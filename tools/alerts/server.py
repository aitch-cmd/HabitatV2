import os
import re
import html
import json
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from fastmcp import FastMCP
from db.connection import postgres_client
from dotenv import load_dotenv

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alerts_mcp")

alerts_server = FastMCP("listing_alerts")

# SMTP Configuration (Standard App Password authentication)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER") or os.getenv("SENDER_EMAIL")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")


# ---------------------------------------------------------------
# SMTP Email helper using App Password
# ---------------------------------------------------------------

def send_alert_email(to_email: str, listing_data: dict) -> bool:
    """
    Send an HTML alert email using standard SMTP with an App Password.
    Compatible with Gmail, Outlook, or any standard SMTP service.
    """
    if not SMTP_USER or not SMTP_APP_PASSWORD:
        logger.warning(
            "SMTP credentials not configured (set SMTP_USER and SMTP_APP_PASSWORD in .env). "
            f"Skipping actual email delivery to {to_email}."
        )
        return False

    address = html.escape(str(listing_data.get("address", "Unknown")))
    price = html.escape(str(listing_data.get("price", "N/A")))
    bedrooms = html.escape(str(listing_data.get("bedrooms", "N/A")))
    description = html.escape(str(listing_data.get("description", ""))[:200])

    html_content = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;border:1px solid #e2e8f0;border-radius:8px;">
        <h2 style="color:#2b6cb0;margin-top:0;">🏠 New Listing Alert!</h2>
        <p style="color:#4a5568;">A new property matching your alert criteria has been listed:</p>
        <div style="background:#f7fafc;border-left:4px solid #3182ce;padding:16px;margin:16px 0;border-radius:4px;">
            <p style="margin:4px 0;"><strong>Address:</strong> {address}</p>
            <p style="margin:4px 0;"><strong>Monthly Rent:</strong> &#8377;{price}</p>
            <p style="margin:4px 0;"><strong>Bedrooms:</strong> {bedrooms} BHK</p>
            <p style="margin:4px 0;"><strong>Description:</strong> {description}...</p>
        </div>
        <p style="color:#718096;font-size:12px;margin-bottom:0;">— HabitatOS Alerts Notification Engine</p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🏠 New Listing Alert: {address} — ₹{price}/mo"
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        logger.info(f"Alert email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send alert email via SMTP to {to_email}: {e}")
        return False


# ---------------------------------------------------------------
# Index-Backed Alert Matching (No Table Scan, No Polling Worker)
# ---------------------------------------------------------------

def find_matching_alerts(listing_data: dict) -> list[dict]:
    """
    Query PostgreSQL using composite and pattern indexes to retrieve
    ONLY the alerts that match the given listing.
    """
    try:
        price = int(listing_data.get("price") or 0)
        bedrooms = int(listing_data.get("bedrooms") or 0)
        address = str(listing_data.get("address") or "")
        locality = str(listing_data.get("locality") or "")
        city = str(listing_data.get("city") or "")

        sql = """
        SELECT id, email, location, max_price, min_bedrooms
        FROM listing_alerts
        WHERE active = TRUE
          AND (max_price = 0 OR max_price >= %(price)s)
          AND (min_bedrooms = 0 OR min_bedrooms <= %(bedrooms)s)
          AND (
              location IS NULL
              OR location = ''
              OR %(address)s ILIKE ('%%' || location || '%%')
              OR %(locality)s ILIKE ('%%' || location || '%%')
              OR %(city)s ILIKE ('%%' || location || '%%')
          );
        """
        return postgres_client.fetch_all(sql, {
            "price": price,
            "bedrooms": bedrooms,
            "address": address,
            "locality": locality,
            "city": city,
        })
    except Exception as e:
        logger.error(f"Error querying matching alerts from PostgreSQL index: {e}")
        return []


def check_and_notify(listing_data: dict) -> int:
    """
    Triggered when a new listing is saved.
    Directly pulls matching alerts from database indexes and dispatches notifications.
    """
    matching_alerts = find_matching_alerts(listing_data)
    if not matching_alerts:
        return 0

    sent = 0
    for alert in matching_alerts:
        if send_alert_email(alert["email"], listing_data):
            sent += 1

    return sent


# ---------------------------------------------------------------
# TOOL 1 – Create Alert
# ---------------------------------------------------------------

@alerts_server.tool()
def create_alert(
    email: str,
    location: str = "",
    max_price: int = 0,
    min_bedrooms: int = 0,
) -> str:
    """
    Create a listing alert. You'll receive email notifications when a new
    matching property is added.

    Args:
        email:        Recipient email address
        location:     Target city or locality substring match (e.g. 'South Delhi', 'Koramangala')
        max_price:    Maximum monthly rent in INR (0 = no limit)
        min_bedrooms: Minimum number of bedrooms (0 = no limit)

    Returns:
        Confirmation with alert ID and registered filters.
    """
    if not _EMAIL_RE.match(email):
        return "❌ Invalid email address."

    filters = {}
    if location:
        filters["location"] = location.strip()
    if max_price > 0:
        filters["max_price"] = max_price
    if min_bedrooms > 0:
        filters["min_bedrooms"] = min_bedrooms

    insert_sql = """
    INSERT INTO listing_alerts (email, location, max_price, min_bedrooms, filters, active)
    VALUES (%s, %s, %s, %s, %s::jsonb, TRUE)
    RETURNING id;
    """

    try:
        inserted_id = postgres_client.execute_and_return_id(
            insert_sql,
            (email, location.strip(), max_price, min_bedrooms, json.dumps(filters)),
        )
    except Exception as e:
        logger.error(f"Error creating alert in PostgreSQL: {e}")
        return f"❌ Error saving alert to database: {e}"

    filter_desc = []
    if location:
        filter_desc.append(f"Location: {location.strip()}")
    if max_price > 0:
        filter_desc.append(f"Max budget: ₹{max_price:,}")
    if min_bedrooms > 0:
        filter_desc.append(f"Min bedrooms: {min_bedrooms} BHK")

    return f"""
✅ Alert Created Successfully

**Alert ID**: {inserted_id}
**Email**: {email}
**Indexed Filters**: {', '.join(filter_desc) if filter_desc else 'All new listings'}

You will automatically receive an email alert whenever a matching listing is added.
"""


# ---------------------------------------------------------------
# TOOL 2 – List Alerts
# ---------------------------------------------------------------

@alerts_server.tool()
def list_alerts(email: str) -> str:
    """
    List all active alerts registered for an email address.

    Args:
        email: Email address to query
    """
    try:
        alerts = postgres_client.fetch_all(
            "SELECT id, location, max_price, min_bedrooms, created_at FROM listing_alerts WHERE email = %s AND active = TRUE ORDER BY id DESC;",
            (email,),
        )
    except Exception as e:
        logger.error(f"Error listing alerts from PostgreSQL: {e}")
        return f"❌ Error retrieving alerts: {e}"

    if not alerts:
        return f"No active alerts found for {email}."

    lines = [f"**Active Alerts for {email}** ({len(alerts)} total)\n"]
    for a in alerts:
        f_parts = []
        if a.get("location"):
            f_parts.append(f"Location: {a['location']}")
        if a.get("max_price") and a["max_price"] > 0:
            f_parts.append(f"Max ₹{a['max_price']:,}")
        if a.get("min_bedrooms") and a["min_bedrooms"] > 0:
            f_parts.append(f"Min {a['min_bedrooms']}BHK")

        created = a.get("created_at")
        created_str = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else str(created or "N/A")

        lines.append(
            f"- **Alert #{a['id']}** → {', '.join(f_parts) if f_parts else 'All listings'} "
            f"(created {created_str})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------
# TOOL 3 – Delete Alert
# ---------------------------------------------------------------

@alerts_server.tool()
def delete_alert(alert_id: str) -> str:
    """
    Deactivate an alert by its ID.

    Args:
        alert_id: The ID of the alert to deactivate
    """
    try:
        aid = int(str(alert_id).strip())
    except ValueError:
        return f"❌ Invalid alert ID format: {alert_id}. Must be an integer ID."

    try:
        rows_updated = postgres_client.execute(
            "UPDATE listing_alerts SET active = FALSE WHERE id = %s RETURNING id;",
            (aid,),
        )

        if rows_updated > 0:
            return f"✅ Alert #{alert_id} deactivated."
        else:
            return f"❌ Alert #{alert_id} not found or already inactive."
    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        return f"❌ Error deleting alert: {e}"

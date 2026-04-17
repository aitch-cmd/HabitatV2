import os
import json
import logging
from datetime import datetime
from bson import ObjectId
from fastmcp import FastMCP
from db.connection import MongoDBClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alerts_mcp")

alerts_server = FastMCP("listing_alerts")

# ---------- DB & SendGrid Setup ----------
try:
    db_client = MongoDBClient(database_name="rental_database")
    alerts_collection = db_client.database["listing_alerts"]
    logger.info("Alerts: Connected to MongoDB.")
except Exception as e:
    alerts_collection = None
    logger.error(f"Alerts: MongoDB connection failed: {e}")

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")


# ---------------------------------------------------------------
# Email helper
# ---------------------------------------------------------------

def send_alert_email(to_email: str, listing_data: dict) -> bool:
    """Send an email notification for a matching listing."""
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        logger.error("SendGrid credentials not configured")
        return False

    address = listing_data.get("address", "Unknown")
    price = listing_data.get("price", "N/A")
    bedrooms = listing_data.get("bedrooms", "N/A")
    description = listing_data.get("description", "")[:200]

    html_content = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;">
        <h2 style="color:#2d3748;">🏠 New Listing Alert!</h2>
        <p>A new property matching your alert has been listed:</p>
        <div style="background:#f7fafc;border-left:4px solid #4299e1;padding:16px;margin:16px 0;border-radius:4px;">
            <p><strong>Address:</strong> {address}</p>
            <p><strong>Monthly Rent:</strong> ₹{price}</p>
            <p><strong>Bedrooms:</strong> {bedrooms}</p>
            <p><strong>Description:</strong> {description}...</p>
        </div>
        <p style="color:#718096;font-size:12px;">— HabitatOS Alerts</p>
    </div>
    """

    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_email,
        subject=f"🏠 New Listing: {address} — ₹{price}/mo",
        html_content=html_content,
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        logger.info(f"Alert email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send alert email to {to_email}: {e}")
        return False


# ---------------------------------------------------------------
# Alert matching — called by listings server after save
# ---------------------------------------------------------------

def check_and_notify(listing_data: dict) -> int:
    """
    Check all active alerts against a newly saved listing.
    Sends emails for matches. Returns count of notifications sent.
    """
    if alerts_collection is None:
        return 0

    alerts = list(alerts_collection.find({"active": True}))
    sent = 0

    for alert in alerts:
        if _matches(alert, listing_data):
            if send_alert_email(alert["email"], listing_data):
                sent += 1

    return sent


def _matches(alert: dict, listing: dict) -> bool:
    """Check if a listing matches an alert's filters."""
    filters = alert.get("filters", {})

    # Location filter (case-insensitive substring)
    loc = filters.get("location")
    if loc:
        listing_addr = (listing.get("address") or "").lower()
        if loc.lower() not in listing_addr:
            return False

    # Max price filter
    max_price = filters.get("max_price")
    if max_price and listing.get("price"):
        try:
            if int(listing["price"]) > int(max_price):
                return False
        except (ValueError, TypeError):
            pass

    # Min bedrooms filter
    min_beds = filters.get("min_bedrooms")
    if min_beds and listing.get("bedrooms"):
        try:
            if int(listing["bedrooms"]) < int(min_beds):
                return False
        except (ValueError, TypeError):
            pass

    return True


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
    Create a listing alert. The user will receive an email when a new
    listing matching the filters is saved.

    Args:
        email:        Email address to send notifications to
        location:     Location filter (city/area substring match)
        max_price:    Maximum monthly rent (0 = no limit)
        min_bedrooms: Minimum number of bedrooms (0 = no limit)

    Returns:
        Confirmation message with alert ID
    """
    if alerts_collection is None:
        return "❌ MongoDB is not connected."

    filters = {}
    if location:
        filters["location"] = location
    if max_price > 0:
        filters["max_price"] = max_price
    if min_bedrooms > 0:
        filters["min_bedrooms"] = min_bedrooms

    doc = {
        "email": email,
        "filters": filters,
        "active": True,
        "created_at": datetime.now().isoformat(),
    }

    result = alerts_collection.insert_one(doc)

    filter_desc = []
    if location:
        filter_desc.append(f"Location: {location}")
    if max_price > 0:
        filter_desc.append(f"Max price: ₹{max_price}")
    if min_bedrooms > 0:
        filter_desc.append(f"Min bedrooms: {min_bedrooms}")

    return f"""
✅ Alert Created

**Alert ID**: {result.inserted_id}
**Email**: {email}
**Filters**: {', '.join(filter_desc) if filter_desc else 'All listings'}

You'll receive an email when a matching listing is added.
"""


# ---------------------------------------------------------------
# TOOL 2 – List Alerts
# ---------------------------------------------------------------

@alerts_server.tool()
def list_alerts(email: str) -> str:
    """
    List all active alerts for an email address.

    Args:
        email: Email address to look up alerts for

    Returns:
        Formatted list of active alerts
    """
    if alerts_collection is None:
        return "❌ MongoDB is not connected."

    alerts = list(alerts_collection.find({"email": email, "active": True}))

    if not alerts:
        return f"No active alerts found for {email}."

    lines = [f"**Active Alerts for {email}** ({len(alerts)} total)\n"]
    for a in alerts:
        filters = a.get("filters", {})
        f_parts = []
        if filters.get("location"):
            f_parts.append(f"Location: {filters['location']}")
        if filters.get("max_price"):
            f_parts.append(f"Max ₹{filters['max_price']}")
        if filters.get("min_bedrooms"):
            f_parts.append(f"Min {filters['min_bedrooms']}BHK")

        lines.append(
            f"- **{a['_id']}** → {', '.join(f_parts) if f_parts else 'All listings'} "
            f"(created {a.get('created_at', 'N/A')})"
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
        alert_id: The MongoDB ObjectId of the alert to deactivate

    Returns:
        Success or error message
    """
    if alerts_collection is None:
        return "❌ MongoDB is not connected."

    try:
        oid = ObjectId(alert_id)
    except Exception:
        return f"❌ Invalid alert ID format: {alert_id}"

    result = alerts_collection.update_one(
        {"_id": oid},
        {"$set": {"active": False}},
    )

    if result.modified_count == 1:
        return f"✅ Alert {alert_id} deactivated."
    else:
        return f"❌ Alert not found or already inactive: {alert_id}"

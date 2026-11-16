import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
from fastmcp import FastMCP
from db.connection import postgres_client
from tools.listings.utils.parser import ParserListings
from tools.alerts.server import check_and_notify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("listings_mcp")

listings_server = FastMCP("property_listings")

parser = ParserListings()

# ---------------------------------------------------------------
# Formatting & Validation
# ---------------------------------------------------------------

def format_listing_summary(data: Dict[str, Any]) -> str:
    def fmt(v, default="Not specified"):
        return v if v not in ["", None, []] else default

    rental = data.get("rental_terms") or {}

    amenities = data.get("amenities") or {}
    amenities_flat = (
        amenities.get("appliances", []) +
        amenities.get("utilities_included", []) +
        amenities.get("other_amenities", [])
    )

    return f"""
**Property Listing Summary**
━━━━━━━━━━━━━━━━━━━━━━━━━━
**Address**: {fmt(data.get("address"))}
**Monthly Rent**: ₹{fmt(data.get("price"))}
**Bedrooms**: {fmt(data.get("bedrooms"))}
**Bathrooms**: {fmt(data.get("bathrooms"))}

📋 **Description**:
{fmt(data.get("description"))}

**Application Fee**: {fmt(rental.get("application_fee"))}
**Security Deposit**: {fmt(rental.get("security_deposit"))}
**Lease Terms**: {fmt(rental.get("lease_terms"))}
**Availability**: {fmt(rental.get("availability"))}
**Contact**: {fmt(data.get("contact"))}

**Pet Friendly**: {fmt(data.get("pet_friendly"))}
**Amenities**: {", ".join(amenities_flat) if amenities_flat else "None"}
**Listing URL**: {fmt(data.get("listing_url"))}
━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()


def validate_listing(d: Dict[str, Any]) -> Tuple[bool, list]:
    errors = []
    if not d.get("address"):
        errors.append("address missing")
    if not d.get("price"):
        errors.append("price missing")
    if not d.get("bedrooms"):
        errors.append("bedrooms missing")
    if not d.get("description"):
        errors.append("description missing")
    return (len(errors) == 0, errors)

# ---------------------------------------------------------------
# TOOL 1 – Parse
# ---------------------------------------------------------------

@listings_server.tool()
def parse_and_confirm_listing(raw_listing: str) -> str:
    """
    Parse listing text → return normalized JSON + summary.
    (Does NOT save to DB.)
    """
    try:
        parsed = parser.extract(raw_listing)
        
        if not parsed or parsed.get("error"):
            error_msg = parsed.get("error", "Unknown parsing error") if parsed else "Parser returned None"
            logger.error(f"Parsing failed: {error_msg}")
            return f"❌ Error parsing listing: {error_msg}"
        
        if not parsed.get("address") or not parsed.get("price"):
            return "❌ Error: Missing required fields (address or price)"
        
        # Add metadata
        parsed["source"] = "user_created"
        parsed["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        ok, errs = validate_listing(parsed)
        summary = format_listing_summary(parsed)

        return f"""
{summary}

Validation: {"OK" if ok else ", ".join(errs)}

Next actions:
- To save → call **save_listing(parsed_listing_json)**
- To modify → call **update_listing_fields(parsed_listing_json, changes_description)**

Parsed JSON:
```json
{json.dumps(parsed, indent=2)}
```
"""
    except Exception as e:
        logger.error(f"Exception in parse_and_confirm_listing: {str(e)}")
        return f"❌ Error parsing listing: {str(e)}"

# ---------------------------------------------------------------
# TOOL 2 – Update fields
# ---------------------------------------------------------------
@listings_server.tool()
def update_listing_fields(parsed_listing_json: str, changes_description: str) -> str:
    """
    Update the listing data based on natural language edit requests.
    """
    try:
        data = json.loads(parsed_listing_json)
    except (json.JSONDecodeError, ValueError, TypeError):
        return "❌ Invalid JSON input."

    text = changes_description.lower()

    # --- numeric updates ---
    numbers = re.findall(r"\d+", text)

    _BOUNDS = {
        "price":     (1, 10_000_000),
        "bedrooms":  (1, 50),
        "bathrooms": (1, 50),
    }

    def apply_numeric(field):
        if field in text and numbers:
            value = int(numbers[0])
            lo, hi = _BOUNDS.get(field, (1, 10_000_000))
            if lo <= value <= hi:
                data[field] = value

    apply_numeric("price")
    apply_numeric("bedrooms")
    apply_numeric("bathrooms")

    # --- rental terms ---
    rental = data.get("rental_terms") or {}

    if "application fee" in text and numbers:
        rental["application_fee"] = numbers[0]

    if "security deposit" in text and numbers:
        rental["security_deposit"] = numbers[0]

    if "lease" in text:
        m = re.search(r"lease.*?(\d+\s*\w+)", text)
        if m:
            rental["lease_terms"] = m.group(1)

    data["rental_terms"] = rental

    # --- pet friendly ---
    if "pet" in text:
        if any(x in text for x in ["yes", "allowed", "friendly"]):
            data["pet_friendly"] = "yes"
        elif any(x in text for x in ["no", "not allowed"]):
            data["pet_friendly"] = "no"

    # --- amenities updates ---
    amenities = data.get("amenities") or {}

    # add amenity
    m = re.findall(r"(?:add|include)\s+([a-zA-Z0-9 ]+)", text)
    if m:
        amenities.setdefault("other_amenities", [])
        amenities["other_amenities"].extend([x.strip() for x in m])

    # remove amenity
    r = re.findall(r"(?:remove|delete)\s+([a-zA-Z0-9 ]+)", text)
    if r:
        for item in r:
            for k in ["appliances", "utilities_included", "other_amenities"]:
                if item in amenities.get(k, []):
                    amenities[k].remove(item)

    data["amenities"] = amenities

    # --- freeform fields ---
    def str_update(field):
        if field in text:
            m = re.search(field + r"\s*to\s*(.*)", text)
            if m:
                data[field] = m.group(1).strip()

    str_update("address")
    str_update("description")
    str_update("contact")
    str_update("listing_url")

    # update timestamp
    data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    ok, errs = validate_listing(data)
    summary = format_listing_summary(data)

    return f"""
Listing Updated

{summary}

Validation: {"OK" if ok else ", ".join(errs)}

Next:
- Call **save_listing(parsed_listing_json)** to save
- Or modify again

Updated JSON:
```json
{json.dumps(data, indent=2)}
```
"""

# ---------------------------------------------------------------
# TOOL 3 – Save
# ---------------------------------------------------------------
@listings_server.tool()
def save_listing(parsed_listing_json: str) -> str:
    """
    Saves the final listing into PostgreSQL.
    """
    try:
        data = json.loads(parsed_listing_json)
    except (json.JSONDecodeError, ValueError, TypeError):
        return "❌ Invalid JSON."

    data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    # Prepare fields for SQL insertion
    title = data.get("title") or f"{data.get('bedrooms', '')} BHK in {data.get('address', '')}"
    address = data.get("address", "")
    price = int(data.get("price", 0))
    bedrooms = int(data.get("bedrooms", 0)) if data.get("bedrooms") else None
    bathrooms = int(data.get("bathrooms", 0)) if data.get("bathrooms") else None
    description = data.get("description", "")
    property_type = data.get("property_type", "Apartment")
    furnishing_status = data.get("furnishing_status", "Unfurnished")
    contact = data.get("contact", "")
    pet_friendly = data.get("pet_friendly", "unknown")
    listing_url = data.get("listing_url", "")
    source = data.get("source", "user_created")
    last_updated = data.get("last_updated")

    amenities = data.get("amenities") or {}
    amenities_json = json.dumps(amenities) if isinstance(amenities, (dict, list)) else "{}"

    rental_terms = data.get("rental_terms") or {}
    rental_terms_json = json.dumps(rental_terms) if isinstance(rental_terms, (dict, list)) else "{}"
    security_deposit = str(rental_terms.get("security_deposit", "Not specified"))

    # Extract locality / city from address if possible
    parts = [p.strip() for p in address.split(",") if p.strip()]
    city = parts[-1] if len(parts) > 1 else ""
    locality = parts[0] if parts else ""

    insert_sql = """
    INSERT INTO listings (
        title, address, locality, city, price, bedrooms, bathrooms,
        description, property_type, furnishing_status, contact,
        security_deposit, pet_friendly, listing_url, amenities,
        rental_terms, source, last_updated
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s
    ) RETURNING id;
    """

    try:
        inserted_id = postgres_client.execute_and_return_id(
            insert_sql,
            (
                title, address, locality, city, price, bedrooms, bathrooms,
                description, property_type, furnishing_status, contact,
                security_deposit, pet_friendly, listing_url, amenities_json,
                rental_terms_json, source, last_updated,
            ),
        )
    except Exception as e:
        logger.error(f"Error saving listing to PostgreSQL: {e}")
        return f"❌ Error saving to PostgreSQL: {e}"

    # Trigger alert notifications for matching alerts
    notified = check_and_notify(data)
    alert_msg = f"\n📧 Notified {notified} alert subscriber(s)." if notified else ""

    return f"""
✅ Listing Saved Successfully

Database ID: {inserted_id}

Your listing is now live.{alert_msg}
"""

# ---------------------------------------------------------------
# TOOL 4 – Delete
# ---------------------------------------------------------------
@listings_server.tool()
def delete_listing(listing_id: str) -> str:
    """
    Delete a listing from PostgreSQL using its ID.
    
    Args:
        listing_id: The ID of the listing to delete (integer as string)
    
    Returns:
        Success or error message
    """
    try:
        lid = int(listing_id.strip())
    except ValueError:
        return f"❌ Invalid listing ID format: {listing_id}. Must be an integer ID."

    try:
        # First check if the listing exists
        existing = postgres_client.fetch_one(
            "SELECT id, address, price FROM listings WHERE id = %s;",
            (lid,),
        )
        
        if not existing:
            return f"❌ Listing not found with ID: {listing_id}"
        
        address = existing.get("address", "Unknown")
        price = existing.get("price", "N/A")
        
        # Delete the listing
        rows_deleted = postgres_client.execute(
            "DELETE FROM listings WHERE id = %s;",
            (lid,),
        )
        
        if rows_deleted > 0:
            return f"""
✅ Listing Deleted Successfully

**Deleted Listing:**
- ID: {listing_id}
- Address: {address}
- Price: ₹{price}

The listing has been permanently removed from the database.
"""
        else:
            return f"❌ Failed to delete listing with ID: {listing_id}"
            
    except Exception as e:
        logger.error(f"Error deleting listing: {e}")
        return f"❌ Error deleting listing: {e}"

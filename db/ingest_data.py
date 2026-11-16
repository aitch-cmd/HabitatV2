"""
Data ingestion and normalization script for HabitatOS PostgreSQL database.
Injects and standardizes data from two Kaggle datasets:
1. divyanshug40/data-for-houses-available-for-rent (Makaan Delhi rentals)
2. iamsouravbanerjee/house-rent-prediction-dataset (India House Rent dataset)
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import kagglehub

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from db.connection import DATABASE_URL, get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_datasets")


def parse_price(val) -> int:
    """Parse string or numeric rental price into an integer in INR."""
    s = str(val).strip().replace(",", "")
    if s.endswith("L"):
        num = float(s[:-1].strip())
        return int(round(num * 100000))
    elif s.endswith("Cr"):
        num = float(s[:-2].strip())
        return int(round(num * 10000000))
    else:
        try:
            return int(round(float(s)))
        except (ValueError, TypeError):
            return 0


def parse_deposit(val) -> str:
    """Standardize security deposit values."""
    s = str(val).strip().replace(",", "")
    if s in ["No", "No deposit", "nan", "None", "", "0"]:
        return "No deposit"
    if s.endswith("L"):
        num = float(s[:-1].strip())
        return f"₹{int(round(num * 100000)):,}"
    elif s.endswith("Cr"):
        num = float(s[:-2].strip())
        return f"₹{int(round(num * 10000000)):,}"
    try:
        return f"₹{int(round(float(s))):,}"
    except (ValueError, TypeError):
        return s


def load_dataset_1() -> pd.DataFrame:
    """Download and normalize dataset 1 (divyanshug40/data-for-houses-available-for-rent)."""
    logger.info("Fetching Dataset 1 (divyanshug40/data-for-houses-available-for-rent)...")
    path = kagglehub.dataset_download("divyanshug40/data-for-houses-available-for-rent")
    csv_file = Path(path) / "Makaan_data_700pages.csv"
    logger.info(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    clean = pd.DataFrame()
    clean["bedrooms"] = df["Size"].astype(int)
    clean["bathrooms"] = df["Bathroom"].fillna(clean["bedrooms"]).astype(int)
    clean["price"] = df["Rent_price"].apply(parse_price)
    clean["area_sqft"] = df["Area_sqft"].astype(int)
    clean["locality"] = df["Location"].str.strip()
    clean["city"] = "Delhi"
    clean["address"] = clean["locality"] + ", Delhi"
    clean["furnishing_status"] = df["Status"].str.strip()

    # Clean property types
    ptype = df["Property_type"].str.strip().replace({"ApartmentApartment": "Apartment"})
    clean["property_type"] = ptype

    clean["floor"] = None
    clean["tenant_preferred"] = "Any"
    clean["contact"] = df["Seller_type"].str.strip() + " (" + df["Seller_name"].str.strip() + ")"
    clean["facing_direction"] = df["Facing_direction"].fillna("Not specified").str.strip()
    clean["security_deposit"] = df["Security_deposit"].apply(parse_deposit)
    clean["pet_friendly"] = "unknown"
    clean["listing_url"] = ""
    clean["source"] = "kaggle_makaan_delhi"
    clean["posted_on"] = None
    clean["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    clean["title"] = (
        clean["bedrooms"].astype(str)
        + " BHK "
        + clean["property_type"]
        + " in "
        + clean["locality"]
        + ", "
        + clean["city"]
    )

    clean["description"] = clean.apply(
        lambda r: (
            f"{r['bedrooms']} BHK {r['property_type']} ({r['area_sqft']} sqft) for rent in "
            f"{r['locality']}, {r['city']}. Furnishing: {r['furnishing_status']}. "
            f"Facing: {r['facing_direction']}. Monthly Rent: ₹{r['price']:,}. "
            f"Bathrooms: {r['bathrooms']}. Security Deposit: {r['security_deposit']}. "
            f"Contact: {r['contact']}."
        ),
        axis=1,
    )

    clean["amenities"] = clean.apply(
        lambda r: json.dumps({"appliances": [], "utilities_included": [], "other_amenities": []}),
        axis=1,
    )

    clean["rental_terms"] = clean.apply(
        lambda r: json.dumps({
            "rent": f"₹{r['price']:,}/month",
            "security_deposit": r["security_deposit"],
            "lease_terms": "12 months",
            "availability": "Immediate",
        }),
        axis=1,
    )

    logger.info(f"Dataset 1 normalized: {len(clean)} rows.")
    return clean


def load_dataset_2() -> pd.DataFrame:
    """Download and normalize dataset 2 (iamsouravbanerjee/house-rent-prediction-dataset)."""
    logger.info("Fetching Dataset 2 (iamsouravbanerjee/house-rent-prediction-dataset)...")
    path = kagglehub.dataset_download("iamsouravbanerjee/house-rent-prediction-dataset")
    csv_file = Path(path) / "House_Rent_Dataset.csv"
    logger.info(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)

    clean = pd.DataFrame()
    clean["bedrooms"] = df["BHK"].astype(int)
    clean["bathrooms"] = df["Bathroom"].astype(int)
    clean["price"] = df["Rent"].astype(int)
    clean["area_sqft"] = df["Size"].astype(int)
    clean["locality"] = df["Area Locality"].str.strip()
    clean["city"] = df["City"].str.strip()
    clean["address"] = clean["locality"] + ", " + clean["city"]
    clean["furnishing_status"] = df["Furnishing Status"].str.strip()

    # Map area type to sensible property type
    clean["property_type"] = df["Area Type"].str.strip().apply(
        lambda x: "Apartment" if "Area" in str(x) else str(x)
    )
    clean["floor"] = df["Floor"].str.strip()
    clean["tenant_preferred"] = df["Tenant Preferred"].str.strip()
    clean["contact"] = df["Point of Contact"].str.strip()
    clean["facing_direction"] = "Not specified"
    clean["security_deposit"] = "Not specified"
    clean["pet_friendly"] = "unknown"
    clean["listing_url"] = ""
    clean["source"] = "kaggle_house_rent_prediction"
    clean["posted_on"] = df["Posted On"].str.strip()
    clean["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    clean["title"] = (
        clean["bedrooms"].astype(str)
        + " BHK "
        + clean["property_type"]
        + " in "
        + clean["locality"]
        + ", "
        + clean["city"]
    )

    clean["description"] = clean.apply(
        lambda r: (
            f"{r['bedrooms']} BHK {r['property_type']} ({r['area_sqft']} sqft) on {r['floor']} "
            f"for rent in {r['locality']}, {r['city']}. Furnishing: {r['furnishing_status']}. "
            f"Preferred Tenant: {r['tenant_preferred']}. Monthly Rent: ₹{r['price']:,}. "
            f"Bathrooms: {r['bathrooms']}. Contact: {r['contact']}."
        ),
        axis=1,
    )

    clean["amenities"] = clean.apply(
        lambda r: json.dumps({"appliances": [], "utilities_included": [], "other_amenities": []}),
        axis=1,
    )

    clean["rental_terms"] = clean.apply(
        lambda r: json.dumps({
            "rent": f"₹{r['price']:,}/month",
            "security_deposit": r["security_deposit"],
            "lease_terms": "12 months",
            "availability": "Immediate",
        }),
        axis=1,
    )

    logger.info(f"Dataset 2 normalized: {len(clean)} rows.")
    return clean


def generate_seed_sql(combined_df: pd.DataFrame, output_path: Path):
    """
    Generate seed_data.sql for Docker entrypoint automatic database initialization.
    Uses multi-row INSERTs for optimal speed.
    """
    logger.info(f"Generating SQL seed file at {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "title", "address", "locality", "city", "price", "bedrooms", "bathrooms",
        "area_sqft", "furnishing_status", "property_type", "floor", "tenant_preferred",
        "contact", "facing_direction", "security_deposit", "pet_friendly", "listing_url",
        "amenities", "rental_terms", "description", "source", "posted_on", "last_updated"
    ]

    col_names_str = ", ".join(columns)

    def sql_escape(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "NULL"
        s = str(v).replace("'", "''")
        return f"'{s}'"

    batch_size = 500
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("-- Generated seed data for HabitatOS PostgreSQL\n")
        f.write("BEGIN;\n\n")

        for start_idx in range(0, len(combined_df), batch_size):
            batch = combined_df.iloc[start_idx:start_idx + batch_size]
            rows_sql = []
            for _, row in batch.iterrows():
                vals = [
                    sql_escape(row["title"]),
                    sql_escape(row["address"]),
                    sql_escape(row["locality"]),
                    sql_escape(row["city"]),
                    str(int(row["price"])),
                    str(int(row["bedrooms"])) if pd.notna(row["bedrooms"]) else "NULL",
                    str(int(row["bathrooms"])) if pd.notna(row["bathrooms"]) else "NULL",
                    str(int(row["area_sqft"])) if pd.notna(row["area_sqft"]) else "NULL",
                    sql_escape(row["furnishing_status"]),
                    sql_escape(row["property_type"]),
                    sql_escape(row["floor"]),
                    sql_escape(row["tenant_preferred"]),
                    sql_escape(row["contact"]),
                    sql_escape(row["facing_direction"]),
                    sql_escape(row["security_deposit"]),
                    sql_escape(row["pet_friendly"]),
                    sql_escape(row["listing_url"]),
                    f"'{row['amenities']}'::jsonb",
                    f"'{row['rental_terms']}'::jsonb",
                    sql_escape(row["description"]),
                    sql_escape(row["source"]),
                    sql_escape(row["posted_on"]) if row["posted_on"] else "NULL",
                    sql_escape(row["last_updated"]),
                ]
                rows_sql.append(f"({', '.join(vals)})")

            f.write(f"INSERT INTO listings ({col_names_str}) VALUES\n")
            f.write(",\n".join(rows_sql))
            f.write(";\n\n")

        f.write("COMMIT;\n")

    logger.info(f"Successfully generated seed SQL ({output_path.stat().st_size / (1024*1024):.2f} MB).")


def inject_into_postgres(combined_df: pd.DataFrame):
    """Inject rows directly into a running PostgreSQL database."""
    logger.info("Attempting direct insertion into running PostgreSQL database...")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Ensure schema exists
                init_sql_path = BASE_DIR / "db" / "init.sql"
                if init_sql_path.exists():
                    cur.execute(init_sql_path.read_text(encoding="utf-8"))

                logger.info("Database schema initialized/verified.")

                # Insert data in batches
                insert_sql = """
                INSERT INTO listings (
                    title, address, locality, city, price, bedrooms, bathrooms,
                    area_sqft, furnishing_status, property_type, floor, tenant_preferred,
                    contact, facing_direction, security_deposit, pet_friendly, listing_url,
                    amenities, rental_terms, description, source, posted_on, last_updated
                ) VALUES (
                    %(title)s, %(address)s, %(locality)s, %(city)s, %(price)s, %(bedrooms)s, %(bathrooms)s,
                    %(area_sqft)s, %(furnishing_status)s, %(property_type)s, %(floor)s, %(tenant_preferred)s,
                    %(contact)s, %(facing_direction)s, %(security_deposit)s, %(pet_friendly)s, %(listing_url)s,
                    %(amenities)s::jsonb, %(rental_terms)s::jsonb, %(description)s, %(source)s, %(posted_on)s, %(last_updated)s
                );
                """
                records = combined_df.to_dict(orient="records")
                cur.executemany(insert_sql, records)
                logger.info(f"Successfully injected {len(records)} listings directly into PostgreSQL!")
    except Exception as e:
        logger.warning(
            f"Could not connect directly to PostgreSQL ({e}).\n"
            f"Don't worry: `db/seed_data.sql` has been generated and will automatically seed the database when "
            f"`docker compose up -d` is executed!"
        )


def main():
    logger.info("Starting Kaggle datasets download, normalization, and ingestion...")

    df1 = load_dataset_1()
    df2 = load_dataset_2()

    logger.info("Merging datasets into unified schema...")
    combined = pd.concat([df1, df2], ignore_index=True)
    logger.info(f"Total unified listings count: {len(combined)}")

    # 1. Generate docker-compose seed SQL
    seed_path = BASE_DIR / "db" / "seed_data.sql"
    generate_seed_sql(combined, seed_path)

    # 2. Also save cleaned dataset CSV in db/data/ for inspection
    data_dir = BASE_DIR / "db" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "unified_house_rent_listings.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved normalized CSV to {csv_path}")

    # 3. Attempt direct injection if DB is up
    inject_into_postgres(combined)

    logger.info("Data processing complete!")


if __name__ == "__main__":
    main()

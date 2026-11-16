-- HabitatOS PostgreSQL Schema with pgvector
-- Automatically executed on container startup via /docker-entrypoint-initdb.d/

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Table: listings
CREATE TABLE IF NOT EXISTS listings (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    address TEXT NOT NULL,
    locality VARCHAR(255),
    city VARCHAR(100),
    price INTEGER NOT NULL,
    bedrooms INTEGER,
    bathrooms INTEGER,
    area_sqft INTEGER,
    furnishing_status VARCHAR(50),
    property_type VARCHAR(100),
    floor VARCHAR(100),
    tenant_preferred VARCHAR(100),
    contact VARCHAR(255),
    facing_direction VARCHAR(50),
    security_deposit VARCHAR(100),
    pet_friendly VARCHAR(20) DEFAULT 'unknown',
    listing_url TEXT,
    amenities JSONB DEFAULT '{"appliances": [], "utilities_included": [], "other_amenities": []}'::jsonb,
    rental_terms JSONB DEFAULT '{}'::jsonb,
    description TEXT,
    source VARCHAR(100),
    posted_on DATE,
    last_updated VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding vector(384)
);

-- Indexes for listings performance
CREATE INDEX IF NOT EXISTS idx_listings_price ON listings(price);
CREATE INDEX IF NOT EXISTS idx_listings_bedrooms ON listings(bedrooms);
CREATE INDEX IF NOT EXISTS idx_listings_city ON listings(city);
CREATE INDEX IF NOT EXISTS idx_listings_locality ON listings(locality);
CREATE INDEX IF NOT EXISTS idx_listings_property_type ON listings(property_type);

-- Full-text search index across textual fields for BM25-style search
CREATE INDEX IF NOT EXISTS idx_listings_fts ON listings USING gin(
    to_tsvector('english', 
        coalesce(title, '') || ' ' || 
        coalesce(address, '') || ' ' || 
        coalesce(city, '') || ' ' || 
        coalesce(locality, '') || ' ' || 
        coalesce(description, '')
    )
);

-- HNSW Vector Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_listings_embedding ON listings USING hnsw (embedding vector_cosine_ops);

-- Backward compatibility view
CREATE OR REPLACE VIEW tulire_listings AS SELECT * FROM listings;

-- Table: listing_alerts
CREATE TABLE IF NOT EXISTS listing_alerts (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    max_price INTEGER DEFAULT 0,
    min_bedrooms INTEGER DEFAULT 0,
    filters JSONB DEFAULT '{}'::jsonb,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_email ON listing_alerts(email);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON listing_alerts(active);
-- Index for instant SQL alert matching without table scans or workers
CREATE INDEX IF NOT EXISTS idx_alerts_matching ON listing_alerts(active, max_price, min_bedrooms);
CREATE INDEX IF NOT EXISTS idx_alerts_location_trgm ON listing_alerts USING gin(location gin_trgm_ops);

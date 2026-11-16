"""
Stage 1 & Stage 2: Structured SQL Pre-filtering + In-Database Hybrid RAG (pgvector Cosine + tsvector BM25 RRF).
"""

import logging
import torch
from sentence_transformers import SentenceTransformer
from db.connection import postgres_client

logger = logging.getLogger("hybrid_search")


class HybridRAGSearcher:
    """
    Executes 2-stage retrieval inside PostgreSQL:
    Stage 1: Hard SQL constraints (bedrooms, price + 15% buffer, city).
    Stage 2: Hybrid RAG combining Dense Vector Cosine Similarity and Lexical FTS BM25 via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer ({model_name}) on device: {self.device}")
        self.embedder = SentenceTransformer(model_name, device=self.device)

    def search(
        self,
        query_text: str,
        budget: float | None = None,
        target_bhk: int | None = None,
        city: str | None = None,
        top_k: int = 25,
    ) -> list[dict]:
        """
        Execute hybrid in-database search and return top_k candidates with RRF scores.
        """
        # Step 1: Compute query embedding
        query_emb = self.embedder.encode(query_text, normalize_embeddings=True)
        emb_str = f"[{','.join(f'{x:.6f}' for x in query_emb)}]"

        # Step 2: Build Structured SQL Pre-Filter conditions (Stage 1)
        conditions = ["1=1"]
        params: dict = {
            "emb": emb_str,
            "q_text": query_text,
            "top_k": top_k,
        }

        # Price bounds with 15% upper buffer and reasonable lower bound for mid/high budgets
        if budget and budget > 0:
            params["max_price"] = int(budget * 1.15)
            conditions.append("price <= %(max_price)s")
            if budget >= 12000:
                params["min_price"] = int(budget * 0.45)
                conditions.append("price >= %(min_price)s")

        # Target BHK filter (allows exact or nearby if relaxed)
        if target_bhk and target_bhk > 0:
            params["target_bhk"] = target_bhk
            conditions.append("(bedrooms = %(target_bhk)s OR bedrooms IS NULL)")

        # City filter
        if city and city.strip():
            params["city"] = city.strip()
            conditions.append("city ILIKE %(city)s")

        where_clause = " AND ".join(conditions)

        # Step 3: In-Database Hybrid RAG with Reciprocal Rank Fusion (Stage 2)
        sql = f"""
        WITH dense_search AS (
            SELECT id, 
                   1.0 - (embedding <=> %(emb)s::vector) AS dense_sim,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector ASC) AS r_dense
            FROM listings
            WHERE {where_clause} AND embedding IS NOT NULL
            ORDER BY embedding <=> %(emb)s::vector ASC
            LIMIT 100
        ),
        lexical_search AS (
            SELECT id,
                   ts_rank_cd(
                       to_tsvector('english', coalesce(title, '') || ' ' || coalesce(address, '') || ' ' || coalesce(city, '') || ' ' || coalesce(locality, '') || ' ' || coalesce(description, '')),
                       plainto_tsquery('english', %(q_text)s)
                   ) AS lexical_score,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(
                           to_tsvector('english', coalesce(title, '') || ' ' || coalesce(address, '') || ' ' || coalesce(city, '') || ' ' || coalesce(locality, '') || ' ' || coalesce(description, '')),
                           plainto_tsquery('english', %(q_text)s)
                       ) DESC
                   ) AS r_lexical
            FROM listings
            WHERE {where_clause} AND to_tsvector('english', coalesce(title, '') || ' ' || coalesce(address, '') || ' ' || coalesce(city, '') || ' ' || coalesce(locality, '') || ' ' || coalesce(description, '')) @@ plainto_tsquery('english', %(q_text)s)
            LIMIT 100
        ),
        candidate_ids AS (
            SELECT id FROM dense_search
            UNION
            SELECT id FROM lexical_search
        ),
        rrf_combined AS (
            SELECT 
                c.id,
                COALESCE(d.dense_sim, 0.0) AS dense_sim,
                COALESCE(l.lexical_score, 0.0) AS lexical_score,
                (COALESCE(1.0 / (60 + d.r_dense), 0.0) + COALESCE(1.0 / (60 + l.r_lexical), 0.0)) AS rrf_score
            FROM candidate_ids c
            LEFT JOIN dense_search d ON c.id = d.id
            LEFT JOIN lexical_search l ON c.id = l.id
        )
        SELECT l.id, l.title, l.address, l.locality, l.city, l.price, l.bedrooms, l.bathrooms,
               l.area_sqft, l.furnishing_status, l.property_type, l.floor, l.tenant_preferred,
               l.contact, l.facing_direction, l.security_deposit, l.pet_friendly, l.listing_url,
               l.amenities, l.rental_terms, l.description, l.source,
               r.dense_sim, r.lexical_score, r.rrf_score
        FROM rrf_combined r
        JOIN listings l ON r.id = l.id
        ORDER BY r.rrf_score DESC
        LIMIT %(top_k)s;
        """

        try:
            results = postgres_client.fetch_all(sql, params)
            # Fallback if no hybrid matches (e.g. strict query on rare words)
            if not results:
                fallback_sql = f"""
                SELECT id, title, address, locality, city, price, bedrooms, bathrooms,
                       area_sqft, furnishing_status, property_type, floor, tenant_preferred,
                       contact, facing_direction, security_deposit, pet_friendly, listing_url,
                       amenities, rental_terms, description, source,
                       0.5 AS dense_sim, 0.0 AS lexical_score, 0.01 AS rrf_score
                FROM listings
                WHERE {where_clause}
                ORDER BY price ASC
                LIMIT %(top_k)s;
                """
                results = postgres_client.fetch_all(fallback_sql, params)
            return results
        except Exception as e:
            logger.error(f"Error executing hybrid search: {e}")
            return []

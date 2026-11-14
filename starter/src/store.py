"""Persistent vector storage and cosine search using PostgreSQL + pgvector."""
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    """Store chunks by strategy and search them with cosine distance."""

    table_name = "day6_chunks"

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url must not be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")
        self.database_url = database_url
        self.dimensions = int(dimensions)

    def _connect(self) -> psycopg.Connection:
        connection = psycopg.connect(self.database_url)
        register_vector(connection)
        return connection

    def create_schema(self) -> None:
        """Create the extension, durable chunk table, and metadata index."""
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL CHECK (page > 0),
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS day6_chunks_strategy_section_idx
                ON {self.table_name} (strategy, (metadata ->> 'section'))
                """
            )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Replace one strategy's rows transactionally, removing stale chunks."""
        if not strategy:
            raise ValueError("strategy must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Received {len(chunks)} chunks but {len(vectors)} vectors"
            )

        rows = []
        for chunk, values in zip(chunks, vectors):
            if len(values) != self.dimensions:
                raise ValueError(
                    f"Vector for {chunk.chunk_id} has {len(values)} dimensions; "
                    f"expected {self.dimensions}"
                )
            if not chunk.text.strip():
                raise ValueError(f"Chunk {chunk.chunk_id} has empty text")
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    Vector(values),
                )
            )

        insert_sql = f"""
            INSERT INTO {self.table_name}
                (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """

        with self._connect() as connection:
            # Replacing the strategy set also removes chunks made obsolete by
            # changed source text or chunking boundaries.
            connection.execute(
                f"DELETE FROM {self.table_name} WHERE strategy = %s", (strategy,)
            )
            if rows:
                with connection.cursor() as cursor:
                    cursor.executemany(insert_sql, rows)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks ranked by cosine similarity, with optional JSONB filters."""
        if not strategy:
            raise ValueError("strategy must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector has {len(query_vector)} dimensions; "
                f"expected {self.dimensions}"
            )

        where_sql = "strategy = %s"
        params: list[Any] = [strategy]
        if filters:
            where_sql += " AND metadata @> %s"
            params.append(Jsonb(filters))

        query_sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s) AS score
            FROM {self.table_name}
            WHERE {where_sql}
            ORDER BY embedding <=> %s, chunk_id
            LIMIT %s
        """
        params = [Vector(query_vector), *params, Vector(query_vector), top_k]

        with self._connect() as connection:
            rows = connection.execute(query_sql, params).fetchall()

        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

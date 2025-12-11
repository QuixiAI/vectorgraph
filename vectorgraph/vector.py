"""Thin vector-facing wrappers around the async helpers in db.py.
Keeps the public API small while making imports intention-revealing."""
from .db import (
    create_db,
    delete_db,
    vector_add,
    vector_batch_add,
    vector_batch_delete,
    vector_cosine_similarity,
    vector_delete,
    vector_get,
    vector_nearest_neighbors,
    vector_query_by_id,
    vector_rebuild_index,
    vector_stats,
    vector_update,
)

__all__ = [
    "create_db",
    "delete_db",
    "vector_add",
    "vector_batch_add",
    "vector_batch_delete",
    "vector_cosine_similarity",
    "vector_delete",
    "vector_get",
    "vector_nearest_neighbors",
    "vector_query_by_id",
    "vector_rebuild_index",
    "vector_stats",
    "vector_update",
]

"""Synchronous wrappers around the async helpers for quick scripts/tests."""
import asyncio
from functools import partial
from typing import Any, Callable
from . import db as adb


def _run(coro_factory: Callable[..., Any], *args, **kwargs):
    return asyncio.run(coro_factory(*args, **kwargs))


def create_db() -> str:
    return _run(adb.create_db)


def delete_db(db_id: str) -> None:
    return _run(adb.delete_db, db_id)


def graph_create_entity(db_id: str, id: str, name: str, description: str = None) -> dict | None:
    return _run(adb.graph_create_entity, db_id, id, name, description)


def graph_get_entity(db_id: str, id: str) -> dict | None:
    return _run(adb.graph_get_entity, db_id, id)


def graph_create_relationship(db_id: str, source_id: str, target_id: str, relation: str, **properties) -> dict | None:
    return _run(adb.graph_create_relationship, db_id, source_id, target_id, relation, **properties)


def graph_neighbors(db_id: str, entity_id: str, depth: int = 1, relation: str = None) -> list[dict]:
    return _run(adb.graph_neighbors, db_id, entity_id, depth=depth, relation=relation)


def graph_similarity(db_id: str, entity_a: str, entity_b: str, method: str = "neighbors", depth: int = 2) -> float:
    return _run(adb.graph_similarity, db_id, entity_a, entity_b, method=method, depth=depth)


def vector_add(db_id: str, id: str, vector, metadata: dict | None = None) -> dict:
    return _run(adb.vector_add, db_id, id, vector, metadata)


def vector_get(db_id: str, id: str, include_vector: bool = False) -> dict | None:
    return _run(adb.vector_get, db_id, id, include_vector)


def vector_nearest_neighbors(db_id: str, vector, k: int = 10, metadata_filter: dict | None = None, include_vector: bool = False) -> list[dict]:
    return _run(adb.vector_nearest_neighbors, db_id, vector, k=k, metadata_filter=metadata_filter, include_vector=include_vector)


def vector_query_by_id(db_id: str, id: str, k: int = 10, metadata_filter: dict | None = None, include_vector: bool = False) -> list[dict]:
    return _run(adb.vector_query_by_id, db_id, id, k=k, metadata_filter=metadata_filter, include_vector=include_vector)


def cross_join_query(
    db_id: str,
    *,
    vector=None,
    source_id: str | None = None,
    k: int = 10,
    table: str | None = None,
    where: dict | None = None,
    include_neighbors: bool = True,
    depth: int = 1,
    relation: str | None = None,
    include_vector: bool = False,
) -> list[dict]:
    return _run(
        adb.cross_join_query,
        db_id,
        vector=vector,
        source_id=source_id,
        k=k,
        table=table,
        where=where,
        include_neighbors=include_neighbors,
        depth=depth,
        relation=relation,
        include_vector=include_vector,
    )

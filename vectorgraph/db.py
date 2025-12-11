import os
import uuid
import json
import asyncio
import weakref
from typing import Optional, Any, Iterable, Sequence
import re

import asyncpg
from dotenv import load_dotenv


load_dotenv()

# Use env-driven connection details so compose/.env values are honored
# Defaults mirror the packaged Docker stack (see docker-compose.yml/.env.example).
DB_USER = os.getenv("POSTGRES_USER", "vg_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "vg_password")
DB_NAME = os.getenv("POSTGRES_DB", "vg_db")
DB_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

_pools: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncpg.pool.Pool]" = weakref.WeakKeyDictionary()


def _quote_ident(name: str) -> str:
    """Minimal identifier quoting to avoid injection when using dynamic names."""
    return '"' + name.replace('"', '""') + '"'


def _quote_rel(name: str) -> str:
    """Quote relationship/type names using backticks for Cypher."""
    return "`" + name.replace("`", "``") + "`"


def _quote_label(name: str) -> str:
    """Quote label names using backticks for Cypher."""
    return "`" + name.replace("`", "``") + "`"


def _graph_name(db_id: str) -> str:
    return f"graph_{db_id}"


def _escape_literal(value: str) -> str:
    """Escape single quotes for safe literal insertion into format strings."""
    return str(value).replace("'", "''")


TYPE_LABEL = "kg_type"

def _vector_table(db_id: str) -> str:
    return f"vector_{db_id}"


def _vector_index_name(db_id: str) -> str:
    return f"vector_{db_id}_embedding_idx"


def _validate_vector(vec: Sequence[float]) -> list[float]:
    if not isinstance(vec, Sequence):
        raise ValueError("vector must be a sequence of floats")
    vec_list = [float(x) for x in vec]
    if len(vec_list) != 768:
        raise ValueError(f"vector must have dimension 768, got {len(vec_list)}")
    return vec_list


def _vector_param(vec: Sequence[float]) -> str:
    vec_list = _validate_vector(vec)
    return "[" + ", ".join(str(x) for x in vec_list) + "]"


def _normalize_vector(val: Any) -> Optional[list[float]]:
    if val is None:
        return None
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        cleaned = val.strip("[]")
        if not cleaned:
            return []
        return [float(x.strip()) for x in cleaned.split(",")]
    return None


async def _ensure_graph_exists(conn: asyncpg.connection.Connection, graph: str) -> None:
    exists = await conn.fetchval(
        "SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1;", graph
    )
    if exists is None:
        raise ValueError(f"Graph '{graph}' does not exist")


async def _ensure_vector_table(
    conn: asyncpg.connection.Connection, db_id: str
) -> tuple[str, str]:
    table_name = _vector_table(db_id)
    index_name = _vector_index_name(db_id)
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote_ident(table_name)} (
            id uuid PRIMARY KEY,
            embedding vector(768) NOT NULL,
            metadata jsonb DEFAULT '{{}}'::jsonb
        );
        """
    )
    # Ensure metadata column exists on older tables
    await conn.execute(
        f"ALTER TABLE {_quote_ident(table_name)} ADD COLUMN IF NOT EXISTS metadata jsonb DEFAULT '{{}}'::jsonb;"
    )
    # Create HNSW index if missing
    await conn.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = '{_escape_literal(index_name)}'
            ) THEN
                CREATE INDEX {_quote_ident(index_name)}
                ON {_quote_ident(table_name)}
                USING hnsw (embedding vector_cosine_ops);
            END IF;
        END$$;
        """
    )
    return table_name, index_name


async def _cypher_json(
    conn: asyncpg.connection.Connection, graph: str, cypher: str
) -> list[Any]:
    """
    Execute Cypher and return parsed JSON rows.
    """
    graph_literal = graph.replace("'", "''")
    sql = f"SELECT agtype_to_json(v) AS data FROM cypher('{graph_literal}', $$ {cypher} $$) AS (v agtype);"
    rows = await conn.fetch(sql)
    return [json.loads(row["data"]) if isinstance(row["data"], str) else row["data"] for row in rows]


async def _cypher_json_one(
    conn: asyncpg.connection.Connection, graph: str, cypher: str
) -> Optional[Any]:
    rows = await _cypher_json(conn, graph, cypher)
    return rows[0] if rows else None


async def _get_pool() -> asyncpg.pool.Pool:
    loop = asyncio.get_running_loop()
    pool = _pools.get(loop)
    if pool is None:
        pool = await asyncpg.create_pool(
            DB_DSN,
            ssl=False,
            min_size=1,
            max_size=10,
            command_timeout=60.0,
        )
        _pools[loop] = pool
    return pool


async def create_db() -> str:
    """
    Create a new graph (AGE) and vector table, both keyed by a UUID.

    Graph name: graph_{uuid}
    Vector table name: vector_{uuid}
    """
    db_id = uuid.uuid4().hex  # avoid hyphens to keep graph/table identifiers simple
    graph_name = f"graph_{db_id}"
    vector_table = f"vector_{db_id}"

    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')

        # Create AGE graph
        await conn.execute("SELECT create_graph($1);", graph_name)
        # Pre-create labels/edge labels used by helpers
        await conn.execute("SELECT create_vlabel($1, 'Entity');", graph_name)
        await conn.execute("SELECT create_vlabel($1, $2);", graph_name, TYPE_LABEL)
        await conn.execute("SELECT create_elabel($1, 'HAS_TYPE');", graph_name)

        # Create vector table to store embeddings/payloads
        await conn.execute(
            f"""
            CREATE TABLE {_quote_ident(vector_table)} (
                id uuid PRIMARY KEY,
                embedding vector(768) NOT NULL,
                payload jsonb DEFAULT '{{}}'::jsonb,
                metadata jsonb DEFAULT '{{}}'::jsonb
            );
            """
        )

    return db_id


# --------- Graph entity helpers ---------


async def graph_create_entity(db_id: str, id: str, name: str, description: str = None) -> dict:
    graph = _graph_name(db_id)
    desc_literal = f"'{_escape_literal(description)}'" if description is not None else "NULL"
    cypher = f"""
    CREATE (e:Entity {{id: '{_escape_literal(id)}', name: '{_escape_literal(name)}', description: {desc_literal}}})
    RETURN e
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json_one(conn, graph, cypher)


async def graph_get_entity(db_id: str, id: str) -> Optional[dict]:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH (e:Entity {{id: '{_escape_literal(id)}'}})
    RETURN e
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json_one(conn, graph, cypher)


async def graph_find_entities(
    db_id: str,
    name: str = None,
    type: str = None,
    description_contains: str = None,
    limit: int = 50,
) -> list[dict]:
    graph = _graph_name(db_id)
    where_clauses = []
    if name is not None:
        where_clauses.append(f"e.name = '{_escape_literal(name)}'")
    if type is not None:
        where_clauses.append(f"e.type = '{_escape_literal(type)}'")
    if description_contains is not None:
        where_clauses.append(f"e.description CONTAINS '{_escape_literal(description_contains)}'")
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    cypher = f"""
    MATCH (e:Entity)
    {where_sql}
    RETURN e
    LIMIT {limit}
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json(conn, graph, cypher)


async def graph_update_entity(db_id: str, id: str, **fields) -> Optional[dict]:
    graph = _graph_name(db_id)
    allowed = {k: v for k, v in fields.items() if k in {"name", "description", "type"}}
    if not allowed:
        return await graph_get_entity(db_id, id)
    set_parts = []
    for key, val in allowed.items():
        if val is None:
            set_parts.append(f"e.{key} = NULL")
        else:
            set_parts.append(f"e.{key} = '{_escape_literal(str(val))}'")
    set_sql = ", ".join(set_parts)
    cypher = f"""
    MATCH (e:Entity {{id: '{_escape_literal(id)}'}})
    SET {set_sql}
    RETURN e
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json_one(conn, graph, cypher)


async def graph_delete_entity(db_id: str, id: str) -> bool:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH (e:Entity {{id: '{_escape_literal(id)}'}})
    DETACH DELETE e
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_literal = graph.replace("'", "''")
        await conn.execute(
            f"SELECT * FROM cypher('{graph_literal}', $$ {cypher} $$) AS (v agtype);"
        )
        return True


# --------- Types ---------


async def graph_create_type(db_id: str, name: str) -> dict:
    graph = _graph_name(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        existing = await graph_get_type(db_id, name)
        if existing:
            return existing
        cypher = f"""
        CREATE (t:{TYPE_LABEL} {{name: '{_escape_literal(name)}'}})
        RETURN t
        """
        return await _cypher_json_one(conn, graph, cypher)


async def graph_get_type(db_id: str, name: str) -> Optional[dict]:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH (t:{TYPE_LABEL} {{name: '{_escape_literal(name)}'}})
    RETURN t
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json_one(conn, graph, cypher)


async def graph_list_types(db_id: str) -> list[str]:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH (t:{TYPE_LABEL})
    RETURN t.name
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_literal = graph.replace("'", "''")
        rows = await conn.fetch(
            f"SELECT v FROM cypher('{graph_literal}', $$ {cypher} $$) AS (v agtype);"
        )
        return [row["v"] for row in rows]


async def graph_delete_type(db_id: str, name: str) -> bool:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH (t:{_quote_label(TYPE_LABEL)} {{name: '{_escape_literal(name)}'}})
    DETACH DELETE t
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_literal = graph.replace("'", "''")
        await conn.execute(
            f"SELECT * FROM cypher('{graph_literal}', $$ {cypher} $$) AS (v agtype);"
        )
        return True


async def graph_assign_type(db_id: str, entity_id: str, type_name: str) -> Optional[dict]:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH (e:Entity {{id: '{_escape_literal(entity_id)}'}})
    MATCH (t:{TYPE_LABEL} {{name: '{_escape_literal(type_name)}'}})
    MERGE (e)-[r:HAS_TYPE]->(t)
    RETURN r
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json_one(conn, graph, cypher)


# --------- Relationships ---------


async def graph_create_relationship(
    db_id: str,
    source_id: str,
    target_id: str,
    relation: str,
    **properties,
) -> dict:
    graph = _graph_name(db_id)
    rel_id = str(uuid.uuid4())
    props = {"id": rel_id, "relation": relation, **properties}
    props_literal = ", ".join(
        f"{k}: '{_escape_literal(str(v))}'" for k, v in props.items()
    )
    cypher = f"""
    MATCH (a:Entity {{id: '{_escape_literal(source_id)}'}})
    MATCH (b:Entity {{id: '{_escape_literal(target_id)}'}})
    CREATE (a)-[r:{_quote_rel(relation)} {{{props_literal}}}]->(b)
    RETURN r
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        return await _cypher_json_one(conn, graph, cypher)


async def graph_get_relationship(db_id: str, rel_id: str) -> Optional[dict]:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH ()-[r]-()
    WHERE r.id = '{_escape_literal(rel_id)}'
    RETURN r
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        return await _cypher_json_one(conn, graph, cypher)


async def graph_find_relationships(
    db_id: str,
    source_id: str = None,
    target_id: str = None,
    relation: str = None,
) -> list[dict]:
    graph = _graph_name(db_id)
    where_parts = []
    if relation:
        where_parts.append(f"type(r) = '{_escape_literal(relation)}'")
    if source_id:
        where_parts.append(f"a.id = '{_escape_literal(source_id)}'")
    if target_id:
        where_parts.append(f"b.id = '{_escape_literal(target_id)}'")
    where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    cypher = f"""
    MATCH (a:Entity)-[r]-(b:Entity)
    {where_sql}
    RETURN r
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json(conn, graph, cypher)


async def graph_update_relationship(db_id: str, rel_id: str, **properties) -> Optional[dict]:
    graph = _graph_name(db_id)
    if not properties:
        return await graph_get_relationship(db_id, rel_id)
    set_parts = []
    for key, val in properties.items():
        if val is None:
            set_parts.append(f"r.{key} = NULL")
        else:
            set_parts.append(f"r.{key} = '{_escape_literal(str(val))}'")
    set_sql = ", ".join(set_parts)
    cypher = f"""
    MATCH ()-[r]-()
    WHERE r.id = '{_escape_literal(rel_id)}'
    SET {set_sql}
    RETURN r
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json_one(conn, graph, cypher)


async def graph_delete_relationship(db_id: str, rel_id: str) -> bool:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH ()-[r]-()
    WHERE r.id = '{_escape_literal(rel_id)}'
    DELETE r
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_literal = graph.replace("'", "''")
        await conn.execute(
            f"SELECT * FROM cypher('{graph_literal}', $$ {cypher} $$) AS (v agtype);"
        )
        return True


# --------- Traversal ---------


async def graph_neighbors(
    db_id: str, entity_id: str, depth: int = 1, relation: str = None
) -> list[dict]:
    graph = _graph_name(db_id)
    rel_filter = f":{_quote_rel(relation)}" if relation else ""
    cypher = f"""
    MATCH (e:Entity {{id: '{_escape_literal(entity_id)}'}})-[r{rel_filter}*1..{depth}]-(n:Entity)
    RETURN DISTINCT n
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        return await _cypher_json(conn, graph, cypher)


async def graph_paths(
    db_id: str, start_id: str, end_id: str, max_depth: int = 5
) -> list[list[str]]:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH p = (a:Entity {{id: '{_escape_literal(start_id)}'}})-[*1..{max_depth}]-(b:Entity {{id: '{_escape_literal(end_id)}'}})
    RETURN p
    LIMIT 1
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        paths = await _cypher_json(conn, graph, cypher)
    # Extract node ids from path objects if present
    result_paths = []
    for p in paths:
        if isinstance(p, dict) and "vertices" in p:
            ids = []
            for v in p["vertices"]:
                if not isinstance(v, dict):
                    continue
                props = v.get("properties", {})
                ids.append(props.get("id") or v.get("id"))
            result_paths.append(ids)
    return result_paths


async def graph_traverse(
    db_id: str,
    entity_id: str,
    depth: int = 3,
    direction: str = "both",
    relations: Iterable[str] = None,
) -> dict:
    graph = _graph_name(db_id)
    if direction not in {"in", "out", "both"}:
        direction = "both"
    dir_arrow = {
        "in": "<-",
        "out": "->",
        "both": "-",
    }[direction]
    rel_filter = ""
    if relations:
        rels = "|".join(_quote_rel(r) for r in relations)
        rel_filter = f":{rels}"
    cypher = f"""
    MATCH p = (e:Entity {{id: '{_escape_literal(entity_id)}'}}){dir_arrow}[r{rel_filter}*1..{depth}]{dir_arrow}(n:Entity)
    RETURN p
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        paths = await _cypher_json(conn, graph, cypher)
    return {"paths": paths}


async def graph_extract_subgraph(
    db_id: str,
    center_id: str,
    depth: int = 2,
    include_types: bool = True,
) -> dict:
    graph = _graph_name(db_id)
    cypher = f"""
    MATCH p = (e:Entity {{id: '{_escape_literal(center_id)}'}})-[*1..{depth}]-(n)
    RETURN p
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        paths = await _cypher_json(conn, graph, cypher)
    if not include_types:
        # Optionally strip Type nodes from result paths
        filtered = []
        for p in paths:
            if isinstance(p, dict) and "vertices" in p:
                vertices = [v for v in p["vertices"] if v.get("label") != TYPE_LABEL]
                p["vertices"] = vertices
            filtered.append(p)
        paths = filtered
    return {"paths": paths}


async def graph_similarity(
    db_id: str,
    entity_a: str,
    entity_b: str,
    method: str = "neighbors",
    depth: int = 2,
) -> float:
    if method != "neighbors":
        method = "neighbors"
    # Simple Jaccard similarity on neighbor ids
    neighbors_a = await graph_neighbors(db_id, entity_a, depth=depth)
    neighbors_b = await graph_neighbors(db_id, entity_b, depth=depth)
    ids_a = {
        (n.get("properties") or {}).get("id", n.get("id"))
        for n in neighbors_a
        if isinstance(n, dict)
    }
    ids_b = {
        (n.get("properties") or {}).get("id", n.get("id"))
        for n in neighbors_b
        if isinstance(n, dict)
    }
    intersection = len(ids_a & ids_b)
    union = len(ids_a | ids_b)
    return intersection / union if union else 0.0


# --------- Vector operations ---------


async def vector_add(
    db_id: str, id: Optional[str], vector: Sequence[float], metadata: Optional[dict] = None
) -> dict:
    vec = _validate_vector(vector)
    vector_id = id or str(uuid.uuid4())
    meta = metadata or {}
    table = _vector_table(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        await conn.execute(
            f"""
            INSERT INTO {_quote_ident(table)} (id, embedding, metadata)
            VALUES ($1, $2::vector, $3::jsonb)
            ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
            """,
            vector_id,
            _vector_param(vec),
            json.dumps(meta),
        )
        return {"id": vector_id, "metadata": meta}


async def vector_get(db_id: str, id: str, include_vector: bool = False) -> Optional[dict]:
    table = _vector_table(db_id)
    columns = "id, metadata" + (", embedding" if include_vector else "")
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        row = await conn.fetchrow(
            f"SELECT {columns} FROM {_quote_ident(table)} WHERE id = $1;",
            id,
        )
    if not row:
        return None
    result = {"id": str(row["id"]), "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]}
    if include_vector:
        result["vector"] = _normalize_vector(row["embedding"])
    return result


async def vector_update(
    db_id: str,
    id: str,
    vector: Optional[Sequence[float]] = None,
    metadata: Optional[dict] = None,
) -> Optional[dict]:
    table = _vector_table(db_id)
    set_parts = []
    params = []
    if vector is not None:
        set_parts.append(f"embedding = ${len(params)+1}::vector")
        params.append(_vector_param(vector))
    if metadata is not None:
        set_parts.append(f"metadata = ${len(params)+1}::jsonb")
        params.append(json.dumps(metadata))
    if not set_parts:
        return await vector_get(db_id, id, include_vector=False)
    params.append(id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        await conn.execute(
            f"UPDATE {_quote_ident(table)} SET {', '.join(set_parts)} WHERE id = ${len(params)};",
            *params,
        )
    return await vector_get(db_id, id, include_vector=False)


async def vector_delete(db_id: str, id: str) -> bool:
    table = _vector_table(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        result = await conn.execute(
            f"DELETE FROM {_quote_ident(table)} WHERE id = $1;", id
        )
    return "DELETE" in result.upper()


def vector_cosine_similarity(db_id: str, vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    va = _validate_vector(vector_a)
    vb = _validate_vector(vector_b)
    dot = sum(x * y for x, y in zip(va, vb))
    norm_a = sum(x * x for x in va) ** 0.5
    norm_b = sum(y * y for y in vb) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def vector_nearest_neighbors(
    db_id: str,
    vector: Sequence[float],
    k: int = 10,
    metadata_filter: Optional[dict] = None,
    include_vector: bool = False,
) -> list[dict]:
    vec = _validate_vector(vector)
    table = _vector_table(db_id)
    k = min(max(k, 1), 200)
    filter_clause = ""
    params = [ _vector_param(vec), k]
    if metadata_filter:
        filter_clause = "WHERE metadata @> $2"
        params = [_vector_param(vec), json.dumps(metadata_filter), k]
    select_vector = ", embedding" if include_vector else ""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        rows = await conn.fetch(
            f"""
            SELECT id, metadata, embedding <=> $1 AS score{select_vector}
            FROM {_quote_ident(table)}
            {filter_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${len(params)};
            """,
            *params,
        )
    results = []
    for r in rows:
        item = {
            "id": str(r["id"]),
            "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
            "score": float(r["score"]),
        }
        if include_vector:
            item["vector"] = _normalize_vector(r["embedding"])
        results.append(item)
    return results


async def vector_query_by_id(
    db_id: str, id: str, k: int = 10, metadata_filter: Optional[dict] = None, include_vector: bool = False
) -> list[dict]:
    anchor = await vector_get(db_id, id, include_vector=True)
    if not anchor or "vector" not in anchor:
        return []
    # Exclude the anchor itself after query
    neighbors = await vector_nearest_neighbors(
        db_id, anchor["vector"], k=k + 1, metadata_filter=metadata_filter, include_vector=include_vector
    )
    return [n for n in neighbors if n["id"] != id][:k]


def _validate_column(name: str) -> str:
    """
    Ensure we only allow simple identifier columns to avoid injection when building SQL.
    """
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid column name '{name}'")
    return name


async def cross_join_query(
    db_id: str,
    *,
    vector: Optional[Sequence[float]] = None,
    source_id: Optional[str] = None,
    k: int = 10,
    table: Optional[str] = None,
    where: Optional[dict] = None,
    include_neighbors: bool = True,
    depth: int = 1,
    relation: Optional[str] = None,
    include_vector: bool = False,
) -> list[dict]:
    """
    Native vector -> SQL -> graph flow:
    - vector search to get candidates
    - optional SQL filter on a relational table by id + predicates
    - optional graph neighbors for the surviving ids
    """
    if vector is None and source_id is None:
        raise ValueError("Provide either 'vector' or 'source_id'")
    anchor_vector = vector
    if anchor_vector is None and source_id:
        anchor_row = await vector_get(db_id, source_id, include_vector=True)
        if not anchor_row or "vector" not in anchor_row:
            return []
        anchor_vector = anchor_row["vector"]

    base_hits = await vector_nearest_neighbors(
        db_id, anchor_vector, k=k, metadata_filter=None, include_vector=include_vector
    )
    if not base_hits:
        return []

    filtered_hits = base_hits
    relational_rows: dict[str, dict[str, Any]] = {}
    if table:
        table_name = _quote_ident(_validate_column(table))
        candidate_ids = [hit["id"] for hit in base_hits]
        pool = await _get_pool()
        clauses = [f'id = ANY($1::uuid[])']
        params: list[Any] = [candidate_ids]
        if where:
            for key, val in where.items():
                _validate_column(key)
                params.append(val)
                clauses.append(f'{_quote_ident(key)} = ${len(params)}')
        sql = f'SELECT * FROM {table_name} WHERE ' + " AND ".join(clauses)
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        relational_rows = {str(row["id"]): dict(row) for row in rows}
        filtered_hits = [hit for hit in base_hits if hit["id"] in relational_rows]

    if include_neighbors and filtered_hits:
        for hit in filtered_hits:
            hit["neighbors"] = await graph_neighbors(
                db_id, hit["id"], depth=depth, relation=relation
            )

    results = []
    for hit in filtered_hits:
        item = dict(hit)
        if relational_rows:
            item["sql_row"] = relational_rows.get(hit["id"])
        results.append(item)
    return results


async def vector_batch_add(
    db_id: str, items: list[dict], upsert: bool = False
) -> dict:
    table = _vector_table(db_id)
    pool = await _get_pool()
    inserted = 0
    updated = 0
    errors = 0
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        for item in items:
            try:
                vid = item.get("id") or str(uuid.uuid4())
                vec = _validate_vector(item["vector"])
                meta = item.get("metadata") or {}
                if upsert:
                    await conn.execute(
                        f"""
                        INSERT INTO {_quote_ident(table)} (id, embedding, metadata)
                        VALUES ($1, $2::vector, $3::jsonb)
                        ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata
                        """,
                        vid,
                        _vector_param(vec),
                        json.dumps(meta),
                    )
                    updated += 1
                else:
                    await conn.execute(
                        f"""
                        INSERT INTO {_quote_ident(table)} (id, embedding, metadata)
                        VALUES ($1, $2::vector, $3::jsonb)
                        ON CONFLICT DO NOTHING
                        """,
                        vid,
                        _vector_param(vec),
                        json.dumps(meta),
                    )
                    inserted += 1
            except Exception:
                errors += 1
    return {"inserted": inserted, "updated": updated, "errors": errors}


async def vector_batch_delete(db_id: str, ids: list[str]) -> dict:
    if not ids:
        return {"deleted": 0}
    table = _vector_table(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        result = await conn.execute(
            f"DELETE FROM {_quote_ident(table)} WHERE id = ANY($1::uuid[]);",
            ids,
        )
    # result like "DELETE X"
    try:
        count = int(result.split()[-1])
    except Exception:
        count = 0
    return {"deleted": count}


async def vector_stats(db_id: str) -> dict:
    table = _vector_table(db_id)
    index_name = _vector_index_name(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {_quote_ident(table)};")
        has_index = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1);",
            index_name,
        )
    return {"rows": int(count or 0), "has_index": bool(has_index), "index": index_name}


async def vector_rebuild_index(db_id: str) -> bool:
    table = _vector_table(db_id)
    index_name = _vector_index_name(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, f"graph_{db_id}")
        await _ensure_vector_table(conn, db_id)
        await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(index_name)};")
        await conn.execute(
            f"""
            CREATE INDEX {_quote_ident(index_name)}
            ON {_quote_ident(table)}
            USING hnsw (embedding vector_cosine_ops);
            """
        )
    return True


async def vector_run_raw(db_id: str, **kwargs) -> dict:
    raise NotImplementedError("Raw vector operations are not exposed for safety")


# --------- Admin/utility ---------


async def graph_run_cypher(db_id: str, cypher: str, params: dict = None) -> list[Any]:
    graph = _graph_name(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_literal = graph.replace("'", "''")
        if params:
            rows = await conn.fetch(
                f"SELECT agtype_to_json(v) AS data FROM cypher('{graph_literal}', $$ {cypher} $$, $1::json) AS (v agtype);",
                json.dumps(params),
            )
            return [json.loads(row["data"]) if isinstance(row["data"], str) else row["data"] for row in rows]
        sql = f"SELECT agtype_to_json(v) AS data FROM cypher('{graph_literal}', $$ {cypher} $$) AS (v agtype);"
        rows = await conn.fetch(sql)
        return [json.loads(row["data"]) if isinstance(row["data"], str) else row["data"] for row in rows]


async def graph_stats(db_id: str) -> dict:
    graph = _graph_name(db_id)
    cypher_nodes = "MATCH (n) RETURN count(n)"
    cypher_rels = "MATCH ()-[r]-() RETURN count(r)"
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_literal = graph.replace("'", "''")
        nodes = await conn.fetchval(
            f"SELECT v FROM cypher('{graph_literal}', $$ {cypher_nodes} $$) AS (v agtype);"
        )
        rels = await conn.fetchval(
            f"SELECT v FROM cypher('{graph_literal}', $$ {cypher_rels} $$) AS (v agtype);"
        )
    return {"nodes": int(nodes or 0), "relationships": int(rels or 0)}


async def graph_schema(db_id: str) -> dict:
    graph = _graph_name(db_id)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        await _ensure_graph_exists(conn, graph)
        graph_oid = await conn.fetchval(
            "SELECT graphid FROM ag_catalog.ag_graph WHERE name = $1;", graph
        )
        labels_rows = await conn.fetch(
            "SELECT name FROM ag_catalog.ag_label WHERE graph = $1 AND kind = 'v';",
            graph_oid,
        )
        rels_rows = await conn.fetch(
            "SELECT name FROM ag_catalog.ag_label WHERE graph = $1 AND kind = 'e';",
            graph_oid,
        )
    return {
        "labels": [row["name"] for row in labels_rows],
        "relationships": [row["name"] for row in rels_rows],
    }


async def delete_db(db_id: str) -> None:
    """
    Drop the graph and vector table associated with the given UUID.
    """
    graph_name = f"graph_{db_id}"
    vector_table = f"vector_{db_id}"

    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')

        # Drop graph (cascade removes its schema objects)
        await conn.execute("SELECT drop_graph($1, true);", graph_name)

        # Drop vector table if it exists
        await conn.execute(
            f'DROP TABLE IF EXISTS {_quote_ident(vector_table)} CASCADE;'
        )

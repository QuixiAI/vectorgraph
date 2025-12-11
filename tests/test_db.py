import uuid
from vectorgraph.db import create_db, delete_db


async def test_extensions(db_pool):
    async with db_pool.acquire() as conn:
        extensions = await conn.fetch("SELECT extname FROM pg_extension")
        ext_names = {ext["extname"] for ext in extensions}
        required = {"vector", "age", "btree_gist", "pg_trgm", "http"}
        for ext in required:
            assert ext in ext_names, f"{ext} extension not found"
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')
        result = await conn.fetchval("SELECT count(*) FROM ag_catalog.ag_graph")
        assert result >= 0


async def test_create_and_delete_db(db_pool):
    db_id = await create_db()
    graph_name = f"graph_{db_id}"
    vector_table = f"vector_{db_id}"

    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')

        graph_exists = await conn.fetchval(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1;", graph_name
        )
        assert graph_exists == 1

        table_exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", vector_table)
        assert table_exists

    await delete_db(db_id)

    async with db_pool.acquire() as conn:
        await conn.execute("LOAD 'age';")
        await conn.execute('SET search_path = ag_catalog, "$user", public;')

        graph_gone = await conn.fetchval(
            "SELECT 1 FROM ag_catalog.ag_graph WHERE name = $1;", graph_name
        )
        assert graph_gone is None

        table_gone = await conn.fetchval("SELECT to_regclass($1) IS NULL;", vector_table)
        assert table_gone

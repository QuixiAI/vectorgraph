import uuid
from vectorgraph.graph import (
    create_db,
    delete_db,
    graph_assign_type,
    graph_create_entity,
    graph_create_relationship,
    graph_create_type,
    graph_delete_entity,
    graph_delete_relationship,
    graph_delete_type,
    graph_extract_subgraph,
    graph_find_entities,
    graph_find_relationships,
    graph_get_entity,
    graph_get_relationship,
    graph_get_type,
    graph_list_types,
    graph_neighbors,
    graph_paths,
    graph_run_cypher,
    graph_schema,
    graph_similarity,
    graph_stats,
    graph_traverse,
    graph_update_entity,
    graph_update_relationship,
)


def prop(node, key):
    return (node or {}).get("properties", {}).get(key) if isinstance(node, dict) else None


async def test_graph_functions(db_pool):
    db_id = await create_db()
    type_name = f"Type-{uuid.uuid4()}"
    e1_id = f"e1-{uuid.uuid4()}"
    e2_id = f"e2-{uuid.uuid4()}"

    try:
        created_type = await graph_create_type(db_id, type_name)
        assert created_type and prop(created_type, "name") == type_name

        fetched_type = await graph_get_type(db_id, type_name)
        assert fetched_type and prop(fetched_type, "name") == type_name

        types_list = await graph_list_types(db_id)
        normalized_types = {t.strip('"') for t in types_list}
        assert type_name in normalized_types

        e1 = await graph_create_entity(db_id, e1_id, "Alpha", "First entity")
        e2 = await graph_create_entity(db_id, e2_id, "Beta", "Second entity")
        assert e1 and e2

        get_e1 = await graph_get_entity(db_id, e1_id)
        assert get_e1 and prop(get_e1, "id") == e1_id

        found_by_name = await graph_find_entities(db_id, name="Alpha")
        assert any(prop(item, "id") == e1_id for item in found_by_name)

        updated_e1 = await graph_update_entity(db_id, e1_id, description="Updated", type=type_name)
        assert updated_e1 and prop(updated_e1, "description") == "Updated"

        await graph_assign_type(db_id, e1_id, type_name)

        rel = await graph_create_relationship(db_id, e1_id, e2_id, "RELATES", note="hello")
        rel_id = prop(rel, "id") or rel.get("id")
        assert rel_id

        rel_fetched = await graph_get_relationship(db_id, rel_id)
        assert rel_fetched and prop(rel_fetched, "id") == rel_id

        rels_found = await graph_find_relationships(db_id, relation="RELATES")
        assert any(prop(r, "id") == rel_id for r in rels_found)

        rel_updated = await graph_update_relationship(db_id, rel_id, note="updated")
        assert rel_updated and prop(rel_updated, "note") == "updated"

        neighbors = await graph_neighbors(db_id, e1_id, depth=1)
        assert any(prop(n, "id") == e2_id for n in neighbors)

        paths = await graph_paths(db_id, e1_id, e2_id, max_depth=2)
        assert isinstance(paths, list)

        traverse = await graph_traverse(db_id, e1_id, depth=2)
        assert traverse.get("paths")

        subgraph = await graph_extract_subgraph(db_id, e1_id, depth=2)
        assert subgraph.get("paths")

        sim = await graph_similarity(db_id, e1_id, e2_id, depth=1)
        assert 0.0 <= sim <= 1.0

        cypher_rows = await graph_run_cypher(
            db_id, "MATCH (e:Entity) RETURN e.id ORDER BY e.id"
        )
        ids = []
        for row in cypher_rows:
            if isinstance(row, dict):
                ids.append(prop(row, "id") or row.get("id"))
            else:
                ids.append(row)
        assert e1_id in ids and e2_id in ids

        stats = await graph_stats(db_id)
        assert stats["nodes"] >= 2
        assert stats["relationships"] >= 1

        schema_info = await graph_schema(db_id)
        labels = schema_info.get("labels", [])
        assert "Entity" in labels or "entity" in labels
        assert "kg_type" in labels

        assert await graph_delete_relationship(db_id, rel_id)
        after_rel = await graph_get_relationship(db_id, rel_id)
        assert after_rel is None

        assert await graph_delete_entity(db_id, e2_id)
        assert await graph_delete_entity(db_id, e1_id)

        assert await graph_delete_type(db_id, type_name)

    finally:
        await delete_db(db_id)

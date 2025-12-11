import uuid
from vectorgraph.vector import (
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


def make_vec(v: float) -> list[float]:
    return [v] * 768


async def test_vector_functions(db_pool):
    db_id = await create_db()
    try:
        v1_id = str(uuid.uuid4())
        v2_id = str(uuid.uuid4())

        added = await vector_add(db_id, v1_id, make_vec(0.1), {"a": 1})
        assert added["id"] == v1_id

        fetched = await vector_get(db_id, v1_id, include_vector=True)
        assert fetched and fetched["id"] == v1_id and len(fetched.get("vector", [])) == 768

        updated = await vector_update(db_id, v1_id, metadata={"a": 2})
        assert updated and updated["metadata"]["a"] == 2

        await vector_add(db_id, v2_id, make_vec(0.2), {"b": 3})

        nn = await vector_nearest_neighbors(db_id, make_vec(0.1), k=2)
        ids = [item["id"] for item in nn]
        assert v1_id in ids

        by_id = await vector_query_by_id(db_id, v1_id, k=1)
        assert by_id

        batch_items = [
            {"id": str(uuid.uuid4()), "vector": make_vec(0.3), "metadata": {"c": 1}},
            {"id": str(uuid.uuid4()), "vector": make_vec(0.4), "metadata": {"d": 2}},
        ]
        batch_result = await vector_batch_add(db_id, batch_items)
        assert batch_result["inserted"] >= 2

        stats = await vector_stats(db_id)
        assert stats["rows"] >= 4

        del_ids = [batch_items[0]["id"]]
        del_result = await vector_batch_delete(db_id, del_ids)
        assert del_result["deleted"] == 1

        assert await vector_delete(db_id, v2_id)

        assert await vector_rebuild_index(db_id)

        sim = vector_cosine_similarity(db_id, make_vec(1.0), make_vec(1.0))
        assert abs(sim - 1.0) < 1e-6
    finally:
        await delete_db(db_id)

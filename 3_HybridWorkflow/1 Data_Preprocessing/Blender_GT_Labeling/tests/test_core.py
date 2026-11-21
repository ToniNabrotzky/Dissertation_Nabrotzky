from bonsai_joint_labeler.core import (
    next_free_index, format_joint_id, format_pair_id,
    sort_pair_by_guid, pair_key, orphaned_pair_ids,
)


def test_next_free_index_fills_gaps():
    assert next_free_index([]) == 1
    assert next_free_index([1, 2, 3]) == 4
    assert next_free_index([1, 2, 4]) == 3
    assert next_free_index([2, 3]) == 1


def test_formatting_without_brackets():
    assert format_joint_id(1) == "J0001"
    assert format_pair_id(123) == "P0123"


def test_sort_by_guid():
    a, b = {"guid": "B"}, {"guid": "A"}
    assert sort_pair_by_guid(a, b) == (b, a)


def test_pair_key_is_order_independent():
    assert pair_key("A", "B") == pair_key("B", "A")


def test_orphaned_pairs_keep_shared():
    mapping = {
        "J0001": {"P0001", "P0002"},
        "J0002": {"P0002", "P0003"},
    }
    # P0001 nur in J0001 -> verwaist; P0002 auch in J0002 -> bleibt
    assert orphaned_pair_ids(mapping, "J0001") == {"P0001"}
    assert orphaned_pair_ids(mapping, "J0002") == {"P0003"}
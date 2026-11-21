"""
Reine Logik ohne Blender-Abhängigkeit (separat testbar).
"""

from typing import Iterable


def next_free_index(used_indices: Iterable[int]) -> int:
    """
    Liefert den kleinsten freien positiven Index (>= 1).

    Füllt Lücken auf: {1, 2, 4} -> 3 ; {1, 2, 3} -> 4 ; {} -> 1

    :param used_indices: Bereits vergebene Indizes.
    :return: Kleinster nicht belegter Index.
    """
    used = set(used_indices)
    candidate = 1
    while candidate in used:
        candidate += 1
    return candidate


def format_joint_id(index: int) -> str:
    """Formatiert einen Joint-Index, z. B. 1 -> 'J0001'."""
    return f"J{index:04d}"


def format_pair_id(index: int) -> str:
    """Formatiert einen Paar-Index, z. B. 12 -> 'P0012'."""
    return f"P{index:04d}"


def sort_pair_by_guid(elem_a: dict, elem_b: dict) -> tuple:
    """
    Sortiert zwei Element-Dicts aufsteigend nach 'guid'.
    Das Element mit der kleineren GUID wird A, das andere B.

    :param elem_a: Dict mit mindestens dem Schlüssel 'guid'.
    :param elem_b: Dict mit mindestens dem Schlüssel 'guid'.
    :return: Tupel (A, B) sortiert nach GUID.
    """
    return (elem_a, elem_b) if elem_a["guid"] <= elem_b["guid"] else (elem_b, elem_a)


def pair_key(guid_a: str, guid_b: str) -> frozenset:
    """
    Reihenfolge-unabhängiger Schlüssel für ein Kontaktpaar.
    Wird zur Identifikation von Paaren im globalen Paar-Set verwendet.
    """
    return frozenset((guid_a, guid_b))


def orphaned_pair_ids(joint_to_pairs: dict, removed_joint_id: str) -> set:
    """
    Ermittelt Paare, die nach dem Entfernen eines Joints keinem Joint mehr angehören.

    :param joint_to_pairs: Mapping {joint_id: set(pair_id, ...)} VOR dem Löschen.
    :param removed_joint_id: ID des zu löschenden Joints.
    :return: Menge der Paar-IDs, die nur im gelöschten Joint vorkamen.
    """
    removed_pairs = set(joint_to_pairs.get(removed_joint_id, set()))
    still_used = set()
    for joint_id, pair_ids in joint_to_pairs.items():
        if joint_id != removed_joint_id:
            still_used |= set(pair_ids)
    return removed_pairs - still_used


CSV_HEADER = [
    "Joint_ID", "Pair_ID", "Status",  # "Pair_ID" bewusst wie in der Spezifikation
    "Type_A", "Type_B", "GUID_A", "GUID_B", "Name_A", "Name_B",
]
DEFAULT_STATUS = "o"
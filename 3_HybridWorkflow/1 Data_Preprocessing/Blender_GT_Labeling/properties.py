"""
Datenmodell: PropertyGroups an bpy.types.Scene (in .blend persistiert).

Normalisiertes n:m-Modell:
  - jl_pairs  = globales Set eindeutiger Kontaktpaare (inkl. Status)
  - jl_joints = Joints mit Referenzen (pair_refs) auf Paare
"""

import bpy
from bpy.props import (
    CollectionProperty, EnumProperty, IntProperty, StringProperty,
)
from bpy.types import PropertyGroup

from . import core


class JL_PairRef(PropertyGroup):
    """Mitgliedschaft: Verweis eines Joints auf ein Paar im globalen Paar-Set."""

    pair_id: StringProperty(name="Paar-ID (FK)", default="")


class JL_Joint(PropertyGroup):
    """Ein Soll-Joint (Gruppe von Kontaktpaaren)."""

    index: IntProperty(name="Index", default=1)
    joint_id: StringProperty(name="Joint-ID", default="")
    pair_refs: CollectionProperty(type=JL_PairRef)


class JL_Pair(PropertyGroup):
    """Ein eindeutiges Kontaktpaar zweier IFC-Elemente (globales Set)."""

    index: IntProperty(name="Index", default=1)
    pair_id: StringProperty(name="Paar-ID", default="")

    guid_a: StringProperty(name="GUID A")
    guid_b: StringProperty(name="GUID B")
    type_a: StringProperty(name="Typ A")
    type_b: StringProperty(name="Typ B")
    name_a: StringProperty(name="Name A")
    name_b: StringProperty(name="Name B")

    # Nachträglich editierbar, wird in der .blend gespeichert
    status: StringProperty(
        name="Status",
        description="Status des Kontaktpaares, wird im CSV-Export übernommen",
        default=core.DEFAULT_STATUS,
    )


# ---------------------------------------------------------------------
# Dynamischer Enum für den aktiven Joint
# ---------------------------------------------------------------------
_ENUM_CACHE = []


def _joint_enum_items(self, context):
    """
    Dynamischer Enum-Callback: baut die Dropdown-Einträge aus den Joints.

    Blender hält keine eigene Referenz auf die zurückgegebenen Strings,
    daher wird die Liste in einem Modul-Cache gehalten.
    """
    global _ENUM_CACHE
    scene = context.scene if context else bpy.context.scene
    items = [
        (j.joint_id, j.joint_id, f"Joint {j.joint_id}")
        for j in sorted(scene.jl_joints, key=lambda j: j.index)
    ]
    if not items:
        items = [("NONE", "– kein Joint –", "Noch kein Joint angelegt")]
    _ENUM_CACHE = items
    return _ENUM_CACHE


# ---------------------------------------------------------------------
# Helfer für Datenzugriff (nutzen Szene, daher hier statt in core.py)
# ---------------------------------------------------------------------
def get_joint(scene, joint_id):
    """Liefert den Joint mit der gegebenen ID oder None."""
    return next((j for j in scene.jl_joints if j.joint_id == joint_id), None)


def get_pair(scene, pair_id):
    """Liefert das Paar mit der gegebenen ID oder None."""
    return next((p for p in scene.jl_pairs if p.pair_id == pair_id), None)


def find_pair_by_guids(scene, guid_a, guid_b):
    """Sucht im globalen Paar-Set ein Paar mit der GUID-Kombination (reihenfolge-unabhängig)."""
    key = core.pair_key(guid_a, guid_b)
    return next(
        (p for p in scene.jl_pairs if core.pair_key(p.guid_a, p.guid_b) == key),
        None,
    )


def joint_to_pairs_map(scene):
    """Baut das Mapping {joint_id: set(pair_id)} aus der Szene."""
    return {
        j.joint_id: {ref.pair_id for ref in j.pair_refs}
        for j in scene.jl_joints
    }


_CLASSES = (JL_PairRef, JL_Joint, JL_Pair)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.jl_joints = CollectionProperty(type=JL_Joint)
    bpy.types.Scene.jl_pairs = CollectionProperty(type=JL_Pair)
    bpy.types.Scene.jl_active_joint = EnumProperty(
        name="Aktiver Joint",
        description="Joint, dem neue Kontaktpaare zugeordnet werden",
        items=_joint_enum_items,
    )


def unregister():
    del bpy.types.Scene.jl_active_joint
    del bpy.types.Scene.jl_pairs
    del bpy.types.Scene.jl_joints
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
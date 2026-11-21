"""
Operatoren des Addons.
"""

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import core, csv_export, ifc_utils
from .properties import (
    find_pair_by_guids, get_joint, get_pair, joint_to_pairs_map,
)


class JL_OT_add_joint(Operator):
    """Erstellt einen neuen Joint (füllt Lücken in der Nummerierung)"""
    bl_idname = "jl.add_joint"
    bl_label = "Joint hinzufügen"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        # Kleinsten freien Index bestimmen (Lückenfüllung)
        idx = core.next_free_index(j.index for j in scene.jl_joints)

        joint = scene.jl_joints.add()
        joint.index = idx
        joint.joint_id = core.format_joint_id(idx)

        # Neuen Joint direkt aktiv setzen
        scene.jl_active_joint = joint.joint_id
        self.report({"INFO"}, f"{joint.joint_id} erstellt")
        return {"FINISHED"}


class JL_OT_remove_joint(Operator):
    """Löscht den aktiven Joint. Paare, die noch in anderen Joints vorkommen, bleiben erhalten"""
    bl_idname = "jl.remove_joint"
    bl_label = "Joint löschen"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.jl_joints) > 0

    def execute(self, context):
        scene = context.scene
        active_id = scene.jl_active_joint

        pos = next((i for i, j in enumerate(scene.jl_joints) if j.joint_id == active_id), None)
        if pos is None:
            self.report({"WARNING"}, "Kein Joint aktiv")
            return {"CANCELLED"}

        # 1) Verwaiste Paare bestimmen (nur im gelöschten Joint vorhanden)
        orphans = core.orphaned_pair_ids(joint_to_pairs_map(scene), active_id)

        # 2) Verwaiste Paare aus dem globalen Set entfernen (rückwärts iterieren)
        for i in reversed(range(len(scene.jl_pairs))):
            if scene.jl_pairs[i].pair_id in orphans:
                scene.jl_pairs.remove(i)

        # 3) Joint selbst (inkl. seiner Mitgliedschaften) entfernen
        scene.jl_joints.remove(pos)

        # Verbleibenden Joint aktiv setzen, falls vorhanden
        if len(scene.jl_joints) > 0:
            scene.jl_active_joint = scene.jl_joints[0].joint_id

        kept = len(joint_to_pairs_map(scene)) and "" or ""
        self.report(
            {"INFO"},
            f"{active_id} gelöscht ({len(orphans)} Paar(e) entfernt, geteilte Paare bleiben)",
        )
        return {"FINISHED"}


class JL_OT_add_pair(Operator):
    """Speichert die 2 ausgewählten IFC-Elemente als Kontaktpaar im aktiven Joint"""
    bl_idname = "jl.add_pair"
    bl_label = "Kontaktpaar speichern"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.jl_joints) > 0

    def execute(self, context):
        scene = context.scene
        joint = get_joint(scene, scene.jl_active_joint)
        if joint is None:
            self.report({"ERROR"}, "Kein aktiver Joint")
            return {"CANCELLED"}

        # 1) Auswahl validieren (genau 2 Objekte, beide mit IFC-GUID)
        elements, error = ifc_utils.get_selected_ifc_elements(context)
        if error:
            self.report({"ERROR"}, error)
            return {"CANCELLED"}

        # 2) Nach GUID sortieren: kleinere GUID = A
        a, b = core.sort_pair_by_guid(*elements)

        # 3) Paar im globalen Set suchen, ggf. neu anlegen
        pair = find_pair_by_guids(scene, a["guid"], b["guid"])
        is_new = pair is None
        if is_new:
            idx = core.next_free_index(p.index for p in scene.jl_pairs)
            pair = scene.jl_pairs.add()
            pair.index = idx
            pair.pair_id = core.format_pair_id(idx)
            pair.guid_a, pair.type_a, pair.name_a = a["guid"], a["type"], a["name"]
            pair.guid_b, pair.type_b, pair.name_b = b["guid"], b["type"], b["name"]
            pair.status = core.DEFAULT_STATUS

        # 4) Mitgliedschaft im aktiven Joint prüfen
        if any(ref.pair_id == pair.pair_id for ref in joint.pair_refs):
            self.report({"WARNING"}, f"{pair.pair_id} ist bereits in {joint.joint_id}")
            return {"CANCELLED"}

        # 5) Paar dem Joint zuordnen
        ref = joint.pair_refs.add()
        ref.pair_id = pair.pair_id

        msg = "neu angelegt und" if is_new else "bereits vorhanden, nur"
        self.report({"INFO"}, f"{pair.pair_id} {msg} zu {joint.joint_id} hinzugefügt")
        return {"FINISHED"}


class JL_OT_remove_pair(Operator):
    """Entfernt ein Kontaktpaar aus dem aktiven Joint (Paar bleibt erhalten, falls in anderen Joints vorhanden)"""
    bl_idname = "jl.remove_pair"
    bl_label = "Paar aus Joint entfernen"
    bl_options = {"REGISTER", "UNDO"}

    pair_id: StringProperty()

    def execute(self, context):
        scene = context.scene
        joint = get_joint(scene, scene.jl_active_joint)
        if joint is None:
            return {"CANCELLED"}

        # Mitgliedschaft im aktiven Joint entfernen
        pos = next((i for i, r in enumerate(joint.pair_refs) if r.pair_id == self.pair_id), None)
        if pos is None:
            return {"CANCELLED"}
        joint.pair_refs.remove(pos)

        # Paar komplett löschen, falls es in keinem Joint mehr vorkommt
        still_used = any(
            r.pair_id == self.pair_id for j in scene.jl_joints for r in j.pair_refs
        )
        if not still_used:
            ppos = next((i for i, p in enumerate(scene.jl_pairs) if p.pair_id == self.pair_id), None)
            if ppos is not None:
                scene.jl_pairs.remove(ppos)
        return {"FINISHED"}


class JL_OT_export_csv(Operator):
    """Exportiert Joints und Kontaktpaare als '<ifc_stem> GT_Labels.csv'"""
    bl_idname = "jl.export_csv"
    bl_label = "CSV exportieren"

    @classmethod
    def poll(cls, context):
        return len(context.scene.jl_pairs) > 0

    def execute(self, context):
        try:
            path = csv_export.write_csv(context.scene)
        except (RuntimeError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exportiert: {path}")
        return {"FINISHED"}


_CLASSES = (
    JL_OT_add_joint, JL_OT_remove_joint,
    JL_OT_add_pair, JL_OT_remove_pair,
    JL_OT_export_csv,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
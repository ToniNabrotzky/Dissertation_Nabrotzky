"""
UI: N-Panel im 3D-Viewport.
"""

import bpy
from bpy.types import Panel

from .properties import get_joint, get_pair


class JL_PT_main(Panel):
    """Hauptpanel des Joint Labelers"""
    bl_label = "Joint Labeler"
    bl_idname = "JL_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Joint Labeler"  # Tab-Name im N-Panel

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # --- Joints ---------------------------------------------------
        box = layout.box()
        box.label(text="Soll-Joints", icon="OUTLINER_COLLECTION")
        row = box.row(align=True)
        row.prop(scene, "jl_active_joint", text="")
        row.operator("jl.add_joint", text="", icon="ADD")
        row.operator("jl.remove_joint", text="", icon="TRASH")

        # --- Kontaktpaare des aktiven Joints -------------------------
        box = layout.box()
        box.label(text="Kontaktpaare", icon="LINKED")
        box.operator("jl.add_pair", icon="CON_TRACKTO")

        joint = get_joint(scene, scene.jl_active_joint)
        pairs = []
        if joint is not None:
            pairs = [get_pair(scene, r.pair_id) for r in joint.pair_refs]
            pairs = sorted((p for p in pairs if p is not None), key=lambda p: p.index)

        if pairs:
            col = box.column(align=True)
            for p in pairs:
                row = col.row(align=True)
                # Paar-ID + Typen (nicht editierbar)
                sub = row.row(align=True)
                sub.label(text=f"{p.pair_id}: {p.type_a} ↔ {p.type_b}")
                # Status (editierbar, kompakt)
                status_col = row.row(align=True)
                status_col.ui_units_x = 3.5
                status_col.prop(p, "status", text="")
                # Paar aus aktivem Joint entfernen
                op = row.operator("jl.remove_pair", text="", icon="X")
                op.pair_id = p.pair_id
        else:
            box.label(text="Keine Paare für diesen Joint", icon="INFO")

        # --- Export ---------------------------------------------------
        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.operator("jl.export_csv", icon="FILE_TICK")


def register():
    bpy.utils.register_class(JL_PT_main)


def unregister():
    bpy.utils.unregister_class(JL_PT_main)
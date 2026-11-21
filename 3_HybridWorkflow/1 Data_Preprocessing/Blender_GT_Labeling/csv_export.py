"""
CSV-Export der Joints und Kontaktpaare.

Format: Pro Joint werden dessen Paare als Zeilen geschrieben,
danach folgt eine Leerzeile (nur leere Werte) zur besseren Lesbarkeit.
"""

import csv
from pathlib import Path

import bpy

from . import core, ifc_utils
from .properties import get_pair


def build_output_path() -> Path:
    """
    Baut den Zielpfad '<blend_dir>/<ifc_stem> GT_Labels.csv'.

    :raises RuntimeError: Wenn .blend ungespeichert oder kein IFC geladen ist.
    """
    if not bpy.data.filepath:
        raise RuntimeError("Blender-Datei ist noch nicht gespeichert.")
    stem = ifc_utils.get_ifc_stem()
    if not stem:
        raise RuntimeError("Kein IFC-Modell in Bonsai geladen.")
    return Path(bpy.path.abspath("//")) / f"{stem} GT_Labels.csv"


def write_csv(scene) -> Path:
    """
    Schreibt alle Joints samt ihrer Kontaktpaare in die CSV-Datei.

    Sortierung: Joints nach Index, Paare innerhalb eines Joints nach Paar-Index.
    Nach jeder Joint-Gruppe wird eine Leerzeile eingefügt.
    Der Status wird aus der Paar-Property übernommen (nachträglich editierbar).

    :return: Pfad der geschriebenen Datei.
    """
    path = build_output_path()
    empty_row = [""] * len(core.CSV_HEADER)

    # utf-8-sig: Excel-freundlich bei Umlauten in Bauteilnamen
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(core.CSV_HEADER)

        for joint in sorted(scene.jl_joints, key=lambda j: j.index):
            # Referenzierte Paare auflösen und nach Paar-Index sortieren
            pairs = [get_pair(scene, ref.pair_id) for ref in joint.pair_refs]
            pairs = sorted((p for p in pairs if p is not None), key=lambda p: p.index)

            # Joints ohne Paare überspringen (keine leere Gruppe schreiben)
            if not pairs:
                continue

            for p in pairs:
                writer.writerow([
                    joint.joint_id, p.pair_id, p.status or core.DEFAULT_STATUS,
                    p.type_a, p.type_b, p.guid_a, p.guid_b, p.name_a, p.name_b,
                ])
            # Leerzeile nach jedem Joint
            writer.writerow(empty_row)
    return path
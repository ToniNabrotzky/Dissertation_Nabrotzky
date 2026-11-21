"""
Adapter zu Bonsai (BlenderBIM) für IFC-Informationen.

Kapselt alle Zugriffe auf Bonsai/ifcopenshell, damit API-Änderungen
nur hier angepasst werden müssen.
"""

from pathlib import Path
from typing import Optional

import bpy


def _get_ifc_file():
    """Liefert das aktuell geladene ifcopenshell-File-Objekt oder None."""
    try:
        # Bonsai (neuere Versionen)
        import bonsai.tool as tool
        return tool.Ifc.get()
    except Exception:
        pass
    try:
        # Fallback: Legacy-BlenderBIM
        from blenderbim.bim.ifc import IfcStore
        return IfcStore.get_file()
    except Exception:
        return None


def get_ifc_stem() -> Optional[str]:
    """
    Liefert den Dateinamen (ohne Endung) des geladenen IFC-Modells.
    Bevorzugt Bonsais Property 'ifc_file', sonst Fallback via IfcStore.
    """
    try:
        path = bpy.context.scene.BIMProperties.ifc_file
        if path:
            return Path(path).stem
    except AttributeError:
        pass
    try:
        from bonsai.bim.ifc import IfcStore
        if IfcStore.path:
            return Path(IfcStore.path).stem
    except Exception:
        pass
    return None


def get_element_info(obj) -> Optional[dict]:
    """
    Ermittelt IFC-Informationen eines Blender-Objekts.

    :param obj: Blender-Objekt.
    :return: Dict {guid, type, name} oder None, wenn das Objekt kein IFC-Element ist.
    """
    ifc_file = _get_ifc_file()
    if ifc_file is None:
        return None

    # Bonsai speichert die IFC-Step-ID am Objekt als 'BIMObjectProperties.ifc_definition_id'
    try:
        step_id = obj.BIMObjectProperties.ifc_definition_id
    except AttributeError:
        return None
    if not step_id:
        return None

    try:
        entity = ifc_file.by_id(step_id)
    except Exception:
        return None

    guid = getattr(entity, "GlobalId", None)
    if not guid:
        return None

    return {
        "guid": guid,
        "type": entity.is_a(),                        # z. B. 'IfcSlab'
        "name": getattr(entity, "Name", None) or "",  # Bauteilname
    }


def get_selected_ifc_elements(context):
    """
    Prüft die Auswahl und liefert IFC-Infos aller selektierten Objekte.

    :return: (elements, fehlermeldung)
             - elements: Liste von Dicts (nur bei Erfolg gefüllt)
             - fehlermeldung: None bei Erfolg, sonst Text
    """
    selected = list(context.selected_objects)
    if len(selected) != 2:
        return [], f"Genau 2 Objekte auswählen (aktuell: {len(selected)})."

    infos = []
    for obj in selected:
        info = get_element_info(obj)
        if info is None:
            return [], f"'{obj.name}' besitzt keine IFC-GUID."
        infos.append(info)

    if infos[0]["guid"] == infos[1]["guid"]:
        return [], "Beide Objekte haben dieselbe GUID."
    return infos, None
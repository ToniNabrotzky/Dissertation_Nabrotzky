"""
Bonsai Joint Labeler
====================
Blender-Addon zum Erstellen von Soll-Joints und Kontaktpaaren aus
IFC-Elementen (Bonsai) und deren Export als Ground-Truth-Labels (CSV).
"""

bl_info = {
    "name": "Bonsai Joint Labeler",
    "author": "Toni Nabrotzky",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D-Viewport > N-Panel > Joint Labeler",
    "description": "Soll-Joints und Kontaktpaare aus Bonsai-IFC-Elementen labeln und als CSV exportieren",
    "category": "Object",
}

from . import properties, operators, panels

# Reihenfolge ist wichtig: Properties -> Operatoren -> UI
_MODULES = (properties, operators, panels)


def register():
    """Registriert alle Module des Addons in definierter Reihenfolge."""
    for module in _MODULES:
        module.register()


def unregister():
    """Deregistriert alle Module in umgekehrter Reihenfolge."""
    for module in reversed(_MODULES):
        module.unregister()


if __name__ == "__main__":
    register()
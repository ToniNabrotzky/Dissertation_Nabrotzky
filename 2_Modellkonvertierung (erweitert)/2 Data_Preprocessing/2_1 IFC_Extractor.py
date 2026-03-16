import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element
import json
import os
from pathlib import Path

class IfcExtractor:
    def __init__(self, ifc_path):
        self.ifc_path = Path(ifc_path)
        self.model = ifcopenshell.open(str(self.ifc_path))
        self.settings = ifcopenshell.geom.settings()
        # self.settings.set(self.settings.USE_PYTHON_OPENCASCADE, True)

    def get_file_metadata(self):
        """Extrahiert Pfad-Informationen"""
        return{
            "full_name": self.ifc_path.name, # Ergebnis: "21_22 L_TWP_Tragwerksmodell.ifc"
            "stem": self.ifc_path.stem, # Ergebnis: "21_22 L_TWP_Tragwerksmodell"
            "folder": self.ifc_path.parent.name # Ergebnis: "1 Model_Database"
        }
    
    def is_load_bearing(self, product):
        """Prüft, ob das Bauteil als tragend definiert ist"""
        psets = element.get_psets(product)
        for pset_name, properties in psets.items():
            if "LoadBearing" in properties:
                return bool(properties["LoadBearing"])
        # Fallback: Manche Programme exportieren es als "Tragendes Bauteil"
        # Soll alles getestet werden:
        # return True    
        return False
    
    def extract_geometry_features(self, product):
        """Berechnet L, B, H, Position und Kennzahlen"""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            # 1. Position (Schwerpunkt des Bauteils)
            # Wir nutzen die Matrix der Shape-Geometrie für präzise Daten
            matrix = shape.transformation.matrix.data
            pos_x, pos_y, pos_z = matrix[3], matrix[7], matrix[11] # Translation aus der Transformationsmatrix

            # 2. Bounding Box (AABB als Annäherung für L, B, H)
            verts = shape.geometry.verts
            x_coords = [verts[i] for i in range(0, len(verts), 3)]
            y_coords = [verts[i+1] for i in range(0, len(verts), 3)]
            z_coords = [verts[i+2] for i in range(0, len(verts), 3)]

            l_abs = max(x_coords) - min(x_coords)
            b_abs = max(y_coords) - min(y_coords)
            h_abs = max(z_coords) - min(z_coords)

            # 3. Normiertes L, B, H Tupel
            max_dim = max(l_abs, b_abs, h_abs, 1e-6) # 1e-6 verhindert Division durch Null
            normalized_tuple = (round(l_abs/max_dim, 3), round(b_abs/max_dim, 3), round(h_abs/max_dim, 3))

            # 4. Volumen und Fläche (über OpenCascade falls verfügbar)
            # Vereinfachte Dummy-Werte, falls keine BaseQuantities vorhanden sind
            # (Hier könnte man später eine spezialisierte Funktion einbauen)

            volume = 0.0
            area = 0.0

            qsets = element.get_psets(product, qtos_only=True)
            for qset in qsets.values():
                volume = qset.get('NetVolume', volume)
                area = qset.get('NetSurfaceArea', area)

            ratio_av = round(area / volume, 4) if volume > 0 else 0

            return {
                "position": (pos_x, pos_y, pos_z),
                "dimensions": (round(l_abs, 3), round(b_abs, 3), round(h_abs, 3)),
                "norm_dimensions": normalized_tuple,
                "volume": round(volume, 4),
                "area": round(area, 4),
                "ratio_av": (ratio_av)
            }
        
        except Exception as e:
            return None
        
    def process(self):
        """Hauptprozess: Filtert und extrahiert Daten"""
        extracted_nodes = []
        products = self.model.by_type("IfcProduct")

        print(f"Analysiere {len(products)} Objekte...")

        counter = 0
        for prod in products:
            if self.is_load_bearing(prod):
                geom_data = self.extract_geometry_features(prod)

                if geom_data:
                    node= {
                        "node_id": counter,
                        "guid": prod.GlobalId,
                        "name": prod.Name or "Unnamed",
                        "entity": prod.is_a(),
                        "features": geom_data
                    }
                    extracted_nodes.append(node)
                    counter += 1
        
        return extracted_nodes

def save_to_json(data, metadata, output_dir):
    """Speichert die Liste sauber als JSON-Datei"""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{metadata['stem']} Nodes.json"

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f">>>ERFOLG! {len(data)} Knoten extrahiert in: {file_path}")
    return




if __name__ == "__main__":
    # 1. Pfad zum IFC-Modell
    ifc_file_input = "../1 Model_Database/21_22 L_TWP_Tragwerksmodell.ifc"

    # 2. Extraktor initialisieren und Daten extrahieren
    if Path(ifc_file_input).exists():
        extractor = IfcExtractor(ifc_file_input)
        metadata = extractor.get_file_metadata()
        nodes = extractor.process()

    # 3. Ergebnisse als JSON speichern
    output_directory = "./2_1 Graph_Generation"
    save_to_json(nodes, metadata, output_directory)



# from pathlib import Path

# # Dein Eingabepfad
# input_path_str = "../1 Model_Database/21_22 L_TWP_Tragwerksmodell.ifc"
# path_obj = Path(input_path_str)

# # 1. Der reine Dateiname mit Endung
# file_name = path_obj.name 
# # Ergebnis: "21_22 L_TWP_Tragwerksmodell.ifc"

# # 2. Der Dateiname OHNE Endung (ideal für Export-Dateinamen)
# file_stem = path_obj.stem 
# # Ergebnis: "21_22 L_TWP_Tragwerksmodell"

# # 3. Überprüfung, ob die Datei wirklich existiert
# if path_obj.exists():
#     print(f"Datei {file_stem} wurde gefunden.")
# else:
#     print("Pfad ist ungültig!")
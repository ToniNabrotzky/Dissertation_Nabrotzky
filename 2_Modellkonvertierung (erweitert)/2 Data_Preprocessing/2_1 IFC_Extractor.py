import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element
import json
import os
import numpy as np
from pathlib import Path

class IfcExtractor:
    def __init__(self, ifc_path):
        self.ifc_path = Path(ifc_path)
        self.model = ifcopenshell.open(str(self.ifc_path))
        self.settings = ifcopenshell.geom.settings()
        # self.settings.set("use-python-opencascade", True)
        # self.settings.set(self.settings.USE_PYTHON_OPENCASCADE, True)
        # Sicherstellen, dass das Mesh trianguliert ist
        # self.settings.set(self.settings.USE_WORLD_COORDS, True)

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
    
    def _estimate_mesh_properties(self, verts, faces):
        """Schätzt daas Volumen und die Overfläche auf Mesh-Daten
        Oberfläche (Surface Area): Ein Bauteil-Mesh besteht aus vielen kleinen Dreiecken. 
        Die Gesamtfläche ist einfach die Summe der Flächen aller Dreiecke.
        Die Fläche eines Dreiecks mit den Eckpunkten A, B, C berechnet man über das Kreuzprodukt:
            Fläche = 1/2 |(B-A) X (C-A)|
        
        Volumen (Mesh Volume): Hier wird der Satz des Gauß (Divergenzsatz) benutzt.
        Für jedes Dreieck wird das "signierte Volumen" einer Pyramide berechnet,
        die vom Koordinatenursprung (0, 0, 0) aufgespannt wird.
            Zeigt das Dreieck "vom Ursprung weg", ist das Teilvolumen positiv
            Zeit es "zum Ursprung hin", ist es negativ
        Die Summe ergibt exakt das Volumen des geschlossenen Körpers:
            V = SUM( 1/6*(A.(AxB)))
        """
        
        total_area = 0.0
        total_volume = 0.0
        
        # Gruppiere flache Liste verts in (x,y,z) Punkte
        nodes = [np.array(verts[i:i+3]) for i in range(0, len(verts), 3)]
        
        # Iteriere über die Faces (jeweils 3 Indizes bilden ein Dreieck)
        for i in range(0, len(faces), 3):
            try:
                # Hole die drei Eckpunkte des Dreiecks
                idx_a, idx_b, idx_c = faces[i], faces[i+1], faces[i+2]
                A, B, C = nodes[idx_a], nodes[idx_b], nodes[idx_c]
                
                # --- Flächenberechnung (Kreuzprodukt) ---
                cross_product = np.cross(B - A, C - A)
                area = 0.5 * np.linalg.norm(cross_product)
                total_area += area
                
                # --- Volumenberechnung (Spatprodukt/Signed Volume) ---
                # Volumen = 1/6 * |A · (B x C)|
                volume_contribution = np.dot(A, np.cross(B, C)) / 6.0
                total_volume += volume_contribution
            except IndexError:
                continue

        return round(total_area, 4), round(abs(total_volume), 4)


    def extract_geometry_features(self, product):
        """Berechnet L, B, H, Position und Kennzahlen"""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            verts = shape.geometry.verts
            faces = shape.geometry.faces # Indices der Dreiecke
            # print("\tshape:", shape) # Debug
            # print("\tverts:", verts) # Debug
            # print("\faces:", faces) # Debug

            if not shape or not verts or not faces:
                return None
            # print("\tNOT SHAP, NOT VERTS: CHECKED") # Debug
            
            # 1. Position aus Welt-Koordinaten
            # Transformationsmatrix enthält Position und Rotation
            matrix = shape.transformation.matrix
            # print("\tmatrix: ", matrix) # Debug
            """Erklärung zum Aufbau der Matrix (R= Rotation, X/Y/Z sind Koordinaten)
             matrix = [
                1, 0, 0, 29.08
                0, 1, 0, 25,38
                0, 0, 1, -1,74
                0, 0, 0, 1.0
                ]
             Row-Major-Form: [R, R, R, X, R, R, R, Y, R, R, R, Z, 0, 0, 0, 1]
             Column-Major-Form: [R, R, R, 0,  R, R, R, 0,  R, R, R, 0,  X, Y, Z, 1]
            Beipiel als Liste: (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 29.0812976912, 25.3782442783, -1.735, 1.0)
            Matrix ist im Column-Major-Format, daher:"""
            # pos_x, pos_y, pos_z = matrix[3], matrix[7], matrix[11] # Indices für Row-Variante
            pos_x, pos_y, pos_z = matrix[12], matrix[13], matrix[14] # Indices für Column-Variante
            position = (round(pos_x, 3), round(pos_y, 3), round(pos_z, 3))
            print(f"\tPosition: {position}") # Debug

            # 2. Bounding Box (AABB als Annäherung für L, B, H)
            x_coords = [verts[i] for i in range(0, len(verts), 3)]
            y_coords = [verts[i+1] for i in range(0, len(verts), 3)]
            z_coords = [verts[i+2] for i in range(0, len(verts), 3)]

            l_abs = max(x_coords) - min(x_coords)
            b_abs = max(y_coords) - min(y_coords)
            h_abs = max(z_coords) - min(z_coords)            
            dimensions = (round(l_abs, 3), round(b_abs, 3), round(h_abs, 3))
            print(f"\tDimensions: {dimensions}") # Debug

            # 3. Normiertes L, B, H Tupel
            max_dim = max(l_abs, b_abs, h_abs, 1e-6) # 1e-6 verhindert Division durch Null
            normalized_tuple = (round(l_abs/max_dim, 3), round(b_abs/max_dim, 3), round(h_abs/max_dim, 3))
            print(f"\tNormalized Dimensions: {normalized_tuple}") # Debug

            # 4. Volumen und Fläche (über OpenCascade falls verfügbar)
            # Versuch 1: Aus den BaseQuantities des Modells lesen
            volume = 0.0
            area = 0.0

            qsets = element.get_psets(product, qtos_only=True)
            for qset in qsets.values():
                volume = qset.get('NetVolume', qset.get('Volume', volume))
                area = qset.get('NetSurfaceArea', qset.get('SurfaceArea', area))

            # Versuch 2: Über die Mesh-Daten schätzen, falls Volumen oder Fläche nicht gefunden
            if volume <= 0 or area <= 0:
                est_area, est_vol = self._estimate_mesh_properties(verts, faces)
                area = area if area > 0 else est_area
                volume = volume if volume > 0 else est_vol
            print(f"\tvolume: {volume}, area: {area}") # Debug

            ratio_av = round(area / volume, 4) if volume > 0 else 0
            print(f"\tratio_av: {ratio_av}") # Debug

            return {
                "position": position,
                "dimensions": dimensions,
                "norm_dimensions": normalized_tuple,
                "volume": round(volume, 4),
                "area": round(area, 4),
                "ratio_av": (ratio_av)
            }
        
        except Exception as e:
            print(f"    FEHLER: {e}")
            return None
        
    def process(self):
        """Hauptprozess: Filtert und extrahiert Daten"""
        nodes  = []
        products = self.model.by_type("IfcProduct")

        # Debugging
        count_all = len(products)
        count_load_bearing = 0
        count_with_geom = 0

        print(f"Analysiere {len(products)} Objekte...")

        for i, prod in enumerate(products):
            print(f"\nPRÜFE OBJEKT {i}: {prod.is_a()} - {prod.Name}") # Debug
            print(f"\t{prod}") # Debug

            # SCHRITT 1: Filter
            if not self.is_load_bearing(prod):
                continue
            count_load_bearing += 1

            # SCHRITT 2: Semantik
            name = prod.Name or "Unnamed"
            guid = prod.GlobalId
            entity = prod.is_a()

            # SCHRITT 3: Geometrie
            geom_data = self.extract_geometry_features(prod)
            if not geom_data:
                continue
            count_with_geom += 1

            nodes.append({
                "node_id": len(nodes),
                "guid": guid,
                "name": name,
                "entity": entity,
                "features": geom_data
            })        
        
        print(f"\n--- ANALYSE ERGEBNIS ---")
        print(f"Gesamtanzahl IfcProducts: {count_all}")
        print(f"Davon 'LoadBearing':     {count_load_bearing}")
        print(f"Davon mit Geometrie:     {count_with_geom}")
        return nodes

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
    # ifc_file_input = "../1 Model_Database/20200820IFC4_Convenience_store_Renga_4.1.ifc"

    # 2. Extraktor initialisieren und Daten extrahieren
    if Path(ifc_file_input).exists():
        extractor = IfcExtractor(ifc_file_input)
        metadata = extractor.get_file_metadata()
        nodes = extractor.process()

    # 3. Ergebnisse als JSON speichern
    output_directory = "./2_1 Graph_Generation"
    save_to_json(nodes, metadata, output_directory)
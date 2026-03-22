import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element
import json
import numpy as np
import os


def main():
    # 1. Pfad zum IFC-Modell
    ifc_name = [
        # "21_22 L_TWP_Tragwerksmodell",                  # immer aktiv lassen für Tests
        # "22_23 LTWP_221127_Ifc-Allplan", 
        # "23_24 LTWP-V__Dachtragwerk",                 # erledigt
        # "24_25 LTWP-V_250122_02_Vordachmodell",       # aussortiert - zu simpel
        # "24_25 LTWP-V_250122_Kellermodell",           # aussortiert - zu simpel
        # "20200820IFC4_Convenience_store_Renga_4.1"    # aussortiert - zu simpel
        # "47L" 
        "20220421MODEL REV01",                        # erledigt
        # "202102183458-Model",                         # erledigt
        # "ARchiCAD__20200518Yangsan Pr-HARDWARE",      # aussortiert - unhandlich, schlecht modelliert
        # "Grethes-hus-bok-2",                          # erledigt
        # "Ifc2x3_SampleCastle",                        # erledigt
        # "Vectorworks2016-IFC2x3-EQUA_IDA_ICE"         # erledigt
    ]

    # Klasse instanziieren und Prozess starten
    for ifc in ifc_name:
        extractor = IfcExtractor(ifc)
        extractor.process_all()
    return





class IfcExtractor:
    def __init__(self, model_stem):
        self.model_stem = model_stem

        # --- Pfad-Definition ---
        self.base_dir = os.path.dirname(__file__)
        # Pfad zur Modelldatenbank (ein Ordner hoch, dann in Database)
        self.db_dir = os.path.normpath(os.path.join(self.base_dir, "..", "1 Model_Database"))
        # Ausgabeordner
        self.dir_2_1 = os.path.join(self.base_dir, "2_1 Graph_Generation")

        # Dateipfade zusammenbauen
        self.path_ifc = os.path.join(self.db_dir, f"{self.model_stem}.ifc")
        self.output_json = os.path.join(self.dir_2_1, f"{self.model_stem} Nodes.json")

        # Validierung und Initialisierung
        if not os.path.exists(self.path_ifc):
            raise FileNotFoundError(f"IFC-Datei nicht gefunden unter: {self.path_ifc}")
        
        print(f"Lade Modell: {self.model_stem}...")
        self.model = ifcopenshell.open(str(self.path_ifc))
        self.settings = ifcopenshell.geom.settings()
        # self.settings.set("use-python-opencascade", True)
        # self.settings.set(self.settings.USE_PYTHON_OPENCASCADE, True)
        # Sicherstellen, dass das Mesh trianguliert ist
        # self.settings.set(self.settings.USE_WORLD_COORDS, True)

        # Zähler für die Analyse
        self.count_all = 0
        self.count_load_bearing = 0
        self.count_with_geom = 0
        self.entity_counts = {}


    # def get_file_metadata(self):
    #     """Extrahiert Pfad-Informationen"""
    #     return{
    #         "full_name": self.ifc_path.name, # Ergebnis: "21_22 L_TWP_Tragwerksmodell.ifc" # type: ignore
    #         "stem": self.ifc_path.stem, # Ergebnis: "21_22 L_TWP_Tragwerksmodell"
    #         "folder": self.ifc_path.parent.name # Ergebnis: "1 Model_Database"
    #     }
    
    
    def is_load_bearing(self, product):
        """Prüft, ob das Bauteil als tragend definiert ist"""
        psets = element.get_psets(product)
        for pset_name, properties in psets.items():
            if "LoadBearing" in properties:
                return bool(properties["LoadBearing"])
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
        nodes_list = [np.array(verts[i:i+3]) for i in range(0, len(verts), 3)]
        
        # Iteriere über die Faces (jeweils 3 Indizes bilden ein Dreieck)
        for i in range(0, len(faces), 3):
            try:
                # Hole die drei Eckpunkte des Dreiecks
                idx_a, idx_b, idx_c = faces[i], faces[i+1], faces[i+2]
                A, B, C = nodes_list[idx_a], nodes_list[idx_b], nodes_list[idx_c]
                
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
        """Berechnet L, B, H, Position und Kennzahlen für ein einzelnes Produkt"""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            # print("\tshape:", shape) # Debug
            if not shape or not shape.geometry: # type: ignore
                return None
            
            verts = shape.geometry.verts # type: ignore
            faces = shape.geometry.faces # type: ignore # Indices der Dreiecke
            # print("\tverts:", verts) # Debug
            # print("\faces:", faces) # Debug
            if not verts or not faces:
                return None
            
            # 1. Position aus Welt-Koordinaten
            # Transformationsmatrix enthält Position und Rotation
            matrix = shape.transformation.matrix # type: ignore
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
            origin = (round(pos_x, 3), round(pos_y, 3), round(pos_z, 3))
            origin_x = origin[0]
            origin_y = origin[1]
            origin_z = origin[2]
            print(f"\tPosition: {origin}") # Debug

            # 2. Kordinaten trennen für die Berechnungen
            x_coords = [verts[i] for i in range(0, len(verts), 3)]
            y_coords = [verts[i+1] for i in range(0, len(verts), 3)]
            z_coords = [verts[i+2] for i in range(0, len(verts), 3)]
            num_verts = len(x_coords)

            # 3. Centroid berechnen (Mittelpunkt der Vertices + Origin)
            # Da verts lokal zum Matrix-Urpsrung sind, muss man sie zur Urpsrungsposition hinzurechnen.
            c_x = (sum(x_coords) / num_verts) + pos_x
            c_y = (sum(y_coords) / num_verts) + pos_y
            c_z = (sum(z_coords) / num_verts) + pos_z
            centroid = (round(c_x, 3), round(c_y, 3), round(c_z, 3))
            centroid_x = centroid[0]
            centroid_y = centroid[1]
            centroid_z = centroid[2]
            print(f"\tCentroid: {centroid}") # Debug

            # 4. Bounding Box (AABB als Annäherung für L, B, H)
            l_abs = max(x_coords) - min(x_coords)
            b_abs = max(y_coords) - min(y_coords)
            h_abs = max(z_coords) - min(z_coords)            
            dimensions = (round(l_abs, 3), round(b_abs, 3), round(h_abs, 3))
            print(f"\tDimensions: {dimensions}") # Debug

            # 5. Normiertes L, B, H Tupel
            max_dim = max(l_abs, b_abs, h_abs, 1e-6) # 1e-6 verhindert Division durch Null
            normalized_dim = (round(l_abs/max_dim, 3), round(b_abs/max_dim, 3), round(h_abs/max_dim, 3))
            print(f"\tNormalized Dimensions: {normalized_dim}") # Debug

            # 6. Volumen und Fläche (über OpenCascade falls verfügbar)
            # Versuch 1: Aus den BaseQuantities des Modells lesen
            area = 0.0
            volume = 0.0

            qsets = element.get_psets(product, qtos_only=True)
            for qset in qsets.values():
                area = qset.get('NetSurfaceArea', qset.get('SurfaceArea', area))
                volume = qset.get('NetVolume', qset.get('Volume', volume))

            # Versuch 2: Über die Mesh-Daten schätzen, falls Volumen oder Fläche nicht gefunden
            if volume <= 0 or area <= 0:
                est_area, est_vol = self._estimate_mesh_properties(verts, faces)
                area = area if area > 0 else est_area
                volume = volume if volume > 0 else est_vol
            print(f"\tvolume: {volume}, area: {area}") # Debug

            ratio_av = round(area / volume, 4) if volume > 0 else 0
            print(f"\tratio_av: {ratio_av}") # Debug

            # Rückgabe der geometrischen Attribute
            return {
                "origin": origin,
                "origin_x": origin_x,
                "origin_y": origin_y,
                "origin_z": origin_z,
                "centroid": centroid,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "centroid_z": centroid_z,
                "dimensions": dimensions,
                "norm_dimensions": normalized_dim,
                "area": round(area, 4),
                "volume": round(volume, 4),
                "ratio_av": (ratio_av),
                # "mesh": {
                #     "vertices": verts,
                #     "faces": faces
                # }
            }
        
        except Exception as e:
            print(f"\tFEHLER: {e}")
            return None
        

    def process_all(self):
        """Hauptprozess: Filtert und extrahiert Daten"""
        nodes  = []
        products = self.model.by_type("IfcProduct")

        self.count_all = len(products)
        print(f"Analysiere {self.count_all} Objekte...")

        for i, prod in enumerate(products):
            print(f"\nPRÜFE OBJEKT {i}: {prod.is_a()} - {prod.Name}") # Debug
            print(f"\t{prod}") # Debug

            # SCHRITT 1: Filter Tragend
            if not self.is_load_bearing(prod):
                continue
            self.count_load_bearing += 1

            # SCHRITT 2: Geometrie
            geom_data = self.extract_geometry_features(prod)
            if not geom_data:
                continue
            self.count_with_geom += 1
            
            # SCHRITT 3: Semantik
            name = prod.Name or "Unnamed"
            guid = prod.GlobalId
            entity = prod.is_a()
            self.entity_counts[entity] = self.entity_counts.get(entity, 0) + 1

            nodes.append({
                "node_id": len(nodes),
                "guid": guid,
                "name": name,
                "entity": entity,
                "features": geom_data
            })

        self.save_to_json(nodes)


    def save_to_json(self, nodes):
        """Exportiert die Daten als JSON-Datei"""
        if not os.path.exists(self.dir_2_1):
            os.makedirs(self.dir_2_1)
        
        export_data = {
            "graph_info": {
                "model_stem": self.model_stem,
                "entites": self.entity_counts
            },
            "stats_2_1": {
                "count_all_ifc_products": self.count_all,
                "count_load_bearing": self.count_load_bearing,
                "count_with_geometry": self.count_with_geom
            },
            "nodes": nodes
        }
        
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=4)
        
        print(f">>> ERFOLG! {len(nodes)} Knoten extrahiert in: {self.output_json}")
        
        print(f"\n--- ANALYSE ERGEBNIS für {self.model_stem} ---")
        print(f"Anzahl IfcProducts: \t{self.count_all}")
        print(f"Davon 'LoadBearing': \t{self.count_load_bearing}")
        print(f"Davon Geometrie: \t{self.count_with_geom}")
        print(f"Datei gespeichert: \t{os.path.basename(self.output_json)}")
        return nodes





if __name__ == "__main__":
    main()

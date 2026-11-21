import os
import glob
import json
import math
import numpy as np
from scipy.spatial import cKDTree # type: ignore
import ifcopenshell
import ifcopenshell.geom
from datetime import datetime

# =========================================================
# MAIN & STEUERUNG
# =========================================================
def main():
    """
    Zentrale Steuerung des Skripts.
    Hier können die auszuwertenden IfcStem-Namen definiert werden.
    Ist die Liste leer, werden alle IFC-Modelle im Ordner verarbeitet.
    """
    ifc_names = [
        "21_22 L_TWP_Tragwerksmodell",
        "23_24 LTWP-V__Dachtragwerk",
        "20220421MODEL REV01",
        "202102183458-Model",
        "Grethes-hus-bok-2",
        "Grethes-hus-bok-2",
        # "model_1",
        # "model_3_1D",
        # "model_3_2D",
        # "model_4",
        # "model_7",
        # "model_8",
        # "model_9", # 5,0m erforderlich
        # "model_10_1D",
        # "model_10_2D",
        "Parametrisches Modell Index_00000",
        "Parametrisches Modell Index_00050",
        "Parametrisches Modell Index_00055",
        "Parametrisches Modell Index_00100",
        "Parametrisches Modell Index_00250",
    ]
    
    pipeline = GraphRepairPipeline(ifc_names)
    pipeline.run()

# =========================================================
# KONFIGURATION
# =========================================================
# Selber RBF_GAMMA Wert wie in der Vorlage zur korrekten Berechnung
RBF_GAMMA = 0.0001
RESOLUTION = 0.025  # 2,5 cm Abtastung für Kanten (wie in der Vorlage)

class GraphRepairPipeline:
    """
    Klasse zum Reparieren von fehlerhaften Distanzen und Proximity-Activations
    in bereits exportierten JSON-Graphen.
    """
    def __init__(self, model_names=None):
        # Pfade analog zur Vorlage definieren
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_db = os.path.normpath(os.path.join(self.base_dir, "..", "0 Model_Database"))
        self.dir_graphs = os.path.join(self.base_dir, "2_4 Graph_Generation")
        
        # Auszuwertende Modelle übernehmen (leere Liste = alle)
        self.model_names = model_names if model_names is not None else []
        
        # IFC Geometrie-Einstellungen initialisieren
        self.settings = ifcopenshell.geom.settings()
        
    def log(self, msg):
        """Hilfsfunktion für formatierten Konsolen-Output."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {msg}")

    def run(self):
        """Sucht die auszuwertenden JSON-Dateien und startet den Reparatur-Vorgang."""
        if not self.model_names:
            # Falls Liste leer ist, suche alle JSON Graphen im Ordner
            search_pattern = os.path.join(self.dir_graphs, "*Graph.json")
            json_files = glob.glob(search_pattern)
        else:
            # Sonst generiere Dateipfade aus der ifc_names Liste
            json_files = [os.path.join(self.dir_graphs, f"{name} Graph.json") for name in self.model_names]
            
        # Prüfen ob die generierten/gefundenen Pfade wirklich existieren
        valid_files = [f for f in json_files if os.path.exists(f)]
        
        if not valid_files:
            self.log(f"Keine zu verarbeitenden JSON-Dateien gefunden.")
            return

        self.log(f"Starte Reparatur für {len(valid_files)} Graphen...")
        
        for json_path in valid_files:
            self.repair_single_graph(json_path)
            
        self.log("Alle Graphen wurden erfolgreich repariert!")

    def repair_single_graph(self, json_path):
        """Verarbeitet einen einzelnen Graphen, misst Distanzen neu und speichert ihn."""
        filename = os.path.basename(json_path)
        self.log(f"{'-'*50}\nPrüfe Datei: {filename}")
        
        # 1. JSON Einlesen
        with open(json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
            
        model_stem = graph_data.get("graph_info", {}).get("model_stem", "")
        if not model_stem:
            # Fallback, falls 'model_stem' nicht exakt so heißt
            model_stem = filename.replace(" Graph.json", "")
            
        ifc_path = os.path.join(self.dir_db, f"{model_stem}.ifc")
        if not os.path.exists(ifc_path):
            self.log(f"ÜBERSPRUNGEN: Zugehörige IFC {ifc_path} nicht gefunden!")
            return
            
        self.log(f"Lade IFC-Modell: {model_stem}")
        model = ifcopenshell.open(str(ifc_path))
        
        # 2. Benötigte GUIDs aus den Kanten des JSON extrahieren
        edges = graph_data.get("edges", [])
        needed_guids = set()
        for edge in edges:
            meta = edge.get("meta", {})
            if "guid_a" in meta and "guid_b" in meta:
                needed_guids.add(meta["guid_a"])
                needed_guids.add(meta["guid_b"])
                
        self.log(f"Extrahiere Geometrie für {len(needed_guids)} referenzierte Bauteile...")
        
        # 3. Geometrie nur für die Bauteile berechnen, die im Graph existieren
        geom_cache = {}
        for guid in needed_guids:
            product = model.by_guid(guid)
            if product:
                geom_data = self.extract_test_points(product)
                if geom_data is not None:
                    geom_cache[guid] = geom_data
                else:
                    self.log(f"WARNUNG: Konnte keine Geometrie für GUID {guid} erstellen.")
            else:
                self.log(f"WARNUNG: Bauteil mit GUID {guid} in IFC nicht gefunden.")

        # 4. Kanten durchgehen und Werte neu berechnen
        recalculated_count = 0
        self.log("Berechne Distanzen und Proximity Activation neu...")
        
        # KD-Trees für alle zwischengespeicherten Geometrien vorbereiten
        trees = {guid: cKDTree(pts) for guid, pts in geom_cache.items()}

        for edge in edges:
            guid_a = edge.get("meta", {}).get("guid_a")
            guid_b = edge.get("meta", {}).get("guid_b")
            
            if guid_a in geom_cache and guid_b in geom_cache:
                pts_a = geom_cache[guid_a]
                pts_b = geom_cache[guid_b]
                
                # Exakte Messung via KDTree
                dist_a_to_b, _ = trees[guid_a].query(pts_b, k=1, workers=-1)
                final_distance_m = np.min(dist_a_to_b) # in [m]
                
                # Wandle die Distanz von [m] in [mm] um (Wie in der Vorlage)
                final_dist_mm = final_distance_m * 1000.0
                
                # Proximity Activation berechnen
                proximity = math.exp(-RBF_GAMMA * (final_dist_mm ** 2))
                
                # Werte im JSON-Objekt überschreiben
                edge["features"]["distance"] = round(final_dist_mm, 4)
                edge["features"]["proximity_activation"] = round(proximity, 6)
                recalculated_count += 1
                
        # 5. JSON Datei überschreiben
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)
            
        self.log(f"ERFOLG: {recalculated_count} Kanten in '{filename}' repariert und überschrieben.")

    def extract_test_points(self, product):
        """
        Extrahiert die abgetastete Punktewolke ('test_points') eines Bauteils 
        exakt so, wie in der Vorlage _14_ definiert.
        """
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            if not shape or not getattr(shape, 'geometry', None):
                return None
            
            verts = shape.geometry.verts #type: ignore
            faces = shape.geometry.faces #type: ignore
            matrix_1d = shape.transformation.matrix #type: ignore
            
            # Vertices in numpy arrays umwandeln und in globale Koordinaten transformieren
            local_verts = np.array(verts).reshape(-1, 3)
            matrix_4x4 = np.array(matrix_1d).reshape(4, 4)
            homogeneous_verts = np.hstack((local_verts, np.ones((local_verts.shape[0], 1))))
            global_verts = homogeneous_verts.dot(matrix_4x4)[:, :3]
            
            # Eindeutige Kanten extrahieren 
            faces_array = np.array(faces).reshape(-1, 3)
            edges_set = set()
            for f in faces_array:
                edges_set.add(tuple(sorted((f[0], f[1]))))
                edges_set.add(tuple(sorted((f[1], f[2]))))
                edges_set.add(tuple(sorted((f[2], f[0]))))

            # Geometrie an Kanten abtasten (exakt wie in Vorlage)                
            sampled_points = []
            for e in edges_set:
                p1 = global_verts[e[0]]
                p2 = global_verts[e[1]]
                length = np.linalg.norm(p2 - p1)

                if length > RESOLUTION:
                    num_steps = int(np.ceil(length / RESOLUTION))
                    if num_steps > 1000:
                        num_steps = 1000

                    t = np.linspace(0, 1, num_steps + 1).reshape(-1, 1)
                    sampled = p1 + t * (p2 - p1)
                    sampled_points.append(sampled)
            
            if sampled_points:
                # Füge die abgetasteten Punkte den ursprünglichen Vertices hinzu
                test_points = np.vstack(sampled_points)
            else:
                test_points = global_verts

            return test_points
            
        except Exception as e:
            # Bei Geometriefehlern None zurückgeben
            return None

if __name__ == "__main__":
    main()
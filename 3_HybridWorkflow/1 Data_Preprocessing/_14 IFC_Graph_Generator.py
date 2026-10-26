"""
_14 IFC_Graph_Generator.py
==========================
Struktur des Skripts:
1. MAIN & STEUERUNG
2. KONFIGURATION
3. PIPELINE KLASSE (GNNPipeline)
   Teil A - Versuch #2 GNN-Machbarkeitsstudie:
       Generierung des Kontaktgraphen (Knoten & Kanten inkl. Merkmale)
       A.1 Dateneingang (IFC-Extraktion & Labels)
       A.2 Knoten & Knotenmerkmale (homogener / heterogener Graph)
       A.3 Kanten & Kantenmerkmale (Flächenprojektion & Verdeckung)
       A.4 Graph-Export & Visualisierung
   Teil B - Versuch #1 RBC-Benchmark:
       Identifikation und Bewertung von Joints (Hyperkanten)
       B.1 Soll-Joints: Rekonstruktion aus der Ground Truth (IFC-basiert)
       B.2 Ist-Joints: Extraktion aus dem Analysemodell
       B.3 Validierung: Matching, Fehlercodes (E0-E7), Statistik
"""
import os
import glob
import json
import math
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree #type: ignore
from datetime import datetime
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element
import ifcopenshell.util.placement as placement

# =========================================================
# MAIN & STEUERUNG
# =========================================================
def main():
    """
    Zentrale Steuerung des Skripts.
    Hier können die auszuwertenden IfcStem-Namen definiert werden.
    Ist die Liste leer, werden alle IFC-Modelle im Database-Ordner verarbeitet.
    """
    ifc_names = [
        # "21_22 L_TWP_Tragwerksmodell",
        # "23_24 LTWP-V__Dachtragwerk",
        # "20220421MODEL REV01",
        # "202102183458-Model",
        # "Grethes-hus-bok-2",
        # "Grethes-hus-bok-2",
        # "model_1",
        # "model_3_1D",
        # "model_3_2D",
        # "model_4",
        # "model_7",
        # "model_8",
        "model_9", # 5,0m erforderlich
        # "model_10_1D",
        # "model_10_2D",
        # "Parametrisches Modell Index_00000",
        # "Parametrisches Modell Index_00050",
        # "Parametrisches Modell Index_00055",
        # "Parametrisches Modell Index_00100",
        # "Parametrisches Modell Index_00250",
    ]
    
    pipeline = GNNPipeline(ifc_names)
    pipeline.run()

# =========================================================
# KONFIGURATION
# =========================================================
# --- Versuch #2 GNN-Machbarkeitsstudie ---
# A.1 Dateneingang: IFC-Extraktion
ALLOWED_ENTITIES = [
    "IfcBeam", "IfcColumn", "IfcSlab", "IfcWall", "IfcWallStandardCase",
    "IfcFooting", "IfcMember", "IfcPile", "IfcPlate", "IfcBuildingElementProxy"
]

# A.3 Kanten: Distanzmessung und Gauß-Kernel
RBF_GAMMA = 0.0001 # Hyperparameter - Umwandlung euklidische Distanz in GNN-Merkmal (Feature). Default = 0.0001
MAX_DIST = 3.0  # Einheitliche Grenze in [m] für Distanzberechnung (zwischen AABBs und Exakt-Messung). Default = 3.0

# A.4 Visualisierung: Farben und Marker je IFC-Entität
VIS_MAPPING = {
    "IfcBeam": ("#2E8B57", "h"), "IfcColumn": ("#FF4500", "p"),
    "IfcSlab": ("#B38719", "o"),
    "IfcWall": ("#808080", "H"), "IfcWallStandardCase": ("#A9A9A9", "D"),
    "IfcFooting": ("#8B4513", "s"), "IfcMember": ("#4682B4", "v"),
    "IfcPile": ("#000000", "|"), "IfcPlate": ("#DDA0DD", "^"),
    "IfcBuildingElementProxy": ("#FFD700", "*")
}

# --- Versuch #1 RBC-Benchmark ---
# B.1 Soll-Joints: Rekonstruktion aus der Ground Truth
JOINT_EPSILON = 0.005  # [m] Kontakttoleranz eps in [m]. HINWEIS: Sensitivitätsstudie (Empfehlung: 1-5 mm). Default = 0.005
JOINT_TAU = 0.2        # [-] Schwellenwert tau für Dimensionalitätsverhältnisse L2/L1 und L3/L1 (linien-/flächen-/punktförmig). Default = 0.2
JOINT_DELTA_SCALE = 1.0 # [-] Skalierung der Joint-Toleranz delta mit der größten Bauteildicke max(t_a, t_b). Default = 1.0

# =========================================================
# PIPELINE KLASSE
# =========================================================
class GNNPipeline:
    """
    Hauptklasse des Hybrid-Workflows. Aufbau in zwei Versuchsteilen:

    Versuch #2 GNN-Machbarkeitsstudie (Teil A):
        Generierung des Kontaktgraphen: IFC-Extraktion, Berechnung der
        Knoten- und Kantenmerkmale, Export und Visualisierung des Graphen.
    Versuch #1 RBC-Benchmark (Teil B):
        Identifikation und Bewertung von Joints (Hyperkanten):
        B.1 Soll-Joints aus der Ground Truth (IFC-modellbasiert),
        B.2 Ist-Joints aus dem Analysemodell,
        B.3 Validierung (Matching, Fehlercodes E0-E7, Statistik).
    """

    # ---------------------------------------------------------
    # INITIALISIERUNG, LOGGING & STEUERUNG
    # ---------------------------------------------------------
    def __init__(self, model_names=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_db = os.path.normpath(os.path.join(self.base_dir, "..", "0 Model_Database"))
        self.dir_gt = os.path.join(self.base_dir, "1 Ground_Truth")
        self.dir_rbc = os.path.join(self.base_dir, "3 RBC_Labels")
        self.dir_out = os.path.join(self.base_dir, "4 Graph_Generation")
        
        os.makedirs(self.dir_out, exist_ok=True)

        if not model_names:
            search_pattern = os.path.join(self.dir_db, "*.ifc")
            self.model_names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(search_pattern)]
        else:
            self.model_names = model_names
            
        self.logs = []


    def log(self, msg, print_console=True):
        """Speichert eine Nachricht als Protokoll und gibt sie optional in der Konsole aus."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        
        if print_console:
            print(log_msg)
            
        for line in str(log_msg).split('\n'):
            clean_msg = line.rstrip()
            if clean_msg:
                self.logs.append(clean_msg)


    def run(self):
        """Startet die Pipeline für alle definierten Modelle."""
        self.log(f"Starte Verarbeitung für {len(self.model_names)} Modelle...")
        for stem in self.model_names:
            self.logs = [] # Reset logs for each model
            self.process_model(stem)


    def process_model(self, stem):
        """
        Führt alle Arbeitsschritte für ein einzelnes IFC-Modell durch:
        - Versuch #2 GNN-Machbarkeitsstudie: Kontaktgraph generieren (A.1-A.4)
        - Versuch #1 RBC-Benchmark: Joints identifizieren und bewerten (B.1-B.3)
        """
        self.log(f"\n{'='*50}\nVerarbeite Modell: {stem}\n{'='*50}")

        # =====================================================
        # VERSUCH #2 GNN-MACHBARKEITSSTUDIE: KONTAKTGRAPH
        # =====================================================
        # --- A.1 Dateneingang: IFC-Extraktion & Labels ---
        nodes, stats_ifc = self.extract_ifc_data(stem)
        if not nodes:
            self.log(f"Abbruch: Keine auswertbaren Knoten für {stem} gefunden.")
            return

        gt_pairs = self.load_ground_truth(stem)
        rbc_dict = self.load_rbc_labels(stem)

        # --- A.2 Knoten & Knotenmerkmale (homogener / heterogener Graph) ---
        node_features_hom, node_features_het = self.compute_node_features(nodes)

        # --- A.3 Kanten & Kantenmerkmale ---
        edges, stats_edges = self.calculate_edges(nodes, gt_pairs, rbc_dict)
        edges = self.compute_edge_features(nodes, edges)

        # --- A.4 Visualisierung ---
        # Der JSON-Export erfolgt am Ende von process_model, damit die
        # Joint-Ergebnisse (Versuch #1) in derselben Datei enthalten sind.
        self.plot_graph(stem, nodes, edges, mode="2D")
        self.plot_graph(stem, nodes, edges, mode="3D")

        # =====================================================
        # VERSUCH #1 RBC-BENCHMARK: JOINTS (HYPERKANTEN)
        # =====================================================
        # --- B.1 Soll-Joints: Rekonstruktion aus der Ground Truth (IFC) ---
        soll_joints = self.reconstruct_gt_joints(stem, nodes, gt_pairs)
        soll_joints = self.classify_gt_joints(soll_joints)

        # --- B.2 Ist-Joints: Extraktion aus dem Analysemodell ---
        ist_joints = self.extract_analysis_joints(stem)

        # --- B.3 Validierung: Matching, Fehlercodes & Statistik ---
        matching = self.match_joints(soll_joints, ist_joints)
        error_codes = self.classify_joint_errors(matching) # Fehlertaxonomie
        joint_stats = self.evaluate_joint_statistics(matching, error_codes)

        # --- Export: Graph-Datei inkl. aller Joint-Ergebnisse (Soll, Ist, Validierung) ---
        # Hinweis: Erfolgt am Ende, damit die in B.1-B.3 berechneten Daten enthalten sind.
        self.export_graph(stem, nodes, edges, stats_ifc, stats_edges,
                          soll_joints=soll_joints, ist_joints=ist_joints,
                          matching=matching, error_codes=error_codes,
                          joint_stats=joint_stats)

        # --- Log speichern ---
        log_path = os.path.join(self.dir_out, f"{stem} Protokoll.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.logs))


    # =========================================================
    # VERSUCH #2 GNN-MACHBARKEITSSTUDIE:
    # GENERIERUNG DES KONTAKTGRAPHEN
    # =========================================================
    # Kernaufgabe: Erstellung der Knoten und Kanten des Kontaktgraphen.
    # Ein Teil der Daten wird eingelesen (IFC, GT-/RBC-Labels), weitere
    # Informationen werden eigens hier im Skript berechnet
    # (Knotenmerkmale in A.2, Kantenmerkmale in A.3).
    # ---------------------------------------------------------
    # A.1 DATENEINGANG: IFC-EXTRAKTION & MATHEMATIK
    # ---------------------------------------------------------
    def extract_ifc_data(self, stem):
        """
        Extrahiert Knoten aus dem IFC-Modell inkl. Geometrie.
        
        Mathematischer Hintergrund:
        Oberfläche (Surface Area): Summe der Flächen aller Dreiecke.
        $ \text{Fläche} = \frac{1}{2} |\vec{AB} \times \vec{AC}| $
        
        Volumen (Mesh Volume): Divergenzsatz (Satz von Gauß).
        $ V = \sum \left( \frac{1}{6} \vec{A} \cdot (\vec{B} \times \vec{C}) \right) $
        """
        ifc_path = os.path.join(self.dir_db, f"{stem}.ifc")
        if not os.path.exists(ifc_path):
            self.log(f"FEHLER: {ifc_path} nicht gefunden.")
            return [], {}
        
        try:
            self.log(f"Lade IFC-Modell: {ifc_path}")
            model = ifcopenshell.open(str(ifc_path))
            settings = ifcopenshell.geom.settings()
        except Exception as e:
            self.log(f"Kritischer Fehler beim Öffnen der IFC: {e}")
            return [], {}

        products = model.by_type("IfcElement")
        nodes = []
        
        count_all = len(products)
        count_lb = 0
        count_geom = 0
        entity_counts = {}
        
        self.log(f"Analysiere {count_all} IFC-Elemente...")

        for i, prod in enumerate(products):
            if not self.is_load_bearing(prod):
                continue
            count_lb += 1
                
            geom_data = self.get_geometry_data(prod, settings)
            if not geom_data:
                continue
            count_geom += 1

            entity = prod.is_a()
            target_entity = entity if entity in ALLOWED_ENTITIES else "IfcBuildingElementProxy"
            entity_counts[target_entity] = entity_counts.get(target_entity, 0) + 1

            nodes.append({
                "node_id": len(nodes),
                "guid": str(prod.GlobalId).strip(),
                "name": prod.Name or "Unnamed",
                "entity_processed": target_entity,
                "x": geom_data["centroid_x"],
                "y": geom_data["centroid_y"],
                "z": geom_data["centroid_z"],
                "origin_x": geom_data["origin_x"],
                "origin_y": geom_data["origin_y"],
                "origin_z": geom_data["origin_z"],
                "dimensions": geom_data["dimensions"],
                "geom_raw": geom_data["geom_raw"], # Wird vor dem Export entfernt
                "features": {
                    "entity_one_hot_idx": ALLOWED_ENTITIES.index(target_entity),
                    "norm_dimensions": geom_data["norm_dimensions"],
                    "area": geom_data["area"],
                    "volume": geom_data["volume"],
                    "ratio_av": geom_data["ratio_av"],
                }
            })
            
            if i > 0 and i % 100 == 0:
                self.log(f"DEBUG: {i}/{count_all} Elemente geprüft...")
            
        stats = {
            "count_all_ifc_products": count_all,
            "count_load_bearing": count_lb,
            "count_with_geometry": count_geom,
            "entity_distribution": entity_counts
        }
        
        self.log(f"Extraktion abgeschlossen. {count_geom} gültige Knoten erstellt.")
        return nodes, stats


    def is_load_bearing(self, product):
        try:
            for _, props in element.get_psets(product).items():
                if "LoadBearing" in props:
                    return bool(props["LoadBearing"])
        except: pass
        return False


    def get_geometry_data(self, product, settings):
        """Generiert trianguliertes Mesh und AABB für ein Bauteil."""
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            if not shape or not getattr(shape, 'geometry', None):
                return None
            
            # OpenCascade Geometry
            verts = shape.geometry.verts #type: ignore
            faces = shape.geometry.faces #type: ignore
            
            matrix_1d = shape.transformation.matrix #type: ignore
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
            pos_x, pos_y, pos_z = matrix_1d[12], matrix_1d[13], matrix_1d[14]
            origin = (round(pos_x, 3), round(pos_y, 3), round(pos_z, 3))
            
            x_coords = [verts[i] for i in range(0, len(verts), 3)]
            y_coords = [verts[i+1] for i in range(0, len(verts), 3)]
            z_coords = [verts[i+2] for i in range(0, len(verts), 3)]
            num_verts = len(x_coords)
            
            # Centroid (Lokal zu global)
            c_x = (sum(x_coords) / num_verts) + pos_x
            c_y = (sum(y_coords) / num_verts) + pos_y
            c_z = (sum(z_coords) / num_verts) + pos_z
            
            l_abs = max(x_coords) - min(x_coords)
            b_abs = max(y_coords) - min(y_coords)
            h_abs = max(z_coords) - min(z_coords)
            dimensions = (round(l_abs, 3), round(b_abs, 3), round(h_abs, 3))
            
            max_dim = max(l_abs, b_abs, h_abs, 1e-6)
            norm_dimensions = (round(l_abs/max_dim, 3), round(b_abs/max_dim, 3), round(h_abs/max_dim, 3))

            # Globale Vertices für KDTree
            local_verts = np.array(verts).reshape(-1, 3)
            matrix_4x4 = np.array(matrix_1d).reshape(4, 4)
            homogeneous_verts = np.hstack((local_verts, np.ones((local_verts.shape[0], 1))))
            global_verts = homogeneous_verts.dot(matrix_4x4)[:, :3]
            
            aabb_min, aabb_max = np.min(global_verts, axis=0), np.max(global_verts, axis=0)

            # Eindeutige Kanten extrahieren 
            faces_array = np.array(faces).reshape(-1, 3)
            triangles = global_verts[faces_array]

            edges_set = set()
            for f in faces_array:
                # Eindeutige Kanten des Meshes extrahieren (sortiert, um Duplikate zu vermeiden)
                edges_set.add(tuple(sorted((f[0], f[1]))))
                edges_set.add(tuple(sorted((f[1], f[2]))))
                edges_set.add(tuple(sorted((f[2], f[0]))))

            # Geometrie abtasten                
            sampled_points = []
            resolution = 0.025 # Alle 2,5 cm einen Test-Punkt erzeugen
            for e in edges_set:
                p1 = global_verts[e[0]]
                p2 = global_verts[e[1]]
                length = np.linalg.norm(p2 - p1)

                if length > resolution:
                    num_steps = int(np.ceil(length / resolution))
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

            # Fläche / Volumen berechnen
            area, volume = 0.0, 0.0
            for f in faces_array:
                A, B, C = global_verts[f[0]], global_verts[f[1]], global_verts[f[2]]
                area += 0.5 * np.linalg.norm(np.cross(B - A, C - A))
                volume += np.dot(A, np.cross(B, C)) / 6.0
                
            volume = abs(volume)
            ratio_av = round(area / volume, 4) if volume > 0 else 0

            return {
                "origin_x": origin[0], "origin_y": origin[1], "origin_z": origin[2],
                "centroid_x": round(c_x, 3), "centroid_y": round(c_y, 3), "centroid_z": round(c_z, 3),
                "dimensions": dimensions,
                "norm_dimensions": norm_dimensions,
                "area": round(area, 4),
                "volume": round(volume, 4),
                "ratio_av": ratio_av,
                "geom_raw": {
                    "vertices": global_verts, # Originale Eckpunkte
                    "triangles": triangles,
                    "test_points": test_points, # Verdichtete abgetastete Punkte
                    "aabb": (aabb_min, aabb_max)
                }
            }
        except Exception as e:
            return None


    # ---------------------------------------------------------
    # A.1 DATENEINGANG: LABELING LESE-FUNKTIONEN
    # ---------------------------------------------------------
    def load_ground_truth(self, stem):
        """Lädt GT-Labels. Rückgabe ist ein Set aus sortierten Tupeln der GUIDs."""
        path = os.path.join(self.dir_gt, f"{stem} GT_Labels.csv")
        gt_pairs = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8-sig') as f:
                # Fallback falls Komma statt Semikolon genutzt wurde
                reader = csv.DictReader(f, delimiter=';')
                if not reader.fieldnames or 'GUID_A' not in reader.fieldnames:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    g_a, g_b = row.get("GUID_A", "").strip(), row.get("GUID_B", "").strip()
                    status = row.get("Status", "").strip()
                    
                    # Nur Verbindungen mit dem Status "o" gelten als echte Referenzwahrheit
                    if status == "o" and g_a and g_b and g_a != "---" and g_b != "---":
                        gt_pairs.add(tuple(sorted([g_a, g_b])))
                        
            self.log(f"GT-Labels geladen: {len(gt_pairs)} echte Referenz-Paare (gefiltert auf Status 'o')")
        else:
            self.log(f"WARNUNG: Keine GT-Labels unter {path} gefunden.")
        return gt_pairs


    def load_rbc_labels(self, stem):
        """Lädt regelbasierte Labels. Rückgabe: Dict {pair_tuple: Status}"""
        path = os.path.join(self.dir_rbc, f"{stem} RFEM_Labels.csv")
        rbc_dict = {}
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8-sig') as f:
                # Fallback falls Komma statt Semikolon genutzt wurde
                reader = csv.DictReader(f, delimiter=';')
                if not reader.fieldnames or 'IFC_GUID_A' not in reader.fieldnames:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=',')
                for row in reader:
                    g_a = row.get("IFC_GUID_A", "").strip()
                    g_b = row.get("IFC_GUID_B", "").strip()
                    status = row.get("Label", "TN").strip()
                    if g_a and g_b:
                        rbc_dict[tuple(sorted([g_a, g_b]))] = status
            self.log(f"RBC-Labels geladen: {len(rbc_dict)} Einträge -- \tAchtung: Fehlende Kanten werden auf GT aufgestockt. Zahl ist somit entsprechend der 'FP's größer.")
        else:
            self.log(f"WARNUNG: Keine RBC-Labels unter {path} gefunden.")
        return rbc_dict


    # ---------------------------------------------------------
    # A.2 KNOTEN & KNOTENMERKMALE
    # ---------------------------------------------------------
    def compute_node_features(self, nodes):
        """
        Bündelt die Berechnung der Knotenmerkmale des Kontaktgraphen.
        Es werden zwei Merkmalsstrukturen parallel erzeugt:
        1. Homogener Graph: einheitliche Merkmalsstruktur für alle Bauteile.
        2. Heterogener Graph: getrennte Merkmalsstrukturen für 1D- und 2D-Bauteile.
        """
        features_hom = self.compute_node_features_homogeneous(nodes)
        features_het = self.compute_node_features_heterogeneous(nodes)
        return features_hom, features_het


    def compute_node_features_homogeneous(self, nodes):
        """
        PLATZHALTER: Einheitliche Knotenmerkmale für den homogenen Graphen.
        Geplante Inhalte:
        - Orientierung, Ort, Ebene
        - Geometrische Merkmale: Primär- und Sekundärmaß, Volumen, Masse, Schlankheit
        - Kovarianzmatrix
        - Lage im Gebäude
        """
        self.log("PLATZHALTER: Homogene Knotenmerkmale noch nicht implementiert.")
        features = {}
        # TODO: Orientierung, Ort, Ebene berechnen
        # TODO: Primär-/Sekundärmaß, Volumen, Masse, Schlankheit berechnen
        # TODO: Kovarianzmatrix berechnen
        # TODO: Lage im Gebäude bestimmen
        return features


    def compute_node_features_heterogeneous(self, nodes):
        """
        PLATZHALTER: Knotenmerkmale für den heterogenen Graphen mit
        Unterscheidung nach Bauteil-Dimensionalität:
        - 1D-Bauteile (z. B. Stützen, Balken)
        - 2D-Bauteile (z. B. Wände, Decken)
        Merkmalsinhalte analog zum homogenen Graphen.
        """
        self.log("PLATZHALTER: Heterogene Knotenmerkmale (1D/2D) noch nicht implementiert.")
        features = {"1D": {}, "2D": {}}
        # TODO: Aufteilung der Knoten in 1D- und 2D-Bauteile
        # TODO: Dimensionsspezifische Merkmale je Knotentyp berechnen
        return features


    # ---------------------------------------------------------
    # A.3 KANTEN & KANTENMERKMALE
    # ---------------------------------------------------------
    def dist_aabb(self, aabb1, aabb2):
        """Berechnet den euklidischen Abstand zweier AABBs."""
        min1, max1 = aabb1
        min2, max2 = aabb2
        dx = max(0, max(min1[0] - max2[0], min2[0] - max1[0]))
        dy = max(0, max(min1[1] - max2[1], min2[1] - max1[1]))
        dz = max(0, max(min1[2] - max2[2], min2[2] - max1[2]))
        return math.sqrt(dx**2 + dy**2 + dz**2)


    def calculate_edges(self, nodes, gt_pairs, rbc_dict):
        """
        Bildet die Kanten basierend auf Distanzen und ergänzt die GT/RBC Labels.
        
        Gauß-Kernel Transformation (RBF):
        $ \text{proximity} = \exp(-\gamma \cdot d^2) $
        """
        self.log("Starte Distanzmessungen und Kantenbildung...")
        edges = []
        
        # KDTree mit verdichteten 'test_points'
        trees = {n["guid"]: cKDTree(n["geom_raw"]["test_points"]) for n in nodes}
        stats_rbc = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        total_edges = 0
        gt_edges_count = 0

        # self.log(f"DEBUG: GT-Pairs: {gt_pairs}")

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n_a, n_b = nodes[i], nodes[j]
                
                # 1. Schneller AABB Filter
                d_aabb = self.dist_aabb(n_a["geom_raw"]["aabb"], n_b["geom_raw"]["aabb"])
                
                if d_aabb > MAX_DIST:
                    continue
                
                # 2. Exakte Messung via KDTree
                dist_a_to_b, _ = trees[n_a["guid"]].query(n_b["geom_raw"]["test_points"], k=1, workers=-1)
                final_distance = np.min(dist_a_to_b) # in [m]
                
                if final_distance > MAX_DIST:
                    continue
                
                # Wandle die Distanz von [m] in [mm] um
                final_dist_mm = final_distance * 1000.0
                    
                # 3. Kante erstellen
                proximity = math.exp(-RBF_GAMMA * (final_dist_mm ** 2))
                
                pair = tuple(sorted([str(n_a["guid"]).strip(), str(n_b["guid"]).strip()]))
                is_gt = pair in gt_pairs
                # self.log(f"DEBUG: Pair gebaut: {pair} | is_GT: {is_gt}")
                
                rbc_status = rbc_dict.get(pair)
                if not rbc_status:
                    # Logische Ableitung, falls Paar nicht in RFEM auftaucht
                    rbc_status = "FN" if is_gt else "TN"
                
                stats_rbc[rbc_status] = stats_rbc.get(rbc_status, 0) + 1
                total_edges += 1
                if is_gt: gt_edges_count += 1

                edges.append({
                    "source": n_a["node_id"],
                    "target": n_b["node_id"],
                    "features": {
                        "distance": round(final_dist_mm, 4),
                        "proximity_activation": round(proximity, 6),
                        "load_transfer": is_gt,
                        "rbc_converted": rbc_status in ["TP", "FP"],
                        "rbc_status": rbc_status
                    },
                    "meta": {
                        "guid_a": n_a["guid"],
                        "guid_b": n_b["guid"],
                        "entity_a": n_a["entity_processed"],
                        "entity_b": n_b["entity_processed"],
                        "name_a": n_a["name"],
                        "name_b": n_b["name"],
                    }
                })
        
        # Performance Metriken berechnen
        tp, tn = stats_rbc.get("TP", 0), stats_rbc.get("TN", 0)
        fp, fn = stats_rbc.get("FP", 0), stats_rbc.get("FN", 0)
        
        accuracy = (tp + tn) / total_edges if total_edges > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        stats_edges = {
            "total_edges": total_edges,
            "gt_edges": gt_edges_count,
            "gt_ratio_percentage": round((gt_edges_count / total_edges * 100), 2) if total_edges > 0 else 0.0,
            "rbc_metrics": {
                "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1_Score": round(f1_score, 4)
            }
        }
        
        self.log(f"DEBUG: Kanten berechnet. Total: {total_edges} | GT: {gt_edges_count} | RBC-TP: {tp}")
        return edges, stats_edges


    def compute_edge_features(self, nodes, edges):
        """
        Bündelt die Berechnung der erweiterten Kantenmerkmale des Kontaktgraphen.
        Grundlage: Projektion der Flächen der volumetrischen Elemente
        (Grund-, Mantel- und Schrägflächen von Stützen, Balken, Wänden, Decken).
        """
        edges = self.compute_face_projection_features(nodes, edges)
        edges = self.compute_occlusion_features(nodes, edges)
        return edges


    def compute_face_projection_features(self, nodes, edges):
        """
        PLATZHALTER: Merkmale aus der Flächenprojektion:
        - Schnitt-Indikator
        - Relativer Durchdringungsgrad
        - Kontaktflächenanteil
        - Projektionsabstand
        - Aktive Flächenklasse
        """
        self.log("PLATZHALTER: Kantenmerkmale (Flächenprojektion) noch nicht implementiert.")
        # TODO: Grund-, Mantel- und Schrägflächen projizieren und Merkmale ableiten
        return edges


    def compute_occlusion_features(self, nodes, edges):
        """
        PLATZHALTER: Verdeckungsmerkmale zur Erkennung, ob Bauteile direkt
        verbunden sind oder indirekt über Dritte:
        - Relative Lage und Schwerkraftbezug
        - Winkelbeziehungen
        - Stoßtyp-Klassifikation
        - Regelkonformitäts-Scores (weiche Expertenregeln)
        """
        self.log("PLATZHALTER: Verdeckungsmerkmale noch nicht implementiert.")
        # TODO: Verdeckungsanalyse und Merkmalsableitung
        return edges


    # ---------------------------------------------------------
    # A.4 GRAPH-EXPORT & VISUALISIERUNG
    # ---------------------------------------------------------
    def export_graph(self, stem, nodes, edges, stats_ifc, stats_edges,
                     soll_joints=None, ist_joints=None, matching=None,
                     error_codes=None, joint_stats=None):
        """
        Exportiert den Kontaktgraphen als JSON-Datei.
        Die Datei enthält neben Knoten und Kanten auch eine eigene Sektion
        'joints' mit den Ergebnissen aus Versuch #1 (Soll-/Ist-Joints und Validierung).

        Args:
            stem (str): Modellname ohne Dateiendung.
            nodes (list): Knotenliste des Kontaktgraphen.
            edges (list): Kantenliste des Kontaktgraphen.
            stats_ifc (dict): Statistik der IFC-Extraktion.
            stats_edges (dict): Statistik der Kantenbildung und RBC-Metriken.
            soll_joints (list, optional): Rekonstruierte Soll-Joints (B.1).
            ist_joints (list, optional): Extrahierte Ist-Joints (B.2).
            matching (dict, optional): Zuordnungstabelle Soll <-> Ist (B.3).
            error_codes (dict, optional): Fehlercodes E0-E7 je Joint (B.3).
            joint_stats (dict, optional): Statistische Auswertung (B.3).

        Returns:
            None
        """
        # Bereinigung der unhandlichen Geometriedaten vor dem Export
        clean_nodes = []
        for n in nodes:
            n_copy = n.copy()
            n_copy.pop("geom_raw", None) 
            clean_nodes.append(n_copy)

        output_data = {
            "graph_info": {
                "model_stem": stem,
                "preprocessing_config": {
                    "rbf_gamma": RBF_GAMMA,
                    "max_dist": MAX_DIST,
                    "joint_epsilon": JOINT_EPSILON,
                    "joint_tau": JOINT_TAU,
                    "joint_delta_scale": JOINT_DELTA_SCALE
                },
                "stats_IFC_extraction": stats_ifc,
                "stats_Edges_and_RBC": stats_edges
            },
            "nodes": clean_nodes,
            "edges": edges,
            "joints": {
                "config": {
                    "epsilon": JOINT_EPSILON,
                    "tau": JOINT_TAU,
                    "delta_scale": JOINT_DELTA_SCALE
                },
                "soll_joints": soll_joints if soll_joints is not None else [],
                "ist_joints": ist_joints if ist_joints is not None else [],
                "validation": {
                    "matching": matching if matching is not None else {},
                    "error_codes": error_codes if error_codes is not None else {},
                    "statistics": joint_stats if joint_stats is not None else {}
                }
            }
        }
        
        path = os.path.join(self.dir_out, f"{stem} Graph.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        self.log(f"Graph JSON gespeichert unter: {os.path.basename(path)}")


    def plot_graph(self, stem, nodes, edges, mode="2D"):
        """Erstellt Plots der Bauwerksgraphen als 2D- oder 3D-Projektion"""
        fig = plt.figure(figsize=(12, 10))
        
        if mode == "3D":
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')
            pos = {n["node_id"]: (n["x"], n["y"], n["z"]) for n in nodes}
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            G = nx.Graph()
            for n in nodes: G.add_node(n["node_id"])
            for e in edges: G.add_edge(e["source"], e["target"])
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            
        sorted_edges = sorted(edges, key=lambda e: e["features"]["load_transfer"])
        
        for edge in sorted_edges:
            u, v = edge["source"], edge["target"]
            is_gt = edge["features"]["load_transfer"]
            c, a, lw, ls = ("darkblue", 0.9, 0.8, "-") if is_gt else ("gray", 0.4, 0.5, "--")
            
            if mode == "3D":
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 
                        color=c, alpha=a, linewidth=lw, linestyle=ls, zorder=1)
            else:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                        color=c, alpha=a, linewidth=lw, linestyle=ls, zorder=1)
                
        for ent in ALLOWED_ENTITIES:
            ent_nodes = [n for n in nodes if n.get("entity_processed") == ent]
            if not ent_nodes: continue
            
            node_ids = [n["node_id"] for n in ent_nodes]
            x = [pos[n_id][0] for n_id in node_ids]
            y = [pos[n_id][1] for n_id in node_ids]
            color, marker = VIS_MAPPING.get(ent, ("black", "o"))
            
            if mode == "3D":
                z = [pos[n_id][2] for n_id in node_ids]
                ax.scatter(x, y, z, c=color, marker=marker, s=150, label=ent, edgecolors='white', linewidths=0.5, zorder=5) #type: ignore
            else:
                ax.scatter(x, y, c=color, marker=marker, s=150, label=ent, edgecolors='white', linewidths=0.5, zorder=5)
                
        ax.set_title(f"Graph ({mode}): {stem}")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        
        if mode == "3D":
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]') #type: ignore
        else:
            ax.set_axis_off()
            
        fig.subplots_adjust(right=0.85)
        
        filename_png = os.path.join(self.dir_out, f"{stem} {mode}.png")
        plt.savefig(filename_png, dpi=300)
        plt.close(fig)
        self.log(f"Plot erstellt: {os.path.basename(filename_png)}")


    # =========================================================
    # VERSUCH #1 RBC-BENCHMARK: JOINTS
    # =========================================================
    # Joints sind Hyperkanten. Sie gehören nicht direkt zum
    # Kontaktgraphen, beziehen sich aber auf dessen Bestandteile.
    # ---------------------------------------------------------
    # B.1 SOLL-JOINTS: REKONSTRUKTION AUS DER GROUND TRUTH (IFC-BASIERT)
    # ---------------------------------------------------------
    # Methodenreihenfolge: interne Hilfsfunktionen (_) alphabetisch,
    # danach Orchestrierung, danach weitere Unterfunktionen.
    # ---------------------------------------------------------
    def _attribute_joints(self, components, contact_regions, gt_pairs):
        """
        Schritt 3 der Joint-Rekonstruktion: Attribuiert die geclusterten Joints.
        Berechnet Position, Geometrietyp, Bauteilmenge und den induzierten GT-Teilgraph.

        Args:
            components (list[set]): Zusammenhangskomponenten (Indexmengen) aus dem Koinzidenz-Graph.
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Attribuierte Soll-Joints mit Schlüsseln:
                'position', 'geometry_type', 'member_guids', 'gt_subgraph_edges'.
        """
        joints = []
        for comp in components:
            # Sammle alle Kontaktregionen und Bauteile dieser Komponente
            comp_regions = [contact_regions[i] for i in comp]
            member_guids = set()
            for r in comp_regions:
                member_guids.add(r["guid_a"])
                member_guids.add(r["guid_b"])

            # Position: Schwerpunkt über alle Regions-Schwerpunkte
            positions = np.array([r["centroid"] for r in comp_regions])
            position = np.mean(positions, axis=0)

            # Geometrietyp: PCA über die Vereinigung aller Regionspunkte
            all_points = np.vstack([r["points"] for r in comp_regions])
            eigenvalues, _, L = self._pca_contact_region(all_points) # L = Ausdehnungen der Kontaktregion
            geometry_type = self._classify_dimensionality(L)

            # Induzierter GT-Teilgraph: Welche Paare innerhalb des Joints tragen Last ab?
            member_list = sorted(list(member_guids))
            gt_subgraph_edges = []
            for i in range(len(member_list)):
                for j in range(i + 1, len(member_list)):
                    pair = (member_list[i], member_list[j])
                    if pair in gt_pairs:
                        gt_subgraph_edges.append(pair)

            joints.append({
                "position": [round(float(p), 4) for p in position],
                "geometry_type": geometry_type,
                "member_guids": member_list,
                "gt_subgraph_edges": gt_subgraph_edges,
                "eigenvalues": [round(float(l), 6) for l in eigenvalues],
                "lengths": [round(float(l), 4) for l in L]
            })
        return joints


    def _classify_dimensionality(self, L):
        """
        Klassifiziert die Dimensionalität einer Kontaktregion anhand der Ausdehnungen.
        Deckt die Überschneidung von volumetrischen und dimensionsreduzierten Elementen ab.
        Entscheidungsregel (Versuch #1 Teil A):
            L2/L1 < tau  und L3/L1 < tau  -> linienhaft (2 Ausdehnungen sind deutlich kleiner als Primärachse)
            L3/L1 < tau  und L2/L1 >= tau -> flächig (1 Ausdehnung ist deutlich kleiner als Primärachse)
            sonst                          -> punktförmig

        Args:
            L (np.ndarray): Ausdehnungen [L1, L2, L3] entlang der Hauptachsen (L1 >= L2 >= L3).

        Returns:
            str: 'linienhaft', 'flaechig' oder 'punktuell'.
        """
        if L[0] < 1e-9: # z.B. Idealisierte 1D-Bauteile
            return "punktuell"
        r21 = L[1] / L[0]
        r31 = L[2] / L[0]
        if r21 < JOINT_TAU and r31 < JOINT_TAU: # z.B. Wand-Wand, Wand-Decke, Decke-Decke
            return "linienhaft"
        if r31 < JOINT_TAU and r21 >= JOINT_TAU: # z.B. Balken-Balken, Stütze-Decke
            return "flaechig"
        return "punktuell" # Evtl. zu unsauber? Dann bräuchte es "sonstige", falls das relevant wird


    def _cluster_contact_regions(self, contact_regions, nodes_by_guid):
        """
        Schritt 2 der Joint-Rekonstruktion: Clustert Kontaktregionen zu Joints.
        Zwei Regionen R_ab und R_cd gehören zum selben Joint, wenn sie räumlich
        koinzident sind: Hausdorff-Distanz d_H(R_ab, R_cd) <= delta.
        Die Toleranz delta skaliert mit der größten Bauteildicke max(t_a, t_b).
        Die transitive Hülle wird über Connected Components gebildet.

        Args:
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1.
            nodes_by_guid (dict): Mapping GUID -> Knoten-Dictionary.

        Returns:
            list[set]: Zusammenhangskomponenten; jede Menge enthält Indizes
                       auf contact_regions, die zum selben Joint gehören.
        """
        n = len(contact_regions)
        if n == 0:
            return []

        # Koinzidenz-Graph aufbauen: Knoten = Regionen, Kante = Koinzidenz
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                r1 = contact_regions[i]
                r2 = contact_regions[j]
                p1 = r1["points"]
                p2 = r2["points"]

                # Hausdorff-Distanz zwischen den Punktmengen (symmetrisch)
                d_h = self._hausdorff_distance(p1, p2)

                # Joint-Toleranz delta: skaliert mit größter Bauteildicke beider Regionen
                t1 = self._get_max_thickness(r1, nodes_by_guid)
                t2 = self._get_max_thickness(r2, nodes_by_guid)
                delta = JOINT_DELTA_SCALE * max(t1, t2)

                if d_h <= delta:
                    G.add_edge(i, j)

        # Transitive Hülle: Zusammenhangskomponenten (abgeschlossene Inseln)
        # bilden die Joint-Cluster, bei denen man innerhalb eines Clusters 
        # von jedem Punkt aus jeden anderen darin erreichen kann.
        return list(nx.connected_components(G))


    def _compute_contact_regions(self, nodes, gt_pairs):
        """
        Schritt 1 der Joint-Rekonstruktion: Bestimmt die Kontaktregion R_ab
        für jedes Ground-Truth-Paar (a, b).
        Definition (Versuch #1 Teil A):
            R_(a,b) = { x in R^3 | d(x, dS_a) <= eps  und  d(x, dS_b) <= eps }
        mit S_a, S_b = Volumenkörper, eps = Kontakttoleranz (JOINT_EPSILON).
        Praktisch: Schnittmenge der abgetasteten Randpunktwolken beider Bauteile
        (Auflösung 2,5 cm) über cKDTree.

        Args:
            nodes (list[dict]): Knotenliste mit 'geom_raw' und 'guid'.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Kontaktregionen mit Schlüsseln:
                'guid_a', 'guid_b', 'points', 'centroid',
                'eigenvalues', 'lengths', 'contact_type'.
        """
        self.log(f"B.1 Schritt 1: Bestimme Kontaktregionen für {len(gt_pairs)} GT-Paare (eps = {JOINT_EPSILON*1000:.1f} mm)...")
        nodes_by_guid = {n["guid"]: n for n in nodes}
        contact_regions = []

        for guid_a, guid_b in gt_pairs:
            node_a = nodes_by_guid.get(guid_a)
            node_b = nodes_by_guid.get(guid_b)
            if not node_a or not node_b:
                continue

            pts_a = node_a["geom_raw"]["test_points"]
            pts_b = node_b["geom_raw"]["test_points"]

            # Abstand aller Punkte von a zu b und umgekehrt
            dist_a_to_b, _ = cKDTree(pts_b).query(pts_a, k=1, workers=-1)
            dist_b_to_a, _ = cKDTree(pts_a).query(pts_b, k=1, workers=-1)

            # Schnittmenge: Punkte, die beide Ränder innerhalb eps berühren
            mask_a = dist_a_to_b <= JOINT_EPSILON
            mask_b = dist_b_to_a <= JOINT_EPSILON
            contact_points = np.vstack([pts_a[mask_a], pts_b[mask_b]])

            if len(contact_points) == 0:
                continue

            # Charakterisierung der Kontaktregion
            centroid = np.mean(contact_points, axis=0)
            eigenvalues, _, L = self._pca_contact_region(contact_points)
            contact_type = self._classify_dimensionality(L)

            contact_regions.append({
                "guid_a": guid_a,
                "guid_b": guid_b,
                "points": contact_points,
                "centroid": centroid,
                "eigenvalues": eigenvalues,
                "lengths": L,
                "contact_type": contact_type
            })

        self.log(f"B.1 Schritt 1: {len(contact_regions)} Kontaktregionen gefunden.")
        return contact_regions


    def _get_max_thickness(self, region, nodes_by_guid):
        """
        Ermittelt die maximale Bauteildicke der beiden an einer Kontaktregion
        beteiligten Bauteile. Als Näherung dient das kleinste der drei
        Primärmaße (AABB-Dimensionen) jedes Bauteils.

        Args:
            region (dict): Kontaktregion mit 'guid_a' und 'guid_b'.
            nodes_by_guid (dict): Mapping GUID -> Knoten-Dictionary.

        Returns:
            float: Maximale Bauteildicke in [m].
        """
        t_max = 0.0
        for guid in (region["guid_a"], region["guid_b"]):
            node = nodes_by_guid.get(guid)
            if node and "dimensions" in node:
                t = min(node["dimensions"])
                t_max = max(t_max, t)
        return t_max


    def _hausdorff_distance(self, pts_a, pts_b):
        """
        Berechnet die symmetrische Hausdorff-Distanz zwischen zwei Punktmengen.
            d_H(A, B) = max( sup_{a in A} inf_{b in B} ||a-b||,
                             sup_{b in B} inf_{a in A} ||a-b|| )

        Args:
            pts_a (np.ndarray): Punktmenge A, Shape (n, 3).
            pts_b (np.ndarray): Punktmenge B, Shape (m, 3).

        Returns:
            float: Hausdorff-Distanz in [m].
        """
        if len(pts_a) == 0 or len(pts_b) == 0:
            return float('inf')
        d_a_to_b = np.min(cKDTree(pts_b).query(pts_a, k=1, workers=-1)[0])
        d_b_to_a = np.min(cKDTree(pts_a).query(pts_b, k=1, workers=-1)[0])
        return max(d_a_to_b, d_b_to_a)


    def _pca_contact_region(self, points):
        """
        Führt eine Hauptkomponentenanalyse (PCA) über eine Punktwolke durch.
        Liefert die Eigenwerte lambda_1 >= lambda_2 >= lambda_3, die zugehörigen
        Eigenvektoren (Hauptachsen) und die Ausdehnungen L_i entlang dieser Achsen.

        Args:
            points (np.ndarray): Punktwolke, Shape (n, 3).

        Returns:
            tuple: (eigenvalues, eigenvectors, lengths)
                eigenvalues (np.ndarray): Eigenwerte der Kovarianzmatrix (absteigend).
                eigenvectors (np.ndarray): Hauptachsen als Spalten.
                lengths (np.ndarray): Ausdehnungen [L1, L2, L3] in [m].
        """
        if len(points) < 2:
            return np.zeros(3), np.eye(3), np.zeros(3)

        cov = np.cov(points.T) # cov = Kovarianzmatrix (Verteilung der Punkte um Massezentrum)
        #Eigenvektor: 3D-Richtung der Hauptachsen.    Eigenwert: Skalare Länge der Ausdehnung in diese Richtungen
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Absteigend sortieren
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Ausdehnungen: Projektion auf Hauptachsen
        proj = points @ eigenvectors # Punkte werden so gedreht, dass sie orthogonal zu Hauptachsen liegen
        lengths = np.max(proj, axis=0) - np.min(proj, axis=0) # Bounding-Box-Dimensionen des Kontaktvolumens

        return eigenvalues, eigenvectors, lengths


    def reconstruct_gt_joints(self, stem, nodes, gt_pairs):
        """
        Orchestriert die Rekonstruktion der Soll-Joints aus der Ground Truth.
        Ablauf (geometrisch gestützt, siehe Anhang Teil A):
            Schritt 1: Kontaktregion pro GT-Paar bestimmen
            Schritt 2: Kontaktregionen zu Joints clustern
            Schritt 3: Joints attribuieren

        Args:
            stem (str): Modellname ohne Dateiendung.
            nodes (list[dict]): Knotenliste mit Geometriedaten.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare.

        Returns:
            list[dict]: Attribuierte Soll-Joints.
        """
        self.log("B.1: Starte Soll-Joint-Rekonstruktion (IFC-basiert)...")
        nodes_by_guid = {n["guid"]: n for n in nodes}

        # Schritt 1: Kontaktregionen bestimmen
        contact_regions = self._compute_contact_regions(nodes, gt_pairs)

        # Schritt 2: Kontaktregionen zu Joints clustern
        components = self._cluster_contact_regions(contact_regions, nodes_by_guid)

        # Schritt 3: Joints attribuieren
        joints = self._attribute_joints(components, contact_regions, gt_pairs)

        self.log(f"B.1: {len(joints)} Soll-Joints rekonstruiert.")
        return joints


    def classify_gt_joints(self, joints):
        """
        PLATZHALTER: Klassifikation der Joints über die drei Achsen der
        Dimensionalität (Valenz, Idealisierungsversatz, Morphologie).
        """
        joints = self._classify_joint_valence(joints)
        joints = self._classify_joint_offset(joints)
        joints = self._classify_joint_morphology(joints)
        return joints


    def _classify_joint_valence(self, joints):
        """
        PLATZHALTER: Anschlussgrad (Valenz) -> kombinatorische Komplexität.
        - Primärmaß: Valenz |J|
        - Sekundärmaß: Dimensionsheterogenität, Innertopologische Dichte
          (Kanten im induzierten GT-Teilgraph / mögliche Kanten)
        """
        # TODO: Valenz, Dimensionsheterogenität und Innertopologische Dichte bestimmen
        return joints


    def _classify_joint_offset(self, joints):
        """
        PLATZHALTER: Idealisierungsversatz -> geometrische Komplexität.
        1. Versatzvektor definieren
        2. Kennzahlen ableiten
        3. Von paarweisen Werten zum Joint aggregieren
        """
        # TODO: Versatzvektor, Kennzahlen und Joint-Aggregation berechnen
        return joints


    def _classify_joint_morphology(self, joints):
        """
        PLATZHALTER: Topologisch-geometrischer Anschlusstyp (Morphologie).
        - Berührungsmodus: Wird bereits in Schritt 1 als 'contact_type'
          bzw. als Joint-'geometry_type' bestimmt.
        - Anschlusskonfiguration (I-, L-, T-, X-, Y-Anschluss)
        - Winkelcharakteristik
        """
        # TODO: Anschlusskonfiguration und Winkelcharakteristik bestimmen
        return joints


    # ---------------------------------------------------------
    # B.2 IST-JOINTS: EXTRAKTION AUS DEM ANALYSEMODELL
    # ---------------------------------------------------------
    def extract_analysis_joints(self, stem):
        """
        PLATZHALTER: Räumliche Ermittlung der Ist-Joints aus den
        Verbindungsmechanismen des Analysemodells:
        - Geteilter Knoten
        - Geteilte Linie / Knoten auf Flächenrand
        - Explizite Kopplungselemente
        Mit GUID-Rückverfolgung, damit später beide Joint-Cluster
        (Soll und Ist) verglichen werden können.
        HINWEIS: Große Teile liegen bereits in '_13_RFEM_Analyser.py' vor.
        """
        self.log("PLATZHALTER: Ist-Joint-Extraktion (Analysemodell) noch nicht implementiert.")
        joints = []
        # TODO: Verbindungsmechanismen auswerten (geteilter Knoten, geteilte Linie, Kopplungselemente)
        # TODO: GUID-Rückverfolgung implementieren
        return joints


    # ---------------------------------------------------------
    # B.3 VALIDIERUNG: MATCHING, FEHLERCODES & STATISTIK
    # ---------------------------------------------------------
    def match_joints(self, soll_joints, ist_joints):
        """
        PLATZHALTER: Beantwortet 'Welcher Ist-Joint gehört zu welchem Soll-Joint?'
        Zuordnungstabelle über Matching:
        - Informationsquellen: Bauteilmenge (Jaccard) + räumliche Nähe
          (globale Optimierung, Ungarische Methode)
        - Ergebnis: 3 Gruppen: gematchte Paare, unmatched Soll, unmatched Ist
        """
        self.log("PLATZHALTER: Joint-Matching noch nicht implementiert.")
        matching = {"matched": [], "unmatched_soll": [], "unmatched_ist": []}
        # TODO: Jaccard-Ähnlichkeit der Bauteilmengen + räumliche Nähe
        # TODO: Globale Optimierung (Ungarische Methode)
        return matching


    def classify_joint_errors(self, matching):
        """
        PLATZHALTER: Beantwortet 'Was ist genau an jedem Joint schiefgegangen?'
        Fehlertaxonomie: topologische Fehlercodes E0 bis E7 inkl.
        zugehörigem Entscheidungsbaum.
        """
        self.log("PLATZHALTER: Fehlerklassifikation (E0-E7) noch nicht implementiert.")
        error_codes = {}
        # TODO: Entscheidungsbaum der Fehlercodes E0-E7 je Joint anwenden
        return error_codes


    def evaluate_joint_statistics(self, matching, error_codes):
        """
        PLATZHALTER: Beantwortet 'Welche Anschlusstypen sind systematisch fehleranfällig?'
        - Deskriptive Auswertung: O-Klasse (Versatz) als Merkmal;
          Kreuztabellen Merkmal x Fehlertyp
        - Konfirmatorische Auswertung: logistische Regression und Odds Ratios
        - Explorative Auswertung: interpretierbare Regelextraktion
          (Entscheidungsbaum / Assoziationsregeln / SHAP-Werte;
          getrennte Modelle pro Fehlertyp)
        """
        self.log("PLATZHALTER: Statistische Auswertung noch nicht implementiert.")
        stats = {}
        # TODO: Deskriptive Kreuztabellen erstellen
        # TODO: Logistische Regression und Odds Ratios berechnen
        # TODO: Regelextraktion pro Fehlertyp durchführen
        return stats

if __name__ == "__main__":
    main()
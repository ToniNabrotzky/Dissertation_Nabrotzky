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
    Ist die Liste leer, werden alle IFC-Modelle im Ordner verarbeitet.
    """
    ifc_names = [
        # "21_22 L_TWP_Tragwerksmodell",
        # "23_24 LTWP-V__Dachtragwerk",
        # "20220421MODEL REV01",
        # "202102183458-Model",
        # "Grethes-hus-bok-2",
        # "Grethes-hus-bok-2",
        # "model_1",
        "model_3_1D",
        "model_3_2D",
        "model_4",
        "model_7",
        # "model_8",
        # "model_9", # 5,0m erforderlich
        "model_10_1D",
        "model_10_2D",
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
RBF_GAMMA = 0.0001
MAX_DIST = 3.0  # Einheitliche 3.0m Grenze für AABB und Exakt-Messung

ALLOWED_ENTITIES = [
    "IfcFooting", "IfcBeam", "IfcColumn", "IfcMember", "IfcSlab",
    "IfcWallStandardCase", "IfcWall", "IfcPlate", "IfcPile", "IfcBuildingElementProxy"
]

VIS_MAPPING = {
    "IfcFooting": ("#8B4513", "s"), "IfcBeam": ("#2E8B57", "h"),
    "IfcColumn": ("#FF4500", "p"), "IfcMember": ("#4682B4", "v"),
    "IfcWall": ("#808080", "H"), "IfcWallStandardCase": ("#A9A9A9", "D"),
    "IfcSlab": ("#B38719", "o"), "IfcPlate": ("#DDA0DD", "^"),
    "IfcPile": ("#000000", "|"), "IfcBuildingElementProxy": ("#FFD700", "*")
}

# =========================================================
# PIPELINE KLASSE
# =========================================================
class GNNPipeline:
    """
    Hauptklasse zur Verarbeitung von IFC-Modellen, Berechnung geometrischer Eigenschaften,
    Bestimmung von Distanzen und Erstellung von GNN-Graphen.
    """
    def __init__(self, model_names=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_db = os.path.normpath(os.path.join(self.base_dir, "..", "0 Model_Database"))
        self.dir_gt = os.path.join(self.base_dir, "2_1 Ground_Truth")
        self.dir_rbc = os.path.join(self.base_dir, "2_3 RBC_Labels")
        self.dir_out = os.path.join(self.base_dir, "2_4 Graph_Generation")
        
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
        """Führt alle Arbeitsschritte für ein einzelnes IFC-Modell durch."""
        self.log(f"\n{'='*50}\nVerarbeite Modell: {stem}\n{'='*50}")
        
        # 1. IFC Extraktion
        nodes, stats_ifc = self.extract_ifc_data(stem)
        if not nodes:
            self.log(f"Abbruch: Keine auswertbaren Knoten für {stem} gefunden.")
            return
        
        # 2. Labels einlesen
        gt_pairs = self.load_ground_truth(stem)
        rbc_dict = self.load_rbc_labels(stem)
        
        # 3. Kanten berechnen
        edges, stats_edges = self.calculate_edges(nodes, gt_pairs, rbc_dict)
        
        # 4. Graphen-Daten zusammenstellen und exportieren
        self.export_graph(stem, nodes, edges, stats_ifc, stats_edges)
        
        # 5. Visualisierung
        self.plot_graph(stem, nodes, edges, mode="2D")
        self.plot_graph(stem, nodes, edges, mode="3D")
        
        # 6. Log speichern
        log_path = os.path.join(self.dir_out, f"{stem} Protokoll.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.logs))

    # ---------------------------------------------------------
    # IFC EXTRAKTION & MATHEMATIK
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
    # LABELING LESE-FUNKTIONEN
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
    # DISTANZEN & KANTEN
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

    # ---------------------------------------------------------
    # EXPORT & VISUALISIERUNG
    # ---------------------------------------------------------
    def export_graph(self, stem, nodes, edges, stats_ifc, stats_edges):
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
                    "max_dist": MAX_DIST
                },
                "stats_IFC_extraction": stats_ifc,
                "stats_Edges_and_RBC": stats_edges
            },
            "nodes": clean_nodes,
            "edges": edges
        }
        
        path = os.path.join(self.dir_out, f"{stem} Graph.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        self.log(f"Graph JSON gespeichert unter: {os.path.basename(path)}")

    def plot_graph(self, stem, nodes, edges, mode="2D"):
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

if __name__ == "__main__":
    main()
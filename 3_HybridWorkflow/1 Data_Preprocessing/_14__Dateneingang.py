"""
_14_Dateneingang.py
===================
Gemeinsamer Dateneingang für beide Experimente.

Das Modul wird von '_14 IFC_Graph_Generator.py' eingebunden und stellt die
Klasse 'Dateneingang' bereit (Mixin der Klasse GNNPipeline).

Struktur:
1. KONFIGURATION (Dateneingang)
2. KLASSE Dateneingang
   - IFC-Extraktion inkl. Geometrie (Knoten des Kontaktgraphen)
   - Einlesen der Labels (Ground Truth, regelbasierte Labels)
"""
import os
import csv
from typing import Callable

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element

# =========================================================
# KONFIGURATION (Dateneingang)
# =========================================================
# IFC-Extraktion: Bauteilarten, die als Knoten übernommen werden
ALLOWED_ENTITIES = [
    "IfcBeam", "IfcColumn", "IfcSlab", "IfcWall", "IfcWallStandardCase",
    "IfcFooting", "IfcMember", "IfcPile", "IfcPlate", "IfcBuildingElementProxy"
]


# =========================================================
# DATENEINGANG: IFC-EXTRAKTION & LABELS
# =========================================================
class Dateneingang:
    """
    Liest die Eingangsdaten ein, die beide Experimente gemeinsam nutzen:
    Bauteile samt Geometrie aus dem IFC-Modell sowie die Labels.

    Erwartet von der Hauptklasse (GNNPipeline): die Pfade dir_db, dir_gt,
    dir_rbc sowie die Methode log().
    """
    dir_db: str
    dir_gt: str
    dir_rbc: str
    log: Callable[..., None]

    # ---------------------------------------------------------
    # IFC-EXTRAKTION & MATHEMATIK
    # ---------------------------------------------------------
    def _get_geometry_data(self, product, settings):
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


    def _is_load_bearing(self, product):
        """Prüft über das Property 'LoadBearing', ob ein Bauteil tragend ist."""
        try:
            for _, props in element.get_psets(product).items():
                if "LoadBearing" in props:
                    return bool(props["LoadBearing"])
        except: pass
        return False


    def extract_ifc_data(self, stem):
        r"""
        Extrahiert Knoten aus dem IFC-Modell inkl. Geometrie.
        
        Mathematischer Hintergrund:
        Oberfläche (Surface Area): Summe der Flächen aller Dreiecke.
        $ \text{Fläche} = \frac{1}{2} |\vec{AB} \times \vec{AC}| $
        
        Volumen (Mesh Volume): Divergenzsatz (Satz von Gauß).
        $ V = \sum \left( \frac{1}{6} \vec{A} \cdot (\vec{B} \times \vec{C}) \right) $

        Args:
            stem (str): Modellname ohne Dateiendung.

        Returns:
            tuple: (nodes, stats)
                nodes (list[dict]): Knoten mit Geometrie ('geom_raw') und Basismerkmalen.
                stats (dict): Statistik der Extraktion.
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
            if not self._is_load_bearing(prod):
                continue
            count_lb += 1
                
            geom_data = self._get_geometry_data(prod, settings)
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


    # ---------------------------------------------------------
    # LABELS EINLESEN
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

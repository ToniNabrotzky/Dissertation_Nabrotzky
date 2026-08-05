import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element
import ifcopenshell.util.placement as placement
import numpy as np
import os
import itertools
import csv
import time
from datetime import datetime
from scipy.spatial import cKDTree #type: ignore

def main():
    start_time = time.time()
    # Umwandlung in ein datetime-Objekt (lokale Zeit)
    dt_object = datetime.fromtimestamp(start_time)
    # Formatierung als Uhrzeit-String (Stunden:Minuten:Sekunden)
    uhrzeit_str = dt_object.strftime("%H:%M:%S")

    print(f"Starte Skript um {uhrzeit_str} Uhr...")

    # Konstanten für die Distanzberechnung zentral definieren
    BUFFER_DIST = 0.62  # Ziel-Toleranz: Bauteile mit (z.B. <= 25cm) Abstand gelten als verbunden (Kontakt)
    TRIA_DIST = 0.75     # Such-Radius: Erst bei grober Näherung (z.B. < 50cm) startet die teure, exakte Detailprüfung

    # Hier eine Liste der IFC-Dateinamen (ohne .ifc Dateiendung) eintragen
    # Die Dateien müssen in der in __init__ definierten Ordnerstruktur liegen
    ifc_names = [
        # "model_1",
        "model_2",
        # "model_3_1D",
        # "model_3_2D",
        "model_4",
        "model_5",
        "model_6",
        # "model_7",
        # "model_8",
        # "model_9",
        # "model_10_1D",
        # "model_10_2D",
    ]
    ifc_names = [
        "model_4",
    ]

    for ifc in ifc_names:
        analyzer = IfcDistanceAnalyzer(ifc)
        # Übergebe True für Triangulation (Genauigkeitsstufe 2), False für rein AABB
        # Nutze use_kdtree=True für die schnelle scipy Punkt-zu-Punkt Suche
        analyzer.process_and_measure(use_exact_mesh=True, use_kdtree=False,
                                     buffer_dist=BUFFER_DIST, tria_dist=TRIA_DIST)

    end_time = time.time()
    # Umwandlung in ein datetime-Objekt (lokale Zeit)
    dt_object = datetime.fromtimestamp(start_time)
    # Formatierung als Uhrzeit-String (Stunden:Minuten:Sekunden)
    uhrzeit_str = dt_object.strftime("%H:%M:%S")
    print(f"...Ende Skript um {uhrzeit_str} Uhr")

    dauer_sekunden = end_time - start_time
    # Umwandlung in Minuten, Sekunden
    minuten, sekunden = divmod(dauer_sekunden, 60)
    print(f"\nGesamtdauer der Berechnung für {len(ifc_names)} Modelle: {int(minuten)} Minuten und {int(sekunden)} Sekunden")


class IfcDistanceAnalyzer:
    def __init__(self, model_stem):
        self.model_stem = model_stem

        # Pfade definieren
        self.base_dir = os.path.dirname(__file__) # dirname(os.path.abspath(__file__))
        self.ifcmodels_dir = os.path.normpath(os.path.join(self.base_dir, "..", "0 Model_Database"))
        self.path_ifc = os.path.join(self.ifcmodels_dir, f"{self.model_stem}.ifc")
        
        # Ausgabeordner definieren ("2_1 Ground_Truth")
        self.output_dir = os.path.join(self.base_dir, "2_1 Ground_Truth")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Ausgabeordner erstellt: {self.output_dir}")   
        
        if not os.path.exists(self.path_ifc):
            print(f"Fehler: IFC-Datei nicht gefunden unter {self.path_ifc}")
            self.model = None
            return
        
        print(f"\n___Lade Modell: {self.model_stem}...")
        try:
            self.model = ifcopenshell.open(str(self.path_ifc))
            self.settings = ifcopenshell.geom.settings()
            print(f"\t{self.model_stem} erfolgreich geladen.")
        except Exception as e:
            print(f"\tKritischer Fehler beim Öffnen der Datei: {e}")
            self.model = None

    def is_load_bearing(self, product):
        """Prüft, ob ein Bauteil das Attribut 'LoadBearing' = True besitzt."""
        try:
            psets = element.get_psets(product)
            for pset_name, properties in psets.items():
                if "LoadBearing" in properties:
                    return bool(properties["LoadBearing"])
        except Exception:
            pass
        return False
            
    def get_z_origin(self, product):
        """Liest die Z-Koordinate des lokalen Ursprungs (ObjectPlacement) eines IFC-Elements aus"""
        try:
            placement_obj = product.ObjectPlacement
            if placement_obj:
                # Liefert eine 4x4 Transformationsmatrix
                matrix = placement.get_local_placement(placement_obj)
                # Die Z-Translation befindet sich in Zeile 2, Spalte 3 (0-basiert)
                return float(matrix[2, 3])
        except Exception as e:
            pass
        return 0.0

    def get_geometry_data(self, product):
        """Extrahiert das triangulierte Mesh eines Elements in globale Koordinaten inkl. Kanten-Abtastung"""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            if not shape or not shape.geometry: # type: ignore
                return None
            
            verts = shape.geometry.verts # type: ignore
            faces = shape.geometry.faces # type: ignore
            if not verts or not faces:
                return None
            
            # Gruppiere Listen
            local_verts = np.array(verts).reshape(-1, 3)
            faces_idx = np.array(faces).reshape(-1, 3)
            
            # Transformationsmatrix (4x4) anwenden (Column-Major aus ifcopenshell)
            matrix = np.array(shape.transformation.matrix).reshape(4, 4) # type: ignore
            homogeneous_verts = np.hstack((local_verts, np.ones((local_verts.shape[0], 1))))
            global_verts = homogeneous_verts.dot(matrix)[:, :3]
            
            # Hole alle Dreiecke
            triangles = global_verts[faces_idx]
            
            # Eindeutige Kanten extrahieren
            edges = set()
            for f in faces_idx:
                for i in range(3):
                    v1, v2 = f[i], f[(i+1)%3]
                    edges.add(tuple(sorted((v1, v2))))
                    
            # Abtasten der Kanten (Resolution = 0.025 m = 2.5 cm)
            resolution = 0.025 
            edge_points = []
            for e in edges:
                p1 = global_verts[e[0]]
                p2 = global_verts[e[1]]
                length = np.linalg.norm(p2 - p1)
                
                if length > 1e-5:
                    num_steps = int(np.ceil(length / resolution))
                    if num_steps > 1000:
                        num_steps = 1000
                    
                    t = np.linspace(0, 1, num_steps + 1).reshape(-1, 1)
                    sampled = p1 + t * (p2 - p1)
                    edge_points.append(sampled)
            
            if edge_points:
                test_points = np.vstack(edge_points)
            else:
                test_points = global_verts
                
            return {
                "vertices": global_verts,
                "triangles": triangles,
                "test_points": test_points
            }
        except Exception as e:
            print(f"Fehler bei Geometrie-Extraktion für {product.Name}: {e}")
            return None
            
    def get_aabb(self, vertices):
        """Liefert die Bounding Box Min/Max Koordinaten"""
        return np.min(vertices, axis=0), np.max(vertices, axis=0)

    def distance_aabb(self, aabb1, aabb2):
        """Berechnet den Abstand zweier Bounding Boxes (AABB) im 3D-Raum. Gibt 0 zurück wenn sie überlappen."""
        min1, max1 = aabb1
        min2, max2 = aabb2
        
        dx = max(0, max(min1[0] - max2[0], min2[0] - max1[0]))
        dy = max(0, max(min1[1] - max2[1], min2[1] - max1[1]))
        dz = max(0, max(min1[2] - max2[2], min2[2] - max1[2]))
        
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def point_to_segment_distance(self, p, a, b):
        """Berechnet den Abstand von Punkten p zu einem Liniensegment ab"""
        ab = b - a
        ap = p - a
        denom = np.maximum(np.sum(ab * ab), 1e-8)
        t = np.sum(ap * ab, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        closest = a + t.reshape(-1, 1) * ab
        return np.sqrt(np.sum((p - closest)**2, axis=1))

    def point_to_triangle_distance(self, p, v0, v1, v2):
        """Berechnet den kürzesten Abstand von Punkten p zu einem Dreieck v0, v1, v2"""
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        normal = np.cross(v0v1, v0v2)
        area2 = np.sum(normal**2)
        
        if area2 < 1e-8:
            return self.point_to_segment_distance(p, v0, v1)
            
        n = normal / np.sqrt(area2)
        
        v0p = p - v0
        dist_plane = np.abs(np.sum(v0p * n, axis=1))
        
        proj = p - dist_plane.reshape(-1, 1) * n
        
        edge0 = v1 - v0
        vp0 = proj - v0
        c0 = np.cross(edge0, vp0)
        
        edge1 = v2 - v1
        vp1 = proj - v1
        c1 = np.cross(edge1, vp1)
        
        edge2 = v0 - v2
        vp2 = proj - v2
        c2 = np.cross(edge2, vp2)
        
        inside = (np.sum(c0 * n, axis=1) >= -1e-5) & \
                 (np.sum(c1 * n, axis=1) >= -1e-5) & \
                 (np.sum(c2 * n, axis=1) >= -1e-5)
                 
        dist = np.full(p.shape[0], np.inf)
        dist[inside] = dist_plane[inside]
        
        outside = ~inside
        if np.any(outside):
            p_out = p[outside]
            d0 = self.point_to_segment_distance(p_out, v0, v1)
            d1 = self.point_to_segment_distance(p_out, v1, v2)
            d2 = self.point_to_segment_distance(p_out, v2, v0)
            dist[outside] = np.minimum(d0, np.minimum(d1, d2))
            
        return dist

    def exact_mesh_distance(self, geomA, geomB, aabbA, aabbB, max_dist=0.05):
        """
        Berechnet den genauen Abstand zwischen zwei Mesh-Geometrien.
        Nutzt AABB auf Dreiecksebene, um irrelevante Dreiecke frühzeitig auszufiltern.
        """
        min_dist = float('inf')
        
        pts_A = geomA["test_points"]
        for tri in geomB["triangles"]:
            # 1. AABB des einzelnen Dreiecks berechnen
            tri_min = np.min(tri, axis=0)
            tri_max = np.max(tri, axis=0)
            
            # 2. Prüfen, ob das Dreieck überhaupt in Reichweite von Bauteil A ist
            if self.distance_aabb(aabbA, (tri_min, tri_max)) > max_dist:
                continue # Dreieck ist zu weit weg, überspringen!
                
            dists = self.point_to_triangle_distance(pts_A, tri[0], tri[1], tri[2])
            current_min = np.min(dists)
            if current_min < min_dist:
                min_dist = current_min
                
        pts_B = geomB["test_points"]
        for tri in geomA["triangles"]:
            # 1. AABB des einzelnen Dreiecks berechnen
            tri_min = np.min(tri, axis=0)
            tri_max = np.max(tri, axis=0)
            
            # 2. Prüfen, ob das Dreieck überhaupt in Reichweite von Bauteil B ist
            if self.distance_aabb(aabbB, (tri_min, tri_max)) > max_dist:
                continue # Dreieck ist zu weit weg, überspringen!
                
            dists = self.point_to_triangle_distance(pts_B, tri[0], tri[1], tri[2])
            current_min = np.min(dists)
            if current_min < min_dist:
                min_dist = current_min
                
        return min_dist

    def distance_kdtree(self, tree_A, geomB):
        """
        Berechnet den Abstand mit scipy's cKDTree (Punkt-zu-Punkt).
        Der Urpsungsbaum A wurde bereits gebaut und hier übergeben.
        Das verhindert eine für jeden Vergleich eine redundante Neubildung des Baumes.
        """
        pts_B = geomB["test_points"]

        # Frage den vorbereiteten Baum mit den Punkten von Bauteil B ab.
        # k=1 bedeutet: Finde nur den EINEN allernächsten Nachbarn.
        # workers=-1 bedeutet: Nutze alle verfügbaren CPU-Kerne für maximale Geschwindigkeit!
        distances, _ = tree_A.query(pts_B, k=1, workers=-1)
        
        # Das Ergebnis ist eine Liste der kürzesten Abstände für jeden Punkt aus B.
        # Davon nehmen wir den kleinsten Wert:
        return np.min(distances)

    def process_and_measure(self, use_exact_mesh=False, use_kdtree=False, buffer_dist=0.25, tria_dist=0.5):
        """Hauptprozess: Sammelt Bauteile, sortiert nach Z-Koordinate, berechnet Distanzen und exportiert CSV."""
        if getattr(self, 'model', None) is None:
            return
            
        # Sammeln aller physischen Bauteile (IfcElement deckt Wände, Decken, Träger etc. ab)
        products = self.model.by_type("IfcElement") # type: ignore
        elements_data = []
        
        print("\nExtrahiere tragende Bauteile und Geometrie...")
        for prod in products:
            if not self.is_load_bearing(prod):
                continue
                
            geom_data = self.get_geometry_data(prod)
            if geom_data is None:
                continue
                
            aabb = self.get_aabb(geom_data["vertices"])
            z_origin = self.get_z_origin(prod)

            # KD-Tree direkt hier 1x pro Bauteil aufbauen, wenn die Funktion aktiv ist (Caching!)
            # Das spart Rechenzeit, da er nicht bei jedem Vergleich neu gebaut werden muss.
            kdtree_obj = None
            if use_kdtree:
                kdtree_obj = cKDTree(geom_data["test_points"])
            
            elements_data.append({
                "guid": prod.GlobalId,
                "name": prod.Name or "Unnamed",
                "type": prod.is_a(),
                "geom": geom_data,
                "aabb": aabb,
                "z_origin": z_origin,
                "kdtree": kdtree_obj
            })

        # 1. Bauteile nach Z-Koordinate des Ursprungs absteigend sortieren (höchstes zuerst)
        elements_data.sort(key=lambda x: x["z_origin"], reverse=True)
        
        print(f"\t{len(elements_data)} Bauteile mit Geometrie bereit zur Berechnung.")
        print(f"""\tStarte Distanz-Berechnungen (
              Level 1 AABB, 
              Level 2 Exakt: {use_exact_mesh},
              Level 2 KDTree: {use_kdtree}...""")
        
        results = []
        
        # 2. Vergleich aller Bauteil-Paare.
        # Da elements_data sortiert ist, ist elemA immer das höhere (oder gleich hohe) Bauteil im Vergleich zu elemB.
        # itertools.combinations verhindert zudem A-B und B-A Duplikate.
        for elemA, elemB in itertools.combinations(elements_data, 2):
            
            # Genauigkeitsstufe 1: AABB Distanz
            aabb_dist = self.distance_aabb(elemA["aabb"], elemB["aabb"])
            
            is_connected = False
            
            if aabb_dist <= buffer_dist:
                # Treffer durch AABB
                is_connected = True
                
            # !!! Variante A dauert in Berechnung länger als Variante B !!!
            elif use_kdtree and aabb_dist < tria_dist:
                # Genauigkeitsstufe 2 Variante A: Scipy KD-Tree (mit gecachtem Baum)
                kdtree_dist = self.distance_kdtree(elemA["kdtree"], elemB["geom"])
                
                if kdtree_dist <= buffer_dist:
                    is_connected = True

            elif use_exact_mesh and aabb_dist < tria_dist:
                # Genauigkeitsstufe 2 Variante B: Exakte Triangulationsmessung (mit Dreiecks-AABB)
                exact_dist = self.exact_mesh_distance(elemA["geom"], elemB["geom"], elemA["aabb"], elemB["aabb"], buffer_dist)
                
                if exact_dist <= buffer_dist:
                    # Treffer durch exakte Triangulationsmessung
                    is_connected = True
                    
            if is_connected:
                # Verbindung für kurzfristige Liste (results) speichern
                results.append({
                    "Status": "auto",
                    "GUID_A": elemA["guid"],
                    "Name_A": elemA["name"],
                    "Type_A": elemA["type"],
                    "GUID_B": elemB["guid"],
                    "Name_B": elemB["name"],
                    "Type_B": elemB["type"],
                })
                # print(f"Verbindung gefunden: [{elemA['name']}] <--> [{elemB['name']}]")
        
        print(f"\nAbschluss: Es wurden {len(results)} gültige Verbindungen (<= {buffer_dist*100}cm) ermittelt.")
        
        # 3. CSV Export
        if not results:
            print("\tKeine Verbindungen gefunden. Export wird übersprungen.")
            return
            
        out_csv_path = os.path.join(self.output_dir, f"{self.model_stem} GT.csv")
        
        try:
            with open(out_csv_path, mode='w', newline='', encoding='utf-8') as f:
                # Semikolon als Standard-Delimiter für deutsche Excel-Systeme (oftmals präferiert)
                # Alternativ kann delimiter=',' genutzt werden, falls das das exakte Format von GT_Labels erfordert
                writer = csv.writer(f, delimiter=';')
                
                # Header schreiben
                writer.writerow(["Status", "Type_A", "GUID_A", "Name_A", "Type_B", "GUID_B", "Name_B"])

                previous_guid_a = None
                
                # Daten schreiben. Die Reihenfolge in 'results' ist bereits automatisch nach
                # Z-Koordinate von A sortiert, da elements_data sortiert an 'combinations' übergeben wurde!
                for r in results:
                    current_guid_a = r["GUID_A"]

                    # Wenn sich die GUID von Element A ändert, entsteht eine Leerzeile für optische Blöcke und besserer Erkennbarkeit
                    if previous_guid_a is not None and current_guid_a != previous_guid_a:
                        writer.writerow(["", "", "", "", "", "", ""])

                    writer.writerow([
                        r["Status"],
                        # r["Type_A"], r["GUID_A"], r["Name_A"],
                        # r["Type_B"], r["GUID_B"], r["Name_B"],
                        r["Type_A"], r["Type_B"],
                        r["GUID_A"], r["GUID_B"],
                        r["Name_A"], r["Name_B"],
                    ])

                    # Update für den nächsten Schleifendurchlauf
                    previous_guid_a = current_guid_a
                    
            print(f"\nErgebnisse erfolgreich exportiert nach: {out_csv_path}")
        except Exception as e:
            print(f"\nFehler beim Schreiben der CSV-Datei: {e}")


if __name__ == "__main__":
    main()
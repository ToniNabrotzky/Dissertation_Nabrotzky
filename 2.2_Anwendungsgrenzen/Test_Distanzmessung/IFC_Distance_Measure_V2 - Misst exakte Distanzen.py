import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element
import numpy as np
import os
import itertools


def main():
    # Hier einfach den Namen (ohne .ifc Dateiendung) eintragen
    # Die Datei muss im selben Ordner liegen wie dieses Python-Skript
    ifc_names = [
        # "Dein_Ifc_Modell_Name", 
        # "21_22 L_TWP_Tragwerksmodell",
        "model_0",
    ]

    for ifc in ifc_names:
        analyzer = IfcDistanceAnalyzer(ifc)
        analyzer.process_and_measure()


class IfcDistanceAnalyzer:
    def __init__(self, model_stem):
        self.model_stem = model_stem

        self.base_dir = os.path.dirname(__file__)
        self.path_ifc = os.path.join(self.base_dir, f"{self.model_stem}.ifc")
        
        if not os.path.exists(self.path_ifc):
            print(f"Fehler: IFC-Datei nicht gefunden unter {self.path_ifc}")
            return
        
        print(f"Lade Modell: {self.model_stem}...")
        try:
            self.model = ifcopenshell.open(str(self.path_ifc))
            self.settings = ifcopenshell.geom.settings()
            print("IFC-Modell erfolgreich geladen.\n")
        except Exception as e:
            print(f"Kritischer Fehler beim Öffnen der Datei: {e}")
            del self.model
        
    def is_load_bearing(self, product):
        psets = element.get_psets(product)
        for pset_name, properties in psets.items():
            if "LoadBearing" in properties:
                return bool(properties["LoadBearing"])
        return False
    
    def get_geometry_data(self, product):
        """Extrahiert das triangulierte Mesh eines Elements in globale Koordinaten inkl. Kanten-Abtastung"""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            if not shape or not shape.geometry:
                return None
            
            verts = shape.geometry.verts
            faces = shape.geometry.faces
            if not verts or not faces:
                return None
            
            # Gruppiere Listen
            local_verts = np.array(verts).reshape(-1, 3)
            faces_idx = np.array(faces).reshape(-1, 3)
            
            # Transformationsmatrix (4x4) anwenden (Column-Major)
            matrix = np.array(shape.transformation.matrix).reshape(4, 4)
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
            # Dadurch wird sichergestellt, dass auch sich kreuzende Geometrien 
            # oder parallele Wand-Kanten den korrekten Abstand finden.
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
        """Berechnet den Abstand zweier Bounding Boxes (AABB) im 3D-Raum"""
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

    def exact_mesh_distance(self, geomA, geomB):
        """
        Berechnet den genauen Abstand zwischen zwei Mesh-Geometrien,
        indem Kanten-Abtastpunkte gegen die Dreiecke des anderen Elements getestet werden.
        """
        min_dist = float('inf')
        
        pts_A = geomA["test_points"]
        for tri in geomB["triangles"]:
            dists = self.point_to_triangle_distance(pts_A, tri[0], tri[1], tri[2])
            current_min = np.min(dists)
            if current_min < min_dist:
                min_dist = current_min
                
        pts_B = geomB["test_points"]
        for tri in geomA["triangles"]:
            dists = self.point_to_triangle_distance(pts_B, tri[0], tri[1], tri[2])
            current_min = np.min(dists)
            if current_min < min_dist:
                min_dist = current_min
                
        return min_dist

    def process_and_measure(self):
        """Hauptprozess: Filtert tragende Bauteile, extrahiert Geometrie, berechnet Distanzen"""
        if getattr(self, 'model', None) is None:
            return
            
        products = self.model.by_type("IfcProduct")
        elements_data = []
        
        print("Extrahiere tragende Bauteile und Geometrie...")
        for prod in products:
            if not self.is_load_bearing(prod):
                continue
                
            geom_data = self.get_geometry_data(prod)
            if geom_data is None:
                continue
                
            aabb = self.get_aabb(geom_data["vertices"])
            
            elements_data.append({
                "guid": prod.GlobalId,
                "name": prod.Name or "Unnamed",
                "geom": geom_data,
                "aabb": aabb
            })

        print(f"{len(elements_data)} tragende Bauteile mit Geometrie bereit zur Berechnung.\n")
        print("Starte Distanz-Berechnungen...")
        
        results = []
        
        # Vergleich aller Bauteil-Paare
        for elemA, elemB in itertools.combinations(elements_data, 2):
            aabb_dist = self.distance_aabb(elemA["aabb"], elemB["aabb"])
            
            # Nur genaue Prüfung durchführen, wenn die AABB Distanz < 1.0m ist
            if aabb_dist < 1.0:
                exact_dist = self.exact_mesh_distance(elemA["geom"], elemB["geom"])
                
                results.append({
                    "elemA_name": elemA["name"],
                    "elemA_guid": elemA["guid"],
                    "elemB_name": elemB["name"],
                    "elemB_guid": elemB["guid"],
                    "distance": exact_dist
                })
                
                print(f"[{elemA['name']} ({elemA['guid']})] <--> [{elemB['name']} ({elemB['guid']})]: {exact_dist:.3f} m")
        
        print(f"\nAbschluss: Es wurden {len(results)} Kontaktpunkte / Näherungen (< 1.0m) ermittelt.")

if __name__ == "__main__":
    main()

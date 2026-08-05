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
        "21_22 L_TWP_Tragwerksmodell",
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
    
    def get_global_vertices(self, product):
        """Extrahiert das triangulierte Mesh eines Elements in globale Koordinaten"""
        try:
            shape = ifcopenshell.geom.create_shape(self.settings, product)
            if not shape or not shape.geometry:
                return None
            
            verts = shape.geometry.verts
            if not verts:
                return None
            
            # Gruppiere flache Liste [x,y,z,x,y,z...] in Matrix (N, 3)
            local_verts = np.array(verts).reshape(-1, 3)
            
            # Transformationsmatrix (4x4) des IfcShape aufbauen
            # In ifcopenshell ist shape.transformation.matrix ein flaches Tupel im Column-Major Format
            matrix = np.array(shape.transformation.matrix).reshape(4, 4)
            
            # Homogene Koordinaten ([x, y, z, 1]) für die Multiplikation hinzufügen
            ones = np.ones((local_verts.shape[0], 1))
            homogeneous_verts = np.hstack((local_verts, ones))
            
            # Da ifcopenshell Matrizen Column-Major sind, können wir bei (N, 4) Inputs
            # direkt mit der reshape(4,4) Matrix post-multiplizieren. 
            global_verts = homogeneous_verts.dot(matrix)[:, :3]
            
            return global_verts
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

    def exact_vertex_distance(self, verts1, verts2):
        """Berechnet den kürzesten euklidischen Abstand zwischen allen Vertex-Paaren der Elemente"""
        min_dist = float('inf')
        # Um enorme Speicherbelastung (NxM Matrix) zu verhindern, layern wir über verts1
        for v in verts1:
            dists = np.sqrt(np.sum((verts2 - v)**2, axis=1))
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
                
            global_verts = self.get_global_vertices(prod)
            if global_verts is None:
                continue
                
            aabb = self.get_aabb(global_verts)
            
            elements_data.append({
                "guid": prod.GlobalId,
                "name": prod.Name or "Unnamed",
                "vertices": global_verts,
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
                exact_dist = self.exact_vertex_distance(elemA["vertices"], elemB["vertices"])
                
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

import os
import re
import uuid
import json
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx # WICHTIG: Python 3.14 hat Bug. Erfordere 3.12!!!

# Versuche RFEM Module zu importieren. Dies setzt eine funktionierende RFEM 6 Python API Umgebung voraus.
try:
    from RFEM.initModel import Model
    from RFEM.enums import ObjectTypes
    from RFEM.Tools.GetObjectNumbersByType import GetObjectNumbersByType
except ImportError:
    print("RFEM Module konnten nicht importiert werden. Sicherstellen, dass die RFEM 6 Python API installiert ist.")

# Konfiguration für Modellnamen (IFC_STEM) - Dient als Basis für alle Datei-Ein/Ausgaben
IFC_STEM = "model_4"

# Konfiguration für die Visualisierung in Anlehnung an Vorlage 14
VIS_MAPPING = {
    "Stab": ("#4682B4", "v"),     # Stahlblau, Dreieck nach unten
    "Fläche": ("#808080", "H"),  # Grau, Hexagon (groß)
    "Unbekannt": ("#000000", "o") # Schwarz, Kreis
    }

# Dateinamen für den Export
CSV_FILENAME = f"{IFC_STEM} RFEM_Labels.csv"
JSON_FILENAME = f"{IFC_STEM} RFEM_Graph.json"
GRAPH_BASENAME = f"{IFC_STEM} RFEM_Graph_2D"


def main():
    """
    Ablauf:
    - Verbindung aufbauen
    - Analyse durchführen (Topologie & Geometrie)
    - Bauteile extrahieren
    - Labeling durch Abgleich mit Ground Truth
    - Dateien (CSV & JSON) exportieren
    - Graph plotten und speichern
    """
    print("="*60)
    print(f"Starte automatisierte RFEM Analyse für Modell: {IFC_STEM}...")

    # Pfadstruktur anlegen
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dir_2_1 = os.path.join(base_dir, "2_1 Ground_Truth")
    dir_2_2 = os.path.join(base_dir, "2_2 Analysemodell")
    dir_2_3 = os.path.join(base_dir, "2_3 RBC_Labels")

    # Verbindung zu RFEM über SOAP-API
    try:
        Model(new_model=False, model_name="") #type: ignore
    except Exception as e:
        print(f"Fehler: Konnte keine Verbindung zu RFEM herstellen: {e}")
        return
    
    if not hasattr(Model, 'clientModel') or Model.clientModel is None: #type: ignore
        print(f"Verbindungsfehler: Aktives Modell in RFEM konnte nicht initialisiert werden.")
    
    service = Model.clientModel.service #type: ignore
    print(f"\tErfolgreich verbunden!")

    # 1. Zuordnung aufbauen
    print(f"\n1. Erstelle Zuordnung von Knoten/Linien zu übergeordneten Bauteilen...")
    node_maps, line_maps = build_parent_maps(service)

    # 2. Verbindungen / Kopplungen auslesen
    print(f"\n2. Identifiziere Bauteil-Verbindungen aus den generierten Kopplungen...")
    edges, csv_data = analyze_connections(service, node_maps, line_maps)

    # 3. Bauteile für den Graph laden
    print(f"\n3. Erstelle Graphen-Knoten...")
    graph_nodes = extract_graph_elements(service)

    # 4. Ground-Truth (GT) Labeling durchführen
    print(f"\n4. Gleiche Verbindungen mit Ground Truth ab (Labeling TP/FP/FN)...")
    gt_pairs = load_ground_truth(dir_2_1, IFC_STEM)
    csv_data, edges = apply_labels_and_add_fn(csv_data, edges, gt_pairs, graph_nodes)

    # --- CSV EXPORT ---
    print(f"\nStarte CSV-Export von {CSV_FILENAME}...")
    csv_path = os.path.join(dir_2_3, CSV_FILENAME)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=";")

        # Header
        writer.writerow(['Verbindungs_Nr', 'Verbindungs_Art',
                         'Begrenzung_1_Nr', 'Begrenzung_2_Nr',
                         'Elem_A_Typ', 'Elem_A_Nr',
                         'Elem_B_Typ', 'Elem_B_Nr',
                         'IFC_GUID_A', 'IFC_GUID_B',
                         'Typus', 'Label', 'Kommentar'])
        writer.writerows(csv_data)
    print(f"\tCSV-Export erfolgreich: {CSV_FILENAME} ({len(csv_data)} Verbindungen)")

    # --- JSON EXPORT (Graph) ---
    print(f"\nStarte Export von {JSON_FILENAME}...")
    json_path = os.path.join(dir_2_3, JSON_FILENAME)
    # Entferne die temporäre 'Node_ID' aus dem Export, um exakt die Vorlage zu matchen
    clean_json_output = []
    for node in graph_nodes:
        clean_node = {"Graph_ID": node["Graph_ID"], "RFEM": node["RFEM"]}
        clean_json_output.append(clean_node)
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(clean_json_output, f, indent=4, ensure_ascii=False)
    print(f"\tJSON-Export erfolgreich: {JSON_FILENAME} ({len(clean_json_output)} Elemente)")

    # --- GRAPH PLOTTEN ---
    try:
        print(f"\nErstelle Graph-Plot (2D Topologie)...")
        plot_rfem_graph(graph_nodes, edges, dir_2_3)
    except Exception as e:
        print(f"Fehler bei der Graphen-Generierung (Plot): {e}")

    print("="*60)
    print(f"...Analyse vollständig abgeschlossen!")


"""ANALYSE HILFSFUNKTIONEN"""
def uuid_to_ifc_guid(uuid_string: str):
    """
    Konvertiert eine standardmäßige 36-stellige UUID (aus RFEM) in eine 
    22-stellige IFC-GUID (Base64 Encoding).
    """
    try:
        b = uuid.UUID(str(uuid_string)).bytes
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"

        def encode(num, length):
            """Hilfsfunktion zum Trennen der Zeichen"""
            res = ""
            for _ in range(length):
                res = chars[num % 64] + res
                num //= 64
            return res
        
        res = []
        res.append(encode(b[0], 2))
        res.append(encode((b[1] << 16) | (b[2] << 8) | b[3], 4))
        res.append(encode((b[4] << 16) | (b[5] << 8) | b[6], 4))
        res.append(encode((b[7] << 16) | (b[8] << 8) | b[9], 4))
        res.append(encode((b[10] << 16) | (b[11] << 8) | b[12], 4))
        res.append(encode((b[13] << 16) | (b[14] << 8) | b[15], 4))

        return "".join(res)
    except Exception as e:
        return f"Fehler bei Umrechnung {e}"
    
def extract_ifc_guid(obj_str: str):
    """
    Sucht mittels regulärer Ausdrücke (Regex) im String-Repräsentanten eines RFEM-Objektes nach UUIDs oder IFC-GUIDs und gibt die IFC-GUID zurück.
    """
    match = re.search(r'id_for_export_import\s*=\s*["\']?([a-fA-F0-9\-]{36})["\']?', obj_str)
    if match:
        return uuid_to_ifc_guid(match.group(1))
    
    # Fallback, falls direkt eine GlobalId hinterlegt ist
    match_global = re.search(r'GlobalId.*?([a-zA-Z0-9_\$]{22})', obj_str, re.IGNORECASE)
    if match_global:
        return match_global.group(1)
    return None


def parse_id_list(id_string: str):
    """
    Wandelt RFEM-typische ID-Strings (z.B. "2 4 7") in eine saubere Liste von Ganzzahlen (Integers) um, damit diese im Code iteriert werden können.
    """
    if not id_string or str(id_string).strip() in ['Keine', 'None', '']:
        return []
    # Verhindern, dass aus Python-Objekt-Strings falsche Zahlen extrahiert werden
    if '<' in str(id_string) and '>' in str(id_string):
        return []
    return [int(n) for n in re.findall(r'\d+', str(id_string))]

def parse_int_safely(s: str):
    """
    Extrahiert sicher die erste Ziffer/Zahl aus einem String.
    Wichtig für fehlerhafte oder unsaubere API-Rückgaben
    """
    if '<' in str(s) and '>' in str(s):
        return 0
    match = re.search(r'\d+', str(s))
    return int(match.group()) if match else 0

def build_parent_maps(service):
    """
    Erstellt Dictionaries (Maps), die Knoten und Linien zu ihren übergeordneten Bauteilen (Stäbe, Flächen) zuordnen.
    Dies ist essenziel, um bei einer Kopplung (die Knoten oder Linien verbindet) herauszufinden, welche Bauteile miteinander interagieren.
    Rückgabe: (node_to_parents, line_to_parents)
    """
    print(f"\tErstelle Topologie-Zuordnungen (Parent-Maps)...")
    node_to_parents = {}
    line_to_parents = {}

    def add_to_map(mapping, child_no, parent):
        """Hilfsfunktion zum Einsortieren in Mapping-Liste"""
        if child_no <= 0: return
        if child_no not in mapping:
            mapping[child_no] = []
        if parent not in mapping[child_no]:
            mapping[child_no].append(parent)
    
    def map_line_and_nodes(line_no, parent):
        """Hilfsfunktion zum Mappen einer Linie und all ihrer (internen, begrenzenden) Knoten"""
        add_to_map(line_to_parents, line_no, parent)
        try:
            line_obj = service.get_line(line_no)

            # Geometrie-Knoten und mögliche interne Knoten abfragen
            for attr in ['definition_nodes', 'nodes_on_line']: # alt: 'connected_nodes'
                nodes_str = getattr(line_obj, attr, '')
                for nodes_no in parse_id_list(nodes_str):
                    add_to_map(node_to_parents, nodes_no, parent)
        except Exception:
            pass # Linie nicht gefunden oder Zugriffsfehler

    # 1. STÄBE durchsuchen und deren Knoten/Linien mappen
    try:
        member_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_MEMBER) #type: ignore
        if not isinstance(member_numbers, (list, tuple)):
            member_numbers = parse_id_list(str(member_numbers))
        
        for no in member_numbers:
            member = service.get_member(no)
            if str(getattr(member, 'type', '')) == 'TYPE_RIGID':
                continue # Starre Stäbe (Knotenkopplungen) sind Verbindungen, keine Eltern-Bauteile
            
            parent = ('Stab', member.no)

            # Ein Stab hängt an einer Linie. Wir mappen diese Linie mit ihren Knoten
            line_no = getattr(member, 'line', 0) # --> RFEM indiziert Objekte beginnend mit 1
            if line_no > 0:
                map_line_and_nodes(line_no, parent)
            
            # Falls der Stab direkt interne Knoten hat (Nodes on Member)
            for node_no in parse_id_list(getattr(member, 'nodes_on_member', '')):
                add_to_map(node_to_parents, node_no, parent)  
    except Exception as e:
        print(f"Warnung bei Stab-Mapping: {e}")
    
    # 2. FLÄCHEN durchsuchen und deren Knoten/Linien mappen
    try:
        surface_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_SURFACE) #type: ignore
        if not isinstance(surface_numbers, (list, tuple)):
            surface_numbers = parse_id_list(str(surface_numbers))
        
        for no in surface_numbers:
            surface = service.get_surface(no)
            parent = ('Fläche', surface.no)

            # A) Begrenzungslinien inkl. deren Knoten mappen
            for l_no in parse_id_list(getattr(surface, 'boundary_lines', '')):
                map_line_and_nodes(l_no, parent)
                
            # B) Interne (integrierte) Linien inkl. deren Knoten mappen
            for l_no in parse_id_list(getattr(surface, 'integrated_lines', '')): # alt: 'internal_lines'
                map_line_and_nodes(l_no, parent)
                
            # C) Direkte interne (integrierte) Knoten mappen (die auf keiner Linie liegen)
            for n_no in parse_id_list(getattr(surface, 'integrated_nodes', '')):
                add_to_map(node_to_parents, n_no, parent)               
    except Exception as e:
        print(f"Warnung bei Flächen-Mapping: {e}")
    
    # 3. Generierte/Interne Knoten rückwärts durchsuchen
    try:
        node_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_NODE) #type: ignore
        if not isinstance(node_numbers, (list, tuple)):
            node_numbers = parse_id_list(str(node_numbers))
            
        for no in node_numbers:
            try:
                node = service.get_node(no)
                line_no = parse_int_safely(getattr(node, 'on_line_reference_line', getattr(node, 'line', 0))) #type: ignore
                if line_no > 0:
                    for parent in line_to_parents.get(line_no, []):
                        add_to_map(node_to_parents, no, parent)
                        
                member_no = parse_int_safely(getattr(node, 'on_member_reference_member', getattr(node, 'member', 0))) #type: ignore
                if member_no > 0:
                    add_to_map(node_to_parents, no, ('Stab', member_no))
            except Exception:
                pass
    except Exception as e:
        print(f"Warnung bei Knoten-Mapping: {e}")
    
    return node_to_parents, line_to_parents


"""GEOMETRIE HILFSFUNKTIONEN FÜR RECHNERISCHE ANALYSE"""
def geo_dist(p1, p2):
    """Berechnet den euklidischen Abstand zweier 3D-Punkte."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def geo_vec_sub(p1, p2):
    """Subtrahiert zwei 3D-Vektoren voneinander."""
    return (p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2])

def geo_dot(v1, v2):
    """Berechnet das Skalarprodukt (Dot-Product) zweier Vektoren."""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def geo_plane_from_poly(poly):
    """Erstellt den Normalenvektor einer Fläche basierend auf ihren Randpolygonen (Newell-Methode)."""
    nx, ny, nz = 0, 0, 0
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i+1)%len(poly)]
        nx += (p1[1] - p2[1]) * (p1[2] + p2[2])
        ny += (p1[2] - p2[2]) * (p1[0] + p2[0])
        nz += (p1[0] - p2[0]) * (p1[1] + p2[1])
    mag = math.sqrt(nx**2 + ny**2 + nz**2)
    if mag == 0: return None, None
    return (nx/mag, ny/mag, nz/mag), poly[0]

def geo_point_in_poly(pt, poly, normal):
    """Prüft per Ray-Casting auf der dominanten 2D-Projektion, ob ein Punkt innerhalb eines Polygons liegt."""
    nx, ny, nz = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if nx > ny and nx > nz: ax1, ax2 = 1, 2
    elif ny > nz: ax1, ax2 = 0, 2
    else: ax1, ax2 = 0, 1
    
    u, v = pt[ax1], pt[ax2]
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        ui, vi = poly[i][ax1], poly[i][ax2]
        uj, vj = poly[j][ax1], poly[j][ax2]
        if ((vi > v) != (vj > v)) and (u < (uj - ui) * (v - vi) / (vj - vi + 1e-9) + ui):
            inside = not inside
        j = i
    return inside

def geo_segment_dist(p1, p2, p3, p4):
    """Berechnet den kürzesten Abstand zweier 3D-Linienelemente (Vektor-Mathematik) für kreuzende Stäbe."""
    u = geo_vec_sub(p2, p1)
    v = geo_vec_sub(p4, p3)
    w = geo_vec_sub(p1, p3)
    a, b, c = geo_dot(u, u), geo_dot(u, v), geo_dot(v, v)
    d, e = geo_dot(u, w), geo_dot(v, w)
    D = a*c - b*b
    
    if D < 1e-8:
        sc = 0.0
        tc = (d/b if b>c else e/c) if (b>c or c>0) else 0.0
    else:
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D
        
    if sc < 0.0: 
        sc = 0.0; tc = e / c if c > 0 else 0.0
    elif sc > 1.0: 
        sc = 1.0; tc = (e + b) / c if c > 0 else 0.0
    if tc < 0.0:
        tc = 0.0; sc = -d / a if a > 0 else 0.0
    elif tc > 1.0:
        tc = 1.0; sc = (-d + b) / a if a > 0 else 0.0
        
    sc, tc = max(0.0, min(1.0, sc)), max(0.0, min(1.0, tc))
    pt_s = (p1[0]+u[0]*sc, p1[1]+u[1]*sc, p1[2]+u[2]*sc)
    pt_t = (p3[0]+v[0]*tc, p3[1]+v[1]*tc, p3[2]+v[2]*tc)
    return geo_dist(pt_s, pt_t), sc, tc

def geo_line_plane_intersect(p1, p2, plane_origin, plane_normal):
    """Ermittelt den 3D-Schnittpunkt einer Linie mit einer Fläche/Ebene (Durchstechen)."""
    u = geo_vec_sub(p2, p1)
    dot = geo_dot(plane_normal, u)
    if abs(dot) < 1e-8: return None # parallel, durchstößt nicht
    w = geo_vec_sub(plane_origin, p1)
    fac = geo_dot(plane_normal, w) / dot
    if 0.0 < fac < 1.0: # Punkt liegt streng zwischen den Endpunkten
        return (p1[0]+u[0]*fac, p1[1]+u[1]*fac, p1[2]+u[2]*fac)
    return None


"""LABELING HILFSFUNKTIONEN"""
def load_ground_truth(gt_dir, ifc_stem):
    """Sucht die zugehörige Ground Truth CSV im Unterordner und liest alle Kantenpaare ein."""
    gt_path = os.path.join(gt_dir, f"{ifc_stem} GT_Labels.csv")
    gt_pairs = set()
    
    if not os.path.exists(gt_path):
        print(f"\t\t[WARNUNG] Ground Truth Datei nicht gefunden: {gt_path}")
        return gt_pairs
        
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        # Fallback falls Komma statt Semikolon genutzt wurde
        if not reader.fieldnames or 'GUID_A' not in reader.fieldnames:
            f.seek(0)
            reader = csv.DictReader(f, delimiter=',')
            
        for row in reader:
            guid_a = row.get("GUID_A", "").strip()
            guid_b = row.get("GUID_B", "").strip()
            status = row.get("Status", "").strip().lower()
            
            # Nur laden, wenn die GUIDs valide sind UND der Status "o" ist
            if guid_a and guid_b and guid_a != "---" and guid_b != "---" and status == "o":
                # Tuple alphabetisch sortieren für ungerichteten Abgleich
                pair = tuple(sorted([guid_a, guid_b]))
                gt_pairs.add(pair)
                
    return gt_pairs

def apply_labels_and_add_fn(csv_rows, edges, gt_pairs, graph_nodes):
    """Bewertet bestehende CSV-Zeilen (TP/FP) und hängt fehlende GT-Verbindungen als FN an."""
    found_gt_pairs = set()
    labeled_edges = [] # Speichert Kanten inkl. Label
    
    # 1. Label iterativ vergeben (TP / FP)
    for i, row in enumerate(csv_rows):
        guid_a = row[8] # IFC_GUID_A
        guid_b = row[9] # IFC_GUID_B
        
        if not guid_a or not guid_b:
            row[11] = "FP" # Ohne GUID kann kein Abgleich stattfinden
            label = "FP"
        else:            
            pair = tuple(sorted([str(guid_a), str(guid_b)]))
            
            if pair in gt_pairs:
                row[11] = "TP"
                label = "TP"
                found_gt_pairs.add(pair)
            else:
                row[11] = "FP"
                label = "FP"
        
        # Kante mit Label in neue Liste übernehmen (gleicher Index wie csv_rows)
        if i < len(edges):
            labeled_edges.append((edges[i][0], edges[i][1], label))
            
    # 2. Ermittle False Negatives (FN) - In GT, aber nicht in RFEM gefunden
    missing_gt = gt_pairs - found_gt_pairs
    
    # 3. Füge FNs der CSV (und ggf. den Kanten) hinzu
    if missing_gt:
        print(f"\tFüge {len(missing_gt)} fehlende Verbindungen (FN) am Ende hinzu.")
        
        # Reverse-Lookup Map: GUID -> Graph_Node_ID für Kantendarstellung
        guid_to_node = {}
        for n in graph_nodes:
            guid = n["RFEM"].get("IFC_GUID")
            if guid:
                guid_to_node[guid] = n["Node_ID"]
                
        # Nächste freie ID ermitteln
        max_id = max([int(r[0]) for r in csv_rows if str(r[0]).isdigit()] + [0])
        
        for guid_a, guid_b in missing_gt:
            max_id += 1
            node_a = guid_to_node.get(guid_a)
            node_b = guid_to_node.get(guid_b)
            
            # Typus aus der Node-ID ableiten falls vorhanden
            typ_a, nr_a = (node_a.split('_')[0], node_a.split('_')[1]) if node_a else ("Unbekannt", "")
            typ_b, nr_b = (node_b.split('_')[0], node_b.split('_')[1]) if node_b else ("Unbekannt", "")
            
            csv_rows.append([
                max_id, "Missing (GT)", 
                "", "",
                typ_a, nr_a, 
                typ_b, nr_b,
                guid_a, guid_b,
                "GroundTruth", "FN", "In RFEM-Analyse nicht identifiziert"
            ])
            
            # Nur Kanten hinzufügen, wenn beide Elemente im Graph existieren
            if node_a and node_b:
                labeled_edges.append((node_a, node_b, "FN"))
                
    return csv_rows, labeled_edges

"""HAUPTFUNKTIONEN"""
def analyze_connections(service, node_to_parents, line_to_parents):
    """
    Identifiziert alle Kopplungen (Starre Stäbe, Rigid Links) im Modell.
    Findet heraus, welche übergeordneten Bauteile (Stäbe, Flächen) dadurch verbunden werden und erstellt die Kanten (Edges) für den Graphen.
    Die Kanten werden separat als .csv exportiert.
    """
    print(f"\nAnalysiere Verbindungen (Kopplungen)...")
    edges = []
    csv_rows = []

    # Cache für GUIDs, um bei vielen geometrischen Verbindungen API-Anfragen zu sparen
    guid_cache = {}

    def get_guid(obj_type, no):
        """Hilfskunktion zum Holen der GUID eines Elements."""
        cache_key = f"{obj_type}_{no}"
        if cache_key in guid_cache:
            return guid_cache[cache_key]
        try:
            api_type = 'member' if obj_type == 'Stab' else 'surface'
            if api_type == 'member': obj = service.get_member(no)
            else: obj = service.get_surface(no)
            guid = extract_ifc_guid(str(obj)) or ""
            guid_cache[cache_key] = guid
            return guid
        except: return ""

    # ==========================================
    # TEIL 1: KNOTENKOPPLUNGEN (Stäbe vom Typ Rigid)
    # ==========================================
    print("TEIL 1: KNOTENKOPPLUNGEN (Starre Stäbe)")
    print("-" * 60)
    try:
        member_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_MEMBER) #type: ignore
        if not isinstance(member_numbers, (list, tuple)): member_numbers = parse_id_list(str(member_numbers))

        for no in member_numbers:
            member = service.get_member(no)
            if str(member.type) == 'TYPE_RIGID':
                start_node = parse_int_safely(getattr(member, 'node_start', '0'))
                end_node = parse_int_safely(getattr(member, 'node_end', '0'))

                parents_a = node_to_parents.get(start_node, [])
                parents_b = node_to_parents.get(end_node, [])

                # Alle möglichen Kombinationen prüfen
                for pa_type, pa_no in parents_a:
                    for pb_type, pb_no in parents_b:
                        # Selbstbezüge ausschließen
                        if pa_type == pb_type and pa_no == pb_no: continue

                        node_a_id = f"{pa_type}_{pa_no}"
                        node_b_id = f"{pb_type}_{pb_no}"
                        
                        edges.append((node_a_id, node_b_id))
                        csv_rows.append([
                            member.no, "Knotenkopplung",
                            start_node, end_node,
                            pa_type, pa_no, 
                            pb_type, pb_no,
                            get_guid(pa_type, pa_no), 
                            get_guid(pb_type, pb_no),
                            "Rule", "", ""
                            ])
    except Exception as e:
        print(f"Keine Stäbe im Modell gefunden.")
        print(f"Fehler bei Knotenkopplungen: {e}")
    
    # ==========================================
    # TEIL 2: LINIENKOPPLUNGEN (Rigid Link)
    # ==========================================
    print("TEIL 2: STARRE KOPPLUNGEN (Spezielle Objekte --> Rigid Links)")
    print("-" * 60)
    try:
        rigid_link_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_RIGID_LINK) #type: ignore
        if not isinstance(rigid_link_numbers, (list, tuple)): rigid_link_numbers = parse_id_list(str(rigid_link_numbers))

        for no in rigid_link_numbers:
            link = service.get_rigid_link(no)
            line1 = parse_int_safely(getattr(link, 'line1', '0'))
            line2 = parse_int_safely(getattr(link, 'line2', '0'))
            surface_id = parse_int_safely(getattr(link, 'surface', '0'))
            
            parents_a = line_to_parents.get(line1, [])
            parents_b = []
            if line2 > 0:
                parents_b = line_to_parents.get(line2, [])
            elif surface_id > 0:
                parents_b = [('Fläche', surface_id)]
                
            for pa_type, pa_no in parents_a:
                for pb_type, pb_no in parents_b:
                    if pa_type == pb_type and pa_no == pb_no: continue
                    
                    node_a_id = f"{pa_type}_{pa_no}"
                    node_b_id = f"{pb_type}_{pb_no}"
                    
                    edges.append((node_a_id, node_b_id))
                    csv_rows.append([
                        link.no, "Linienkopplung",
                        line1, line2 if line2 > 0 else surface_id,
                        pa_type, pa_no, 
                        pb_type, pb_no,
                        get_guid(pa_type, pa_no), 
                        get_guid(pb_type, pb_no),
                        "Rule", "", ""
                        ])
    except Exception as e:
        print(f"Keine Flächen im Modell gefunden.")
        print(f"Fehler bei Linienkopplungen: {e}")
    
    # ==========================================
    # TEIL 3: GEOMETRIEKOPPLUNGEN (Gemeinsame Knoten / Linien)
    # ==========================================
    print("TEIL 3: GEOMETRISCHE KOPPLUNGEN (Rechnerisch)")
    print("-" * 60)

    geom_conn_id = 1
    TOL = 0.001 # Toleranz von 1mm für Floating-Point Vergleiche
    
    # --- Daten-Cache für rechnerische Geometrie ---
    coords_cache = {}
    member_cache = {} 
    surface_cache = {}
    
    def get_coords(node_id):
        if node_id in coords_cache: return coords_cache[node_id]
        try:
            nd = service.get_node(node_id)
            c = (getattr(nd, 'coordinate_1', 0.0), getattr(nd, 'coordinate_2', 0.0), getattr(nd, 'coordinate_3', 0.0))
            coords_cache[node_id] = c
            return c
        except: return None

    # Stäbe und deren Endkoordinaten abrufen
    try:
        member_nums = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_MEMBER) #type: ignore
        member_nums = parse_id_list(str(member_nums)) if not isinstance(member_nums, (list, tuple)) else member_nums
        for member_no in member_nums:
            mem = service.get_member(member_no)
            if str(getattr(mem, 'type', '')) == 'TYPE_RIGID': continue
            ns = parse_int_safely(getattr(mem, 'node_start', 0)) #type: ignore
            ne = parse_int_safely(getattr(mem, 'node_end', 0)) #type: ignore
            if ns > 0 and ne > 0:
                cs, ce = get_coords(ns), get_coords(ne)
                if cs and ce:
                    member_cache[member_no] = {'nodes': [ns, ne], 'coords': [cs, ce]}
    except: pass
    
    # Ebene Flächen und deren Randkoordinaten abrufen
    try:
        s_nums = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_SURFACE) #type: ignore
        s_nums = parse_id_list(str(s_nums)) if not isinstance(s_nums, (list, tuple)) else s_nums
        for s_no in s_nums:
            surf = service.get_surface(s_no)
            if 'GEOMETRY_PLANE' not in str(getattr(surf, 'geometry', '')): continue
            
            b_lines = parse_id_list(getattr(surf, 'boundary_lines', ''))
            b_nodes = []
            for l_no in b_lines:
                try:
                    line = service.get_line(l_no)
                    for n in parse_id_list(getattr(line, 'definition_nodes', '')):
                        if n not in b_nodes: b_nodes.append(n)
                except: pass
                
            b_coords = [c for c in [get_coords(n) for n in b_nodes] if c]
            if len(b_coords) >= 3:
                normal, origin = geo_plane_from_poly(b_coords)
                if normal:
                    surface_cache[s_no] = {'lines': b_lines, 'nodes': b_nodes, 'coords': b_coords, 'normal': normal, 'origin': origin}
    except: pass

    # --- Rechnerischer Abgleich ---
    # 3.1 Stäbe vs Stäbe (Duplikate & Kreuzungen)
    m_ids = list(member_cache.keys())
    for i in range(len(m_ids)):
        for j in range(i+1, len(m_ids)):
            ma_no, mb_no = m_ids[i], m_ids[j]
            ma, mb = member_cache[ma_no], member_cache[mb_no]
            
            connected = False
            # Knoten-Duplikat (Punkte liegen übereinander)
            for idx_a, na in enumerate(ma['nodes']):
                for idx_b, nb in enumerate(mb['nodes']):
                    if geo_dist(ma['coords'][idx_a], mb['coords'][idx_b]) < TOL:
                        edges.append((f"Stab_{ma_no}", f"Stab_{mb_no}"))
                        csv_rows.append([
                            geom_conn_id, "Geometriekopplung", 
                            na, nb, 
                            "Stab", ma_no, 
                            "Stab", mb_no, 
                            get_guid("Stab", ma_no), get_guid("Stab", mb_no), 
                            "Topology", "", "Knoten-Duplikat (gleiche Koordinate)"
                            ])
                        geom_conn_id += 1
                        connected = True
            
            if connected: continue
            
            # Linien-Kreuzung (Kreuzung in der Mitte zweier Stäbe)
            dist_val, sc, tc = geo_segment_dist(ma['coords'][0], ma['coords'][1], mb['coords'][0], mb['coords'][1])
            if dist_val < TOL and 0.01 < sc < 0.99 and 0.01 < tc < 0.99:
                edges.append((f"Stab_{ma_no}", f"Stab_{mb_no}"))
                csv_rows.append([
                    geom_conn_id, "Geometriekopplung", 
                    ma['nodes'][0], mb['nodes'][0], 
                    "Stab", ma_no, 
                    "Stab", mb_no, 
                    get_guid("Stab", ma_no), get_guid("Stab", mb_no), 
                    "Topology", "", "Linien-Kreuzung (Schnittpunkt)"
                    ])
                geom_conn_id += 1

    # 3.2 Stäbe vs Flächen (Knoten auf Ebene & Durchstechen)
    for ma_no, ma in member_cache.items():
        for sb_no, sb in surface_cache.items():
            connected = False
            # Knoten auf Flächenebene
            for idx_a, na in enumerate(ma['nodes']):
                pt = ma['coords'][idx_a]
                if abs(geo_dot(geo_vec_sub(pt, sb['origin']), sb['normal'])) < TOL:
                    if geo_point_in_poly(pt, sb['coords'], sb['normal']):
                        edges.append((f"Stab_{ma_no}", f"Fläche_{sb_no}"))
                        csv_rows.append([
                            geom_conn_id, "Geometriekopplung", 
                            na, sb['lines'][0] if sb['lines'] else 0, 
                            "Stab", ma_no, "Fläche", sb_no, 
                            get_guid("Stab", ma_no), get_guid("Fläche", sb_no), 
                            "Topology", "", "Knoten auf Flächenebene"
                            ])
                        geom_conn_id += 1
                        connected = True
                        break
            if connected: continue
            
            # Stab durchstößt Fläche (Linien-Kreuzung / Durchdringung)
            intersect = geo_line_plane_intersect(ma['coords'][0], ma['coords'][1], sb['origin'], sb['normal'])
            if intersect and geo_point_in_poly(intersect, sb['coords'], sb['normal']):
                edges.append((f"Stab_{ma_no}", f"Fläche_{sb_no}"))
                csv_rows.append([
                    geom_conn_id, "Geometriekopplung", 
                    ma['nodes'][0], sb['lines'][0] if sb['lines'] else 0, 
                    "Stab", ma_no, 
                    "Fläche", sb_no, 
                    get_guid("Stab", ma_no), get_guid("Fläche", sb_no), 
                    "Topology", "", "Linien-Kreuzung (Schnittpunkt)"
                    ])
                geom_conn_id += 1

    # 3.3 Flächen vs Flächen (Rand auf Ebene & Kreuzung)
    s_ids = list(surface_cache.keys())
    for i in range(len(s_ids)):
        for j in range(i+1, len(s_ids)):
            sa_no, sb_no = s_ids[i], s_ids[j]
            sa, sb = surface_cache[sa_no], surface_cache[sb_no]
            
            connected = False
            for k in range(len(sa['coords'])):
                p1, p2 = sa['coords'][k], sa['coords'][(k+1)%len(sa['coords'])]
                
                # Randlinie auf Flächenebene
                if abs(geo_dot(geo_vec_sub(p1, sb['origin']), sb['normal'])) < TOL and abs(geo_dot(geo_vec_sub(p2, sb['origin']), sb['normal'])) < TOL:
                    mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
                    if geo_point_in_poly(mid, sb['coords'], sb['normal']):
                        edges.append((f"Fläche_{sa_no}", f"Fläche_{sb_no}"))
                        csv_rows.append([
                            geom_conn_id, "Geometriekopplung", 
                            sa['lines'][k] if len(sa['lines'])>k else 0, sb['lines'][0] if sb['lines'] else 0, 
                            "Fläche", sa_no, "Fläche", sb_no, 
                            get_guid("Fläche", sa_no), get_guid("Fläche", sb_no), 
                            "Topology", "", "Randlinie auf Flächenebene"
                            ])
                        geom_conn_id += 1
                        connected = True
                        break
                
                # Flächen-Kreuzung (Randlinie durchstößt andere Fläche)
                if not connected:
                    intersect = geo_line_plane_intersect(p1, p2, sb['origin'], sb['normal'])
                    if intersect and geo_point_in_poly(intersect, sb['coords'], sb['normal']):
                        edges.append((f"Fläche_{sa_no}", f"Fläche_{sb_no}"))
                        csv_rows.append([
                            geom_conn_id, "Geometriekopplung", 
                            sa['lines'][k] if len(sa['lines'])>k else 0, sb['lines'][0] if sb['lines'] else 0, 
                            "Fläche", sa_no, 
                            "Fläche", sb_no, 
                            get_guid("Fläche", sa_no), get_guid("Fläche", sb_no), 
                            "Topology", "", "Flächen-Kreuzung (Durchdringung)"
                            ])
                        geom_conn_id += 1
                        connected = True
                        break
            
            if connected: continue

            # Gegenprobe (Fläche B vs Fläche A)
            for k in range(len(sb['coords'])):
                p1, p2 = sb['coords'][k], sb['coords'][(k+1)%len(sb['coords'])]
                
                if abs(geo_dot(geo_vec_sub(p1, sa['origin']), sa['normal'])) < TOL and abs(geo_dot(geo_vec_sub(p2, sa['origin']), sa['normal'])) < TOL:
                    mid = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2)
                    if geo_point_in_poly(mid, sa['coords'], sa['normal']):
                        edges.append((f"Fläche_{sb_no}", f"Fläche_{sa_no}"))
                        csv_rows.append([
                            geom_conn_id, "Geometriekopplung", 
                            sb['lines'][k] if len(sb['lines'])>k else 0, sa['lines'][0] if sa['lines'] else 0, 
                            "Fläche", sb_no, 
                            "Fläche", sa_no, 
                            get_guid("Fläche", sb_no), get_guid("Fläche", sa_no), 
                            "Topology", "", "Randlinie auf Flächenebene"
                            ])
                        geom_conn_id += 1
                        connected = True
                        break
                
                if not connected:
                    intersect = geo_line_plane_intersect(p1, p2, sa['origin'], sa['normal'])
                    if intersect and geo_point_in_poly(intersect, sa['coords'], sa['normal']):
                        edges.append((f"Fläche_{sb_no}", f"Fläche_{sa_no}"))
                        csv_rows.append([geom_conn_id, "Geometriekopplung", sb['lines'][k] if len(sb['lines'])>k else 0, sa['lines'][0] if sa['lines'] else 0, "Fläche", sb_no, "Fläche", sa_no, get_guid("Fläche", sb_no), get_guid("Fläche", sa_no), "Topology", "", "Flächen-Kreuzung (Durchdringung)"])
                        geom_conn_id += 1
                        connected = True
                        break
    
    return edges, csv_rows

def extract_graph_elements(service):
    """
    Sammelt alle tragenden Stäbe und Flächen aus dem RFEM-Modell.
    Bereitet diese als Knoten (Nodes) für den JSON-Graph auf. 
    """
    print(f"\nExtrahiere Bauteile für den Graphen...")
    graph_nodes = []
    graph_id = 1

    # STÄBE
    try:
        member_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_MEMBER) #type: ignore
        if not isinstance(member_numbers, (list, tuple)): member_numbers = parse_id_list(str(member_numbers))
        
        for no in member_numbers:
            member = service.get_member(no)
            member_type = str(getattr(member, 'type', 'Unbekannt'))
            if member_type == 'TYPE_RIGID': continue # Überspringen, da dies Kanten sind

            line_no = getattr(member, 'line', 0)
            line = service.get_line(line_no)
            line_type = str(getattr(line, 'type', 'Unbekannt'))

            is_curved = 'TYPE_LINE' not in line_type and 'TYPE_POLYLINE' not in line_type
            
            rfem_data = {
                "Nr": member.no,
                "Kategorie": "Stab",
                "Type": member_type,
                "IFC_GUID": extract_ifc_guid(str(member)),
                "analytische_Länge": getattr(member, 'length', None),
                "is_curved": is_curved,
                "is_rotated": getattr(member, 'rotation_angle', 0.0) != 0.0,
                "zugehoerige_Linien": parse_id_list(getattr(member, 'line', '')),
                "zugehoerige_Knoten": parse_id_list(getattr(member, 'nodes', ''))
            }
            graph_nodes.append({"Graph_ID": graph_id, "RFEM": rfem_data, "Node_ID": f"Stab_{member.no}"})
            graph_id += 1
    except Exception as e: print(f"Fehler bei Stäben: {e}")

    # FLÄCHEN
    try:
        surface_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_SURFACE) #type: ignore
        if not isinstance(surface_numbers, (list, tuple)): surface_numbers = parse_id_list(str(surface_numbers))
        
        for no in surface_numbers:
            surface = service.get_surface(no)
            geom_type = str(getattr(surface, 'geometry', ''))
            
            rfem_data = {
                "Nr": surface.no,
                "Kategorie": "Fläche",
                "Type": f"{geom_type} / {getattr(surface, 'type', 'Unbekannt')}".strip(" /"),
                "IFC_GUID": extract_ifc_guid(str(surface)),
                "analytische_Länge": getattr(surface, 'area', None), # Analytische Länge = Fläche für 2D
                "is_curved": 'GEOMETRY_PLANE' not in geom_type if geom_type else False,
                "is_rotated": getattr(surface, 'axes_rotation_angle', 0.0) != 0.0,
                "zugehoerige_Linien": parse_id_list(getattr(surface, 'boundary_lines', '')),
                "zugehoerige_Knoten": parse_id_list(getattr(surface, 'nodes', ''))
            }
            graph_nodes.append({"Graph_ID": graph_id, "RFEM": rfem_data, "Node_ID": f"Fläche_{surface.no}"})
            graph_id += 1
    except Exception as e: print(f"Fehler bei Flächen: {e}")

    return graph_nodes

def plot_rfem_graph(graph_nodes, edges, output_dir):
    """
    Erstellt eine 2D-Visualisierung des ermittelten Tragwerks-Graphen.
    Stäbe und Flächen werden unterschiedlich dargestellt.
    Speichert das Ergebnis als PNG und SVG.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')

    G = nx.Graph()
    
    # Knoten zum NetworkX Graph hinzufügen
    for node in graph_nodes:
        n_id = node["Node_ID"]
        kat = node["RFEM"]["Kategorie"]
        G.add_node(n_id, kategorie=kat)
        
    # Kanten inkl. Label hinzufügen
    valid_nodes = set(G.nodes())
    for u, v, label in edges:
        # Nur zeichnen, wenn beide Knoten existieren
        if u in valid_nodes and v in valid_nodes:
            G.add_edge(u, v, label=label)

    # Automatisches Layout (Spring Layout) generieren, da wir keine direkten 3D Koordinaten spezifiziert haben.
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Gefilterte Kantenlisten für die farbige Darstellung aufbauen
    tp_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == 'TP']
    fp_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == 'FP']
    fn_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('label') == 'FN']

    # Kanten getrennt voneinander und farblich zeichnen
    # nx.draw_networkx_edges(G, pos, ax=ax, edge_color="darkblue", width=1.0, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=tp_edges, ax=ax, edge_color="green", width=2.0, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=fp_edges, ax=ax, edge_color="red", width=1.5, alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=fn_edges, ax=ax, edge_color="orange", width=2.0, style="dashed", alpha=0.9)

    # Knoten zeichnen gruppenweise
    for kategorie in ["Stab", "Fläche"]:
        nodelist = [n for n, attr in G.nodes(data=True) if attr.get('kategorie') == kategorie]
        if not nodelist: continue
        
        color, marker = VIS_MAPPING.get(kategorie, VIS_MAPPING["Unbekannt"])
        
        # Manuelles Plotten mittels Scatter für konsistente Legendenerstellung
        x = [pos[n][0] for n in nodelist]
        y = [pos[n][1] for n in nodelist]
        
        ax.scatter(x, y, c=color, marker=marker, s=200, label=kategorie,
                   edgecolors='white', linewidths=0.5, zorder=5)

    # IDs plotten
    for node_id, coords in pos.items():
        # Extrahieren der Nummer aus dem String (z.B. "Stab_5" -> "5")
        nr = node_id.split("_")[1] if "_" in node_id else node_id
        ax.text(coords[0], coords[1] + 0.04, str(nr),
                fontsize=8, color='black', fontweight='bold', zorder=10, 
                horizontalalignment='center')

    ax.set_title("RFEM Analysemodell: Bauteile & Kopplungen (Topologie)")

    
    # Eigene Kanten-Elemente für die Legende erstellen
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    edge_legends = [
        mlines.Line2D([], [], color='green', linewidth=2.0, label='TP (Korrekt gefunden)'),
        mlines.Line2D([], [], color='red', linewidth=1.5, label='FP (Fälschlich gefunden)'),
        mlines.Line2D([], [], color='orange', linewidth=2.0, linestyle='--', label='FN (Nicht gefunden)')
    ]

    # Vorhandene Knoten-Legende auslesen und neue Kanten-Legenden anhängen
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(edge_legends)
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    ax.set_axis_off()
    fig.subplots_adjust(right=0.85)
    
    # Speichern
    plt.savefig(os.path.join(output_dir, f"{GRAPH_BASENAME}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{GRAPH_BASENAME}.svg"), format='svg', bbox_inches='tight')
    plt.close(fig)
    print("\tPlots erfolgreich als PNG und SVG gespeichert.")
  


if __name__ == "__main__":
    main()
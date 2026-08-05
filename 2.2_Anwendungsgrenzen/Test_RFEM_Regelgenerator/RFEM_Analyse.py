from RFEM.initModel import Model
from RFEM.enums import ObjectTypes
from RFEM.Tools.GetObjectNumbersByType import GetObjectNumbersByType
import re
import uuid
import json
import os
import csv

# Hilfsvariable für den Debug-Output
debug_bim_printed = False

def uuid_to_ifc_guid(uuid_string):
    """
    Konvertiert die 36-stellige RFEM-UUID in die 22-stellige IFC-GUID.
    Dies ist der offizielle IFC-Standard-Algorithmus (Base64 Encoding).
    """
    try:
        b = uuid.UUID(str(uuid_string)).bytes
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        
        def encode(num, length):
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
        return f"Fehler bei Umrechnung: {e}"

def extract_ifc_guid(obj_str):
    """Extrahiert die GUID als reinen String aus den Objektdaten."""
    match = re.search(r'id_for_export_import\s*=\s*["\']?([a-fA-F0-9\-]{36})["\']?', obj_str)
    if match:
        return uuid_to_ifc_guid(match.group(1))
    match_global = re.search(r'GlobalId.*?([a-zA-Z0-9_\$]{22})', obj_str, re.IGNORECASE)
    if match_global:
        return match_global.group(1)
    return None

def parse_id_list(id_string):
    """Wandelt einen String aus IDs (z.B. '1 2 3') in eine Integer-Liste um."""
    if not id_string or str(id_string).strip() in ['Keine', 'None', '']:
        return []
    # Verhindern, dass aus Python-Objekt-Strings falsche Zahlen extrahiert werden
    if '<' in str(id_string) and '>' in str(id_string):
        return []
    return [int(n) for n in re.findall(r'\d+', str(id_string))]

def get_bim_info(service, obj_type, no):
    """Liest die RFEM-ID (UUID) aus den Objektdaten und konvertiert sie."""
    global debug_bim_printed
    
    if no is None or str(no) == 'Keine' or str(no) == 'Unbekannt' or str(no).strip() == '':
        return "Fehlende ID"
        
    try:
        num_match = re.search(r'\d+', str(no))
        if not num_match:
            return f"Ungültiges Format ({no})"
        clean_no = int(num_match.group())

        if obj_type == 'line':
            obj = service.get_line(clean_no)
        elif obj_type == 'surface':
            obj = service.get_surface(clean_no)
        elif obj_type == 'member':
            obj = service.get_member(clean_no)
        else:
            return "Unbekannter Typ"
        
        # Das gesamte Objekt in Text umwandeln (Das ist dein alter funktionierender Ansatz!)
        obj_str = str(obj)
        
        # Wir suchen explizit nach der Zeile: id_for_export_import = "..."
        match = re.search(r'id_for_export_import\s*=\s*["\']?([a-fA-F0-9\-]{36})["\']?', obj_str)
        
        if match:
            rfem_uuid = match.group(1)
            # ====================================================================
            # HIER WIRD UMGERECHNET: Die gefundene RFEM-ID geht in die Funktion!
            # ====================================================================
            ifc_guid = uuid_to_ifc_guid(rfem_uuid)
            info = f"RFEM-ID: {rfem_uuid} -> IFC-GUID: {ifc_guid}"
        else:
            # Fallback: Vielleicht heißt es GlobalId in den Parametern
            match_global = re.search(r'GlobalId.*?([a-zA-Z0-9_\$]{22})', obj_str, re.IGNORECASE)
            if match_global:
                info = f"Direkte IFC-GUID gefunden: {match_global.group(1)}"
            else:
                info = "Keine UUID in den Objektdaten gefunden"
            
        # Den Debug-Ausdruck für das ERSTE Objekt zwingend beibehalten
        if not debug_bim_printed:
            print(f"\n  --- DEBUG-INFO: ROHDATEN DES GEKOPPELTEN OBJEKTS ({obj_type.upper()} {clean_no}) ---")
            print(f"  {obj_str}")
            print("  ----------------------------------------------------------------------\n")
            debug_bim_printed = True
        
        return info
    except Exception as e:
        return f"Fehler beim Abruf: {e}"

def parse_int_safely(s):
    """Extrahiert die erste gefundene Ganzzahl aus einem String, gibt 0 zurück, wenn nichts gefunden wird."""
    if '<' in str(s) and '>' in str(s):
        return 0
    match = re.search(r'\d+', str(s))
    return int(match.group()) if match else 0

def build_parent_maps(service):
    """
    Erstellt zwei Dictionaries, die Knoten und Linien ihren übergeordneten
    Elementen (Stäbe, Flächen) zuordnen, um die Suche zu beschleunigen.
    Rückgabe: (node_to_parents, line_to_parents)
    """
    node_to_parents = {}
    line_to_parents = {}

    def add_to_map(mapping, child_no, parent):
        if child_no <= 0: return
        if child_no not in mapping:
            mapping[child_no] = []
        if parent not in mapping[child_no]:
            mapping[child_no].append(parent)

    # Stäbe durchlaufen
    try:
        member_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_MEMBER)
        if not isinstance(member_numbers, (list, tuple)):
            member_numbers = parse_id_list(str(member_numbers))
            
        for no in member_numbers:
            member = service.get_member(no)
            
            # WICHTIG: Starre Stäbe (Knotenkopplungen) ausgrenzen!
            if str(getattr(member, 'type', '')) == 'TYPE_RIGID':
                continue
                
            parent = ('Stab', member.no)
            
            for attr in ['node_start', 'node_end', 'nodes', 'internal_nodes', 'nodes_on_member']:
                for n_no in parse_id_list(getattr(member, attr, '')):
                    add_to_map(node_to_parents, n_no, parent)
                
            for attr in ['line']:
                for l_no in parse_id_list(getattr(member, attr, '')):
                    add_to_map(line_to_parents, l_no, parent)
                    try:
                        line_obj = service.get_line(l_no)
                        for l_attr in ['definition_nodes', 'nodes_on_line', 'connected_nodes']:
                            for n_no in parse_id_list(getattr(line_obj, l_attr, '')):
                                add_to_map(node_to_parents, n_no, parent)
                    except Exception:
                        pass
                
    except Exception as e:
        print(f"Warnung beim Erstellen der Stab-Parent-Map: {e}")

    # Flächen durchlaufen
    try:
        surface_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_SURFACE)
        if not isinstance(surface_numbers, (list, tuple)):
            surface_numbers = parse_id_list(str(surface_numbers))
            
        for no in surface_numbers:
            surface = service.get_surface(no)
            parent = ('Flaeche', surface.no)
            
            for attr in ['boundary_lines', 'integrated_lines', 'internal_lines']:
                for l_no in parse_id_list(getattr(surface, attr, '')):
                    add_to_map(line_to_parents, l_no, parent)
                    try:
                        line_obj = service.get_line(l_no)
                        for l_attr in ['definition_nodes', 'nodes_on_line', 'connected_nodes']:
                            for n_no in parse_id_list(getattr(line_obj, l_attr, '')):
                                add_to_map(node_to_parents, n_no, parent)
                    except Exception:
                        pass
            
                
            for attr in ['nodes', 'integrated_nodes', 'internal_nodes']:
                for n_no in parse_id_list(getattr(surface, attr, '')):
                    add_to_map(node_to_parents, n_no, parent)
    except Exception as e:
        print(f"Warnung beim Erstellen der Flächen-Parent-Map: {e}")
        
    # Rückwärtssuche über alle Knoten (für generierte/interne Knoten)
    try:
        node_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_NODE)
        if not isinstance(node_numbers, (list, tuple)):
            node_numbers = parse_id_list(str(node_numbers))
            
        for no in node_numbers:
            try:
                node = service.get_node(no)
                l_no = parse_int_safely(getattr(node, 'on_line_reference_line', getattr(node, 'line', 0)))
                if l_no > 0:
                    for p in line_to_parents.get(l_no, []):
                        add_to_map(node_to_parents, no, p)
                        
                m_no = parse_int_safely(getattr(node, 'on_member_reference_member', getattr(node, 'member', 0)))
                if m_no > 0:
                    add_to_map(node_to_parents, no, ('Stab', m_no))
            except Exception:
                pass
    except Exception:
        pass
        
    return node_to_parents, line_to_parents

def analyze_couplings(n=10, csv_filename="rfem_couplings.csv"):
    print("Verbinde mit RFEM 6 über SOAP...")
    try:
        Model(new_model=False, model_name="")
    except Exception:
        pass
        
    if not hasattr(Model, 'clientModel') or Model.clientModel is None:
        print("Verbindungsfehler: Das aktive RFEM-Modell konnte nicht initialisiert werden.")
        return

    print("Erfolgreich verbunden!")
    print("=" * 60)
    
    # Parent-Maps erstellen, um die Suche nach Bauteilen zu beschleunigen
    print("Erstelle Zuordnung von Knoten/Linien zu übergeordneten Bauteilen...")
    node_to_parents, line_to_parents = build_parent_maps(Model.clientModel.service)
    print("Zuordnung abgeschlossen.")
    
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    csv_data = []

    def get_guid(obj_type, no):
        if no is None or str(no).strip() in ['Keine', 'Unbekannt', 'None', '', '0', '[]']:
            return ""
        try:
            no_int = int(re.search(r'\d+', str(no)).group())
            service = Model.clientModel.service
            
            type_map = {'stab': 'member', 'flaeche': 'surface', 'knoten': 'node'}
            api_type = type_map.get(obj_type.lower(), obj_type.lower())

            if api_type == 'node': obj = service.get_node(no_int)
            elif api_type == 'line': obj = service.get_line(no_int)
            elif api_type == 'surface': obj = service.get_surface(no_int)
            elif api_type == 'member': obj = service.get_member(no_int)
            else: return ""
            return extract_ifc_guid(str(obj)) or ""
        except Exception:
            return ""

    # ==========================================
    # TEIL 1: KNOTENKOPPLUNGEN (Starre Stäbe)
    # ==========================================
    print("TEIL 1: KNOTENKOPPLUNGEN (Starre Stäbe)")
    print("-" * 60)
    try:
        member_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_MEMBER)
    except Exception:
        member_numbers = []

    if member_numbers:
        rigid_members = []
        for no in member_numbers:
            member = Model.clientModel.service.get_member(no)
            if str(member.type) == 'TYPE_RIGID':
                rigid_members.append(member)
        
        total_rigid = len(rigid_members)
        print(f"{total_rigid} Starre Stäbe gefunden. (Ausgabe: 1., jedes n. und letztes)")
        
        for index, member in enumerate(rigid_members):
            start_node = parse_int_safely(getattr(member, 'node_start_node', getattr(member, 'node_start', '0')))
            end_node = parse_int_safely(getattr(member, 'node_end_node', getattr(member, 'node_end', '0')))

            parents_a = node_to_parents.get(start_node, [])
            parents_b = node_to_parents.get(end_node, [])

            has_self_connection = False
            valid_pairs = []
            for pa_type, pa_no in parents_a:
                for pb_type, pb_no in parents_b:
                    if pa_type == pb_type and pa_no == pb_no: 
                        has_self_connection = True
                        continue
                    pair = ("Knoten", start_node, "Knoten", end_node, pa_type, pa_no, pb_type, pb_no)
                    if pair not in valid_pairs:
                        valid_pairs.append(pair)
            
            is_first = (index == 0)
            is_nth = ((index + 1) % n == 0)
            is_last = (index == total_rigid - 1)
            
            if not valid_pairs:
                if is_first or is_nth or is_last:
                    print(f"[Stab-Kopplung] Nr. {member.no} (Index {index + 1}/{total_rigid}): Verbindet Knoten {start_node} mit Knoten {end_node}")
                    if has_self_connection:
                        print(f"  -> [WARNUNG] Konnte nur Selbstverbindungen finden (ignoriert).")
                    else:
                        print(f"  -> [WARNUNG] Konnte keine Bauteile zuordnen. (Interne/lose Knoten?)")
                    print("-" * 40)
                continue

            for b1_t, b1_n, b2_t, b2_n, pa_type, pa_no, pb_type, pb_no in valid_pairs:
                guid_a = get_guid(pa_type, pa_no)
                guid_b = get_guid(pb_type, pb_no)
                csv_data.append([
                    member.no, "Knotenkopplung",
                    b1_t, b1_n, b2_t, b2_n,
                    pa_type, pa_no, pb_type, pb_no,
                    guid_a, guid_b, "Rule", ""
                ])
            
            if is_first or is_nth or is_last:
                print(f"[Stab-Kopplung] Nr. {member.no} (Index {index + 1}/{total_rigid}): Verbindet Knoten {start_node} mit Knoten {end_node}")
                if has_self_connection:
                    print(f"  -> [WARNUNG] Selbstverbindung wurde herausgefiltert.")
                for b1_t, b1_n, b2_t, b2_n, pa_type, pa_no, pb_type, pb_no in valid_pairs:
                    guid_a = get_guid(pa_type, pa_no)
                    guid_b = get_guid(pb_type, pb_no)
                    print(f"  -> Bauteil A ({pa_type} {pa_no} via {b1_t} {b1_n}) <-> Bauteil B ({pb_type} {pb_no} via {b2_t} {b2_n})")
                    print(f"     GUID A: {guid_a} | GUID B: {guid_b}")
                print("-" * 40)
    else:
        print("Keine Stäbe im Modell gefunden.")

    print("\n" + "=" * 60)

    # ==========================================
    # TEIL 2: LINIENKOPPLUNGEN (Rigid Links)
    # ==========================================
    print("TEIL 2: STARRE KOPPLUNGEN (Spezielle Objekte -> Rigid Links)")
    print("-" * 60)
    
    try:
        rigid_link_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_RIGID_LINK)
    except Exception as e:
        print(f"Fehler bei der Abfrage: {e}")
        rigid_link_numbers = []

    if rigid_link_numbers:
        total = len(rigid_link_numbers)
        print(f"{total} Starre Kopplung(en) gefunden. (Ausgabe: 1., jedes n. und letztes)")
        print("-" * 60)
        
        for index, no in enumerate(rigid_link_numbers):
            link = Model.clientModel.service.get_rigid_link(no)
            
            line1_str, line2_str, surface_id_str = getattr(link, 'line1', '0'), getattr(link, 'line2', '0'), getattr(link, 'surface', '0')
            line1 = parse_int_safely(line1_str)

            parents_a = line_to_parents.get(line1, [])
            parents_b_raw = []

            surface_id = parse_int_safely(surface_id_str)
            line2 = parse_int_safely(line2_str)
            
            # Priorität: Wenn eine explizite Linie (line2) da ist, nutze diese.
            if line2 > 0:
                for pb_t, pb_n in line_to_parents.get(line2, []):
                    parents_b_raw.append(("Linie", line2, pb_t, pb_n))
            # Fallback: Wenn keine Linie da ist, aber eine Fläche, nutze die Fläche direkt.
            elif surface_id > 0:
                parents_b_raw.append(('Flaeche', surface_id, 'Flaeche', surface_id))

            has_self_connection = False
            valid_pairs = []
            for pa_type, pa_no in parents_a:
                for b2_t, b2_n, pb_type, pb_no in parents_b_raw:
                    if pa_type == pb_type and pa_no == pb_no: 
                        has_self_connection = True
                        continue
                    pair = ("Linie", line1, b2_t, b2_n, pa_type, pa_no, pb_type, pb_no)
                    if pair not in valid_pairs:
                        valid_pairs.append(pair)

            is_first = (index == 0)
            is_nth = ((index + 1) % n == 0)
            is_last = (index == total - 1)
            
            if not valid_pairs:
                if is_first or is_nth or is_last:
                    print(f"[Starre Kopplung] Nr. {link.no} (Index {index + 1}/{total}):")
                    print(f"  (Basis: Bezugslinie: {line1_str}, Flächenlinie: {line2_str}, Fläche: {surface_id_str})")
                    if has_self_connection:
                        print(f"  -> [WARNUNG] Konnte nur Selbstverbindungen finden (ignoriert).")
                    else:
                        print(f"  -> [WARNUNG] Konnte keine Bauteile zuordnen. (Interne/lose Linien?)")
                    print("-" * 40)
                continue

            for b1_t, b1_n, b2_t, b2_n, pa_type, pa_no, pb_type, pb_no in valid_pairs:
                guid_a = get_guid(pa_type, pa_no)
                guid_b = get_guid(pb_type, pb_no)
                csv_data.append([
                    link.no, "Linienkopplung",
                    b1_t, b1_n, b2_t, b2_n,
                    pa_type, pa_no, pb_type, pb_no,
                    guid_a, guid_b, "Rule", ""
                ])
            
            if is_first or is_nth or is_last:
                print(f"[Starre Kopplung] Nr. {link.no} (Index {index + 1}/{total}):")
                print(f"  (Basis: Bezugslinie: {line1_str}, Flächenlinie: {line2_str}, Fläche: {surface_id_str})")
                if has_self_connection:
                    print(f"  -> [WARNUNG] Selbstverbindung wurde herausgefiltert.")
                for b1_t, b1_n, b2_t, b2_n, pa_type, pa_no, pb_type, pb_no in valid_pairs:
                    guid_a = get_guid(pa_type, pa_no)
                    guid_b = get_guid(pb_type, pb_no)
                    print(f"  -> Bauteil A ({pa_type} {pa_no} via {b1_t} {b1_n}) <-> Bauteil B ({pb_type} {pb_no} via {b2_t} {b2_n})")
                    print(f"     GUID A: {guid_a} | GUID B: {guid_b}")
                print(f"  -> Typ: {getattr(link, 'type', 'Unbekannt')} | Typus: Rule | Label: ")
                print("-" * 40)
    else:
        print("Keine 'Starren Kopplungen' gefunden.")
        
    # --- CSV SPEICHERN ---
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Verbindungs_Nr', 'Verbindungs_Art', 
                        #  'Begrenzung_1_Typ', 
                         'Begrenzung_1_Nr', 
                        #  'Begrenzung_2_Typ', 
                         'Begrenzung_2_Nr', 
                         'Elem_A_Typ', 'Elem_A_Nr', 
                         'Elem_B_Typ', 'Elem_B_Nr', 
                         'IFC_GUID_A', 'IFC_GUID_B', 
                         'Typus', 'Label', 'Comment'])
        writer.writerows(csv_data)
    print(f"\nAlle {len(csv_data)} Verbindungen erfolgreich in CSV exportiert: '{csv_filename}'")

def generate_rfem_graph_json(filename="rfem_graph.json"):
    # --- Pfad-Definitionen ---
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, filename)
    
    print(f"\nGeneriere RFEM-Graph und speichere in JSON-Datei unter:\n{filepath}...")
    
    # Verbindung nur aufbauen, wenn noch nicht geschehen, um Abbrüche zu vermeiden
    if not hasattr(Model, 'clientModel') or Model.clientModel is None:
        try:
            Model(new_model=False, model_name="")
        except Exception as e:
            print(f"Verbindungsfehler: Das aktive RFEM-Modell konnte nicht initialisiert werden. ({e})")
            return

    service = Model.clientModel.service
    graph_nodes = []
    graph_id = 1

    def get_numbers_safely(obj_type):
        """Hilfsfunktion zum sicheren Abrufen der Objektnummern, fängt Fehler ab."""
        try:
            nums = GetObjectNumbersByType(obj_type)
            # Falls die API unerwartet einen String wie "1 2 3" statt einer Liste zurückgibt
            if not isinstance(nums, (list, tuple)):
                nums = parse_id_list(str(nums))
            return nums
        except Exception as e:
            print(f"Warnung beim Abrufen von {obj_type.name}: {e}")
            return []

    # Abrufen aller relevanten Typen
    line_numbers = get_numbers_safely(ObjectTypes.E_OBJECT_TYPE_LINE)
    surface_numbers = get_numbers_safely(ObjectTypes.E_OBJECT_TYPE_SURFACE)
    node_numbers = get_numbers_safely(ObjectTypes.E_OBJECT_TYPE_NODE)
    member_numbers = get_numbers_safely(ObjectTypes.E_OBJECT_TYPE_MEMBER)
    
    # print(f"Elemente gefunden: {len(node_numbers)} Knoten, {len(line_numbers)} Linien, {len(member_numbers)} Stäbe, {len(surface_numbers)} Flächen.")
    print(f"Elemente gefunden: {len(member_numbers)} Stäbe, {len(surface_numbers)} Flächen.")

    # # --- KNOTEN VERARBEITEN ---
    # for no in node_numbers:
    #     try:
    #         node = service.get_node(no)
    #         obj_str = str(node)
    #         rfem_data = {
    #             "Nr": node.no,
    #             "Kategorie": "Knoten",
    #             "Type": str(getattr(node, 'type', 'Unbekannt')),
    #             "IFC_GUID": extract_ifc_guid(obj_str),
    #             "Koordinaten": [getattr(node, 'coordinate_1', 0.0), getattr(node, 'coordinate_2', 0.0), getattr(node, 'coordinate_3', 0.0)],
    #             "zugehoerige_Linien": [],
    #             "zugehoerige_Knoten": []
    #         }
    #         graph_nodes.append({"Graph_ID": graph_id, "RFEM": rfem_data})
    #         graph_id += 1
    #     except Exception as e:
    #         print(f"Fehler bei Knoten {no}: {e}")


    # # --- LINIEN VERARBEITEN ---
    # for no in line_numbers:
    #     try:
    #         line = service.get_line(no)
    #         obj_str = str(line)
    #         line_type = str(getattr(line, 'type', 'Unbekannt'))
            
    #         # Typischerweise ist eine Linie gekrümmt, wenn sie keine normale Linie oder Polylinie ist
    #         is_curved = 'TYPE_LINE' not in line_type and 'TYPE_POLYLINE' not in line_type
            
    #         rfem_data = {
    #             "Nr": line.no,
    #             "Kategorie": "Linie",
    #             "Type": line_type,
    #             "IFC_GUID": extract_ifc_guid(obj_str),
    #             "analytische_Laenge": getattr(line, 'length', None),
    #             "is_curved": is_curved,
    #             "is_rotated": getattr(line, 'rotation_angle', 0.0) != 0.0,
    #             "zugehoerige_Linien": [], # Linien bestehen nicht aus untergeordneten Linien
    #             "zugehoerige_Knoten": parse_id_list(getattr(line, 'definition_nodes', ''))
    #         }
            
    #         graph_nodes.append({"Graph_ID": graph_id, "RFEM": rfem_data})
    #         graph_id += 1
    #     except Exception as e:
    #         print(f"Fehler bei Linie {no}: {e}")


    # --- STÄBE VERARBEITEN ---
    for no in member_numbers:
        try:
            member = service.get_member(no)
            member_type = str(getattr(member, 'type', 'Unbekannt'))
            
            # Starre Stäbe im Graph ignorieren
            if member_type == 'TYPE_RIGID':
                continue
                
            obj_str = str(member)
            rfem_data = {
                "Nr": member.no,
                "Kategorie": "Stab",
                "Type": member_type,
                "IFC_GUID": extract_ifc_guid(obj_str),
                "analytische_Laenge": getattr(member, 'length', None),
                "is_curved": False,
                "is_rotated": getattr(member, 'rotation_angle', 0.0) != 0.0,
                "zugehoerige_Linien": parse_id_list(getattr(member, 'line', '')),
                "zugehoerige_Knoten": parse_id_list(getattr(member, 'nodes', ''))
            }
            graph_nodes.append({"Graph_ID": graph_id, "RFEM": rfem_data})
            graph_id += 1
        except Exception as e:
            print(f"Fehler bei Stab {no}: {e}")


    # --- FLÄCHEN VERARBEITEN ---
    for no in surface_numbers:
        try:
            surface = service.get_surface(no)
            obj_str = str(surface)
            geom_type = str(getattr(surface, 'geometry', ''))
            
            rfem_data = {
                "Nr": surface.no,
                "Kategorie": "Flaeche",
                "Type": f"{geom_type} / {getattr(surface, 'type', 'Unbekannt')}".strip(" /"),
                "IFC_GUID": extract_ifc_guid(obj_str),
                "analytische_Laenge": getattr(surface, 'area', None), # Analytische Länge einer Fläche interpretieren wir hier als Area/Flächeninhalt
                "is_curved": 'GEOMETRY_PLANE' not in geom_type if geom_type else False,
                "is_rotated": getattr(surface, 'axes_rotation_angle', 0.0) != 0.0,
                "zugehoerige_Linien": parse_id_list(getattr(surface, 'boundary_lines', '')),
                "zugehoerige_Knoten": parse_id_list(getattr(surface, 'nodes', '')) # RFEM-Flächen haben oft keine direkten Knoten in den Attributen
            }
            
            graph_nodes.append({"Graph_ID": graph_id, "RFEM": rfem_data})
            graph_id += 1
        except Exception as e:
            print(f"Fehler bei Fläche {no}: {e}")

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graph_nodes, f, indent=4, ensure_ascii=False)
    print(f"Graph erfolgreich lokal gespeichert! ({len(graph_nodes)} Elemente)")

if __name__ == "__main__":
    analyze_couplings()
    generate_rfem_graph_json()
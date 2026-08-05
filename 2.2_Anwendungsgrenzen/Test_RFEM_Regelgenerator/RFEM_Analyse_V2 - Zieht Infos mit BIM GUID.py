from RFEM.initModel import Model
from RFEM.enums import ObjectTypes
from RFEM.Tools.GetObjectNumbersByType import GetObjectNumbersByType
import re
import uuid

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

def analyze_couplings(n=100):
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
            is_first = (index == 0)
            is_nth = ((index + 1) % n == 0)
            is_last = (index == is_nth - 1)
            
            if not (is_first or is_nth or is_last):
                continue
                
            start_node = getattr(member, 'node_start_node', getattr(member, 'node_start', 'Unbekannt'))
            end_node = getattr(member, 'node_end_node', getattr(member, 'node_end', 'Unbekannt'))
            print(f"[Stab-Kopplung] Nr. {member.no} (Index {index + 1}/{total_rigid}): Verbindet Knoten {start_node} mit Knoten {end_node}")
            
            bim_info = get_bim_info(Model.clientModel.service, 'member', member.no)
            print(f"  -> {bim_info}")
            print("-" * 40)
    else:
        print("Keine Stäbe im Modell gefunden.")

    print("\n" + "=" * 60)

    # ==========================================
    # TEIL 2: STARRE KOPPLUNGEN (Rigid Links)
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
            is_first = (index == 0)
            is_nth = ((index + 1) % n == 0)
            is_last = (index == total - 1)
            
            if not (is_first or is_nth or is_last):
                continue
                
            link = Model.clientModel.service.get_rigid_link(no)
            
            print(f"[Starre Kopplung] Nr. {link.no} (Index {index + 1}/{total}):")
            
            link_type = getattr(link, 'type', 'Unbekannt')
            line1 = getattr(link, 'line1', 'Keine')
            line2 = getattr(link, 'line2', 'Keine')
            surface_id = getattr(link, 'surface', 'Keine')
            
            print(f"  -> Typ: {link_type}")
            print(f"  -> Bezugslinie: {line1} | Flächenlinie: {line2} | Fläche: {surface_id}")
            
            # Hier rufen wir die umgerechneten Daten ab!
            print(f"  -> BIM-Daten Bezugslinie ({line1}): {get_bim_info(Model.clientModel.service, 'line', line1)}")
            print(f"  -> BIM-Daten Flächenlinie ({line2}): {get_bim_info(Model.clientModel.service, 'line', line2)}")
            print(f"  -> BIM-Daten Fläche ({surface_id}): {get_bim_info(Model.clientModel.service, 'surface', surface_id)}")
            
            if is_first:
                print("\n  --- DEBUG-INFO FÜR DAS ERSTE RIGID-LINK-OBJEKT ---")
                print(f"  {link}")
                print("  ---------------------------------------\n")
                
            print("-" * 40)
    else:
        print("Keine 'Starren Kopplungen' gefunden.")

if __name__ == "__main__":
    analyze_couplings(n=1)
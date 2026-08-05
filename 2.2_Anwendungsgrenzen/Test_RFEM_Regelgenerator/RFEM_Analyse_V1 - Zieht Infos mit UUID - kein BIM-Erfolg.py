from RFEM.initModel import Model
from RFEM.enums import ObjectTypes
from RFEM.Tools.GetObjectNumbersByType import GetObjectNumbersByType

def get_bim_info(service, obj_type, no):
    """Hilfsfunktion: Liest Name, GUID und Kommentar eines gekoppelten Objekts aus"""
    if not no or str(no) == 'Keine' or str(no) == 'Unbekannt':
        return "N/A"
    try:
        if obj_type == 'line':
            obj = service.get_line(int(no))
        elif obj_type == 'surface':
            obj = service.get_surface(int(no))
        elif obj_type == 'member':
            obj = service.get_member(int(no))
        else:
            return "Unbekannter Typ"
        
        # Attribute sicher abrufen
        name = getattr(obj, 'name', '')
        uuid = getattr(obj, 'id_for_export_import', '')
        comment = getattr(obj, 'comment', '')
        
        info = []
        if name: info.append(f"Name: {name}")
        if uuid: info.append(f"UUID: {uuid}")
        if comment: info.append(f"Info: {comment}")
        
        return " | ".join(info) if info else "Keine IFC/BIM-Daten hinterlegt"
    except Exception:
        return "Nicht abrufbar"

def analyze_couplings():
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
        for no in member_numbers:
            member = Model.clientModel.service.get_member(no)
            if str(member.type) == 'TYPE_RIGID':
                start_node = getattr(member, 'node_start_node', getattr(member, 'node_start', 'Unbekannt'))
                end_node = getattr(member, 'node_end_node', getattr(member, 'node_end', 'Unbekannt'))
                print(f"[Stab-Kopplung] Nr. {member.no}: Verbindet Knoten {start_node} mit Knoten {end_node}")

                bim_info = get_bim_info(Model.clientModel.service, 'member', member.no)
                print(f"  -> IFC/BIM-Daten (Stab {member.no}): {bim_info}")
    else:
        print("Keine Stäbe im Modell gefunden.")

    print("\n" + "=" * 60)

    # ==========================================
    # TEIL 2: STARRE KOPPLUNGEN (Spezielle Objekte)
    # ==========================================
    print("TEIL 2: STARRE KOPPLUNGEN (Spezielle Objekte -> Rigid Links)")
    print("-" * 60)
    
    try:
        # Abfragen aller IDs vom Typ 'Rigid Link'
        rigid_link_numbers = GetObjectNumbersByType(ObjectTypes.E_OBJECT_TYPE_RIGID_LINK)
    except Exception as e:
        print(f"Fehler bei der Abfrage der Starren Kopplungen: {e}")
        rigid_link_numbers = []

    if rigid_link_numbers:
        print(f"{len(rigid_link_numbers)} Starre Kopplung(en) gefunden. Analysiere...")
        print("-" * 60)
        
        for index, no in enumerate(rigid_link_numbers):
            # Das Objekt 'Starre Kopplung' aus RFEM abrufen
            link = Model.clientModel.service.get_rigid_link(no)
            
            print(f"[Starre Kopplung] Nr. {link.no}:")
            
            # Die korrekten Felder gemäß Debug-Output auslesen
            link_type = getattr(link, 'type', 'Unbekannt')
            line1 = getattr(link, 'line1', 'Keine')
            line2 = getattr(link, 'line2', 'Keine')
            surface_id = getattr(link, 'surface', 'Keine')

            print(f"  -> Typ: {link_type}")
            print(f"  -> Bezugslinie: {line1} | Flächenlinie: {line2} | Fläche: {surface_id}")

            print(f"  -> BIM-Daten Bezugslinie ({line1}): {get_bim_info(Model.clientModel.service, 'line', line1)}")
            print(f"  -> BIM-Daten Flächenlinie ({line2}): {get_bim_info(Model.clientModel.service, 'line', line2)}")
            print(f"  -> BIM-Daten Fläche ({surface_id}): {get_bim_info(Model.clientModel.service, 'surface', surface_id)}")
                        
            # WICHTIG: Beim ersten Treffer (index == 0) drucken wir das komplette Objekt aus!
            if index == 0:
                print("\n  --- DEBUG-INFO FÜR DAS ERSTE OBJEKT ---")
                print("  So sieht die Datenstruktur direkt aus RFEM aus:")
                print(f"  {link}")
                print("  ---------------------------------------\n")
                
            print("-" * 40)
    else:
        print("Keine 'Starren Kopplungen' gefunden.")

if __name__ == "__main__":
    analyze_couplings()
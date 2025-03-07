import os
import ifcopenshell
import ifcopenshell.util.element as element
import ifcopenshell.geom as geom
import json

from pathlib import Path


"""Hauptfunktion zum Extrahieren von Daten aus IFC-Dateien"""
def extract_ifc_data(ifc_folder_path):
    """Extrahiert Daten aus IFC-Dateien und speichert sie in JSON-Dateien"""
    ## Durchsuche folder_path nach allen IFC-Dateien
    if not ifc_folder_path.exists():
        print(f"__Ordner nicht gefunden. Erstelle Pfad in: {ifc_folder_path}")
        ifc_files = None
    else:
        print(f"__Ordner gefunden. Suche IFC-Dateien in: {ifc_folder_path}")
        ifc_files = [datei for datei in os.listdir(ifc_folder_path) if datei.lower().endswith('.ifc')]
        # print(f"IFC-Dateien: {ifc_files}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    if not ifc_files:
        print(f"  __Keine IFC-Dateien gefunden. Extraktion abgebrochen.")
    else:
        print(f"  __IFC-Dateien gefunden. Starte Extraktion...")

        for ifc_file in ifc_files:
            ## Öffne Ifc-Datei
            ifc_file_path = ifc_folder_path / ifc_file
            model = ifcopenshell.open(ifc_file_path)
            print(f"\n\n  __Analysierte Datei: {ifc_file}")

            ## Liste zum Speichern der extrahierten Daten
            extracted_data = []

            ## Extrahiere Atribute
            for i, ifc_object in enumerate(model.by_type("IfcProduct")):
                """Quelle zum Lesen: https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifckernel/lexical/ifcproduct.htm"""
                ## Extrahieren der Attribute
                obj_attributes = get_attributes(ifc_object)
                obj_type = obj_attributes['type']
                obj_name = obj_attributes['name']
                obj_guid = obj_attributes['guid']
                obj_id = i
                # print("DEBUG_Attritbutes: ", obj_attributes, " | id: ", obj_id) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Extrahiere Merkmale aus P-Sets
                if ifc_object.Representation is not None:
                    obj_properties = element.get_psets(ifc_object)
                    # print("DEBUG_Props:", obj_type, " | psets:", obj_properties) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    ## Filtere nach Load-Bearing-Objekten
                    if is_load_bearing(obj_properties):
                        ## Extrahiere geometrische Daten
                        geometry = get_geometric_data(ifc_object, obj_name)
                        obj_position = geometry['position']
                        obj_orientation = geometry['orientation']
                        obj_verts_local = geometry['vertices_local']
                        obj_verts_global = geometry['vertices_global']
                        obj_edges = geometry['edges']
                        obj_faces = geometry['faces']
                        brep_data = geometry['breb_data']

                        ## Extrahieren aller Daten im Dictionary (Muss im load-bearing-Block sein)
                        # print(f"__Speichere Daten zum Objekt: {ifc_object}")
                        extracted_data.append({
                            'id': obj_id,
                            'ifc_type': obj_type,
                            'name': obj_name,
                            'guid': obj_guid,
                            'obj_properties': obj_properties,
                            'position': obj_position,
                            'orientation': obj_orientation,
                            'vertices_local': obj_verts_local,
                            'vertices_global': obj_verts_global,
                            'edges': obj_edges,
                            'faces': obj_faces,
                            'brep_data': brep_data
                            })
                    else:
                        # print(f"Objekt: Typ = {obj_type}, Name = {obj_name}, GUID = {obj_guid} ist nicht tragendes Element")
                        continue
                else:
                    # print(f"Objekt: Typ = {obj_type}, Name = {obj_name}, GUID = {obj_guid} ist kein geometrisches Element")
                    continue
            
            ## Speichere Daten in JSON-Datei
            export_data_to_json(extracted_data, ifc_folder_path, ifc_file)
            pass

    """
    KURZE ERKLÄRWEISE ALGORITHMUS:
    1. Durchsuche Ordner nach IFC-Dateien
    2. Für jede IFC-Datei:
        2.1 Öffne die Datei
        Für jedes Objekt in dieser Datei:
            2.2 Extrahiere die Attribute
            2.3 Extrahiere die P-Sets
            2.4 Filtere nach Load-Bearing-Objekten
                2.5 Extrahiere die geometrischen Daten  
                2.6 Speichere die Daten in einem Dictionary
    3. Speichere das Dictionary in einer JSON-Datei im script_dir

    If ifc_files:
        for ifc_file in ifc_files:
            model = ifc.open(ifc_file_path)
            extracted_data = []
            for object in model.by_type('IfcProduct'):
                # extrahiere attribute
                if object.representation:
                    # extrahieren der Psets
                    if object is_Load_Bearing:
                        # extrahiere geometrische Infos
                        extracted_data.append = Dictionary mit extrahierten Daten
            return extracted_data

            # save extracted_data as JSON in script_dir
    Else:
        Fehlermeldung
    """


"""Hilfsfunktionen zum Extrahieren von Daten aus IFC-Dateien"""
def get_ifc_folder_paths(script_dir, folder_names):
    """Gibt eine Liste von Pfaden zu den IFC-Ordnern zurück"""
    database_path = script_dir.parent / '1_1 Modelldatenbank'    
    ifc_folder_paths = [database_path / folder_name for folder_name in folder_names]
    # print(f"Database-Path - {database_path.exists()}: {database_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # print(f"IFC-Folder-Paths: {ifc_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return ifc_folder_paths


def get_attributes(ifc_object):
    """Extrahiert die Attribute eines IFC-Objekts"""
    ## IFC-Type und GUID extrahieren
    attributes = {}
    obj_type = ifc_object.is_a()
    obj_name = ifc_object.Name if hasattr(ifc_object, "Name") else None
    obj_guid = ifc_object.GlobalId if hasattr(ifc_object, "GlobalId") else None

    attributes['type'] = obj_type
    attributes['name'] = obj_name
    attributes['guid'] = obj_guid
    # print("Obj-Attributes:", attributes) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return attributes


def is_load_bearing(object_properties):
        """Prüft, ob ein Objekt tragend ist"""
        for pset_name, pset_data in object_properties.items():
            # print(f"DEBUG_IsLoadBearing: Durchsuche: {pset_name} mit {pset_data}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if 'LoadBearing' in pset_data and pset_data['LoadBearing']:
                # print(f"  In Pset '{pset_name}': LoadBearing ist True")
                return True
            else:
                # print(f"  In Pset '{pset_name}': LoadBearing ist nicht True oder nicht vorhanden")
                continue


def get_geometric_data(ifc_object, obj_name):
    """Extrahiert die geometrischen Informationen eines IFC-Objekts"""
    ## Extrahiere aus ObjectPlacement
    try:
        obj_position = ifc_object.ObjectPlacement.RelativePlacement.Location.Coordinates
        obj_orientation = ifc_object.ObjectPlacement.RelativePlacement.Axis.DirectionRatios
        # print(f"DEBUG_Geom: Position: {obj_position} | Orientation: {obj_orientation}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    except:
        obj_position = None
        obj_orientation = None
        # print(f"__Geometrische Extraktion (position, orientation) fehlerhaft für: {obj_name}")

    ## Extrahiere aus Geom
    try:
        settings = ifcopenshell.geom.settings()
        # settings.set(settings.USE_PYTHON_OPENCASCADE, True)

        shape = geom.create_shape(settings, inst=ifc_object)        
        verts_flat = shape.geometry.verts        
        edges_flat = shape.geometry.edges        
        faces_flat = shape.geometry.faces
        # print("DEBUG_Shape:", shape) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("DEBUG_Verts:", verts) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("DEBUG_Edges:", edges) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # # print("DEBUG_Faces:", faces) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        verts_local = [(verts_flat[i], verts_flat[i+1], verts_flat[i+2]) for i in range(0, len(verts_flat), 3)]
        verts_global = [(v[0] + obj_position[0], v[1] + obj_position[1], v[2] + obj_position[2]) for v in verts_local]
        edges = [(edges_flat[i], edges_flat[i+1]) for i in range(0, len(edges_flat), 2)]
        # print(f"DEBUG_Verts_sorted for {obj_name}:", verts_local) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(f"DEBUG_Verts_global for {obj_name}:", verts_global) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(f"DEBUG_Edges_sorted for {obj_name}:", edges_sorted) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        brep_data = None
        if hasattr(shape.geometry, 'brep_data'):
            brep_data = shape.geometry.brep_data
            print("DEBUG_Brep:", brep_data) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    except:
        print(f"__Geometrische Extraktion fehlerhaft für: {obj_name}")
    
    ## Erstelle Geometry-Dictionary
    geometry = {
        'position': obj_position, # Position im globalen Koordinatensystem
        'orientation': obj_orientation, # Orientierung des Objektes im globalen Koordinatensystem
        'vertices_local': verts_local, # Knotenpunkte (X, Y, Z) im lokalen Koordinatensystem (Bezug zu Position und Richtung)
        'vertices_global': verts_global, # Knotenpunkte (X, Y, Z) im globalen Koordinatensystem
        'edges': edges, # Indices der Knotenpunkte, welche die Kante bilden
        'faces': faces_flat, # Indices der Lnotenpunkte, welche die Fläche bilden
        'breb_data': brep_data, # Daten zur Beschreibung, falls Objekt BREP ist.
    }    
    return geometry



def export_data_to_json(extracted_data, ifc_folder_path, ifc_file):
    """Speichert die extrahierten Daten in einer JSON-Datei"""
    ## Definiere Namen und Pfad der JSON-Datei
    # normpath normalisiert Pfad, sodass er keine doppelten Schrägstriche oder abschließendes "\" enthält
    json_folder_name = os.path.basename(os.path.normpath(ifc_folder_path)) # Name des letzten Ordners aus Pfad
    json_folder_path = script_dir / json_folder_name
    # print(f"script_dir: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print(f"json_folder_name: {json_folder_name}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print(f"json_folder_path: {json_folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    os.makedirs(json_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden    

    json_file_name = ifc_file[:-4] + '.json' # Ersetzt '.ifc' mit '.json' | Alternative: ifc_file.replace('.ifc', '.json')
    # print(f"json_file_name: {json_file_name}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    json_file_path = json_folder_path / json_file_name
    # print(f"json_file_path: {json_file_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Struktur der JSON-Datei
    json_data = {
        "Bauteile": extracted_data
    }

    ## Speichere Parameter als .json
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)    
    print(f"__Export IFC-Daten erfolgreich für {json_file_name} in {json_file_path}")





if __name__ == "__main__":
    print("___Intitialisiere Skript___")
    # script_dir = Path.cwd() # Alternative: Path(__file__).parent
    script_dir = Path(__file__).parent
    # print(f"Script-Dir - {script_dir.exists()}: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # ifc_folder_names = ["Modell_2_Parametrisch IfcOpenShell", "Modell_2_Tutorial BimVision"]
    ifc_folder_names = ["Modell_2_DataBase", "Modell_2_Parametrisch IfcOpenShell"]
    ifc_folder_paths = get_ifc_folder_paths(script_dir, ifc_folder_names)

    for ifc_folder_path in ifc_folder_paths:
        # print(f"IFC-Folder-Path - {folder_path.exists()}:  {folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        extract_ifc_data(ifc_folder_path)
    
    print("___Skript beendet___")




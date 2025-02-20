import os
import ifcopenshell
import ifcopenshell.util.element as element
import ifcopenshell.geom as geom
import json

from pathlib import Path


"""Hauptfunktionen zum Extrahieren von Daten aus IFC-Dateien"""
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
            for ifc_object in model.by_type("IfcProduct"):
                """Quelle zum Lesen: https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifckernel/lexical/ifcproduct.htm"""
                ## Extrahieren der Attribute
                obj_type, obj_name, obj_guid = get_attributes(ifc_object)
                # print("DEBUG_Attritbutes: type:", obj_type, " | name:", obj_name, " | guid:", obj_guid)

                ## Extrahiere Merkmale aus P-Sets
                if ifc_object.Representation is not None:
                    obj_properties = element.get_psets(ifc_object)
                    # print("DEBUG_Props:", obj_type, " | psets:", obj_properties) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    ## Filtere nach Load-Bearing-Objekten
                    if is_load_bearing(obj_properties):
                        ## Extrahiere geometrische Daten
                        try:
                            obj_position = ifc_object.ObjectPlacement.RelativePlacement.Location.Coordinates
                            obj_orientation = ifc_object.ObjectPlacement.RelativePlacement.Axis.DirectionRatios
                            # print(f"DEBUG_Geom: Position: {obj_position} | Orientation: {obj_orientation}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                            settings = ifcopenshell.geom.settings()
                            # settings.set(settings.USE_PYTHON_OPENCASCADE, True)

                            shape = geom.create_shape(settings, inst=ifc_object)
                            # print("DEBUG_Shape:", shape) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            verts = shape.geometry.verts
                            # print("DEBUG_Verts:", verts) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            edges = shape.geometry.edges
                            # print("DEBUG_Edges:", edges) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            faces = shape.geometry.faces
                            # print("DEBUG_Faces:", faces) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        except:
                            print(f"__Geometrische Extraktion fehlerhaft für: {obj_name}")
                            continue

                        brep_data = None
                        if hasattr(shape.geometry, 'brep_data'):
                            brep_data = shape.geometry.brep_data
                            print("DEBUG_Brep:", brep_data) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                        ## Extrahieren aller Daten im Dictionary (Muss im load-bearing-Block sein)
                        # print(f"__Speichere Daten zum Objekt: {ifc_object}")
                        extracted_data.append({
                            'ifc_type': obj_type,
                            'name': obj_name,
                            'guid': obj_guid,
                            'obj_properties': obj_properties,
                            'position': obj_position,
                            'orientation': obj_orientation,
                            'verts': verts,
                            'edges': edges,
                            'faces': faces,
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
    # print(f"Database-Path - {database_path.exists()}: {database_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ifc_folder_paths = [database_path / folder_name for folder_name in folder_names]
    # print(f"IFC-Folder-Paths: {ifc_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return ifc_folder_paths


def get_attributes(ifc_object):
    """Extrahiert die Attribute eines IFC-Objekts"""
    ## IFC-Type und GUID extrahieren
    attributes = {}
    obj_type = ifc_object.is_a()
    obj_name = ifc_object.Name if hasattr(ifc_object, "Name") else None
    obj_guid = ifc_object.GlobalId if hasattr(ifc_object, "GlobalId") else None

    attributes['obj_type'] = obj_type
    attributes['obj_name'] = obj_name
    attributes['obj_guid'] = obj_guid
    # print("Obj-Attributes:", attributes)
    return obj_type, obj_name, obj_guid


def is_load_bearing(object_properties):
        """Prüft, ob ein Objekt tragend ist"""
        for pset_name, pset_data in object_properties.items():
            # print(f"DEBUG_IsLoadBearing: Durchsuche: {pset_name} mit {pset_data}")
            if 'LoadBearing' in pset_data and pset_data['LoadBearing']:
                # print(f"  In Pset '{pset_name}': LoadBearing ist True")
                return True
            else:
                # print(f"  In Pset '{pset_name}': LoadBearing ist nicht True oder nicht vorhanden")
                continue


def export_data_to_json(extracted_data, ifc_folder_path, ifc_file):
    """Speichert die extrahierten Daten in einer JSON-Datei"""
    ## Definiere Namen und Pfad der JSON-Datei
    # normpath normalisiert Pfad, sodass er keine doppelten Schrägstriche oder abschließendes "\" enthält
    json_folder_name = os.path.basename(os.path.normpath(ifc_folder_path)) # Name des letzten Ordners aus Pfad
    # print(f"json_folder_name: {json_folder_name}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    json_folder_path = script_dir / json_folder_name
    os.makedirs(json_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden    

    json_file_name = ifc_file[:-4] + '.json' # Ersetzt '.ifc' mit '.json' | Alternative: ifc_file.replace('.ifc', '.json')
    # print(f"json_file_name: {json_file_name}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    json_file_path = json_folder_path / json_file_name
    # print(f"json_file_path: {json_file_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Speichere Parameter als .json
    with open(json_file_path, 'w') as json_file:
        json.dump(extracted_data, json_file, indent=4)    
    print(f"__Export IFC-Daten erfolgreich für {json_file_name}")





if __name__ == "__main__":
    print("___Intitialisiere Skript___")
    script_dir = Path.cwd() # Alternative: Path(__file__).parent
    # print(f"Script-Dir - {script_dir.exists()}: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ifc_folder_names = ["Modell_3_Übungsmodell", "Modell_2_DataBase"]
    ifc_folder_paths = get_ifc_folder_paths(script_dir, ifc_folder_names)

    for ifc_folder_path in ifc_folder_paths:
        # print(f"IFC-Folder-Path - {folder_path.exists()}:  {folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        extract_ifc_data(ifc_folder_path)




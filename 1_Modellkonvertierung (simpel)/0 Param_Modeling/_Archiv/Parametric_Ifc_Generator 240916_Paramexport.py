# Einheitenangabe in m
print("___Intitialisiere Skript___")


# Für Schritt 3
import itertools
# Für Schritt 4
import numpy as np
# Für Schritt 5
import ifcopenshell
import ifcopenshell.api
import uuid
# Für Schritt 6
import os
import json


"""1. Definition von Parametern"""
class BuildingParameters:
    def __init__(self, grid_x_field, grid_y_field, grid_x_size, grid_y_size,
                 floors, floor_height, slab_thickness, foundation_thickness,
                 corner_columns, edge_columns, inner_columns):
        self.grid_x_field = grid_x_field          # Anzahl an Feldern in x-Richtung
        self.grid_y_field = grid_y_field          # Anzahl an Feldern in y-Richtung
        self.grid_x_size = grid_x_size            # Ausdehnung einer Gitterzelle in x-Richtung
        self.grid_y_size = grid_y_size            # Ausdehnung einer Gitterzelle in y-Richtung
        self.floors = floors                      # Geschossanzahl
        self.floor_height = floor_height          # Geschosshöhe
        self.slab_thickness = slab_thickness      # Deckendicke
        self.foundation_thickness = foundation_thickness  # Dicke der Bodenplatte
        self.corner_columns = corner_columns      # Eckstützen (True/False)
        self.edge_columns = edge_columns          # Randstützen (True/False)
        self.inner_columns = inner_columns        # Innenstützen (True/False)

    def __repr__(self):
        return (f"BuildingParameters("
                f"grid_x_field={self.grid_x_field}, grid_y_field={self.grid_y_field}, "
                f"grid_x_size={self.grid_x_size}, grid_y_size={self.grid_y_size}, "
                f"floors={self.floors}, floor_height={self.floor_height}, "
                f"slab_thickness={self.slab_thickness}, foundation_thickness={self.foundation_thickness}, "
                f"corner_columns={self.corner_columns}, edge_columns={self.edge_columns}, inner_columns={self.inner_columns})")


"""2. Bildung verschiedener Ranges für die Parameter"""
param_ranges = {
    # 'grid_x_field': [3, 4, 5],
    # 'grid_y_field': [2, 4, 5],
    # 'grid_x_size': [5.0, 7.5, 10.0],
    # 'grid_y_size': [5.0, 7.5, 10.0],
    # 'floors': [1, 2, 3],
    # 'floor_height': [3.0, 4.0],
    # 'slab_thickness': [0.2, 0.3],
    # 'foundation_thickness': [0.6],  # Dicke der Bodenplatte
    # 'corner_columns': [True, False],
    # 'edge_columns': [True, False],
    # 'inner_columns': [True, False],

    'grid_x_field': [3, 4, 5],
    'grid_y_field': [2, 4, 5],
    'grid_x_size': [5.0, 10.0],
    'grid_y_size': [5.0, 7.5],
    'floors': [2],
    'floor_height': [3.0],
    'slab_thickness': [0.2],
    'foundation_thickness': [0.6],  # Dicke der Bodenplatte
    'corner_columns': [False],
    'edge_columns': [True],
    'inner_columns': [True],
}


"""3. Kombination dieser Parameter zu einem Parameterset"""
def generate_parameter_combinations(param_ranges):
    # param_ranges.items gibt Listen von Tupeln wie etwa: [('grid_x_field', [3, 4, 5]), ('grid_y_field', [2, 4, 5]), ...]] zurück. Das Sternchen * entpackt Werte als separate Argumente
    # zip-Funktion gruppiert die entpackten Argumente aus den Tupeln in keys und values
    keys, values = zip(*param_ranges.items())
    combinations = [BuildingParameters(*combo) for combo in itertools.product(*values)]
    return combinations

## Generieren aller möglichen Parameterkombinationen
parameter_combinations = generate_parameter_combinations(param_ranges)
print(f"Beispiel Indices Parameterkombinationen: 0 bis {len(parameter_combinations)-1}")

## Beispiel: Ausgabe der ersten Kombinationen
for param_set in parameter_combinations[:5]:
    print(f"Beipsiel Param_Kombi: {param_set}")





"""4. Erstellen der Gebäudegeometrie"""
## Hilfsfunktion zum Erstellen des Rasters
def create_grid(grid_x_field, grid_y_field, grid_x_size, grid_y_size):
    x_positions = np.linspace(0, grid_x_size * grid_x_field, grid_x_field + 1)
    y_positions = np.linspace(0, grid_y_size * grid_y_field, grid_y_field + 1)
    
    grid = np.array(np.meshgrid(x_positions, y_positions)).T.reshape(-1, 2)

    return grid

## Hilfsfunktion zum Erstellen der Stützen
def create_columns(grid_points, grid_x_field, grid_y_field, corner_columns, edge_columns, inner_columns):
    columns = []
    
    for i, (x, y) in enumerate(grid_points):
        is_corner = (i in [0, grid_y_field, len(grid_points) - 1 - grid_y_field, len(grid_points) - 1])
        is_edge = (i % grid_x_field == 0 or i % grid_x_field == grid_x_field - 1 or i < grid_x_field or i >= len(grid_points) - grid_x_field)

        if is_corner and corner_columns:
            columns.append({'type': 'column', 'position_type': 'corner', 'position': (x, y)})
        elif not is_corner and is_edge and edge_columns:
            columns.append({'type': 'column', 'position_type': 'edge', 'position': (x, y)})
        elif not is_corner and not is_edge and inner_columns:
            columns.append({'type': 'column', 'position_type': 'inner', 'position': (x, y)})

    return columns


## Hilfsfunktion zum Erstellen der Bodenplatte
def create_foundation(grid_x_field, grid_y_field, grid_x_size, grid_y_size, foundation_thickness):    
    extent_x = grid_x_size * grid_x_field
    extent_y = grid_y_size * grid_y_field

    foundation = {
        'type': 'slab',
        'position': (extent_x/2, extent_y/2, 0),  # Bodenplatte auf Höhe 0
        'floor': 0,
        'size': (extent_x, extent_y),
        'thickness': foundation_thickness,
        'is_foundation': True  # Kennzeichnung als Bodenplatte
    }
    return foundation


## Hilfsfunktion zum Erstellen der Geschossdecken
def create_slab(floor, grid_x_field, grid_y_field, grid_x_size, grid_y_size, slab_thickness, floor_height):
    extent_x = grid_x_size * grid_x_field
    extent_y = grid_y_size * grid_y_field
    
    slab = {
        'type': 'slab',
        'position': (extent_x/2, extent_y/2, floor * (floor_height + slab_thickness)),
        'floor': floor - 1,
        'size': (extent_x, extent_y),
        'thickness': slab_thickness,
    }
    return slab


## Funktion zur Erstellung des Gebäudes mit Stützen, Decken und Bodenplatte
def create_building_geometry(parameters):
    ## Erstelle das Raster
    grid_points = create_grid(parameters.grid_x_field, parameters.grid_y_field, parameters.grid_x_size, parameters.grid_y_size)
    
    ## Beispiel: Ausgabe der ersten 5 Rasterpunkte
    # for point in grid_points[:15]:
    #     print(f"Beispiel Gitterpunkte: {point}")
    
    ## Erstelle die Bodenplatte
    building_geometry = [create_foundation(parameters.grid_x_field, parameters.grid_y_field,
                                           parameters.grid_x_size, parameters.grid_y_size,
                                           parameters.foundation_thickness)]
    
    ## Erstelle die Stützen und Decken für jedes Geschoss
    columns = create_columns(grid_points, parameters.grid_x_field, parameters.grid_y_field,
                                parameters.corner_columns, parameters.edge_columns, parameters.inner_columns)

    for floor in range(parameters.floors):
        z = floor * (parameters.floor_height + parameters.slab_thickness)
        
        ## Füge die Stützen hinzu
        for column in columns:
            building_geometry.append({
                'type': column['type'],
                'position_type': column['position_type'],
                'position': (column['position'][0], column['position'][1], z),
                'floor': floor,
                'size': (0.2, 0.2),
                'height': parameters.floor_height,
            })
        
        ## Füge die Decken hinzu
        slab = create_slab(floor + 1, parameters.grid_x_field, parameters.grid_y_field, 
                           parameters.grid_x_size, parameters.grid_y_size, parameters.slab_thickness,
                           parameters.floor_height)
        if slab: # Nur hinzufügen, wenn slab nicht None ist
            building_geometry.append(slab)
    
    
    ## Beispiel: Ausgabe der ersten Gebäudegeometrien:
    for element in building_geometry[:5]:
        print(f"Beispiel Gebäudegeometrie: {element}")
    
    return building_geometry

## Beispiel: Geometrie für die erste Parameterkombination erstellen
# example_params = parameter_combinations[0]
# building_geometry = create_building_geometry(example_params)





"""5. Erstellen der IFC-Datei"""
## Hilfsunktion zur Generierung von GUIDs
def generate_guid():
    return ifcopenshell.guid.compress(uuid.uuid1().hex)


## Hilfsfunktion zur Erstellung von Stützen
def create_column_geometry(element, context, model, column):
    # Erstelle die Basisgeometrie (z.B. ein Rechteck entsprechend der Größe der Decke)
    rect_profile = model.createIfcRectangleProfileDef(
        ProfileType="AREA",
        ProfileName=None,
        Position=model.createIfcAxis2Placement2D(
            Location=model.createIfcCartesianPoint((0.0, 0.0))
        ),
        XDim=element['size'][0],
        YDim=element['size'][1],
    )

    # Erstelle die Extrusion
    extrusion = model.createIfcExtrudedAreaSolid(
        SweptArea=rect_profile,
        Position=model.createIfcAxis2Placement3D(
            Location=model.createIfcCartesianPoint((0.0, 0.0, 0.0))
        ),
        ExtrudedDirection=model.createIfcDirection((0.0, 0.0, 1.0)),
        Depth=element['height']
    )

    # Erstelle das Produkt und weise Geometrie zu
    column.Representation = model.createIfcProductDefinitionShape(
        Representations=[model.createIfcShapeRepresentation(
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[extrusion]
        )]
    )

    # Erstelle das IFC-Objekt
    position = element['position']
    element_position = tuple(float(coord) for coord in position)

    column.ObjectPlacement = model.createIfcLocalPlacement(
        RelativePlacement=model.createIfcAxis2Placement3D(
            Location=model.createIfcCartesianPoint(element_position)
        )
    )


## Hilfsfunktion zur Generierung von Decken
def create_slab_geometry(element, context, model, slab):
    # Erstellen der Grundfläche (Rechteck)
    rect_profile = model.createIfcRectangleProfileDef(
        ProfileType="AREA",
        ProfileName=None,
        Position=model.createIfcAxis2Placement2D(
            Location=model.createIfcCartesianPoint((0.0, 0.0))
        ),
        XDim=element['size'][0],
        YDim=element['size'][1],
    )
    
    # Erstellen der Extrusion
    extrusion = model.createIfcExtrudedAreaSolid(
        SweptArea=rect_profile,
        Position=model.createIfcAxis2Placement3D(
            Location=model.createIfcCartesianPoint((0.0, 0.0, 0.0))
        ),
        ExtrudedDirection=model.createIfcDirection((0.0, 0.0, -1.0)),
        Depth=element['thickness']
    )
    
    # Erstellen des Produkts und Zuweisung einer geometrischen Repräsentation
    slab.Representation = model.createIfcProductDefinitionShape(
        Representations=[model.createIfcShapeRepresentation(
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[extrusion]
        )]
    )
    
    # Erstelle das IFC-Objekt
    position = element['position']
    element_position = tuple(float(coord) for coord in position)

    slab.ObjectPlacement = model.createIfcLocalPlacement(
        RelativePlacement=model.createIfcAxis2Placement3D(
            Location=model.createIfcCartesianPoint(element_position)
        )
    )


## Funktion zur Erstellung der IFC-Datei
def create_ifc(parameters, building_geometry, full_path):
    ## IFC-Datei initialisieren
    model = ifcopenshell.api.run("project.create_file")

    ## Projekt erstellen
    project = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcProject", name="Building Project")
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model", context_identifier="Building")

    ## Gelände (Site) erstellen
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Building Site")
    site.GlobalId = generate_guid()
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, product=site)

    ## Gebäude erstellen
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Building")
    building.GlobalId = generate_guid()
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, product=building)

    ## Geschosse erstellen
    storeys = {} # Speichere Geschoss für spätere Verwendung
    for floor in (range(parameters.floors)):
        if floor == 0:
            floor_name = f"EG"
        else:
            floor_name = f"{floor}. OG"
        
        storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name=floor_name)
        storey.GlobalId = generate_guid()
        ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, product=storey)
        storeys[floor] = storey
    
    ## Material erstellen
    Beton = ifcopenshell.api.run("material.add_material", model, name="Stahlbeton", category="Concrete") #--> Cocnrete als Kategorie lassen?
    
    ## Element_Counter für jeden Typ erstellen
    element_counters = {}

    for element in building_geometry:
        # Initialisiere Counter für Elementtyp
        element_type = element['type']
        if element_type not in element_counters:
            element_counters[element_type] = 1

    ## Geometrie erstellen
    for element in building_geometry:
        # Hole aktuellen Counter für Elementtyp
        element_type = element['type']
        counter = element_counters[element_type]

        # Definiere Name
        if element_type == 'slab' and 'is_foundation' in element:
            name = f"Bodenplatte-001"
        else:
            name = f"{element_type.capitalize()}-{counter:03d}"
            element_counters[element_type] += 1
        
        # Definiere Geschoss
        floor_number = element['floor']
        storey = storeys[floor_number]
        
        # Stütze oder Decke erstellen
        if element['type'] == 'column':
            column = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcColumn", name=name)
            column.GlobalId = generate_guid()
            ifcopenshell.api.run("spatial.assign_container", model, product=column, relating_structure=storey)
            ifcopenshell.api.run("material.assign_material", model, product=column, material=Beton)
            
            create_column_geometry(element, context, model, column)
        
        elif element['type'] == 'slab':
            slab = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSlab", name=name)
            slab.GlobalId = generate_guid()
            ifcopenshell.api.run("spatial.assign_container", model, product=slab, relating_structure=storey)
            ifcopenshell.api.run("material.assign_material", model, product=slab, material=Beton)

            create_slab_geometry(element, context, model, slab)


    ## IFC-Datei speichern
    model.write(full_path)

## Beispielaufruf zur Erstellung der IFC-Datei
# filepath = r"H:\1_Modelldatenbank\Modell_Parametrischer_Code"
# filename = "building_model"
# export_ifc(building_geometry, filepath, filename)





"""6. Export der IFC-Datei"""
## Hilfsfunktion zum Speichern der Parameter als .txt und .json
def export_params_to_files(params_dict, folder_dir, index):
    # Erstelle Dateinamen
    text_file_name = os.path.join(folder_dir, f"Parameter_Index_{index:03d}.txt")
    json_file_name = os.path.join(folder_dir, f"Parameter_Index_{index:03d}.json")

    # Speichere Parameter als .txt
    with open(text_file_name, 'w') as txt_file:
        txt_file.write(f"Parameter Set für Index {index}:\n")
        for key, value in params_dict.items():
            txt_file.write(f"{key}: {value}\n")
    
    # Speichere Parameter als .json
    with open(json_file_name, 'w') as json_file:
        json.dump(params_dict, json_file, indent=4)
    
    print(f"Parameter für Index {index} wurden erfolgreich als .txt und .json exportiert.")


## Funktion zum Exportieren eines Parametersets
def export_solo_ifc(index: int):
    print(f"___Starte IFC-Ausgabe für IFC-Modell. Aktueller Index: {index}")

    # Erhalte Parameterset basierend auf Index aus Kombinationsset
    params = parameter_combinations[index]
    params_dict = vars(params) # Wandelt params in Dict um: {'grid_x_field': 4, 'grid_y_field': 2, 'grid_x_size': 5.0, 'grid_y_size': 5.0, 'floors': 2, ...}
    # params_dict = params.__dict__ # Wandelt params in Dict um: {'grid_x_field': 4, 'grid_y_field': 2, 'grid_x_size': 5.0, 'grid_y_size': 5.0, 'floors': 2, ...}
    
    # Erstelle Geometrie basierend auf Parameterset
    building_geometry = create_building_geometry(params)

    # Überprüfe und Erstelle Dateipfad
    folder_path = os.path.dirname(os.path.realpath(__file__))

    folder_name = (
        f"{len(parameter_combinations)}-"
        f"Grid_{min(param_ranges['grid_x_field'])}x{min(param_ranges['grid_y_field'])}_"
        f"{max(param_ranges['grid_x_field'])}x{max(param_ranges['grid_y_field'])}-"
        f"Size_{min(param_ranges['grid_x_size'])}x{min(param_ranges['grid_y_size'])}_"
        f"{max(param_ranges['grid_x_size'])}x{max(param_ranges['grid_y_size'])}-"
        f"Floors_{min(param_ranges['floors'])}_{max(param_ranges['floors'])}-"
        f"FH_{min(param_ranges['floor_height'])}_{max(param_ranges['floor_height'])}-"
        f"ST_{min(param_ranges['slab_thickness'])}_{max(param_ranges['slab_thickness'])}-"
        f"FT_{min(param_ranges['foundation_thickness'])}_{max(param_ranges['foundation_thickness'])}-"
        f"CC_{''.join(map(str, sorted(set(int(c) for c in param_ranges['corner_columns']))))}-"
        f"EC_{''.join(map(str, sorted(set(int(c) for c in param_ranges['edge_columns']))))}-"
        f"IC_{''.join(map(str, sorted(set(int(c) for c in param_ranges['inner_columns']))))}"
    )

    folder_dir = os.path.join(folder_path, folder_name)
    os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

    file_name = f"Parametrisches_Modell Index_{index:03d}"
    suffix = ".ifc"

    full_path = os.path.join(folder_dir, file_name + suffix)
    
    # Erstelle und exportiere IFC-Datei
    create_ifc(params, building_geometry, full_path)

    # Exportiere Parameter in .txt und .json
    # export_params_to_files(params_dict, folder_dir, index)


## Funktion zum Exportieren aller Parametersets
def export_range_ifc():
    print(f"___Start IFC Ausgabe für alle IFC-Modelle. Anzahl an Parametersets: {len(parameter_combinations)}")
    
    for index in range(len(parameter_combinations)):
        export_solo_ifc(index)


## Exportiere IFC-Datei(en)
# export_solo_ifc(12)

# export_range_ifc()



"""
TODO
- Struktur in modulare Weise umbauen entsprechend der Schritte: 1. Parameterset und -kombination, 2. Geometrie erstellen, 3. IFC-Datei erzeugen, 4. IFC-Export (Solo/Serie)
- Attribut LoadBearing den Bauteilen hinzufügen

- Richtungsanpassung der Stützen (Orientierung in in Richtung X und Y)
    Basierend darauf dann Querschnittsmaße anpassbar machen
- Alignment der Stützen je nach Ausrichtung anpassen, sodass sie bündig mit Plattenrand abschließen.

- IFC-typische Materialien definieren (Beton, Stahl, MW, Holz -> Vorbild aus ArchiCAD BIM-Modul wählen)


- Fehler bei Stützenhöhe einbauen, damit nicht alle Stützen die Decke berühren. Höhe soll nur 0.8 oder 0.95 mal die Geschosshöhe annehmen
- Fehler bei Stützenplatzierung: Einige Stützen sollen über dem Raster hinaus gesetzt werden, die nicht tragen
- Fehler bei Decken und Bodenplatte: Ausdehnung muss über Stützen hinaus gehen bzw. kürzer sein, dann wäre Punkt oben auch abgedeckt


"""




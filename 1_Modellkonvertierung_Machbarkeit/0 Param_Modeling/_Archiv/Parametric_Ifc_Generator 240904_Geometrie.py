import itertools
import numpy as np

"""1. Definition von Parametern"""
class BuildingParameters:
    def __init__(self, grid_x_count, grid_y_count, grid_x_size, grid_y_size,
                 floors, floor_height, slab_thickness, foundation_thickness,
                 corner_columns, edge_columns, inner_columns):
        self.grid_x_count = grid_x_count          # Anzahl an Feldern in x-Richtung
        self.grid_y_count = grid_y_count          # Anzahl an Feldern in y-Richtung
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
                f"grid_x_count={self.grid_x_count}, grid_y_count={self.grid_y_count}, "
                f"grid_x_size={self.grid_x_size}, grid_y_size={self.grid_y_size}, "
                f"floors={self.floors}, floor_height={self.floor_height}, "
                f"slab_thickness={self.slab_thickness}, foundation_thickness={self.foundation_thickness}, "
                f"corner_columns={self.corner_columns}, edge_columns={self.edge_columns}, inner_columns={self.inner_columns})")

"""2. Bildung verschiedener Ranges für die Parameter"""
param_ranges = {
    'grid_x_count': [3, 4, 5],
    'grid_y_count': [3, 4, 5],
    'grid_x_size': [5.0, 7.5, 10.0],
    'grid_y_size': [5.0, 7.5, 10.0],
    'floors': [1, 2, 3],
    'floor_height': [3.0, 3.5, 4.0],
    'slab_thickness': [0.2, 0.3],
    'foundation_thickness': [0.6],  # Dicke der Bodenplatte
    'corner_columns': [True, False],
    'edge_columns': [True, False],
    'inner_columns': [True, False]
}

"""3. Kombination dieser Parameter zu einem Parameterset"""
def generate_parameter_combinations(param_ranges):
    keys, values = zip(*param_ranges.items())
    combinations = [BuildingParameters(*combo) for combo in itertools.product(*values)]
    return combinations

## Generieren aller möglichen Parameterkombinationen
parameter_combinations = generate_parameter_combinations(param_ranges)

## Beispiel: Ausgabe der ersten 5 Kombinationen
for param_set in parameter_combinations[:5]:
    print(f"Beipsiel Param_Kombi: {param_set}")

"""4. Erstellen der Gebäudegeometrie"""
## 1. Funktion zur Erstellung des Rasters
def create_grid(grid_x_count, grid_y_count, grid_x_size, grid_y_size):
    x_positions = np.linspace(0, grid_x_size * (grid_x_count - 1), grid_x_count)
    y_positions = np.linspace(0, grid_y_size * (grid_y_count - 1), grid_y_count)
    return np.array(np.meshgrid(x_positions, y_positions)).T.reshape(-1, 2)

## 2. Funktion zur Stützenplatzierung
def place_columns(grid_points, grid_x_count, grid_y_count, corner_columns, edge_columns, inner_columns):
    columns = []
    
    for i, (x, y) in enumerate(grid_points):
        is_corner = (i in [0, grid_x_count - 1, len(grid_points) - grid_x_count, len(grid_points) - 1])
        is_edge = (i % grid_x_count == 0 or i % grid_x_count == grid_x_count - 1 or i < grid_x_count or i >= len(grid_points) - grid_x_count)
        
        if is_corner and corner_columns:
            columns.append({'type': 'column', 'position_type': 'corner', 'position': (x, y)})
        elif is_edge and edge_columns:
            columns.append({'type': 'column', 'position_type': 'edge', 'position': (x, y)})
        elif not is_edge and not is_corner and inner_columns:
            columns.append({'type': 'column', 'position_type': 'inner', 'position': (x, y)})

    return columns

## 3. Funktion zur Erstellung der Geschossdecken
def create_slab(floor, grid_x_count, grid_y_count, grid_x_size, grid_y_size, slab_thickness, floor_height):
    if floor == 0:
        return None

    slab = {
        'type': 'slab',
        'position': (0, 0, floor * (floor_height + slab_thickness)),
        'size': (grid_x_size * (grid_x_count - 1), grid_y_size * (grid_y_count - 1)),
        'thickness': slab_thickness
    }
    return slab

## 4. Funktion zur Erstellung der Bodenplatte
def create_foundation(grid_x_count, grid_y_count, grid_x_size, grid_y_size, foundation_thickness):
    foundation = {
        'type': 'slab',
        'position': (0, 0, 0),  # Bodenplatte auf Höhe 0
        'size': (grid_x_size * (grid_x_count - 1), grid_y_size * (grid_y_count - 1)),
        'thickness': foundation_thickness,
        'is_foundation': True  # Kennzeichnung als Bodenplatte
    }
    return foundation

## 5. Funktion zur Erstellung des Gebäudes mit Stützen, Decken und Bodenplatte
def create_building_geometry(parameters):
    ## Erstelle das Raster
    grid_points = create_grid(parameters.grid_x_count, parameters.grid_y_count, parameters.grid_x_size, parameters.grid_y_size)
    
    ## Beispiel: Ausgabe der ersten 5 Rasterpunkte
    for point in grid_points[:5]:
        print(f"Beispiel Rasterpunkte: {point}")
    
    ## Erstelle die Bodenplatte
    building_geometry = [create_foundation(parameters.grid_x_count, parameters.grid_y_count,
                                           parameters.grid_x_size, parameters.grid_y_size,
                                           parameters.foundation_thickness)]
    
    ## Erstelle die Stützen und Decken für jedes Geschoss
    for floor in range(parameters.floors):
        z = floor * (parameters.floor_height + parameters.slab_thickness)
        columns = place_columns(grid_points, parameters.grid_x_count, parameters.grid_y_count,
                                parameters.corner_columns, parameters.edge_columns, parameters.inner_columns)
        
        ## Füge die Stützen hinzu
        for column in columns:
            building_geometry.append({
                'type': column['type'],
                'position_type': column['position_type'],
                'position': (column['position'][0], column['position'][1], z)
            })
        
        ## Füge die Decken hinzu
        slab = create_slab(floor, parameters.grid_x_count, parameters.grid_y_count, 
                           parameters.grid_x_size, parameters.grid_y_size, parameters.slab_thickness,
                           parameters.floor_height)
        if slab: # Nur hinzufügen, wenn slab nicht None ist
            building_geometry.append(slab)
    
    return building_geometry

## Beispiel: Geometrie für die erste Parameterkombination erstellen
example_params = parameter_combinations[0]
building_geometry = create_building_geometry(example_params)

## Beispiel: Ausgabe der ersten 15 Elemente:
for element in building_geometry[:15]:
    print(f"Beispiel Geometrie: {element}")

import itertools

# 1. Definition von Parametern
class BuildingParameters:
    def __init__(self, grid_x_count, grid_y_count, grid_x_size, grid_y_size,
                 floors, floor_height, slab_thickness, corner_columns, edge_columns, inner_columns):
        self.grid_x_count = grid_x_count          # Anzahl an Feldern in x-Richtung
        self.grid_y_count = grid_y_count          # Anzahl an Feldern in y-Richtung
        self.grid_x_size = grid_x_size            # Ausdehnung einer Gitterzelle in x-Richtung
        self.grid_y_size = grid_y_size            # Ausdehnung einer Gitterzelle in y-Richtung
        self.floors = floors                      # Geschossanzahl
        self.floor_height = floor_height          # Geschosshöhe
        self.slab_thickness = slab_thickness      # Deckendicke
        self.corner_columns = corner_columns      # Eckstützen (True/False)
        self.edge_columns = edge_columns          # Randstützen (True/False)
        self.inner_columns = inner_columns        # Innenstützen (True/False)

    def __repr__(self):
        return (f"BuildingParameters("
                f"grid_x_count={self.grid_x_count}, grid_y_count={self.grid_y_count}, "
                f"grid_x_size={self.grid_x_size}, grid_y_size={self.grid_y_size}, "
                f"floors={self.floors}, floor_height={self.floor_height}, "
                f"slab_thickness={self.slab_thickness}, corner_columns={self.corner_columns}, "
                f"edge_columns={self.edge_columns}, inner_columns={self.inner_columns})")


# 2. Bildung verschiedener Ranges für die Parameter
param_ranges = {
    'grid_x_count': [3, 4, 5],
    'grid_y_count': [3, 4, 5],
    'grid_x_size': [5.0, 7.5, 10.0],
    'grid_y_size': [5.0, 7.5, 10.0],
    'floors': [1, 2, 3],
    'floor_height': [3.0, 3.5, 4.0],
    'slab_thickness': [0.2, 0.3],
    'corner_columns': [True, False],
    'edge_columns': [True, False],
    'inner_columns': [True, False]
}

# 3. Kombination dieser Parameter zu einem Parameterset
def generate_parameter_combinations(param_ranges):
    keys, values = zip(*param_ranges.items())
    combinations = [BuildingParameters(*combo) for combo in itertools.product(*values)]
    return combinations

# Generieren aller möglichen Parameterkombinationen
parameter_combinations = generate_parameter_combinations(param_ranges)

# Beispiel: Ausgabe der ersten 5 Kombinationen
for param_set in parameter_combinations[:5]:
    print(param_set)

# Für Schritt 2 (Parameterkombination)
import itertools
# Für Schritt 3 (Geometrieerstellung)
import numpy as np
import random
# Für Schritt 4 (IFC-Erstellung)
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
# Für Schritt 5 (Export)
import os
import json



"""Hauptfunktion"""
def main():
    """Hauptfunktion zum Erzeugen von IFC-Dateien"""

    ## Definiere Parameterbereich
    global param_values
    param_values = {
        ## Parameter 2D-Gitter
        'grid_x_n': [2, 4],
        'grid_y_n': [3, 5],
        'grid_x_size': [5.0, 8], # in [m]
        'grid_y_size': [7.5, 10.0], # in [m]
        'grid_x_offset': [0.0, 4.0], # in [m]
        'grid_y_offset': [0.0, 3.0], # in [m]
        'grid_rotation': [0.0], # in [degree] --> Bauteile passen sich Drehung nicht mit an
        ## Parameter Geschoss
        'floors': [2, 4, 5],
        'floor_height': [2.8], # in [m]
        ## Parameter Bauteil - Platten
        'slab_thickness': [0.3], # in [m]
        'foundation_thickness': [0.6],  # in [m]
        ## Parameter Bauteil - Stützen
        'corner_columns': [True, False],
        'edge_columns': [True, False],
        'inner_columns': [True, False],
        'column_profile': [ProfileFactory.Rec_20_30] # in [m] --> [('rectangle', x_dim, y_dim)], [('circle', radius)], [('i-profile', h, b, t_f, t_w)]
    }
    
    ## Generiere alle möglichen Parameterkombinationen
    global param_combination
    param_combination = BuildingParameters.generate_parameter_combination(param_values)

    BuildingParameters.Beispiel_Param_Kombi_Indices() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # BuildingParameters.Beispiel_Param_Kombi_Set(max_index = 6) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Exportiere IFC-Modelle
    folder_path = get_folder_path()
    # index_list = [0, 1, 2] # Leere Liste beudeutet alle Kombinationen
    index_list = [] # Leere Liste beudeutet alle Kombinationen

    export_ifc_model(folder_path, index_list)



"""1. Definition Parameter für das Bauwerk"""
class BuildingParameters:
    """Klasse zum Speichern der Parameter für ein Bauwerk"""
    def __init__(self, grid_x_n, grid_y_n, grid_x_size, grid_y_size, grid_x_offset, grid_y_offset, grid_rotation,
                 floors, floor_height, slab_thickness, foundation_thickness,
                 corner_columns, edge_columns, inner_columns, column_profile):
        ## Parameter 2D-Gitter
        self.grid_x_n = grid_x_n                  # Anzahl an Feldern in x-Richtung
        self.grid_y_n = grid_y_n                  # Anzahl an Feldern in y-Richtung
        self.grid_x_size = grid_x_size            # Ausdehnung einer Gitterzelle in x-Richtung
        self.grid_y_size = grid_y_size            # Ausdehnung einer Gitterzelle in y-Richtung
        self.grid_x_offset = grid_x_offset        # Verschiebung in X-Richtung
        self.grid_y_offset = grid_y_offset        # Verschiebung in Y-Richtung
        self.grid_rotation = grid_rotation        # Verschiebung in X-Richtung
        ## Parameter Geschoss
        self.floors = floors                      # Anzahl an Geschossen
        self.floor_height = floor_height          # Geschosshöhe (OKRD bis OKRD)
        ## Parameter Bauteil - Platten
        self.slab_thickness = slab_thickness      # Deckendicke
        self.foundation_thickness = foundation_thickness  # Dicke der Bodenplatte
        ## Parameter Bauteil - Stützen
        self.corner_columns = corner_columns      # Eckstützen (True/False)
        self.edge_columns = edge_columns          # Randstützen (True/False)
        self.inner_columns = inner_columns        # Innenstützen (True/False)
        self.column_profile = column_profile      # Stützenprofil mit entsprechenden Parametern


    def __repr__(self):
        return (f"BuildingParameters("
                f"grid_x_n={self.grid_x_n}, grid_y_n={self.grid_y_n}, "
                f"grid_x_size={self.grid_x_size}, grid_y_size={self.grid_y_size}, "
                f"floors={self.floors}, floor_height={self.floor_height}, "
                f"slab_thickness={self.slab_thickness}, foundation_thickness={self.foundation_thickness}, "
                f"corner_columns={self.corner_columns}, edge_columns={self.edge_columns}, inner_columns={self.inner_columns}, column_profile={self.column_profile})")



    """2. Kombination Parameter zu mehreren Parametersets"""
    @staticmethod
    def generate_parameter_combination(param_values):
        """Funktion zur Kombination der Parameter zu Parametersets"""
        ## Erklärung Syntax
        # param_values.items() gibt aus Dictionary Liste von Tupeln wie etwa: [('grid_x_n', [3, 4, 5]), ('grid_y_n', [2, 4, 5]), ...] zurück.
        # Das Sternchen * entpackt Werte als separate Argumente
        # zip-Funktion gruppiert die entpackten Argumente aus den Tupeln in keys und values
        keys, values = zip(*param_values.items())
        combinations = [BuildingParameters(*combo) for combo in itertools.product(*values)]
        return combinations

    ## Beispiel: Ausgabe der Range der Parametersets
    @staticmethod
    def Beispiel_Param_Kombi_Indices():
        print(f"_Beispiel Indices der Parametersets: 0 bis {len(param_combination)-1}")

    ## Beispiel: Ausgabe der ersten n Parametersets
    @staticmethod
    def Beispiel_Param_Kombi_Set(max_index):
        for index, param_set in enumerate(param_combination[:int(max_index)]):
            print(f"_Beipsiel Param_Set {index}: {param_set}")



"""3. Generierung Geometrie"""
class BuildingGeometry:
    """Klasse zum Erstellen der Gebäudegeometrie"""
    def __init__(self, parameters):
        self.parameters = parameters
        self.grid_points = None
        self.geometry = []
    

    def create_grid(self):
        """Funktion zur Erstellung des Grids"""
        grid_x_n, grid_y_n = self.parameters.grid_x_n, self.parameters.grid_y_n
        grid_x_size, grid_y_size = self.parameters.grid_x_size, self.parameters.grid_y_size
        offset_x, offset_y = self.parameters.grid_x_offset, self.parameters.grid_y_offset
        angle = np.radians(self.parameters.grid_rotation) # Grad -> Radiant

        ## Erstelle reguläre Gitterpunkte
        x_positions = np.linspace(0, grid_x_size * grid_x_n, grid_x_n + 1)
        y_positions = np.linspace(0, grid_y_size * grid_y_n, grid_y_n + 1)
        grid = np.array(np.meshgrid(x_positions, y_positions)).T.reshape(-1, 2) #reshape(-1 = Länge der ersten Dimension wird angepasst, 2 = Jede Zeile enthält 2 Werte)
        # print("ohne offset:", grid) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Drehe Grid um Winkel α
        if angle != 0:
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            grid = np.dot(grid, rotation_matrix.T)
        # print("mit Drehung:", grid) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Verschiebe Grid
        grid[:,0] += offset_x
        grid[:,1] += offset_y
        # print("mit offset:", grid) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        self.grid_points = grid
        return self.grid_points
    

    @staticmethod
    def is_point_on_edge(px, py, x1, y1, x2, y2, tolerance = 1e-5):
        """Funktion zur Überprüfung mit dem Kreuzprodukt, ob ein Punkt auf einem Vektor liegt"""
        cross_product = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross_product) > tolerance:
            return False
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
    

    def create_foundation(self):
        """Funktion zur Erstellung der Bodenplatte"""
        extent_x = self.parameters.grid_x_size * self.parameters.grid_x_n
        extent_y = self.parameters.grid_y_size * self.parameters.grid_y_n

        foundations = [] # WIP, # Soll neben Bodenplatte auch Streifenfundamente an den Kanten erstellen. Streifenfundamente = strip foundation
        foundation = {
            'name': 'Bodenplatte',
            'type': 'slab',
            'position': (self.grid_points[0][0] + extent_x/2, self.grid_points[0][1] + extent_y/2, 0), # Oberkante Bodenplatte bei z=0
            'rotation': 0.0,
            'floor': -1,
            'extent': (extent_x, extent_y),
            'modify_x': 1.0,
            'modify_y': 1.0,
            'thickness': self.parameters.foundation_thickness,
            'is_foundation': True # Kennzeichnung für Fundamentgeschoss
        }
        return foundation
    

    def create_slab(self, z, floor, floor_name, index: int):
        """Funktion zur Erstellung von Geschossdecken"""
        grid_x_size = self.parameters.grid_x_size
        grid_y_size = self.parameters.grid_y_size
        grid_x_n = self.parameters.grid_x_n
        grid_y_n = self.parameters.grid_y_n
        extent_x = grid_x_size * grid_x_n
        extent_y = grid_y_size * grid_y_n
        position = (self.grid_points[0][0] + extent_x/2, self.grid_points[0][1] + extent_y/2, z + self.parameters.floor_height)

        ## Fehlereinbau
        if ErrorFactory.should_apply_error(index, probability=0.2):
            random.seed() # Seed zurücksetzen
            scale_x = ErrorFactory.modify_slab_extent(self, grid_x_size, grid_x_n)
            scale_y = ErrorFactory.modify_slab_extent(self, grid_y_size, grid_y_n)
            print(f"MODIFY: Geschossdecke {floor_name} mit {np.round(scale_x*100, 3)}% und  {np.round(scale_y*100, 3)}%")
        else:
            scale_x = scale_y = 1.0
        
        extent_x = extent_x * scale_x
        extent_y = extent_y * scale_y

        slab = {
            'name': f'Decke {floor_name}',
            'type': 'slab',
            # 'position': (self.grid_points[0][0] + extent_x/2, self.grid_points[0][1] + extent_y/2, z + self.parameters.floor_height),
            'position': position,
            'rotation': 0.0,
            'floor': floor,
            'extent': (extent_x, extent_y),
            'modify_x': scale_x,
            'modify_y': scale_y,
            'thickness': self.parameters.slab_thickness,
            'is_foundation': False # Kennzeichnung für Fundamentgeschoss
        }
        return slab
    
    
    def create_columns(self, z, floor, floor_name, column_profile):
        """Funktion zur Erstellung von Stützen"""
        if self.grid_points is None:
            raise ValueError("Grid must be created before columns.")
        
        columns = []
        grid_points = self.grid_points
        grid_x_n = self.parameters.grid_x_n
        grid_y_n = self.parameters.grid_y_n
        extent_x = self.parameters.grid_x_size * self.parameters.grid_x_n
        extent_y = self.parameters.grid_y_size * self.parameters.grid_y_n
        # strong_axis = 'x' if extent_x > extent_y else 'y' # In which direction strong axis is needed
        strong_axis = 'y' # fixed due to further issues with rotation later on

        ## Definiere x (Breite) und y (Höhe) des Querschnitts
        if column_profile[0] == 'rectangle':
            profile_x, profile_y = column_profile[1:]
            material = "Beton_C3037"
        elif column_profile[0] == 'circle':
            profile_x = profile_y = column_profile[1]
            material = "Beton_C3037"
        elif column_profile[0] == 'i-profile':
            profile_x = column_profile[2]
            profile_y = column_profile[1]
            material = "Stahl_S235"
        else:
            raise ValueError(f"Unbekannter Profiltyp: {column_profile[0]}")
        
        ## Definiere Eckpunkte des Gitters
        corner_points = {
            "bottom_left": self.grid_points[0],
            "top_left": self.grid_points[grid_y_n],
            "bottom_right": self.grid_points[len(self.grid_points) - 1 - grid_y_n],
            "top_right": self.grid_points[len(self.grid_points) - 1]
        }
        cp_0_x, cp_0_y = corner_points['bottom_left'][0], corner_points['bottom_left'][1]
        cp_1_x, cp_1_y = corner_points['bottom_right'][0], corner_points['bottom_right'][1]
        cp_2_x, cp_2_y = corner_points['top_right'][0], corner_points['top_right'][1]
        cp_3_x, cp_3_y = corner_points['top_left'][0], corner_points['top_left'][1]

        ## Definiere Stützenmerkmale
        for i, (x, y) in enumerate(self.grid_points):
            ## Definiere Lage des Punktes
            is_corner = (x == cp_0_x and y == cp_0_y or 
                         x == cp_1_x and y == cp_1_y or                      
                         x == cp_2_x and y == cp_2_y or                      
                         x == cp_3_x and y == cp_3_y
                         )            
            on_edge_lower = BuildingGeometry.is_point_on_edge(x, y, cp_0_x, cp_0_y, cp_1_x, cp_1_y)
            on_edge_right = BuildingGeometry.is_point_on_edge(x, y, cp_1_x, cp_1_y, cp_2_x, cp_2_y)
            on_edge_upper = BuildingGeometry.is_point_on_edge(x, y, cp_2_x, cp_2_y, cp_3_x, cp_3_y)
            on_edge_left = BuildingGeometry.is_point_on_edge(x, y, cp_3_x, cp_3_y, cp_0_x, cp_0_y)
            is_edge = on_edge_lower or on_edge_right or on_edge_upper or on_edge_left

            ## Definiere Rotation und Alignment
            rotation = 0.0 if strong_axis == 'y' else 270.0
            x_offset, y_offset = 0.0, 0.0
            
            if strong_axis == 'y':
                y_offset += profile_y/2 if on_edge_lower else 0
                x_offset += -profile_x/2 if on_edge_right else 0
                y_offset += -profile_y/2 if on_edge_upper else 0
                x_offset += profile_x/2 if on_edge_left else 0
            else:
                y_offset += profile_x/2 if on_edge_lower else 0
                x_offset += -profile_y/2 if on_edge_right else 0
                y_offset += -profile_x/2 if on_edge_upper else 0
                x_offset += profile_y/2 if on_edge_left else 0

            x += x_offset
            y += y_offset

            ## Stützendaten hinzufügen
            column_data = i, x, y, z, rotation, floor, floor_name, column_profile, material
            # print('col_data: ', column_data) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if is_corner and self.parameters.corner_columns:
                position_type = 'corner'
                columns.append(self.add_column_data(*column_data, position_type))
            elif not is_corner and is_edge and self.parameters.edge_columns:
                if on_edge_lower or on_edge_upper:
                    position_type = 'edge_x'
                elif on_edge_left or on_edge_right:
                    position_type = 'edge_y'
                else:
                    position_type = 'edge'
                columns.append(self.add_column_data(*column_data, position_type))
            elif not is_corner and not is_edge and self.parameters.inner_columns:
                position_type = 'inner'
                columns.append(self.add_column_data(*column_data, position_type))
            else:
                pass
            # print(f"INDEX: {i}, position_type: {position_type}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return columns
    

    def add_column_data(self, i, x, y, z, rotation, floor, floor_name, column_profile, material, position_type):
        """Ergänzt Stütze mit Merkmalen"""
        name = f'Stütze {position_type} {floor_name} {i:02d}'

        ## Fehlereinbau
        height = self.parameters.floor_height - self.parameters.slab_thickness
        if ErrorFactory.should_apply_error(i + z, probability=0.26):
            random.seed() # Seed zurücksetzen
            scale = ErrorFactory.modify_column_height(self, height)
            height = height * scale
            print(f"MODIFY: Stütze {name} mit {np.round(scale*100, 3)}%")
        else:
            scale = 1.0

        return{'name': name, 
               'type': 'column', 
               'position_type': position_type, 
               'position': (x, y, z),
               'rotation': rotation,
               'floor': floor, 
               'profile': column_profile,
               'material': material,
               'height': height,
               'modify_heigth': scale
        }
    

    def generate_geometry(self, index: int):
        """Funktion zum Erstellen der Geometrie"""
        ## Generierung Fundament
        foundation = self.create_foundation()
        self.geometry.append(foundation)

        for current_floor in range(self.parameters.floors):
            z = current_floor * self.parameters.floor_height

            if current_floor == 0:
                floor_name = 'EG'
            else:
                floor_name = f'OG{current_floor}'

            ## Generierung Deckenplatten
            slab = self.create_slab(z, current_floor, floor_name, index)
            self.geometry.append(slab)

            ## Generierung Stützen
            columns = self.create_columns(z, current_floor, floor_name, self.parameters.column_profile)
            for column in columns:
                self.geometry.append(column)
                        
        return self.geometry
    
    ## Beispiel: Ausgabe der ersten n Geometrieobjekte
    def Beispiel_Building_Geometry(self, max_index):
        for index, object in enumerate(self.geometry[:max_index]):
            print(f"_Beipsiel Geometry_Objetct {index:02d}: {object}")


class ErrorFactory:
    """Klasse zum Erzeugen von Modellfehlern"""
    @staticmethod
    def should_apply_error(index, probability= 0.3):
        random.seed(index)
        """Funktion zur Entscheidung, ob ein Fehler angewendet werden soll"""
        return random.random() < probability
    
    
    def modify_slab_extent(self, grid_size, grid_n):
        """Funktion zur Modifikation der Deckenplatte"""
        ## Überprüfe Stützenprofil
        column_profile = self.parameters.column_profile
        if column_profile[0] == 'circle':
            extent_profile = column_profile[1]
        else:
            extent_profile = max(column_profile[1], column_profile[2])
        
        ## Bestimme Grenzwerte für Modifikation
        min_variation = extent_profile / 2 / grid_size # min halbes Stützenprofil als Faktor
        max_variation = 1 / (grid_n + 1) # max 1/3 bei 2 Feldern, 1/4 bei 3 Feldern, etc...
        # print("variations: ", min_variation, max_variation) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Modifiziere Abmessungsfaktor
        if random.random() < 0.5:
            # Platte wird vergrößert
            scale = np.round(random.uniform(1 + min_variation, 1 + max_variation), 5)
        else:
            # Platte wird verkleinert
            scale = np.round(random.uniform(1 - max_variation, 1 - min_variation), 5)

        return scale
    
    
    def modify_column_height(self, height):
        """Funktion zur Modifikation der Deckenplatte"""
        min_variation = 0.05 / height # min 5cm als Faktor (z.B. 0.01 bei 5m oder 0.004 bei 12,5m)
        max_variation = 0.5 # max halbe Stützenhöhe
        # print("variations: ", min_variation, max_variation) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        scale = np.round(random.uniform(1 - max_variation, 1 - min_variation), 5)
        return scale



"""4. Generierung IFC-Modell"""
class ProfileFactory:
    """Klasse zur Erzeugung von Querschnittsprofilen"""
    ## Profil-Werte in [m]
    # ('rectangle', x_dim, y_dim)
    Rec_15_25 = ('rectangle', 0.15, 0.25)
    Rec_15_35 = ('rectangle', 0.15, 0.35)
    Rec_20_20 = ('rectangle', 0.2, 0.2)
    Rec_20_30 = ('rectangle', 0.2, 0.3)
    Rec_30_20 = ('rectangle', 0.3, 0.2)
    Rec_40_20 = ('rectangle', 0.4, 0.2)
    Rec_40_30 = ('rectangle', 0.4, 0.3)
    # ('circle', radius)
    Circle_20 = ('circle', 0.2)
    # ('i-profile', h, b, t_f, t_w)
    IPE400 = ('i-profile', 0.4, 0.18, 13.5/1000, 8.6/1000)
    HEA400 = ('i-profile', 0.39, 0.3, 19/1000, 11/1000)
    HEB400 = ('i-profile', 0.4, 0.3, 24/1000, 13.5/1000)

    
    @staticmethod
    def create_rectangle_profile(model, x_dim, y_dim):
        """Erstellt ein rechteckiges Profil."""
        return model.createIfcRectangleProfileDef(
            ProfileType="AREA",
            ProfileName=None,
            Position=model.createIfcAxis2Placement2D(
                Location=model.createIfcCartesianPoint((0.0, 0.0))
            ),
            XDim=x_dim,
            YDim=y_dim
        )
    

    @staticmethod
    def create_circle_profile(model, radius):
        """Erstellt ein kreisförmiges Profil."""
        return model.createIfcCircleProfileDef(
            ProfileType="AREA",
            ProfileName=None,
            Position=model.createIfcAxis2Placement2D(
                Location=model.createIfcCartesianPoint((0.0, 0.0))
            ),
            Radius=radius
        )
    

    @staticmethod
    def create_i_profile(model, h, b, t_f, t_w):
        """Erstellt ein I-Profil."""
        return model.createIfcIShapeProfileDef(
            ProfileType="AREA",
            ProfileName=None,
            Position=model.createIfcAxis2Placement2D(
                Location=model.createIfcCartesianPoint((0.0, 0.0))
            ),
            OverallDepth=h,
            OverallWidth=b,
            FlangeThickness=t_f,
            WebThickness=t_w
        )


class IfcFactory:
    """Klasse zur Erzeugung von IFC-Dateien"""
    @staticmethod
    def generate_guid():
        """Funktion zur Erstellung einer IFC-stabilen GUID."""
        return ifcopenshell.guid.new()
    
    
    @staticmethod
    def create_ifc_slab(element, context, model, slab, profile):
        """Funktion zur Erstellung einer IFC-Slab"""
        ## Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea = profile,
            Position = model.createIfcAxis2Placement3D(
                Location = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection=model.createIfcDirection((0.0, 0.0, -1.0)),
            Depth=element['thickness']
        )

        ## Erstelle IfcProduct und bestimme geometrische Repräsentation
        slab.Representation = model.createIfcProductDefinitionShape(
            Representations = [model.createIfcShapeRepresentation(
                ContextOfItems=context,
                RepresentationIdentifier="Body",
                RepresentationType="SweptSolid",
                Items=[extrusion]
            )]
        )

        ## Erstelle IFC-Objekt
        position = element['position']
        element_position = tuple(float(coord) for coord in position)

        slab.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement = model.createIfcAxis2Placement3D(
                Location = model.createIfcCartesianPoint(element_position),
                Axis=model.createIfcDirection((0.0, 0.0, 1.0)),  
                RefDirection=model.createIfcDirection((float(np.cos(np.radians(element['rotation']))), 
                                                       float(np.sin(np.radians(element['rotation']))), 
                                                       0.0))
            )
        )
    

    @staticmethod
    def create_ifc_column(element, context, model, column, profile,):
        """Funktion zur Erstellung einer IFC-Column"""
        ## Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea = profile,
            Position = model.createIfcAxis2Placement3D(
                Location = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection = model.createIfcDirection((0.0, 0.0, 1.0)),
            Depth = element['height']
        )

        ## Erstelle IfcProduct und bestimme geometrische Repräsentation
        column.Representation = model.createIfcProductDefinitionShape(
            Representations = [model.createIfcShapeRepresentation(
                ContextOfItems=context,
                RepresentationIdentifier="Body",
                RepresentationType="SweptSolid",
                Items=[extrusion]
            )]
        )

        ## Erstelle IFC-Objekt
        position = element['position']
        element_position = tuple(float(coord) for coord in position)

        column.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement = model.createIfcAxis2Placement3D(
                Location=model.createIfcCartesianPoint(element_position),
                Axis=model.createIfcDirection((0.0, 0.0, 1.0)),  
                RefDirection=model.createIfcDirection((float(np.cos(np.radians(element['rotation']))), 
                                                       float(np.sin(np.radians(element['rotation']))), 
                                                       0.0))
            )
        )


    @staticmethod
    def add_pset_slab_common(model, slab, Status = "New", AcousticRating = "N/A", FireRating = "N/A", PitchAngle: float = 0, Combustible: bool = False, SurfaceSpreadOfFlame = "N/A", Compartmentation = "N/A",
                             IsExternal: bool = False, ThermalTransmittance: float = 0, LoadBearing: bool = True):
        """Funktion zum Hinzufügen eines Property-Sets"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = slab, name = "Pset_SlabCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={
            "Status": Status,
            "AcousticRating": AcousticRating,
            "FireRating": FireRating,
            "PitchAngle": PitchAngle,
            "Combustible": Combustible,
            "SurfaceSpreadOfFlame": SurfaceSpreadOfFlame,
            "Compartmentation": Compartmentation,
            "IsExternal": IsExternal,
            "ThermalTransmittance": ThermalTransmittance,
            "LoadBearing": LoadBearing            
        })

        
    @staticmethod
    def add_pset_column_common(model, column, Status = "New", Slope: float = 0, Roll: float = 0, 
                               IsExternal: bool = False, ThermalTransmittance: float = 0, LoadBearing: bool = True, FireRating = "N/A"):
        """Funktion zum Hinzufügen eines Property-Sets"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = column, name = "Pset_ColumnCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
            "Status": Status,
            "Slope": Slope,
            "Roll": Roll,
            "IsExternal": IsExternal,
            "ThermalTransmittance": ThermalTransmittance,
            "LoadBearing": LoadBearing,
            "FireRating": FireRating  
        })
    

    @staticmethod
    def add_pset_ErrorInfo(model, element, attributes):
        """Funktion zum Hinzufügen eines Property-Sets"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = element, name = "Pset_ErrorInfo")
        # print("pset_Error: ", attributes) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        if attributes['type'] == 'slab':
            ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
                "floor": attributes['floor'],
                "modify_x": attributes['modify_x'],
                "modify_y": attributes['modify_y'],
            })
        elif attributes['type'] == 'column':
            ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
                "floor": attributes['floor'],
                "modify_heigth": attributes['modify_heigth'],
                "position_type": attributes['position_type'],
            })
        else:
            raise ValueError(f"Unbekannter Typ von element in building_geometry: {element['type']}")
    
    
    def generate_ifc_model(self, parameters, building_geometry, index):
        """Funktion zum Erstellen eines IFC-Modells"""
        ## Initialisiere IFC-Datei und Projektstruktur
        model = ifcopenshell.api.run("project.create_file")

        ## Entitäten erstellen
        project = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcProject", name = "Building Project")
        site = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcSite", name = "Building Site")
        site.GlobalId = self.generate_guid()
        building = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcBuilding", name = "Building")
        building.GlobalId = self.generate_guid()
        context = ifcopenshell.api.context.add_context(model, context_type = "Model", context_identifier = "Building")

        ##  Hierarchien herstellen
        ifcopenshell.api.run("aggregate.assign_object", model, products = [site], relating_object = project)
        ifcopenshell.api.run("aggregate.assign_object", model, products = [building], relating_object = site)
        ifcopenshell.api.run("aggregate.assign_object", model, products = [building], relating_object = site)


        ## Initialisiere Geschosse
        storeys = {}
        for floor in range(-1, parameters.floors):
            if floor == -1:
                floor_name = "Fundmanet"
            elif floor == 0:
                floor_name = "EG"
            else:
                floor_name = f"OG{floor}"            
            storey = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcBuildingStorey", name = floor_name)
            storey.GlobalId = self.generate_guid()
            ifcopenshell.api.aggregate.assign_object(model, products = [storey], relating_object = building)
            storeys[floor] = storey
                
        ## Initialisiere Materialien
        materials = {
            'Beton_C3037': ifcopenshell.api.run("material.add_material", model, name = "C30/37", category = "Concrete"),
            'Stahl_S235': ifcopenshell.api.run("material.add_material", model, name = "S235", category = "Steel"),
            'Stahl_S355': ifcopenshell.api.run("material.add_material", model, name = "S355", category = "Steel")
        }
        

        ## Initialisiere Bauteile       
        for element in building_geometry:
            storey = storeys[element['floor']]

            if element['type'] == 'slab':
                # print("Element[Slab]: ", element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                slab = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcSlab", name= element['name'])
                slab.GlobalId = self.generate_guid()
                ifcopenshell.api.spatial.assign_container(model, products= [slab], relating_structure=storey)
                ifcopenshell.api.material.assign_material(model, products= [slab], material = materials['Beton_C3037'])
                
                ## Erstelle Grundfläche
                profile_type, profile_params = 'rectangle', element['extent']                              
                x_dim, y_dim = profile_params
                profile = ProfileFactory.create_rectangle_profile(model, x_dim, y_dim)
                
                ## Erstelle die Geschossdecke
                self.create_ifc_slab(element, context, model, slab, profile)
                self.add_pset_slab_common(model= model, slab= slab)
                self.add_pset_ErrorInfo(model= model, element= slab, attributes= element)
            
            elif element['type'] == 'column':
                # print("Element[Column]: ", element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                column = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcColumn", name = element['name'])
                column.GlobalId = self.generate_guid()
                ifcopenshell.api.spatial.assign_container(model, products= [column], relating_structure=storey)
                ifcopenshell.api.material.assign_material(model, products= [column], material= materials[element['material']])

                ## Erstelle die Grundfläche
                profile_type, *profile_params = element['profile']
                if profile_type == 'rectangle':
                    profile = ProfileFactory.create_rectangle_profile(model, *profile_params)
                elif profile_type == 'circle':
                    profile = ProfileFactory.create_circle_profile(model, *profile_params)
                elif profile_type == 'i-profile':
                    profile = ProfileFactory.create_i_profile(model, *profile_params)
                else:
                    raise ValueError(f"Unbekannter Profiltyp: {profile_type}")
                
                ## Erstelle die Stütze
                self.create_ifc_column(element, context, model, column, profile)
                self.add_pset_column_common(model= model, column= column)
                self.add_pset_ErrorInfo(model= model, element= column, attributes= element)
            else:
                raise ValueError(f"Unbekannter Typ von element in building_geometry: {element['type']}")
        return model


"""5. Export Daten"""
def get_folder_path():
    """Ermittelt aktuellen absoluten Pfad des Python-Skriptes"""
    file_path = os.path.realpath(__file__) # Gibt absoluten Pfad zum Python-Skript
    folder_path = os.path.dirname(file_path) # Gibt absoluten Pfad zum lokalen Ordner
    return folder_path


def export_parametersets(paramameters_dict, folder_path, index: int):
    """Funktion zum Exportieren des Parametersets"""
    folder_name = "Parameterset"
    folder_dir = os.path.join(folder_path, folder_name)
    os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

    file_name = f"Parameter_Index_{index:03d}"
    text_file_path = os.path.join(folder_dir, file_name + ".txt")
    json_file_path = os.path.join(folder_dir, file_name + ".json")

    ## Speichere Parameter als .txt
    with open(text_file_path, 'w') as txt_file:
        txt_file.write(f"Parameterset für Index {index}:\n")
        for key, value in paramameters_dict.items():
            txt_file.write(f"{key}: {value}\n")
    
    ## Speichere Parameter als .json
    with open(json_file_path, 'w') as json_file:
        json.dump(paramameters_dict, json_file, indent=4)    
    print(f"__Export Parameterset erfolgreich für Index {index} als .txt und .json")


def export_building_geometry(geometry, folder_path, index: int):
    """Funktion zum Exportieren der Geometrie"""
    folder_name = "BuildingGeometry"
    folder_dir = os.path.join(folder_path, folder_name)
    os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

    file_name = f"Geometry_Index_{index:03d}"
    text_file_path = os.path.join(folder_dir, file_name + ".txt")
    json_file_path = os.path.join(folder_dir, file_name + ".json")

    ## Speichere Parameter als .txt
    with open(text_file_path, 'w') as txt_file:
        txt_file.write(f"Gebäudegeometrie für Index {index}:\n")
        for element in geometry:
            for key, value in element.items():
                txt_file.write(f"{key}: {value}\n")
            txt_file.write("\n")
    
    ## Speichere Parameter als .json
    with open(json_file_path, 'w') as json_file:
        json.dump(geometry, json_file, indent=4)
    print(f"__Export Gebäudegeometrie erfolgreich für Index {index} als .txt und .json")


def export_ifc_model(folder_path, index_list):
    """Funktion zum Exportieren von IFC-Dateien bestimmter Kombinationen basierend auf ihrem Index"""
    if index_list:
        print(f"\n __ Start Export für spezifische IFC-Modelle. Indizes: {index_list}")
    else:
        index_list = range(len(param_combination))
        print(f"\n __ Start Export für alle IFC-Modelle. Anzahl an Parametersets: {len(index_list)}")
    
    for index in index_list:
        print(f"\n __Start Export IFC-Modell. Aktueller Index: {index}")

        ## Erhalte Parameterset basierend auf Index
        parameters = param_combination[index]
        parameters_dict = vars(parameters) # vars() wandelt params in Dict um: {'grid_x_field': 4, 'grid_y_field': 2, 'grid_x_size': 5.0, 'grid_y_size': 5.0, 'floors': 2, ...}
        # paramameters_dict = params.__dict__ # Alternative zur Umwandlung in ein Dictionary
        # print(parameters) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print(parameters_dict) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ## Generiere Geometrie
        building_geometry = BuildingGeometry(parameters)
        grid_points = building_geometry.create_grid()
        # print("erstelltes Grid:", grid_points) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        geometry = building_geometry.generate_geometry(index)
        # print(geometry) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Generiere IFC-Modell
        ifc_factory = IfcFactory()
        ifc_model = ifc_factory.generate_ifc_model(parameters, geometry, index)

        ## Überprüfe und erstelle Ordnerpfad für IFC-Dateien
        folder_name = (
            f"{len(parameters_dict)} - "
            f"Grid_{min(param_values['grid_x_n'])}x{min(param_values['grid_y_n'])}_"
            f"{max(param_values['grid_x_n'])}x{max(param_values['grid_y_n'])} - "
            f"Size_{min(param_values['grid_x_size'])}x{min(param_values['grid_y_size'])}_"
            f"{max(param_values['grid_x_size'])}x{max(param_values['grid_y_size'])} - "
            f"Floors_{min(param_values['floors'])}_{max(param_values['floors'])} - "
            f"FH_{min(param_values['floor_height'])}_{max(param_values['floor_height'])} - "
            # f"ST_{min(param_values['slab_thickness'])}_{max(param_values['slab_thickness'])} - "
            # f"FT_{min(param_values['foundation_thickness'])}_{max(param_values['foundation_thickness'])} - "
            f"CC{''.join(map(str, sorted(set(int(c) for c in param_values['corner_columns']))))}_"
            f"EC{''.join(map(str, sorted(set(int(c) for c in param_values['edge_columns']))))}_"
            f"IC{''.join(map(str, sorted(set(int(c) for c in param_values['inner_columns']))))}"
        )
        folder_dir = os.path.join(folder_path, folder_name)
        os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

        file_name = f"Parametrisches_Modell Index_{index:05d}"
        suffix = ".ifc"

        full_path = os.path.join(folder_dir, file_name + suffix)

        ## Exportiere Parameterset
        export_parametersets(parameters_dict, folder_dir, index)

        ## Exportiere Gebäudegeometrie
        export_building_geometry(geometry, folder_dir, index)

        ## Exportiere IFC-Datei
        ifc_model.write(full_path)
        print(f"__Export IFC-Modell erfolgreich für {index} in Pfad: {full_path}")





if __name__ == "__main__":
    print("___Intitialisiere Skript___")

    main()

    print("___Skript erfolgreich beendet___")


    
"""
TODO
- Rotation vervollständigen, sodass sich Bauteile dem Winkel mitdrehen

"""


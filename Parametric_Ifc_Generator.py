# Für Schritt 2 (Parameterkombination)
import itertools
# Für Schritt 3 (Geometrieerstellung)
import numpy as np
# Für Schritt 4 (IFC-Erstellung)
import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
# Für Schritt 5 (Export)
import os
import json



"""1. Definition Parameter für das Bauwerk"""
class BuildingParameters:
    """Klasse zum Speichern der Parameter für ein Bauwerk"""
    def __init__(self, grid_x_n, grid_y_n, grid_x_size, grid_y_size,
                 floors, floor_height, slab_thickness, foundation_thickness,
                 corner_columns, edge_columns, inner_columns, column_profile):
        # Parameter 2D-Gitter
        self.grid_x_n = grid_x_n                  # Anzahl an Feldern in x-Richtung
        self.grid_y_n = grid_y_n                  # Anzahl an Feldern in y-Richtung
        self.grid_x_size = grid_x_size            # Ausdehnung einer Gitterzelle in x-Richtung
        self.grid_y_size = grid_y_size            # Ausdehnung einer Gitterzelle in y-Richtung
        # Parameter Geschoss
        self.floors = floors                      # Anzahl an Geschossen
        self.floor_height = floor_height          # Geschosshöhe (OKRD bis OKRD)
        # Parameter Bauteil - Platten
        self.slab_thickness = slab_thickness      # Deckendicke
        self.foundation_thickness = foundation_thickness  # Dicke der Bodenplatte
        # Parameter Bauteil - Stützen
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
def generate_parameter_combination(param_values):
    ## Erklärung Syntax
    # param_values.items() gibt aus Dictionary Liste von Tupeln wie etwa: [('grid_x_n', [3, 4, 5]), ('grid_y_n', [2, 4, 5]), ...] zurück.
    # Das Sternchen * entpackt Werte als separate Argumente
    # zip-Funktion gruppiert die entpackten Argumente aus den Tupeln in keys und values
    keys, values = zip(*param_values.items())
    combinations = [BuildingParameters(*combo) for combo in itertools.product(*values)]
    return combinations

def Beispiel_Param_Kombi_Indices():
    print(f"_Beispiel Indices der Parametersets: 0 bis {len(param_combination)-1}")

## Beispiel: Ausgabe der ersten n Parametersets
def Beispiel_Param_Kombi_Set(max_index):
    for index, param_set in enumerate(param_combination[:int(max_index)]):
        print(f"_Beipsiel Param_Set {index}: {param_set}")



"""3. Generierung Geometrie"""
class BuildingGeometry:
    """Klasse zum erstellen der Gebäudegeometrie"""
    def __init__(self, parameters):
        self.parameters = parameters
        self.grid_points = None
        self.geometry = []
    

    def create_grid(self):
        x_positions = np.linspace(0, self.parameters.grid_x_size * self.parameters.grid_x_n, self.parameters.grid_x_n + 1)
        y_positions = np.linspace(0, self.parameters.grid_y_size * self.parameters.grid_y_n, self.parameters.grid_y_n + 1)
        self.grid_points = np.array(np.meshgrid(x_positions, y_positions)).T.reshape(-1, 2) #reshape(-1 = Länge der ersten Dimension wird angepasst, 2 = Jede Zeile enthält 2 Werte)
        return self.grid_points
    

    def create_foundation(self):
        extent_x = self.parameters.grid_x_size * self.parameters.grid_x_n
        extent_y = self.parameters.grid_y_size * self.parameters.grid_y_n

        foundations = [] # WIP, # Soll neben Bodenplatte auch Streifenfundamente an den Kanten erstellen. Streifenfundamente = strip foundation
        foundation = {
            'name': 'Bodenplatte',
            'type': 'slab',
            'position': (self.grid_points[0][0] + extent_x/2, self.grid_points[0][1] + extent_y/2, 0), # Oberkante Bodenplatte bei z=0
            'floor': -1,
            'extent': (extent_x, extent_y),
            'thickness': self.parameters.foundation_thickness,
            'is_foundation': True # Kennzeichnung für Fundamentgeschoss
        }
        return foundation
    

    def create_slab(self, z, floor, floor_name):
        extent_x = self.parameters.grid_x_size * self.parameters.grid_x_n
        extent_y = self.parameters.grid_y_size * self.parameters.grid_y_n

        slab = {
            'name': f'Decke {floor_name}',
            'type': 'slab',
            'position': (self.grid_points[0][0] + extent_x/2, self.grid_points[0][1] + extent_y/2, z + self.parameters.floor_height),
            'floor': floor,
            'extent': (extent_x, extent_y),
            'thickness': self.parameters.slab_thickness,
            'is_foundation': False # Kennzeichnung für Fundamentgeschoss
        }
        return slab
    
    
    def create_columns(self, z, floor, floor_name, column_profile):
        if self.grid_points is None:
            raise ValueError("Grid must be created before columns.")
        
        columns = []
        grid_x_n = self.parameters.grid_x_n
        grid_y_n = self.parameters.grid_y_n

        for i, (x, y) in enumerate(self.grid_points):
            is_corner = (i in [0,                                       # erster Index (Punkt) im Raster (unten links)
                               grid_y_n,                                # letzter Punkt in erster Reihe (unten rechts)
                               len(self.grid_points) - 1 - grid_y_n,    # erster Punkt in letzter Reihe (oben links)
                               len(self.grid_points) - 1])              # letzter Index (Punkt) im Raster (oben rechts)
            
            is_edge = (i % grid_x_n == 0 or                             # Punkte in erster Spalte (links)
                       i % grid_x_n == grid_x_n -1 or                   # Punkte in letzter Spalte (rechts)
                       i < grid_x_n or                                  # Punkte in erster Reihe (unten)
                       i >= len(self.grid_points) - grid_x_n)           # Punkte in letzter Reihe (oben)
                                    
            column_data = i, x, y, z, floor, floor_name, column_profile

            if is_corner and self.parameters.corner_columns:
                position_type = 'corner'
                columns.append(self.add_column_data(*column_data, position_type))
            elif not is_corner and is_edge and self.parameters.edge_columns:
                position_type = 'edge'
                columns.append(self.add_column_data(*column_data, position_type))
            elif not is_corner and not is_edge and self.parameters.inner_columns:
                position_type = 'inner'
                columns.append(self.add_column_data(*column_data, position_type))
        return columns
    
    def add_column_data(self, i, x, y, z, floor, floor_name, column_profile, position_type):
        return{'name': f'Stütze {floor_name} {i:02d}', 
               'type': 'column', 
               'position_type': position_type, 
               'position': (x, y, z), 
               'floor': floor, 
               'profile': column_profile,
               'height': self.parameters.floor_height - self.parameters.slab_thickness
        }
    

    def generate_geometry(self):
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
            slab = self.create_slab(z, current_floor, floor_name)
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



"""4. Generierung IFC-Modell"""
class ProfileFactory:
    """Klasse zur Erzeugung von Querschnittsprofilen"""
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
        """Erstellt eine IFC-stabile GUID."""
        return ifcopenshell.guid.new()
    
    
    @staticmethod
    def create_ifc_slab(element, context, model, slab, profile):
        # Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea = profile,
            Position = model.createIfcAxis2Placement3D(
                Location = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection=model.createIfcDirection((0.0, 0.0, -1.0)),
            Depth=element['thickness']
        )

        # Erstelle IfcProduct und bestimme geometrische Repräsentation
        slab.Representation = model.createIfcProductDefinitionShape(
            Representations = [model.createIfcShapeRepresentation(
                ContextOfItems=context,
                RepresentationIdentifier="Body",
                RepresentationType="SweptSolid",
                Items=[extrusion]
            )]
        )

        # Erstelle IFC-Objekt
        position = element['position']
        element_position = tuple(float(coord) for coord in position)

        slab.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement = model.createIfcAxis2Placement3D(
                Location = model.createIfcCartesianPoint(element_position)
            )
        )
    

    @staticmethod
    def create_ifc_column(element, context, model, column, profile):
        # Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea = profile,
            Position = model.createIfcAxis2Placement3D(
                Location = model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection = model.createIfcDirection((0.0, 0.0, 1.0)),
            Depth = element['height']
        )

        # Erstelle IfcProduct und bestimme geometrische Repräsentation
        column.Representation = model.createIfcProductDefinitionShape(
            Representations = [model.createIfcShapeRepresentation(
                ContextOfItems=context,
                RepresentationIdentifier="Body",
                RepresentationType="SweptSolid",
                Items=[extrusion]
            )]
        )

        # Erstelle IFC-Objekt
        position = element['position']
        element_position = tuple(float(coord) for coord in position)

        column.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement = model.createIfcAxis2Placement3D(
                Location=model.createIfcCartesianPoint(element_position)
            )
        )


    @staticmethod
    def add_pset_slab_common(model, slab, Status = "New", AcousticRating = "N/A", FireRating = "N/A", PitchAngle: float = 0, Combustible: bool = False, SurfaceSpreadOfFlame = "N/A", Compartmentation = "N/A",
                             IsExternal: bool = False, ThermalTransmittance: float = 0, LoadBearing: bool = True):
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

    
    # @staticmethod
    # def add_pset_column_common(model, column, Status = "New", Slope: float = 0, Roll: float = 0,
    #                            IsExternal: bool = False, ThermalTransmittance = 2):
    #     pset = ifcopenshell.api.run("pset.add_pset", model, product = column, name = "Pset_ColumnCommon")
    #     ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={
    #         "Status": Status,
    #         "Slope": Slope,
    #         "Roll": Roll,
    #         "IsExternal": IsExternal,
    #         "ThermalTransmittance": ThermalTransmittance,
    #         # "LoadBearing": LoadBearing,
    #         # "FireRating": FireRating  
    #     })
    
    @staticmethod
    def add_pset_column_common(model, column, Status = "New", Slope: float = 0, Roll: float = 0, 
                               IsExternal: bool = False, ThermalTransmittance: float = 0, LoadBearing: bool = True, FireRating = "N/A"):
        pset = ifcopenshell.api.run("pset.add_pset", model, product = column, name = "Pset_ColumnCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset=pset, properties={
            "Status": Status,
            "Slope": Slope,
            "Roll": Roll,
            "IsExternal": IsExternal,
            "ThermalTransmittance": ThermalTransmittance,
            "LoadBearing": LoadBearing,
            "FireRating": FireRating  
        })
    
    
    
    def generate_ifc_model(self, parameters, building_geometry):
        ## Initialisiere IFC-Datei und Projektstruktur
        model = ifcopenshell.api.run("project.create_file")

        project = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcProject", name = "Building Project")
        context = ifcopenshell.api.run("context.add_context", model, context_type = "Model", context_identifier = "Building")

        site = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcSite", name = "Building Site")
        site.GlobalId = self.generate_guid()
        ifcopenshell.api.run("aggregate.assign_object", model, relating_object = project, product = site)

        building = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcBuilding", name = "Building")
        building.GlobalId = self.generate_guid()
        ifcopenshell.api.run("aggregate.assign_object", model, relating_object = site, product = building)

        ## Initialisiere Geschosse
        storeys = {}
        for floor in range(-1, parameters.floors):
            if floor == -1:
                floor_name = "Fundmanet"
            elif floor == 0:
                floor_name = "EG"
            else:
                floor_name = f"OG{floor}"            
            storey = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcBuildingStorey", name = floor_name)
            storey.GlobalId = self.generate_guid()
            ifcopenshell.api.run("aggregate.assign_object", model, relating_object = building, product = storey)
            storeys[floor] = storey
                
        ## Initialisiere Materialien
        Beton = ifcopenshell.api.run("material.add_material", model, name = "Stahlbeton", category = "Concrete")

        ##Initialisiere Bauteile       
        # for index, object in enumerate(building_geometry):
        #     print(f"_Beipsiel Geometry_Objetct {index:02d}: {object}")

        for element in building_geometry:
            storey = storeys[element['floor']]

            if element['type'] == 'slab':
                slab = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcSlab", name = element['name'])
                slab.GlobalId = self.generate_guid()
                ifcopenshell.api.run("spatial.assign_container", model, product=slab, relating_structure=storey)
                ifcopenshell.api.run("material.assign_material", model, product=slab, material=Beton)
                
                # Erstelle Grundfläche
                profile_type, profile_params = 'rectangle', element['extent']                
                # x_dim, y_dim = (dim/2 for dim in profile_params)                
                x_dim, y_dim = profile_params
                profile = ProfileFactory.create_rectangle_profile(model, x_dim, y_dim)
                # Erstelle die Decke
                self.create_ifc_slab(element, context, model, slab, profile)
                self.add_pset_slab_common(model= model, slab= slab)
            
            elif element['type'] == 'column':
                column = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcColumn", name = element['name'])
                column.GlobalId = self.generate_guid()
                ifcopenshell.api.run("spatial.assign_container", model, product=column, relating_structure=storey)
                ifcopenshell.api.run("material.assign_material", model, product=column, material=Beton)

                # Erstelle die Grundfläche
                profile_type, *profile_params = element['profile']
                if profile_type == 'rectangle':
                    profile = ProfileFactory.create_rectangle_profile(model, *profile_params)
                elif profile_type == 'circle':
                    profile = ProfileFactory.create_circle_profile(model, *profile_params)
                elif profile_type == 'i-profile':
                    profile = ProfileFactory.create_i_profile(model, *profile_params)
                else:
                    raise ValueError(f"Unbekannter Profiltyp: {profile_type}")
                # Erstelle die Stütze
                self.create_ifc_column(element, context, model, column, profile)
                self.add_pset_column_common(model= model, column= column)
            else:
                raise ValueError(f"Unbekannter Typ von element in building_geometry: {element['type']}")
        return model


"""5. Export Daten"""
def get_folder_path():
    file_path = os.path.realpath(__file__) # Gibt absoluten Pfad zum Python-Skript
    folder_path = os.path.dirname(file_path) # Gibt absoluten Pfad zum lokalen Ordner
    return folder_path


def export_parametersets(paramameters_dict, folder_path, index: int):
    folder_name = "Parameterset"
    folder_dir = os.path.join(folder_path, folder_name)
    os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

    file_name = f"Parameter_Index_{index:03d}"
    text_file_path = os.path.join(folder_dir, file_name + ".txt")
    json_file_path = os.path.join(folder_dir, file_name + ".json")

    # Speichere Parameter als .txt
    with open(text_file_path, 'w') as txt_file:
        txt_file.write(f"Parameterset für Index {index}:\n")
        for key, value in paramameters_dict.items():
            txt_file.write(f"{key}: {value}\n")
    
    # Speichere Parameter als .json
    with open(json_file_path, 'w') as json_file:
        json.dump(paramameters_dict, json_file, indent=4)    
    print(f"__Parameter für Index {index} wurden erfolgreich als .txt und .json exportiert.")


def export_building_geometry(geometry, folder_path, index: int):
    folder_name = "BuildingGeometry"
    folder_dir = os.path.join(folder_path, folder_name)
    os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

    file_name = f"Geometry_Index_{index:03d}"
    text_file_path = os.path.join(folder_dir, file_name + ".txt")
    json_file_path = os.path.join(folder_dir, file_name + ".json")


    # Speichere Parameter als .txt
    with open(text_file_path, 'w') as txt_file:
        txt_file.write(f"Gebäudegeometrie für Index {index}:\n")
        for element in geometry:
            for key, value in element.items():
                txt_file.write(f"{key}: {value}\n")
            txt_file.write("\n")
    
    # Speichere Parameter als .json
    with open(json_file_path, 'w') as json_file:
        json.dump(geometry, json_file, indent=4)   
    print(f"Gebäudegeometrie für Index {index} wurden erfolgreich als .txt und .json exportiert.")


def export_ifc_solo_model(folder_path, index: int):
    print(f"__Export IFC-Modell. Aktueller Index: {index}")

    # Erhalte Parameterset basierend auf Index
    parameters = param_combination[index]
    parameters_dict = vars(parameters) # Wandelt params in Dict um: {'grid_x_field': 4, 'grid_y_field': 2, 'grid_x_size': 5.0, 'grid_y_size': 5.0, 'floors': 2, ...}
    # paramameters_dict = params.__dict__ # Alternative zur Umwandlung in ein Dictionary
    print(parameters) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(parameters_dict) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # Generiere Geometrie
    building_geometry = BuildingGeometry(parameters)
    grid_points = building_geometry.create_grid()
    geometry = building_geometry.generate_geometry()
    print(geometry) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Generiere IFC-Modell
    ifc_factory = IfcFactory()
    ifc_model = ifc_factory.generate_ifc_model(parameters, geometry)

    # Überprüfe und erstelle Ordnerpfad für IFC-Dateien
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

    file_name = f"Parametrisches_Modell Index_{index:03d}"
    suffix = ".ifc"

    full_path = os.path.join(folder_dir, file_name + suffix)

    # Exportiere Parameterset
    export_parametersets(parameters_dict, folder_dir, index)

    # Exportiere Gebäudegeometrie
    export_building_geometry(geometry, folder_dir, index)

    # Exportiere IFC-Datei
    ifc_model.write(full_path)
    print(f"__Export erfolgreich in Pfad: {full_path}")

def export_ifc_range_model(folder_path):
    print(f"__Start Export für alle IFC-Modelle. Anzahl an Parametersets: {len(param_combination)}")

    for index in range(len(param_combination)):
        export_ifc_solo_model(folder_path, index)



if __name__ == "__main__":
    print("___Intitialisiere Skript___")

    """1. Definition Parameter für das Bauwerk"""
    param_values = {
        # Parameter 2D-Gitter
        'grid_x_n': [3, 4, 5],
        'grid_y_n': [2, 4, 5],
        'grid_x_size': [5.0, 10.0], # in [m]
        'grid_y_size': [5.0, 7.5], # in [m]
        # Parameter Geschoss
        'floors': [2, 4, 6],
        'floor_height': [3.0], # in [m]
        # Parameter Bauteil - Platten
        'slab_thickness': [0.2], # in [m]
        'foundation_thickness': [0.6],  # in [m]
        # Parameter Bauteil - Stützen
        'corner_columns': [True],
        'edge_columns': [True],
        'inner_columns': [True],
        'column_profile': [('rectangle',0.2, 0.2)] # in [m] --> [('rectangle', x_dim, y_dim)], [('circle', radius)], [('i-profile', h, b, t_f, t_w)]
    }    

    
    """2. Kombination Parameter zu einem Parameterset"""
    ## Generiere alle möglichen Parameterkombinationen
    param_combination = generate_parameter_combination(param_values)

    # Beispiel_Param_Kombi_Indices() # TESTPRINT !!!
    # Beispiel_Param_Kombi_Set(max_index = 6) # TESTPRINT !!!

    
    """3. Generierung Geometrie"""
    # ### Testbereich eröffnet...
    # parameters = param_combination[0] # ESSENZIELL !!!
    # building_geometry = BuildingGeometry(parameters) # ESSENZIELL !!!

    # ## Kleine Tests mit 1 Geschoss
    # current_floor = 0
    # current_z = current_floor * parameters.floor_height
    # floor_name = 'EG'

    # grid_points = building_geometry.create_grid() # ESSENZIELL !!!
    # # print(grid_points), print(grid_points[-1])

    # foundation = building_geometry.create_foundation()
    # # print(foundation)

    # slab = building_geometry.create_slab(current_z, current_floor, floor_name)
    # # print(slab)
    
    # columns = building_geometry.create_columns(current_z, current_floor, floor_name, ['circle', 0.3])
    # # print(columns), print(len(columns))

    # ## Kleine Tests mit allen Geschossen
    # geometry = building_geometry.generate_geometry() # ESSENZIELL !!!
    # building_geometry.Beispiel_Building_Geometry(max_index = 20) # TESTPRINT !!!
    # ### ...Testbereich geschlossen

    
    """4. Generierung IFC-Modell"""
    # ## Erstelle Ifc-Modell
    # ifc_factory = IfcFactory() # ESSENZIELL
    # ifc_model = ifc_factory.generate_ifc_model(parameters, geometry) # ESSENZIELL

    
    """5. Export Daten"""
    ## Beziehe Dateipfad
    folder_path = get_folder_path()

    # Exportiere einzelnes IFC-Modell
    export_ifc_solo_model(folder_path, index=0)

    ## Exportiere alle Kombinationen für IFC-Modelle
    # export_ifc_range_model(folder_path)


    
"""
TODO
- Attribut LoadBearing den Bauteilen hinzufügen

- Richtungsanpassung der Stützen (Orientierung in in Richtung X und Y)
- Alignment der Stützen je nach Ausrichtung anpassen, sodass sie bündig mit Plattenrand abschließen.

- IFC-typische Materialien definieren (Beton, Stahl, MW, Holz -> Vorbild aus ArchiCAD BIM-Modul wählen)


- Fehler bei Stützenhöhe einbauen, damit nicht alle Stützen die Decke berühren. Höhe soll nur 0.8 oder 0.95 mal die 'height' betragen
- Fehler bei Stützenplatzierung: Einige Stützen sollen über dem Raster hinaus gesetzt werden, die nicht tragen
- Fehler bei Decken und Bodenplatte: Ausdehnung muss über Stützen hinaus gehen bzw. kürzer sein, dann wäre Punkt oben auch abgedeckt
- !!!! Geometrie entsprechend attributieren mit 'label_fullheight' oder 'labelfullextent' use je nach Einbau des Fehlers. !!!


"""

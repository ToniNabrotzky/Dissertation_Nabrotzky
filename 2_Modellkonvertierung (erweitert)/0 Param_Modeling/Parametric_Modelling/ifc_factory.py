import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid
import math
import numpy as np
# import random


class IfcFactory:
    """Speichert auf Basis einer Bauwerksgeometrie, einem Parametersatz sowie einer Gitterstruktur"""
    def __init__(self, grid_points, params, building_geometry): 
        self.grid_points = grid_points
        self.params = params
        self.building_geometry = building_geometry
        # seed = int(params.index) + len(self.grid_points) + len(building_geometry)
        # random.seed(seed)
        return
    
    """Hauptfunktion"""
    def generate_ifc_model(self):
        """Erzeugt für angegebene Geometrie mithilfe des Parametersatzes und der Gitterstruktur eine Repräsentation im IFC-Schema"""
        print("___Starte Generierung des IFC-Modells") # Demo
        ### Initialisiere IFC-Datei und Projektstruktur
        model = ifcopenshell.api.run("project.create_file")

        ## Entitäten erstellen
        project = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcProject", name = "Building Project")
        site = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcSite", name = "Building Site")
        site.GlobalId = self.generate_guid()
        building = ifcopenshell.api.root.create_entity(model, ifc_class = "IfcBuilding", name = "Building")
        building.GlobalId = self.generate_guid()
        context = ifcopenshell.api.context.add_context(model, context_type = "Model", context_identifier = "Building")

        ## Definition der Einheiten
        length_unit = model.createIfcSIUnit(UnitType="LENGTHUNIT", Name="METRE")
        area_unit = model.createIfcSIUnit(UnitType="AREAUNIT", Name="SQUARE_METRE")
        volume_unit = model.createIfcSIUnit(UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
        unit_assignment = model.createIfcUnitAssignment([length_unit, area_unit, volume_unit])
        model.by_type("IfcProject")[0].UnitsInContext = unit_assignment

        ##  Hierarchien herstellen
        ifcopenshell.api.run("aggregate.assign_object", model, products = [site], relating_object = project)
        ifcopenshell.api.run("aggregate.assign_object", model, products = [building], relating_object = site)
        ifcopenshell.api.run("aggregate.assign_object", model, products = [building], relating_object = site)

        ## Initialisiere Geschosse
        storeys = {}
        for floor in range(-1, self.params.floors):
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
            'Holz_GL24h': ifcopenshell.api.run("material.add_material", model, name = "GL24h", category = "Wood"),
            'Stahl_S235': ifcopenshell.api.run("material.add_material", model, name = "S235", category = "Steel"),
            'Stahl_S355': ifcopenshell.api.run("material.add_material", model, name = "S355", category = "Steel")
        }

        ### Initialisiere Bauteile
        for element in self.building_geometry:
            storey = storeys[element['floor']]
            
            ## Geschossdecken
            if element['type'] == 'slab':
                # print("Element[Slab]: ", element) # Debug
                slab = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcSlab", name= element['name'])
                slab.GlobalId = self.generate_guid()
                ifcopenshell.api.spatial.assign_container(model, products= [slab], relating_structure= storey)
                ifcopenshell.api.material.assign_material(model, products= [slab], material = materials['Beton_C3037'])
                
                ## Erstelle Grundfläche
                # print("element profile: ", element['profile']) # Debug
                x_dim, y_dim = element['profile']['x_dim'], element['profile']['y_dim']
                profile = self.create_rectangle_profile(model, x_dim, y_dim)
                # print("PROFILE: ", id(profile), ",", profile, element['name']) # Debug
                
                ## Erstelle die Geschossdecke
                self.create_ifc_slab(element, context, model, slab, profile)
                self.add_pset_slab_common(model= model, slab= slab)
                # self.add_pset_ErrorInfo(model= model, element= slab, attributes= element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            ## Wände
            elif element['type'] == 'wall':
                # print("Element[Wall]: ", element) # Debug
                wall = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcWallStandardCase", name= element['name'])
                wall.GlobalId = self.generate_guid()
                ifcopenshell.api.spatial.assign_container(model, products= [wall], relating_structure= storey)
                ifcopenshell.api.material.assign_material(model, products= [wall], material = materials['Beton_C3037'])

                ## Erstelle Grundfläche
                x_dim, y_dim = element['length'], element['thickness']
                profile = self.create_rectangle_profile(model, x_dim, y_dim)
                # print("PROFILE: ", id(profile), ",", profile, element['name']) # Debug

                ## Erstelle die Geschossdecke
                self.create_ifc_wall(element, context, model, wall, profile)
                self.add_pset_wall_common(model= model, wall= wall)
                # self.add_pset_ErrorInfo(model= model, element= slab, attributes= element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            ## Balken
            elif element['type'] == 'beam':
                # print("Element[Beam]: ", element) # Debug
                beam = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcBeam", name= element['name'])
                beam.GlobalId = self.generate_guid()
                ifcopenshell.api.spatial.assign_container(model, products= [beam], relating_structure= storey)
                ifcopenshell.api.material.assign_material(model, products= [beam], material = materials['Beton_C3037'])

                ## Erstelle Grundfläche
                beam_profile = element['profile'] # -> Aus Klassenobjekt eine spezifische Variable, welche als Dictionary gespeichert ist
                profile_type = beam_profile['type']

                
                # _Variante_2: Kompaktes Entpacken - Variante_1 bei Column
                params_map = {
                    'rectangle': ['x_dim', 'y_dim'],
                    'circle': ['radius'],
                    'i-profile': ['h', 'b', 't_f', 't_w']
                    }
                
                param_keys = params_map.get(profile_type)

                if param_keys is None:
                    raise ValueError(f"Unbekannter Profiltyp: {profile_type}")
                
                profile_params = [beam_profile[k] for k in param_keys]

                if profile_type == 'rectangle':
                    profile = self.create_rectangle_profile(model, *profile_params)
                elif profile_type == 'circle':
                    profile = self.create_circle_profile(model, *profile_params)
                elif profile_type == 'i-profile':
                    profile = self.create_i_profile(model, *profile_params)
                # print("PROFILE: ", id(profile), ",", profile, element['name']) # Debug

                ## Erstelle die Geschossdecke
                self.create_ifc_beam(element, context, model, beam, profile)
                self.add_pset_beam_common(model= model, beam= beam)
                # self.add_pset_ErrorInfo(model= model, element= slab, attributes= element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            ## Stützen
            elif element['type'] == 'column':
                # print("Element[Column]: ", element) # Debug
                column = ifcopenshell.api.run("root.create_entity", model, ifc_class = "IfcColumn", name = element['name'])
                column.GlobalId = self.generate_guid()
                ifcopenshell.api.spatial.assign_container(model, products= [column], relating_structure= storey)
                ifcopenshell.api.material.assign_material(model, products= [column], material= materials['Beton_C3037'])
                # ifcopenshell.api.material.assign_material(model, products= [column], material= materials[element['material']]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Erstelle die Grundfläche
                column_profile = element['profile'] # -> Aus Klassenobjekt eine spezifische Variable, welche als Dictionary gespeichert ist
                profile_type = column_profile['type']

                # _Variante_1: Explizite Übergabe (Sauber, leichter nachvollziehbar) - Variante_2 bei Beam
                if profile_type == 'rectangle':
                    profile = self.create_rectangle_profile(
                        model,
                        column_profile['x_dim'],
                        column_profile['y_dim']
                    )

                elif profile_type == 'circle':
                    profile = self.create_circle_profile(
                        model,
                        column_profile['radius']
                    )

                elif profile_type == 'i-profile':
                    profile = self.create_i_profile(
                        model,
                        column_profile['h'],
                        column_profile['b'],
                        column_profile['t_f'],
                        column_profile['t_w']
                    )
                else:
                    raise ValueError(f"Unbekannter Profiltyp: {profile_type}")
                # print("PROFILE: ", id(profile), ",", profile, ",", element['name']) # Debug
                
                ## Erstelle die Stütze
                self.create_ifc_column(element, context, model, column, profile)
                self.add_pset_column_common(model= model, column= column)
            #     self.add_pset_ErrorInfo(model= model, element= column, attributes= element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            ## Unbekannter Bauteiltyp
            # else: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #     raise ValueError(f"Unbekannter Typ von element in building_geometry: {element['type']}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print("___Erfolgreiche Generierung des IFC-Modells") # Demo
        return model


    """Hilfsfunktionen"""
    @staticmethod
    def generate_guid():
        """Funktion zur Erstellung einer IFC-stabilen GUID."""
        return ifcopenshell.guid.new()
    

    ## Bauelemente
    @staticmethod
    def create_ifc_slab(element, context, model, slab, profile):
        """Erzeugt eine IFC-konforme Geschossdecke."""
        ## Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea= profile,
            Position= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection= model.createIfcDirection((0.0, 0.0, -1.0)),
            Depth= element['thickness']
        )

        ## Erstelle geometrische Repräsentation
        slab.Representation = model.createIfcProductDefinitionShape(
            Representations = [model.createIfcShapeRepresentation(
                ContextOfItems=context,
                RepresentationIdentifier="Body",
                RepresentationType="SweptSolid",
                Items=[extrusion]
            )]
        )

        ## Erstelle Positionierung
        element_position = tuple(float(coord) for coord in element['position'])

        rotation = np.radians(element['rotation']) # Grad zu Rad
        ref_dir = (
            float(np.cos(rotation)),
            float(np.sin(rotation)),
            0.0
        )

        slab.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint(element_position),
                Axis= model.createIfcDirection((0.0, 0.0, 1.0)),
                RefDirection= model.createIfcDirection(ref_dir)
            )
        )
        return


    @staticmethod
    def create_ifc_wall(element, context, model, wall, profile):
        """Erzeugt eine IFC-konforme Wand."""
        ## Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea= profile,
            Position= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection= model.createIfcDirection((0.0, 0.0, 1.0)),
            Depth= element['height']
        )

        ## Erstelle geometrische Repräsentation
        wall.Representation = model.createIfcProductDefinitionShape(
            Representations= [model.createIfcShapeRepresentation(
                ContextOfItems= context,
                RepresentationIdentifier= "Body",
                RepresentationType= "SweptSolid",
                Items= [extrusion]
            )]
        )

        ## Erstelle Positionierung
        element_position = tuple(float(coord) for coord in element['position'])

        rotation = np.radians(element['rotation']) # Grad zu Rad
        ref_dir = (
            float(np.cos(rotation)),
            float(np.sin(rotation)),
            0.0
        )

        wall.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint(element_position),
                Axis= model.createIfcDirection((0.0, 0.0, 1.0)),
                RefDirection= model.createIfcDirection(ref_dir)
            )
        )
        return
    

    @staticmethod
    def create_ifc_beam(element, context, model, beam, profile):
        """Erzeugt einen IFC-konformen Balken."""
        axis = element['axis'] # z.B. 'x' oder 'y'
        grid_rotation = element['grid_rotation'] # z.B. 45.0 in [°]

        ## Basisrichtung bestimmen
        if axis == 'x':
            base_x, base_y = 1.0, 0.0 # --> Basisvektor für Bauteile in x-Richtung orientiert
        else:
            base_x, base_y = 0.0, 1.0 # --> Basisvektor für Bauteile in y-Richtung orientiert
        
        ## Drehung um Z-Achse
        angle_rad = math.radians(grid_rotation) # Grad in Rad    
        ex_x = base_x * math.cos(angle_rad) - base_y * math.sin(angle_rad)
        ex_y = base_x * math.sin(angle_rad) + base_y * math.cos(angle_rad)

        ## Extrusionsrichtung bestimmen
        extrusion_dir = (ex_x, ex_y, 0.0) # lokale Z-Achse

        ## Referenzrichtung bestimmen (90° CCW zu Extrusionsrichtung in XY)
        ref_x = -ex_y
        ref_y = ex_x
        ref_dir = (ref_x, ref_y, 0.0)

        """neue Variante"""
        # print("extrusion_dir: ", element['extrusion_dir'][0], element['extrusion_dir'][1], element['extrusion_dir'][2]) # Debug
        # print("ref_dir: ", element['ref_dir']) # Debug
        ## Erstelle Positionierung
        element_position = tuple(float(coord) for coord in element['position'])
        # extrusion_dir = element['extrusion_dir'] # Ausrichtung des Elementes (lokale Z-Achse)
        # x1 = element['extrusion_dir'][0]
        # x0 = element['extrusion_dir'][1]
        # ex = (x1, x0, 0.0)
        # # extrusion_dir = (element['extrusion_dir'][0], element['extrusion_dir'][1], element['extrusion_dir'][2])  # Ausrichtung des Elementes (lokale Z-Achse)
        # extrusion_dir = ex  # Ausrichtung des Elementes (lokale Z-Achse)
        # # extrusion_dir = (x1, x0, x0)  # Ausrichtung des Elementes (lokale Z-Achse)
        # # extrusion_dir = (1.0, 0.0, 0.0)  # Ausrichtung des Elementes (lokale Z-Achse)
        # # ref_dir = element['ref_dir'] # Ausrichtung des Elementes (lokale X-Achse)
        # # ref_dir = (element['ref_dir'][0], element['ref_dir'][1], element['ref_dir'][2]) # Ausrichtung des Elementes (lokale X-Achse)
        # ref_dir = (0.0, 1.0, 0.0) # Ausrichtung des Elementes (lokale X-Achse)


        ## Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea= profile,
            Position= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection= model.createIfcDirection((0.0, 0.0, 1.0)),
            Depth= element['length']
        )


        ## Erstelle geometrische Repräsentation
        beam.Representation = model.createIfcProductDefinitionShape(
            Representations= [model.createIfcShapeRepresentation(
                ContextOfItems= context,
                RepresentationIdentifier= "Body",
                RepresentationType= "SweptSolid",
                Items= [extrusion]
            )]
        )

        ## Erstelle Objekt-Positionierung
        beam.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint(element_position),
                Axis= model.createIfcDirection(extrusion_dir),
                RefDirection= model.createIfcDirection(ref_dir)
            )
        )


        """alte Variante"""
        # ## Erstelle Extrusion
        # extrusion = model.createIfcExtrudedAreaSolid(
        #     SweptArea= profile,
        #     Position= model.createIfcAxis2Placement3D(
        #         Location= model.createIfcCartesianPoint((0.0, 0.0, 0.0))
        #     ),
        #     ExtrudedDirection= model.createIfcDirection((0.0, 0.0, 1.0)),
        #     Depth= element['length']
        # )

        # ## Erstelle geometrische Repräsentation
        # beam.Representation = model.createIfcProductDefinitionShape(
        #     Representations= [model.createIfcShapeRepresentation(
        #         ContextOfItems= context,
        #         RepresentationIdentifier= "Body",
        #         RepresentationType= "SweptSolid",
        #         Items= [extrusion]
        #     )]
        # )

        # ## Erstelle Positionierung
        # element_position = tuple(float(coord) for coord in element['position'])

        # rotation = np.radians(element['rotation']) # Grad zu Rad
        # ref_dir = (
        #     float(np.cos(rotation)),
        #     float(np.sin(rotation)),
        #     0.0
        # )

        # beam.ObjectPlacement = model.createIfcLocalPlacement(
        #     RelativePlacement= model.createIfcAxis2Placement3D(
        #         Location= model.createIfcCartesianPoint(element_position),
        #         Axis= model.createIfcDirection((0.0, 0.0, 1.0)),
        #         RefDirection= model.createIfcDirection(ref_dir)
        #     )
        # )
        return

    
    @staticmethod
    def create_ifc_column(element, context, model, column, profile):
        """Erzeugt eine IFC-konforme Stütze."""
        ## Erstelle Extrusion
        extrusion = model.createIfcExtrudedAreaSolid(
            SweptArea= profile,
            Position= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint((0.0, 0.0, 0.0))
            ),
            ExtrudedDirection= model.createIfcDirection((0.0, 0.0, 1.0)),
            Depth= element['height']
        )

        ## Erstelle geometrische Repräsentation
        column.Representation= model.createIfcProductDefinitionShape(
            Representations= [model.createIfcShapeRepresentation(
                ContextOfItems= context,
                RepresentationIdentifier= "Body",
                RepresentationType= "SweptSolid",
                Items= [extrusion]
            )]
        )

        ## Erstelle Positionierung
        element_position = tuple(float(coord) for coord in element['position'])

        rotation = np.radians(element['rotation']) # Grad zu Rad
        ref_dir = (
            float(np.cos(rotation)),
            float(np.sin(rotation)),
            0.0
        )

        column.ObjectPlacement = model.createIfcLocalPlacement(
            RelativePlacement= model.createIfcAxis2Placement3D(
                Location= model.createIfcCartesianPoint(element_position),
                Axis= model.createIfcDirection((0.0, 0.0, 1.0)),
                RefDirection= model.createIfcDirection(ref_dir)
            )
        )
        return
    
        
    ## Querschnittsprofile
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
    

    ## P-Sets
    @staticmethod
    def add_pset_slab_common(model, slab, Status= "New", AcousticRating= "N/A", FireRating= "N/A", PitchAngle: float= 0, Combustible: bool= False, SurfaceSpreadOfFlame= "N/A", Compartmentation= "N/A", IsExternal: bool= False, ThermalTransmittance: float= 0, LoadBearing: bool= True):
        """Erstellt ein allgemeines Property-Set für Decken"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = slab, name = "Pset_SlabCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties={
            "Status": Status, # Phase des Bauteils: 'New', 'Existing', 'Demolish', 'Temporary'
            "AcousticRating": AcousticRating, # Schallschutzklasse
            "FireRating": FireRating, # Feuerwiderstandsklasse
            "PitchAngle": PitchAngle, # Neigungswinkel gegenüber der Horizontalen (0 Grad)
            "Combustible": Combustible, # Bool, ob brennbar
            "SurfaceSpreadOfFlame": SurfaceSpreadOfFlame, # Beschreibung Brandverhalten
            "Compartmentation": Compartmentation, # Bool, ob Abgrenzung eines Brandabschnitts
            "IsExternal": IsExternal, # Bool, ob außenliegend
            "ThermalTransmittance": ThermalTransmittance, # Wärmedurchgangskoeffizient (U-Wert)
            "LoadBearing": LoadBearing, # Bool, ob lasttragend
        })
        return
    

    @staticmethod
    def add_pset_wall_common(model, wall, Status= "New", AcousticRating= "N/A", FireRating= "N/A", Combustible: bool= False, SurfaceSpreadOfFlame= "N/A", ThermalTransmittance: float= 0, IsExternal: bool= False, LoadBearing: bool= True, ExtendToStructure= True, Compartmentation= "N/A"):
        """Erstellt ein allgemeines Property-Set für Stützen"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = wall, name = "Pset_WallCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
            "Status": Status, # Phase des Bauteils: 'New', 'Existing', 'Demolish', 'Temporary'
            "AcousticRating": AcousticRating, # Schallschutzklasse
            "FireRating": FireRating, # Feuerwiderstandsklasse
            "Combustible": Combustible, # Bool, ob brennbar
            "SurfaceSpreadOfFlame": SurfaceSpreadOfFlame, # Beschreibung Brandverhalten
            "ThermalTransmittance": ThermalTransmittance, # Wärmedurchgangskoeffizient (U-Wert)
            "IsExternal": IsExternal, # Bool, ob außenliegend
            "LoadBearing": LoadBearing, # Bool, ob lasttragend
            "ExtendToStructure": ExtendToStructure, # Bool, ob Wand raumhoch ist
            "Compartmentation": Compartmentation, # Bool, ob Abgrenzung eines Brandabschnitts
        })
        return
    

    @staticmethod
    def add_pset_beam_common(model, beam, Status= "New", Span= 0, Slope= 0, Roll= 0, IsExternal: bool= False, ThermalTransmittance: float= 0, LoadBearing: bool= True, FireRating= "N/A"):
        """Erstellt ein allgemeines Property-Set für Stützen"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = beam, name = "Pset_BeamCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
            "Status": Status, # Phase des Bauteils: 'New', 'Existing', 'Demolish', 'Temporary'
            "Span": Span, # Lichte Spannweite für statische Anforderung
            "Slope": Slope, # Neigungswinkel gegenüber der Horizontalen (0 Grad)
            "Roll": Roll, # Kippwinkel relativ zur Vertikalen (0 Grad)
            "IsExternal": IsExternal, # Bool, ob außenliegend
            "ThermalTransmittance": ThermalTransmittance, # Wärmedurchgangskoeffizient (U-Wert)
            "LoadBearing": LoadBearing, # Bool, ob lasttragend
            "FireRating": FireRating, # Feuerwiderstandsklasse
        })
        return

        
    @staticmethod
    def add_pset_column_common(model, column, Status= "New", Slope: float= 0, Roll: float= 0, IsExternal: bool= False, ThermalTransmittance: float= 0, LoadBearing: bool= True, FireRating= "N/A"):
        """Erstellt ein allgemeines Property-Set für Stützen"""
        pset = ifcopenshell.api.run("pset.add_pset", model, product = column, name = "Pset_ColumnCommon")
        ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
            "Status": Status, # Phase des Bauteils: 'New', 'Existing', 'Demolish', 'Temporary'
            "Slope": Slope, # Neigungswinkel gegenüber der Horizontalen (0 Grad)
            "Roll": Roll, # Drehwinkel relativ zur globalen X-Ausrichtung (0 Grad)
            "IsExternal": IsExternal, # Bool, ob außenliegend
            "ThermalTransmittance": ThermalTransmittance, # Wärmedurchgangskoeffizient (U-Wert)
            "LoadBearing": LoadBearing, # Bool, ob lasttragend
            "FireRating": FireRating, # Feuerwiderstandsklasse
        })
        return
    

    # @staticmethod
    # def add_pset_ErrorInfo(model, element, attributes):
    #     """Funktion zum Hinzufügen eines Property-Sets"""
    #     pset = ifcopenshell.api.run("pset.add_pset", model, product = element, name = "Pset_ErrorInfo")
    #     # print("pset_Error: ", attributes) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    #     if attributes['type'] == 'slab':
    #         ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
    #             "floor": attributes['floor'],
    #             "modify_x": attributes['modify_x'],
    #             "modify_y": attributes['modify_y'],
    #         })
    #     elif attributes['type'] == 'column':
    #         ifcopenshell.api.run("pset.edit_pset", model, pset= pset, properties= {
    #             "floor": attributes['floor'],
    #             "modify_heigth": attributes['modify_heigth'],
    #             "position_type": attributes['position_type'],
    #         })
    #     else:
    #         raise ValueError(f"Unbekannter Typ von element in building_geometry: {element['type']}")
    
import math
import random

from Parametric_Modelling import parameter_factory


class BuildingGeometry:
    """Speichert Bauwerksgeometrie auf Basis einer Gitterstruktur"""
    def __init__(self, grid_points, params): 
        self.grid_points = grid_points
        self.params = params
        self.elements = []
        seed = int(params.index) + len(self.grid_points)
        random.seed(seed)
        return

    """Hauptfunktion"""
    def generate_gemeotry(self):
        """Erzeugt für alle Geschosse zufällig in X- oder Y-Achse ausgerichtet die Bauelemente (Gründung, Decken, Wände, Balken und Stützen)"""
        print("___Starte Generierung der Bauwerksgeometrie___") # Demo
        ### Erstelle Gründung        
        ## Bodenplatte erzeugen
        print(f"GENERATE_GEOMETRY: Erstelle Gründungsgeometrie (Bodenplatte)") # Demo        
        axis = 'x' 
        floor = 0
        floor_name = 'EG'

        max_x_dim = max(self.params.column_profile['x_dim'], self.params.beam_profile['x_dim'], self.params.wall_thickness)
        max_y_dim = max(self.params.column_profile['y_dim'], self.params.beam_profile['y_dim'], self.params.wall_thickness)

        center_point = next((p for p in self.grid_points if p['pos_type'] == 'center' and p['floor'] == floor), None)
        Px, Py, Pz = center_point['x'], center_point['y'], center_point['z']
        position = (Px, Py, Pz)

        grid_x_size = self.params.grid_x_size
        grid_y_size = self.params.grid_y_size
        grid_x_n = self.params.grid_x_n
        grid_y_n = self.params.grid_y_n
        extent_x = grid_x_size * grid_x_n + max_x_dim
        extent_y = grid_y_size * grid_y_n + max_y_dim
        profile = parameter_factory.ProfileFactory('rectangle', extent_x, extent_y).profile

        attributes = (
            True, 
            floor, floor_name, 
            center_point, position, 
            self.params.grid_rotation, 
            profile, 
            self.params.foundation_thickness
        )
        slab = self.create_slab(*attributes)
        self.elements.append(slab)
        
        
        ## Streifenfundamente erzeugen
        print(f"GENERATE_GEOMETRY: Erstelle Gründungsgeometrie (Streifenfundamente)") # Demo
        for axis in ['y', 'x']: # Erst x-Richtung, dann y-Richtung
            num_rows, direction, floor_name = self.process_rows(axis, floor)           

            for row_num in range(num_rows):
                ## Durchlaufe alle Reihen für ausgewählte Achse
                row_points = [p for p in self.grid_points if p[f'{direction}_row'] == row_num and p['floor'] == floor ]
                print(f"ROW_POINTS: Punkte-IDs im Geschoss {floor} in {axis}-Achse für Reihe {row_num}: ", [point['id'] for point in row_points]) # Demo

                ## Start- und Endpunkt
                p1, p2 = random.sample(row_points, 2) # Start- und Endpunkt eines Streifenfundaments
                p_start, p_end = sorted([p1, p2], key= lambda p: p['id']) # Bringe Punkte in korrekte Reihenfolge
                # foundation_ids = {p_start['id'], p_end['id']}
                # print(f"STREIFENFUNDAMENTPUNKTE für {row_num}: ", p_start['id'], p_end['id']) # Debug
                print(self.params.strip_foundation_profile)

                ## Position des Balkens (Am Startpunkt)
                z_pos = p_start['z']  - self.params.foundation_thickness - self.params.strip_foundation_profile['y_dim'] / 2
                position = (p_start['x'], p_start['y'], z_pos)

                ## Balkenlänge
                length = math.dist((p_start['x'], p_start['y']), (p_end['x'], p_end['y']))
                
                ## Balkenrotation
                dx = p_end['x'] - p_start['x']
                dy = p_end['y'] - p_start['y']
                rotation = math.degrees(math.atan2(dy, dx))  # in °
                grid_rotation = self.params.grid_rotation # in °
                # print("ROT: ", rotation, "GRID_ROT: ", grid_rotation) # Debug

                attributes = (
                            True, axis, 
                            floor, floor_name, 
                            p_start, p_end, 
                            position, 
                            rotation, grid_rotation,
                            self.params.strip_foundation_profile, 
                            length, 
                            row_num, num_rows
                        )
                beam = self.create_beam(*attributes)
                self.elements.append(beam)
        
        
        ### Bauteile erzeugen
        ## Durchlaufe alle Geschosse
        for floor in range(self.params.floors):
            ## Durchlaufe zufällig ausgewählte Achse
            axis = random.choice(['x', 'y'])
            print(f"\nGENERATE_GEOMETRY: ___Erstelle Bauteile für Geschoss {floor} in {'x' if axis == 'y' else 'y'}-Richtung___") # Demo
            num_rows, direction, floor_name = self.process_rows(axis, floor)   
            
            
            ## Geschossdecken erzeugen
            print(f"GENERATE_GEOMETRY: Erstelle Bauteile (Geschossdecke)") # Demo 
            center_point = next((p for p in self.grid_points if p['pos_type'] == 'center' and p['floor'] == floor), None)
            Px, Py, Pz = center_point['x'], center_point['y'], center_point['z'] + self.params.floor_height
            position = (Px, Py, Pz)

            grid_x_size = self.params.grid_x_size
            grid_y_size = self.params.grid_y_size
            grid_x_n = self.params.grid_x_n
            grid_y_n = self.params.grid_y_n
            extent_x = grid_x_size * grid_x_n + max_x_dim
            extent_y = grid_y_size * grid_y_n + max_y_dim
            profile = parameter_factory.ProfileFactory('rectangle', extent_x, extent_y).profile

            attributes = (
                False, 
                floor, floor_name, 
                center_point, position, 
                self.params.grid_rotation, 
                profile, 
                self.params.slab_thickness
            )
            slab = self.create_slab(*attributes)
            self.elements.append(slab)

            ## Restliche Bauteile erzeugen
            print(f"GENERATE_GEOMETRY: Erstelle Bauteile (Wände, Balken, Stützen)") # Demo 
            for row_num in range(num_rows):
                ## Durchlaufe alle Reihen für ausgewählte Achse
                row_points = [p for p in self.grid_points if p[f'{direction}_row'] == row_num and p['floor'] == floor ]
                print(f"ROW_POINTS: Punkte-IDs im Geschoss {floor} in {axis}-Achse für Reihe {row_num}: ", [point['id'] for point in row_points]) # Demo
                
                
                ## Wände erzeugen
                # print(f"GENERATE_GEOMETRY: Erstelle Wände für Geschoss {floor} in {direction}-Richtung") # Debug
                if not random.random() < self.params.wall_prob:
                    # print(f"WANDPUNKTE für {row_num}: ", False) # Debug
                    wall_ids = set()
                else:
                    ## Start- und Endpunkt
                    p1, p2 = random.sample(row_points, 2) # Start- und Endpunkt einer Wand
                    p_start, p_end = sorted([p1, p2], key= lambda p: p['id']) # Bringe Pnkte wieder in korrekte Reihenfolge
                    wall_ids = {p_start['id'], p_end['id']}
                    # print(f"WANDPUNKTE für {row_num}: ", p_start['id'], p_end['id']) # Debug

                    ## Position der Wand (Mittelpunkt zwischen Start- und Endpunkt)
                    x_pos = (p_start['x'] + p_end['x']) / 2
                    y_pos = (p_start['y'] + p_end['y']) / 2
                    z_pos = p_start['z']  # gleiche Höhe wie Startpunkt
                    position = (x_pos, y_pos, z_pos)

                    ## Wandlänge
                    length = math.dist((p_start['x'], p_start['y']), (p_end['x'], p_end['y']))
                    dx = p_end['x'] - p_start['x']
                    dy = p_end['y'] - p_start['y']
                    rotation = math.degrees(math.atan2(dy, dx))  # in °
                    
                    attributes = (
                        floor, floor_name, 
                        p_start, p_end, 
                        position, 
                        rotation, 
                        self.params.wall_thickness, 
                        length, 
                        row_num, num_rows
                    )
                    wall = self.create_wall(*attributes)
                    self.elements.append(wall)
                
                
                ## Balken erzeugen
                # print(f"GENERATE_GEOMETRY: Erstelle Balken für Geschoss {floor} in {direction}-Richtung") # Debug
                beam_segments = []
                if wall_ids:
                    # Wenn eine Wand existiert, muss ein Balken nur in den Feldern ohne Wand platziert werden
                    wall_points = [p for p in row_points if p['id'] in wall_ids]
                    wall_start, wall_end = sorted(wall_points, key=lambda p: p['id']) # -> Falls es Probleme gibt wegen Sortierung, dann simpel: = wall_points[0], wall_points[1]
                    # print("WALL_POINT_IDS: ", wall_points) # Debug
                    # print("WALL_START_END: ", wall_start, wall_end) # Debug

                    ## Start- und Endpunkt
                    if wall_start['id'] == row_points[0]['id'] and wall_end['id'] == row_points[-1]['id']:
                        # Wand geht über die ganze Reihe --> es braucht keinen Balken
                        beam_segments = []
                    elif wall_start['id'] == row_points[0]['id']:
                        # Wand beginnt am Reihenanfang --> Balken geht vom Wandende bis zum Reihenende
                        beam_segments.append((wall_end, row_points[-1]))
                    elif wall_end['id'] == row_points[-1]['id']:
                        # Wand geht bis zum Reihenende --> Balken geht vom Reihenanfang bis zum Wandanfang
                        beam_segments.append((row_points[0], wall_start))
                    else:
                        # Wand ist mitten in der Reihe --> Beide vorherigen Optionen treffen zu
                        beam_segments.append((row_points[0], wall_start))
                        beam_segments.append((wall_end, row_points[-1]))
                else:
                    # Es exisitert keine Wand --> Balken geht über ganze Reihe
                    beam_segments.append((row_points[0], row_points[-1]))
                
                for p_start, p_end in beam_segments:
                    ## Final sicherstellen, dass der Balken zwei verschiedene Punkte hat
                    if p_start['id'] != p_end['id']:
                        ## Position des Balkens (Am Startpunkt)
                        z_pos = p_start['z']  + self.params.floor_height - self.params.slab_thickness - self.params.beam_profile['y_dim'] / 2
                        position = (p_start['x'], p_start['y'], z_pos)

                        ## Balkenlänge
                        length = math.dist((p_start['x'], p_start['y']), (p_end['x'], p_end['y']))

                        ## Balkenrotation
                        dx = p_end['x'] - p_start['x']
                        dy = p_end['y'] - p_start['y']
                        rotation = math.degrees(math.atan2(dy, dx))  # in °
                        grid_rotation = self.params.grid_rotation # in °
                        # print("ROT: ", rotation, "GRID_ROT: ", grid_rotation) # Debug

                        attributes = (
                            False, axis, 
                            floor, floor_name, 
                            p_start, p_end, 
                            position, 
                            rotation, grid_rotation,
                            self.params.beam_profile, 
                            length, 
                            row_num, num_rows
                        )
                        beam = self.create_beam(*attributes)
                        self.elements.append(beam)

                
                ## Stützen erzeugen
                # print(f"GENERATE_GEOMETRY: Erstelle Stützen für Geschoss {floor} in {direction}-Richtung") # Debug
                if wall_ids:
                    # Wand vorhanden --> Nur Punkte außerhalb der Wand erlaubt
                    wall_points = [p for p in row_points if p['id'] in wall_ids]
                    wall_start, wall_end = sorted(wall_points, key=lambda p: p['id']) # -> Falls es Probleme gibt wegen Sortierung, dann simpel: = wall_points[0], wall_points[1]
                    allowed_points = [p for p in row_points if p['id'] < wall_start['id'] or p['id'] > wall_end['id']]
                else:
                    # Keine Wand vorhanden --> alle Punkte erlaubt
                    allowed_points = row_points

                for p in allowed_points:
                    if random.random() < self.params.column_prob:
                        attributes = (
                            floor, floor_name, 
                            p, 
                            self.params.grid_rotation, 
                            self.params.column_profile, 
                            row_num, num_rows
                        )
                        column = self.create_column(*attributes)
                        self.elements.append(column)
        
        print("___Erfolgreiche Generierung der Gebäudegeometrie___") # Demo
        return self.elements
    

    """Hilfsfunktionen"""
    def process_rows(self, axis= 'x', floor= 0):
        """Verarbeitet die Reihen entlang einer angegebenen Achsrichtung zur Bauteilerzeugung"""
        ## Bestimme Anzahl an Reihen in Achsenrichtung
        if axis == 'x':
            ## Achsen hier vertauschen, da 'x' nicht als Spaltenindex, sondern als Richtung zu sehen ist.
            num_rows = self.params.grid_y_n + 1
            direction = 'y'
        elif axis == 'y':
            ## Achsen hier vertauschen, da 'y'nicht als Zeilenindex, sondern als Richtung zu sehen ist.
            num_rows = self.params.grid_x_n + 1
            direction = 'x'
        else:
            ValueError("Falsche Achse ausgewählt. Wähle aus 'x' oder 'y'")

        ## Definiere Geschossnamen
        if floor == 0:
                floor_name = 'EG'
        else:
            floor_name = f'OG{floor}'
        
        print(f"PROCESS_ROWS: Parallel zur {axis}-Achse verlaufen im Geschoss '{floor_name}' {num_rows} Reihen") # Demo
        return num_rows, direction, floor_name
    
    
    def create_slab(self, is_foundation, floor, floor_name, p, position, rotation, profile, thickness):
        """Erzeugt die geometrische Repräsentation einer Geschossdecke."""
        return{
            ## semantische Attribute
            'foundation': True if is_foundation else False, # Gründungsbauteil
            'type': 'slab', # Typ des Bauteils
            'name': f"{'Bodenplatte' if is_foundation else 'Decke'} {floor_name if not is_foundation else ''} p_{p['id']}", # Bauteilbezeichnung
            'point_id': p['id'], # interne ID (semantische Identifikation)
            ## geometrische Attribute
            'position': position, # XYZ-Position im Raum
            'modify_position': None, # möglicher Fehlermodifikator
            'rotation': rotation, # Ausrichtung/Rotation des Elementes
            'profile': profile, # Querschnittsprofil
            # 'modify_x': scale_x, # möglicher Fehlermodifikator
            # 'modify_y': scale_y, # möglicher Fehlermodifikator
            'thickness': thickness, # Querschnittsprofil
            ## alphanumerische Attribute
            # 'is_foundation': False # Kennzeichnung für Fundamentgeschoss
            ## topologische Attribute
            'floor': -1 if is_foundation else floor, # Zugehöriges Geschoss
        }
    

    def create_wall(self, floor, floor_name, p_start, p_end, position, rotation, thickness, length, row_num, num_rows):
        """Erzeugt die geometrische Repräsentation einer Wand."""
        if self.params.slab_thickness:
            height = self.params.floor_height - self.params.slab_thickness
        else:
            height = self.params.floor_height - self.params.slab_beam_profile['y_dim'] # Falls Deckebalken eingeführt sind, so korrekt?
            
        return{
            ## semantische Attribute
            'foundation': False, # Gründungsbauteil
            'type': 'wall', # Typ des Bauteils
            'name': f"Wand {floor_name} p_{p_start['id']}-{p_end['id']}", # Bauteilbezeichnung
            'start_id': p_start['id'], # interne ID (semantische Identifikation)
            'end_id': p_end['id'], # interne ID (semantische Identifikation)
            ## geometrische Attribute
            'position': position, # XYZ-Position im Raum
            'modify_position': None, # möglicher Fehlermodifikator
            'rotation': rotation, # Ausrichtung/Rotation des Elementes
            'thickness': thickness, # Querschnittsprofil
            'length': length, # Länge des Bauteils
            'height': height, # Bauteilhöhe
            ## alphanumerische Attribute
            'is_external': (row_num == 0 or row_num == num_rows-1), # Bool ob außenliegendes Bauteil
            ## topologische Attribute
            'floor': floor, # Zugehöriges Geschoss
        }
    

    def create_beam(self, is_foundation, axis_dir, floor, floor_name, p_start, p_end, position, rotation, grid_rotation, profile, length, row_num, num_rows):
        """Erzeugt die geometrische Repräsentation eines Balkens."""
        if is_foundation:
            name = f"Streifenfundament_{axis_dir}"
        else:
            name = f"Balken_{axis_dir} {floor_name}"
        
        return {
            ## semantische Attribute
            'foundation': True if is_foundation else False, # Gründungsbauteil
            'type': 'beam', # Typ des Bauteils
            'name': f"{name} p_{p_start['id']}-{p_end['id']}", # Bauteilbezeichnung
            'start_id': p_start['id'], # interne ID (semantische Identifikation)
            'end_id': p_end['id'], # interne ID (semantische Identifikation)
            'axis': axis_dir,
            ## geometrische Attribute
            'position': position, # XYZ-Position im Raum
            'modify_position': None, # möglicher Fehlermodifikator
            'rotation': rotation, # Rotation des Elementes
            'grid_rotation': grid_rotation, # Rotation des Elementes
            'profile': profile, # Querschnittsprofil
            'length': length, # Länge des Bauteils
            ## alphanumerische Attribute
            'is_external': (row_num == 0 or row_num == num_rows-1), # Bool ob außenliegendes Bauteil
            ## topologische Attribute
            'floor': -1 if is_foundation else floor, # Zugehöriges Geschoss
        }


    def create_column(self, floor, floor_name, p, rotation, profile, row_num, num_rows):
        """Erzeugt die geometrische Repräsentation einer Stütze."""
        if self.params.slab_thickness:
            height = self.params.floor_height - self.params.slab_thickness
        else:
            height = self.params.floor_height - self.params.slab_beam_profile['y_dim'] # Falls Deckebalken eingeführt sind, so korrekt?

        return {
            ## semantische Attribute
            'foundation': False, # Gründungsbauteil
            'type': 'column', # Typ des Bauteils
            'name': f"Stütze {floor_name} p: {p['id']}", # Bauteilbezeichnung
            'pos_type': p['pos_type'], # Positionstyp (z.B. Eckstütze)
            'point_id': p['id'], # interne ID (semantische Identifikation)
            ## geometrische Attribute
            'position': (p['x'], p['y'], p['z']), # XYZ-Position im Raum
            'modify_position': None, # möglicher Fehlermodifikator
            'rotation': rotation, # Ausrichtung/Rotation des Elementes
            'profile': profile, # Querschnittsprofil
            'height': height, # Bauteilhöhe
            'modify_height': None, # möglicher Fehlermodifikator
            ## alphanumerische Attribute
            'is_external': (row_num == 0 or row_num == num_rows-1), # Bool ob außenliegendes Bauteil
            ## topologische Attribute
            'floor': floor, # Zugehöriges Geschoss
        }
    

    """Extra Methoden"""
import json # export-metode - vlt nicht nötig?
import matplotlib.pyplot as plt
import numpy as np
import os # export-methode - vlt nicht nötig?
import random
from mpl_toolkits.mplot3d import Axes3D


class GridFactory:
    """Speichert eine räumliche Gitterstruktur auf Basis eines Parametersets"""
    def __init__(self, params, # index,
                #  grid_x_n, grid_y_n, grid_x_size, grid_y_size, grid_rotation, grid_x_offset, grid_y_offset, # 2D-Gitterstruktur
                #  floors, floor_height
                 ): # 3D-Gitterstruktur
        ## Parameterset
        self.params = params

        ## Index
        self.index = self.params.index

        ## Parameter Gitterstruktur
        self.grid_x_n = self.params.grid_x_n                # Anzahl an Feldern in x-Richtung
        self.grid_y_n = self.params.grid_y_n                # Anzahl an Feldern in y-Richtung
        self.grid_x_size = self.params.grid_x_size          # Ausdehnung einer Gitterzelle in x-Richtung
        self.grid_y_size = self.params.grid_y_size          # Ausdehnung einer Gitterzelle in y-Richtung
        self.rotation = np.radians(self.params.grid_rotation)      # Drehung im mathematisch positiven Sinn; Umrechnung in Radiant
        self.offset_x = self.params.grid_x_offset           # Verschiebung in X-Richtung
        self.offset_y = self.params.grid_y_offset           # Verschiebung in Y-Richtung
        
        ## Parameter Geschoss
        self.floors = self.params.floors                            # Anzahl an Geschossen
        self.floor_height = self.params.floor_height                # Geschosshöhe (OKRD bis OKRD)

        ## Gitterpunkte
        self.points = []
        self.create_3d_grid()
        return # explizites Ende (wird zukünftig für alle Methoden so umgesetzt für verbesserte Lesbarkeit)


    def __repr__(self):
        ## Einmal komplette Ausgabe aller Punkte:
        # point_ids = [point['id'] for point in self.points]
        # return f"GridFactory(index={self.index}, point_ids={point_ids})"
        
        ## Einmal zeilenweise Ausgabe aller Punkte:
        repr_string = f"GridFactory mit {len(self.points)} Punkten:\n"
        for idx, p in enumerate(self.points):
                repr_string += f"""  Punkt {idx}: id= {p['id']}, \t(x= {p['x']:.2f}, y= {p['y']:.2f}, z= {p['z']:.2f}), \tpos_type= {p['pos_type']}, \tx_row= {p['x_row']}, \ty_row= {p['y_row']} \tfloor= {p['floor']}\n"""
        return repr_string

    
    """Hauptfunktionen"""
    def create_3d_grid(self):
        """Erstellt eine räumliche Gitterstruktur auf Basis eines Parametersets"""
        print("___Starte Generierung der Gitterstruktur___") # Demo
        id_counter = 0

        ### 3D-Rasterpunkte erzeugen
        for i_x in range(self.grid_x_n + 1):
            for i_y in range(self.grid_y_n + 1):
                x = i_x * self.grid_x_size # i_x ist der Spaltenindex, beginnend bei 0 --> Für alle Reihen in X-Richtung ist der jeweils erste Punkt: x_row == 0.
                y = i_y * self.grid_y_size # i_y ist der Zeilenindex, beginnend bei 0 --> Für alle Reihen in Y-Richtung ist der jeweils der erste Punkt y_row == 0.

                ## Drehung um den Ursprung (0, 0)
                x_rot, y_rot = self.rotate_point(x, y, self.rotation)

                ## Verschiebung der Rasterpunkte
                x_final = x_rot + self.offset_x
                y_final = y_rot + self.offset_y

                ## Kategorisierung der Lage
                if (i_x == 0 or i_x == self.grid_x_n) and (i_y == 0 or i_y == self.grid_y_n):
                    pos_type = 'corner'
                elif (i_x == 0 or i_x == self.grid_x_n) or (i_y == 0 or i_y == self.grid_y_n):
                    pos_type = 'edge'
                else:
                    pos_type = 'inner'
                
                ## Punkte je Geschoss erzeugen
                for floor in range(self.floors):
                    z = floor * self.floor_height
                    point = {
                        'id': id_counter,
                        'x': x_final,
                        'y': y_final,
                        'z': z,
                        'floor': floor,
                        'pos_type': pos_type,
                        'x_row': i_x,
                        'y_row': i_y,
                    }
                    self.points.append(point)
                    id_counter += 1 # PS: id_counter wird hier für die Mittelpunkte am Ende erhöht, sodass keine Dopplungen aufkommen ;)

        ### 3D-Mittelpunkte erzeugen
        self.add_center_point(id_counter)

        ### Punkte nach Geschossen sortieren
        # self.points.sort(key=lambda p: (p['floor'], p['id']))
        self.points.sort(key=lambda p: (p['floor']))
        
        print("___Erfolgreiche Generierung der Gitterstruktur___") # Demo
        return
    
    
    """Hilfsfunktionen"""
    @staticmethod
    def rotate_point(x, y, angle_rad):
        """Rotiert einen Punkt (x, y) um den Ursprung um angle_deg Grad"""
        # angle_rad = np.radians(angle_deg)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        x_rot = cos_theta * x - sin_theta * y
        y_rot = sin_theta * x + cos_theta * y
        return x_rot, y_rot
    

    def add_center_point(self, start_id):
        """Fügt Mittelpunkt je Geschoss hinzu"""
        ## Ermittle Eckpunkte
        min_x = min(p['x'] for p in self.points if p['floor'] == 0)
        max_x = max(p['x'] for p in self.points if p['floor'] == 0)
        min_y = min(p['y'] for p in self.points if p['floor'] == 0)
        max_y = max(p['y'] for p in self.points if p['floor'] == 0)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        ## Mittelpunkte je Geschoss erzeugen
        for floor in range(self.floors):
            z = floor * self.floor_height
            id = start_id + floor
            center_point = {
                'id': id,
                'x': center_x,
                'y': center_y,
                'z': z,
                'floor': floor,
                'pos_type': 'center',
                'x_row': -1,
                'y_row': -1,
            }
            self.points.append(center_point)
        return

    
    """Extra Methoden"""
    def get_points(self):
        """Gibt die Punkte der Gitterstruktur zurück"""
        return self.points
    

    def plot_3d_grid(self, points):
        """Erstellt ein 3D-Plot der Gitterstruktur"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        zs = [p['z'] for p in points]

        ax.scatter(xs, ys, zs, c='b', marker='o')
        ax.set_xlabel('X-Achse [m]')
        ax.set_ylabel('Y-Achse [m]')
        ax.set_zlabel('Z-Achse [m]')

        ax.set_title('3D-Gitterpunkte')
        plt.tight_layout()
        plt.show()
        return


#     def export_grid_as_json(self, filename):
#         """Speichert Gitterstruktur als JSON-Datei"""
#         """GILT ES BESTIMMT NOCH WAS DEN PFAD BETRIFFT ZU ÜBERARBEITEN"""

#         data = {
#         "index": self.index,
#         "points": []
#         }

#         for point in self.points:
#                 point_data = {
#                 "id": point["id"],
#                 "x": point["x"],
#                 "y": point["y"],
#                 "z": point["z"],
#                 "floor": point["floor"],
#                 "position_type": point["position_type"],
#                 "x_row": point["x_row"],
#                 "y_row": point["y_row"],
#                 "x_col_index": point["x_col_index"],
#                 "y_col_index": point["y_col_index"]
#                 }
#                 data["points"].append(point_data)
        
#         with open(filename, 'w') as f:
#                 json.dump(data, f, indent=4)
        
#         print(f"→ Gitter als JSON exportiert: {os.path.abspath(filename)}")
#         pass # return None wäre genauso sinnvoll. Beides verdeutlicht, dass aus dieser Funktion absichtlich nichts zurückkomt.
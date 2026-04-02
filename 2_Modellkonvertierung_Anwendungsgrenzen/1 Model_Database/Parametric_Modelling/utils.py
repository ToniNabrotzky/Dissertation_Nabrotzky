import json
import os
from pathlib import Path



class PathFactory:
    """Speichert die benötigten Pfade"""
    def __init__(self, script_path, script_folder, value_domain, parameter_space):
        self.script_path = script_path # --> Absoluter Pfad zum ausführenden Skript
        self.script_folder = script_folder # --> Absoluter Pfad zum Ordner des ausführenden Skripts
        self.value_domain = value_domain
        self.parameter_space = parameter_space
        return
    
    def get_script_path(self):
        """Gib den Pfad zur Datei der aktuellen Datei zurück"""
        return self.script_path
    
    def get_script_folder(self):
        """Gib den Pfad zum Ordner der aktuellen Datei zurück"""
        return self.script_folder
    
    def create_export_path(self):
        """Gibt den Pfad zum Export der Daten zurück"""
        param_values = self.value_domain
        folder_name = (
            f"{len(self.parameter_space)} - "
            f"Grid_{min(param_values['grid_x_n'])}x{min(param_values['grid_y_n'])}_"
            f"{max(param_values['grid_x_n'])}x{max(param_values['grid_y_n'])} - "
            f"Size_{min(param_values['grid_x_size'])}x{min(param_values['grid_y_size'])}_"
            f"{max(param_values['grid_x_size'])}x{max(param_values['grid_y_size'])} - "
            f"Floors_{min(param_values['floors'])}_{max(param_values['floors'])} - "
            f"FH_{min(param_values['floor_height'])}_{max(param_values['floor_height'])}"
        )

        export_path = os.path.join(self.script_folder, folder_name)
        os.makedirs(export_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden 
        return export_path



class Exporter:
    """Speichert verschiedene Daten für den Export zur Archivierung oder späteren Weiterverarbeitung"""
    def __init__(self, grid_points, params, building_geometry, ifc_model): 
        self.grid_points = grid_points
        self.params = params
        self.building_geometry = building_geometry
        self.ifc_model = ifc_model
        return
    

    """Extra Methoden"""
    @staticmethod
    def export_data_as_txt(data: dict, folder_name: str, folder_path: str, index: int):
        """Exportiert als Dictionary abgespeicherte Daten als Text-Datei im lokalen Ordner"""
        folder_dir = os.path.join(folder_path, folder_name)
        os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

        file_name = f"{folder_name}_Index_{index:05d}"
        text_file_path = os.path.join(folder_dir, file_name + ".txt")

        ## Speichere Parameter als .txt
        with open(text_file_path, 'w') as txt_file:
            txt_file.write(f"{folder_name} für Index {index}:\n")
            for key, value in data.items():
                txt_file.write(f"{key}: {value}\n")
          
        print(f"__Export {folder_name} erfolgreich für Index {index} als .txt")
        return
    
    @staticmethod
    def export_data_as_json(data: dict, folder_name: str, folder_path: str, index: int):
        """Exportiert als Dictionary abgespeicherte Daten als Text-Datei im lokalen Ordner"""
        folder_dir = os.path.join(folder_path, folder_name)
        os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

        file_name = f"{folder_name}_Index_{index:05d}"
        json_file_path = os.path.join(folder_dir, file_name + ".json")
        
        ## Speichere Parameter als .json
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)    
        print(f"__Export {folder_name} erfolgreich für Index {index} als .json")
        return


    @staticmethod
    def export_data_as_ifc(data, folder_name: str, folder_path: str, index: int):
        """Exportiert ein IFC-Modell als eine .ifc-Datei im lokalen Ordner"""
        folder_dir = os.path.join(folder_path, folder_name)
        os.makedirs(folder_dir, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden

        file_name = f"{folder_name} Index_{index:05d}"
        ifc_file_path = os.path.join(folder_dir, file_name + ".ifc")

        ## Exportiere IFC-Datei
        data.write(ifc_file_path)
        print(f"__Export IFC-Modell erfolgreich für Index {index} in Pfad: {ifc_file_path}")
        return
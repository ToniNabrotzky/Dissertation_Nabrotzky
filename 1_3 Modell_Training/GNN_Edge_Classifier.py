import os
import json
import networkx as nx

from pathlib import Path



"""Hauptfunktion"""
def main():
    """Hauptfunktion zum Trainieren des GNN-Modells"""
    global script_dir
    script_dir = Path(__file__).parent
    # script_dir = Path.cwd() # Alternative: Path(__file__).parent
    # print(f"Script-Dir - {script_dir.exists()}: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Teste einzelnes GNN
    json_folder = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch Stahlbeton\JSON_Graph"
    json_file_name = "Parametrisches_Modell Index_016.json"
    json_file_path = f"{json_folder}\\{json_file_name}"
    # json_file = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch Stahlbeton\JSON_Graph\Parametrisches_Modell Index_016.json"
    train_GNN(json_file_path)
    
    ## Trainiere seriell GNN für alle JSON-Dateien
    # json_folder_names = ["Modell_2_Parametrisch IfcOpenShell", "Modell_2_Tutorial BimVision"]
    json_folder_names = ["Modell_2_DataBase", "Modell_2_Parametrisch Stahlbeton"]
    # json_folder_names = []
    json_folder_paths = get_folder_paths(script_dir, json_folder_names)

    for json_folder_path in json_folder_paths:
        # print(f"json-Folder-Path - {json_folder_path.exists()}:  {json_folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Durchsuche folder_path nach allen JSON-Dateien
        if not json_folder_path.exists():
            print(f"__Ordner nicht gefunden. Fehlender Pfad: {json_folder_path}")
        else:
            print(f"__Ordner gefunden. Suche JSON-Dateien in: {json_folder_path}")
            json_files = [datei for datei in os.listdir(json_folder_path) if datei.lower().endswith('.json')]
            # print(f"JSON-Dateien: {json_files}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            for json_file in json_files:
                # print('json_file (generate_graph_from_json): ', json_file) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                print(f"\n  __Analysiere Datei: {json_file}")


def train_GNN(json_file_path):
    print("Hallo, wir trainieren hier GNNs")



"""Hilfsfunktionen """
def get_folder_paths(script_dir, folder_names):
    """Gibt eine Liste von Pfaden zu den Ordnern zurück"""  
    json_folder_paths = [script_dir / folder_name / 'JSON_IFC' for folder_name in folder_names]
    # # print(f"Folder-Paths: {json_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return json_folder_paths




if __name__ == "__main__":
    print("___Intitialisiere Skript___")

    main()

    print("___Skript erfolgreich beendet___")
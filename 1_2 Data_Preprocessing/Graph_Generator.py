import os
import json
import networkx as nx
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


"""Hauptfunktion"""
def main():
    """Hauptfunktion zum Erstellen des Graphen und der Kantenlabels"""
    global script_dir
    script_dir = Path(__file__).parent
    # script_dir = Path.cwd() # Alternative: Path(__file__).parent
    # print(f"Script-Dir - {script_dir.exists()}: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Teste einen gespeicherten Graphen
    # graph_folder = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch Stahlbeton\Graph_Save"
    # graph_file_name = "Parametrisches_Modell Index_005.json"
    # graph_file_path = f"{graph_folder}\\{graph_file_name}"
    # plot_graph_3D(graph_file_path)
    
    
    ## Erstelle seriell Graphen aus allen IFC-Extracts
    extracts_folder_names = ["Modell_2_DataBase", "Modell_2_Parametrisch Stahlbeton"]
    extracts_folder_paths = get_folder_paths(script_dir, extracts_folder_names)

    for extracts_folder_path in extracts_folder_paths:
        # print(f"extracts-Folder-Path - {extracts_folder_path.exists()}:  {extracts_folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        generate_graph_from_extracts(extracts_folder_path)


def generate_graph_from_extracts(extracts_folder_path):
    """Extrahiert Daten aus JSON-Dateien und speichert sie als Graph"""
    ## Durchsuche folder_path nach allen JSON-Dateien
    if not extracts_folder_path.exists():
        print(f"__Ordner nicht gefunden. Fehlender Pfad: {extracts_folder_path}")
    else:
        print(f"__Ordner gefunden. Suche JSON-Dateien in: {extracts_folder_path}")
        json_files = [datei for datei in os.listdir(extracts_folder_path) if datei.lower().endswith('.json')]
        # print(f"JSON-Dateien: {json_files}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
        if not json_files:
            print(f"  __Keine JSON-Dateien gefunden. Extraktion abgebrochen.")
        else:
            print(f"  __JSON-Dateien gefunden. Starte Extraktion...")

            for json_file in json_files:
                # print('json_file (generate_graph_from_extracts): ', json_file) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                print(f"\n  __Analysiere Datei: {json_file}")

                ## Öffne JSON-Datei
                extracts_file_path = extracts_folder_path / json_file

                ## Extrahiere Elemente
                elements = get_elements_from_json(extracts_file_path)

                ## Graph erstellen
                G = create_graph_from_elements(elements)
                print("Graph unverbunden: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Kanten erstellen
                edges = create_edges_from_elements(elements)
                # print("Kanten: ", edges) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Kanten dem Graphen hinzufügen
                G.add_edges_from(edges)
                print("Graph mit Kanten: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Speicher Graph und Kantenliste
                graph_folder_path = Path(extracts_file_path).parent.parent / "Graph_Save"
                os.makedirs(graph_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden
                graph_file_name = Path(extracts_file_path).name # Mit Dateiendung. Ohne geht über ".stem"
                graph_file_path = graph_folder_path / graph_file_name
                # print('graph_file_path :', graph_file_path) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                save_graph_to_json(G, graph_file_path)

                ## Plotte und speicher den Graphen
                plot_graph_3D(graph_file_path)




"""Hilfsfunktionen """
def get_folder_paths(script_dir, folder_names):
    """Gibt eine Liste von Pfaden zu den Ordnern zurück"""  
    extracts_folder_paths = [script_dir / folder_name / 'IFC_Extract' for folder_name in folder_names]
    # # print(f"Folder-Paths: {json_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return extracts_folder_paths


def get_elements_from_json(json_file):
    """Extrahiert Daten aus JSON-Datei"""
    # print('json_file (get_elements_from_json): ', json_file) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    with open(json_file, "r") as f:
        data = json.load(f)
        elements = {}

        for element in data['Bauteile']:
            error_info = element.get('obj_properties', {}).get('Pset_ErrorInfo', {})
            # print('error_info: ', error_info) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            attr = {
                # general attributes
                'id': element['id'],
                'ifc_class': element['ifc_type'],
                'name': element['name'],
                'guid': element['guid'],
                # geometric attributes
                'position': tuple(element['position']),
                'orientation': tuple(element['orientation']),
                'vertices_local': element['vertices_local'],
                'vertices_global': element['vertices_global'],
                'edges': element['edges'],
                # error attributes
                'floor': error_info.get('floor', None),
                'error_modify_x': error_info.get('modify_x', 1.0),
                'error_modify_y': error_info.get('modify_y', 1.0),
                'error_modify_height': error_info.get('modify_heigth', 1.0),
                'error_position_type': error_info.get('position_type', None),

            }
            # print("\n element:", element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("\n attr:", attr) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("position: ", attr['position']) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("vertices_global last: ", attr['vertices_global'][-1]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("error_position_type: ", attr['error_position_type']) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            elements[element['id']] = attr
        return elements


def create_graph_from_elements(elements):
    """Erstellt unverbundenen Graphen aus JSON-Datei"""
    G = nx.Graph()

    ifc_class_encoder = {
        'IfcBuildingElementProxy': 0,
        'IfcBeam': 1,
        'IfcColumn': 2,
        'IfcFoundation': 3,
        'IfcSlab': 4,
        'IfcWall': 5,
        }

    for node_id, element in elements.items():
        node_id = element['id']
        node_features = {
                # 'ifc_class': element['ifc_class'],
                'ifc_class': ifc_class_encoder.get(element['ifc_class'], 0),
                'position': tuple(element['position']),
                'orientation': tuple(element['orientation']),
                'vertices_local': element['vertices_local'],
                'vertices_global': element['vertices_global'],
                'edges': element['edges'],
        }
        # print("\n element:", element) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("\n attributes:", attributes) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("position: ", attributes['position']) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("vertices_global: ", attributes['vertices_global'][-1]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        G.add_node(node_id, **node_features)
    return G


def create_edges_from_elements(elements):
    """Erstellt Kanten aus JSON-Datei"""
    edges = []

    """Regelsatz für Kanten:
    - Verbinde Elemente vom Typ IfcSlab und IfcColumn auf der selben Etage.
    - Verbinde Elemente vom Typ IfcColumn mit IfcSlab aus darunterliegenden Etage.

    Ausnahmen:
    - Wenn bei IfcColumn 'modify_heigth' != 1.0, dan keine Verbindung mit IfcSlab auf selber Etage
    - Wenn bei IfcSlab 'modify_x' bzw. modify_y != 1.0, dann keine Verbindung mit IfcColumn
      mit 'position_type' == 'edge_y' bzw. 'edge_x' oder 'position_type' == 'corner'
      auf der selben Etage und der darüberliegenden Etage
    """

    for node_id, element in elements.items():
        # Überprüfe nach Verbindungen für IfcSlab
        if element['ifc_class'] == 'IfcSlab':
            for other_id, other_element in elements.items():
                # Suche nach IfcColumn
                if other_element['ifc_class'] == 'IfcColumn':
                    # Überprüfe Elemente auf selber Etage
                    if element['floor'] == other_element['floor']:
                        # Überprüfe Ausnahmen
                        # print('node_id und other_id selbe Etage: ', node_id, other_id) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        if (element['error_modify_x'] < 1.0 and 
                            other_element['error_position_type'] in ['edge_y', 'corner']):
                            # print(node_id, " und ", other_id, "Ausnahme selbe Etage wegen modify_x") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            continue
                        if (element['error_modify_y'] < 1.0 and 
                            other_element['error_position_type'] in ['edge_x', 'corner']):
                            # print(node_id, " und ", other_id, "Ausnahme selbe Etage wegen modify_y") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            continue
                        if other_element['error_modify_height'] == 1.0:
                            edges.append((node_id, other_id))
                            edges.append((other_id, node_id))
                        else:
                            # print(node_id, " und ", other_id, "Ausnahme selbe Etage wegen modify_height") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            continue
                    # Überprüfe Elemente auf darüberliegender Etage
                    elif element['floor'] == other_element['floor'] - 1:
                        # print('node_id und other_id obige Etage: ', node_id, other_id) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # Überprüfe Ausnahmen
                        if (element['error_modify_x'] < 1.0 and 
                            other_element['error_position_type'] in ['edge_y', 'corner']):
                            # print(node_id, " und ", other_id, "Ausnahme obige Etage wegen modify_x") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            continue
                        if (element['error_modify_y'] < 1.0 and 
                            other_element['error_position_type'] in ['edge_x', 'corner']):
                            # print(node_id, " und ", other_id, "Ausnahme obige Etage wegen modify_y") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            continue
                        edges.append((node_id, other_id))
                        edges.append((other_id, node_id))     
    return edges


def save_graph_to_json(G, graph_file_path):
    """Speichert den Graphen und die Kantenliste in einer JSON-Datei"""
    graph_data = {
        'nodes': {node: data for node, data in G.nodes(data= True)},
        'edges': list(G.edges())
    }

    ## Speichere Graph und Kantenliste in JSON-Datei
    with open(graph_file_path, "w") as f:
        json.dump(graph_data, f, indent= 4)
    print(f"  __Graph und Kantenliste gespeichert in: {graph_file_path}")


def plot_graph_3D(graph_file_path):
    """Plottet den 3D-Graphen mit den Knotenpositionen aus einer JSON-Datei und speichert Bild"""
    print("file_path: ", graph_file_path)
    with open(graph_file_path, "r") as f:
        graph_data = json.load(f)
    
    G = nx.Graph()
    for node, data in graph_data['nodes'].items():
        node = str(node) # Ensure node ID is a string
        # print(f"Adding node: {node} with data: {data}")  # Debug print
        G.add_node(node, **data)
    # print(f"Graph after adding nodes: {G.nodes(data=True)}")  # Debug print long
    print(f"Graph after adding nodes: {G}")  # Debug print short

    edges = [(str(edge[0]), str(edge[1])) for edge in graph_data['edges']]
    # print(f"Adding edges: {edges}")  # Debug print

    ## Bugfixing: Check for nodes in edges that are not in the initial set of nodes
    initial_nodes = set(G.nodes())
    edge_nodes = set(node for edge in edges for node in edge)
    missing_nodes = edge_nodes - initial_nodes
    if missing_nodes:
        print(f"Warning: The following nodes are referenced in edges but were not in the initial set of nodes: {missing_nodes}")

    G.add_edges_from(edges)
    # print(f"Graph after adding edges: {G.edges(data=True)}")  # Debug print
    print(f"Graph with nodes and edges: {G}")  # Debug print


    """Plottet den 3D-Graphen mit den Knotenpositionen"""
    pos = {node: data['position'] for node, data in G.nodes(data= True)}
    color_map = {
        1: 'aquamarine', # IfcBeam
        1: 'aquamarine', # IfcBeam
        2: 'darkorange', # IfcColumn
        3: 'darkolivegreen', # IfcFoundation
        4: 'mediumblue', # IfcSlab
        5: 'firebrick', # IfcWall
        
    }
    """Liste mit allen statisch relevanten Bauteilen:
    Stütze, Wand, Balken, Decke, Fundament, 
    Pfahl, Treppe, Rampe, Dach"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection= '3d')

    ## Plotte Knoten
    for node, (x, y, z) in pos.items():
        node_color = color_map.get(G.nodes[node]['ifc_class'], 'gray')
        ax.scatter(x, y, z, color= node_color, s= 50)
        ax.text(x, y, z, s= node, fontsize= 10)

    ## Plotte Kanten
    for edge in G.edges():
        x= [pos[edge[0]][0], pos[edge[1]][0]]
        y= [pos[edge[0]][1], pos[edge[1]][1]]
        z= [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color= 'black')    
    
    ## Speicher und plotte den Graphen
    image_folder_path = Path(graph_file_path).parent.parent / "Graph_Image"
    os.makedirs(image_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden
    image_file_name = Path(graph_file_path).stem + '.png' # Dateiname ohne Suffix
    image_file_path = image_folder_path / image_file_name
    print('image_file_path: ', image_file_path)
    plt.savefig(image_file_path)
    
    # plt.show()





if __name__ == "__main__":
    print("___Intitialisiere Skript___")

    main()

    print("___Skript erfolgreich beendet___")
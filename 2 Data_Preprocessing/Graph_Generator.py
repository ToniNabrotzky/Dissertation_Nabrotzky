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

    ## Teste Plotten und Speichern eines Bildes eines  gespeicherten Graphen
    # graph_folder = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch Stahlbeton\Graph_Save"
    # graph_file_name = "Parametrisches_Modell Index_005.json"
    # graph_file_path = f"{graph_folder}\\{graph_file_name}"
    # plot_graph_3D(graph_file_path)
    
    
    ## Erstelle seriell Graphen aus allen IFC-Extracts
    global Save_Images
    Save_Images = False

    extracts_folder_names = ["Modell_2_Parametrisch Stahlbeton 00000_01535"]
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
                print("Graph unverbunden - Erstellung: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Kanten erstellen
                edges = create_edges_from_elements(elements)
                # print("Kanten: ", edges) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Kanten dem Graphen hinzufügen
                edges = create_edges_from_elements(elements)
                for edge in edges:
                    G.add_edge(edge['source'], edge['target'], label=edge['label'])
                    # print(f"Added edge to graph: {edge['source']} -> {edge['target']}, label: {edge['label']}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                print("Graph mit Kanten - Erstellung: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                ## Speicher Graph und Kantenliste
                graph_folder_path = Path(extracts_file_path).parent.parent / "Graph_Save"
                os.makedirs(graph_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden
                graph_file_name = Path(extracts_file_path).name # Mit Dateiendung. Ohne geht über ".stem"
                graph_file_path = graph_folder_path / graph_file_name
                # print('graph_file_path :', graph_file_path) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                save_graph_to_json(G, graph_file_path)

                ## Plotte und speicher den Graphen
                if Save_Images:
                    plot_graph_3D(graph_file_path)




"""Hilfsfunktionen """
def get_folder_paths(script_dir, folder_names):
    """Gibt eine Liste von Pfaden zu den Ordnern zurück"""  
    extracts_folder_paths = [script_dir / folder_name / 'IFC_Extract' for folder_name in folder_names]
    # # print(f"Folder-Paths: {json_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return extracts_folder_paths


def pad_or_truncate(data, max_length, var_type= "verts"):
    """Bringt die vertices auf eine feste Länge durch Padding oder Trunkieren.
    
    :param data: Die Eingabedaten (Liste von Listen oder Werten).
    :param max_length: Die gewünschte Länge der Ausgabe.
    :param var_type: Der Typ der Daten ("verts" für 3D-Koordinaten oder "edges" für Knoten-IDs).
    :return: Die gepaddete oder getrunkierte Liste.
    """
    if len(data) > max_length:
        # Trunkieren, wenn die Länge größer als max_length ist
        return data[:max_length]  # Trunkieren
    else:
        # Padding, wenn die Länge kleiner als max_length ist
        if var_type == "verts":
            # Padding mit [0.0, 0.0, 0.0] als 3er-Liste für 3D-Koordinaten
            return data + [[0.0, 0.0, 0.0]] * (max_length - len(data))
        elif var_type == "edges":
            # Padding mit [0, 0] für Knoten-IDs
            return data + [[0, 0]] * (max_length - len(data))
        else:
            raise ValueError(f"Nicht unterstütze Ausgabe: {var_type}")


def get_elements_from_json(json_file):
    """Extrahiert Daten aus JSON-Datei"""
    # print('json_file (get_elements_from_json): ', json_file) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    with open(json_file, "r") as f:
        data = json.load(f)
        elements = {}

        for element in data['Bauteile']:
            error_info = element.get('obj_properties', {}).get('Pset_ErrorInfo', {})
            # print('error_info: ', error_info) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            ## Padding und Trunkieren der Vertices
            padding = False
            if padding:
                max_length = 12 # Beispielhaft haben alle rechteckigen Körper 8 verts, IPE hat 24 - ist variabel bei anderen
                vertices_local = element['vertices_local']
                vertices_local = pad_or_truncate(vertices_local, max_length, var_type= "verts")
                vertices_global = element['vertices_global']
                vertices_global = pad_or_truncate(vertices_global, max_length, var_type= "verts")
                edges = element['edges']
                edges = pad_or_truncate(edges, max_length, var_type= "edges")
            else:
                vertices_local = element['vertices_local']
                vertices_global = element['vertices_global']
                edges = element['edges']

            attr = {
                # alphanumeric features
                'id': element['id'], #Type: Int
                'ifc_class': element['ifc_type'], #Type: Int
                'name': element['name'], #Type: String
                'guid': element['guid'], #Type: Int
                # geometric features
                'position': tuple(element['position']), #Type: Tuple (konstant)
                'orientation': tuple(element['orientation']), #Type: Tuple (konstant)
                'vertices_local': vertices_local, #Type: Liste (variabel)
                'vertices_global': vertices_global, #Type: Liste (variabel)
                'edges': edges, #Type: Liste (variabel)
                # error features
                'floor': error_info.get('floor', None), #Type: Int
                'error_modify_x': error_info.get('modify_x', 1.0), #Type: Int
                'error_modify_y': error_info.get('modify_y', 1.0), #Type: Int
                'error_modify_height': error_info.get('modify_heigth', 1.0), #Type: Int
                'error_position_type': error_info.get('position_type', None), #Type: String ('edge_y', etc.)

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

    ## Ergänze Knotenmerkmale
    for node_id, element in elements.items():
        node_id = element['id']
        node_features = {
                ## 'ifc_class': element['ifc_class'], # -> Wäre ein Text, was nicht geht
                'ifc_class': ifc_class_encoder.get(element['ifc_class'], 0),
                'position': tuple(element['position']),
                ## 'orientation': tuple(element['orientation']),
                ## 'vertices_local': element['vertices_local'],
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
    node_ids = list(elements.keys())

    ## Erstelle alle möglichen Kanten
    for source_id in node_ids:
        for target_id in node_ids:
            source_element = elements[source_id]
            target_element = elements[target_id]
            # print(f"______Überprüfe {source_id}={source_element['ifc_class']} mit {target_id}={target_element['ifc_class']}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # Standardmäßig: label = 0 (Regeln nicht erfüllt)
            label = 0

            """Regelsatz für Kanten:
                - Verbinde Elemente vom Typ IfcSlab und IfcColumn auf der selben Etage.
                - Verbinde Elemente vom Typ IfcColumn mit IfcSlab aus darunterliegenden Etage.

                Ausnahmen:
                - Wenn bei IfcColumn 'modify_heigth' != 1.0, dan keine Verbindung mit IfcSlab auf selber Etage
                - Wenn bei IfcSlab 'modify_x' bzw. modify_y != 1.0, dann keine Verbindung mit IfcColumn
                mit 'position_type' == 'edge_y' bzw. 'edge_x' oder 'position_type' == 'corner'
                auf der selben Etage und der darüberliegenden Etage
                """

            # Selbstverbindungen werden nicht geprüft und erhalten autoamtisch Label 0
            if source_id != target_id:
                ## Überprüfe Regeln für Elemente für IfcSlab und IfcColumn
                if source_element['ifc_class'] == 'IfcSlab' and target_element['ifc_class'] == 'IfcColumn':
                    # Überprüfe Ausnahmen auf gleicher Etage
                    if source_element['floor'] == target_element['floor']:
                        if not(
                            (source_element['error_modify_x'] < 1.0 and target_element['error_position_type'] in ['edge_y', 'corner']) or
                            (source_element['error_modify_x'] < 1.0 and target_element['error_position_type'] in ['edge_x', 'corner']) or
                            target_element['error_modify_height'] != 1.0
                        ):
                            label = 1
                    # Überprüfe Ausnahmen auf darüberliegender Etage
                    elif source_element['floor'] == target_element['floor'] -1:
                        if not(
                            (source_element['error_modify_x'] < 1.0 and target_element['error_position_type'] in ['edge_y', 'corner']) or
                            (source_element['error_modify_x'] < 1.0 and target_element['error_position_type'] in ['edge_x', 'corner'])
                        ):
                            label = 1
            
            ## Füge Kante mit dem Label hinzu
            # print(f"  __Creating edge: {source_id} -> {target_id}, label: {label}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                
            if label == 1:
                edges.append({'source': source_id, 'target': target_id, 'label': label})
                edges.append({'source': target_id, 'target': source_id, 'label': label}) # Bidirektionale Kante        
    return edges


def save_graph_to_json(G, graph_file_path):
    """Speichert den Graphen und die Kantenliste in einer JSON-Datei"""
    file_id = Path(graph_file_path).stem

    graph_data = {
        'file_id': file_id,
        'nodes': {node: data for node, data in G.nodes(data= True)},
        'edges': [
            {'source': edge[0], 'target': edge[1], 'label': G.edges[edge].get('label', 0)}
            for edge in G.edges()
        ]
    }

    ## Speichere Graph und Kantenliste in JSON-Datei
    with open(graph_file_path, "w") as f:
        json.dump(graph_data, f, indent= 4)
    print(f"  __Graph und Kantenliste gespeichert in: {graph_file_path}")


def plot_graph_3D(graph_file_path):
    """Plottet den 3D-Graphen mit den Knotenpositionen aus einer JSON-Datei und speichert Bild"""
    ## Lade und verarbeite Daten
    # print("file_path: ", graph_file_path) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    with open(graph_file_path, "r") as f:
        graph_data = json.load(f)
    
    # Lade Graphen
    G = nx.Graph()
    for node, data in graph_data['nodes'].items():
        node = int(node) # Absichern, dass node ID eine Ganzzahl ist
        # print(f"Adding node: {node} with data: {data}")  # Debug print
        G.add_node(node, **data)
    # print(f"Graph unverbunden - Plot: {G.nodes(data=True)}")  # Debug print long
    # print(f"Graph unverbunden - Plot: {G}")  # Debug print short

    # Lade Kanten
    edges = [(edge['source'], edge['target']) for edge in graph_data['edges']]
    # print(f"Adding edges: {edges}")  # Debug print

    ## Bugfixing: Check for nodes in edges that are not in the initial set of nodes
    initial_nodes = set(G.nodes())
    edge_nodes = set(node for edge in edges for node in edge)
    missing_nodes = edge_nodes - initial_nodes
    if missing_nodes:
        print(f"Warning: The following nodes are referenced in edges but were not in the initial set of nodes: {missing_nodes}")

    G.add_edges_from(edges)
    # print(f"Graph mit Kanten - Plot: {G.edges(data=True)}")  # Debug print
    # print(f"Graph mit Kanten - Plot: {G}")  # Debug print


    ## Speicher und Plotte den Graphen
    pos = {node: data['position'] for node, data in G.nodes(data= True)} # "if 'position' in data" vor geschweifte Klammer?
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
    
    ## Speicher und plotte den Graphen...
    image_folder_path = Path(graph_file_path).parent.parent / "Graph_Image"
    os.makedirs(image_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden
    ## ...als png
    image_file_name = Path(graph_file_path).stem + '.png' # Dateiname ohne Suffix
    image_file_path = image_folder_path / image_file_name
    print('image_file_path: ', image_file_path)
    plt.savefig(image_file_path)

    ## ...als svg
    image_file_name = Path(graph_file_path).stem + '.svg' # Dateiname ohne Suffix
    image_file_path = image_folder_path / image_file_name
    print('image_file_path: ', image_file_path)
    plt.savefig(image_file_path)
    plt.close(fig)
    
    # plt.show()





if __name__ == "__main__":
    print("___Intitialisiere Skript___")

    main()

    print("___Skript erfolgreich beendet___")
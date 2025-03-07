import json
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""Hauptfunktion"""
def get_elements_from_json(json_file):
    """Extrahiert Daten aus JSON-Datei"""
    with open(json_file, "r") as f:
        data = json.load(f)
        elements = {}

        for element in data['Bauteile']:
            error_info = element.get('obj_properties', {}).get('Pset_ErrorInfo', {})
            # print('error_info: ', error_info) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            attr = {
                # General attributes
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



"""Hilfsfunktionen"""
def plot_graph_2D(G):
    """Plottet den 2D-Graphen mit den Knotenpositionen"""
    pos = {node: data['position'] for node, data in G.nodes(data= True)}

    nx.draw(G, pos,with_labels= True, node_size= 50, node_color= 'green', font_size= 8)
    plt.show()


def plot_graph_3D(G):
    """Plottet den 3D-Graphen mit den Knotenpositionen"""
    pos = {node: data['position'] for node, data in G.nodes(data= True)}
    color_map = {
        0: 'gray', # IfcBuildingElementProxy
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

    for node, (x, y, z) in pos.items():
        node_color = color_map.get(G.nodes[node]['ifc_class'], 'gray')
        ax.scatter(x, y, z, color= node_color, s= 50)
        ax.text(x, y, z, s= node, fontsize= 10)

    for edge in G.edges():
        x= [pos[edge[0]][0], pos[edge[1]][0]]
        y= [pos[edge[0]][1], pos[edge[1]][1]]
        z= [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color= 'black')
    
    plt.show()


## Modellpfade
Param_Modell_001 = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch IfcOpenShell\Parametrisches_Modell Index_001.json"
CoreHouse = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch IfcOpenShell\CoreHouse_VanDerRohe.json"
BimVision_47L = r"H:\1_2 Data_Preprocessing\Modell_2_Tutorial BimVision\47L.json"





if __name__ == "__main__":

    json_file = Param_Modell_001 
    elements = get_elements_from_json(json_file)

    ## Graph erstellen
    G = create_graph_from_elements(elements)
    # G = create_graph_from_json(json_file)
    print("Graph leer: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # plot_graph_3D(G)

    ## Kanten erstellen
    edges = create_edges_from_elements(elements)
    # print("Kanten: ", edges) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Kanten dem Graphen hinzufügen
    G.add_edges_from(edges)
    print("Graph mit Kanten: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    plot_graph_3D(G)
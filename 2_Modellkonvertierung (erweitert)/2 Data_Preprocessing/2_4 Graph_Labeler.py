import json
import csv
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


# Konfiguration der festen IFC-Entitäten für konsistentes One-Hot-Encoding
ALLOWED_ENTITIES = [
    "IfcFooting",
    "IfcBeam",
    "IfcColumn",
    "IfcMember",
    "IfcSlab",
    "IfcWallStandardCase",
    "IfcWall",
    "IfcPlate",
    "IfcPile",
    "IfcBuildingElementProxy"
]

# Konfiguration für Visualisierung: (Farbe, Marker-Form)
VIS_MAPPING = {
    "IfcFooting": ("#8B4513", "s"),           # Braun, Quadrat
    "IfcBeam": ("#2E8B57", "h"),              # Seegrün, Hexagon
    "IfcColumn": ("#FF4500", "p"),            # Orange-Rot, Pentagon
    "IfcMember": ("#4682B4", "v"),            # Stahlblau, Dreieck unten
    "IfcWall": ("#808080", "H"),              # Grau, Hexagon (groß)
    "IfcWallStandardCase": ("#A9A9A9", "D"),  # Dunkelgrau, Diamant
    "IfcSlab": ("#B38719", "o"),              # Gold dunkel, Kreis
    "IfcPlate": ("#DDA0DD", "^"),             # Pflaume, Dreieck oben
    "IfcPile": ("#000000", "|"),              # Schwarz, Vertikale Linie
    "IfcBuildingElementProxy": ("#FFD700", "*") # Gold hell, Stern
}

"""Parameter für Gauß-Kernel (RBF)
Ein kleinerer Gamma-Wert lässt die Aktivierung langsamer abfallen.
Beispiel: d=50mm -> Gamma 0.0001 -> exp(-0.0001 * 2500) = 0.77 (hohe Aktivierung)
Beispiel: d=500mm -> Gamma 0.0001 -> exp(-0.0001 * 250000) = 1.3e-11 (praktisch 0)
"""
RBF_GAMMA = 0.0001


def plot_graph(graph_data, output_dir, model_stem, mode="2D"):
    """
    Erstellt einen Plot des Graphen und speichert ihn ab.
    """
    fig = plt.figure(figsize=(12, 10))
    if mode == "3D":
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # --- Kanten zeichnen ---
    # Zuerst werden nichttragende (unten), dann tragende Kanten (oben) gezeichnet
    # Damit überdecken die GT-Kanten die nicht-tragenden nach dem Maler-Algorithmus-Prinzip
    # Lambda extrahiert den Wert e["features"]["load_transfer"] für den Sortier-Vergleich
    sorted_edges = sorted(edges, key=lambda e: e["features"]["load_transfer"])

    for edge in sorted_edges:
        n_a = next(n for n in nodes if n["node_id"] == edge["source"])
        n_b = next(n for n in nodes if n["node_id"] == edge["target"])

        is_gt = edge["features"]["load_transfer"]
        color = "blue" if is_gt else "gray"
        alpha = 0.9 if is_gt else 0.4
        linewidth = 1.0 if is_gt else 0.5
        linestyle = "-" if is_gt else "--"

        x_coords = [n_a.get("x", 0), n_b.get("x", 0)]
        y_coords = [n_a.get("y", 0), n_b.get("y", 0)]
        z_coords = [n_a.get("z", 0), n_b.get("z", 0)]

        if mode == "3D":
            ax.plot([n_a["x"], n_b["x"]], [n_a["y"], n_b["y"]], [n_a["z"], n_b["z"]], 
                    color=color, alpha=alpha, 
                    linewidth=linewidth, linestyle=linestyle, zorder=1)
        else:
            ax.plot([n_a["x"], n_b["x"]], [n_a["y"], n_b["y"]], 
                    color=color, alpha=alpha,
                    linewidth=linewidth, linestyle=linestyle, zorder=1)
        
    # --- Knoten zeichnen ---
    # Gruppieren nach Typ für effizienteres Plotten (Legende)
    for ent in ALLOWED_ENTITIES:
        # Direkt die Entity zu nehmen, könnte etwas außerhalb der erlaubten Werte ergeben.
        ent_nodes = [n for n in nodes if n.get("entity_processed") == ent]
        if not ent_nodes: continue

        x = [n["x"] for n in ent_nodes]
        y = [n["y"] for n in ent_nodes]
        z = [n["z"] for n in ent_nodes]

        color, marker = VIS_MAPPING.get(ent, ("black", "o"))

        if mode == "3D":
            ax.scatter(x, y, z, c=color, marker=marker, s=60, label=ent, 
                    edgecolors='white', linewidths=0.5, zorder=2)
        else:
            ax.scatter(x, y, c=color, marker=marker, s=60, label=ent, 
                    edgecolors='white', linewidths=0.5, zorder=2)
    
    # --- Graph layouten ---
    ax.set_title(f"Graph Visualisierung ({mode}): {model_stem}")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if mode == "3D": ax.set_zlabel('Z')

    plt.tight_layout()

    # --- Speichern ---
    save_path = os.path.join(output_dir, f"{model_stem} {mode}.png")
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # Schließt den Plot automatisch
    return


def prepare_gnn_graph(model_stem):
    """
    Bereitet einen Graphen für das GNN-Training vor.
    Kombiniert Knotenmerkmale, Distanzmessungen und Ground-Truth-Labels.
    """

    # Basis-Pfad-Definition
    base_dir = os.path.dirname(__file__)
    dir_2_1 = os.path.join(base_dir, "2_1 Graph_Generation")
    dir_2_2 = os.path.join(base_dir, "2_2 Distance_Meassurement")
    dir_2_3 = os.path.join(base_dir, "2_3 Edge_Labeling")
    dir_2_4 = os.path.join(base_dir, "2_4 Graph_Labeled")

    # Dateipfade zusammenbauen
    path_nodes = os.path.join(dir_2_1, f"{model_stem} Nodes.json")
    path_distance = os.path.join(dir_2_2, f"{model_stem} Distance.xlsx")
    path_labels = os.path.join(dir_2_3, f"{model_stem} Labels.json")
    path_output = os.path.join(dir_2_4, f"{model_stem} Graph.json")

    # Validierung der Existenz
    if not all(os.path.exists(p) for p in [path_nodes, path_distance, path_labels]):
        print(f"FEHLER: Nicht alle Quelldateien für {model_stem} gefunden.")
        return
    
    print(f"Verarbeite Modell: {model_stem}...")


    # --- 1. Knoten laden, mappen und verarbeiten ---
    with open(path_nodes, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)

    # guid_to_id = {node['guid']: node['node_id'] for node in nodes_data} # Alte Variante
    guid_to_id = {}
    processed_nodes = []

    # Entitty Mapping Map erstellen (für die Dokumentation im Output)
    entity_map = {entity: i for i, entity in enumerate(ALLOWED_ENTITIES)}

    for node in nodes_data:
        guid = node.get("guid")
        if not guid:
            print(f"WARNUNG: Knoten ohne GUID übersprungen: {node}")
            continue
        
        guid_to_id[guid] = node["node_id"]

        # Zugriff auf Positionsdaten der Knoten
        pos_list = node.get("features", {}).get("position", [0, 0, 0])
        node["x"] = pos_list[0]
        node["y"] = pos_list[1]
        node["z"] = pos_list[2]

        # IFC-Entity Validierung (Fallback auf ProxyElement)
        original_entity = node.get("entity", "IfcBuildingElementProxy")
        if original_entity not in ALLOWED_ENTITIES:
            target_entity = "IfcBuildingElementProxy"
        else:
            target_entity = original_entity

        # One-Hot Index zuweisen
        node["entity_processed"] = target_entity
        node["entity_one_hit_idx"] = entity_map[target_entity]

        # Dokumentation
        processed_nodes.append(node)


    # --- 2. Ground Truth (GT) Labels laden ---
    # Es werden nur Paare gespeichert, die wirklich lasten übertragen (GT).
    gt_pairs = set()
    with open(path_labels, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
        for entry in labels_data:
            if entry.get("Label_Type") == "GT":
                # Sortiertes Tuple für ungerichteten Vergleich
                pair = tuple(sorted([entry["GUID_A"], entry["GUID_B"]]))
                gt_pairs.add(pair)
    

    # --- 3. Kanten und Distanzen verarbeiten ---
    processed_edges = []
    seen_edges = set()

    """Variante für Excel-Dateien"""
    df = pd.read_excel(path_distance)

    for _, row in df.iterrows():
        guid_a = str(row.get("GUID_A", ""))
        guid_b = str(row.get("GUID_B", ""))

        # Distanz ist in der Excel in der Regel ein Float
        try:
            distance = float(row.get("Distanz", 9999.0))
        except:
            distance = 9999.0
        
        # Filter: Keine ungütiltige Zeilen, keine Selbstbezüge
        if not guid_a or not guid_b or guid_a == "---" or guid_b == "---" or guid_a == guid_b:
            continue

        # Filter: Nur GUIDs, die auch in Nodes.json existieren
        if guid_a not in guid_to_id or guid_b not in guid_to_id:
            continue

        # Filter: Duplikate (ungerichteter Graph)
        edge_pair = tuple(sorted([guid_a, guid_b]))
        if edge_pair in seen_edges:
            continue
        seen_edges.add(edge_pair)
        
        # Gauß-Kernel Transformation (RBF)
        # proximity = exp(-gamma * distance^2) --> Wert von 1 bei Distanz=0, ansonsten sinkt der Wert
        proximity = math.exp(-RBF_GAMMA * (distance ** 2))

        # Label bestimmen - Überträgt Last? (Ground Truth Vergleich)
        is_load_bearing = edge_pair in gt_pairs

        processed_edges.append({
            "source": guid_to_id[guid_a],
            "target": guid_to_id[guid_b],
            "features": {
                "distance": distance,
                "proximity_activation": round(proximity, 6),
                "load_transfer": is_load_bearing
            },
            "meta": {
                "guid_a": guid_a,
                "guid_b": guid_b
            }
        })


    # """Variante für csv-Dateien"""
    # with open(path_distance, 'r', encoding='utf-8-sig') as f:
    #     # DictReader wird genutzt für die Spalten GUID_A, GUID_B und Distanz
    #     reader = csv.DictReader(f, delimiter=',') # Nutzt Header-Zeile
    #     for row in reader:
    #         guid_a = row.get("GUID_A")
    #         guid_b = row.get("GUID_B")
    #         distance = row.get("Distanz")

    #         # Filter: Keine ungütiltige Zeilen, keine Selbstbezüge, keine Duplikate
    #         if not guid_a or not guid_b or guid_a == "---" or guid_b == "---" or guid_a == guid_b:
    #             continue

    #         # Filter: Nur GUIDs, die auch in Nodes.json existieren
    #         if guid_a not in guid_to_id or guid_b not in guid_to_id:
    #             continue

    #         # Filter: Duplikate (ungerichteter Graph)
    #         edge_pair = tuple(sorted(guid_a, guid_b))
    #         if edge_pair in seen_edges:
    #             continue
    #         seen_edges.add(edge_pair)

    #         # # Distanz parsen 8Entfernung von " mm" und Komma-Korrektur)
    #         # try:
    #         #     distance = float(distance.replace(" mm", "").replace(",", "."))
    #         # except (ValueError, AttributeError):
    #         #     distance = 9999.0
            
    #         # Gauß-Kernel Transformation (RBF)
    #         # proximity = exp(-gamma * distance^2) --> Wert von 1 bei Distanz=0, ansonsten sinkt der Wert
    #         proximity = math.exp(-RBF_GAMMA * (distance ** 2))

    #         # Label bestimmen - Überträgt Last? (Ground Truth Vergleich)
    #         is_load_bearing = edge_pair in gt_pairs

    #         processed_edges.append({
    #             "source": guid_to_id[guid_a],
    #             "target": guid_to_id[guid_b],
    #             "features": {
    #                 "distance": distance,
    #                 "proximity_activation": round(proximity, 6),
    #                 "load_transfer": is_load_bearing
    #             },
    #             "meta": {
    #                 "guid_a": guid_a,
    #                 "guid_b": guid_b
    #             }
    #         })
    

    # --- 4. Metadaten berechnen ---
    node_count = len(processed_nodes)
    edge_count = len(processed_edges)
    gt_count = len([e for e in processed_edges if e['features']['load_transfer']])
    gt_ratio = (gt_count / edge_count * 100) if edge_count > 0 else 0


    # --- 5. Export ---
    output_data = {
        "graph_info": {
            "model_stem": model_stem,
            "rbf_gamma": RBF_GAMMA,
            "entities": ALLOWED_ENTITIES,
            "stats": {
                "nodes": node_count,
                "edges": edge_count,
                "gt_edges": gt_count,
                "gt_percentage": round(gt_ratio, 2)
            }
        },
        "nodes": processed_nodes,
        "edges": processed_edges
    }
    
    os.makedirs(dir_2_4, exist_ok=True)    
    with open(path_output, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f">>>Fertig! Graph gespeichert unter: {path_output}")
    # Formatierte Ausgabe (gekürzt für Lesbarkeit)
    print(f"Graph gespeichert: {os.path.basename(path_output)}")
    print(f"N: {node_count} | E: {edge_count} | GT: {gt_count} ({round(gt_ratio, 1)}%)")

    # --- 6. Visualisierung ---
    print("Erstelle Plots...")
    plot_graph(output_data, dir_2_4, model_stem, mode="2D")
    plot_graph(output_data, dir_2_4, model_stem, mode="3D")
    print(f"Visualisierungen in {os.path.basename(dir_2_4)} gespeichert.")





# Beispielaufruf
if __name__ == "__main__":
    prepare_gnn_graph("21_22 L_TWP_Tragwerksmodell")
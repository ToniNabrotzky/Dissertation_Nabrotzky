import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
"""Merke: SVGs sind noch immer weiß und buggy"""


def main():
    # prepare_gnn_graph("21_22 L_TWP_Tragwerksmodell")
    # prepare_gnn_graph("23_24 LTWP-V__Dachtragwerk")
    # prepare_gnn_graph("20220421MODEL REV01")
    return


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


def plot_graph(graph_data, output_dir, model_stem, mode="2D", pos_nodes="Centroid"):
    """
    Erstellt einen Plot des Graphen und speichert ihn ab.
    2D: Automatische Anordnung mittels NetworkX (Spring Layout).
    3D: Räumliche Anordnung basierend auf Koordinaten.
    """
    fig = plt.figure(figsize=(12, 10))
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    if not nodes:
        print(f"Warnung: Keine Knoten zum Plotten für {model_stem} vorhanden.")
        plt.close(fig)
        return

    # Zugriff auf Positionsdaten der Knoten für das Plotting
    if mode == "3D":
        ax = fig.add_subplot(111, projection='3d')
        # Sicherstellen, dass die Achsen einen Hintergrund haben
        ax.set_facecolor('white')

        # Echte Urpsrungskoordinaten in 3D
        if pos_nodes == "Centroid":
            pos = {n["node_id"]: 
                   (n["features"]["centroid_x"], 
                    n["features"]["centroid_y"], 
                    n["features"]["centroid_z"]) 
                    for n in nodes}
        else:
            pos = {n["node_id"]: 
                   (n["features"]["origin_x"], 
                    n["features"]["origin_y"], 
                    n["features"]["origin_z"]) 
                    for n in nodes}
    else:
        ax = fig.add_subplot(111)
        # Sicherstellen, dass die Achsen einen Hintergrund haben
        ax.set_facecolor('white')

        # Automatisches Knotenlayout in 2D
        G = nx.Graph()
        for n in nodes:
            G.add_node(n["node_id"])
        for e in edges:
            G.add_edge(e["source"], e["target"])
        
        # spring_layout ordnet Knoten so an, dass Kanten sich möglichst wenig kreuzen
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)    

    # --- Kanten zeichnen ---
    # Zuerst werden nichttragende (unten), dann tragende Kanten (oben) gezeichnet
    # Damit überdecken die GT-Kanten die nicht-tragenden nach dem Maler-Algorithmus-Prinzip
    # Lambda extrahiert den Wert e["features"]["load_transfer"] für den Sortier-Vergleich
    sorted_edges = sorted(edges, key=lambda e: e["features"]["load_transfer"])

    for edge in sorted_edges:
        u, v = edge["source"], edge["target"]
        is_gt = edge["features"]["load_transfer"]

        color = "darkblue" if is_gt else "gray"
        alpha = 0.9 if is_gt else 0.4
        lw = 0.8 if is_gt else 0.5 # linewidth
        ls = "-" if is_gt else "--" # linestyle

        if mode == "3D":
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 
                    color=color, alpha=alpha, linewidth=lw, linestyle=ls, zorder=1)
        else:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                    color=color, alpha=alpha, linewidth=lw, linestyle=ls, zorder=1)
        
    # --- Knoten zeichnen ---
    # Gruppieren nach Typ für effizienteres Plotten (Legende)
    for ent in ALLOWED_ENTITIES:
        # Direkt die Entity zu nehmen, könnte etwas außerhalb der erlaubten Werte ergeben.
        ent_nodes = [n for n in nodes if n.get("entity_processed") == ent]
        if not ent_nodes: continue

        node_ids = [n["node_id"] for n in ent_nodes]
        x = [pos[n_id][0] for n_id in node_ids]
        y = [pos[n_id][1] for n_id in node_ids]

        color, marker = VIS_MAPPING.get(ent, ("black", "o"))

        if mode == "3D":
            z = [pos[n_id][2] for n_id in node_ids]
            ax.scatter(x, y, z, c=color, marker=marker, s=150, label=ent,  # type: ignore
                    edgecolors='white', linewidths=0.5, zorder=5)
        else:
            ax.scatter(x, y, c=color, marker=marker, s=150, label=ent, 
                    edgecolors='white', linewidths=0.5, zorder=5)
    
    # --- Knoten-IDs plotten ---
    for n in nodes:
        nid = n["node_id"]
        coords = pos[nid]

        if mode == "3D":
            # Kleiner Offset in Z-Richtung für bessere Lesbarkeit
            ax.text(coords[0], coords[1], coords[2] + 0.05, str(nid),  # type: ignore
                    fontsize=8, color='black', fontweight='bold', zorder=10)
        else:
            # Kleiner Offset in Y-Richtung
            ax.text(coords[0], coords[1] + 0.01, str(nid),
                    fontsize=8, color='black', fontweight='bold', zorder=10) # type: ignore
            
    
    # --- Graph layouten ---
    if mode == "3D":
        title_type = f"Graph ({mode} - {pos_nodes}): {model_stem}"
    else:
        title_type = f"Graph ({mode}): {model_stem}"
    ax.set_title(title_type)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    if mode == "3D":
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]') # type: ignore
    else:
        # Im Auto-Layout machen Koordinatenachsen keinen Sinn
        ax.set_axis_off()

    # Layout optimieren und Rendern erzwingen
    # plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.canvas.draw()

    # --- Speichern ---
    filename_png = f"{model_stem} {mode}{f' {pos_nodes}' if mode == '3D' else ''}.png"
    filename_svg = f"{model_stem} {mode}{f' {pos_nodes}' if mode == '3D' else ''}.svg"

    # Speichern als PNG
    save_path = os.path.join(output_dir, filename_png)
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # Schließt den Plot automatisch

    # Speichern als SVG
    save_path = os.path.join(output_dir, filename_svg)
    plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=False)
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


    # --- 1. Knoten laden imd Metadaten extrahieren ---
    with open(path_nodes, 'r', encoding='utf-8') as f:
        nodes_raw = json.load(f)
    
    graph_info = nodes_raw.get("graph_info", {})
    nodes_data = nodes_raw.get('nodes', [])

    # --- Mapping initialisieren für den schnellen Zugriff auf Meta-Daten ---
    guid_to_id = {}
    guid_to_info = {}
    processed_nodes = []
    entity_map = {entity: i for i, entity in enumerate(ALLOWED_ENTITIES)}
    
    # --- Vorbereitung der Normierung ---
    # Sammeln aller Koordinaten, um Min/Max für das gesamte Modell zu bestimmen
    try:
        all_x = [n["features"]["centroid_x"] for n in nodes_data]
        all_y = [n["features"]["centroid_y"] for n in nodes_data]
        all_z = [n["features"]["centroid_z"] for n in nodes_data]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        min_z, max_z = min(all_z), max(all_z)
    except KeyError:
        # Fallback falls centroid_x fehlt
        min_x = max_x = min_y = max_y = min_z = max_z = 0

    for node in nodes_data:
        guid = node.get("guid")
        if not guid:
            print(f"WARNUNG: Knoten ohne GUID übersprungen: {node}")
            continue
        
        # Mapping für die Kanten-Metadaten
        guid_to_id[guid] = node["node_id"]
        guid_to_info[guid] = {
            "name": node.get("name", "Unknown"),
            "entity": node.get("entity", "IfcBuildingElementProxy")
        }

        # Zugriff auf Positionsdaten der Knoten aus den features.
        # Primär zählt centroid. Falls nicht vorhanden wird der origin gewählt.
        feat = node["features"]
        cx = feat.get("centroid_x", feat.get("origin_x", 0))
        cy = feat.get("centroid_y", feat.get("origin_y", 0))
        cz = feat.get("centroid_z", feat.get("origin_z", 0))

        # Flache Koordinaten für Plotting
        node["x"], node["y"], node["z"] = cx, cy, cz

        # Normierung berechnen (0.0 bis 1.0)
        feat["normed_x"] = round((cx - min_x) / (max_x - min_x) if max_x != min_x else 0.0, 6)
        feat["normed_y"] = round((cy - min_y) / (max_y - min_y) if max_y != min_y else 0.0, 6)
        feat["normed_z"] = round((cz - min_z) / (max_z - min_z) if max_z != min_z else 0.0, 6)

        # IFC-Entity Validierung (Fallback auf ProxyElement)
        original_entity = node.get("entity", "IfcBuildingElementProxy")
        if original_entity not in ALLOWED_ENTITIES:
            target_entity = "IfcBuildingElementProxy"
        else:
            target_entity = original_entity

        # One-Hot Index zuweisen
        node["entity_processed"] = target_entity
        feat["entity_one_hot_idx"] = entity_map[target_entity]

        # --- Dokumentation ---
        # Neue Knoten-Struktur: x, y, z VOR features
        new_node = {
            "node_id": node["node_id"],
            "guid": guid,
            "name": guid_to_info[guid]["name"],
            "entity": guid_to_info[guid]["entity"],
            "entity_processed": target_entity,
            "x": cx,
            "y": cy,
            "z": cz,
            "features": feat
        }
        processed_nodes.append(new_node)
        # processed_nodes.append(node)


    # --- 2. Ground Truth (GT) Labels laden ---
    # Es werden nur Paare gespeichert, die wirklich lasten übertragen (GT).
    gt_pairs = set()
    # stats_rule_conversion = {}

    with open(path_labels, 'r', encoding='utf-8') as f:
        labels_raw = json.load(f)

    stats_rule_conversion = labels_raw.get("stats_Rule_conversion", {})
    labels_list = labels_raw.get('edge_labels', [])

    for entry in labels_list:
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

        # Label bestimmen - Überträgt Last? (Ground Truth Vergleich mit gemessenen Distanzen)
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
                "guid_b": guid_b,
                "name_a": guid_to_info[guid_a]["name"],
                "name_b": guid_to_info[guid_b]["name"],
                "entity_a": guid_to_info[guid_a]["entity"],
                "entity_b": guid_to_info[guid_b]["entity"]
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
    # Ohne Kopie greifen beide Variablen auf selben Speicherplatz zurück
    final_graph_info = graph_info.copy()
    final_graph_info.update({
        "stats_Rule_conversion": stats_rule_conversion,
        "stats_GNN_preprocessing": {
            "nodes": node_count,
            "edges": edge_count,
            "gt_edges": gt_count,
            "gt_percentage": round(gt_ratio, 2)
        },
        "preprocessing_config": {
            "rbf_gamma": RBF_GAMMA,
            "allowed_entites": ALLOWED_ENTITIES
        }
    })

    output_data = {
        "graph_info": final_graph_info,
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
    plot_graph(output_data, dir_2_4, model_stem, mode="3D", pos_nodes= "Origin")
    print(f"Visualisierungen in {os.path.basename(dir_2_4)} gespeichert.")





# Beispielaufruf
if __name__ == "__main__":
    main()
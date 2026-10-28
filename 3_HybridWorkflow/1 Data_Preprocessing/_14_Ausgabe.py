"""
_14_Ausgabe.py
==============
Gemeinsame Ausgabe beider Experimente.

Das Modul wird von '_14 IFC_Graph_Generator.py' eingebunden und stellt die
Klasse 'Ausgabe' bereit (Mixin der Klasse GNNPipeline).

Struktur:
1. KONFIGURATION (Ausgabe)
2. KLASSE Ausgabe
   - Graph-Export: eine gemeinsame JSON-Datei mit den Ergebnissen
     von Experiment 1 (Sektion 'joints') und Experiment 2 (Knoten & Kanten)
   - Visualisierung des Kontaktgraphen (2D / 3D)
"""
import os
import json
from typing import Callable

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from _14_Dateneingang import ALLOWED_ENTITIES
from _14_Experiment1_RBC_Fehlertaxonomie import JOINT_EPSILON, JOINT_TAU, JOINT_DELTA_SCALE
from _14_Experiment2_GNN_Machbarkeitsstudie import RBF_GAMMA, MAX_DIST

# =========================================================
# KONFIGURATION (Ausgabe)
# =========================================================
# Visualisierung: Farben und Marker je IFC-Entität
VIS_MAPPING = {
    "IfcBeam": ("#2E8B57", "h"), "IfcColumn": ("#FF4500", "p"),
    "IfcSlab": ("#B38719", "o"),
    "IfcWall": ("#808080", "H"), "IfcWallStandardCase": ("#A9A9A9", "D"),
    "IfcFooting": ("#8B4513", "s"), "IfcMember": ("#4682B4", "v"),
    "IfcPile": ("#000000", "|"), "IfcPlate": ("#DDA0DD", "^"),
    "IfcBuildingElementProxy": ("#FFD700", "*")
}


# =========================================================
# AUSGABE: EXPORT & VISUALISIERUNG
# =========================================================
class Ausgabe:
    """
    Schreibt die Ergebnisse beider Experimente in eine gemeinsame Graph-Datei
    und erstellt die Plots des Kontaktgraphen.

    Erwartet von der Hauptklasse (GNNPipeline): den Pfad dir_out sowie die
    Methode log().
    """
    dir_out: str
    log: Callable[..., None]

    # ---------------------------------------------------------
    # GRAPH-EXPORT (BEIDE EXPERIMENTE IN EINER DATEI)
    # ---------------------------------------------------------
    def export_graph(self, stem, nodes, edges, stats_ifc, stats_edges,
                     soll_joints=None, ist_joints=None, matching=None,
                     error_codes=None, joint_stats=None):
        """
        Exportiert den Kontaktgraphen als JSON-Datei.
        Die Datei enthält neben Knoten und Kanten auch eine eigene Sektion
        'joints' mit den Ergebnissen aus Experiment 1 (Soll-/Ist-Joints und Validierung).

        Args:
            stem (str): Modellname ohne Dateiendung.
            nodes (list): Knotenliste des Kontaktgraphen.
            edges (list): Kantenliste des Kontaktgraphen.
            stats_ifc (dict): Statistik der IFC-Extraktion.
            stats_edges (dict): Statistik der Kantenbildung und RBC-Metriken.
            soll_joints (list, optional): Rekonstruierte Soll-Joints (1.A).
            ist_joints (list, optional): Extrahierte Ist-Joints (1.C).
            matching (dict, optional): Zuordnungstabelle Soll <-> Ist (1.D).
            error_codes (dict, optional): Fehlercodes E0-E7 je Joint (1.D).
            joint_stats (dict, optional): Statistische Auswertung (1.D).

        Returns:
            None
        """
        # Bereinigung der unhandlichen Geometriedaten vor dem Export
        clean_nodes = []
        for n in nodes:
            n_copy = n.copy()
            n_copy.pop("geom_raw", None) 
            clean_nodes.append(n_copy)

        output_data = {
            "graph_info": {
                "model_stem": stem,
                "preprocessing_config": {
                    "rbf_gamma": RBF_GAMMA,
                    "max_dist": MAX_DIST,
                    "joint_epsilon": JOINT_EPSILON,
                    "joint_tau": JOINT_TAU,
                    "joint_delta_scale": JOINT_DELTA_SCALE
                },
                "stats_IFC_extraction": stats_ifc,
                "stats_Edges_and_RBC": stats_edges
            },
            "nodes": clean_nodes,
            "edges": edges,
            "joints": {
                "config": {
                    "epsilon": JOINT_EPSILON,
                    "tau": JOINT_TAU,
                    "delta_scale": JOINT_DELTA_SCALE
                },
                "soll_joints": soll_joints if soll_joints is not None else [],
                "ist_joints": ist_joints if ist_joints is not None else [],
                "validation": {
                    "matching": matching if matching is not None else {},
                    "error_codes": error_codes if error_codes is not None else {},
                    "statistics": joint_stats if joint_stats is not None else {}
                }
            }
        }
        
        path = os.path.join(self.dir_out, f"{stem} Graph.json")
        with open(path, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        self.log(f"Graph JSON gespeichert unter: {os.path.basename(path)}")


    # ---------------------------------------------------------
    # VISUALISIERUNG
    # ---------------------------------------------------------
    def plot_graph(self, stem, nodes, edges, mode="2D"):
        """Erstellt Plots der Bauwerksgraphen als 2D- oder 3D-Projektion"""
        fig = plt.figure(figsize=(12, 10))
        
        if mode == "3D":
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')
            pos = {n["node_id"]: (n["x"], n["y"], n["z"]) for n in nodes}
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            G = nx.Graph()
            for n in nodes: G.add_node(n["node_id"])
            for e in edges: G.add_edge(e["source"], e["target"])
            pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
            
        sorted_edges = sorted(edges, key=lambda e: e["features"]["load_transfer"])
        
        for edge in sorted_edges:
            u, v = edge["source"], edge["target"]
            is_gt = edge["features"]["load_transfer"]
            c, a, lw, ls = ("darkblue", 0.9, 0.8, "-") if is_gt else ("gray", 0.4, 0.5, "--")
            
            if mode == "3D":
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 
                        color=c, alpha=a, linewidth=lw, linestyle=ls, zorder=1)
            else:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                        color=c, alpha=a, linewidth=lw, linestyle=ls, zorder=1)
                
        for ent in ALLOWED_ENTITIES:
            ent_nodes = [n for n in nodes if n.get("entity_processed") == ent]
            if not ent_nodes: continue
            
            node_ids = [n["node_id"] for n in ent_nodes]
            x = [pos[n_id][0] for n_id in node_ids]
            y = [pos[n_id][1] for n_id in node_ids]
            color, marker = VIS_MAPPING.get(ent, ("black", "o"))
            
            if mode == "3D":
                z = [pos[n_id][2] for n_id in node_ids]
                ax.scatter(x, y, z, c=color, marker=marker, s=150, label=ent, edgecolors='white', linewidths=0.5, zorder=5) #type: ignore
            else:
                ax.scatter(x, y, c=color, marker=marker, s=150, label=ent, edgecolors='white', linewidths=0.5, zorder=5)
                
        ax.set_title(f"Graph ({mode}): {stem}")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        
        if mode == "3D":
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]') #type: ignore
        else:
            ax.set_axis_off()
            
        fig.subplots_adjust(right=0.85)
        
        filename_png = os.path.join(self.dir_out, f"{stem} {mode}.png")
        plt.savefig(filename_png, dpi=300)
        plt.close(fig)
        self.log(f"Plot erstellt: {os.path.basename(filename_png)}")

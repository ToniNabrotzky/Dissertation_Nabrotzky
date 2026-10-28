"""
_14 IFC_Graph_Generator.py
==========================
Hauptdatei (Einstiegspunkt) des Hybrid-Workflows.

Struktur des Gesamtskripts (aufgeteilt auf mehrere Dateien):
1. '_14 IFC_Graph_Generator.py'   (diese Datei)
       MAIN & STEUERUNG, KONFIGURATION (Verweis), PIPELINE KLASSE
2. '_14_Dateneingang.py'
       Gemeinsamer Dateneingang: IFC-Extraktion und Einlesen der Labels
3. '_14_Experiment1_RBC_Fehlertaxonomie.py'
       Experiment 1: Identifikation und Bewertung von Joints (Hyperkanten)
       1.A Rekonstruktion von Anschlussstellen (Soll-Joints)
       1.B Klassifizierungssystem für Anschlussstellen
       1.C Extraktion von Anschlussstellen aus dem Analysemodell
       1.D Validierung: Matching, Fehlercodes, Statistik
4. '_14_Experiment2_GNN_Machbarkeitsstudie.py'
       Experiment 2: Generierung des Kontaktgraphen
       2.A Knotenmerkmale
       2.B Kantenmerkmale
       2.C Vereinheitlichung und Normalisierung für GNN-Training
5. '_14_Ausgabe.py'
       Graph-Export (eine gemeinsame JSON-Datei) und Visualisierung
"""
import os
import glob
from datetime import datetime

from _14_Dateneingang import Dateneingang
from _14_Experiment1_RBC_Fehlertaxonomie import Experiment1RBCFehlertaxonomie
from _14_Experiment2_GNN_Machbarkeitsstudie import Experiment2GNNMachbarkeitsstudie
from _14_Ausgabe import Ausgabe

# =========================================================
# MAIN & STEUERUNG
# =========================================================
def main():
    """
    Zentrale Steuerung des Skripts.
    Hier können die auszuwertenden IfcStem-Namen definiert werden.
    Ist die Liste leer, werden alle IFC-Modelle im Database-Ordner verarbeitet.
    """
    ifc_names = [
        # "21_22 L_TWP_Tragwerksmodell",
        # "23_24 LTWP-V__Dachtragwerk",
        # "20220421MODEL REV01",
        # "202102183458-Model",
        # "Grethes-hus-bok-2",
        # "Grethes-hus-bok-2",
        # "model_1",
        # "model_3_1D",
        # "model_3_2D",
        # "model_4",
        # "model_7",
        # "model_8",
        "model_9", # 5,0m erforderlich
        # "model_10_1D",
        # "model_10_2D",
        # "Parametrisches Modell Index_00000",
        # "Parametrisches Modell Index_00050",
        # "Parametrisches Modell Index_00055",
        # "Parametrisches Modell Index_00100",
        # "Parametrisches Modell Index_00250",
    ]
    
    pipeline = GNNPipeline(ifc_names)
    pipeline.run()

# =========================================================
# KONFIGURATION
# =========================================================
# Die Parameter liegen thematisch gruppiert im jeweiligen Modul:
#   Dateneingang:  '_14_Dateneingang.py'
#       -> ALLOWED_ENTITIES
#   Experiment 1:  '_14_Experiment1_RBC_Fehlertaxonomie.py'
#       -> 1.A JOINT_EPSILON, JOINT_TAU, JOINT_DELTA_SCALE
#   Experiment 2:  '_14_Experiment2_GNN_Machbarkeitsstudie.py'
#       -> 2.B RBF_GAMMA, MAX_DIST
#   Ausgabe:       '_14_Ausgabe.py'
#       -> VIS_MAPPING
# Die Werte werden in der Graph-Datei unter 'preprocessing_config' dokumentiert.

# =========================================================
# PIPELINE KLASSE
# =========================================================
class GNNPipeline(Dateneingang, Experiment1RBCFehlertaxonomie,
                  Experiment2GNNMachbarkeitsstudie, Ausgabe):
    """
    Hauptklasse des Hybrid-Workflows. Die Fachlogik liegt als Mixin-Klassen
    in eigenen Dateien:

    Dateneingang:
        IFC-Extraktion und Einlesen der Labels.
    Experiment 1 RBC-Fehlertaxonomie (Experiment1RBCFehlertaxonomie):
        1.A Soll-Joints, 1.B Klassifizierung, 1.C Ist-Joints,
        1.D Validierung (Matching, Fehlercodes E0-E7, Statistik).
    Experiment 2 GNN-Machbarkeitsstudie (Experiment2GNNMachbarkeitsstudie):
        2.A Knotenmerkmale, 2.B Kantenmerkmale,
        2.C Vereinheitlichung und Normalisierung für das GNN-Training.
    Ausgabe:
        Graph-Export (gemeinsame JSON-Datei) und Visualisierung.
    """

    # ---------------------------------------------------------
    # INITIALISIERUNG, LOGGING & STEUERUNG
    # ---------------------------------------------------------
    def __init__(self, model_names=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_db = os.path.normpath(os.path.join(self.base_dir, "..", "0 Model_Database"))
        self.dir_gt = os.path.join(self.base_dir, "1 Ground_Truth")
        self.dir_rbc = os.path.join(self.base_dir, "3 RBC_Labels")
        self.dir_out = os.path.join(self.base_dir, "4 Graph_Generation")
        
        os.makedirs(self.dir_out, exist_ok=True)

        if not model_names:
            search_pattern = os.path.join(self.dir_db, "*.ifc")
            self.model_names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(search_pattern)]
        else:
            self.model_names = model_names
            
        self.logs = []


    def log(self, msg, print_console=True):
        """Speichert eine Nachricht als Protokoll und gibt sie optional in der Konsole aus."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        
        if print_console:
            print(log_msg)
            
        for line in str(log_msg).split('\n'):
            clean_msg = line.rstrip()
            if clean_msg:
                self.logs.append(clean_msg)


    def run(self):
        """Startet die Pipeline für alle definierten Modelle."""
        self.log(f"Starte Verarbeitung für {len(self.model_names)} Modelle...")
        for stem in self.model_names:
            self.logs = [] # Reset logs for each model
            self.process_model(stem)


    def process_model(self, stem):
        """
        Führt alle Arbeitsschritte für ein einzelnes IFC-Modell durch:
        - Dateneingang: IFC-Extraktion und Labels
        - Experiment 1 RBC-Fehlertaxonomie: Joints identifizieren und bewerten (1.A-1.D)
        - Experiment 2 GNN-Machbarkeitsstudie: Kontaktgraph generieren (2.A-2.C)
        - Ausgabe: Plots und gemeinsame Graph-Datei
        """
        self.log(f"\n{'='*50}\nVerarbeite Modell: {stem}\n{'='*50}")

        # =====================================================
        # DATENEINGANG: IFC-EXTRAKTION & LABELS
        # =====================================================
        nodes, stats_ifc = self.extract_ifc_data(stem)
        if not nodes:
            self.log(f"Abbruch: Keine auswertbaren Knoten für {stem} gefunden.")
            return

        gt_pairs = self.load_ground_truth(stem)
        rbc_dict = self.load_rbc_labels(stem)

        # =====================================================
        # EXPERIMENT 1 RBC-FEHLERTAXONOMIE: JOINTS (HYPERKANTEN)
        # =====================================================
        # --- 1.A Rekonstruktion von Anschlussstellen (Soll-Joints) ---
        soll_joints = self.reconstruct_gt_joints(stem, nodes, gt_pairs)

        # --- 1.B Klassifizierungssystem für Anschlussstellen ---
        soll_joints = self.classify_gt_joints(soll_joints)

        # --- 1.C Extraktion von Anschlussstellen aus dem Analysemodell (Ist-Joints) ---
        ist_joints = self.extract_analysis_joints(stem)

        # --- 1.D Validierung: Matching, Fehlercodes & Statistik ---
        matching = self.match_joints(soll_joints, ist_joints)
        error_codes = self.classify_joint_errors(matching) # Fehlertaxonomie
        joint_stats = self.evaluate_joint_statistics(matching, error_codes)

        # =====================================================
        # EXPERIMENT 2 GNN-MACHBARKEITSSTUDIE: KONTAKTGRAPH
        # =====================================================
        # --- 2.A Knotenmerkmale (homogener / heterogener Graph) ---
        node_features_hom, node_features_het = self.compute_node_features(nodes)

        # --- 2.B Kantenmerkmale ---
        edges, stats_edges = self.calculate_edges(nodes, gt_pairs, rbc_dict)
        edges = self.compute_edge_features(nodes, edges)

        # --- 2.C Vereinheitlichung und Normalisierung für GNN-Training ---
        gnn_data = self.prepare_gnn_data(nodes, edges, node_features_hom, node_features_het)

        # =====================================================
        # AUSGABE: PLOTS & GEMEINSAME GRAPH-DATEI
        # =====================================================
        self.plot_graph(stem, nodes, edges, mode="2D")
        self.plot_graph(stem, nodes, edges, mode="3D")

        # Eine Datei für beide Experimente (Kontaktgraph + Joints + Validierung)
        self.export_graph(stem, nodes, edges, stats_ifc, stats_edges,
                          soll_joints=soll_joints, ist_joints=ist_joints,
                          matching=matching, error_codes=error_codes,
                          joint_stats=joint_stats)

        # --- Log speichern ---
        log_path = os.path.join(self.dir_out, f"{stem} Protokoll.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.logs))


if __name__ == "__main__":
    main()

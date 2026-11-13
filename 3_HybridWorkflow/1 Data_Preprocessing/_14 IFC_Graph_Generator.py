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
import math
from datetime import datetime
from numbers import Real

from _14__Dateneingang import Dateneingang
from _14__Experiment1_RBC import Experiment1RBCFehlertaxonomie
from _14__Experiment2_GNN import Experiment2GNNMachbarkeitsstudie
from _14__Ausgabe import Ausgabe

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
        # "model_3_2D_ohneDach",
        # "model_4",
        "model_7",
        # "model_8",
        # "model_9", # 5,0m erforderlich
        # "model_10_1D",
        # "model_10_1D_ohneDach",
        # "model_10_2D",
        # "Parametrisches Modell Index_00000",
        # "Parametrisches Modell Index_00050",
        # "Parametrisches Modell Index_00055",
        # "Parametrisches Modell Index_00100",
        # "Parametrisches Modell Index_00250",
    ]
    
    # Leere Liste: Standardwert; gefüllte Liste: Sensitivitätsstudie in Metern.
    experiment_1 = True
    experiment_2 = False
    sensitivity_study = []  # z. B. [0.025, 0.05, 0.1]


    pipeline = GNNPipeline(ifc_names)
    pipeline.run(
        experiment_1=experiment_1,
        experiment_2=experiment_2,
        sensitivity_study=sensitivity_study,
    )

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
        """Speichert eine Nachricht als Protokoll und gibt sie optional in der Konsole aus. Ein leerer String erzeugt eine Leerzeile ohne Zeitstempel."""
        if msg == "":
            if print_console:
                print()
            self.logs.append("")
            return

        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {msg}"
        
        if print_console:
            print(log_msg)
            
        for line in str(log_msg).split('\n'):
            clean_msg = line.rstrip()
            if clean_msg:
                self.logs.append(clean_msg)
                

    def run(self, experiment_1=True, experiment_2=False,
            sensitivity_study=[]):
        """Lädt die gemeinsamen Eingangsdaten und startet die gewählten Experimente.

        Eine leere Epsilon-Liste verwendet den Standardwert aus Experiment 1.
        Eine gefüllte Liste startet eine Sensitivitätsstudie mit Werten in Metern.
        """
        if not experiment_1 and not experiment_2:
            print("Keine Experimente aktiviert.")
            return

        if experiment_1:
            if not isinstance(sensitivity_study, list):
                raise TypeError("Die Epsilon-Werte müssen als Liste angegeben werden.")
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   for value in sensitivity_study):
                raise ValueError("Epsilon-Werte müssen reelle Zahlen sein.")
            try:
                epsilons = [float(value) for value in sensitivity_study]
            except OverflowError as exc:
                raise ValueError("Epsilon-Werte müssen endlich und positiv sein.") from exc
            if any(not math.isfinite(value) or value <= 0 for value in epsilons):
                raise ValueError("Epsilon-Werte müssen endlich und positiv sein.")
            if len(set(epsilons)) != len(epsilons):
                raise ValueError("Die Epsilon-Liste darf keine Duplikate enthalten.")

        for stem in self.model_names:
            self.logs = []
            # Protokoll - Process
            self.log(f"\n{'='*50}\nVerarbeite Modell: {stem}\n{'='*50}")

            # =====================================================
            # DATENEINGANG: IFC-EXTRAKTION UND GROUND TRUTH
            # =====================================================
            nodes, stats_ifc = self.extract_ifc_data(stem)

            if not nodes:
                # Protokoll - Process
                self.log(f"ABBRUCH Process: Keine auswertbaren Knoten für {stem} gefunden.")
                if experiment_1:
                    self._save_protocol(stem, 1)
                if experiment_2:
                    self._save_protocol(stem, 2)
                continue

            gt_pairs = self.load_ground_truth(stem)
            input_logs = self.logs.copy()

            if experiment_1:
                try:
                    self.process_experiment_1(
                        stem,
                        nodes,
                        stats_ifc,
                        gt_pairs,
                        sensitivity_study=sensitivity_study,
                    )
                finally:
                    self._save_protocol(stem, 1)

            if experiment_2:
                self.logs = input_logs.copy()
                try:
                    self.process_experiment_2(stem, nodes, stats_ifc, gt_pairs)
                finally:
                    self._save_protocol(stem, 2)
    
    def process_experiment_1(
        self, stem, nodes, stats_ifc, gt_pairs,
        sensitivity_study=[],
    ):
        """Führt Schritt 1–3 von Experiment 1 aus und exportiert die Ergebnisse."""
        if sensitivity_study:
            results = self._run_epsilon_study(nodes, gt_pairs, epsilons=sensitivity_study)

            for study_epsilon, result in results.items():
                self.export_soll_joints(
                    stem,
                    nodes,
                    stats_ifc,
                    result["contact_regions"],
                    result["soll_joints"],
                    result["joint_classes"],
                    epsilon=study_epsilon,
                    filename_suffix=f" eps_{study_epsilon!r}m",
                    gt_pairs=gt_pairs,
                )
        else:
            # EXPERIMENT 1, SCHRITT 1: KONTAKTPAARE UND KONTAKTREGIONEN
            contact_regions = self._compute_contact_regions(
                nodes, gt_pairs, epsilon=None
            )
            
            # EXPERIMENT 1, SCHRITT 2: SOLL-JOINTS CLUSTERN
            soll_joints = self._build_soll_joints(
                nodes, gt_pairs, contact_regions
            )
            
            # EXPERIMENT 1, SCHRITT 3: SOLL-JOINTS KLASSIFIZIEREN
            joint_classes = self._classify_soll_joints(
                nodes, gt_pairs, contact_regions, soll_joints, epsilon=None
            )
            
            # EXPERIMENT 1, DATENAUSGANG
            self.export_soll_joints(
                stem,
                nodes,
                stats_ifc,
                contact_regions,
                soll_joints,
                joint_classes,
                epsilon=None,
                gt_pairs=gt_pairs,
            )


    def process_experiment_2(self, stem, nodes, stats_ifc, gt_pairs):
        """Platzhalter: Experiment 2 und dessen Datenausgabe folgen später."""
        pass


    def _save_protocol(self, stem, experiment):
        log_path = os.path.join(
            self.dir_out, f"{stem} Protokoll Exp_{experiment}.txt"
        )
        with open(log_path, "w", encoding="utf-8") as file:
            file.write("\n".join(self.logs))


if __name__ == "__main__":
    main()

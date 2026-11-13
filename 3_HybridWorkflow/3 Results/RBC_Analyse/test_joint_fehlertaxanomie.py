import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import Joint_Fehlertaxanomie as analyse


A = "0000000000000000000001"
B = "0000000000000000000002"
C = "0000000000000000000003"
AB = (A, B)
AC = (A, C)


class JointAnalyseTests(unittest.TestCase):
    def test_joint_bewertung_und_fehlerprioritaet(self):
        self.assertEqual(analyse.bewerte_joint({AB, AC}, {AB: {"TP"}, AC: {"TP"}}), "korrekt")
        self.assertEqual(analyse.bewerte_joint({AB, AC}, {AB: {"FN"}}), "falsch")
        self.assertEqual(analyse.bewerte_joint({AB, AC}, {AB: {"TP"}}), "unbewertet")
        self.assertEqual(analyse.bewerte_joint({AB}, {AB: {"FP"}}), "unbewertet")
        self.assertEqual(analyse.bewerte_joint({AB}, {AB: {"TP", "FN"}}), "unbewertet")
        self.assertEqual(analyse.bewerte_joint({AB}, None), "unbewertet")
        self.assertEqual(set(analyse.gruende_unbewertet({AB, AC}, {AB: {"FP"}})),
                         {"kein RBC-Label", "FP/TN auf Soll-Paar"})

    def test_csv_duplikate_ungueltige_guids_und_widerspruch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", encoding="utf-8", newline="") as datei:
                writer = csv.writer(datei, delimiter=";")
                writer.writerow(["IFC_GUID_A", "IFC_GUID_B", "Label"])
                writer.writerows([
                    [A, B, "TP"], [B, A, "TP"], [A, C, "FN"],
                    [C, A, "FP"], ["IfcColumn", B, "FN"],
                ])
            warnungen = []
            labels, zeilen = analyse.lese_rbc(path, "modell", warnungen)
        self.assertEqual(labels, {AB: {"TP"}, AC: {"FN", "FP"}})
        self.assertEqual(zeilen["TP"], 2)
        self.assertEqual(len(warnungen), 2)
        result = analyse.metriken(labels)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 0, 0))
        self.assertEqual(result["widerspruechliche_Paare"], 1)

    def test_metriken_und_nicht_definierte_werte(self):
        result = analyse.metriken({AB: {"TP"}, AC: {"FN"}, (B, C): {"FP"}})
        self.assertEqual((result["Precision"], result["Recall"], result["F1"]), (0.5, 0.5, 0.5))
        leer = analyse.metriken({})
        self.assertIsNone(leer["Precision"])
        self.assertIsNone(leer["Recall"])
        self.assertIsNone(leer["F1"])

    def test_auswertung_mit_unvollstaendigen_modellen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joints_dir, rbc_dir, out_dir = (root / name for name in ("joints", "rbc", "out"))
            joints_dir.mkdir()
            rbc_dir.mkdir()
            klasse = {
                "joint_id": "J1", "valence_class": "V2", "h": 1, "p": [1.0], "o_class": "O0",
                "gamma_max": 0, "k_max": "K1", "configuration": "L",
                "angle_class": "orthogonal",
            }
            for stem in ("komplett", "ohne_rbc"):
                daten = {
                    "model_stem": stem,
                    "contact_pairs": [{"pair_id": "P1", "guid_a": A, "guid_b": B}],
                    "soll_joints": [{"joint_id": "J1", "pair_ids": ["P1"]}],
                    "joint_classes": [klasse],
                }
                (joints_dir / f"{stem} Soll_Joints.json").write_text(json.dumps(daten), encoding="utf-8")
            for stem, label in (("komplett", "TP"), ("ohne_joints", "FP")):
                with (rbc_dir / f"{stem} RFEM_Labels.csv").open("w", encoding="utf-8", newline="") as datei:
                    writer = csv.writer(datei, delimiter=";")
                    writer.writerow(["IFC_GUID_A", "IFC_GUID_B", "Label"])
                    writer.writerow([A, B, label])
            # Eine fehlerhafte Datei ohne Gegenstück darf nicht eingelesen werden.
            (joints_dir / "ohne_rbc Soll_Joints.json").write_text("ungültig", encoding="utf-8")
            with patch.multiple(
                analyse, DIR_JOINTS=joints_dir, DIR_RBC=rbc_dir, DIR_OUT=out_dir,
                MODELLE=["komplett", "ohne_rbc", "ohne_joints", "keine_dateien"],
            ), patch.object(analyse, "zeichne_klassen"), patch.object(analyse, "zeichne_bewertung"), \
                    patch.object(analyse, "zeichne_metriken"), redirect_stdout(io.StringIO()):
                analyse.main()
            with (out_dir / "Joint_Einzelbewertung.csv").open(encoding="utf-8-sig", newline="") as datei:
                joints = list(csv.DictReader(datei, delimiter=";"))
            with (out_dir / "RBC_Metriken.csv").open(encoding="utf-8-sig", newline="") as datei:
                metrics = {row["Modell"]: row for row in csv.DictReader(datei, delimiter=";")}
            with (out_dir / "Warnungen.csv").open(encoding="utf-8-sig", newline="") as datei:
                warnings = list(csv.DictReader(datei, delimiter=";"))
            protokoll = (out_dir / "Protokoll RBC_Analyse.txt").read_text(encoding="utf-8")
            self.assertEqual({row["Modell"]: row["Bewertung"] for row in joints},
                             {"komplett": "korrekt"})
            self.assertEqual(metrics["komplett"]["TP"], "1")
            self.assertNotIn("ohne_joints", metrics)
            self.assertEqual(metrics["Gesamt (modellweise)"]["FP"], "0")
            self.assertEqual({row["Modell"] for row in warnings if "übersprungen" in row["Warnung"]},
                             {"ohne_rbc", "ohne_joints", "keine_dateien"})
            self.assertIn("Übersprungene Modelle (3)", protokoll)


if __name__ == "__main__":
    unittest.main()

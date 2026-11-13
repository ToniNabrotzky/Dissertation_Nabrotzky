"""
Joint_Fehlertaxanomie.py
=======================
Beschreibung der Soll-Joints und Auswertung der RBC-Konvertierung.

Eingang: Soll-Joints aus '4 Graph_Generation' und die gleichnamigen
RFEM_Labels.csv aus '3 RBC_Labels'. 
Ausgabe: Tabellen, Diagramme und ein Protokoll in RBC_Analyse. 
Aufruf: python Joint_Fehlertaxanomie.py

Modelle ohne eine der beiden Dateien werden vollständig übersprungen.
Auswertung auf Ebene eindeutiger, ungerichteter IFC-GUID-Paare. CSV-Labels
werden nicht anhand der aktuellen Ground Truth neu vergeben. Ein Soll-Joint
ist falsch, sobald eines seiner Soll-Paare als FN vorliegt; er ist nur dann
korrekt, wenn alle Soll-Paare als TP vorliegen. Fehlende, widersprüchliche
oder als FP/TN bezeichnete Soll-Paare bleiben unbewertet. FP werden nicht
einem Soll-Joint zugerechnet. Ein Paar kann mehreren Soll-Joints angehören.
Die CSV enthält keine vollständige Grundgesamtheit negativer Paare; deshalb
werden Accuracy und TN nicht aus fehlenden Zeilen abgeleitet.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# KONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_JOINTS = BASE_DIR / "1 Data_Preprocessing" / "4 Graph_Generation"
DIR_RBC = BASE_DIR / "1 Data_Preprocessing" / "3 RBC_Labels"
DIR_OUT = Path(__file__).resolve().parent / "RBC_Analyse"

# Leere Liste: alle Modelle mit beiden Dateien in den Eingangsordnern.
MODELLE = []
JOINT_SUFFIX = " Soll_Joints.json"
RBC_SUFFIX = " RFEM_Labels.csv"
GUID_RE = re.compile(r"^[0-3][0-9A-Za-z_$]{21}$")
ACHSEN = {
    # Achse 1: Valenz
    "Valenz": "valence_class",
    "Heterogenitaet": "h",
    "Innendichte": "p",
    # Achse 2: Versatz
    "Versatz": "o_class",
    "Dimensionssprung": "gamma_max",
    # Achse 3: Morphologie
    "Kontakt": "k_max",
    "Konfiguration": "configuration",
    "Winkel": "angle_class",
}
STATUSES = ("korrekt", "falsch", "unbewertet")


def paar(guid_a, guid_b):
    """Gibt ein gültiges, ungerichtetes IFC-Paar oder None zurück."""
    a, b = str(guid_a or "").strip(), str(guid_b or "").strip()
    if not GUID_RE.fullmatch(a) or not GUID_RE.fullmatch(b) or a == b:
        return None
    return tuple(sorted((a, b)))


def lese_rbc(path, modell, warnungen):
    """Liest CSV-Labels und bewahrt widersprüchliche Labels je Paar."""
    labels = defaultdict(set)
    zeilen = Counter()
    with path.open(encoding="utf-8-sig", newline="") as datei:
        reader = csv.DictReader(datei, delimiter=";")
        if not reader.fieldnames or "IFC_GUID_A" not in reader.fieldnames:
            datei.seek(0)
            reader = csv.DictReader(datei, delimiter=",")
        required = {"IFC_GUID_A", "IFC_GUID_B", "Label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: erforderliche Spalten fehlen: {required}")
        for nr, row in enumerate(reader, 2):
            label = (row.get("Label") or "").strip().upper()
            verbindung = paar(row.get("IFC_GUID_A"), row.get("IFC_GUID_B"))
            if label not in {"TP", "FP", "FN", "TN"}:
                zeilen["unbekanntes_Label"] += 1
                warnungen.append((modell, f"{path.name}, Zeile {nr}: unbekanntes Label {label!r}"))
                continue
            zeilen[label] += 1
            if verbindung is None:
                zeilen["ungueltige_Paarzeilen"] += 1
                warnungen.append((
                    modell,
                    f"{path.name}, Zeile {nr}: fehlende/ungültige oder gleiche GUIDs "
                    f"({row.get('IFC_GUID_A')!r}, {row.get('IFC_GUID_B')!r}); nicht als Paar auswertbar",
                ))
                continue
            labels[verbindung].add(label)
    for verbindung, werte in labels.items():
        if len(werte) > 1:
            warnungen.append((modell, f"Widersprüchliche Labels für {verbindung}: {sorted(werte)}"))
    return labels, zeilen


def lade_joints(path):
    """Prüft die Referenzen und liefert Joint-Klassen und Soll-Paare."""
    with path.open(encoding="utf-8-sig") as datei:
        daten = json.load(datei)
    for feld in ("model_stem", "contact_pairs", "soll_joints", "joint_classes"):
        if feld not in daten:
            raise ValueError(f"{path}: Feld {feld} fehlt")
    regionen = {}
    for region in daten["contact_pairs"]:
        pair_id = region["pair_id"]
        verbindung = paar(region["guid_a"], region["guid_b"])
        if verbindung is None or pair_id in regionen:
            raise ValueError(f"{path}: ungültiges oder doppeltes Kontaktpaar {pair_id}")
        regionen[pair_id] = verbindung
    klassen = {}
    for klasse in daten["joint_classes"]:
        joint_id = klasse["joint_id"]
        if joint_id in klassen:
            raise ValueError(f"{path}: doppelte Joint-Klasse {joint_id}")
        klassen[joint_id] = klasse
    joints = []
    ids = set()
    for joint in daten["soll_joints"]:
        joint_id = joint["joint_id"]
        if joint_id in ids or joint_id not in klassen:
            raise ValueError(f"{path}: doppelte Joint-ID oder fehlende Klasse: {joint_id}")
        ids.add(joint_id)
        pair_ids = joint["pair_ids"]
        if not pair_ids or len(set(pair_ids)) != len(pair_ids):
            raise ValueError(f"{path}: leere oder doppelte Paar-IDs in {joint_id}")
        fehlend = set(pair_ids) - regionen.keys()
        if fehlend:
            raise ValueError(f"{path}: unbekannte Paar-IDs in {joint_id}: {sorted(fehlend)}")
        klasse = klassen[joint_id]
        fehlende_felder = [feld for feld in ACHSEN.values() if klasse.get(feld) is None]
        if fehlende_felder:
            raise ValueError(f"{path}: Klassifizierung {joint_id} unvollständig: {fehlende_felder}")
        joints.append((joint_id, klasse, {regionen[pid] for pid in pair_ids}))
    if ids != set(klassen):
        raise ValueError(f"{path}: Klassen ohne Soll-Joint: {sorted(set(klassen) - ids)}")
    return daten["model_stem"], joints


def bewerte_joint(verbindungen, labels):
    """Falsch hat Vorrang vor unbewertet; FP werden nicht zugeordnet."""
    if labels is None:
        return "unbewertet"
    zustaende = [labels.get(verbindung, set()) for verbindung in verbindungen]
    if any(wert == {"FN"} for wert in zustaende):
        return "falsch"
    if all(wert == {"TP"} for wert in zustaende):
        return "korrekt"
    return "unbewertet"


def gruende_unbewertet(verbindungen, labels):
    """Nennt die nicht exklusiven Ursachen für eine unvollständige Bewertung."""
    zustaende = [labels.get(verbindung, set()) for verbindung in verbindungen]
    gruende = []
    if any(not wert for wert in zustaende):
        gruende.append("kein RBC-Label")
    if any(wert & {"FP", "TN"} for wert in zustaende):
        gruende.append("FP/TN auf Soll-Paar")
    if any(len(wert) > 1 for wert in zustaende):
        gruende.append("widersprüchliche Labels")
    return gruende


def metriken(labels):
    """Metriken über auswertbare eindeutige Paare; nicht definierte Werte sind leer."""
    zaehler = Counter(next(iter(werte)) for werte in labels.values() if len(werte) == 1)
    tp, fp, fn = (zaehler[x] for x in ("TP", "FP", "FN"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN_nur_CSV": zaehler["TN"],
        "widerspruechliche_Paare": sum(len(werte) > 1 for werte in labels.values()),
        "Precision": precision, "Recall": recall, "F1": f1,
    }


def csv_schreiben(path, spalten, zeilen):
    with path.open("w", encoding="utf-8-sig", newline="") as datei:
        writer = csv.DictWriter(datei, fieldnames=spalten, delimiter=";")
        writer.writeheader()
        writer.writerows(zeilen)


def klassen_zaehlen(joints):
    """Zählt Achsen, kategoriale Kombinationen und unveränderte Shortcodes."""
    result = defaultdict(Counter)
    for joint in joints:
        for achse in ACHSEN:
            result[(achse, joint[achse])][joint["Bewertung"]] += 1
        kombination = " | ".join(
            f"{achse}={joint[achse]}" for achse in ACHSEN if achse != "Innendichte"
        )
        result[("Kombination", kombination)][joint["Bewertung"]] += 1
        result[("Shortcode", joint["Shortcode"])][joint["Bewertung"]] += 1
    return result


def zeichne_klassen(zaehlung, path):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
    for ax, achse in zip(axes.flat, ACHSEN):
        daten = sorted((klasse, counts) for (a, klasse), counts in zaehlung.items() if a == achse)
        namen = [klasse for klasse, _ in daten]
        werte = [sum(counts.values()) for _, counts in daten]
        ax.barh(namen, werte, color="#4472a1")
        ax.set_title(achse)
        ax.set_xlabel("Anzahl Soll-Joints")
        ax.invert_yaxis()
    fig.suptitle("Datensatz: Verteilung der Joint-Klassifikationen")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def zeichne_bewertung(zaehlung, path):
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
    farben = {"korrekt": "#4c9f70", "falsch": "#d35f5f", "unbewertet": "#a7a7a7"}
    for ax, achse in zip(axes.flat, ACHSEN):
        daten = sorted((klasse, counts) for (a, klasse), counts in zaehlung.items() if a == achse)
        namen = [klasse for klasse, _ in daten]
        links = [0] * len(daten)
        for status in STATUSES:
            werte = [counts[status] for _, counts in daten]
            ax.barh(namen, werte, left=links, label=status, color=farben[status])
            links = [a + b for a, b in zip(links, werte)]
        ax.set_title(achse)
        ax.set_xlabel("Anzahl Soll-Joints")
        ax.invert_yaxis()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3)
    fig.suptitle("RBC-Bewertung der Soll-Joints nach Klassifikation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def zeichne_metriken(zeilen, path):
    if not zeilen:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(zeilen) * 1.3), 5), constrained_layout=True)
    breite = 0.25
    for offset, name in enumerate(("Precision", "Recall", "F1")):
        # Nicht definierte Metriken erhalten keinen Balken, nicht den Wert null.
        xs = [i + (offset - 1) * breite for i, z in enumerate(zeilen) if z[name] is not None]
        werte = [z[name] for z in zeilen if z[name] is not None]
        ax.bar(xs, werte, breite, label=name)
    ax.set_xticks(
        range(len(zeilen)),
        [z["Modell"] + (" *" if z["Datenqualitaet"] != "ohne erkannte Einschränkung" else "") for z in zeilen],
        rotation=30, ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wert")
    ax.set_title("RBC-Paarmetriken (* eingeschränkte Daten; nicht definiert: kein Balken)")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_wert(wert):
    return "nicht definiert" if wert is None else f"{wert:.4f}"


def main():
    if not DIR_JOINTS.is_dir() or not DIR_RBC.is_dir():
        raise FileNotFoundError(f"Eingangsordner fehlen: {DIR_JOINTS} bzw. {DIR_RBC}")
    warnungen = []
    joint_dateien = {
        path.name[:-len(JOINT_SUFFIX)]: path for path in DIR_JOINTS.glob(f"*{JOINT_SUFFIX}")
    }
    rbc_dateien = {
        path.name[:-len(RBC_SUFFIX)]: path for path in DIR_RBC.glob(f"*{RBC_SUFFIX}")
    }
    kandidaten = set(MODELLE) if MODELLE else set(joint_dateien) | set(rbc_dateien)
    uebersprungen = []
    for stem in sorted(kandidaten):
        fehlende_dateien = []
        if stem not in joint_dateien:
            fehlende_dateien.append(f"{stem}{JOINT_SUFFIX}")
        if stem not in rbc_dateien:
            fehlende_dateien.append(f"{stem}{RBC_SUFFIX}")
        if fehlende_dateien:
            meldung = f"Modell übersprungen; Datei(en) fehlen: {', '.join(fehlende_dateien)}"
            uebersprungen.append((stem, meldung))
            warnungen.append((stem, meldung))
    gemeinsame_modelle = sorted(kandidaten & joint_dateien.keys() & rbc_dateien.keys())
    if not gemeinsame_modelle:
        raise FileNotFoundError(f"Keine Modelle mit beiden Eingabedateien. {warnungen}")

    joint_zeilen, paar_zeilen, metrik_zeilen = [], [], []
    rbc_zeilen = {}
    modelle = []
    gesamt_labels = {}
    for stem in gemeinsame_modelle:
        path = joint_dateien[stem]
        model_stem, joints = lade_joints(path)
        if model_stem != stem:
            raise ValueError(f"{path}: model_stem {model_stem!r} stimmt nicht mit Dateiname überein")
        modelle.append(stem)
        labels, zeilen = lese_rbc(rbc_dateien[stem], stem, warnungen)
        rbc_zeilen[stem] = zeilen
        gesamt_labels.update({(stem, *pair): werte for pair, werte in labels.items()})

        zuordnung = defaultdict(list)
        for joint_id, klasse, verbindungen in joints:
            bewertung = bewerte_joint(verbindungen, labels)
            row = {
                "Modell": stem, "Joint_ID": joint_id,
                **{
                    name: str(klasse[feld][0] if isinstance(klasse[feld], list) and len(klasse[feld]) == 1 else klasse[feld])
                    for name, feld in ACHSEN.items()
                },
                "Shortcode": klasse.get("shortcode", ""),
                "Soll_Paare": len(verbindungen), "Bewertung": bewertung,
                "Gruende_unbewertet": "; ".join(gruende_unbewertet(verbindungen, labels)) if bewertung == "unbewertet" else "",
            }
            joint_zeilen.append(row)
            for verbindung in verbindungen:
                zuordnung[verbindung].append(joint_id)
                status = "/".join(sorted(labels.get(verbindung, set())))
                paar_zeilen.append({
                    "Modell": stem, "Joint_ID": joint_id, "GUID_A": verbindung[0],
                    "GUID_B": verbindung[1], "RBC_Label": status or "nicht vorhanden",
                })
        if labels is not None:
            nicht_zugeordnet = Counter()
            fehlende_soll_paare = sorted(set(zuordnung) - labels.keys())
            fp_auf_soll = sum(bool(werte & {"FP", "TN"}) for verbindung, werte in labels.items() if verbindung in zuordnung)
            for verbindung in fehlende_soll_paare:
                warnungen.append((
                    stem, f"Soll-Paar {verbindung} hat kein RBC-Label; "
                    f"betroffene Joints: {', '.join(zuordnung[verbindung])}",
                ))
            for verbindung, werte in sorted(labels.items()):
                # FP werden ausdrücklich nicht auf Soll-Joints übertragen.
                joint_ids = zuordnung.get(verbindung, []) if werte in ({"TP"}, {"FN"}) else []
                if not joint_ids:
                    nicht_zugeordnet["/".join(sorted(werte))] += 1
                if verbindung in zuordnung and werte & {"FP", "TN"}:
                    warnungen.append((stem, f"Soll-Paar {verbindung} hat RBC-Label {sorted(werte)}; nicht als TP/FN verwendbar"))
                paar_zeilen.append({
                    "Modell": stem, "Joint_ID": ",".join(joint_ids),
                    "GUID_A": verbindung[0], "GUID_B": verbindung[1],
                    "RBC_Label": "/".join(sorted(werte)),
                    "Quelle": "RBC",
                })
            if nicht_zugeordnet:
                warnungen.append((stem, f"RBC-Paare ohne Joint-Zuordnung (FP grundsätzlich ausgeschlossen): {dict(nicht_zugeordnet)}"))
            m = metriken(labels)
            eingeschraenkt = (
                zeilen["ungueltige_Paarzeilen"] or zeilen["unbekanntes_Label"]
                or fehlende_soll_paare or fp_auf_soll or m["widerspruechliche_Paare"]
            )
            metrik_zeilen.append({
                "Modell": stem, **m,
                "ungueltige_Paarzeilen": zeilen["ungueltige_Paarzeilen"],
                "unbekannte_Labels": zeilen["unbekanntes_Label"],
                "fehlende_Soll_Paare": len(fehlende_soll_paare),
                "FP_TN_auf_Soll_Paaren": fp_auf_soll,
                "Datenqualitaet": "eingeschränkt" if eingeschraenkt else "ohne erkannte Einschränkung",
            })

    # CSV-Zeilen für Soll-Paare und RBC-Paare sind bewusst getrennt.
    for row in paar_zeilen:
        row.setdefault("Quelle", "Soll-Joint")
    zaehlung = klassen_zaehlen(joint_zeilen)
    klassen_zeilen = []
    for (achse, klasse), counts in sorted(zaehlung.items()):
        bewertet = counts["korrekt"] + counts["falsch"]
        klassen_zeilen.append({
            "Achse": achse, "Klasse": klasse, "Gesamt": sum(counts.values()),
            **{status: counts[status] for status in STATUSES},
            "Korrektanteil_bewertet": counts["korrekt"] / bewertet if bewertet else None,
        })

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    csv_schreiben(DIR_OUT / "Joint_Einzelbewertung.csv",
                  ["Modell", "Joint_ID", *ACHSEN, "Shortcode", "Soll_Paare", "Bewertung", "Gruende_unbewertet"], joint_zeilen)
    csv_schreiben(DIR_OUT / "Joint_Klassen.csv",
                  ["Achse", "Klasse", "Gesamt", *STATUSES, "Korrektanteil_bewertet"], klassen_zeilen)
    csv_schreiben(DIR_OUT / "Paar_Zuordnung.csv",
                  ["Modell", "Quelle", "Joint_ID", "GUID_A", "GUID_B", "RBC_Label"], paar_zeilen)
    gesamt = {
        "Modell": "Gesamt (modellweise)", **metriken(gesamt_labels),
        "ungueltige_Paarzeilen": sum(z["ungueltige_Paarzeilen"] for z in metrik_zeilen),
        "unbekannte_Labels": sum(z["unbekannte_Labels"] for z in metrik_zeilen),
        "fehlende_Soll_Paare": sum(z["fehlende_Soll_Paare"] or 0 for z in metrik_zeilen),
        "FP_TN_auf_Soll_Paaren": sum(z["FP_TN_auf_Soll_Paaren"] or 0 for z in metrik_zeilen),
        "Datenqualitaet": "eingeschränkt" if any(z["Datenqualitaet"] != "ohne erkannte Einschränkung" for z in metrik_zeilen) else "ohne erkannte Einschränkung",
    }
    csv_schreiben(DIR_OUT / "RBC_Metriken.csv", list(gesamt), [*metrik_zeilen, gesamt])
    csv_schreiben(DIR_OUT / "Warnungen.csv", ["Modell", "Warnung"],
                  [{"Modell": modell, "Warnung": text} for modell, text in warnungen])
    zeichne_klassen(zaehlung, DIR_OUT / "Joint_Klassen.png")
    zeichne_bewertung(zaehlung, DIR_OUT / "Joint_Bewertung.png")
    zeichne_metriken(metrik_zeilen, DIR_OUT / "RBC_Metriken.png")

    gesamt_status = Counter(row["Bewertung"] for row in joint_zeilen)
    protokoll = [
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}]",
        "==================================================",
        "RBC-ANALYSE UND JOINT-FEHLERTAXONOMIE",
        "==================================================",
        f"Soll-Joints: {DIR_JOINTS}",
        f"RBC-Labels: {DIR_RBC}",
        f"Untersuchte Modelle ({len(modelle)}): {', '.join(modelle)}",
        f"Modelle mit RBC-Datei ({len(metrik_zeilen)}): {', '.join(z['Modell'] for z in metrik_zeilen) or 'keine'}",
        f"Übersprungene Modelle ({len(uebersprungen)}): {', '.join(stem for stem, _ in uebersprungen) or 'keine'}",
        "",
        "METHODIK",
        "Datensatz und RBC-Metriken: nur Modelle mit beiden gleichnamigen Eingabedateien.",
        "Modelle ohne eine oder beide Dateien werden vollständig übersprungen und in Warnungen.csv aufgeführt.",
        "Klassifikation: einzelne Merkmale, kategoriale Kombination ohne kontinuierliche Innendichte p und unveränderter Shortcode.",
        "RBC-Metriken: eindeutige ungerichtete GUID-Paare pro Modell; CSV-Labels unverändert.",
        "Ungültige GUIDs, gleiche GUIDs und widersprüchliche Labels sind aus den Paarmetriken ausgeschlossen.",
        "Metriken eingeschränkter Modelle und deren Gesamtwert sind wegen fehlender/inkonsistenter Daten nicht belastbar.",
        "Auch bei anderen Modellen wurden CSV-Labels nicht unabhängig anhand der Ground Truth geprüft.",
        "Mehrfache CSV-Zeilen desselben Paares werden einmal gezählt. Modelle werden für Gesamtwerte getrennt gezählt.",
        "TN nur falls explizit in CSV vorhanden; keine vollständige negative Grundgesamtheit, keine Accuracy.",
        "Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN); Nenner 0: nicht definiert.",
        "Joint: mindestens ein FN = falsch; alle Soll-Paare TP = korrekt; sonst unbewertet.",
        "FP werden keinem Soll-Joint zugeordnet. Mehrfachzuordnung eines Soll-Paares wird beibehalten.",
        "Die Bewertung basiert auf Bauteilpaaren, nicht auf einem räumlichen Soll-Ist-Joint-Matching.",
        "",
        "1. DATENSATZ UND JOINT-BEWERTUNG",
        "| Modell | Soll-Joints | korrekt | falsch | unbewertet |",
    ]
    for stem in modelle:
        counts = Counter(row["Bewertung"] for row in joint_zeilen if row["Modell"] == stem)
        protokoll.append(f"| {stem} | {sum(counts.values())} | {counts['korrekt']} | {counts['falsch']} | {counts['unbewertet']} |")
    protokoll.append(f"| Gesamt | {len(joint_zeilen)} | {gesamt_status['korrekt']} | {gesamt_status['falsch']} | {gesamt_status['unbewertet']} |")
    protokoll.extend([
        "",
        "Ursachen unbewerteter Joints (Mehrfachnennungen möglich):",
        "| Modell | kein RBC-Label | FP/TN auf Soll-Paar | widersprüchliche Labels |",
    ])
    for stem in modelle:
        counts = Counter(
            grund
            for row in joint_zeilen if row["Modell"] == stem and row["Bewertung"] == "unbewertet"
            for grund in row["Gruende_unbewertet"].split("; ") if grund
        )
        protokoll.append(
            f"| {stem} | {counts['kein RBC-Label']} | {counts['FP/TN auf Soll-Paar']} "
            f"| {counts['widersprüchliche Labels']} |"
        )
    protokoll.extend(["", "2. RBC-LEISTUNGSMETRIKEN", "| Modell | TP | FP | FN | TN (CSV) | Precision | Recall | F1 | Datenqualität |"])
    for row in [*metrik_zeilen, gesamt]:
        protokoll.append(
            f"| {row['Modell']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN_nur_CSV']} | "
            f"{format_wert(row['Precision'])} | {format_wert(row['Recall'])} | {format_wert(row['F1'])} | {row['Datenqualitaet']} |"
        )
    protokoll.extend(["", "Datenqualität je Modell: | Modell | ungültige Paarzeilen | unbekannte Labels | fehlende Soll-Paare | FP/TN auf Soll-Paaren |"])
    for row in metrik_zeilen:
        fehlend = row["fehlende_Soll_Paare"]
        widerspruch = row["FP_TN_auf_Soll_Paaren"]
        protokoll.append(
            f"| {row['Modell']} | {row['ungueltige_Paarzeilen']} | {row['unbekannte_Labels']} | "
            f"{fehlend if fehlend is not None else 'nicht geprüft'} | "
            f"{widerspruch if widerspruch is not None else 'nicht geprüft'} |"
        )
    protokoll.extend(["", "3. CSV-ZEILEN VOR PAARBEREINIGUNG"])
    for stem, counts in rbc_zeilen.items():
        protokoll.append(f"{stem}: {dict(counts)}")
    protokoll.extend([
        "", "4. ZUSAMMENFASSUNG UND HINWEISE",
        f"{len(joint_zeilen)} Soll-Joints: {gesamt_status['korrekt']} korrekt, "
        f"{gesamt_status['falsch']} falsch, {gesamt_status['unbewertet']} unbewertet.",
        f"Warnungen: {len(warnungen)}; Details siehe Warnungen.csv.",
        "Vollständige Klassenhäufigkeiten und Bewertungen: Joint_Klassen.csv.",
        "Einzelne Zuordnungen und nicht zugeordnete RBC-Paare: Paar_Zuordnung.csv.",
        "Alle Diagramme und Tabellen liegen im Ausgabeordner RBC_Analyse.",
    ])
    with (DIR_OUT / "Protokoll RBC_Analyse.txt").open("w", encoding="utf-8") as datei:
        datei.write("\n".join(protokoll) + "\n")
    print("\n".join(protokoll))


if __name__ == "__main__":
    main()
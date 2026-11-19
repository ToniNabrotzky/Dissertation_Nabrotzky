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

# Python-Standardbibliotheken für Dateiverarbeitung, Datenformate, Suche und Statistik importieren.
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Matplotlib für die serverfähige, nicht interaktive Diagrammerzeugung einrichten.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# KONFIGURATION
# =========================================================
# Eingabe- und Ausgabeverzeichnisse aus der Position des Skripts ableiten.
BASE_DIR = Path(__file__).resolve().parent.parent
DIR_JOINTS = BASE_DIR / "1 Data_Preprocessing" / "4 Graph_Generation"
DIR_RBC = BASE_DIR / "1 Data_Preprocessing" / "3 RBC_Labels"
DIR_OUT = Path(__file__).resolve().parent / "RBC_Analyse"

# Dateinamenmuster, Modellfilter und Formatregele für IFC-GUIDs festlegen.
# Leere Liste: alle Modelle mit beiden Dateien in den Eingangsordnern.
MODELLE = []
JOINT_SUFFIX = " Soll_Joints.json"
RBC_SUFFIX = " RFEM_Labels.csv"
GUID_RE = re.compile(r"^[0-3][0-9A-Za-z_$]{21}$")

# Darstellungsachsen den Feldern der Joint-Klassifikation zuordnen.
ACHSEN = {
    # Achse 1: Valenz
    "Valenz": "valence_class",
    "Heterogenität": "h",
    "Innendichte": "p",
    # Achse 2: Versatz
    "Versatz": "o_class",
    "Dimensionssprung": "gamma_max",
    # Achse 3: Morphologie
    "Kontakt": "k_max",
    "Konfiguration": "configuration",
    "Winkel": "angle_class",
}

# Mögliche Bewertungszustände in einheitlicher Reihenfolge definieren.
STATUSES = ("korrekt", "falsch", "unbewertet")


def paar(guid_a, guid_b):
    """Gibt ein gültiges, ungerichtetes IFC-Paar oder None zurück."""
    # Eingaben normalisieren und ungültige oder identische GUIDs ausschließen.
    a, b = str(guid_a or "").strip(), str(guid_b or "").strip()
    if not GUID_RE.fullmatch(a) or not GUID_RE.fullmatch(b) or a == b:
        return None

    # Ein kanonisches, richtungsunabhängiges Paar für spätere Zuordnungen erzeugen.
    return tuple(sorted((a, b)))


def lese_rbc(path, modell, warnungen):
    """Liest CSV-Labels und bewahrt widersprüchliche Labels je Paar."""
    # Zähler und Sammelbehälter für Paarlabels initialisieren.
    labels = defaultdict(set)
    zeilen = Counter()

    # CSV-Datei mit passendem Trennzeichen öffnen und erforderliche Spalten prüfen.
    with path.open(encoding="utf-8-sig", newline="") as datei:
        reader = csv.DictReader(datei, delimiter=";")
        if not reader.fieldnames or "IFC_GUID_A" not in reader.fieldnames:
            datei.seek(0)
            reader = csv.DictReader(datei, delimiter=",")
        required = {"IFC_GUID_A", "IFC_GUID_B", "Label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: erforderliche Spalten fehlen: {required}")

        # Jede Datenzeile validieren, protokollieren und als Paarlabel erfassen.
        for nr, row in enumerate(reader, 2):
            label = (row.get("Label") or "").strip().upper()
            verbindung = paar(row.get("IFC_GUID_A"), row.get("IFC_GUID_B"))
            if label not in {"TP", "FP", "FN", "TN"}:
                zeilen["unbekanntes_Label"] += 1
                warnungen.append((modell, f"{path.name}, Zeile {nr}: unbekanntes Label {label!r}"))
                continue
            zeilen[label] += 1
            if verbindung is None:
                zeilen["ungültige_Paarzeilen"] += 1
                warnungen.append((
                    modell,
                    f"{path.name}, Zeile {nr}: fehlende/ungültige oder gleiche GUIDs "
                    f"({row.get('IFC_GUID_A')!r}, {row.get('IFC_GUID_B')!r}); nicht als Paar auswertbar",
                ))
                continue
            labels[verbindung].add(label)

    # Widersprüchliche Mehrfachlabels als Datenqualitätsproblem festhalten.
    for verbindung, werte in labels.items():
        if len(werte) > 1:
            warnungen.append((modell, f"Widersprüchliche Labels für {verbindung}: {sorted(werte)}"))

    # Normalisierte Labels und Zeilenstatistik an die weitere Auswertung übergeben.
    return labels, zeilen


def lade_joints(path):
    """Prüft die Referenzen und liefert Joint-Klassen und Soll-Paare."""
    # JSON-Datei laden und die strukturell benötigten Felder kontrollieren.
    with path.open(encoding="utf-8-sig") as datei:
        daten = json.load(datei)
    for feld in ("model_stem", "contact_pairs", "soll_joints", "joint_classes"):
        if feld not in daten:
            raise ValueError(f"{path}: Feld {feld} fehlt")

    # Kontaktpaare anhand ihrer eindeutigen IDs für spätere Referenzen indexieren.
    regionen = {}
    for region in daten["contact_pairs"]:
        pair_id = region["pair_id"]
        verbindung = paar(region["guid_a"], region["guid_b"])
        if verbindung is None or pair_id in regionen:
            raise ValueError(f"{path}: ungültiges oder doppeltes Kontaktpaar {pair_id}")
        regionen[pair_id] = verbindung

    # Joint-Klassen anhand ihrer IDs indexieren und Duplikate verhindern.
    klassen = {}
    for klasse in daten["joint_classes"]:
        joint_id = klasse["joint_id"]
        if joint_id in klassen:
            raise ValueError(f"{path}: doppelte Joint-Klasse {joint_id}")
        klassen[joint_id] = klasse

    # Soll-Joints validieren, Klassifikation ergänzen und konkrete GUID-Paare auflösen.
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

    # Sicherstellen, dass jede Klasse genau einem Soll-Joint zugeordnet ist.
    if ids != set(klassen):
        raise ValueError(f"{path}: Klassen ohne Soll-Joint: {sorted(set(klassen) - ids)}")

    # Modellnamen und validierte Joint-Struktur zur Auswertung zurückgeben.
    return daten["model_stem"], joints


def bewerte_joint(verbindungen, labels):
    """Falsch hat Vorrang vor unbewertet; FP werden nicht zugeordnet."""
    # Ohne Labels keine belastbare Joint-Bewertung vornehmen.
    if labels is None:
        return "unbewertet"

    # Labelmengen aller Soll-Paare für die Entscheidungsregeln sammeln.
    zustaende = [labels.get(verbindung, set()) for verbindung in verbindungen]

    # Ein eindeutiger FN-Eintrag macht den Joint falsch und hat Vorrang.
    if any(wert == {"FN"} for wert in zustaende):
        return "falsch"

    # Nur ausschließlich eindeutige TP-Labels machen den Joint korrekt.
    if all(wert == {"TP"} for wert in zustaende):
        return "korrekt"

    # Fehlende, widersprüchliche oder andere Labels als unbewertet behandeln.
    return "unbewertet"


def gruende_unbewertet(verbindungen, labels):
    """Nennt die nicht exklusiven Ursachen für eine unvollständige Bewertung."""
    # Labelzustände der Soll-Paare für die Ursachensuche bestimmen.
    zustaende = [labels.get(verbindung, set()) for verbindung in verbindungen]
    gruende = []

    # Jede auftretende Ursache unabhängig ergänzen, um Mehrfachnennungen zu erlauben.
    if any(not wert for wert in zustaende):
        gruende.append("kein RBC-Label")
    if any(wert & {"FP", "TN"} for wert in zustaende):
        gruende.append("FP/TN auf Soll-Paar")
    if any(len(wert) > 1 for wert in zustaende):
        gruende.append("widersprüchliche Labels")

    # Gesammelte Ursachen für Bericht und CSV zurückgeben.
    return gruende


def metriken(labels):
    """Metriken über auswertbare eindeutige Paare; nicht definierte Werte sind leer."""
    # Nur Paare mit genau einem Label in die Kennzahlenberechnung einbeziehen.
    zaehler = Counter(next(iter(werte)) for werte in labels.values() if len(werte) == 1)
    tp, fp, fn = (zaehler[x] for x in ("TP", "FP", "FN"))

    # Quotienten bei nicht definiertem Nenner bewusst als fehlenden Wert darstellen.
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None

    # Zählungen, Metriken und Qualitätsindikatoren gemeinsam zurückgeben.
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN_nur_CSV": zaehler["TN"],
        "widerspruechliche_Paare": sum(len(werte) > 1 for werte in labels.values()),
        "Precision": precision, "Recall": recall, "F1": f1,
    }


def csv_schreiben(path, spalten, zeilen):
    # Zeilen im festgelegten Spaltenformat als UTF-8-CSV mit Semikolontrennung speichern.
    with path.open("w", encoding="utf-8-sig", newline="") as datei:
        writer = csv.DictWriter(datei, fieldnames=spalten, delimiter=";")
        writer.writeheader()
        writer.writerows(zeilen)


def klassen_zaehlen(joints):
    """Zählt Achsen, kategoriale Kombinationen und unveränderte Shortcodes."""
    # Bewertungszustände getrennt nach Einzelachsen und zusammengesetzten Klassen zählen.
    result = defaultdict(Counter)
    for joint in joints:
        for achse in ACHSEN:
            result[(achse, joint[achse])][joint["Bewertung"]] += 1
        kombination = " | ".join(
            f"{achse}={joint[achse]}" for achse in ACHSEN if achse != "Innendichte"
        )
        result[("Kombination", kombination)][joint["Bewertung"]] += 1
        result[("Shortcode", joint["Shortcode"])][joint["Bewertung"]] += 1

    # Aggregierte Klassenhäufigkeiten für CSV und Diagramme zurückgeben.
    return result


def zeichne_klassen(zaehlung, path):
    # Diagrammfläche mit einem Teilplot je Klassifikationsachse anlegen.
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)

    # Klassenverteilung je Achse als horizontales Balkendiagramm darstellen.
    for ax, achse in zip(axes.flat, ACHSEN):
        daten = sorted((klasse, counts) for (a, klasse), counts in zaehlung.items() if a == achse)
        namen = [klasse for klasse, _ in daten]
        werte = [sum(counts.values()) for _, counts in daten]
        ax.barh(namen, werte, color="#4472a1")
        ax.set_title(achse)
        ax.set_xlabel("Anzahl Soll-Joints")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#cccccc", linewidth=0.8)
        ax.invert_yaxis()

    # Überschrift ergänzen, Bild speichern und Grafikressourcen freigeben.
    fig.suptitle("Datensatz: Verteilung der Joint-Klassifikationen")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def zeichne_bewertung(zaehlung, path):
    # Diagrammfläche und gemeinsame Bewertungsfarben vorbereiten.
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
    farben = {"korrekt": "#4c9f70", "falsch": "#d35f5f", "unbewertet": "#a7a7a7"}

    # Bewertungsanteile je Klassenausprägung als gestapelte Balken zeichnen.
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
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#cccccc", linewidth=0.8)
        ax.invert_yaxis()

    # Gemeinsame Legende und Titel setzen und die fertige Abbildung speichern.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=3)
    fig.suptitle("RBC-Bewertung der Soll-Joints nach Klassifikation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def zeichne_metriken(zeilen, path):
    # Bei fehlenden Modellzeilen keine leere Grafik erzeugen.
    if not zeilen:
        return

    # Flexible Diagrammfläche und Balkenbreite für Modellvergleiche vorbereiten.
    fig, ax = plt.subplots(figsize=(max(8, len(zeilen) * 1.3), 5), constrained_layout=True)
    breite = 0.25

    # Precision, Recall und F1 nur für definierte Werte nebeneinander plotten.
    for offset, name in enumerate(("Precision", "Recall", "F1")):
        # Nicht definierte Metriken erhalten keinen Balken, nicht den Wert null.
        xs = [i + (offset - 1) * breite for i, z in enumerate(zeilen) if z[name] is not None]
        werte = [z[name] for z in zeilen if z[name] is not None]
        ax.bar(xs, werte, breite, label=name)

    # Achsen, Warnmarkierung, Legende und Titel des Metrikvergleichs setzen.
    ax.set_xticks(
        range(len(zeilen)),
        [z["Modell"] + (" *" if z["Datenqualitaet"] != "ohne erkannte Einschränkung" else "") for z in zeilen],
        rotation=30, ha="right",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Wert")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#cccccc", linewidth=0.8)
    ax.set_title("RBC-Paarmetriken (* eingeschränkte Daten; nicht definiert: kein Balken)")
    ax.legend()

    # Metrikdiagramm speichern und Ressourcen schließen.
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_wert(wert):
    # Fehlende Metriken verständlich benennen und vorhandene Werte einheitlich formatieren.
    return "nicht definiert" if wert is None else f"{wert:.4f}"


def main():
    # Vor Beginn sicherstellen, dass beide Eingabeverzeichnisse vorhanden sind.
    if not DIR_JOINTS.is_dir() or not DIR_RBC.is_dir():
        raise FileNotFoundError(f"Eingangsordner fehlen: {DIR_JOINTS} bzw. {DIR_RBC}")

    # Warnungsbehälter und Dateizuordnungen nach Modellnamen aufbauen.
    warnungen = []
    joint_dateien = {
        path.name[:-len(JOINT_SUFFIX)]: path for path in DIR_JOINTS.glob(f"*{JOINT_SUFFIX}")
    }
    rbc_dateien = {
        path.name[:-len(RBC_SUFFIX)]: path for path in DIR_RBC.glob(f"*{RBC_SUFFIX}")
    }

    # Zu untersuchende Kandidaten aus Filterliste oder vorhandenen Dateien bestimmen.
    kandidaten = set(MODELLE) if MODELLE else set(joint_dateien) | set(rbc_dateien)
    uebersprungen = []

    # Kandidaten mit fehlender Gegenstückdatei protokollieren und überspringen.
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

    # Schnittmenge vollständig auswertbarer Modelle bilden und leere Auswertung verhindern.
    gemeinsame_modelle = sorted(kandidaten & joint_dateien.keys() & rbc_dateien.keys())
    if not gemeinsame_modelle:
        raise FileNotFoundError(f"Keine Modelle mit beiden Eingabedateien. {warnungen}")

    # Gemeinsame Ergebniscontainer für modellweise und globale Auswertung vorbereiten.
    joint_zeilen, paar_zeilen, metrik_zeilen = [], [], []
    rbc_zeilen = {}
    modelle = []
    gesamt_labels = {}

    # Jedes vollständige Modell laden, bewerten und mit den RBC-Paaren zusammenführen.
    for stem in gemeinsame_modelle:
        # Modellstamm aus Datei validieren und das Modell als verarbeitet markieren.
        path = joint_dateien[stem]
        model_stem, joints = lade_joints(path)
        if model_stem != stem:
            raise ValueError(f"{path}: model_stem {model_stem!r} stimmt nicht mit Dateiname überein")
        modelle.append(stem)

        # RBC-Labels lesen und für modellübergreifende Metriken eindeutig qualifizieren.
        labels, zeilen = lese_rbc(rbc_dateien[stem], stem, warnungen)
        rbc_zeilen[stem] = zeilen
        gesamt_labels.update({(stem, *pair): werte for pair, werte in labels.items()})

        # Zuordnungsinformationen für Joint- und Paarberichte sammeln.
        zuordnung = defaultdict(list)

        # Jeden Soll-Joint bewerten und seine Klassifikation sowie Paarzuordnung ausgeben.
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

        # Paarweise Datenqualität und nicht zugeordnete RBC-Paare zusätzlich prüfen.
        if labels is not None:
            nicht_zugeordnet = Counter()
            fehlende_soll_paare = sorted(set(zuordnung) - labels.keys())
            fp_auf_soll = sum(bool(werte & {"FP", "TN"}) for verbindung, werte in labels.items() if verbindung in zuordnung)

            # Fehlende Label für referenzierte Soll-Paare als Warnung dokumentieren.
            for verbindung in fehlende_soll_paare:
                warnungen.append((
                    stem, f"Soll-Paar {verbindung} hat kein RBC-Label; "
                    f"betroffene Joints: {', '.join(zuordnung[verbindung])}",
                ))

            # RBC-Zeilen mit erlaubter Zuordnung in die Paarübersicht übernehmen.
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

            # Nicht zugeordnete Paare zusammenfassen und Modellmetriken berechnen.
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

    # Herkunft der Soll-Paarzeilen markieren, bevor beide Quellarten kombiniert werden.
    # CSV-Zeilen für Soll-Paare und RBC-Paare sind bewusst getrennt.
    for row in paar_zeilen:
        row.setdefault("Quelle", "Soll-Joint")

    # Klassenhäufigkeiten in Zeilen für die zusammengefasste CSV überführen.
    zaehlung = klassen_zaehlen(joint_zeilen)
    klassen_zeilen = []
    for (achse, klasse), counts in sorted(zaehlung.items()):
        bewertet = counts["korrekt"] + counts["falsch"]
        klassen_zeilen.append({
            "Achse": achse, "Klasse": klasse, "Gesamt": sum(counts.values()),
            **{status: counts[status] for status in STATUSES},
            "Korrektanteil_bewertet": counts["korrekt"] / bewertet if bewertet else None,
        })

    # Ausgabeordner anlegen und alle tabellarischen Analyseergebnisse speichern.
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    csv_schreiben(DIR_OUT / "Joint_Einzelbewertung.csv",
                  ["Modell", "Joint_ID", *ACHSEN, "Shortcode", "Soll_Paare", "Bewertung", "Gruende_unbewertet"], joint_zeilen)
    csv_schreiben(DIR_OUT / "Joint_Klassen.csv",
                  ["Achse", "Klasse", "Gesamt", *STATUSES, "Korrektanteil_bewertet"], klassen_zeilen)
    csv_schreiben(DIR_OUT / "Paar_Zuordnung.csv",
                  ["Modell", "Quelle", "Joint_ID", "GUID_A", "GUID_B", "RBC_Label"], paar_zeilen)

    # Modellweise Qualitäts- und Fehlerzähler zu einer Gesamtzeile verdichten.
    gesamt = {
        "Modell": "Gesamt (modellweise)", **metriken(gesamt_labels),
        "ungueltige_Paarzeilen": sum(z["ungueltige_Paarzeilen"] for z in metrik_zeilen),
        "unbekannte_Labels": sum(z["unbekannte_Labels"] for z in metrik_zeilen),
        "fehlende_Soll_Paare": sum(z["fehlende_Soll_Paare"] or 0 for z in metrik_zeilen),
        "FP_TN_auf_Soll_Paaren": sum(z["FP_TN_auf_Soll_Paaren"] or 0 for z in metrik_zeilen),
        "Datenqualitaet": "eingeschränkt" if any(z["Datenqualitaet"] != "ohne erkannte Einschränkung" for z in metrik_zeilen) else "ohne erkannte Einschränkung",
    }

    # Metriken und Warnungen sowie alle Diagramme in den Ausgabeordner schreiben.
    csv_schreiben(DIR_OUT / "RBC_Metriken.csv", list(gesamt), [*metrik_zeilen, gesamt])
    csv_schreiben(DIR_OUT / "Warnungen.csv", ["Modell", "Warnung"],
                  [{"Modell": modell, "Warnung": text} for modell, text in warnungen])
    zeichne_klassen(zaehlung, DIR_OUT / "Joint_Klassen.png")
    zeichne_bewertung(zaehlung, DIR_OUT / "Joint_Bewertung.png")
    zeichne_metriken(metrik_zeilen, DIR_OUT / "RBC_Metriken.png")

    # Gesamtbewertung und statischen Kopf des textlichen Analyseprotokolls vorbereiten.
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

    # Modellweise Bewertungsverteilungen als Markdown-Tabelle an das Protokoll anhängen.
    for stem in modelle:
        counts = Counter(row["Bewertung"] for row in joint_zeilen if row["Modell"] == stem)
        protokoll.append(f"| {stem} | {sum(counts.values())} | {counts['korrekt']} | {counts['falsch']} | {counts['unbewertet']} |")
    protokoll.append(f"| Gesamt | {len(joint_zeilen)} | {gesamt_status['korrekt']} | {gesamt_status['falsch']} | {gesamt_status['unbewertet']} |")

    # Ursachen unbewerteter Joints mit Mehrfachnennungen zusammenfassen.
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

    # Tabellenabschnitt der Paarmetriken für alle Modelle und den Gesamtwert ergänzen.
    protokoll.extend(["", "2. RBC-LEISTUNGSMETRIKEN", "| Modell | TP | FP | FN | TN (CSV) | Precision | Recall | F1 | Datenqualität |"])
    for row in [*metrik_zeilen, gesamt]:
        protokoll.append(
            f"| {row['Modell']} | {row['TP']} | {row['FP']} | {row['FN']} | {row['TN_nur_CSV']} | "
            f"{format_wert(row['Precision'])} | {format_wert(row['Recall'])} | {format_wert(row['F1'])} | {row['Datenqualitaet']} |"
        )

    # Datenqualitätsindikatoren je Modell im Bericht ausweisen.
    protokoll.extend(["", "Datenqualität je Modell: | Modell | ungültige Paarzeilen | unbekannte Labels | fehlende Soll-Paare | FP/TN auf Soll-Paaren |"])
    for row in metrik_zeilen:
        fehlend = row["fehlende_Soll_Paare"]
        widerspruch = row["FP_TN_auf_Soll_Paaren"]
        protokoll.append(
            f"| {row['Modell']} | {row['ungueltige_Paarzeilen']} | {row['unbekannte_Labels']} | "
            f"{fehlend if fehlend is not None else 'nicht geprüft'} | "
            f"{widerspruch if widerspruch is not None else 'nicht geprüft'} |"
        )

    # Ursprüngliche CSV-Zeilenstatistik vor Bereinigung in das Protokoll aufnehmen.
    protokoll.extend(["", "3. CSV-ZEILEN VOR PAARBEREINIGUNG"])
    for stem, counts in rbc_zeilen.items():
        protokoll.append(f"{stem}: {dict(counts)}")

    # Abschließende Kernergebnisse und Verweise auf erzeugte Dateien zusammenstellen.
    protokoll.extend([
        "", "4. ZUSAMMENFASSUNG UND HINWEISE",
        f"{len(joint_zeilen)} Soll-Joints: {gesamt_status['korrekt']} korrekt, "
        f"{gesamt_status['falsch']} falsch, {gesamt_status['unbewertet']} unbewertet.",
        f"Warnungen: {len(warnungen)}; Details siehe Warnungen.csv.",
        "Vollständige Klassenhäufigkeiten und Bewertungen: Joint_Klassen.csv.",
        "Einzelne Zuordnungen und nicht zugeordnete RBC-Paare: Paar_Zuordnung.csv.",
        "Alle Diagramme und Tabellen liegen im Ausgabeordner RBC_Analyse.",
    ])

    # Protokoll persistent speichern und identischen Inhalt auf der Konsole anzeigen.
    with (DIR_OUT / "Protokoll RBC_Analyse.txt").open("w", encoding="utf-8") as datei:
        datei.write("\n".join(protokoll) + "\n")
    print("\n".join(protokoll))


if __name__ == "__main__":
    # Skriptablauf nur bei direkter Ausführung starten.
    main()
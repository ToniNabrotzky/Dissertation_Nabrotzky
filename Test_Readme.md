# Diss_Versuchsreihe_2 RBC-GNN-Konvertierung

Dieses Repository beschreibt, ob ein Graph Neural Network (GNN) Verbindungen zwischen tragenden IFC-Bauteilen erkennen und damit automatisierte Regeln zur Modellkonvertierung ergänzen kann. Die Ergebnisse sollen mit der Konvertierungsleistung von RBC verglichen werden.

## Ziel und Versuchsaufbau

Ein IFC-Modell wird als Graph dargestellt:

- **Knoten** sind tragende Bauteile mit IFC-Identität, Entitätstyp und geometrischen Merkmalen.
- **Kanten** stehen für mögliche Bauteilverbindungen. Geometrische Abstände liefern ein Kantenmerkmal; Ground-Truth-Labels geben an, ob das Bauteilpaar Last überträgt.
- Ein **GINE-basierter Edge Classifier** sagt für jede Kante eine binäre Lastübertragung (`load_transfer`) voraus.

Die allgemeinen Schritte sind: IFC-Modelle vorbereiten, Knoten- und Verbindungsinformationen extrahieren, Labels zuordnen, Trainingsgraphen erzeugen, GNN trainieren und die Vorhersagen gegen die Ground Truth auswerten. Zusätzlich wird in `2.2_Anwendungsgrenzen` ein regelbasierter Konvertierungsbenchmark (RBC) aus RFEM-Analysen erstellt, um die Regelresultate mit Ground Truth und GNN einordnen zu können.

## Ordnerstruktur

### `2.1_Anwendungsgrenzen`

| Ordner                   | Inhalt und Zweck                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------- |
| `0 Param_Modeling`     | Python-Skripte zur parametrischen Erzeugung von IFC-Modellen und zugehörige Protokolle.          |
| `1 Model_Database`     | Reale IFC-Modelle, parametrische Modellvarianten sowie Blender-, Bild- und Archivdateien.         |
| `2 Data_Preprocessing` | Extraktion und Aufbereitung der Trainingsgraphen in mehreren nummerierten Verarbeitungsschritten. |
| `3 Model_Training`     | GNN-Training, Inferenzanalyse und Sanity-Checks.                                                  |
| `4 Results`            | Trainingslogs, Sanity-Check-Protokolle, Auswertungen und Metrikplots.                             |

Der Preprocessing-Ordner enthält unter anderem IFC-Knotenextraktion, Distanzmessungen, SAF-/RFEM-Analyse, Label-Erzeugung und die Zusammenführung zum gelabelten Graph. Die begleitende Datei [`Dokumentenstruktur.txt`](<2.1_Anwendungsgrenzen/2%20Data_Preprocessing/Dokumentenstruktur.txt>) beschreibt den vorgesehenen Ablauf und die erwarteten Zwischenergebnisse.

### `2.2_Anwendungsgrenzen`

| Ordner / Datei               | Inhalt und Zweck                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `0 Model_Database`         | Reale IFC-Modelle, Modellfamilien `model_1` bis `model_10`, parametrische Varianten sowie archivierte Modelle. |
| `1 Data_Preprocessing`     | Ground Truth, RFEM-Analysemodelle, RBC-Graphen und Labels sowie erzeugte Trainingsgraphen und Hilfsskripte.        |
| `3 Model_Training`         | GNN-Klassifikator, Evaluator und Trainingsläufe für unterschiedliche Datenbankzusammenstellungen.                |
| `4 Results`                | Auswertungsplots, RFEM-Regelmetriken und Beispiele vorhergesagter Verbindungen.                                    |
| `Test_Distanzmessung`      | Vergleich von Distanzmessungsvarianten.                                                                            |
| `Test_RFEM_Regelgenerator` | Prototypen und Daten des RFEM-Regelgenerators.                                                                     |
| `_00 Workflow.md`          | Beschreibung der Datenbereinigung sowie des Ground-Truth-, RBC- und Graph-Aufbaus.                                 |

Die Nummerierung der Unterordner in `1 Data_Preprocessing` dokumentiert die Verarbeitungsschritte (Ground Truth, Analysemodell, RBC-Labels und Graph-Generation); sie ist nicht identisch mit der Nummerierung von `2.1`.

## Versuche und Unterschiede

### Datenmodelle

**2.1** kombiniert mehrere reale Tragwerksmodelle mit parametrisch erzeugten Varianten. Die parametrischen IFCs erlauben kontrollierte Änderungen an den Modellparametern; daneben liegen unter anderem unterschiedliche Projekt- und Beispielmodelle vor. Die Trainingslogs zeigen beispielsweise Läufe mit fünf realen Modellen (DB5) und mit zehn Modellen einschließlich parametrischer Varianten (DB10).

**2.2** ergänzt die realen und parametrischen Modelle um gezielt konstruierte Versuchsmodelle (`model_1` bis `model_10`), darunter explizit als `1D` und `2D` benannte Varianten. Dadurch können einfache bzw. geometrisch andersartige Anwendungsfälle separat betrachtet werden. Die Trainingsläufe verwenden verschiedene Zusammenstellungen, etwa DB5, DB9, DB13, DB14 und DB19; es handelt sich also nicht um einen einzigen festen Datensatz.

### Datenaufbereitung

In **2.1** wird die Verarbeitung als mehrstufige Pipeline organisiert:

1. Tragende IFC-Elemente und ihre Geometrie werden extrahiert.
2. Bauteilabstände werden gemessen.
3. SAF-/RFEM-Informationen liefern Verbindungs- bzw. Labeldaten.
4. Knoten, Abstände und Ground Truth werden zu Trainingsgraphen zusammengeführt.

Die aufbereiteten Graphen enthalten geometrische und semantische Knotenmerkmale, Abstand bzw. daraus abgeleitete `proximity_activation` als Kantenmerkmal sowie `load_transfer` als Zielwert.

In **2.2** ist der Ablauf stärker auf einen reproduzierbaren Vergleich zwischen manueller Ground Truth und Regelkonvertierung ausgerichtet. Laut
[`_00 Workflow.md`](<2.2_Anwendungsgrenzen/_00%20Workflow.md>) werden IFC-Modelle zunächst auf tragende Elemente bereinigt, die Ground Truth wird anhand der Lastflüsse erstellt und RFEM erzeugt die regelbasierten Kopplungen. Ein Analyser exportiert RBC-Graph und Labels; die Graph-Generation verbindet diese Informationen mit IFC-Merkmalen und Distanzen. Daneben untersuchen eigene Testskripte alternative Distanzmessungen und Varianten des RFEM-Regelgenerators.

### GNN-Training

Beide Bereiche verwenden in den dokumentierten Läufen überwiegend einen GINE-Edge-Classifier und Leave-One-Graph-Out (LOO): Je Durchlauf wird ein Modellgraph zurückgehalten und das Training mit den übrigen Graphen durchgeführt. Ein protokollierter gemeinsamer Basissatz umfasst 128 Hidden Units, fünf Message-Passing-Layer, ELU, Dropout 0,2, Lernrate 0,001, 500 Epochen, Batchgröße 12 und einen Entscheidungsschwellwert von 0,5. Die Skripte enthalten auch Optionen für andere Layer-Typen und Feature-Abwandlungen.

Der deutlichste Unterschied liegt in der **Merkmalsauswahl**:

- Die protokollierten Basisläufe in **2.1** verwenden ein breites Set mit Koordinaten (Origin, Centroid und normierte Position), Abmessungen,
  normierten Abmessungen, Fläche, Volumen, Flächen-Volumen-Verhältnis und IFC-Entitätsindex. Zusätzliche Versuche variieren unter anderem
  Hidden-Dimension, Layerzahl und Feature-Nutzung.
- Die späteren Versuche in **2.2** nutzen häufig ein reduziertes Set aus normierten Abmessungen, Fläche, Volumen, Flächen-Volumen-Verhältnis und
  IFC-Entitätsindex. Ein Schwerpunkt liegt auf Datensatz-Zusammenstellungen, aktualisierten Distanzen und Modellgruppen wie komplexen bzw. TUM-Modellen.

**Hinweis zur Interpretation der LOO-Ergebnisse:** In den vorliegenden Trainingsskripten wird der zurückgehaltene Graph während des Trainings wiederholt zur F1-basierten Auswahl des besten Checkpoints herangezogen. Die protokollierten LOO-Kennzahlen sind deshalb nicht als vollständig unabhängige Testleistung zu verstehen; die zurückgehaltenen Daten beeinflussen die Modellauswahl.

### Ergebnisse

Die Logs in **2.1** zeigen für einen DB5-Basislauf eine durchschnittliche Accuracy von rund **64,3 %** und einen F1-Score von **0,72**. Ein DB10-Lauf
protokolliert rund **64,6 % Accuracy** und **0,67 F1**. Die Ordner `4 Results` enthalten außerdem Varianten zu Trainingsparametern bzw. Features,
Sanity-Checks und Ergebnisplots.

Die Ergebnisse in **2.2** hängen sichtbar von der Datensatz-Auswahl ab:

- DB14 mit aktualisierten Distanzen: rund **65,3 % Accuracy**, **0,69 F1**.
- DB9 mit TUM-Modellen: rund **71,6 % Accuracy**, **0,71 F1**.
- Ein Lauf mit 19 Graphen: rund **57,3 % Accuracy**, **0,47 F1**.
- DB5 mit parametrischen Modellen: rund **60,9 % Accuracy**, **0,64 F1**.

Diese Zahlen sind Beispiele aus den vorhandenen Protokollen, keine direkte Rangliste: Datenbasis, Merkmalsauswahl und Versuchsfrage unterscheiden sich. Insbesondere zeigen die 1D-/2D-Versuche, dass eine hohe Accuracy allein irreführend sein kann; bei einzelnen Testgraphen stehen trotz hoher Accuracy F1-Werte von 0 im Log. Daher sollten F1, Precision und Recall sowie die Zusammensetzung des jeweiligen Datensatzes gemeinsam betrachtet werden.

## Kurzfazit

`2.1_Anwendungsgrenzen` legt die Grundlage für die IFC-zu-GNN-Pipeline und untersucht vor allem reale und parametrische Modelle sowie Merkmals- und Trainingsvarianten. `2.2_Anwendungsgrenzen` führt die systematische Ground-Truth- und RBC-Gegenüberstellung weiter und prüft mit zusätzlichen Modelltypen, Datensatzmischungen und Distanz-/Regelgeneratorvarianten gezielt, wo das Verfahren funktioniert oder an Grenzen stößt. Die Ergebnisse sind als explorative Versuchsprotokolle zu lesen und wegen der beschriebenen Checkpoint-Auswahl nicht als unabhängige Generalisierungsnachweise.

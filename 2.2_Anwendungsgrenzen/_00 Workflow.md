# Workflow-Beschreibung

## __PHASE__0: Bereinigung der Datenbasis

### Schritt 0: IFC-Modelle vorbereiten

Aus eigener, separater Datenbank IFC-Modelle als Strukturmodelle für die Anwendung des Workflows vorbereiten:

- Störende Elemente entfernen (architektonische/nicht-tragende Elemente und Fundamente)
- Bauteilattribuierung sicherstellen (Load-Bearing = True)
- (evtl. falsche Schichtaufbauten editieren/löschen)

Verwendung der Software Blender mit BonsaiBIM sowie dem Browsertool IfcLite.

## __PHASE__1: Referenzwahrheit (GT) und RBC-Benchmark erstellen

### Schritt 1: Ground-Truth

IFC-Modell manuell betrachten und die statischen Lastflüsse identifizieren.

Die GUIDs der verbundenen Elemente A und B als **IfcStem GT_Labels.csv** speichern.

### Schritt 2: Analysemodell erzeugen (RBC-Conversion)

IFC-Modell in RFEM importieren und auf Basis eines festen Regelsatzes automatisiert Bauteil-Kopplungen erzeugen lassen.

### Schritt 3: RBC-Labels

Python-Skript starten, um **IfcStem RFEM_Graph.json** zu erstellen und **IfcStem RFEM_Labels.csv** zu speichern.

- Für **RFEM_Labels** wird **GT_Labels** herangezogen, um das exakte Label (TP, FP, TN, FN) zu bestimmen. Später soll daraus eine Leistungsstatistik gebildet werden.

### Schritt 4: Graph erstellen

Schritt 4.1: Datenextraktion und semantische Filterung

- Die als lasttragend definierten Bauteile werden identifiziert und herausgefiltert.

Schritt 4.2: Knotendefinition und Merkmalsharmonisierung

- Die semantischen und geometrischen Merkmale werden von den Bauteilen extrahiert.
- Diese werden verarbeitet, um sie für das spätere GNN-Training zu harmonisieren.

Schritt 4.3: Kanteninitialisierung und räumliches Encoding

- Es wird eine Distanzberechnung zwischen den Bauteilen durchgeführt (erst grob mit BoundingBoxes, dann exakt mit Triangulation)
- Distanzen werden mithilfe von Radial Basis Function (RBF) weiterverarbeitet und als Kantenmerkmal gespeichert.

Schritt 4.4: Export
- Kanten werden mit exaktem Label (TP, FP, TN, FN) beschriftet. Dafür wird **GT_Labels** herangezogen.
- Statistik zum IFC-Modell wird erstellt.
- Statistik zum RBC-Benchmark wird mithilfe von **RFEM_Labels** erstellt.
- Statistik zum Preprocessing für das GNN-Training wird erstellt.
- Graph wird als **IfcStem Graph.json** gespeichert.


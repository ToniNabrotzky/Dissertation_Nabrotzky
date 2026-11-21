# Hybrid-RBC-GNN-Workflow

Dieses Repository beschreibt einen hybriden Workflow, der RBC-Konvertierung und Graph Neural Networks (GNNs) für die Untersuchung und Klassifizierung von Bauteilverbindungen in IFC-Modellen zusammenführt.

## Ziel

Der Workflow soll die Ergebnisse regelbasierter Konvertierung mit GNN-basierten Ergebnissen kombinieren. Dazu werden IFC-Modelle mit gezielt gestalteten geometrischen Anschlussstellen betrachtet. Für diese Fälle werden Soll-Joints gebildet und klassifiziert; Abweichungen werden im Rahmen einer Fehlertaxonomie eingeordnet.

GNNs werden gezielt auf den relevanten Fehlerstellen trainiert. Der Ansatz umfasst HeteroGNNs und Modell-Ensembles. Ihre Ergebnisse werden mit den RBC-Ergebnissen zusammengeführt und ausgewertet, um mögliche Leistungsfortschritte zu validieren.

## Grober Ablauf

1. IFC-Modelle mit ausgewählten geometrischen Anschlussstellen entwerfen.
2. RBC-Konvertierung durchführen, Soll-Joints bestimmen und klassifizieren.
3. Fehlerstellen und Fehlertypen untersuchen und für das GNN-Training nutzen.
4. HeteroGNNs und Modell-Ensembles trainieren und ihre Ergebnisse mit RBC vergleichen.
5. Die kombinierten Ergebnisse auswerten und mögliche Verbesserungen validieren.

## Ordnerstruktur

- `0 Model_DataBase` – IFC-Modelle und zugehörige Modelldaten.
- `1 Data_Preprocessing` – Aufbereitung der Daten für den Workflow.
- `2 GNN_Training` – Training und Anwendung der GNN-Ansätze.
- `3 Results` – Ergebnisse und Auswertungen.

Die Struktur dient der übergeordneten Einordnung des hybriden Ansatzes. Details zu einzelnen Versuchen und Verarbeitungsschritten sind in den jeweiligen Ordnern dokumentiert.

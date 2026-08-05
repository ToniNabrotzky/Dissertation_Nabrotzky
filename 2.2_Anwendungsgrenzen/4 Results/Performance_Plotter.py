import os
import matplotlib.pyplot as plt

# 1. Datenbasis
accuracy =  [65, 76, 56, 33, 73,
             73, 69, 75, 73,
             71, 65, 56, 64, 66,
             48, 62, 71, 68, 69, 73]
precision = [68, 76, 53, 33, 64,
             60, 54, 63, 70,
             61, 61, 52, 50, 63,
             42, 60, 68, 61, 54, 70]
recall =    [89, 100, 94, 100, 92,
             64, 80, 88, 92,
             60, 83, 99, 65, 81,
             97, 87, 93, 94, 80, 92]
f1_score =  [77, 87, 68, 50, 76,
             62, 64, 73, 79,
             60, 70, 68, 57, 71,
             59, 71, 79, 74, 64, 80]

if len(accuracy) != len(precision) or len(accuracy) != len(recall) or len(accuracy) != len(f1_score):
    print(f"Fehler: Datenbasis haben ungleiche Längen")
    pass

range_max = len(accuracy) + 1
models = [f"{i}" for i in range(1, range_max)] # Platzhalter für die X-Achse

# 2. Pfad-Architektur und Verzeichnismanagement
base_dir = os.path.dirname(__file__)
dir_results = os.path.join(base_dir, "4_1_Evaluation_Plots") # Beispieldirektory analog zu deiner Struktur

# Erstelle das Verzeichnis, falls es noch nicht existiert
os.makedirs(dir_results, exist_ok=True)
save_path = os.path.join(dir_results, "metrics_comparison.png")

# 3. Visualisierungs-Setup
plt.figure(figsize=(10, 6))

# Plot der 4 Metriken mit Kreuzen (marker='x') und durchgehenden Linien (linestyle='-')
plt.plot(models, accuracy, marker='x', linestyle='-', color='#1f77b4', label='Accuracy', markersize=8, linewidth=2)
plt.plot(models, precision, marker='x', linestyle='-', color='#ff7f0e', label='Precision', markersize=8, linewidth=2)
plt.plot(models, recall, marker='x', linestyle='-', color='#2ca02c', label='Recall', markersize=8, linewidth=2)
plt.plot(models, f1_score, marker='x', linestyle='-', color='#d62728', label='F1 Score', markersize=8, linewidth=2)

# 4. Wissenschaftliche Formatierung (Englisch)
plt.title('Performance Metrics Comparison per Model', fontsize=14, fontweight='bold')
plt.xlabel('Evaluated Models', fontsize=12)
plt.ylabel('Score [%]', fontsize=12)

# Achsenskalierung (0 bis 105% für Puffer am oberen Rand)
plt.ylim(0, 105) 

# Grid für bessere Lesbarkeit der exakten Werte
plt.grid(True, linestyle='--', alpha=0.6)

# Legende
plt.legend(loc='lower left', fontsize=12)

plt.tight_layout()

# 5. Export und Anzeige
plt.savefig(save_path, dpi=300, format='png') # 300 dpi für Publikationsqualität
# plt.show()
# plt.close()

print(f"Plot successfully saved to: {save_path}")
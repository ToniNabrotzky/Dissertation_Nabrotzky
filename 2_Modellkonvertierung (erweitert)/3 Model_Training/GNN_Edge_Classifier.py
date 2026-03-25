import copy
import json
import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, GATv2Conv, SAGEConv, NNConv
from torch_geometric.loader import DataLoader


# ==========================================
# KONFIGURATION (Modularer Aufbau)
# ==========================================
"""
    weight -> GNN Layer can use weight values on adjacency matrix
    type -> GNN Layer can use different edge types/relations
    attr -> GNN Layer can use edge features
    edge_weight: GCNConv, GCN2Conv
    edge_type: RGCNConv
    edge_attr: GATConv, GATv2Conv, GINEConv, NNConv
    nichts: EdgeConv, SAGEConv
"""
CONFIG = {
    # Siehe Doku: https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/nn.html
    # Auswahl: GATv2, GINE, NN
    "model_type": "GINE", # Default= GINE
    "hidden_dim": 128,   # Default= 128 | Sinvolle Annahmen: 32, 64, 128
    "num_layers": 5,    # Default= 5 | Anzahl der Message-Passing-Schichten
    # Auswahl: ELU, ReLU, LeakyReLU
    "activation": "ELU",
    "dropout": 0.2, # Default= 0.1
    "lr": 0.001, # Default= 0.001
    "epochs": 500,      # Default= 500 | Sanity Check braucht meist etwas länger zum Overfitten (1000-2000 Epochen)
    "edge_dim": 1,       # Default= 1 | Dimensionen der Kantenmerkmale (z.B. nur Distanz) | 0 deaktiviert Kantenmerkmale
    "batch_size": 12,
    "threshold": 0.5,    # Default= 0.5 | Sicherheit, die das Modell braucht um eine Kante als WAHR zu labeln.
    "model_save_dir": f"Test_00_DB5_DefaultSettings", # Ordner für die pth-Dateien
    "node_features": [   # Ausgewählte Knotenmerkmale
        "origin",               # 3 Features (x, y, z lokal)
        "centroid",             # 3 Features (x, y, z lokal)
        "centroid_x",           # 1 Feature (global X)
        "centroid_y",           # 1 Feature (global Y)
        "centroid_z",           # 1 Feature (global Z)
        "dimensions",           # 3 Features (L, B, H)
        "norm_dimensions",      # 3 Features (L, B, H normiert)
        "area",                 # 1 Feature
        "volume",               # 1 Feature
        "ratio_av",             # 1 Feature
        "normed_x",             # 1 Feature (normed X)
        "normed_y",             # 1 Feature (normed Y)
        "normed_z",             # 1 Feature (normed Z)
        "entity_one_hot_idx"    # 1 Feature (Klassen-Index)
    ]
}


class Logger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logs = []

    def log(self, msg, print_console=True):
        """Speichert eine Nachricht als Protokoll und gibt sie in der Konsole aus"""
        if print_console:
            print(msg)
            
        for line in str(msg).split('\n'):
            clean_msg = line.rstrip() # Entfernt Leerzeichen am Ende
            if clean_msg:
                self.logs.append(clean_msg)
        return 

    def save_to_file(self):
        """Schreibt alle gesammelten Logs in die Ziel-Datei"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.logs))
        print(f"\nProtokoll erfolgreich gespeichert unter: {self.file_path}")


class GNNEdgeClassifier(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, 
                 num_layers, model_type, activation, dropout):
        super(GNNEdgeClassifier, self).__init__()

        self.layers = nn.ModuleList()
        self.model_type = model_type
        self.edge_dim = CONFIG['edge_dim']

        # Aktivierungsfunktion wählen
        if activation == "ReLU":
            self.act = nn.ReLU()
        elif activation == "LeakyReLU":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ELU()

        # Regularisierung, die Elemente null gegen Overfitting. Alternative: L2
        self.dropout = nn.Dropout(dropout)

        # Initialer Layer: Transformiert Input-Features auf Hidden-Dimension
        self.layers.append(self.create_conv_layer(node_in_dim, hidden_dim, edge_in_dim)) # type: ignore

        # Hidden Layers
        for _ in range(num_layers -1):
            self.layers.append(self.create_conv_layer(hidden_dim, hidden_dim, edge_in_dim)) # type: ignore

        # Edge Predictor (MLP zur Klassifizierung der Kanten)
        # Kombination von zwei Knotenrepräsentationen (daher hidden_dim * 2)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # Binärer Output (Lastabtrag ja/nein)
        )
        
    
    def create_conv_layer(self, in_dim, out_dim, edge_dim):
        """Erstellt den gewünschten GNN-Layer Typ"""
        # Wenn edge_dim == 0, wird ein Platzhalter-Kanal von 1 für Layer, die edge_attr erzwingne (wie GINE)
        eff_edge_dim = max(1, self.edge_dim)

        if self.model_type == "GINE":
            # GINE braucht ein MLP für die Knoten-Transformation
            nn_gin = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                self.act,
                nn.Linear(out_dim, out_dim)
            )
            return GINEConv(nn=nn_gin, edge_dim=eff_edge_dim)
        
        elif self.model_type == "GATv2":
            # GATv2 nutzt Attention-Mechanismen | Kann edge_dim=None händeln
            return GATv2Conv(in_dim, out_dim, 
                             edge_dim=self.edge_dim if self.edge_dim > 0 else None)
        
        elif self.model_type == "NN":
            # NNConv nutzt ein MLP, um aus Kantenmerkmalen eine Gewichtsmatrix zu erzeugen
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, in_dim * out_dim)
            )
            return NNConv(in_dim, out_dim, nn=edge_nn)
        else:
            return SAGEConv(in_dim, out_dim)
        
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 0. edge_dim prüfen: Wenn edge_dim == 0, werden Kantenmerkmale neutralisiert
        if self.edge_dim == 0:
            # Erzeuge DummyVektor aus 1en, falls Layer (wie GINE) ihn erwartet
            edge_attr = torch.ones((edge_attr.size(0), 1), device=edge_attr.device)

        # 1. Message Passing (Knotenrepräsentation lernen)
        for conv in self.layers:
            if self.model_type == "GINE" or self.model_type == "NN":
                # Modulare Übergabe basierend auf Layer-Typ und Config
                x = conv(x, edge_index, edge_attr)
            elif self.model_type == "GATv2" and self.edge_dim > 0:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            x = self.act(x)
            x = self.dropout(x)

        # 2. Edge Decoding (Kantenmerkmale vorhersagen)
        # Holen der Repräsentationen der Start- (row) und Zielknoten (col) jeder Kante
        row, col = edge_index
        # Konkatenation der Knotenmerkmale für jede Kante
        edge_feat = torch.cat([x[row], x[col]], dim=1)

        # Rückgabe der Rohwerte (Logits) für den BCE Loss
        return self.edge_predictor(edge_feat).squeeze(-1)
    
    
def load_graph_data(file_path):
    """Lädt einen Graphen und bereitet ihn für PyG vor"""
    with open(file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # --- Knotenmerkmale ---
    nodes = graph_data['nodes']
    node_features_list = []

    for node in nodes:
        feat = []
        # --- MODULARE AUSWAHL DER KNOTENMERKMALE ---
        for feature_name in CONFIG["node_features"]:
            val = node['features'][feature_name]
            if isinstance(val, list):
                feat.extend(val)
            else:
                feat.append(val)
        # -------------------------------------------
        node_features_list.append(feat)
    
    x = torch.tensor(node_features_list, dtype=torch.float)

    # --- Kantenmerkmale ---
    edges = graph_data['edges']
    # Da Node IDs bereits 0 bis N-1 sind, können sie direkt gemapped werden
    edge_index = torch.tensor([[e['source'], e['target']] for e in edges],
                              dtype=torch.long).t().contiguous()
    
    # Nutze 'proximity_activation' aus den edge-features
    edge_attr = torch.tensor([[float(e['features']['proximity_activation'])] for e in edges], dtype=torch.float)

    # Labels: Nutze 'load_transfer' (True/False wir zu 1.0/0.0)
    y = torch.tensor([float(e['features']['load_transfer']) for e in edges],dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, filename=file_path.name)


def train_one_fold(train_graphs, test_graph, fold_nr, save_dir, logger):
    """Führt einen Fold der Kreuzvalidierung durch und speichert das beste Modell dieses Folds"""
    logger.log(f"\n--- Starte Fold {fold_nr+1} | Test-Modell: {test_graph.filename} ---")

    train_loader = DataLoader(train_graphs, batch_size=CONFIG['batch_size'], shuffle=True)

    model = GNNEdgeClassifier(
        node_in_dim=test_graph.num_node_features,
        edge_in_dim=test_graph.edge_attr.size(CONFIG["edge_dim"]),
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG["num_layers"],
        model_type=CONFIG["model_type"],
        activation=CONFIG["activation"],
        dropout=CONFIG["dropout"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    # BCEWithLogitsLoss kombiniert Sigmoid + Binary Cross Entropy für Stabilität
    pos_weight = None # Bei Bedarf z.B. 0.5 oder 1
    if pos_weight:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor(pos_weight))
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    best_fold_f1 = -1.0
    best_model_state = None

    # --- Training ---
    print(f"Training mit {test_graph.num_node_features} Knotenmerkmalen gestartet...")

    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validierung (auf einem Test-Graphen) zur Speicherung des besten Zustands
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(test_graph)
                val_preds = (torch.sigmoid(val_out) > CONFIG['threshold']).float()
                f1 = f1_score(test_graph.y.cpu(), val_preds.cpu(), zero_division=0)

                if f1 > best_fold_f1:
                    best_fold_f1 = f1
                    best_model_state = copy.deepcopy(model.state_dict())
            model.train()
        
        if (epoch + 1) % 100 == 0:
            logger.log(f"  Epoch {epoch+1:03d} | Loss: {total_loss/len(train_loader):.4f} | Best Val-F1: {best_fold_f1:.2f}")
        
    # Lade den besten Zustand für die finale Evaluation des Folds
    model.load_state_dict(best_model_state) # type: ignore

    # Speichere Modell dieses Folds ab
    fold_model_path = save_dir / f"model_fold_{fold_nr+1}.pth"
    torch.save(best_model_state, fold_model_path)

    # Berechne finale Leistungsmetriken
    model.eval()
    with torch.no_grad():
        out = model(test_graph)
        preds = (torch.sigmoid(out) > CONFIG['threshold']).float().cpu().numpy()
        y_true = test_graph.y.cpu().numpy()
        
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

    logger.log(f"RESULT Fold {fold_nr+1} (Threshold {CONFIG['threshold']}): Acc: {acc:.2%}, F1: {f1:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}")
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "state_dict": best_model_state}


def run_loo_training():
    """Führt Training als Kreuzvalidierung (Leave-One-Out) durch"""
    script_dir = Path(__file__).parent
    # Korrigierter Pfad zum Ordner der Trainingsdaten
    data_path = script_dir.parent / "2 Data_Preprocessing" / "2_4 Graph_Labeled"

    # Modell-Speicherpfad vorbereiten
    save_dir = script_dir / CONFIG['model_save_dir']
    save_dir.mkdir(exist_ok=True)

    # Logger initialisieren
    logger = Logger(save_dir / "LOO_Training_Log.txt")

    # Suche nach Dateien, die 'Graph' im Namen haben und auf .json enden
    json_files = list(data_path.glob("*Graph.json"))
    if len(json_files) < 2:
        # print(f"DEBUG: Suche in {data_path.absolute()}") # Debug
        logger.log("FEHLER: Zu wenige Graphen für Leave-One-Out gefunden.")
        return
    
    logger.log(f"Lade {len(json_files)} Graphen in den Speicher...")
    all_data = [load_graph_data(f) for f in json_files]

    # Logge Konfiguration der Architektur
    logger.log(f"CONFIG: ", print_console=False)
    logger.log(f"\tmodel_type: {CONFIG['model_type']}", print_console=False)
    logger.log(f"\thidden_dim: {CONFIG['hidden_dim']}", print_console=False)
    logger.log(f"\tnum_layers: {CONFIG['num_layers']}", print_console=False)
    logger.log(f"\tactivation: {CONFIG['activation']}", print_console=False)
    logger.log(f"\tdropout: {CONFIG['dropout']}", print_console=False)
    logger.log(f"\tlr: {CONFIG['lr']}", print_console=False)
    logger.log(f"\tepochs: {CONFIG['epochs']}", print_console=False)
    logger.log(f"\tedge_dim: {CONFIG['edge_dim']}", print_console=False)
    logger.log(f"\tbatch_size: {CONFIG['batch_size']}", print_console=False)
    logger.log(f"\tthreshold: {CONFIG['threshold']}", print_console=False)
    logger.log(f"\tnode_features: {', '.join(CONFIG['node_features'])}", print_console=False)

    # Initiiere Statistik-Variablen
    all_results = []
    global_best_f1 = -1.0
    global_best_state = None

    # Leave-One-Out Loop
    for i in range(len(all_data)):
        test_graph = all_data[i]
        train_graphs = all_data[:i] + all_data[i+1:]
        
        result = train_one_fold(train_graphs, test_graph, i, save_dir, logger)
        all_results.append(result)

        # Tracke das absolut beste Modell über alle Folds hinweg
        if result["f1"] > global_best_f1:
            global_best_f1 = result["f1"]
            global_best_state = result["state_dict"]

    # Speichere das global beste Modell
    if global_best_state:
        torch.save(global_best_state, save_dir / "best_model_overall.pth")

    # Finale Zusammenfassung
    avg_acc = np.mean([r['acc'] for r in all_results])
    avg_f1 = np.mean([r['f1'] for r in all_results])
    avg_prec = np.mean([r['prec'] for r in all_results])
    avg_rec = np.mean([r['rec'] for r in all_results])

    logger.log("\n" + "="*40)
    logger.log("LOO-KREUZVALIDIERUNG ABSCHLUSS")
    logger.log(f"Anzahl Modelle: {len(all_data)}")
    logger.log(f"Durchschnittliche Accuracy:  {avg_acc:.2%}")
    logger.log(f"Durchschnittlicher F1-Score: {avg_f1:.2f}")
    logger.log(f"Durchschnittliche Precision: {avg_prec:.2f}")
    logger.log(f"Durchschnittlicher Recall:    {avg_rec:.2f}")
    logger.log(f"Modelle gespeichert in: {save_dir}")
    logger.log("="*40)

    # Am Ende Protokoll speichern
    logger.save_to_file()
    return





if __name__ == "__main__":
    run_loo_training()





"""Kurzes Tutorial zu Modelltypen:
GINE (Graph Isomorphism Network with Edge features): Mein Favorit für dein Projekt. Es ist mathematisch sehr mächtig und bindet Kantenmerkmale (Distanz) direkt in den Aggregationsprozess ein. Es ist ideal, wenn die bloße Existenz einer Kante nicht reicht, sondern ihre physikalische Eigenschaft (Länge) wichtig ist.

GATv2 (Graph Attention Network): Nutzt "Aufmerksamkeit". Das Modell lernt selbst, welche Nachbarn wichtiger sind (z. B. eine dicke Stütze wichtiger als eine dünne Wand). GATv2 ist die stabilere Version des ursprünglichen GAT.

NN (Neural Network Convolution / NNConv): Dies ist der flexibelste Typ. Hier wird ein eigenes kleines Netzwerk trainiert, das aus deinen Kantenmerkmalen (Distanz) eine Filter-Matrix für die Knotenberechnung erstellt.
"""
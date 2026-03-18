import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, GATv2Conv, SAGEConv, NNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from pathlib import Path

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
    "model_type": "GINE",
    "hidden_dim": 128,   # Sinvolle Annahmen: 32, 64, 128
    "num_layers": 3,    # Anzahl der Message-Passing-Schichten
    # Auswahl: ELU, ReLU, LeakyReLU
    "activation": "ELU",
    "dropout": 0.1,
    "lr": 0.001,
    "epochs": 2000,      # Sanity Check braucht meist etwas länger zum Overfitten
    "edge_dim": 1       # Dimensionen der Kantenmerkmale (z.B. nur Distanz)
}

class GNNEdgeClassifier(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, 
                 num_layers, model_type, activation, dropout):
        super(GNNEdgeClassifier, self).__init__()

        self.layers = nn.ModuleList()
        self.model_type = model_type

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
            nn.Linear(hidden_dim, 1) # Binärer Output (Lastabtrag ja/nein)
        )
    
    def create_conv_layer(self, in_dim, out_dim, edge_dim):
        """Erstellt den gewünschten GNN-Layer Typ"""
        if self.model_type == "GINE":
            # GINE braucht ein MLP für die Knoten-Transformation
            nn_gin = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                self.act,
                nn.Linear(out_dim, out_dim)
            )
            return GINEConv(nn=nn_gin, edge_dim=edge_dim)
        
        elif self.model_type == "GATv2":
            # GATv2 nutzt Attention-Mechanismen
            return GATv2Conv(in_dim, out_dim, edge_dim=edge_dim)
        
        elif self.model_type == "NN":
            # NNConv nutzt ein MLP, um aus Kantenmerkmalen eine Gewichtsmatrix zu erzeugen
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, in_dim * out_dim)
            )
            return NNConv(in_dim, out_dim, nn=edge_nn)
    
    def forward(self, x, edge_index, edge_attr):
        # 1. Message Passing (Knotenrepräsentation lernen)
        for i, conv in enumerate(self.layers):
            if self.model_type in ["GINE", "GATv2", "NN"]:
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
    
def load_single_sanity_graph(file_path):
    """Lädt einen Graphen und bereitet ihn für PyG vor"""
    with open(file_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # --- Knotenmerkmale ---
    nodes = graph_data['nodes']
    node_features_list = []

    for node in nodes:
        feat = []
        # --- MODULARE AUSWAHL DER KNOTENMERKMALE ---
        # Abgewählte Merkmale müssen auskommentiert werden
        feat.extend(node['features']['position'])         # 3 Features (x, y, z lokal)
        feat.extend(node['features']['dimensions'])       # 3 Features (L, B, H)
        feat.extend(node['features']['norm_dimensions'])  # 3 Features (normiert)
        feat.append(node['features']['volume'])           # 1 Feature
        feat.append(node['features']['area'])             # 1 Feature
        feat.append(node['features']['ratio_av'])         # 1 Feature
        feat.append(node['x'])                            # 1 Feature (global X)
        feat.append(node['y'])                            # 1 Feature (global Y)
        feat.append(node['z'])                            # 1 Feature (global Z)
        feat.append(node['entity_one_hit_idx'])           # 1 Feature (Klassen-Index)
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
    y = torch.tensor([float(e['features']['load_transfer']) for e in edges],
                     dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def run_sanity_check():
    script_dir = Path(__file__).parent
    # Korrigierter Pfad zum Ordner der Trainingsdaten
    data_path = script_dir.parent / "2 Data_Preprocessing" / "2_4 Graph_Labeled"

    # Suche nach Dateien, die Graph im Namen haben und auf .json enden
    json_files = list(data_path.glob("*Graph.json"))
    if not json_files:
        print(f"DEBUG: Suche in {data_path.absolute()}")
        print("FEHLER: Kein pasender Labeled Graph gefunden.")
        return
    
    target_file = json_files[0]
    print(f"--- Sanity Check: Lade {target_file.name} ---")

    try:
        data = load_single_sanity_graph(target_file)
    except Exception as e:
        print(f"FEHLER: Kein Parsen der Daten möglich: {e}")
        return
    
    model = GNNEdgeClassifier(
        node_in_dim=data.num_node_features,
        edge_in_dim=data.edge_attr.size(CONFIG["edge_dim"]), # type: ignore
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        model_type=CONFIG["model_type"],
        activation=CONFIG["activation"],
        dropout=CONFIG["dropout"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    # BCEWithLogitsLoss kombiniert Sigmoid + Binary Cross Entropy für Stabilität
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"Training mit {data.num_node_features} Knotenmerkmalen gestartet...")

    model.train()
    for epoch in range(CONFIG["epochs"]):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if (epoch +1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                # Umwandlung der Logits in Wahrscheinlichkeiten und dann in Klassen (0/1)
                preds = (torch.sigmoid(out) > 0.5).float()
                correct = (preds == data.y).sum().item()
                acc = correct / data.y.size(0) # type: ignore
                print(f"Epoche {epoch+1:03d} | Loss: {loss.item():.4f} | Accuracy: {acc:.2%}")
    
    print(f"--- Sanity Check abgeschlossen ---")

if __name__ == "__main__":
    run_sanity_check()










"""Kurzes Tutorial zu Modelltypen:
GINE (Graph Isomorphism Network with Edge features): Mein Favorit für dein Projekt. Es ist mathematisch sehr mächtig und bindet Kantenmerkmale (Distanz) direkt in den Aggregationsprozess ein. Es ist ideal, wenn die bloße Existenz einer Kante nicht reicht, sondern ihre physikalische Eigenschaft (Länge) wichtig ist.

GATv2 (Graph Attention Network): Nutzt "Aufmerksamkeit". Das Modell lernt selbst, welche Nachbarn wichtiger sind (z. B. eine dicke Stütze wichtiger als eine dünne Wand). GATv2 ist die stabilere Version des ursprünglichen GAT.

NN (Neural Network Convolution / NNConv): Dies ist der flexibelste Typ. Hier wird ein eigenes kleines Netzwerk trainiert, das aus deinen Kantenmerkmalen (Distanz) eine Filter-Matrix für die Knotenberechnung erstellt.
"""
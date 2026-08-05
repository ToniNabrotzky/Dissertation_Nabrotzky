import json
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, GATv2Conv, SAGEConv, NNConv

def main():
    # ifc_name = "model_7 Graph.json"
    # ifc_name = "model_1 Graph.json"
    # ifc_name = "20220421MODEL REV01 Graph.json"

    # === PFADE ANPASSEN ===
    script_dir = Path(__file__).parent
    
    # Passe diesen Pfad zu deiner JSON-Datei an
    json_graph_file = script_dir.parent / "1 Data_Preprocessing" / "2_4 Graph_Generation" / ifc_name
    
    # Pfad zum trainierten Modell (wurde von deinem LOO-Training erstellt)
    trained_model_file = script_dir / CONFIG['model_save_dir'] / "best_model_overall.pth"
    
    # Prüfe, ob Dateien existieren
    if not json_graph_file.exists():
        print(f"Graph-JSON nicht gefunden: {json_graph_file}")
    elif not trained_model_file.exists():
        print(f"Modell-Datei nicht gefunden: {trained_model_file}")
    else:
        # Ausführung
        find_gnn_heroes(json_graph_file, trained_model_file)

# ==========================================
# KONFIGURATION
# ==========================================
# WICHTIG: Diese Config muss exakt mit der aus deinem Trainingsskript übereinstimmen!
CONFIG = {
    "model_save_dir": "Test_04_DB14_AktualisierteDistanzen",
    "model_type": "GINE",
    "hidden_dim": 128,
    "num_layers": 5,
    "activation": "ELU",
    "dropout": 0.2,
    "threshold": 0.5,
    "edge_dim": 1,
    "node_features": [
        "norm_dimensions",
        "area",
        "volume",
        "ratio_av",
        "entity_one_hot_idx"
    ]
}

# ==========================================
# MODELL-KLASSE (Kopie aus Trainingsskript)
# ==========================================
class GNNEdgeClassifier(nn.Module):
    """Exakte Kopie der Architektur, damit die trainierten Gewichte geladen werden können."""
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers, model_type, activation, dropout):
        super(GNNEdgeClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.model_type = model_type
        self.edge_dim = CONFIG['edge_dim']

        self.act = nn.ReLU() if activation == "ReLU" else nn.LeakyReLU() if activation == "LeakyReLU" else nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.layers.append(self.create_conv_layer(node_in_dim, hidden_dim, edge_in_dim))
        for _ in range(num_layers - 1):
            self.layers.append(self.create_conv_layer(hidden_dim, hidden_dim, edge_in_dim))

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def create_conv_layer(self, in_dim, out_dim, edge_dim):
        eff_edge_dim = max(1, self.edge_dim)
        if self.model_type == "GINE":
            nn_gin = nn.Sequential(nn.Linear(in_dim, out_dim), self.act, nn.Linear(out_dim, out_dim))
            return GINEConv(nn=nn_gin, edge_dim=eff_edge_dim)
        elif self.model_type == "GATv2":
            return GATv2Conv(in_dim, out_dim, edge_dim=self.edge_dim if self.edge_dim > 0 else None)
        elif self.model_type == "NN":
            edge_nn = nn.Sequential(nn.Linear(edge_dim, in_dim * out_dim))
            return NNConv(in_dim, out_dim, nn=edge_nn)
        else:
            return SAGEConv(in_dim, out_dim)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.edge_dim == 0:
            edge_attr = torch.ones((edge_attr.size(0), 1), device=edge_attr.device)

        for conv in self.layers:
            if self.model_type in ["GINE", "NN"] or (self.model_type == "GATv2" and self.edge_dim > 0):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col]], dim=1)
        return self.edge_predictor(edge_feat).squeeze(-1)

# ==========================================
# INFERENZ-LOGIK
# ==========================================
def extract_pyg_data_from_json(graph_data):
    """Konvertiert JSON in PyTorch Geometric Tensoren für die Modell-Eingabe."""
    # 1. Knotenmerkmale extrahieren
    node_features_list = []
    for node in graph_data['nodes']:
        feat = []
        for feature_name in CONFIG["node_features"]:
            val = node['features'][feature_name]
            if isinstance(val, list):
                feat.extend(val)
            else:
                feat.append(val)
        node_features_list.append(feat)
    
    x = torch.tensor(node_features_list, dtype=torch.float)

    # 2. Kantenmerkmale extrahieren
    edges = graph_data['edges']
    edge_index = torch.tensor([[e['source'], e['target']] for e in edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[float(e['features']['proximity_activation'])] for e in edges], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def find_gnn_heroes(json_path, model_pth_path):
    """Findet Kanten, die der RBC verpasst hat, das GNN aber korrigiert."""
    print(f"Analysiere Graph: {json_path.name}")
    
    # 1. Lade JSON Daten (wir brauchen sie später für die GUIDs und Namen)
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_json = json.load(f)
        
    edges_meta = graph_json['edges']
    
    # 2. Erzeuge PyG Data Objekt
    pyg_data = extract_pyg_data_from_json(graph_json)
    
    # 3. Initialisiere Modell und lade Gewichte
    model = GNNEdgeClassifier(
        node_in_dim=pyg_data.num_node_features,
        edge_in_dim=pyg_data.edge_attr.size(CONFIG["edge_dim"]), #type: ignore
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG["num_layers"],
        model_type=CONFIG["model_type"],
        activation=CONFIG["activation"],
        dropout=CONFIG["dropout"]
    )
    
    print(f"Lade trainierte Gewichte aus: {model_pth_path.name}")
    model.load_state_dict(torch.load(model_pth_path, map_location=torch.device('cpu')))
    model.eval() # GANZ WICHTIG: Dropout deaktivieren!
    
    # 4. Inferenz durchführen
    with torch.no_grad():
        out_logits = model(pyg_data)
        probabilities = torch.sigmoid(out_logits).numpy() # Transformiere zu 0.0 - 1.0
    
    # 5. Logik anwenden: Suche nach den "Rettungen" des Modells
    worked_examples = []
    
    # Da PyG den Edge_Index aus unserer JSON-Liste aufgebaut hat, 
    # ist die Reihenfolge der Vorhersagen exakt gleich der Reihenfolge der JSON-Edges!
    for i, edge in enumerate(edges_meta):
        # Ground Truth & RBC Status lesen (direkt aus JSON)
        gt_is_connected = edge['features']['load_transfer']
        rbc_status = edge['features']['rbc_status'] # TP, TN, FP, FN
        
        # GNN Vorhersage
        gnn_prob = probabilities[i]
        gnn_is_connected = bool(gnn_prob > CONFIG["threshold"])
        
        # UNSERE MAGISCHE BEDINGUNG:
        # Der RBC hat es als 'False Negative' (FN) markiert (Bauteile nicht verbunden, obwohl sie es in Wahrheit sind)
        # UND das GNN sagt korrekt, dass sie verbunden sind (> 0.5)
        if rbc_status == "FN" and gnn_is_connected:
            worked_examples.append({
                "edge_index": i,
                "prob": float(gnn_prob),
                "guid_a": edge['meta']['guid_a'],
                "guid_b": edge['meta']['guid_b'],
                "name_a": edge['meta']['name_a'],
                "name_b": edge['meta']['name_b'],
                "entity_a": edge['meta']['entity_a'],
                "entity_b": edge['meta']['entity_b']
            })
            
    # 6. Ergebnisse präsentieren
    print("\n" + "="*60)
    print(f"GNN HEROES GEFUNDEN: {len(worked_examples)} Verbindungen gerettet!")
    print("="*60)
    
    for idx, ex in enumerate(worked_examples):
        print(f"\nBeispiel {idx + 1} (Konfidenz: {ex['prob']:.1%}):")
        print(f"  Bauteil A: [{ex['entity_a']}] {ex['name_a']} (GUID: {ex['guid_a']})")
        print(f"  Bauteil B: [{ex['entity_b']}] {ex['name_b']} (GUID: {ex['guid_b']})")
        
    # Speichere die Liste als CSV für eine einfache Untersuchung im BIM Viewer
    if worked_examples:
        export_path = json_path.parent / f"{json_path.stem}_WorkedExamples.csv"
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write("GNN_Probability,GUID_A,GUID_B,Entity_A,Entity_B,Name_A,Name_B\n")
            for ex in worked_examples:
                f.write(f"{ex['prob']:.4f},{ex['guid_a']},{ex['guid_b']},{ex['entity_a']},{ex['entity_b']},\"{ex['name_a']}\",\"{ex['name_b']}\"\n")
        print(f"\n--> Liste exportiert nach: {export_path}")
        print("    Diese CSV kannst du nun nutzen, um die GUIDs im IFC-Viewer zu markieren.")
    else:
        print("\nDas GNN konnte in diesem spezifischen Modell keine FN-Fehler des RBC beheben.")


if __name__ == "__main__":
    main()
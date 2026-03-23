import json
import torch
import torch.nn as nn
from torch_geometric.data import Data
from pathlib import Path


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


# Architektur muss exakt so definiert werden wie im Training
class GNNEdgeClassifier(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers, model_type, activation, dropout):
        super(GNNEdgeClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.model_type = model_type
        
        #  Aktivierungsfunktion (muss identisch zum Training sein)
        if activation == "ELU":
            self.act = nn.ELU()
        elif activation == "LeakyReLU":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()        
        
        # Layer-Logik (muss identisch zum Training sein)
        from torch_geometric.nn import GINEConv

        
        def create_conv(in_d, out_d):
            # MLP Transformation für GINE
            mlp = nn.Sequential(
                nn.Linear(in_d, out_d), 
                self.act, 
                nn.Linear(out_d, out_d)
                )
            return GINEConv(nn=mlp, edge_dim=1) # edge_dim sollte edge_dim sein

        # Schichten aufbauen
        self.layers.append(create_conv(node_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(create_conv(hidden_dim, hidden_dim))

        # Edge Predictor aufbauen (Zusammenführen von 2 Knoten)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)

        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col]], dim=1)
        return self.edge_predictor(edge_feat).squeeze(-1)


def run_inference(model_path, graph_json_path):
    # Logger initialisieren
    logger = Logger(model_path.parent / f"Inference_Report_{graph_json_path.stem}.txt")

    # 1. Daten laden
    with open(graph_json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Knotenmerkmale extrahieren (identisch zum Training!)
    nodes = graph_data['nodes']
    node_feats = []
    for n in nodes:
        f = []
        f.extend(n['features']['origin'])
        f.extend(n['features']['centroid'])
        f.append(n['features']['centroid_x'])
        f.append(n['features']['centroid_y'])
        f.append(n['features']['centroid_z'])
        f.extend(n['features']['dimensions'])
        f.extend(n['features']['norm_dimensions'])
        f.append(n['features']['area'])
        f.append(n['features']['volume'])
        f.append(n['features']['ratio_av'])
        f.append(n['features']['normed_x'])
        f.append(n['features']['normed_y'])
        f.append(n['features']['normed_z'])
        f.append(n['features']['entity_one_hot_idx'])
        node_feats.append(f)
    
    x = torch.tensor(node_feats, dtype=torch.float)
    edges_list = graph_data['edges']
    edge_index = torch.tensor([[e['source'], e['target']] for e in edges_list], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[float(e['features']['proximity_activation'])] for e in edges_list], dtype=torch.float)
    y_true = torch.tensor([float(e['features']['load_transfer']) for e in edges_list], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_true)

    # 2. Modell laden
    # Parameter müssen exakt der CONFIG des Trainings entsprechen
    model = GNNEdgeClassifier(
        node_in_dim=x.size(1), 
        edge_in_dim=edge_attr.size(1), 
        hidden_dim=128, 
        num_layers=3, 
        model_type="GINE", 
        activation="ELU",
        dropout=0.1
    )

    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("Modell erfolgreich geladen.")
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return

    model.eval()

    # 3. Vorhersage
    with torch.no_grad():
        logits = model(data)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    # 4. Tabellarische Analyse der Ergebnisse
    logger.log(f"\nAnalyse für Modell: {graph_json_path.name}")
    header = f"{'Kante':<10} | {'Name A -> Name B':<30} | {'Typ A -> Typ B':<25} | {'GT':<2} | {'Pr':<2} | {'Prob':<6} | {'Status'}"
    # logger.log(f"{'Kante':<12} | {'GT':<5} | {'Pred':<5} | {'Prob':<2} | {'Status'}")
    logger.log(header)
    logger.log("-" * 60)

    errors = 0
    for i in range(len(edges_list)):
        gt = int(y_true[i].item())
        pr = int(preds[i].item())
        prob = probs[i].item()

        meta = edges_list[i].get('meta', {})
        name_a = meta.get('name_a', 'unbekannt')
        name_b = meta.get('name_b', 'unbekannt')
        entity_a = meta.get('entity_a', 'unbekannt')
        entity_b = meta.get('entity_b', 'unbekannt')
        
        is_error = (gt != pr)
        if is_error:
            errors += 1
            
        # Primär werden die Fehler ausgegeben, um die Analyse darauf zu fokussieren
        if is_error or (prob > 0.45 and prob < 0.55):
            edge_ids = f"{edges_list[i]['source']}->{edges_list[i]['target']}"
            names = f"{name_a[:13]}->{name_b[:13]}"
            types = f"{entity_a[:11]}->{entity_b[:11]}"
            status = "FEHLER" if is_error else "UNSICHER"

            row = f"{edge_ids:<10} | {names:<30} | {types:<25} | {gt:<2} | {pr:<2} | {prob:.4f} | {status}"
            logger.log(row)
            # logger.log(f"{edge_ids:<12} | {gt:<4} | {pr:<4} | {prob:.4f} | {status}")

    acc = (preds == y_true).sum().item() / len(y_true)
    logger.log(f"\nZusammenfassung:")
    logger.log(f"Gesamtanzahl Kanten: {len(y_true)}")
    logger.log(f"Fehlklassifizierungen: {errors}")
    logger.log(f"Genauigkeit (Accuracy): {acc:.2%}")
    logger.log(f"\nGesamtgenauigkeit auf diesem Modell: {acc:.2%}")

    # Am Ende Protokoll speichern
    logger.save_to_file()
    return


if __name__ == "__main__":
    # Pfade anpassen
    SCRIPT_DIR = Path(__file__).parent
    MODEL_FILE = SCRIPT_DIR / "saved_models" / "best_model_overall.pth"
    # Wähle hier eine Datei aus dem Preprocssing Ordner
    GRAPH_FILE = SCRIPT_DIR.parent / "2 Data_Preprocessing" / "2_4 Graph_Labeled" / "21_22 L_TWP_Tragwerksmodell Graph.json"
    
    if MODEL_FILE.exists() and GRAPH_FILE.exists():
        run_inference(MODEL_FILE, GRAPH_FILE)
    else:
        print(f"Datei nicht gefunden.\nModell: {MODEL_FILE.exists()}\nGraph: {GRAPH_FILE.exists()}")
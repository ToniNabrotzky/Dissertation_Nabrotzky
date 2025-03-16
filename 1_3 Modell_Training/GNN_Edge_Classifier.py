import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt



"""Hauptfunktion"""
def main():
    """Hauptfunktion zum Trainieren des GNN-Modells"""
    global script_dir
    script_dir = Path(__file__).parent
    # script_dir = Path.cwd() # Alternative: Path(__file__).parent
    # print(f"Script-Dir - {script_dir.exists()}: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Teste einzelnes GNN
    # json_folder_path = r"H:\1_2 Data_Preprocessing\Modell_2_Parametrisch Stahlbeton\JSON_Graph"
    # train_GNN(json_folder_path)
    
    
    ## Trainiere seriell GNN für alle JSON-Dateien
    json_folder_names = ["Modell_2_Parametrisch Stahlbeton Index 000_099"]
    # json_folder_names = []
    json_folder_paths = get_folder_paths(script_dir, json_folder_names)

    for json_folder_path in json_folder_paths:
        # print(f"json-Folder-Path - {json_folder_path.exists()}:  {json_folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Durchsuche folder_path nach allen JSON-Dateien
        if not json_folder_path.exists():
            print(f"__Ordner nicht gefunden. Fehlender Pfad: {json_folder_path}")
        else:
            print(f"__Ordner gefunden. Suche JSON-Dateien in: {json_folder_path}")
            train_GNN(json_folder_path)


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_predictor = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu() # Alternative: Leaky ReLU, ELU
        x = self.conv2(x, edge_index)

        # Kantenklassifikation (Vorhersage der Beziehungen)
        edge_index = data.edge_index
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_predictions = self.edge_predictor(edge_features)
        return edge_predictions


def train_GNN(json_folder_path):
    print(f"__Starte Training für: {json_folder_path}")
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
    dataset = []

    ## Erstelle Datenset mit Graphen und beschrifteten Kanten
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        G, edges = load_graph_from_json(json_file_path)
        data = create_data_from_graph(G, edges)
        dataset.append(data)
    
    ## Erstelle Trainingsbatches, zufällig gemischt
    # Batchsize 1-32: more frequent updates, can help escape local minima but may lead to noisy gradient
    # Batchsize 64-256: Smoother gradients, more stable training, but requires more memory
    loader = DataLoader(dataset, batch_size= 64, shuffle= True)

    model = GNN(input_dim= dataset[0].num_node_features, 
                hidden_dim= 32, 
                output_dim= 2)
    optimizer = optim.Adam(model.parameters(), lr= 0.01) # Alternative: Stochastic Gradient Descent (SGD)
    criterion = nn.CrossEntropyLoss() # Alternative: MSE, Hinge Loss, Focal Loss, KL Divergence

    best_loss = float('inf')
    loss_values = []

    for epoch in range(100):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            # print(f"out: {out}") # Debugging statement
            # print(f"data.y.numel: {data.y.numel()}") # Debugging statement
            # Absichern, dass gelabelte Daten vorhanden sind.
            if data.y.numel() > 0:
                # print(f"out shape: {out.shape}, data.y shape: {data.y.shape}") # Debugging statement
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                print("Keine beschrifteten Daten vorhanden!!!")
        
        avg_loss = total_loss / len(loader)
        loss_values.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        # Speicher bestes Modell (Parameterset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, json_folder_path, best= True)
    
    save_model(model, json_folder_path, best= False)
    plot_loss(loss_values)





"""Hilfsfunktionen """
def get_folder_paths(script_dir, folder_names):
    """Gibt eine Liste von Pfaden zu den Ordnern zurück"""  
    json_folder_paths = [script_dir.parent / "1_2 Data_Preprocessing" / folder_name / 'JSON_Graph' for folder_name in folder_names]
    # # print(f"Folder-Paths: {json_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return json_folder_paths


def load_graph_from_json(json_file_path):
    """Lädt den 3D-Graphen mit den Knotenpositionen aus einer JSON-Datei"""
    with open(json_file_path, "r") as f:
        graph_data = json.load(f)
    
    # Automatic inspection of graph
    num_nodes = len(graph_data["nodes"])
    num_edges = len(graph_data["edges"])
    print(f"Loaded file_path: {json_file_path}")  # Debugging statement
    print(f"num_nodes: {num_nodes}") # Debugging statement
    print(f"num_edges: {num_edges}") # Debugging statement
    # print(f"Loaded nodes: {G.nodes}")  # Debugging statement
    # print(f"Loaded edges: {edges}")  # Debugging statement
    
    G = nx.Graph()
    for node, data in graph_data['nodes'].items():
        node = int(node) # Ensure node ID is a string
        G.add_node(node, **data)

    edges = [(int(edge[0]), int(edge[1])) for edge in graph_data['edges']]
    return G, edges


def create_data_from_graph(G, edges):
    """Erstellt ein PyTorch-Geometric-Data-Objekt aus einem Netzwerkx-Graphen"""
    node_features = []
    for _, data in G.nodes(data= True):
        for value in data.values():
            if isinstance(value, list):
                node_features.extend(value)
            else:
                node_features.append(value)    
    # print("node_features: ", node_features)

    # Flatten the node_features list
    flat_node_features = [item for sublist in node_features for item in (sublist if isinstance(sublist, list) else [sublist])]
    x = torch.tensor(flat_node_features, dtype=torch.float).view(len(G.nodes), -1)

    # Create all possible edge combinations
    possible_edges = list(combinations(G.nodes, 2))
    edge_index = torch.tensor(possible_edges, dtype=torch.long).t().contiguous()

    # Create all labels for edge classification
    edge_set = set(edges)
    y = torch.tensor([1 if (u, v) in edge_set or (v, u) in edge_set else 0 for u, v in possible_edges], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def save_model(model, json_folder_path, best= False):
    ## Speicher Modell in Graph-Format
    model_folder = script_dir / "GNN_Model"
    os.makedirs(model_folder, exist_ok= True)
    model_path = model_folder / ("gnn_model_best.pth" if best else "gnn_model_last.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    ## Speicher Modell in JSON-Format
    model_json_path = model_folder / ("gnn_model_best.json" if best else "gnn_model_last.json")
    model_params = {k: v.tolist() for k, v in model.state_dict().items()}
    with open(model_json_path, 'w') as f:
        json.dump(model_params, f)
    print(f"Model parameters saved to {model_json_path}")


def plot_loss(loss_values):
    plt.figure()
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    print("___Intitialisiere Skript___")

    main()

    print("___Skript erfolgreich beendet___")
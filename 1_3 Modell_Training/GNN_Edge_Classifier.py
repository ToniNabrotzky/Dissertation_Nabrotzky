import logging
import json
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s %(message)s')



"""Hauptfunktion"""
def main():
    """Hauptfunktion zum Trainieren des GNN-Modells"""
    global script_dir
    script_dir = Path(__file__).parent
    ## Alternative Wege
    # script_dir = Path.cwd() # Alternative: Path(__file__).parent
    # print(f"Script-Dir - {script_dir.exists()}: {script_dir}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    """Trainiere GNN in aus einem direkten Dataset"""
    # json_folder_path = r"H:\1_2 Data_Preprocessing\"Modell_2_Parametrisch Stahlbeton 00000_01535\JSON_Graph"
    # train_GNN(json_folder_path)
    
    
    """Trainiere seriell GNN aus mehreren Datasets"""
    graph_folder_names = ["Modell_2_Parametrisch Stahlbeton 00000_01535"] # Standardordner für alle Daten
    # graph_folder_names = ["Modell_2_Parametrisch Stahlbeton 01530_01535"]
    graph_folder_paths = get_folder_paths(script_dir, graph_folder_names)

    for graph_folder_path in graph_folder_paths:
        # print(f"graph-Folder-Path - {graph_folder_path.exists()}:  {graph_folder_path}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ## Durchsuche folder_path nach allen JSON-Dateien
        if not graph_folder_path.exists():
            print(f"__Ordner nicht gefunden. Fehlender Pfad: {graph_folder_path}")
        else:
            print(f"__Ordner gefunden. Suche JSON-Dateien in: {graph_folder_path}")
            
            ## Umleitung der Standardausgabe zu Konsole und Teft-File
            plotter = Plotter(script_dir, graph_folder_path)
            original_stdout = sys.stdout
            with open(plotter.model_folder_path / 'terminal_output.txt', 'w') as f:
                sys.stdout = Tee(sys.stdout, f)
                train_GNN(graph_folder_path)
                sys.stdout = original_stdout


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        ### Festlegung des Modelltyps
        ## Modelltypen: GCNConv (Default alt), GATConv, SAGEConv (Default neu)
        ## Aggregation erfolgt standardmäßig über Mittelwert (mean). Alternative: Somme (add) oder Pooling (LSTM oder Max-Pooling)
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr= 'mean') 
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr= 'mean') 
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr= 'mean')
        self.dropout = nn.Dropout(p= 0.2) # Default= 0.5, neu= 0.2 | Regularisierung, die Elemente nullt. Alternative: L2        
        if Kriterium == 'BCEWithLogitsLoss':
            self.edge_predictor = nn.Linear(hidden_dim * 2, 1)  # Ändere output_dim auf 1
        else:
            self.edge_predictor = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        ### Netzwerkarchitektur definieren
        ## Netzwerkarchitektur ReLU
        # x, edge_index = data.x, data.edge_index
        # x = self.conv1(x, edge_index).relu() # Default: relu | Alternative: Leaky ReLU, ELU
        # x = self.dropout(x) # Vermeidung von overfitting
        # x = self.conv2(x, edge_index)
        # x = self.dropout(x) # Vermeidung von overfitting

        ## Netzwerkarchitektur Leaky ReLU oder ELU
        x, edge_index = data.x, data.edge_index
        # x = F.leaky_relu(self.conv1(x, edge_index), negative_slope= 0.01) # Leaky ReLU
        x = F.elu(self.conv1(x, edge_index), alpha= 1.0) # ELU
        x = self.dropout(x) # Vermeidung von overfitting
        # x = F.leaky_relu(self.conv2(x, edge_index), negative_slope= 0.01) # Leaky ReLU
        x = F.elu(self.conv2(x, edge_index), alpha= 1.0) # ELU
        x = self.dropout(x) # Vermeidung von overfitting
        # x = F.leaky_relu(self.conv3(x, edge_index), negative_slope= 0.01) # Leaky ReLU
        x = F.elu(self.conv3(x, edge_index), alpha= 1.0) # ELU
        x = self.dropout(x) # Vermeidung von overfitting
        

        ### Kantenklassifikation (Vorhersage der Beziehungen)
        edge_index = data.edge_index
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim= 1)
        if Kriterium == 'BCEWithLogitsLoss':
            edge_predictions = self.edge_predictor(edge_features).squeeze(-1)  # Entferne die letzte Dimension
        else:
            edge_predictions = self.edge_predictor(edge_features)
        return edge_predictions


def train_GNN(graph_folder_path):
    """Führt das Training des GNN um. Zuerst werden die Daten in Datensätze und Batches aufgeteilt.
    Dann werden einige Hyperparameter bestimmt und die Epochen für das Training durchlaufen.
    Anschließend findet die Fehlerberechnung und Auswertung ab, die abgespeichert wird."""

    print(f"__Starte Training für: {graph_folder_path}")
    json_files = [f for f in os.listdir(graph_folder_path) if f.endswith('.json')]

    ## Statistik zum Graphen für Ausgewogenheit der Labels vorbereiten
    num_graphs = 0 # Insgesamt eingeladene Graphen
    num_nodes = 0 # Insgesamt eingeladene Knoten
    num_edges = 0 # Insgesamt eingeladene Kanten (positive und negative Ereignisse zusammen) = mögliche Kanten
    num_positives = 0 # Anzahl eingeladener Kanten mit positivem Ereignis
    num_negatives = 0 # Anzahl eingeladener Kanten mit negativem Ereignis
    max_percentage = 0 # Größter Anteil positiver Ereignisse aus einem Graphen

    ## Protokolliere angesetzte Parameter:
    Train_Prozent = 0.3 # Default= 0.3 Wie viel Prozent sollen an Temp gehen? Rest bleibt bei Train.
    Val_Prozent = 0.5 # Default= 0.5 Wie viel Prozent sollen an Test gehen? Rest von oben bleibt bei Val.
    Hidden = 32 # Default= 32
    Lernrate = 0.01 # Default= 0.01
    global Kriterium
    Kriterium = 'BCEWithLogitsLoss'
    ## Auswahl= 'CrossEntropyLoss' oder 'CrossEntropyLoss_mit_Gewichtsklassifizierung' oder 'Focal_Loss' oder 'BCEWithLogitsLoss'
    Epochenzahl = 100 # Default= 100
    Batch_Größe = 32 # Default= 256, neu 32
    ## Batchsize 1-32: häufigere (more frequent) updates, kann aus lokalen Minima führen, aber evtl zu rauschendem (noisy) Gradienten
    ## Batchsize 64-256: Weichere Gradienten, stabileres Training, aber benötigt mehr Arbeitsspeicher
    print(f"""
    Übersicht zur Modellarchitektur:
          Modelltyp= GraphSAGE
          Input | Hidden: {Hidden} (Conv1, ReLU, Dropout, Conv2, Dropout, Conv3, Dropout) | EdgePredictor: Hidden*2 | Out: 1 (bei BCE; sonst 2)
          Aktivierungsfunktion = ELU
          Dropout= 0.2
    Hyperparameter:
          Aufteilung Datensätze (Training, Validierung, Test): {(1 - Train_Prozent) * 100} / {(Train_Prozent * (1 - Val_Prozent)) * 100} / {Train_Prozent * Val_Prozent * 100}
          optimizer= Stochastic Gradient Decent (SGD), LR= {Lernrate}
          Criterion= {Kriterium}
          Early Stopping: Aktiviert
    Training:
          Epochenzahl= {Epochenzahl}
          BatchSize= {Batch_Größe}
          """)
    

    ### Teile Datensätze und teile es in Batches ein
    dataset = []

    for json_file in json_files:
        json_file_path = os.path.join(graph_folder_path, json_file)
        ## Lade unverbundenen Graphen und Kanten als beschriftete Daten
        G, edges, graph_id = load_graph_from_json(json_file_path)
        # edges_sorted = {(edge['source'], edge['target']) for edge in edges}
        # print(f"edges aus 'train_GNN': {sorted(edges_sorted)}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # for edge in edges:
        #     print(f"edge aus 'train_GNN': {edge['label'] == 1}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ## Bugfixing um zu sehen, ob überhaupt positive Labels erzeugt werden vom GNN
        # edges = add_random_edges(G, int(len(edges)), positives= 0.5)
        # print(f"random edges aus 'train_GNN': ", edges) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ## Statistik zum Graphen für Ausgewogenheit der Labels sammeln
        total_nodes = len(G.nodes)
        total_edges = len(edges) # edges inkludiert positive sowie negative Kanten
        positive_edges = sum(e['label'] == 1 for e in edges) # Alternative: sum([1 for edge in edges if edge['label'] == 1])
        negative_edges = sum(e['label'] == 0 for e in edges) # Alternative: sum([1 for edge in edges if edge['label'] == 0])
        percentage = positive_edges / total_edges
        print(f"  __Modell_ID: {graph_id} \tGraph mit {total_nodes} Knoten und {positive_edges} Kanten \tTotal Edges: {total_edges}, Positive Labels: {percentage:.2%}")

        num_graphs += 1
        num_nodes += total_nodes
        num_edges += total_edges
        num_positives += positive_edges
        num_negatives += negative_edges
        max_percentage = percentage if percentage > max_percentage else max_percentage
        # print(f"Vergleich max. Positives: {max_percentage:.2%} mit akt. Positives: {percentage:.2%}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        ## Dataset anreichern
        data = create_data_from_graph(G, edges, graph_id)
        dataset.append(data)
    
    ## Statistik zum Graphen für Ausgewogenheit der Labels ausgeben
    print(f"  __Statistiken über die eingelesenen Daten:")
    print(f"\tGraphen: {num_graphs}, Knoten: {num_nodes}, Kanten: {num_positives} | mögl. Kanten: {num_edges} | Mittelwert Anteil Positives: {num_positives / num_edges:.2%} | Max. Anteil Positives: {max_percentage:.2%}")


    ### Erstelle Datensätze (Training, Validierung, Test) und Einteilung in Batches (Loader)
    ## Aufteilung der Graphen in Trainings-, Validierungs- und Testdatensatz
    ## random_state ist wie ein Random-Seed für Shuffle später
    train_graphs, temp_graphs = train_test_split(dataset, test_size= Train_Prozent, random_state= 42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size= Val_Prozent, random_state= 42)
    
    ## Erstellen der Kantenlisten für jeden Datensatz
    train_edges = [edge for graph in train_graphs for edge in graph.edge_index.t().tolist()]
    val_edges = [edge for graph in val_graphs for edge in graph.edge_index.t().tolist()]
    test_edges = [edge for graph in test_graphs for edge in graph.edge_index.t().tolist()]

    ## Erstellen der DataLoader
    train_loader = DataLoader(train_graphs, batch_size= Batch_Größe, shuffle= True)
    val_loader = DataLoader(val_graphs, batch_size= Batch_Größe, shuffle= False)
    test_loader = DataLoader(test_graphs, batch_size= Batch_Größe, shuffle= False)

    ## Überprüfe Gesamtsumme der Teildatensätze sowie Disjunktheit der Graphen (wegen Unstimmigkeit mit Confusion Matrix)
    num_train_graphs = len(train_graphs)
    num_val_graphs = len(val_graphs)
    num_test_graphs = len(test_graphs)
    num_total_graphs = num_train_graphs + num_val_graphs + num_test_graphs
    print(f"\tAnzahl der Graphen in den 3 Datensätzen: {num_train_graphs}, {num_val_graphs}, {num_test_graphs} | Gesamt: {num_total_graphs}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    num_train_edges = len(train_edges)
    num_val_edges = len(val_edges)
    num_test_edges = len(test_edges)
    num_total_edges = num_train_edges + num_val_edges + num_test_edges
    print(f"\tAnzahl der Kanten in den 3 Datensätzen: {num_train_edges}, {num_val_edges}, {num_test_edges} | Gesamt: {num_total_edges}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    train_graph_ids = [graph.graph_id for graph in train_graphs]
    val_graph_ids = [graph.graph_id for graph in val_graphs]
    test_graph_ids = [graph.graph_id for graph in test_graphs]
    train_vs_val = len(set(train_graph_ids) & set(val_graph_ids))
    train_vs_test = len(set(train_graph_ids) & set(test_graph_ids))
    val_vs_test = len(set(val_graph_ids) & set(test_graph_ids))
    print(f"\tÜberschneidungen zwischen Train und Val: {train_vs_val} | Train und Test: {train_vs_test} | Val und Test: {val_vs_test}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Cross-Validierung bei Aufteilung der Datensets
    kf = KFold(n_splits= 5)
    for train_index, val_index in kf.split(dataset):
        train_subset = [dataset[i] for i in train_index]
        val_subset = [dataset[i] for i in val_index]
        train_loader = DataLoader(train_subset, batch_size= Batch_Größe, shuffle= True)
        val_loader = DataLoader(val_subset, batch_size= Batch_Größe, shuffle= False)

    ### Definition Hyperparameter
    model = GNN(input_dim= dataset[0].num_node_features, hidden_dim= Hidden, output_dim= 2)
    optimizer = optim.Adam(model.parameters(), lr= Lernrate) # Alternative: Stochastic Gradient Descent (SGD)
    # optimizer = optim.SGD(model.parameters(), lr= Lernrate, momentum= 0.9)

    ## Definition der Verlustfunktion
    # Var1: Berechnung mit CrossEntropyLoss
    if Kriterium == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss() # Alternative: BCEWithLogitsLoss, MSE, Hinge Loss, Focal Loss, KL Divergence
    
    # Var2 Kriterium: Berechnung der Klassengewichte zur Berücksichtigung der wenigen positiven Ergebnisse
    elif Kriterium == 'CrossEntropyLoss_mit_Gewichtsklassifizierung':
        positive_weight = len(edges) / num_positives
        negative_weight = len(edges) / num_negatives
        class_weights = torch.tensor([negative_weight, positive_weight], dtype= torch.float)
        criterion = nn.CrossEntropyLoss(weight= class_weights)
    
    # Var3 Kriterium: Berechnung mit Focal Loss
    elif Kriterium == 'Focal_Loss':
        criterion = FocalLoss(alpha= 0.25, gamma= 2)
    
    # Var4: Berechnung mit binärer CrossEntropy
    elif Kriterium == 'BCEWithLogitsLoss':
        positive_weight = num_negatives / num_positives  # Gewichtung für positive Labels
        criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor(positive_weight))

    else:
        raise NotImplementedError("Dies Variante der Verlustberechnung wurde noch nicht integriert")
    
    ## Aktiviere externe Classes
    early_stopping = EarlyStopping(patience= 20, min_delta= 0.02)
    plotter = Plotter(script_dir, graph_folder_path)

    ## Listen für Auswertung
    best_loss = float('inf')
    loss_values = [] # Verlust pro Lernepoche
    train_losses = [] # Verluste im Training
    val_losses = [] # Verluste in Validierung
    train_accuracies = [] # Genauigkeit im Training
    val_accuracies = [] # Genauigkeit in Validierung
    test_accuracies = [] # Genauigkeit in Test

    train_precisions = []  # Precision im Training
    train_recalls = []     # Recall im Training
    train_f1_scores = []   # F1-Score im Training
    val_precisions = []    # Precision in Validierung
    val_recalls = []       # Recall in Validierung
    val_f1_scores = []     # F1-Score in Validierung
    test_precisions = []   # Precision im Test
    test_recalls = []      # Recall im Test
    test_f1_scores = []    # F1-Score im Test

    ### Epochenweises Training des GNN
    for epoch in range(Epochenzahl):
        ### Setze Modell in Trainingsmodus für spezifisches Verhalten (Dropout, BatchNorm, etc.)
        total_loss = 0
        model.train()

        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            # print(f"out shape: {out.shape}, data.y shape: {data.y.shape}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            ## Absichern, dass gelabelte Daten vorhanden sind.
            if data.y.numel() > 0:                
                if Kriterium == 'BCEWithLogitsLoss':
                    loss = criterion(out, data.y.float())  # Konvertiere data.y zu FloatTensor
                else:
                    loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            else:
                print("Keine beschrifteten Daten vorhanden!!!")
        
        ## Fehlerberechnung der Verlustfunktion (Training)
        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)
        train_losses.append(avg_loss)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss}")

        ## Modellauswertung mit Trainingsdatenset
        train_accuracy, train_precision, train_recall, train_f1, train_con_matrix = evaluate_model(model, train_loader)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1_scores.append(train_f1)

        ### Setze Modell in Validierungsmodus
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                
                ## Absichern, dass gelabelte Daten vorhanden sind.
                if data.y.numel() > 0:
                    if Kriterium == 'BCEWithLogitsLoss':
                        loss = criterion(out, data.y.float())  # Konvertiere data.y zu FloatTensor
                    else:
                        loss = criterion(out, data.y)
                    val_loss += loss.item()
        
        ## Fehlerberechnung der Verlustfunktion (Validierung)
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        ## Modellauswertung mit Validierungsdatenset
        val_accuracy, val_precision, val_recall, val_f1, val_con_matrix = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)   
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)     

        ## Modellauswertung mit Testdatenset
        test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)

        ### Konsolen-Output für Dokumentation
        print(f"__Epoch: {epoch +1 }:")
        print(f"\tTrain Accuracy: {train_accuracy}, Precision: {train_accuracy}, Recall: {train_accuracy}, F1-Score: {train_accuracy}")
        print(f"\tVal Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1-Score: {val_f1}")
        print(f"\tTest Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1-Score: {test_f1}")
        print(f"\tTrain Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")
        
        ## Speicher bestes Parameterset
        if avg_loss < best_loss:
            best_loss = avg_loss
            plotter.save_model(model, best= True)
        
        ## Überprüfe Lernfortschritt
        if early_stopping(avg_loss):
            print(f"__Training Abbruch bei Epoche {epoch + 1} aufgrund Early Stopping")
            break
        
    ## Speichern und Plotten
    plotter.save_model(model, best= False)
    plotter.plot_loss_curve(loss_values)
    plotter.plot_learning_curves(train_losses, val_losses)
    plotter.plot_accuracies(train_accuracies, val_accuracies, test_accuracies)
    plotter.plot_confusion_matrix(test_conf_matrix)
    plotter.plot_precision_recall_f1(test_precisions, test_recalls, test_f1_scores)





"""Hilfsfunktionen """
def get_folder_paths(script_dir, folder_names):
    """Gibt eine Liste von Pfaden zu den Ordnern zurück"""  
    graph_folder_paths = [script_dir.parent / "1_2 Data_Preprocessing" / folder_name / 'Graph_Save' for folder_name in folder_names]
    # # print(f"folder-paths: {json_folder_paths}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return graph_folder_paths


def load_graph_from_json(json_file_path):
    """Lädt den 3D-Graphen mit den Knotenpositionen aus einer JSON-Datei"""
    with open(json_file_path, "r") as f:
        graph_data = json.load(f)
    
    ## Automatische Analyse des Graphen
    num_nodes = len(graph_data["nodes"])
    num_edges = len(graph_data["edges"])
    logging.info(f"Loaded file_path: {json_file_path}, num_nodes: {num_nodes}, num_edges: {num_edges}")
    # print(f"Graphdaten: {num_nodes} Geladene Knoten und {num_edges} geladene Kanten") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ## Erstelle Graphen
    G = nx.Graph()
    for node, data in graph_data['nodes'].items():
        node = int(node) # Absichern, dass node ID eine Ganzzahl ist
        G.add_node(node, **data) # Unverbundener Graph mit Knoten und Knotenmerkmalen --> Kein G.add_edges()...
    # print("Graph: ", G) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Erstelle Kanten
    existing_edges = {(edge['source'], edge['target']) for edge in graph_data['edges']}
    possible_edges = set(combinations(G.nodes, 2)) # set == {(), (), ()}
    # print("Kanten_existent sortiert: ", sorted(existing_edges)) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print("Kanten_möglich sortiert: ", sorted(possible_edges)) # Gibt Liste vom Set #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    edges = [] # Wird eine Liste mit Dicts aus allen möglichen Kanten
    ## Labeln der möglichen Kanten
    for edge in possible_edges:
        label = 1 if edge in existing_edges or (edge[1], edge[0]) in existing_edges else 0
        edges.append({'source': edge[0], 'target': edge[1], 'label': label})
    
    graph_id = graph_data['file_id']
    return G, edges, graph_id


def create_data_from_graph(G, edges, graph_id):
    """Erstellt ein PyTorch-Geometric-Data-Objekt aus einem Netzwerkx-Graphen"""
    ## Extrahiere Knotenmerkmale
    node_features = []
    for _, data in G.nodes(data= True):
        for value in data.values():
            if isinstance(value, list):
                node_features.extend(value)
            else:
                node_features.append(value)
    # print(f"node_features: {node_features})

    ## Linearisiere/Gätte/Flache die Merkmalliste der Knoten ab
    flat_node_features = [item for sublist in node_features for item in (sublist if isinstance(sublist, list) else [sublist])]
    x = torch.tensor(flat_node_features, dtype= torch.float).view(len(G.nodes), -1)

    ## Verwende die Kanten aus 'edges' für edge_index
    edge_index = torch.tensor([(edge['source'], edge['target']) for edge in edges], dtype=torch.long).t().contiguous()

    ## Extrahiere die Labels direkt aus 'edges' für y
    y = torch.tensor([edge['label'] for edge in edges], dtype= torch.long)
    # print(f"y aus create_data_from_graph: {y}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## Erste Data-Objekt
    data = Data(x= x, edge_index= edge_index, y= y)
    data.graph_id = graph_id  # Füge die 'graph_id' der 'data' als Attribut hinzu
    return data


def evaluate_model(model,loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)

            if Kriterium == 'BCEWithLogitsLoss':
                preds = (torch.sigmoid(out) > 0.5).long()  # Sigmoid-Aktivierung und Schwellenwert von 0.5
            else:
                preds = torch.argmax(out, dim= 1) # Gibt den Index der Klasse mit der höchsten Wahrscheinlichkeit für jede Zeile zurück
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    print(f"\tAnzahl tatsächlicher Labels und vorhergesagter Labels: {len(all_labels)} und {len(all_preds)}") #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ## Berechnung der Metriken
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division= 1)
    recall = recall_score(all_labels, all_preds, zero_division= 1)
    f1 = f1_score(all_labels, all_preds, zero_division= 1)

    ## Berechnung der Confusion-Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, conf_matrix


def add_random_edges(G, num_edges: int, positives):
    """Bugfixing-Funktion zum erstellen zufäller gelabelter Daten"""
    nodes = list(G.nodes)
    choices = random.choices #Beispiel: choices(['red', 'black', 'green'], [18, 18, 2], k=6) -> ['red', 'green', 'black', 'black', 'red', 'black']
    ## Erstellt Anzahl möglicher Kanten mit zufällig vielen positiven
    random_edges = [{'source': random.choice(nodes), 'target': random.choice(nodes), 'label': choices([0,1], [1-positives, positives])[0]} for _ in range(num_edges)]
    return random_edges


class Tee:
    """Spaltet Output in mehrere Wege auf"""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush # Absicherung, dass die Ausgabe direkt gespeichert wird
    
    def flush(self):
        for f in self.files:
            f.flush


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss


class EarlyStopping:
    def __init__(self, patience= 5, min_delta= 0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class Plotter:
    def __init__(self, script_dir, graph_folder_path):
        self.script_dir = script_dir
        self.model_folder_name = Path(graph_folder_path).parent.name
        self.model_folder_path = self.script_dir / self.model_folder_name
        os.makedirs(self.model_folder_path, exist_ok= True)

    
    def save_model(self, model, best= False):
        ## Speicher Modell in Graph-Format
        model_path = self.model_folder_path / ("gnn_model_best.pth" if best else "gnn_model_last.pth")
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")

        ## Speicher Modell in JSON-Format
        model_path_json = self.model_folder_path / ("gnn_model_best.json" if best else "gnn_model_last.json")
        model_params = {k: v.tolist() for k, v in model.state_dict().items()}
        with open(model_path_json, 'w') as f:
            json.dump(model_params, f)
        logging.info(f"Model parameters saved to {model_path_json}")
    

    def save_terminal_output(self, output):
        """Leitet Terminal Output in .txt um. Bei 'with open...' muss immer der ganze Pfad hinein"""
        file_name = 'terminal_output.txt'
        file_path = self.model_folder_path / file_name
        with open(file_path, 'w') as f:
            f.write(output)
        logging.info(f"Temrinal output saved to {file_path}")
    

    """NIRGENDS EINGEBAUT: SPÄTER DANN FÜR ERGEBNISSE"""
    def plot_predicted_graph_3D(self, G):
        """Plottet den 3D-Graphen mit den vorhergesagten Verbindungen"""
        pos = {node: data['position'] for node, data in G.nodes(data=True)}
        color_map = {
            0: 'gray',  # IfcBuildingElementProxy
            1: 'aquamarine',  # IfcBeam
            2: 'darkorange',  # IfcColumn
            3: 'darkolivegreen',  # IfcFoundation
            4: 'mediumblue',  # IfcSlab
            5: 'firebrick',  # IfcWall
        }
        """Liste mit allen statisch relevanten Bauteilen:
        Stütze, Wand, Balken, Decke, Fundament, 
        Pfahl, Treppe, Rampe, Dach"""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ## Plotte Knoten
        for node, (x, y, z) in pos.items():
            node_color = color_map.get(G.nodes[node]['ifc_class'], 'gray')
            ax.scatter(x, y, z, color= node_color, s= 50)
            ax.text(x, y, z, s= node, fontsize= 10)

        ## Plotte Kanten
        for edge in G.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            z = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x, y, z, color='black')
        
        plt.title('GNN_Graph')
        plt.legend()

        ## Speicher und plotte den Graphen
        GNN_image_folder_path = self.model_folder_path / "GNN_Graph_Image"
        os.makedirs(GNN_image_folder_path, exist_ok=True) # Ordner erstellen, falls noch nicht vorhanden
        GNN_image_file_name = 'Variable GNN name.png'
        GNN_image_file_path = GNN_image_folder_path / GNN_image_file_name
        plt.savefig(GNN_image_file_path)
        print(f"GNN-Graph gespeichert als {GNN_image_file_name}")

        # plt.show()
        logging.info(f"GNN-Graph plot saved as {GNN_image_file_name}")
        

    def plot_loss_curve(self, loss_values):
        """Plottet die Verlustkurve für das Training"""
        plt.figure()
        plt.plot(loss_values, label= 'Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
    
        ## Speicher und Plotte Bild
        file_name = 'training_loss.png'
        plt.savefig(self.model_folder_path / file_name)
        print(f"Diagramm gespeichert als {file_name}")

        # plt.show()
        logging.info(f"Training loss plot saved as {file_name}")


    def plot_learning_curves(self, train_losses, val_losses):
        """Plottet die Trainings- und Validierungsverluste"""
        print(f"\tTrain Losses: {train_losses}")
        print(f"\tVal Losses: {val_losses}")

        plt.figure()
        plt.plot(train_losses, label= 'Training Loss')
        plt.plot(val_losses, label= 'Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()

        ## Speicher und Plotte Bild
        file_name = 'learning_curves.png'
        plt.savefig(self.model_folder_path / file_name)
        print(f"Diagramm gespeichert als {file_name}")

        # plt.show()
        logging.info(f"Learning curves plot saved as {file_name}")
    

    def plot_accuracies(self, train_accuracies, val_accuracies, test_accuracies):
        """Plottet die Genauigkeiten für Training, Validierung und Test"""
        plt.figure()
        plt.plot(train_accuracies, label= 'Train Accuracy')
        plt.plot(val_accuracies, label= 'Validation Accuracy')
        plt.plot(test_accuracies, label= 'Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracies Over Epochs')
        plt.legend()

        ## Speicher und Plotte Bild
        file_name = 'accuracies.png'
        plt.savefig(self.model_folder_path / file_name)
        print(f"Diagramm gespeichert als {file_name}")

        # plt.show()
        logging.info(f"Accuracies plot saved as {file_name}")
    

    def plot_confusion_matrix(self, conf_matrix):
        """Plottet die Cnfusion-Matrix der Testdaten"""
        plt.figure(figsize= (6, 6))
        plt.imshow(conf_matrix, interpolation= 'nearest', cmap= plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(len(conf_matrix)), range(len(conf_matrix)))
        plt.yticks(range(len(conf_matrix)), range(len(conf_matrix)))

        ## Zahlen in die Matrix schreiben
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[i])):
                plt.text(j, i, conf_matrix[i, j], ha= 'center', va= 'center', color= 'red')
        
        ## Speicher und Plotte Bild
        file_name = 'confusion_matrix.png'
        plt.savefig(self.model_folder_path / file_name)
        print(f"Confusion Matrix gespeichert als {file_name}")

        # plt.show()
        logging.info(f"Confusion matrix plot saved as {file_name}")
    
    
    def plot_precision_recall_f1(self, precisions, recalls, f1_scores):
        """Plottet Precision, Recall und F1-Score über die Epochen"""
        plt.figure()
        plt.plot(precisions, label='Precision')
        plt.plot(recalls, label='Recall')
        plt.plot(f1_scores, label='F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1-Score Over Epochs')
        plt.legend()

        # Speicher und Plotte Bild
        file_name = 'precision_recall_f1.png'
        plt.savefig(self.model_folder_path / file_name)
        print(f"Diagramm gespeichert als {file_name}")

        # plt.show()
        logging.info(f"Precision, Recall, and F1-Score plot saved as {file_name}")









if __name__ == "__main__":
    logging.info("___Intitialisiere Skript___")
    
    main()

    logging.info("___Skript erfolgreich beendet___")
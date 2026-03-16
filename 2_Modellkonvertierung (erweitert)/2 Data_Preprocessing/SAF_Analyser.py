import pandas as pd

def analyze_saf_connections(file_path, start_index=0, end_index=None):
    print(f"Lese Datei ein: {file_path}...\n")
    
    try:
        # Einlesen der definierten Tabellenblätter
        df_nodes = pd.read_excel(file_path, sheet_name='StructuralPointConnection')
        df_curve = pd.read_excel(file_path, sheet_name='StructuralCurveMember')
        df_rib = pd.read_excel(file_path, sheet_name='StructuralCurveMemberRib')
        df_surface = pd.read_excel(file_path, sheet_name='StructuralSurfaceMember')
    except Exception as e:
        print(f"Fehler beim Einlesen der Tabellenblätter. Bitte prüfen, ob alle existieren: {e}")
        return

    # Dictionary zur Speicherung initialisieren (Schlüssel = Knoten-Name)
    node_connections = {str(node): {'1D_Elements': [], '2D_Elements': []} 
                        for node in df_nodes['Name']}
    
    # Set zum Sammeln von Knoten, die im Bauteil stehen, aber nicht in der Knotenliste
    missing_nodes = set()

    # Hilfsfunktion zum Zuweisen der Bauteile zu den jeweiligen Knoten
    def add_connections(element_name, nodes_string, element_type):
        if pd.notna(nodes_string):
            nodes = str(nodes_string).split(';') #split erzeugt Liste
            for node in nodes:
                node = node.strip() #split entfernt evtl. Leerzeichen (nur zur Sicherheit)
                if node in node_connections:
                    node_connections[node][element_type].append(str(element_name))
                elif node: # Wnn der Knoten nicht leer ist, aber im Dictionary fehlt
                    missing_nodes.add(node)

    # Hinweis: .iterrows gibt immer 2 Werte zurück (Zeilenindex und Zellenwert) - deshalb brauche ich _ als Wegwerf-Variable
    # 1. Stabelemente (StructuralCurveMember)
    for _, row in df_curve.iterrows():
        add_connections(row['Name'], row['Nodes'], '1D_Elements')

    # 2. Rippen/Stabelemente (StructuralCurveMemberRib)
    for _, row in df_rib.iterrows():
        add_connections(row['Name'], row['Nodes'], '1D_Elements')

    # 3. Flächenelemente (StructuralSurfaceMember)
    for _, row in df_surface.iterrows():
        element_name = row['Name']
        
        # Normale Randknoten verarbeiten
        if 'Nodes' in row:
            add_connections(element_name, row['Nodes'], '2D_Elements')
            
        # Interne Knoten verarbeiten (mit exakter Schreibweise)
        if 'Internal nodes' in row:
            add_connections(element_name, row['Internal nodes'], '2D_Elements')

    # 4. Dynamische Printausgabe der gewählten Range im Titel durch Slicing
    print(f"=== Verbindungsanalyse der Knoten (Index {start_index} bis {end_index if end_index else 'Ende'}) ===")
    for node_id, connections in list(node_connections.items())[start_index:end_index]:
        print(f"Knoten {node_id}:")
        
        # Set wird verwendet, um eventuelle Duplikate bei fehlerhaften SAFs zu filtern
        elements_1d = list(set(connections['1D_Elements']))
        elements_2d = list(set(connections['2D_Elements']))
        
        str_1d = ", ".join(elements_1d) if elements_1d else "Keine"
        str_2d = ", ".join(elements_2d) if elements_2d else "Keine"
        
        print(f"  - 1D-Elemente: {str_1d}")
        print(f"  - 2D-Elemente: {str_2d}")
        print("-" * 30)

        # Ausgabe der Warnung am Ende des Skripts
        if missing_nodes:
            print("\n!!! WARNUNG: Inkonsistente Knotenreferenzen !!!")
            print("Folgende Knoten wurden in Bauteilen referenziert, existieren aber nicht im Tabellenblatt 'StructuralPointConnection':")
            print(", ".join(sorted(missing_nodes)))
            print("-" * 30)

# Ausführung starten (Pfad entsprechend anpassen!)
analyze_saf_connections('21_22 L_TWP_Tragwerksmodell 0D Ausr Train.xlsx', end_index= 5)
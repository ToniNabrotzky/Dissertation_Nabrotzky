import pandas as pd

def analyze_saf_connections(file_path, start_index=0, end_index=None):
    print(f"Lese Datei ein: {file_path}...\n")
    
    try:
        # Einlesen der definierten Tabellenblätter
        df_nodes = pd.read_excel(file_path, sheet_name='StructuralPointConnection')
        df_curve = pd.read_excel(file_path, sheet_name='StructuralCurveMember')
        print("\n--- DEBUG INFO ---")
        print("Alle erkannten Spaltennamen:", df_curve.columns.tolist())
        print("Exakter Dateninhalt der ersten Zeile:\n", df_curve.iloc[0].to_dict())
        print("------------------\n")
        df_rib = pd.read_excel(file_path, sheet_name='StructuralCurveMemberRib')
        df_surface = pd.read_excel(file_path, sheet_name='StructuralSurfaceMember')
    except Exception as e:
        print(f"Fehler beim Einlesen der Tabellenblätter. Bitte prüfen, ob alle existieren: {e}")
        return
    
    # Feste Reihenfolge für Namensgebung
    type_order = ['Stütze', 'Balken', 'Verband', 'Platte', 'Wand']

    # Mapping von SAF-Typen (Englisch) zu Output-Typen (Deutsch)
    type_mapping = {
        'Column': 'Stütze',
        'Beam': 'Balken',
        'Member': 'Verband',
        'Slab': 'Platte',
        'Wall': 'Wand'
    }

    # Dictionary zur Speicherung initialisieren (Schlüssel = Knoten-Name)
    # Dictionary speichert pro Knoten weiteres Dictionary: {Elementname: Elementtyp}
    # Dies verhindert doppelte Zählungen desselben Elements am selben Knoten
    node_connections = {str(node): {} for node in df_nodes['Name']}
    
    # Set zum Sammeln von Knoten, die im Bauteil stehen, aber nicht in der Knotenliste
    missing_nodes = set()

    # Hilfsfunktion zum Zuweisen der Bauteile zu den jeweiligen Knoten
    def add_connections(element_name, nodes_string, saf_type):
        if pd.notna(nodes_string) and pd.notna(saf_type):
            mapped_type = type_mapping.get(str(saf_type).strip()) # strip entfernt evtl. Leerzeichen

            if not mapped_type:
                # Fall ein unbekannter Typ in der Excel steht, wird er hier übersprungen
                return

            nodes = str(nodes_string).split(';') # split erzeugt Liste
            for node in nodes:
                node = node.strip()
                if node in node_connections:
                    # Speichert/Überschreibt das Element mit seinem Typ
                    node_connections[node][str(element_name)] = mapped_type
                elif node: 
                    # Wenn der Knoten nicht leer ist, aber im Dictionary fehlt
                    missing_nodes.add(node)

    # Hinweis: .iterrows gibt immer 2 Werte zurück (Zeilenindex und Zellenwert) - deshalb brauche ich _ als Wegwerf-Variable
    # 1. Stabelemente (StructuralCurveMember)
    for _, row in df_curve.iterrows():
        saf_type = row['Type'] # Typ aus Spalte "Type" lesen
        # print(saf_type) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # add_connections(row['Name'], row['Nodes'], 'Column') # !!!!!!!!!!!!!!!!!!!!!!!!!!
        if 'Type' in row:
            add_connections(row['Name'], row['Nodes'], saf_type)
            # add_connections(row['Name'], row['Nodes'], 'Column') # !!!!!!!!!!!!!!!!!!!!!!!!

    # 2. Rippen/Stabelemente (StructuralCurveMemberRib)
    for _, row in df_rib.iterrows():
        # Typ ist immer 'Beam', ignoriert die Spalte
        add_connections(row['Name'], row['Nodes'], 'Beam')

    # 3. Flächenelemente (StructuralSurfaceMember)
    for _, row in df_surface.iterrows():
        if 'Type' in row:
            element_name = row['Name']
            saf_type = row['Type'] # Typ aus Spalte "Type" lesen
        
            # Normale Randknoten verarbeiten
            if 'Nodes' in row:
                add_connections(element_name, row['Nodes'], saf_type)
                
            # Interne Knoten verarbeiten (mit exakter Schreibweise)
            if 'Internal nodes' in row:
                add_connections(element_name, row['Internal nodes'], saf_type)

    # 4. Dynamische Printausgabe der gewählten Range im Titel durch Slicing
    print(f"=== Verbindungsanalyse der Knoten (Index {start_index} bis {end_index if end_index else 'Ende'}) ===")
    # print(node_connections) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for node_id, elements_dict in list(node_connections.items())[start_index:end_index]:
        # Duplikate bei den Elementnamen entfernen
        total_elements = len(elements_dict)

        # Logik für Bezeichnung der Anschlussvariante (Kategoriename)
        if total_elements <= 1:
            category_name = "..."
        else:
            # Zählt die Vorkommen der Typen für diesen Knoten
            type_counts = {t: 0 for t in type_order}
            for elem_type in elements_dict.values():
                type_counts[elem_type] += 1
            
            # Baut den String in der strikten Reihenfolge auf
            name_parts = []
            for t in type_order:
                if type_counts[t] > 0:
                    name_parts.append(f"{t}{type_counts[t]}")
            
            category_name = "_".join(name_parts)

        # Printausgabe
        element_names = list(elements_dict.keys())
        print(f"Knoten {node_id}:")
        print(f"  - Kategorie: {category_name}")
        print(f"  - Beteiligte Elemente ({total_elements}): {', '.join(element_names) if element_names else 'Keine'}")
        print("-" * 30)
        
        # Ausgabe der Warnung am Ende des Skripts
        if missing_nodes:
            print("\n!!! WARNUNG: Inkonsistente Knotenreferenzen !!!")
            print(", ".join(sorted(missing_nodes)))
            print("-" * 30)


# Ausführung starten (Pfad entsprechend anpassen!)
analyze_saf_connections('21_22 L_TWP_Tragwerksmodell 0D Ausr Train.xlsx', end_index= 85)
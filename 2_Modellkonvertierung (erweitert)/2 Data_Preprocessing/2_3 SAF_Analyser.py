import pandas as pd

def analyze_saf_connections(
        file_path, output_csv_path='verbindungsanalyse.csv',
        start_index=0, end_index=None
        ):
    print(f"Lese Datei ein: {file_path}...\n")
    
    try:
        #  Einlesen der definierten Tabellenblätter
        df_nodes = pd.read_excel(file_path, sheet_name='StructuralPointConnection')
        df_curve = pd.read_excel(file_path, sheet_name='StructuralCurveMember')
        df_rib = pd.read_excel(file_path, sheet_name='StructuralCurveMemberRib')
        df_edge = pd.read_excel(file_path, sheet_name='StructuralCurveEdge')
        df_surface = pd.read_excel(file_path, sheet_name='StructuralSurfaceMember')
    except Exception as e:
        print(f"Fehler beim Einlesen der Tabellenblätter. Bitte prüfen, ob alle existieren: {e}")
        return

    # Spaltennamen bereinigen
    for df in [df_nodes, df_curve, df_rib, df_surface, df_edge]:
        df.columns = df.columns.str.strip()

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
    missing_nodes = set()
    unrecognized_types = set()

    # Hilfsfunktion zum Zuweisen der Bauteile zu den jeweiligen Knoten
    def add_connections(element_name, nodes_string, saf_type):
        if pd.notna(nodes_string) and pd.notna(saf_type):
            mapped_type = type_mapping.get(str(saf_type).strip())
            
            if not mapped_type:
                # Unbekannter Typ, wird hier gesammelt und später ausgegeben
                unrecognized_types.add(str(saf_type))
                return 

            nodes = str(nodes_string).split(';') # SPlit erzeugt Liste
            for node in nodes:
                node = node.strip()
                if node in node_connections:
                    # Speichert/Überschreibt das Element mit seinem Typ
                    node_connections[node][str(element_name)] = mapped_type
                elif node:
                    # Wenn der Knoten nicht leer ist, aber im Dictionary fehlt, wird er hier gesammelt
                    missing_nodes.add(node)

    # Lookup-Dictionary für Flächenelemente (Name -> Typ)
    # Erforderlich, da in StructuralCurveEdge der Typ nicht mehr steht
    """Hinweis für folgenden Code:
    - .iterrows gibt immer 2 Werte zurück (Zeilenindex und Zellenwert)
    - Deshalb brauche ich _ als Wegwerf-Variable
    - Alte Variante row['Type'] == row.get('Type') 
    - get gibt None zurück, wenn Spalte fehlt, statt Fehler zu werfen"""

    surface_types = {}
    for _, row in df_surface.iterrows():
        if pd.notna(row.get('Name')) and pd.notna(row.get('Type')):
            surface_types[str(row.get('Name'))] = str(row.get('Type'))

    # 1. Stabelemente (StructuralCurveMember)
    for _, row in df_curve.iterrows():
        add_connections(row.get('Name'), row.get('Nodes'), row.get('Type'))

    # 2. Rippen (StructuralCurveMemberRib))
    for _, row in df_rib.iterrows():
        add_connections(row.get('Name'), row.get('Nodes'), 'Beam')

    # 3. Flächenelemente (StructuralSurfaceMember)
    for _, row in df_surface.iterrows():
        element_name = row.get('Name')
        saf_type = row.get('Type')
        # Normale Ranknoten verarbeiten
        if pd.notna(row.get('Nodes')):
            add_connections(element_name, row.get('Nodes'), saf_type)
        # Interne Knoten verarbeiten
        if pd.notna(row.get('Internal nodes')):
            add_connections(element_name, row.get('Internal nodes'), saf_type)

    # 4. Kanten von Flächenelementen (StructuralCurveEdge)
    for _, row in df_edge.iterrows():
        element_name = row.get('2D member')
        nodes_string = row.get('Nodes')
        
        if pd.notna(element_name) and pd.notna(nodes_string):
            # Typ aus dem Lookup-Dictionary abrufen
            saf_type = surface_types.get(str(element_name))
            if saf_type:
                add_connections(element_name, nodes_string, saf_type)

    # 5. Dynamische Printausgabe der gewählten Range im Titel durch Slicing
    print(f"""\n=== Verbindungsanalyse der Knoten (Index 
          {start_index} bis {end_index if end_index else 'Ende'}) ===""")
    
    # Liste für die CSV-Ausgabe initialisieren
    export_data = []
    
    for node_id, elements_dict in list(node_connections.items())[start_index:end_index]:
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

        element_names = list(elements_dict.keys())
        elements_string = ', '.join(element_names) if element_names else 'Keine'
        
        # Konsolenausgabe - detailliert
        # print(f"Knoten {node_id}:")
        # print(f"  - Kategorie: {category_name}")
        # print(f"  - Beteiligte Elemente ({total_elements}): {elements_string}")
        # print("-" * 30)

        # Konsolenausgabe - kompakt
        # print(f"Knoten {node_id}: {category_name}")
        # print(f"{category_name}")

        # Datenzeile für CSV-Format anfügen
        export_data.append({
            'Knoten': node_id,
            'Anschlussvariante': category_name,
            'Anzahl Elemente': total_elements,
            'Elemente': elements_string
        })

    # DataFrame aus der Liste erstellen und als CSV speichern
    if export_data:
        df_export = pd.DataFrame(export_data)
        df_export.to_csv(output_csv_path, index=False, 
                            sep=';', encoding='utf-8-sig')
        print(f"\n>>> ERFOLG: Die Ergebnisse wurden in '{output_csv_path}' gespeichert. <<<")

    # Ausgabe der Warnung am Ende des Skripts
    if unrecognized_types:
        print("\n!!! ANALYSE: Unbekannte Bauteiltypen gefunden !!!")
        print("Folgende Werte konnten nicht zugeordnet werden:")
        print(", ".join(unrecognized_types))
        
    if missing_nodes:
        print("\n!!! WARNUNG: Inkonsistente Knotenreferenzen !!!")
        print(", ".join(sorted(missing_nodes)))


# Ausführung starten (Pfad entsprechend anpassen!)
analyze_saf_connections('./Analysemodelle/21_22 L_TWP_Tragwerksmodell 0D Ausr Anschluss.xlsx')
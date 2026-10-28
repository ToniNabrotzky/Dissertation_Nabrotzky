"""
_14_Experiment1_RBC_Fehlertaxonomie.py
======================================
Experiment 1 RBC-Benchmark: Identifikation und Bewertung von Joints
(Anschlussstellen) als Grundlage einer Fehlertaxonomie regelbasierter Konverter.

Das Modul wird von '_14 IFC_Graph_Generator.py' eingebunden und stellt die
Klasse 'Experiment1RBCFehlertaxonomie' bereit (Mixin der Klasse GNNPipeline).

Joints sind Hyperkanten. Sie gehören nicht direkt zum Kontaktgraphen,
beziehen sich aber auf dessen Bestandteile.

Struktur:
1. KONFIGURATION (Experiment 1)
2. KLASSE Experiment1RBCFehlertaxonomie
   1.A Rekonstruktion von Anschlussstellen (Soll-Joints, IFC-/GT-basiert)
   1.B Klassifizierungssystem für Anschlussstellen
   1.C Extraktion von Anschlussstellen aus dem Analysemodell (Ist-Joints)
   1.D Validierung: Matching, Fehlercodes (E0-E7), Statistik
"""
from typing import Callable

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree #type: ignore

# =========================================================
# KONFIGURATION (Experiment 1 RBC-Fehlertaxonomie)
# =========================================================
# 1.A Rekonstruktion von Anschlussstellen (Soll-Joints)
JOINT_EPSILON = 0.005  # [m] Kontakttoleranz eps in [m]. HINWEIS: Sensitivitätsstudie (Empfehlung: 1-5 mm). Default = 0.005
JOINT_TAU = 0.2        # [-] Schwellenwert tau für Dimensionalitätsverhältnisse L2/L1 und L3/L1 (linien-/flächen-/punktförmig). Default = 0.2
JOINT_DELTA_SCALE = 1.0 # [-] Skalierung der Joint-Toleranz delta mit der größten Bauteildicke max(t_a, t_b). Default = 1.0

# 1.B Klassifizierungssystem: noch keine Parameter
# 1.C Extraktion aus dem Analysemodell: noch keine Parameter
# 1.D Validierung: noch keine Parameter


# =========================================================
# EXPERIMENT 1 RBC-FEHLERTAXONOMIE
# =========================================================
class Experiment1RBCFehlertaxonomie:
    """
    Identifikation und Bewertung von Joints:
    1.A Rekonstruktion von Anschlussstellen (Soll-Joints),
    1.B Klassifizierungssystem für Anschlussstellen,
    1.C Extraktion von Anschlussstellen aus dem Analysemodell (Ist-Joints),
    1.D Validierung (Matching, Fehlercodes E0-E7, Statistik).

    Erwartet von der Hauptklasse (GNNPipeline): die Methode log().
    """
    log: Callable[..., None]

    # ---------------------------------------------------------
    # 1.A REKONSTRUKTION VON ANSCHLUSSSTELLEN (SOLL-JOINTS)
    #     Grundlage: Ground Truth + IFC-Geometrie
    # ---------------------------------------------------------
    # Methodenreihenfolge: interne Hilfsfunktionen (_) alphabetisch,
    # danach Orchestrierung, danach weitere Unterfunktionen.
    # ---------------------------------------------------------
    def _attribute_joints(self, components, contact_regions, gt_pairs):
        """
        Schritt 3 der Joint-Rekonstruktion: Attribuiert die geclusterten Joints.
        Berechnet Position, Geometrietyp, Bauteilmenge und den induzierten GT-Teilgraph.

        Args:
            components (list[set]): Zusammenhangskomponenten (Indexmengen) aus dem Koinzidenz-Graph.
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Attribuierte Soll-Joints mit Schlüsseln:
                'position', 'geometry_type', 'member_guids', 'gt_subgraph_edges'.
        """
        joints = []
        for joint_index, comp in enumerate(components, start=1):
            # Sammle alle Kontaktregionen und Bauteile dieser Komponente
            comp_regions = [contact_regions[i] for i in comp]
            contact_region_ids = [region["region_id"] for region in comp_regions]
            member_guids = set()
            for r in comp_regions:
                member_guids.add(r["guid_a"])
                member_guids.add(r["guid_b"])

            # Position: Schwerpunkt über alle Regions-Schwerpunkte
            positions = np.array([r["centroid"] for r in comp_regions])
            position = np.mean(positions, axis=0)

            # Geometrietyp: PCA über die Vereinigung aller Regionspunkte
            all_points = np.vstack([r["points"] for r in comp_regions])
            eigenvalues, _, L = self._pca_contact_region(all_points) # L = Ausdehnungen der Kontaktregion
            geometry_type = self._classify_dimensionality(L)

            # Induzierter GT-Teilgraph: Welche Paare innerhalb des Joints tragen Last ab?
            member_list = sorted(list(member_guids))
            gt_subgraph_edges = []
            for i in range(len(member_list)):
                for j in range(i + 1, len(member_list)):
                    pair = (member_list[i], member_list[j])
                    if pair in gt_pairs:
                        gt_subgraph_edges.append(pair)

            joint = {
                "joint_id": f"J{joint_index:04d}",
                "contact_region_ids": contact_region_ids,
                "position": [round(float(p), 4) for p in position],
                "geometry_type": geometry_type,
                "member_guids": member_list,
                "gt_subgraph_edges": gt_subgraph_edges,
                "eigenvalues": [round(float(l), 6) for l in eigenvalues],
                "lengths": [round(float(l), 4) for l in L]
            }
            joints.append(joint)
            self.log(
                f"#Debug Schritt 3 Joint {joint['joint_id']}: "
                f"contact_region_ids={contact_region_ids}; member_guids={member_list}; "
                f"position={joint['position']}; geometry_type={geometry_type}; "
                f"gt_subgraph_edges={gt_subgraph_edges}",
                print_console=False
            )
        return joints


    def _classify_dimensionality(self, L):
        """
        Klassifiziert die Dimensionalität einer Kontaktregion anhand der Ausdehnungen.
        Deckt die Überschneidung von volumetrischen und dimensionsreduzierten Elementen ab.
        Entscheidungsregel (Experiment 1, Abschnitt 1.A):
            L2/L1 < tau  und L3/L1 < tau  -> linienhaft (2 Ausdehnungen sind deutlich kleiner als Primärachse)
            L3/L1 < tau  und L2/L1 >= tau -> flächig (1 Ausdehnung ist deutlich kleiner als Primärachse)
            sonst                          -> punktförmig

        Args:
            L (np.ndarray): Ausdehnungen [L1, L2, L3] entlang der Hauptachsen (L1 >= L2 >= L3).

        Returns:
            str: 'linienhaft', 'flaechig' oder 'punktuell'.
        """
        if L[0] < 1e-9: # z.B. Idealisierte 1D-Bauteile
            return "punktuell"
        r21 = L[1] / L[0]
        r31 = L[2] / L[0]
        if r21 < JOINT_TAU and r31 < JOINT_TAU: # z.B. Wand-Wand, Wand-Decke, Decke-Decke
            return "linienhaft"
        if r31 < JOINT_TAU and r21 >= JOINT_TAU: # z.B. Balken-Balken, Stütze-Decke
            return "flaechig"
        return "punktuell" # Evtl. zu unsauber? Dann bräuchte es "sonstige", falls das relevant wird


    def _cluster_contact_regions(self, contact_regions, nodes_by_guid):
        """
        Schritt 2 der Joint-Rekonstruktion: Clustert Kontaktregionen zu Joints.
        Zwei Regionen R_ab und R_cd gehören zum selben Joint, wenn sie räumlich
        koinzident sind: Hausdorff-Distanz d_H(R_ab, R_cd) <= delta.
        Die Toleranz delta skaliert mit der größten Bauteildicke max(t_a, t_b).
        Die transitive Hülle wird über Connected Components gebildet.

        Args:
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1.
            nodes_by_guid (dict): Mapping GUID -> Knoten-Dictionary.

        Returns:
            list[set]: Zusammenhangskomponenten; jede Menge enthält Indizes
                       auf contact_regions, die zum selben Joint gehören.
        """
        n = len(contact_regions)
        if n == 0:
            # Debug: Schritt 2 hat ohne gefundene Kontaktregionen nichts zu clustern.
            self.log("#Debug Schritt 2: Keine Kontaktregionen; es werden keine Joints geclustert.", print_console=False)
            return []

        # Koinzidenz-Graph aufbauen: Knoten = Regionen, Kante = Koinzidenz
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Debug: Jede Regionspaar-Prüfung samt d_H, delta und Entscheidung protokollieren.
        self.log(f"#Debug Schritt 2: Prüfe {n * (n - 1) // 2} Regionspaare; Kriterium d_H <= delta.", print_console=False)
        for i in range(n):
            for j in range(i + 1, n):
                r1 = contact_regions[i]
                r2 = contact_regions[j]
                p1 = r1["points"]
                p2 = r2["points"]

                # Hausdorff-Distanz zwischen den vollständigen Punktmengen (symmetrisch)
                d_h = self._hausdorff_distance(p1, p2)

                # Joint-Toleranz delta: skaliert mit größter Bauteildicke beider Regionen
                t1 = self._get_max_thickness(r1, nodes_by_guid)
                t2 = self._get_max_thickness(r2, nodes_by_guid)
                delta = JOINT_DELTA_SCALE * max(t1, t2)

                # #Debug: GUID-Paare machen auch ohne Blick auf interne Indizes nachvollziehbar,
                # welche Regionen für eine mögliche Zusammenführung verglichen wurden.
                merge = d_h <= delta
                self.log(
                    f"#Debug Schritt 2 Vergleich: {r1['region_id']} ({r1['guid_a']}|{r1['guid_b']}) "
                    f"<-> {r2['region_id']} ({r2['guid_a']}|{r2['guid_b']}); "
                    f"d_H={d_h:.6f} m; t1={t1:.6f} m; t2={t2:.6f} m; "
                    f"delta={delta:.6f} m; zusammenfuehren={merge}",
                    print_console=False
                )
                if merge:
                    G.add_edge(i, j)

        # Transitive Hülle: Zusammenhangskomponenten (abgeschlossene Inseln)
        # bilden die Joint-Cluster, bei denen man innerhalb eines Clusters 
        # von jedem Punkt aus jeden anderen darin erreichen kann.
        components = sorted(
            (sorted(component) for component in nx.connected_components(G)),
            key=lambda component: component[0]
        )
        # #Debug: Explizite Zuordnung der Kontaktregion-IDs zu jedem resultierenden Cluster.
        for joint_index, component in enumerate(components, start=1):
            region_ids = [contact_regions[index]["region_id"] for index in component]
            member_guids = sorted({
                guid
                for index in component
                for guid in (contact_regions[index]["guid_a"], contact_regions[index]["guid_b"])
            })
            self.log(
                f"#Debug Schritt 2 Cluster J{joint_index:04d}: "
                f"contact_regions={region_ids}; member_guids={member_guids}",
                print_console=False
            )
        return [set(component) for component in components]


    def _compute_contact_regions(self, nodes, gt_pairs):
        """
        Schritt 1 der Joint-Rekonstruktion: Bestimmt die Kontaktregion R_ab
        für jedes Ground-Truth-Paar (a, b).
        Definition (Experiment 1, Abschnitt 1.A):
            R_(a,b) = { x in R^3 | d(x, dS_a) <= eps  und  d(x, dS_b) <= eps }
        mit S_a, S_b = Volumenkörper, eps = Kontakttoleranz (JOINT_EPSILON).
        Praktisch: Schnittmenge der abgetasteten Randpunktwolken beider Bauteile
        (Auflösung 2,5 cm) über cKDTree.

        Args:
            nodes (list[dict]): Knotenliste mit 'geom_raw' und 'guid'.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Kontaktregionen mit Schlüsseln:
                'guid_a', 'guid_b', 'points', 'centroid',
                'eigenvalues', 'lengths', 'contact_type'.
        """
        self.log(f"1.A Schritt 1: Bestimme Kontaktregionen für {len(gt_pairs)} GT-Paare (eps = {JOINT_EPSILON*1000:.1f} mm)...")
        # #Debug Schritt 1: Paarstatus und alle erzeugten Regionsdaten landen im Protokoll,
        # nicht in der Konsole. Die 2,5-cm-Abtastung ist grober als eps=5 mm und kann
        # deshalb Kontaktpaare ohne passende Sample-Punkte ergeben.
        self.log(
            f"#Debug Schritt 1 Konfiguration: epsilon={JOINT_EPSILON:.6f} m; "
            "Mesh-Sampling=0.025 m; erwartete Einschränkung: epsilon kleiner als Sampling-Auflösung.",
            print_console=False
        )
        nodes_by_guid = {n["guid"]: n for n in nodes}
        contact_regions = []
        missing_pairs = []

        for guid_a, guid_b in sorted(gt_pairs):
            node_a = nodes_by_guid.get(guid_a)
            node_b = nodes_by_guid.get(guid_b)
            if not node_a or not node_b:
                missing_pairs.append((guid_a, guid_b, "GUID nicht in den extrahierten Knoten gefunden"))
                self.log(
                    f"#Debug Schritt 1 Paar ohne Region: GUID_A={guid_a}; GUID_B={guid_b}; "
                    "Grund=GUID nicht in den extrahierten Knoten gefunden",
                    print_console=False
                )
                continue

            pts_a = node_a["geom_raw"]["test_points"]
            pts_b = node_b["geom_raw"]["test_points"]

            # Abstand aller Punkte von a zu b und umgekehrt
            dist_a_to_b, _ = cKDTree(pts_b).query(pts_a, k=1, workers=-1)
            dist_b_to_a, _ = cKDTree(pts_a).query(pts_b, k=1, workers=-1)

            # Schnittmenge: Punkte, die beide Ränder innerhalb eps berühren
            mask_a = dist_a_to_b <= JOINT_EPSILON
            mask_b = dist_b_to_a <= JOINT_EPSILON
            contact_points = np.vstack([pts_a[mask_a], pts_b[mask_b]])

            if len(contact_points) == 0:
                missing_pairs.append((guid_a, guid_b, "keine Rand-Samples innerhalb epsilon"))
                self.log(
                    f"#Debug Schritt 1 Paar ohne Region: GUID_A={guid_a}; GUID_B={guid_b}; "
                    f"Grund=keine Rand-Samples innerhalb epsilon ({JOINT_EPSILON:.6f} m)",
                    print_console=False
                )
                continue

            # Charakterisierung der Kontaktregion
            centroid = np.mean(contact_points, axis=0)
            eigenvalues, _, L = self._pca_contact_region(contact_points)
            contact_type = self._classify_dimensionality(L)

            region = {
                "region_id": f"CR_{len(contact_regions) + 1:04d}",
                "guid_a": guid_a,
                "guid_b": guid_b,
                "points": contact_points,
                "centroid": centroid,
                "eigenvalues": eigenvalues,
                "lengths": L,
                "contact_type": contact_type
            }
            contact_regions.append(region)

            # #Debug Schritt 1: Vollständige Punktwolke plus abgeleitete Schlüssel
            # ausgeben, damit die Regionsbildung extern nachvollzogen werden kann.
            self.log(
                f"#Debug Schritt 1 Kontaktregion {region['region_id']}: "
                f"keys={list(region.keys())}; GUID_A={guid_a}; GUID_B={guid_b}; "
                f"point_count={len(contact_points)}; centroid={np.round(centroid, 6).tolist()}; "
                f"eigenvalues={np.round(eigenvalues, 8).tolist()}; "
                f"lengths={np.round(L, 6).tolist()} m; contact_type={contact_type}; "
                f"points_m={np.round(contact_points, 6).tolist()}",
                print_console=False
            )

        self.log(f"1.A Schritt 1: {len(contact_regions)} Kontaktregionen gefunden.")
        self.log(
            f"#Debug Schritt 1 Zusammenfassung: GT-Paare={len(gt_pairs)}; "
            f"Kontaktregionen={len(contact_regions)}; ohne Region={len(missing_pairs)}; "
            f"fehlende_Paare={missing_pairs}",
            print_console=False
        )
        return contact_regions


    def _get_max_thickness(self, region, nodes_by_guid):
        """
        Ermittelt die maximale Bauteildicke der beiden an einer Kontaktregion
        beteiligten Bauteile. Als Näherung dient das kleinste der drei
        Primärmaße (AABB-Dimensionen) jedes Bauteils.

        Args:
            region (dict): Kontaktregion mit 'guid_a' und 'guid_b'.
            nodes_by_guid (dict): Mapping GUID -> Knoten-Dictionary.

        Returns:
            float: Maximale Bauteildicke in [m].
        """
        t_max = 0.0
        for guid in (region["guid_a"], region["guid_b"]):
            node = nodes_by_guid.get(guid)
            if node and "dimensions" in node:
                t = min(node["dimensions"])
                t_max = max(t_max, t)
        return t_max


    def _hausdorff_distance(self, pts_a, pts_b):
        """
        Berechnet die symmetrische Hausdorff-Distanz zwischen zwei Punktmengen.
            d_H(A, B) = max( sup_{a in A} inf_{b in B} ||a-b||,
                             sup_{b in B} inf_{a in A} ||a-b|| )

        Args:
            pts_a (np.ndarray): Punktmenge A, Shape (n, 3).
            pts_b (np.ndarray): Punktmenge B, Shape (m, 3).

        Returns:
            float: Hausdorff-Distanz in [m].
        """
        if len(pts_a) == 0 or len(pts_b) == 0:
            return float('inf')
        # Hausdorff verwendet je Richtung den groessten Abstand zum nächsten
        # Gegenpunkt. Das Minimum würde nur irgendeine lokale Nähe prüfen und
        # lange Kontaktregionen fälschlich mit punktuellen Regionen verbinden.
        d_a_to_b = np.max(cKDTree(pts_b).query(pts_a, k=1, workers=-1)[0])
        d_b_to_a = np.max(cKDTree(pts_a).query(pts_b, k=1, workers=-1)[0])
        return max(d_a_to_b, d_b_to_a)


    def _pca_contact_region(self, points):
        """
        Führt eine Hauptkomponentenanalyse (PCA) über eine Punktwolke durch.
        Liefert die Eigenwerte lambda_1 >= lambda_2 >= lambda_3, die zugehörigen
        Eigenvektoren (Hauptachsen) und die Ausdehnungen L_i entlang dieser Achsen.

        Args:
            points (np.ndarray): Punktwolke, Shape (n, 3).

        Returns:
            tuple: (eigenvalues, eigenvectors, lengths)
                eigenvalues (np.ndarray): Eigenwerte der Kovarianzmatrix (absteigend).
                eigenvectors (np.ndarray): Hauptachsen als Spalten.
                lengths (np.ndarray): Ausdehnungen [L1, L2, L3] in [m].
        """
        if len(points) < 2:
            return np.zeros(3), np.eye(3), np.zeros(3)

        cov = np.cov(points.T) # cov = Kovarianzmatrix (Verteilung der Punkte um Massezentrum)
        #Eigenvektor: 3D-Richtung der Hauptachsen.    Eigenwert: Skalare Länge der Ausdehnung in diese Richtungen
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Absteigend sortieren
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Ausdehnungen: Projektion auf Hauptachsen
        proj = points @ eigenvectors # Punkte werden so gedreht, dass sie orthogonal zu Hauptachsen liegen
        lengths = np.max(proj, axis=0) - np.min(proj, axis=0) # Bounding-Box-Dimensionen des Kontaktvolumens

        return eigenvalues, eigenvectors, lengths


    def reconstruct_gt_joints(self, stem, nodes, gt_pairs):
        """
        Orchestriert die Rekonstruktion der Soll-Joints aus der Ground Truth.
        Ablauf (geometrisch gestützt, siehe Anhang Teil A):
            Schritt 1: Kontaktregion pro GT-Paar bestimmen
            Schritt 2: Kontaktregionen zu Joints clustern
            Schritt 3: Joints attribuieren

        Args:
            stem (str): Modellname ohne Dateiendung.
            nodes (list[dict]): Knotenliste mit Geometriedaten.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare.

        Returns:
            list[dict]: Attribuierte Soll-Joints.
        """
        self.log("1.A: Starte Soll-Joint-Rekonstruktion (IFC-basiert)...")
        nodes_by_guid = {n["guid"]: n for n in nodes}

        # Schritt 1: Kontaktregionen bestimmen
        contact_regions = self._compute_contact_regions(nodes, gt_pairs)
        # #Debug Schritt 1 -> Schritt 2: Liste der konkreten Region-IDs, die in
        # das Clustering eingehen. Siehe auch vollständige Details oben im Log.
        self.log(
            f"#Debug Uebergabe an Schritt 2: "
            f"contact_regions={[region['region_id'] for region in contact_regions]}",
            print_console=False
        )

        # Schritt 2: Kontaktregionen zu Joints clustern
        components = self._cluster_contact_regions(contact_regions, nodes_by_guid)

        # Schritt 3: Joints attribuieren
        joints = self._attribute_joints(components, contact_regions, gt_pairs)

        self.log(f"1.A: {len(joints)} Soll-Joints rekonstruiert.")
        return joints


    # ---------------------------------------------------------
    # 1.B KLASSIFIZIERUNGSSYSTEM FÜR ANSCHLUSSSTELLEN
    #     Drei Achsen: Valenz, Idealisierungsversatz, Morphologie
    # ---------------------------------------------------------
    def _classify_joint_morphology(self, joints):
        """
        PLATZHALTER: Topologisch-geometrischer Anschlusstyp (Morphologie).
        - Berührungsmodus: Wird bereits in Schritt 1 als 'contact_type'
          bzw. als Joint-'geometry_type' bestimmt.
        - Anschlusskonfiguration (I-, L-, T-, X-, Y-Anschluss)
        - Winkelcharakteristik
        """
        # TODO: Anschlusskonfiguration und Winkelcharakteristik bestimmen
        return joints


    def _classify_joint_offset(self, joints):
        """
        PLATZHALTER: Idealisierungsversatz -> geometrische Komplexität.
        1. Versatzvektor definieren
        2. Kennzahlen ableiten
        3. Von paarweisen Werten zum Joint aggregieren
        """
        # TODO: Versatzvektor, Kennzahlen und Joint-Aggregation berechnen
        return joints


    def _classify_joint_valence(self, joints):
        """
        PLATZHALTER: Anschlussgrad (Valenz) -> kombinatorische Komplexität.
        - Primärmaß: Valenz |J|
        - Sekundärmaß: Dimensionsheterogenität, Innertopologische Dichte
          (Kanten im induzierten GT-Teilgraph / mögliche Kanten)
        """
        # TODO: Valenz, Dimensionsheterogenität und Innertopologische Dichte bestimmen
        return joints


    def classify_gt_joints(self, joints):
        """
        Orchestriert die Klassifikation der Joints über die drei Achsen
        (Valenz, Idealisierungsversatz, Morphologie).
        """
        joints = self._classify_joint_valence(joints)
        joints = self._classify_joint_offset(joints)
        joints = self._classify_joint_morphology(joints)
        return joints


    # ---------------------------------------------------------
    # 1.C EXTRAKTION VON ANSCHLUSSSTELLEN AUS DEM ANALYSEMODELL
    # ---------------------------------------------------------
    def extract_analysis_joints(self, stem):
        """
        PLATZHALTER: Räumliche Ermittlung der Ist-Joints aus den
        Verbindungsmechanismen des Analysemodells:
        - Geteilter Knoten
        - Geteilte Linie / Knoten auf Flächenrand
        - Explizite Kopplungselemente
        Mit GUID-Rückverfolgung, damit später beide Joint-Cluster
        (Soll und Ist) verglichen werden können.
        HINWEIS: Große Teile liegen bereits in '_13_RFEM_Analyser.py' vor.
        """
        self.log("PLATZHALTER: Ist-Joint-Extraktion (Analysemodell) noch nicht implementiert.")
        joints = []
        # TODO: Verbindungsmechanismen auswerten (geteilter Knoten, geteilte Linie, Kopplungselemente)
        # TODO: GUID-Rückverfolgung implementieren
        return joints


    # ---------------------------------------------------------
    # 1.D VALIDIERUNG: MATCHING, FEHLERCODES & STATISTIK
    # ---------------------------------------------------------
    def match_joints(self, soll_joints, ist_joints):
        """
        PLATZHALTER: Beantwortet 'Welcher Ist-Joint gehört zu welchem Soll-Joint?'
        Zuordnungstabelle über Matching:
        - Informationsquellen: Bauteilmenge (Jaccard) + räumliche Nähe
          (globale Optimierung, Ungarische Methode)
        - Ergebnis: 3 Gruppen: gematchte Paare, unmatched Soll, unmatched Ist
        """
        self.log("PLATZHALTER: Joint-Matching noch nicht implementiert.")
        matching = {"matched": [], "unmatched_soll": [], "unmatched_ist": []}
        # TODO: Jaccard-Ähnlichkeit der Bauteilmengen + räumliche Nähe
        # TODO: Globale Optimierung (Ungarische Methode)
        return matching


    def classify_joint_errors(self, matching):
        """
        PLATZHALTER: Beantwortet 'Was ist genau an jedem Joint schiefgegangen?'
        Fehlertaxonomie: topologische Fehlercodes E0 bis E7 inkl.
        zugehörigem Entscheidungsbaum.
        """
        self.log("PLATZHALTER: Fehlerklassifikation (E0-E7) noch nicht implementiert.")
        error_codes = {}
        # TODO: Entscheidungsbaum der Fehlercodes E0-E7 je Joint anwenden
        return error_codes


    def evaluate_joint_statistics(self, matching, error_codes):
        """
        PLATZHALTER: Beantwortet 'Welche Anschlusstypen sind systematisch fehleranfällig?'
        - Deskriptive Auswertung: O-Klasse (Versatz) als Merkmal;
          Kreuztabellen Merkmal x Fehlertyp
        - Konfirmatorische Auswertung: logistische Regression und Odds Ratios
        - Explorative Auswertung: interpretierbare Regelextraktion
          (Entscheidungsbaum / Assoziationsregeln / SHAP-Werte;
          getrennte Modelle pro Fehlertyp)
        """
        self.log("PLATZHALTER: Statistische Auswertung noch nicht implementiert.")
        stats = {}
        # TODO: Deskriptive Kreuztabellen erstellen
        # TODO: Logistische Regression und Odds Ratios berechnen
        # TODO: Regelextraktion pro Fehlertyp durchführen
        return stats

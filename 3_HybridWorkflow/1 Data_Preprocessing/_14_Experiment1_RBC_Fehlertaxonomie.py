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
JOINT_DELTA_SCALE = 1.05 # [-] Skalierung der Joint-Toleranz delta mit der größten Bauteildicke max(t_a, t_b). Default = 1.05 wegen numerischer Ungeauigkeit

JOINT_POINT_EXTENT = 0.05 # [m] Maximale Ausdehnung L1, bis zu der eine Kontaktregion als 0D_Punkt gilt. Muss >= eps/tau sein, damit dünne Oberflächenkontakte nie als 3D_Volumen enden. Default = 0.05
JOINT_SAMPLE_SPACING = 0.025 # [m] Abtastabstand der Dreiecksflächen (Oberflächenkontakt). Default = 0.025
JOINT_MAX_SAMPLES = 5000 # [-] Obergrenze der Abtastpunkte je Bauteil und Paar bzw. der Gitterpunkte im Volumentest. Default = 5000
JOINT_SEGMENT_LENGTH = 1.0 # [m] Maximale Ausdehnung L1 einer Kontaktregion im Clustering. Längere Regionen werden in Stücke dieser Länge zerlegt und begrenzen so die räumliche Reichweite indirekter Verbindungen. HINWEIS: Sensitivität prüfen. Default = 1.0

# Schlüssel einer Kontaktregion (Reihenfolge wie im Dictionary in _compute_contact_regions)
CONTACT_REGION_KEYS = (
    "region_id", "guid_a", "guid_b", "points", "centroid", "eigenvalues",
    "lengths", "contact_type", "contact_method", "volume"
)

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
    def _attribute_joints(self, components, gt_pairs):
        """
        Schritt 3 der Joint-Rekonstruktion: Attribuiert die geclusterten Joints.
        Berechnet Position, Geometrietyp, Bauteilmenge und den induzierten GT-Teilgraph.

        Args:
            components (list[list[dict]]): Je Joint die zugehörigen Teilstücke/Regionen aus Schritt 2
                (Schlüssel 'part_id', 'region_id', 'guid_a', 'guid_b', 'points', 'centroid').
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Attribuierte Soll-Joints mit Schlüsseln:
                'position', 'geometry_type', 'member_guids', 'gt_subgraph_edges'.
        """
        joints = []
        for joint_index, comp_parts in enumerate(components, start=1):
            self.log("", print_console=False) # Leerzeile: Sprung zum nächsten Joint im Protokoll
            # Sammle alle Teilstücke und Bauteile dieses Joints
            contact_region_ids = sorted({part["region_id"] for part in comp_parts})
            contact_part_ids = [part["part_id"] for part in comp_parts]
            member_guids = set()
            for r in comp_parts:
                member_guids.add(r["guid_a"])
                member_guids.add(r["guid_b"])

            # Position: Schwerpunkt über alle Teilstück-Schwerpunkte
            positions = np.array([r["centroid"] for r in comp_parts])
            position = np.mean(positions, axis=0)

            # Geometrietyp: PCA über die Vereinigung aller Teilstückpunkte
            all_points = np.vstack([r["points"] for r in comp_parts])
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
                "contact_part_ids": contact_part_ids,
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
                f"position={joint['position']}; "
                f"geometry_type={geometry_type}; "
                f"Anzahl_Elements={len(member_list)}; "
                f"contact_region_ids={contact_region_ids}; "
                f"contact_part_ids={contact_part_ids}; "
                f"member_guids={member_list}; "
                f"gt_subgraph_edges={gt_subgraph_edges}",
                print_console=False
            )
        self.log("", print_console=False) # Leerzeile nach dem letzten Joint
        return joints


    def _classify_dimensionality(self, L):
        """
        Klassifiziert die Dimensionalität einer Kontaktregion anhand der Ausdehnungen.
        Deckt die Überschneidung von volumetrischen und dimensionsreduzierten Elementen ab.
        Die Ausdehnungen L1 >= L2 >= L3 stammen aus der PCA der Kontaktregion; r21 = L2/L1, r31 = L3/L1.
        Entscheidungsregel (Experiment 1, Abschnitt 1.A), Prüfung in dieser Reihenfolge:
            L1 <= JOINT_POINT_EXTENT  -> 0D_Punkt   (absolute Größe; Verhältnisse sind hier nicht aussagekräftig)
            r31 >= tau                -> 3D_Volumen (alle drei Ausdehnungen groß; r21 >= r31 gilt per Sortierung automatisch)
            r21 >= tau                -> 2D_Fläche  (L3 klein, L2 groß)
            sonst                     -> 1D_Linie   (L2 und L3 klein gegenüber L1)

        Args:
            L (np.ndarray): Ausdehnungen [L1, L2, L3] entlang der Hauptachsen (L1 >= L2 >= L3).

        Returns:
            str: '0D_Punkt', '1D_Linie', '2D_Fläche' oder '3D_Volumen'.
        """
        if L[0] <= JOINT_POINT_EXTENT: # z.B. Eckkontakt
            return "0D_Punkt"
        r21 = L[1] / L[0]
        r31 = L[2] / L[0]
        if r31 >= JOINT_TAU: # Bauteile durchdringen sich (Schnittvolumen), z.B. Stuetze durch Decke
            return "3D_Volumen"
        if r21 >= JOINT_TAU: # z.B. Stuetze auf Decke, Decke auf Decke vollflächig
            return "2D_Fläche"
        return "1D_Linie" # z.B. Wand auf Decke, Wand-Wand


    def _cluster_contact_regions(self, contact_regions, nodes_by_guid):
        """
        Schritt 2 der Joint-Rekonstruktion: Clustert Kontaktregionen zu Joints.

        Ablauf:
            2.1 Zerlegung: Ausgedehnte Regionen (L1 > JOINT_SEGMENT_LENGTH) werden in Teilstücke
                zerlegt (_split_region). Kompakte Regionen bleiben unverändert.
            2.2 Koinzidenzgraph: Knoten = Teilstücke bzw. kompakte Regionen. Eine Kante entsteht
                bei räumlicher Nähe <= delta, mit delta = JOINT_DELTA_SCALE * max(t_a, t_b):
                - beide Knoten sind kompakte Regionen: Hausdorff-Distanz d_H <= delta
                  (die Regionen müssen sich über ihre gesamte Ausdehnung decken)
                - Teilstück <-> Teilstück (andere Region): Mindestabstand d_min <= delta
                  (Berührung genügt, da ein Teilstück nur einen lokalen Ausschnitt abbildet)
                - Teilstück <-> kompakte Region: Mindestabstand d_min <= delta, aber jedes Teilstück
                  wird nur mit seiner nächstgelegenen kompakten Region verbunden. So kann ein
                  Teilstück nie zwei nahe beieinander liegende Joints (z.B. zwei Stützen) verklammern.
                Teilstücke derselben Ursprungsregion werden nie direkt verbunden. Dadurch kann eine
                lange Region (z.B. 5 m Deckenstoß) keine weit entfernten Stellen indirekt
                verbinden: Jede Verbindung reicht höchstens ein Teilstück weit.
            2.3 Transitive Hülle (Connected Components) auf den Teilstücken.
            2.4 Bewertung der Komponenten:
                - Komponente mit kompakter Region oder Teilstücken aus >= 2 Regionen -> Joint.
                - Komponente nur aus Teilstücken EINER Region (kein Partner in der Nähe):
                    * Hat die Region anderswo einen Joint gefunden, wird sie dort bereits
                      abgebildet -> Teilstück verworfen (kein eigener Joint).
                    * Hat die Region nirgends einen Partner, bildet sie als Ganzes einen Joint.

        Args:
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1.
            nodes_by_guid (dict): Mapping GUID -> Knoten-Dictionary.

        Returns:
            list[list[dict]]: Je Joint die Liste der zugehörigen Teilstücke/Regionen
                (Schlüssel 'part_id', 'region_id', 'guid_a', 'guid_b', 'points', 'centroid', 'is_segment').
        """
        if len(contact_regions) == 0:
            # Debug: Schritt 2 hat ohne gefundene Kontaktregionen nichts zu clustern.
            self.log("#Debug Schritt 2: Keine Kontaktregionen; es werden keine Joints geclustert.", print_console=False)
            return []

        # 2.1 Zerlegung ausgedehnter Regionen in Teilstücke
        parts = []
        for region in contact_regions:
            parts.extend(self._split_region(region))
        n = len(parts)
        self.log(
            f"#Debug Schritt 2.1: {len(contact_regions)} Regionen -> {n} Knoten "
            f"({sum(p['is_segment'] for p in parts)} Teilstücke, JOINT_SEGMENT_LENGTH={JOINT_SEGMENT_LENGTH} m)",
            print_console=False
        )

        # 2.2 Koinzidenzgraph: Knoten = Teilstücke/Regionen, Kante = räumliche Nähe
        trees = [cKDTree(p["points"]) for p in parts]
        box_min = np.array([p["points"].min(axis=0) for p in parts])
        box_max = np.array([p["points"].max(axis=0) for p in parts])
        thickness = [self._get_max_thickness(p, nodes_by_guid) for p in parts]
        G = nx.Graph()
        G.add_nodes_from(range(n))

        checked = 0
        nearest_compact = {} # Teilstück-Index -> (Abstand, Index der nächsten kompakten Region)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = parts[i], parts[j]
                if p1["region_id"] == p2["region_id"]:
                    continue # Teilstücke derselben Region nie direkt verbinden

                # Joint-Toleranz delta: skaliert mit größter Bauteildicke beider Knoten
                delta = JOINT_DELTA_SCALE * max(thickness[i], thickness[j])

                # Vorfilter: Der AABB-Abstand ist eine untere Schranke von d_min und d_H
                gap = np.maximum(np.maximum(box_min[i] - box_max[j], box_min[j] - box_max[i]), 0.0)
                if np.linalg.norm(gap) > delta:
                    continue
                checked += 1
                self.log("", print_console=False) # Leerzeile: Sprung zum nächsten Vergleich

                if p1["is_segment"] and p2["is_segment"]:
                    criterion = "d_min"
                    distance = float(np.min(trees[i].query(p2["points"], k=1, workers=-1)[0]))
                    merge = distance <= delta
                    decision = str(merge)
                elif p1["is_segment"] or p2["is_segment"]:
                    criterion = "d_min (nur nächste kompakte Region)"
                    distance = float(np.min(trees[i].query(p2["points"], k=1, workers=-1)[0]))
                    merge = False # Entscheidung erst nach allen Vergleichen
                    decision = "Kandidat" if distance <= delta else "False"
                    if distance <= delta:
                        seg, comp = (i, j) if p1["is_segment"] else (j, i)
                        if seg not in nearest_compact or distance < nearest_compact[seg][0]:
                            nearest_compact[seg] = (distance, comp)
                else:
                    criterion = "d_H"
                    distance = self._hausdorff_distance(p1["points"], p2["points"])
                    merge = distance <= delta
                    decision = str(merge)

                # #Debug: Teilstück-IDs machen nachvollziehbar, welche Ausschnitte verglichen wurden.
                if delta < 1.0:
                    self.log(
                        f"#Debug Schritt 2 Vergleich: {p1['part_id']} ({p1['guid_a']}|{p1['guid_b']}) "
                        f"<-> {p2['part_id']} ({p2['guid_a']}|{p2['guid_b']}); "
                        f"Kriterium={criterion}; Abstand={distance:.6f} m; "
                        f"t1={thickness[i]:.6f} m; t2={thickness[j]:.6f} m; "
                        f"delta={delta:.6f} m; zusammenfuehren={decision}",
                        print_console=False
                    )
                if merge:
                    G.add_edge(i, j)
        # Jedes Teilstück wird nur mit seiner nächsten kompakten Region verbunden
        for seg, (distance, comp) in sorted(nearest_compact.items()):
            G.add_edge(seg, comp)
            self.log(
                f"#Debug Schritt 2.2 Verbindung Teilstück {parts[seg]['part_id']} -> nächste kompakte Region "
                f"{parts[comp]['part_id']}; d_min={distance:.6f} m",
                print_console=False
            )
        self.log("", print_console=False) # Leerzeile nach dem letzten Vergleich
        self.log(
            f"#Debug Schritt 2.2: {n * (n - 1) // 2} mögliche Paare; mit AABB-Abstand <= delta geprüft: {checked}; "
            f"Kanten: {G.number_of_edges()}",
            print_console=False
        )

        # Transitive Hülle: Zusammenhangskomponenten (abgeschlossene Inseln)
        # bilden die Joint-Cluster, bei denen man innerhalb eines Clusters 
        # von jedem Punkt aus jeden anderen darin erreichen kann.
        # 2.3 Zusammenhangskomponenten auf den Teilstücken
        components = sorted(
            (sorted(component) for component in nx.connected_components(G)),
            key=lambda component: component[0]
        )

        # 2.4 Komponenten bewerten: echte Joints vs. alleinstehende Teilstücke einer Region
        joint_components = []
        lone_components = []
        for component in components:
            comp_parts = [parts[k] for k in component]
            region_ids = {part["region_id"] for part in comp_parts}
            if len(region_ids) >= 2 or any(not part["is_segment"] for part in comp_parts):
                joint_components.append(comp_parts)
            else:
                lone_components.append(comp_parts)
        regions_in_joints = {part["region_id"] for comp_parts in joint_components for part in comp_parts}

        self.log("", print_console=False) # Leerzeile vor der Bewertung alleinstehender Teilstücke
        lone_regions = []
        for comp_parts in lone_components:
            region_id = comp_parts[0]["region_id"]
            if region_id in regions_in_joints:
                self.log(
                    f"#Debug Schritt 2.4 Teilstück verworfen (Region {region_id} ist bereits in einem Joint): "
                    f"part_ids={[part['part_id'] for part in comp_parts]}",
                    print_console=False
                )
            elif region_id not in lone_regions:
                lone_regions.append(region_id)
        for region_id in lone_regions:
            # Region ohne Partner in der Nähe: bildet als Ganzes einen eigenen Joint
            joint_components.append([part for part in parts if part["region_id"] == region_id])
            self.log(f"#Debug Schritt 2.4 Region {region_id} ohne Partner: eigener Joint", print_console=False)
        self.log("", print_console=False) # Leerzeile nach der Bewertung

        joint_components.sort(key=lambda comp_parts: min(part["part_id"] for part in comp_parts))
        # #Debug: Explizite Zuordnung der Teilstück-IDs zu jedem resultierenden Cluster.
        for joint_index, comp_parts in enumerate(joint_components, start=1):
            self.log("", print_console=False) # Leerzeile: Sprung zum nächsten Cluster
            member_guids = sorted({guid for part in comp_parts for guid in (part["guid_a"], part["guid_b"])})
            self.log(
                f"#Debug Schritt 2 Cluster J{joint_index:04d}: "
                f"contact_regions={sorted({part['region_id'] for part in comp_parts})}; "
                f"part_ids={[part['part_id'] for part in comp_parts]}; member_guids={member_guids}",
                print_console=False
            )
        self.log("", print_console=False) # Leerzeile nach dem letzten Cluster
        return joint_components


    def _compute_contact_regions(self, nodes, gt_pairs):
        """
        Schritt 1 der Joint-Rekonstruktion: Bestimmt die Kontaktregion R_ab
        für jedes Ground-Truth-Paar (a, b) und klassifiziert sie (0D bis 3D).

        Prüfreihenfolge je Paar (vom billigsten zum teuersten Test):
            Stufe 1: AABB-Vorfilter. Sind die Boxen um mehr als eps getrennt -> kein Kontakt.
            Stufe 2: Volumentest, nur wenn die AABB-Überlappung in allen drei Achsen > eps ist
                     (sonst kann keine Durchdringung > eps existieren). Gitterpunkte der
                     Überlappungsbox, die in beiden Bauteilen liegen, bilden das
                     Schnittvolumen und damit die Kontaktregion (Methode 'Volumen').
            Stufe 3: Oberflächenkontakt, falls kein Schnittvolumen existiert. Abtastpunkte der
                     Dreiecksflächen beider Bauteile mit Abstand <= eps zur Oberfläche des
                     jeweils anderen bilden die Kontaktregion (Methode 'Oberfläche'):
                     R_(a,b) = { x in dS_a, dS_b | d(x, Oberfläche des anderen Bauteils) <= eps }
        Die Klassifikation (0D_Punkt, 1D_Linie, 2D_Fläche, 3D_Volumen) erfolgt per PCA
        (siehe _classify_dimensionality).

        Args:
            nodes (list[dict]): Knotenliste mit 'geom_raw' (Dreiecke, AABB) und 'guid'.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Kontaktregionen mit den Schlüsseln aus CONTACT_REGION_KEYS.
        """
        self.log(f"1.A Schritt 1: Bestimme Kontaktregionen für {len(gt_pairs)} GT-Paare (eps = {JOINT_EPSILON*1000:.1f} mm)...")
        # #Debug Schritt 1: Alle Details landen im Protokoll, nicht in der Konsole.
        self.log(
            f"#Debug Schritt 1 Konfiguration: epsilon={JOINT_EPSILON:.6f} m; tau={JOINT_TAU}; "
            f"point_extent={JOINT_POINT_EXTENT} m; sample_spacing={JOINT_SAMPLE_SPACING} m; "
            f"max_samples={JOINT_MAX_SAMPLES}",
            print_console=False
        )
        self.log(f"#Debug Schritt 1 Schlüssel je Kontaktregion: {list(CONTACT_REGION_KEYS)}", print_console=False)
        nodes_by_guid = {n["guid"]: n for n in nodes}
        contact_regions = []
        missing_pairs = []
        sorted_pairs = sorted(gt_pairs)

        for pair_index, (guid_a, guid_b) in enumerate(sorted_pairs, start=1):
            self.log("", print_console=False) # Leerzeile: Sprung zum nächsten GT-Paar
            node_a = nodes_by_guid.get(guid_a)
            node_b = nodes_by_guid.get(guid_b)
            if not node_a or not node_b:
                missing_pairs.append((guid_a, guid_b, "GUID nicht in den extrahierten Knoten gefunden"))
                self.log(
                    f"#Debug Schritt 1 Paar {pair_index}/{len(sorted_pairs)} ohne Region: "
                    f"GUID_A={guid_a}; GUID_B={guid_b}; Grund=GUID nicht in den extrahierten Knoten gefunden",
                    print_console=False
                )
                continue

            # Stufe 1: AABB-Vorfilter. raw_dims = Überlappung der Boxen je Achse (negativ = Lücke)
            (min_a, max_a) = node_a["geom_raw"]["aabb"]
            (min_b, max_b) = node_b["geom_raw"]["aabb"]
            overlap_min = np.maximum(min_a, min_b)
            overlap_max = np.minimum(max_a, max_b)
            raw_dims = overlap_max - overlap_min
            if np.any(raw_dims < -JOINT_EPSILON):
                missing_pairs.append((guid_a, guid_b, "AABBs um mehr als epsilon getrennt"))
                self.log(
                    f"#Debug Schritt 1 Paar {pair_index}/{len(sorted_pairs)} ohne Region: "
                    f"GUID_A={guid_a}; GUID_B={guid_b}; Grund=AABBs um mehr als epsilon getrennt; "
                    f"aabb_overlap_m={np.round(raw_dims, 6).tolist()}",
                    print_console=False
                )
                continue

            tri_a = node_a["geom_raw"]["triangles"]
            tri_b = node_b["geom_raw"]["triangles"]
            contact_points = None
            method = ""
            volume = 0.0
            d_min = float('nan')

            # Stufe 2: Schnittvolumen (nur möglich, wenn die Boxen sich in allen Achsen > eps überlappen)
            if np.all(raw_dims > JOINT_EPSILON):
                contact_points, volume = self._contact_from_volume(tri_a, tri_b, overlap_min, overlap_max)
                if contact_points is not None:
                    method = "Volumen"

            # Stufe 3: Oberflächenkontakt, falls kein Schnittvolumen gefunden wurde
            if contact_points is None:
                volume = 0.0
                contact_points, d_min = self._contact_from_surfaces(
                    tri_a, tri_b, overlap_min - JOINT_EPSILON, overlap_max + JOINT_EPSILON
                )
                if contact_points is not None:
                    method = "Oberfläche"

            if contact_points is None:
                missing_pairs.append((guid_a, guid_b, "kein Punkt innerhalb epsilon"))
                self.log(
                    f"#Debug Schritt 1 Paar {pair_index}/{len(sorted_pairs)} ohne Region: "
                    f"GUID_A={guid_a}; GUID_B={guid_b}; Grund=kein Punkt innerhalb epsilon "
                    f"({JOINT_EPSILON:.6f} m); kleinster_Oberflaechenabstand_m={d_min:.6f}",
                    print_console=False
                )
                continue

            # Charakterisierung und Klassifikation der Kontaktregion
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
                "contact_type": contact_type,
                "contact_method": method,
                "volume": volume
            }
            contact_regions.append(region)

            # #Debug Schritt 1: Die Schlüssel stehen einmalig oben im Protokoll; hier nur Name=Wert
            self.log(
                f"#Debug Schritt 1 Kontaktregion {region['region_id']} (Paar {pair_index}/{len(sorted_pairs)}): "
                f"GUID_A={guid_a}; GUID_B={guid_b}; centroid={np.round(centroid, 6).tolist()}; "
                f"eigenvalues={np.round(eigenvalues, 8).tolist()}; "
                f"lengths_m={np.round(L, 6).tolist()}; "
                f"contact_method={method}; contact_type={contact_type}; "
                f"volume_m3={volume:.6f}; point_count={len(contact_points)}",
                # f"points_m={np.round(contact_points, 6).tolist()}",
                print_console=False
            )
        self.log("", print_console=False) # Leerzeile nach dem letzten GT-Paar

        type_counts = {}
        for region in contact_regions:
            type_counts[region["contact_type"]] = type_counts.get(region["contact_type"], 0) + 1
        self.log(f"1.A Schritt 1: {len(contact_regions)} Kontaktregionen gefunden.")
        self.log(
            f"#Debug Schritt 1 Zusammenfassung: GT-Paare={len(gt_pairs)}; "
            f"Kontaktregionen={len(contact_regions)}; ohne Region={len(missing_pairs)}; "
            f"Verteilung_contact_type={type_counts}; fehlende_Paare={missing_pairs}",
            print_console=False
        )
        return contact_regions


    def _contact_from_surfaces(self, tri_a, tri_b, box_min, box_max):
        """
        Oberflächenkontakt (Stufe 3): Tastet die Dreiecksflächen beider Bauteile innerhalb
        der Box ab und behält alle Punkte, deren exakter Abstand zur Oberfläche des jeweils
        anderen Bauteils <= eps ist.
        Hinweis: Die 'test_points' aus dem Dateneingang tasten nur Dreieckskanten ab und
        reichen für Flächenkontakte nicht aus; daher werden die Dreiecksflächen hier selbst abgetastet.

        Args:
            tri_a (np.ndarray): Dreiecke von Bauteil a, Shape (n, 3, 3).
            tri_b (np.ndarray): Dreiecke von Bauteil b, Shape (m, 3, 3).
            box_min (np.ndarray): Untere Ecke der um eps aufgeweiteten Überlappungsbox.
            box_max (np.ndarray): Obere Ecke der um eps aufgeweiteten Überlappungsbox.

        Returns:
            tuple: (points, d_min)
                points (np.ndarray | None): Kontaktpunkte (Vereinigung beider Seiten) oder None.
                d_min (float): Kleinster gefundener Oberflächenabstand in [m] (Debug).
        """
        # Abtastquellen: Dreiecke in der Box; Abstandsziele: Dreiecke bis eps um die Box
        source_a = self._select_triangles_in_box(tri_a, box_min, box_max)
        source_b = self._select_triangles_in_box(tri_b, box_min, box_max)
        target_a = self._select_triangles_in_box(tri_a, box_min - JOINT_EPSILON, box_max + JOINT_EPSILON)
        target_b = self._select_triangles_in_box(tri_b, box_min - JOINT_EPSILON, box_max + JOINT_EPSILON)
        if len(source_a) == 0 or len(source_b) == 0:
            return None, float('inf')

        samples_a = self._sample_triangles(source_a, box_min, box_max)
        samples_b = self._sample_triangles(source_b, box_min, box_max)
        d_a = self._point_triangle_distance(samples_a, target_b) if len(samples_a) else np.empty(0)
        d_b = self._point_triangle_distance(samples_b, target_a) if len(samples_b) else np.empty(0)

        d_min = min(d_a.min() if len(d_a) else float('inf'), d_b.min() if len(d_b) else float('inf'))
        contact = np.vstack([samples_a[d_a <= JOINT_EPSILON], samples_b[d_b <= JOINT_EPSILON]])
        return (contact if len(contact) else None), float(d_min)


    def _contact_from_volume(self, tri_a, tri_b, box_min, box_max):
        """
        Schnittvolumen (Stufe 2): Legt ein Gitter in die Überlappungsbox der AABBs und behält
        alle Gitterpunkte, die in beiden Bauteilen liegen (Strahl-Paritaet, siehe _inside_mesh).
        Zuerst wird gegen Bauteil a getestet; nur die verbleibenden Punkte gegen Bauteil b.
        Das Volumen ergibt sich aus Punktanzahl x Zellvolumen.

        Args:
            tri_a (np.ndarray): Dreiecke von Bauteil a, Shape (n, 3, 3).
            tri_b (np.ndarray): Dreiecke von Bauteil b, Shape (m, 3, 3).
            box_min (np.ndarray): Untere Ecke der AABB-Überlappung.
            box_max (np.ndarray): Obere Ecke der AABB-Überlappung.

        Returns:
            tuple: (points, volume)
                points (np.ndarray | None): Punkte des Schnittvolumens oder None.
                volume (float): Geschätztes Schnittvolumen in [m^3].
        """
        dims = box_max - box_min
        # Zellgröße h: so, dass das Gitter etwa JOINT_MAX_SAMPLES Punkte hat, nie feiner als eps
        h = max(JOINT_EPSILON, (float(np.prod(dims)) / JOINT_MAX_SAMPLES) ** (1.0 / 3.0))
        counts = np.maximum(1, np.ceil(dims / h).astype(int))
        grid_axes = [box_min[i] + (np.arange(counts[i]) + 0.5) * dims[i] / counts[i] for i in range(3)]
        grid = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)
        cell_volume = float(np.prod(dims / counts))

        # Strahl in +z: Es werden alle Dreiecke mit Überlappung in x/y benötigt, unabhängig von z
        near_a = self._select_triangles_in_box(tri_a, box_min, box_max, axes=(0, 1))
        grid = grid[self._inside_mesh(grid, near_a)]
        if len(grid) == 0:
            return None, 0.0
        near_b = self._select_triangles_in_box(tri_b, box_min, box_max, axes=(0, 1))
        grid = grid[self._inside_mesh(grid, near_b)]
        if len(grid) == 0:
            return None, 0.0
        return grid, len(grid) * cell_volume


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
        Berechnet die symmetrische (diskrete) Hausdorff-Distanz zwischen zwei Punktmengen A und B.

        Vorgehen:
            1. Gerichtete Distanz A -> B: Für jeden Punkt a aus A wird der Abstand zum nächstgelegenen
               Punkt aus B bestimmt (Minimum über alle b). Von diesen |A| Werten zählt der größte:
                   h(A,B) = max_{a in A} min_{b in B} ||a-b||
            2. Gerichtete Distanz B -> A analog: h(B,A) = max_{b in B} min_{a in A} ||a-b||
            3. Die Hausdorff-Distanz ist der größere der beiden Werte:
                   d_H(A,B) = max( h(A,B), h(B,A) )
        Bedeutung: d_H ist die kleinste Entfernung r, bei der jede Menge vollständig in der
        r-Umgebung der anderen liegt. Sie ist nur klein, wenn sich beide Mengen über ihre gesamte
        Ausdehnung decken. Eine lange und eine kurze Region haben daher auch bei direkter
        Berührung ein großes d_H, weil h(lang, kurz) mindestens die halbe Länge beträgt.
        Zu unterscheiden vom Mindestabstand d_min = min_a min_b ||a-b||, der schon bei einer
        einzelnen Berührungsstelle klein ist.
        Die Mengen sind Abtastpunkte der Kontaktregionen; die Distanz ist deshalb diskret
        (Abweichung höchstens um den Abtastabstand). Die nächsten Nachbarn liefert ein cKDTree.

        Args:
            pts_a (np.ndarray): Punktmenge A, Shape (n, 3).
            pts_b (np.ndarray): Punktmenge B, Shape (m, 3).

        Returns:
            float: Hausdorff-Distanz in [m].
        """
        if len(pts_a) == 0 or len(pts_b) == 0:
            return float('inf')
        # Hausdorff verwendet je Richtung den grösßten Abstand zum nächsten
        # Gegenpunkt. Das Minimum würde nur irgendeine lokale Nähe prüfen und
        # lange Kontaktregionen fälschlich mit punktuellen Regionen verbinden.
        d_a_to_b = np.max(cKDTree(pts_b).query(pts_a, k=1, workers=-1)[0])
        d_b_to_a = np.max(cKDTree(pts_a).query(pts_b, k=1, workers=-1)[0])
        return max(d_a_to_b, d_b_to_a)


    def _inside_mesh(self, points, triangles):
        """
        Punkt-in-Netz-Test per Strahl-Parität: Ein Strahl in +z-Richtung schneidet die Dreiecke;
        eine ungerade Anzahl Schnitte oberhalb des Punkts bedeutet 'innen'. Voraussetzung ist ein
        geschlossenes Netz. Die Punkte werden in x/y um ~1e-7 m verschoben, damit der Strahl keine
        Dreieckskanten exakt trifft (sonst würde die Paritaet kippen).

        Args:
            points (np.ndarray): Testpunkte, Shape (n, 3).
            triangles (np.ndarray): Dreiecke des Bauteils, Shape (m, 3, 3).

        Returns:
            np.ndarray: Boolesche Maske (n,), True = Punkt liegt im Bauteil.
        """
        inside = np.zeros(len(points), dtype=bool)
        if len(triangles) == 0 or len(points) == 0:
            return inside
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        denom = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        valid = np.abs(denom) > 1e-12 # senkrechte Dreiecke haben keine x/y-Fläche
        safe = np.where(valid, denom, 1.0)
        chunk = max(1, 500_000 // len(triangles))
        for start in range(0, len(points), chunk):
            part = points[start:start + chunk]
            px = (part[:, 0] + 1.2345e-7)[:, None]
            py = (part[:, 1] + 2.7183e-7)[:, None]
            pz = part[:, 2][:, None]
            # Baryzentrische Koordinaten l1, l2, l3 des Punkts in der x/y-Projektion des Dreiecks
            l1 = ((b[:, 1] - c[:, 1]) * (px - c[:, 0]) + (c[:, 0] - b[:, 0]) * (py - c[:, 1])) / safe
            l2 = ((c[:, 1] - a[:, 1]) * (px - c[:, 0]) + (a[:, 0] - c[:, 0]) * (py - c[:, 1])) / safe
            l3 = 1.0 - l1 - l2
            hit = valid & (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            z_hit = l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2]
            crossings = np.sum(hit & (z_hit > pz), axis=1)
            inside[start:start + chunk] = (crossings % 2) == 1
        return inside


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


    def _point_triangle_distance(self, points, triangles):
        """
        Exakter Abstand jedes Punkts zur Oberfläche einer Dreiecksmenge.
        Liegt die senkrechte Projektion im Dreieck, ist der Abstand |Ebenenabstand|, sonst
        der Abstand zur nächsten der drei Kanten (Strecken). Ergebnis ist das Minimum über alle Dreiecke.

        Args:
            points (np.ndarray): Punkte, Shape (n, 3).
            triangles (np.ndarray): Dreiecke, Shape (m, 3, 3).

        Returns:
            np.ndarray: Kleinster Abstand je Punkt, Shape (n,).
        """
        if len(triangles) == 0:
            return np.full(len(points), np.inf)
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = b - a, c - b, a - c
        normal = np.cross(ab, -ca)
        length = np.linalg.norm(normal, axis=1)
        valid = length > 1e-12 # entartete Dreiecke nur über ihre Kanten
        unit = np.where(valid[:, None], normal / np.maximum(length, 1e-12)[:, None], 0.0)

        def segment_distance(p, start, vec):
            t = np.clip(np.sum((p - start) * vec, axis=2) / np.maximum(np.sum(vec * vec, axis=1), 1e-18), 0.0, 1.0)
            return np.linalg.norm(p - (start + t[..., None] * vec), axis=2)

        result = np.empty(len(points))
        chunk = max(1, 300_000 // len(triangles))
        for start_idx in range(0, len(points), chunk):
            p = points[start_idx:start_idx + chunk][:, None, :]
            edge_dist = np.minimum(np.minimum(segment_distance(p, a, ab), segment_distance(p, b, bc)),
                                   segment_distance(p, c, ca))
            height = np.sum((p - a) * unit, axis=2) # vorzeichenbehafteter Ebenenabstand
            q = p - height[..., None] * unit        # Projektion auf die Dreiecksebene
            inside = (valid
                      & (np.sum(np.cross(ab, q - a) * unit, axis=2) >= -1e-12)
                      & (np.sum(np.cross(bc, q - b) * unit, axis=2) >= -1e-12)
                      & (np.sum(np.cross(ca, q - c) * unit, axis=2) >= -1e-12))
            result[start_idx:start_idx + chunk] = np.where(inside, np.abs(height), edge_dist).min(axis=1)
        return result


    def _sample_triangles(self, triangles, box_min, box_max):
        """
        Tastet Dreiecksflächen mit einem baryzentrischen Gitter ab (Abstand JOINT_SAMPLE_SPACING)
        und behält nur Punkte innerhalb der Box. Alle drei Dreieckskanten sind enthalten.
        Sehr große Dreiecke werden je Dreieck auf max. 250000 Gitterpunkte begrenzt; über
        JOINT_MAX_SAMPLES hinaus wird gleichmäßig ausgedünnt.

        Args:
            triangles (np.ndarray): Dreiecke, Shape (n, 3, 3).
            box_min (np.ndarray): Untere Ecke der Box.
            box_max (np.ndarray): Obere Ecke der Box.

        Returns:
            np.ndarray: Abtastpunkte, Shape (k, 3).
        """
        h = JOINT_SAMPLE_SPACING
        chunks = [np.empty((0, 3))]
        for a, b, c in triangles:
            ab, ac, bc = b - a, c - a, c - b
            n1 = max(1, int(np.ceil(np.linalg.norm(ab) / h)))
            n2 = max(1, int(np.ceil(np.linalg.norm(ac) / h)))
            n3 = max(1, int(np.ceil(np.linalg.norm(bc) / h)))
            if n1 * n2 > 250_000:
                scale = np.sqrt(250_000 / (n1 * n2))
                n1, n2 = max(1, int(n1 * scale)), max(1, int(n2 * scale))
            u, v = np.meshgrid(np.arange(n1 + 1) / n1, np.arange(n2 + 1) / n2)
            keep = (u + v) <= 1.0 + 1e-9 # Gitterpunkte außerhalb des Dreiecks verwerfen
            chunks.append(a + u[keep, None] * ab + v[keep, None] * ac)
            chunks.append(b + (np.arange(n3 + 1) / n3)[:, None] * bc) # dritte Kante b-c
        points = np.vstack(chunks)
        points = points[np.all((points >= box_min) & (points <= box_max), axis=1)]
        if len(points) > JOINT_MAX_SAMPLES:
            points = points[::int(np.ceil(len(points) / JOINT_MAX_SAMPLES))]
        return points


    def _select_triangles_in_box(self, triangles, box_min, box_max, axes=(0, 1, 2)):
        """
        Wählt die Dreiecke, deren Bounding-Box die Box in den angegebenen Achsen überlappt.

        Args:
            triangles (np.ndarray): Dreiecke, Shape (n, 3, 3).
            box_min (np.ndarray): Untere Ecke der Box.
            box_max (np.ndarray): Obere Ecke der Box.
            axes (tuple): Geprüfte Achsen (0=x, 1=y, 2=z).

        Returns:
            np.ndarray: Ausgewählte Dreiecke, Shape (k, 3, 3).
        """
        idx = list(axes)
        tri_min = triangles.min(axis=1)[:, idx]
        tri_max = triangles.max(axis=1)[:, idx]
        keep = np.all((tri_max >= box_min[idx]) & (tri_min <= box_max[idx]), axis=1)
        return triangles[keep]


    def _split_region(self, region):
        """
        Zerlegt eine ausgedehnte Kontaktregion in Teilstücke (Zellen) der Kantenlänge
        <= JOINT_SEGMENT_LENGTH. Die Zellen liegen im Hauptachsensystem der Region (PCA); nur
        Achsen mit L_i > JOINT_SEGMENT_LENGTH werden geteilt. Kompakte Regionen bleiben ein Stück.

        Args:
            region (dict): Kontaktregion aus Schritt 1 ('points', 'centroid', 'lengths', ...).

        Returns:
            list[dict]: Teilstücke mit 'part_id', 'region_id', 'guid_a', 'guid_b', 'points',
                'centroid' und 'is_segment' (False bei unzerlegter Region).
        """
        points = region["points"]
        lengths = np.asarray(region["lengths"], dtype=float)
        base = {"region_id": region["region_id"], "guid_a": region["guid_a"], "guid_b": region["guid_b"]}
        if lengths[0] <= JOINT_SEGMENT_LENGTH:
            return [{**base, "part_id": region["region_id"], "points": points,
                     "centroid": region["centroid"], "is_segment": False}]

        _, axes, _ = self._pca_contact_region(points)
        coords = (points - region["centroid"]) @ axes # Koordinaten im Hauptachsensystem
        n_cells = np.maximum(1, np.ceil(lengths / JOINT_SEGMENT_LENGTH).astype(int))
        cell_size = np.where(lengths > 0, lengths / n_cells, 1.0)
        cell_idx = np.minimum(((coords - coords.min(axis=0)) / cell_size).astype(int), n_cells - 1)
        _, inverse = np.unique(cell_idx, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)

        parts = []
        for k in range(int(inverse.max()) + 1):
            sub = points[inverse == k]
            parts.append({**base, "part_id": f"{region['region_id']}.{k + 1:02d}", "points": sub,
                          "centroid": sub.mean(axis=0), "is_segment": True})
        return parts


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
        joints = self._attribute_joints(components, gt_pairs)

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

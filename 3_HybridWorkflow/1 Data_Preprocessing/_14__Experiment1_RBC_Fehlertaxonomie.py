"""
_14_Experiment1_RBC_Fehlertaxonomie.py
======================================
Experiment 1: Ermittlung und Dokumentation von Kontaktregionen
für Ground-Truth-Kontaktpaare.

Das Modul wird von '_14 IFC_Graph_Generator.py' eingebunden und stellt die
Klasse 'Experiment1RBCFehlertaxonomie' bereit (Mixin der Klasse GNNPipeline).

Joints sind Hyperkanten. Sie gehören nicht direkt zum Kontaktgraphen,
beziehen sich aber auf dessen Bestandteile.

Struktur (Ist-Stand):
1. KONFIGURATION (Experiment 1)
2. KLASSE Experiment1RBCFehlertaxonomie
   1.A Ermittlung von Kontaktregionen (IFC-/GT-basiert)
   
Struktur (Soll-Ziel):
1. KONFIGURATION (Experiment 1)
2. KLASSE Experiment1RBCFehlertaxonomie
   1.A Rekonstruktion von Anschlussstellen (Soll-Joints, IFC-/GT-basiert)
   1.B Klassifizierungssystem für Anschlussstellen
   1.C Extraktion von Anschlussstellen aus dem Analysemodell (Ist-Joints)
   1.D Validierung: Matching, Fehlercodes (E0-E7), Statistik
"""
from typing import Callable

import numpy as np

# =========================================================
# KONFIGURATION (Experiment 1, Schritt 1)
# =========================================================
# 1.A Kontaktregionen
JOINT_EPSILON = 0.05  # [m] Kontakttoleranz eps in [m]. HINWEIS: Sensitivitätsstudie. Default = 0.05
JOINT_TAU = 0.2        # [-] Schwellenwert tau für Dimensionalitätsverhältnisse L2/L1 und L3/L1 (linien-/flächen-/punktförmig). Default = 0.2
JOINT_DELTA_SCALE = 1.05 # [-] Skalierung der Joint-Toleranz delta mit der größten Bauteildicke max(t_a, t_b). Default = 1.05 wegen numerischer Ungeauigkeit

JOINT_POINT_EXTENT = 0.05 # [m] Maximale Ausdehnung L1, bis zu der eine Kontaktregion als 0D_Punkt gilt. Muss >= eps/tau sein, damit dünne Oberflächenkontakte nie als 3D_Volumen enden. Default = 0.05
JOINT_SAMPLE_SPACING = 0.025 # [m] Abtastabstand der Dreiecksflächen (Oberflächenkontakt). Default = 0.025
JOINT_MAX_SAMPLES = 5000 # [-] Obergrenze der Abtastpunkte je Bauteil und Paar bzw. der Gitterpunkte im Volumentest. Default = 5000

LINEAR_ELEMENT_TYPES = {"IfcBeam", "IfcColumn", "IfcMember", "IfcPile"}
PLANAR_ELEMENT_TYPES = {"IfcSlab", "IfcWall", "IfcWallStandardCase", "IfcPlate"}

# Schlüssel einer Kontaktregion (Reihenfolge wie im Dictionary in _compute_contact_regions)
CONTACT_REGION_KEYS = (
    "region_id", "pair_id", "guid_a", "guid_b", "points", "centroid", "eigenvalues",
    "lengths", "contact_type", "contact_method", "distance_m", "aabb_distance_m"
)

# =========================================================
# EXPERIMENT 1, SCHRITT 1
# =========================================================
class Experiment1RBCFehlertaxonomie:
    """
    Ermittelt und dokumentiert Kontaktregionen für Ground-Truth-Paare.

    Erwartet von der Hauptklasse (GNNPipeline): die Methode log().
    """
    log: Callable[..., None]

    # ---------------------------------------------------------
    # 1.A KONTAKTREGIONEN
    #     Grundlage: Ground Truth + IFC-Geometrie
    # ---------------------------------------------------------
    # Methodenreihenfolge: interne Hilfsfunktionen (_) alphabetisch,
    # danach Orchestrierung, danach weitere Unterfunktionen.
    # ---------------------------------------------------------
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
        if r31 >= JOINT_TAU: # Bauteile durchdringen sich, z. B. Stütze durch Decke
            return "3D_Volumen"
        if r21 >= JOINT_TAU: # z. B. Stütze auf Decke, Decke auf Decke vollflächig
            return "2D_Fläche"
        return "1D_Linie" # z.B. Wand auf Decke, Wand-Wand


    def _compute_contact_regions(self, nodes, gt_pairs):
        """
        Schritt 1 der Joint-Rekonstruktion: Bestimmt die Kontaktregion R_ab
        für jedes Ground-Truth-Paar (a, b) und klassifiziert sie (0D bis 3D).

        Paare mit mehr als eps AABB-Abstand werden verworfen. Für alle übrigen
        Paare bildet die AABB-Nähe die Standard-Kontaktregion. Ein Treffer der
        genaueren Oberflächenprüfung innerhalb eps ersetzt diese AABB-Region.
        Beide Abstände werden getrennt gespeichert. Die Klassifikation erfolgt
        per PCA.

        Args:
            nodes (list[dict]): Knotenliste mit 'geom_raw' (Dreiecke, AABB) und 'guid'.
            gt_pairs (set[tuple]): Ground-Truth-Kontaktpaare (sortierte GUID-Tupel).

        Returns:
            list[dict]: Kontaktregionen mit den Schlüsseln aus CONTACT_REGION_KEYS.
        """
        self.log("", print_console=False) # Leerzeile: Sprung zum nächsten GT-Paar
        self.log(f"{'='*10} EXPERIMENT_1 RBC_Fehlertaxonomie {'='*10}")
        self.log(f"1.A Schritt 1: Bestimme Kontaktregionen für {len(gt_pairs)} GT-Paare (eps = {JOINT_EPSILON*1000:.1f} mm)...")
        self.log(
            f"1.A Schritt 1: Konfiguration: epsilon={100 * JOINT_EPSILON:.2f} cm; tau={JOINT_TAU}; "
            f"point_extent={JOINT_POINT_EXTENT} m; sample_spacing={JOINT_SAMPLE_SPACING} m; "
            f"max_samples={JOINT_MAX_SAMPLES}",
        )
        self.log(f"1.A Schritt 1: Schlüssel je Kontaktregion: {list(CONTACT_REGION_KEYS)}")
        nodes_by_guid = {n["guid"]: n for n in nodes}
        contact_regions = []
        missing_pairs = []
        sorted_pairs = sorted(gt_pairs)

        for pair_index, (guid_a, guid_b) in enumerate(sorted_pairs, start=1):
            # #Debug Schritt 1: Alle Details landen im Protokoll, nicht in der Konsole.
            # self.log("", print_console=False) # Leerzeile: Sprung zum nächsten GT-Paar
            
            pair_id = f"P{pair_index:04d}"
            node_a = nodes_by_guid.get(guid_a)
            node_b = nodes_by_guid.get(guid_b)
            if not node_a or not node_b:
                reason = "GUID nicht in den extrahierten Knoten gefunden"
                missing_pairs.append({
                    "pair_id": pair_id, "guid_a": guid_a, "guid_b": guid_b, "reason": reason
                })
                
                # Protokoll - Kontaktregion für Paar gescheitert (kein Bauteil)
                # self.log(
                #     f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
                #     f"(ohne Kontaktregion): pair_id={pair_id}; "
                #     f"GUID_A={guid_a}; GUID_B={guid_b}; Grund={reason}",
                #     print_console=False
                # )
                if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                    self.log(
                        f"{' '*8}1.A Schritt 1: {pair_index}/{len(sorted_pairs)} Kontaktpaare geprüft...",
                        print_console=True
                    )
                continue

            # AABB-Vorprüfung entspricht einer Erweiterung um eps auf einer Seite.
            (min_a, max_a) = node_a["geom_raw"]["aabb"]
            (min_b, max_b) = node_b["geom_raw"]["aabb"]
            overlap_min = np.maximum(min_a, min_b)
            overlap_max = np.minimum(max_a, max_b)
            raw_dims = overlap_max - overlap_min
            axis_gaps = np.maximum(-raw_dims, 0.0)
            
            aabb_distance = float(np.linalg.norm(axis_gaps))
            aabb_overlap = np.round(raw_dims, 3).tolist()
            
            if np.any(axis_gaps > JOINT_EPSILON):
                reason = "AABB-Abstand größer als Epsilon"
                missing_pairs.append({
                    "pair_id": pair_id, "guid_a": guid_a, "guid_b": guid_b, "reason": reason,
                    "aabb_distance": aabb_distance, "aabb_overlap": aabb_overlap
                })
                
                # Protokoll - Kontaktregion für Paar gescheitert (zu weit entfernt)
                # self.log(
                #     f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
                #     f"(ohne Kontaktregion): pair_id={pair_id}; "
                #     f"GUID_A={guid_a}; GUID_B={guid_b}; Grund={reason}; "
                #     f"AABB-Abstand_m={aabb_distance}; "
                #     f"AABB-Überlappung_m={aabb_overlap}",
                #     print_console=False
                # )
                if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                    self.log(
                        f"{' '*8}1.A Schritt 1: {pair_index}/{len(sorted_pairs)} Kontaktpaare geprüft...",
                        print_console=True
                    )
                continue

            tri_a = node_a["geom_raw"]["triangles"]
            tri_b = node_b["geom_raw"]["triangles"]
            # AABB-Näherung als Standardregion; getrennte Achsen werden auf
            # den Mittelpunkt zwischen den nächstliegenden Boxflächen gesetzt.
            region_min = overlap_min.copy()
            region_max = overlap_max.copy()
            separated_axes = raw_dims < 0.0
            midpoints = (overlap_min + overlap_max) * 0.5
            region_min[separated_axes] = midpoints[separated_axes]
            region_max[separated_axes] = midpoints[separated_axes]
            contact_points = np.array([
                [x, y, z]
                for x in (region_min[0], region_max[0])
                for y in (region_min[1], region_max[1])
                for z in (region_min[2], region_max[2])
            ], dtype=float)
            contact_points = np.unique(contact_points, axis=0)
            method = "AABB"

            # Die Mesh-Prüfung kann die AABB-Methode verfeinern, aber keinen
            # bereits akzeptierten AABB-Treffer verwerfen.
            surface_points, surface_distance = self._contact_from_surfaces(
                tri_a, tri_b, overlap_min - JOINT_EPSILON, overlap_max + JOINT_EPSILON
            )
            if surface_points is not None:
                contact_points = surface_points
                method = "Oberfläche"
            else:
                surface_distance = self._minimum_mesh_distance(tri_a, tri_b)
            distance_m = float(surface_distance) if np.isfinite(surface_distance) else None
            distance_text = f"{surface_distance:.3f}" if distance_m is not None else "nicht verfügbar"

            # Charakterisierung und Klassifikation der Kontaktregion
            centroid = np.mean(contact_points, axis=0)
            eigenvalues, _, L = self._pca_contact_region(contact_points)
            contact_type = self._classify_dimensionality(L)

            region = {
                "region_id": f"CR_{len(contact_regions) + 1:04d}",
                "pair_id": f"P{pair_index:04d}",
                "guid_a": guid_a,
                "guid_b": guid_b,
                "points": contact_points,
                "centroid": centroid,
                "eigenvalues": eigenvalues,
                "lengths": L,
                "contact_type": contact_type,
                "contact_method": method,
                "distance_m": distance_m,
                "aabb_distance_m": round(aabb_distance,6)
            }
            contact_regions.append(region)

            # Protokoll - Kontaktregion für Paar erfolgreich
            # self.log(
            #     f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
            #     f"(Kontaktregion {region['region_id']}): pair_id={pair_id}; "
            #     f"GUID_A={guid_a}; GUID_B={guid_b}; points={len(contact_points)}; "
            #     f"centroid={np.round(centroid, 2).tolist()}; "
            #     f"eigenvalues={np.round(eigenvalues, 3).tolist()}; "
            #     f"lengths_m={np.round(L, 2).tolist()}; "
            #     f"contact_method={method}; contact_type={contact_type}; "
            #     f"surface_distance_m={distance_text}; "
            #     f"aabb_distance_m={aabb_distance:.3f}",
            #     print_console=False
            # )
            if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                self.log(
                    f"{' '*8}1.A Schritt 1: {pair_index}/{len(sorted_pairs)} Kontaktpaare geprüft...",
                    print_console=True
                )

        type_counts = {}
        for region in contact_regions:
            type_counts[region["contact_type"]] = type_counts.get(region["contact_type"], 0) + 1
        # Protokoll - Zusammenfassung Schritt 1
        self.log("", print_console=False) # Leerzeile vor Bericht
        self.log(
            f"1.A Schritt 1: Zusammenfassung: "
            f"\nGT-Paare={len(gt_pairs)} | "
            f"Kontaktregionen={len(contact_regions)} | ohne Region={len(missing_pairs)} | "
            f"Verteilung_contact_type={type_counts} | "
            f"fehlende_Paar_IDs={[missing['pair_id'] for missing in missing_pairs]}",
            print_console=False
        )
        
        # Protokoll - Tabelle - Paare ohne Kontaktregion
        self.log("", print_console=False) # Leerzeile vor Tabelle
        self.log(
            "1.A Schritt 1: Tabelle - Paare ohne Kontaktregion:",
            print_console=False
        )
        self.log(
            "1.A Schritt 1: | pair_id | GUID_A | GUID_B | Grund | aabb_distance | aabb_overlap |",
            print_console=False
        )
        for missing in missing_pairs:
            # Pflichtfelder
            parts = [
                "1.A Schritt 1:",
                missing['pair_id'],
                missing['guid_a'],
                missing['guid_b'],
                missing['reason'],
            ]
            
            # Optionale Felder bei zu großer Distanz
            for key in ("aabb_distance_m", "aabb_overlap_m"):
                if key in missing:
                    parts.append(f"{missing[key]}")
            
            self.log(" | ".join(map(str, parts)) + " |", print_console=False)            
        
        # Protokoll - Tabelle - Paare mit Kontaktregion
        self.log("", print_console=False) # Leerzeile vor Tabelle
        self.log(
            "1.A Schritt 1: Tabelle - Paare mit Kontaktregion:",
            print_console=False
        )
        self.log(
            f"1.A Schritt 1: | pair_id | region_id | GUID_A | GUID_B | "
            f"centroid | eigenvalues | lengths | surface_distance_m | aabb_distance_m |",
            print_console=False
        )
        for region in contact_regions:
            parts = [
                "1.A Schritt 1:",
                region['pair_id'],
                region['region_id'],
                region['guid_a'],
                region['guid_b'],
                # f"{len(region['contact_points'])}",
                np.round(region['centroid'], 2).tolist(),
                np.round(region['eigenvalues'], 3).tolist(),
                np.round(region['lengths'], 2).tolist(),
                # region['contact_method'],
                # region['contact_type'],
                f"{region['distance_m']:.3f}",
                f"{region['aabb_distance_m']:.3f}",
            ]
            
            self.log(" | ".join(map(str, parts)) + " |", print_console=False)
        return contact_regions


    def _minimum_point_cloud_distance(self, points_a, points_b):
        """Berechnet den kleinsten euklidischen Abstand zweier Kontaktpunktwolken.

        Die Berechnung verwendet Punktblöcke, damit nicht die vollständige
        Distanzmatrix beider Punktwolken gleichzeitig im Speicher liegt.
        Zeilen mit ungültigen Koordinaten werden ignoriert. Ist danach eine
        Punktwolke leer, wird unendlicher Abstand zurückgegeben.

        Args:
            points_a (array-like): Erste Punktwolke mit drei Koordinaten je Punkt.
            points_b (array-like): Zweite Punktwolke mit drei Koordinaten je Punkt.

        Returns:
            float: Kürzester Punkt-zu-Punkt-Abstand in den Einheiten der Koordinaten.
        """
        points_a = np.asarray(points_a, dtype=float).reshape(-1, 3)
        points_b = np.asarray(points_b, dtype=float).reshape(-1, 3)
        points_a = points_a[np.all(np.isfinite(points_a), axis=1)]
        points_b = points_b[np.all(np.isfinite(points_b), axis=1)]
        if not len(points_a) or not len(points_b):
            return float("inf")

        min_squared = float("inf")
        block_size = 256
        for start_a in range(0, len(points_a), block_size):
            block_a = points_a[start_a:start_a + block_size]
            norms_a = np.einsum("ij,ij->i", block_a, block_a)[:, None]
            for start_b in range(0, len(points_b), block_size):
                block_b = points_b[start_b:start_b + block_size]
                norms_b = np.einsum("ij,ij->i", block_b, block_b)[None, :]
                squared = norms_a + norms_b - 2.0 * block_a @ block_b.T
                min_squared = min(min_squared, float(np.min(np.maximum(squared, 0.0))))
        return float(np.sqrt(min_squared))


    def _element_category_and_thickness(self, node):
        """Ordnet unterstützte IFC-Bauteile ein und bestimmt ihre Dicke.

        Für lineare Bauteile wird nach absteigender Sortierung der Abmessungen
        die mittlere Abmessung als maximaler Querschnitt verwendet. Für
        planare Bauteile wird die kleinste Abmessung als Dicke verwendet.
        Unbekannte IFC-Typen und ungültige Abmessungen werden nicht geschätzt.

        Args:
            node (dict): Extrahierter IFC-Knoten mit `entity_processed` und
                drei Bauteilabmessungen in Metern.

        Returns:
            tuple: Kategorie (`linear` oder `planar`), Dicke in Metern oder
                `None`, und eine Begründung, falls keine Dicke bestimmbar ist.
        """
        entity_type = node.get("entity_processed")
        if entity_type in LINEAR_ELEMENT_TYPES:
            category = "linear"
        elif entity_type in PLANAR_ELEMENT_TYPES:
            category = "planar"
        else:
            return None, None, f"nicht unterstützter oder mehrdeutiger IFC-Typ {entity_type!r}"

        try:
            dimensions = np.asarray(node.get("dimensions"), dtype=float).reshape(3)
        except (TypeError, ValueError):
            return category, None, "Bauteilabmessungen fehlen oder sind ungültig"
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
            return category, None, "Bauteilabmessungen sind nicht endlich oder nicht positiv"

        sorted_dimensions = np.sort(dimensions)[::-1]
        thickness = sorted_dimensions[1] if category == "linear" else sorted_dimensions[2]
        return category, float(thickness), None


    def _build_soll_joints(self, nodes, gt_pairs, contact_regions):
        """
        Schritt 2 der Joint-Rekonstruktion: Rekonstruiert lokale Soll-Joints 
        aus GT-Paaren und Kontaktregionen.

        Die Ground Truth beschränkt die Suche auf zulässige Bauteilpaare.
        Jedes Joint startet bei einer festen Ankerregion. Weitere Paare werden
        nur über bereits enthaltene GUIDs gesucht und nur dann aufgenommen,
        wenn ihre Kontaktpunktwolke nahe an jeder schon enthaltenen Region
        liegt. Abgelehnte Paare werden nicht als Graph-Zwischenstation genutzt.

        Der Abstandsgrenzwert wird vor jeder Kandidatenprüfung aus den
        charakteristischen Dicken der aktuell enthaltenen Bauteile bestimmt.
        Am Ende enthält jedes Joint eindeutige Bauteil-GUIDs sowie die
        punktgewichtete mittlere Position aller enthaltenen Kontaktpunkte.

        Args:
            nodes (list[dict]): IFC-Knoten mit GUID, Typ und Abmessungen.
            gt_pairs (iterable[tuple[str, str]]): Zulässige GT-Paare als
                GUID-Tupel.
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1,
                jeweils mit GUIDs, IDs und Punktwolke.

        Returns:
            list[dict]: Soll-Joints mit Paar-/Regions-IDs, eindeutigen GUIDs,
                Bauteildaten und Position.
        """
        self.log("", print_console=False) # Leerzeile: Sprung zum nächsten Schritt
        self.log(f"1.A Schritt 2: Clustere Kontaktpärchen zu lokalen Soll-Joints...")

        # Indizes ermöglichen die Nachbarsuche über gemeinsame GUIDs, ohne
        # für jedes Joint wieder alle GT-Paare vollständig zu durchsuchen.
        nodes_by_guid = {node["guid"]: node for node in nodes}
        sorted_pairs = sorted(tuple(sorted(pair)) for pair in gt_pairs)
        pair_ids = {pair: f"P{index:04d}" for index, pair in enumerate(sorted_pairs, start=1)}
        pair_by_region = {
            tuple(sorted((region["guid_a"], region["guid_b"]))): region
            for region in contact_regions
        }
        pairs_by_guid = {}
        for pair in sorted_pairs:
            for guid in pair:
                pairs_by_guid.setdefault(guid, []).append(pair)

        # Bauteildicken und Mehrfachzuordnung werden pro Lauf nur einmal bestimmt.
        thickness_cache = {}
        pair_reuse_cache = {}
        assigned_pairs = set()
        assigned_exclusive_pairs = set()
        soll_joints = []

        def category_and_thickness(guid):
            if guid not in thickness_cache:
                node = nodes_by_guid.get(guid)
                if node is None:
                    result = (None, None, "GUID nicht in den extrahierten Knoten gefunden")
                else:
                    result = self._element_category_and_thickness(node)
                thickness_cache[guid] = result
                if result[1] is None:
                    self.log(
                        f"WARNUNG 1.A Schritt 2: Bauteildicke nicht verwendbar; GUID={guid}; "
                        f"Grund={result[2]}", print_console=False
                    )
            return thickness_cache[guid]

        def pair_allows_multiple_joints(pair):
            if pair not in pair_reuse_cache:
                categories = [category_and_thickness(guid)[0] for guid in pair]
                pair_reuse_cache[pair] = (
                    all(category is not None for category in categories)
                    and "planar" in categories
                )
            return pair_reuse_cache[pair]

        def joint_distance_limit(member_guids, joint_id):
            """Ermittelt den aktuellen Grenzwert; unbekannte Dicken verhindern ihn."""
            thicknesses = [
                (guid, category_and_thickness(guid)[1])
                for guid in sorted(member_guids)
            ]
            available = [(guid, value) for guid, value in thicknesses if value is not None]
            missing = [guid for guid, value in thicknesses if value is None]
            if missing:
                self.log(
                    f"WARNUNG 1.A Schritt 2: Abstandsgrenzwert nicht begründbar; "
                    f"joint_id={joint_id}; GUIDs_ohne_Dicke={missing}",
                    print_console=False
                )
                return None
            max_thickness = max(value for _, value in available)
            distance_limit = JOINT_DELTA_SCALE * max_thickness
            # thickness_text = [(guid, round(value, 2)) for guid, value in available]
            # self.log(
            #     f"#Debug Schritt 2 Grenzwert: joint_id={joint_id}; "
            #     f"Dicken_m={thickness_text}; delta_scale={JOINT_DELTA_SCALE}; "
            #     f"d_max_m={distance_limit:.2f}",
            #     print_console=False
            # )
            return distance_limit

        sorted_regions = sorted(
            contact_regions,
            key=lambda region: (region.get("pair_id", ""), region.get("region_id", ""))
        )
        
        for idx,seed_region in enumerate(sorted_regions):
            seed_pair = tuple(sorted((seed_region["guid_a"], seed_region["guid_b"])))
            if seed_pair in assigned_pairs:
                continue

            # Die erste Region ist der unveränderliche räumliche Startanker.
            joint_id = f"J{len(soll_joints) + 1:04d}"
            member_pairs = [seed_pair]
            member_regions = [seed_region]
            member_guids = set(seed_pair)
            frontier = [(seed_pair, 0)]
            checked_pairs = {seed_pair}
            level_counts = {}
            warned_seven = False
            warned_ten = False

            # #Debug Schritt 2: Alle Details landen im Protokoll, nicht in der Konsole.
            self.log("", print_console=False) # Leerzeile: Sprung zum nächsten GT-Paar
            self.log(
                f"#Debug Schritt 2 Joint-Start: joint_id={joint_id}; "
                f"pair_id={seed_region.get('pair_id')}; "
                f"region_id={seed_region.get('region_id')}; GUIDs={list(seed_pair)}",
                print_console=False
            )

            while frontier:
                # Erzeuge die nächste Ebene ausschließlich aus aufgenommenen
                # Paaren; abgelehnte Kandidaten gelangen nie in diese Frontier.
                candidate_levels = {}
                for current_pair, current_level in frontier:
                    candidate_level = current_level + 1
                    for guid in current_pair:
                        for candidate_pair in pairs_by_guid.get(guid, []):
                            if candidate_pair not in checked_pairs:
                                candidate_levels[candidate_pair] = min(
                                    candidate_level,
                                    candidate_levels.get(candidate_pair, candidate_level)
                                )
                                checked_pairs.add(candidate_pair)

                next_frontier = []
                for candidate_pair in sorted(
                    candidate_levels,
                    key=lambda pair: (pair_ids.get(pair, ""), pair)
                ):
                    level = candidate_levels[candidate_pair]
                    counts = level_counts.setdefault(level, {"checked": 0, "accepted": 0, "pairs": []})
                    counts["checked"] += 1
                    pair_id = pair_ids.get(candidate_pair, "unbekannt")
                    candidate_region = pair_by_region.get(candidate_pair)
                    # #Debug Schritt 2: Alle Details landen im Protokoll, nicht in der Konsole.
                    if candidate_region is None:
                        self.log(
                            f"{' '*8}Kandidat abgelehnt: joint_id={joint_id}; "
                            f"pair_id={pair_id}; Grund=keine Kontaktregion aus Schritt 1",
                            print_console=False
                        )
                        continue
                    if (candidate_pair in assigned_exclusive_pairs
                            and candidate_pair not in member_pairs):
                        self.log(
                            f"{' '*8}Kandidat abgelehnt: joint_id={joint_id}; "
                            f"pair_id={pair_id}; Grund=linienartiges Paar bereits anderem Joint zugeordnet",
                            print_console=False
                        )
                        continue

                    distance_limit = joint_distance_limit(member_guids, joint_id)
                    if distance_limit is None:
                        self.log(
                            f"{' '*8}Kandidat abgelehnt: joint_id={joint_id}; "
                            f"pair_id={pair_id}; Grund=kein begründbarer Abstandsgrenzwert",
                            print_console=False
                        )
                        continue

                    failed_comparison = None
                    # Der Kandidat muss zu jeder bereits enthaltenen Region passen,
                    # nicht nur zum ursprünglichen Startanker.
                    for member_region in member_regions:
                        min_distance = self._minimum_point_cloud_distance(
                            candidate_region["points"], member_region["points"]
                        )
                        if min_distance > distance_limit:
                            failed_comparison = (member_region, min_distance)
                            break
                    if failed_comparison is not None:
                        member_region, min_distance = failed_comparison
                        self.log(
                            f"{' '*8}Kandidat abgelehnt: joint_id={joint_id}; "
                            f"level={level}; pair_id={pair_id}; "
                            f"region_id={candidate_region['region_id']}; "
                            f"Referenzregion={member_region['region_id']}; "
                            f"d_min_m={min_distance:.3f}; d_max_m={distance_limit:.3f}",
                            print_console=False
                        )
                        continue

                    member_pairs.append(candidate_pair)
                    member_regions.append(candidate_region)
                    member_guids.update(candidate_pair)
                    counts["accepted"] += 1
                    counts["pairs"].append(pair_id)
                    next_frontier.append((candidate_pair, level))
                    guid_a, guid_b = candidate_pair
                    self.log(
                        f"{' '*8}Kandidat aufgenommen: joint_id={joint_id}; "
                        f"level={level}; pair_id={pair_id}; "
                        f"region_id={candidate_region['region_id']}; "
                        f"GUID_A={guid_a}; GUID_B={guid_b}",
                        print_console=False
                    )

                    if len(member_guids) >= 7 and not warned_seven:
                        self.log(
                            f"WARNUNG Schritt 2: Joint mit mindestens 7 Bauteilen prüfbedürftig; "
                            f"joint_id={joint_id}; GUID_count={len(member_guids)}",
                            print_console=False
                        )
                        warned_seven = True
                    if len(member_guids) >= 10 and not warned_ten:
                        self.log(
                            f"WARNUNG Schritt 2: Mögliche Überkopplung ab 10 Bauteilen; "
                            f"joint_id={joint_id}; GUID_count={len(member_guids)}",
                            print_console=False
                        )
                        warned_ten = True

                frontier = next_frontier

            # Zusätzliche Suchebenen sind erlaubt, werden aber zur fachlichen
            # Kontrolle im Protokoll hervorgehoben.
            if 2 in level_counts:
                counts = level_counts[2]
                self.log(
                    f"WARNUNG Schritt 2: Suche über Ebene 1 hinaus; joint_id={joint_id}; "
                    f"level=2; geprüft={counts['checked']}; aufgenommen={counts['accepted']}; "
                    f"pair_ids={counts['pairs']}", print_console=False
                )
            deeper_levels = [level for level in level_counts if level >= 3]
            if deeper_levels:
                checked = sum(level_counts[level]["checked"] for level in deeper_levels)
                accepted = sum(level_counts[level]["accepted"] for level in deeper_levels)
                pair_ids_deep = [
                    pair_id for level in sorted(deeper_levels)
                    for pair_id in level_counts[level]["pairs"]
                ]
                self.log(
                    f"WARNUNG Schritt 2: Suche über Ebene 2 hinaus; joint_id={joint_id}; "
                    f"level_ab={min(deeper_levels)}; geprüft={checked}; aufgenommen={accepted}; "
                    f"pair_ids={pair_ids_deep}", print_console=False
                )

            assigned_pairs.update(member_pairs)
            for pair in member_pairs:
                if not pair_allows_multiple_joints(pair):
                    assigned_exclusive_pairs.add(pair)

            # Die Joint-Position ist ein punktgewichteter Schwerpunkt und dient
            # nur der Orientierung, nicht als räumlicher Suchanker.
            point_sum = np.zeros(3, dtype=float)
            point_count = 0
            for region in member_regions:
                points = np.asarray(region["points"], dtype=float).reshape(-1, 3)
                finite_points = points[np.all(np.isfinite(points), axis=1)]
                point_sum += np.sum(finite_points, axis=0)
                point_count += len(finite_points)
            position = (point_sum / point_count).tolist() if point_count else [0.0, 0.0, 0.0]

            elements = []
            for guid in sorted(member_guids):
                node = nodes_by_guid.get(guid, {})
                category, thickness, _ = category_and_thickness(guid)
                elements.append({
                    "guid": guid,
                    "name": node.get("name"),
                    "entity": node.get("entity_processed"),
                    "dimensions": list(node["dimensions"]) if node.get("dimensions") is not None else None,
                    "category": category,
                    "characteristic_thickness_m": thickness,
                })

            joint = {
                "joint_id": joint_id,
                "pair_ids": sorted({pair_ids.get(pair, "unbekannt") for pair in member_pairs}),
                "region_ids": sorted({region["region_id"] for region in member_regions}),
                "guids": sorted(member_guids),
                "guid_count": len(member_guids),
                "elements": elements,
                "position": [round(value, 3) for value in position],
            }
            soll_joints.append(joint)
            
            if len(soll_joints) % 10 == 0 or idx == len(sorted_regions):
                self.log(
                    f"{' '*8}1.A Schritt 2: {joint['joint_id']} erstellt...",
                    print_console=True
                )
            
            self.log(
                f"#Debug Schritt 2 Joint fertig: joint_id={joint_id}; "
                f"pair_ids={joint['pair_ids']}; region_ids={joint['region_ids']}; "
                f"GUID_count={joint['guid_count']}; GUIDs={joint['guids']}; "
                f"position={joint['position']}",
                print_console=False
            )

        guid_counts = [d['guid_count'] for d in soll_joints if "guid_count" in d]
        valenz = sum(guid_counts) / len(guid_counts) if guid_counts else 0        
        
        # Protokoll - Zusammenfassung Schritt 2
        self.log("", print_console=False) # Leerzeile vor Bericht
        self.log(
            f"1.A Schritt 2: Zusammenfassung: "
            f"\nSoll-Joints={len(soll_joints)} | "
            f"Valenz={valenz:.2f} | min_count={min(guid_counts)} | max_count={max(guid_counts)}",
            print_console=False
        )
        
        # Protokoll - Tabelle - Übersichtsliste aller Soll-Joints
        self.log("", print_console=False) # Leerzeile vor Liste
        for joint in soll_joints:
            self.log(
                f"1.A Schritt 2: Soll_Joint_ID: {joint['joint_id']} | "
                f"counts: {joint['guid_count']} | position: {joint['position']} | "
                f"GUIDs: {joint['guids']}"
            )
        return soll_joints


    def _contact_from_surfaces(self, tri_a, tri_b, box_min, box_max):
        """
        Prüft die Dreiecksoberflächen innerhalb der Box auf Abstand <= eps.
        Hinweis: Die 'test_points' aus dem Dateneingang tasten nur Dreieckskanten ab und
        reichen für Flächenkontakte nicht aus; daher werden die Dreiecksflächen hier selbst abgetastet.

        Args:
            tri_a (np.ndarray): Dreiecke von Bauteil a, Shape (n, 3, 3).
            tri_b (np.ndarray): Dreiecke von Bauteil b, Shape (m, 3, 3).
            box_min (np.ndarray): Untere Ecke des lokalen Suchbereichs.
            box_max (np.ndarray): Obere Ecke des lokalen Suchbereichs.

        Returns:
            tuple: (points, d_min)
                points (np.ndarray | None): Kontaktpunkte (Vereinigung beider Seiten) oder None.
                d_min (float): Kleinster gefundener Oberflächenabstand in [m].
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


    def _minimum_mesh_distance(self, triangles_a, triangles_b):
        """Berechnet den kleinsten Abstand zweier Dreiecksnetze."""
        if not len(triangles_a) or not len(triangles_b):
            return float('inf')
        vertices_a = triangles_a.reshape(-1, 3)
        vertices_b = triangles_b.reshape(-1, 3)
        distance = min(
            float(self._point_triangle_distance(vertices_a, triangles_b).min()),
            float(self._point_triangle_distance(vertices_b, triangles_a).min()),
        )

        edges_a = np.concatenate((
            triangles_a[:, [0, 1]],
            triangles_a[:, [1, 2]],
            triangles_a[:, [2, 0]],
        ))
        edges_b = np.concatenate((
            triangles_b[:, [0, 1]],
            triangles_b[:, [1, 2]],
            triangles_b[:, [2, 0]],
        ))

        def edges_intersect_triangles(edges, triangles):
            block_size = max(1, 100_000 // len(triangles))
            a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            edge_1, edge_2 = b - a, c - a
            for start in range(0, len(edges), block_size):
                block = edges[start:start + block_size]
                origin = block[:, None, 0]
                direction = (block[:, 1] - block[:, 0])[:, None, :]
                h = np.cross(direction, edge_2[None, :, :])
                determinant = np.sum(edge_1[None, :, :] * h, axis=2)
                valid = np.abs(determinant) > 1e-12
                inverse = np.divide(
                    1.0,
                    determinant,
                    out=np.zeros_like(determinant),
                    where=valid,
                )
                offset = origin - a[None, :, :]
                u = inverse * np.sum(offset * h, axis=2)
                q = np.cross(offset, edge_1[None, :, :])
                v = inverse * np.sum(direction * q, axis=2)
                t = inverse * np.sum(edge_2[None, :, :] * q, axis=2)
                if np.any(
                    valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
                    & (t >= 0.0) & (t <= 1.0)
                ):
                    return True
            return False

        if edges_intersect_triangles(edges_a, triangles_b):
            return 0.0
        if edges_intersect_triangles(edges_b, triangles_a):
            return 0.0

        starts_b = edges_b[None, :, 0]
        vectors_b = (edges_b[:, 1] - edges_b[:, 0])[None, :, :]

        def point_segment_distance(points, starts, vectors):
            parameter = np.clip(
                np.sum((points - starts) * vectors, axis=2)
                / np.maximum(np.sum(vectors * vectors, axis=2), 1e-18),
                0.0,
                1.0,
            )
            return np.linalg.norm(points - (starts + parameter[..., None] * vectors), axis=2)

        block_size = max(1, 100_000 // len(edges_b))
        for start in range(0, len(edges_a), block_size):
            block = edges_a[start:start + block_size]
            starts_a = block[:, None, 0]
            vectors_a = (block[:, 1] - block[:, 0])[:, None, :]
            starts_a_end = starts_a + vectors_a
            starts_b_end = starts_b + vectors_b

            dists = [
                point_segment_distance(starts_a, starts_b, vectors_b),
                point_segment_distance(starts_a_end, starts_b, vectors_b),
                point_segment_distance(starts_b, starts_a, vectors_a),
                point_segment_distance(starts_b_end, starts_a, vectors_a),
            ]

            w = starts_a - starts_b
            aa = np.sum(vectors_a * vectors_a, axis=2)
            ab = np.sum(vectors_a * vectors_b, axis=2)
            bb = np.sum(vectors_b * vectors_b, axis=2)
            aw = np.sum(vectors_a * w, axis=2)
            bw = np.sum(vectors_b * w, axis=2)
            denominator = aa * bb - ab * ab
            valid = denominator > 1e-18
            s = np.divide(
                ab * bw - bb * aw,
                denominator,
                out=np.zeros_like(denominator),
                where=valid,
            )
            t = np.divide(
                aa * bw - ab * aw,
                denominator,
                out=np.zeros_like(denominator),
                where=valid,
            )
            interior = valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
            interior_distance = np.linalg.norm(
                w + s[..., None] * vectors_a - t[..., None] * vectors_b,
                axis=2,
            )
            dists.append(np.where(interior, interior_distance, np.inf))
            distance = min(distance, float(np.min(np.minimum.reduce(dists))))
            if distance == 0.0:
                return 0.0
        return distance


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

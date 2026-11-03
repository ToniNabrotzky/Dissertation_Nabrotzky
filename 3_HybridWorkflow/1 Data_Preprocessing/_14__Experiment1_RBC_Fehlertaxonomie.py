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

# Schlüssel einer Kontaktregion (Reihenfolge wie im Dictionary in _compute_contact_regions)
CONTACT_REGION_KEYS = (
    "region_id", "pair_id", "guid_a", "guid_b", "points", "centroid", "eigenvalues",
    "lengths", "contact_type", "contact_method", "volume"
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
            f"#Debug Schritt 1 Konfiguration: epsilon={100 * JOINT_EPSILON:.6f} cm; tau={JOINT_TAU}; "
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
            pair_id = f"P{pair_index:04d}"
            self.log("", print_console=False) # Leerzeile: Sprung zum nächsten GT-Paar
            node_a = nodes_by_guid.get(guid_a)
            node_b = nodes_by_guid.get(guid_b)
            if not node_a or not node_b:
                reason = "GUID nicht in den extrahierten Knoten gefunden"
                missing_pairs.append({
                    "pair_id": pair_id, "guid_a": guid_a, "guid_b": guid_b, "reason": reason
                })
                self.log(
                    f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
                    f"(ohne Kontaktregion): pair_id={pair_id}; GUID_A={guid_a}; "
                    f"GUID_B={guid_b}; Grund={reason}",
                    print_console=False
                )
                if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                    self.log(
                        f"Fortschritt Kontaktpaare: {pair_index}/{len(sorted_pairs)}",
                        print_console=True
                    )
                continue

            # Stufe 1: AABB-Vorfilter. raw_dims = Überlappung der Boxen je Achse (negativ = Lücke)
            (min_a, max_a) = node_a["geom_raw"]["aabb"]
            (min_b, max_b) = node_b["geom_raw"]["aabb"]
            overlap_min = np.maximum(min_a, min_b)
            overlap_max = np.minimum(max_a, max_b)
            raw_dims = overlap_max - overlap_min
            if np.any(raw_dims < -JOINT_EPSILON):
                reason = "AABBs um mehr als Epsilon getrennt"
                missing_pairs.append({
                    "pair_id": pair_id, "guid_a": guid_a, "guid_b": guid_b, "reason": reason
                })
                self.log(
                    f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
                    f"(ohne Kontaktregion): pair_id={pair_id}; GUID_A={guid_a}; "
                    f"GUID_B={guid_b}; Grund={reason}; "
                    f"AABB-Überlappung_m={np.round(raw_dims, 6).tolist()}",
                    print_console=False
                )
                if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                    self.log(
                        f"Fortschritt Kontaktpaare: {pair_index}/{len(sorted_pairs)}",
                        print_console=True
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
                reason = "Kein Punkt innerhalb Epsilon"
                missing_pairs.append({
                    "pair_id": pair_id, "guid_a": guid_a, "guid_b": guid_b, "reason": reason
                })
                self.log(
                    f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
                    f"(ohne Kontaktregion): pair_id={pair_id}; GUID_A={guid_a}; "
                    f"GUID_B={guid_b}; Grund={reason}; "
                    f"kleinster_Oberflächenabstand_m={d_min:.6f}",
                    print_console=False
                )
                if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                    self.log(
                        f"Fortschritt Kontaktpaare: {pair_index}/{len(sorted_pairs)}",
                        print_console=True
                    )
                continue

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
                "volume": volume
            }
            contact_regions.append(region)

            self.log(
                f"#Debug Schritt 1 - Paar {pair_index}/{len(sorted_pairs)} "
                f"(Kontaktregion {region['region_id']}): pair_id={pair_id}; "
                f"GUID_A={guid_a}; GUID_B={guid_b}; points={len(contact_points)}; "
                f"centroid={np.round(centroid, 6).tolist()}; "
                f"eigenvalues={np.round(eigenvalues, 8).tolist()}; "
                f"lengths_m={np.round(L, 6).tolist()}; "
                f"contact_method={method}; contact_type={contact_type}; "
                f"volume_m3={volume:.6f}",
                print_console=False
            )
            if pair_index % 50 == 0 or pair_index == len(sorted_pairs):
                self.log(
                    f"Fortschritt Kontaktpaare: {pair_index}/{len(sorted_pairs)}",
                    print_console=True
                )
        self.log("", print_console=False) # Leerzeile nach dem letzten GT-Paar

        type_counts = {}
        for region in contact_regions:
            type_counts[region["contact_type"]] = type_counts.get(region["contact_type"], 0) + 1
        self.log(f"1.A Schritt 1: {len(contact_regions)} Kontaktregionen gefunden.")
        self.log(
            f"#Debug Schritt 1 Zusammenfassung: GT-Paare={len(gt_pairs)}; "
            f"Kontaktregionen={len(contact_regions)}; ohne Region={len(missing_pairs)}; "
            f"Verteilung_contact_type={type_counts}; "
            f"fehlende_Paar_IDs={[missing['pair_id'] for missing in missing_pairs]}",
            print_console=False
        )
        self.log(
            "#Debug Schritt 1 Tabelle Paare ohne Kontaktregion:",
            print_console=False
        )
        self.log(
            "#Debug Schritt 1 | pair_id | GUID_A | GUID_B | Grund |",
            print_console=False
        )
        for missing in missing_pairs:
            self.log(
                f"#Debug Schritt 1 | {missing['pair_id']} | {missing['guid_a']} | "
                f"{missing['guid_b']} | {missing['reason']} |",
                print_console=False
            )
        self.log(
            "#Debug Schritt 1 Kontaktpaar-Tabelle:",
            print_console=False
        )
        self.log(
            "#Debug Schritt 1 | pair_id | region_id | GUID_A | GUID_B |",
            print_console=False
        )
        for region in contact_regions:
            self.log(
                f"#Debug Schritt 1 | {region['pair_id']} | {region['region_id']} | "
                f"{region['guid_a']} | {region['guid_b']} |",
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
        alle Gitterpunkte, die in beiden Bauteilen liegen (Strahl-Parität, siehe _inside_mesh).
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


    def _inside_mesh(self, points, triangles):
        """
        Punkt-in-Netz-Test per Strahl-Parität: Ein Strahl in +z-Richtung schneidet die Dreiecke;
        eine ungerade Anzahl Schnitte oberhalb des Punkts bedeutet 'innen'. Voraussetzung ist ein
        geschlossenes Netz. Die Punkte werden in x/y um ~1e-7 m verschoben, damit der Strahl keine
        Dreieckskanten exakt trifft (sonst würde die Parität kippen).

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

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
   1.A Schritt 1: Kontaktregionen (Experiment1PhaseASchritt1)
   1.A Schritt 2: Soll-Joints (Experiment1PhaseASchritt2)
   1.B Schritt 3: Klassifizierung (Experiment1PhaseB)
   
Struktur (Soll-Ziel):
1. KONFIGURATION (Experiment 1)
2. KLASSE Experiment1RBCFehlertaxonomie
   1.A Rekonstruktion von Anschlussstellen (Soll-Joints, IFC-/GT-basiert)
   1.B Klassifizierungssystem für Anschlussstellen
   1.C Extraktion von Anschlussstellen aus dem Analysemodell (Ist-Joints)
   1.D Validierung: Matching, Fehlercodes (E0-E7), Statistik

1.B Klassifizierung (drei Achsen, konverterunabhängige Prädiktoren):
   Achse 1 Valenz:      val, h, p
   Achse 2 Versatz:     o_ab = c_ideal - c_real, Õ, O0-O3, Γ (Dimensionssprung)
   Achse 3 Morphologie: K0-K3, Konfiguration I/L/T/X/Y/KN, Winkelklasse
Die mechanische Achse entfällt. Ein Komplexitätsindex (JCI) wird bewusst NICHT
berechnet: Gewichte werden erst nach der Regression festgelegt (Teil D.3, H5).
Notation: t = Dicke (thickness); p = Innentopologie-Dichte; ρ = Lateralverhältnis;
Γ = Dimensionssprung |dim(a) - dim(b)|; KN = Multiknoten.
Elementdimensionen: 1 (linear), 2 (planar); 3D-Elementtypen werden nicht betrachtet.
"""
import itertools
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

# 1.B Klassifizierung
JOINT_O_TOL = 0.05          # [-] Õ unterhalb dieser Schwelle gilt der Versatz als O0 (ρ wird dann nicht berechnet). Default = 0.05
JOINT_RHO_AXIAL = 0.2       # [-] ρ < Schwelle -> O1 (nahezu reiner kollinearer Versatz). Default = 0.2
JOINT_RHO_LATERAL = 0.8     # [-] ρ > Schwelle -> O2 (nahezu reiner lateraler Versatz). Default = 0.8
JOINT_ANGLE_TOL = 5.0       # [°] Toleranz für orthogonal (90° ± Tol) und parallel (<= Tol). Default = 5.0
JOINT_END_TOL_SCALE = 0.5   # [-] Rand-Toleranz r_end = eps + Skala * t für die Lage am Bauteilende/-rand. Default = 0.5 (an Validierungsfällen kalibrieren)
JOINT_IDEAL_TOL = 1e-3      # [m] Abstand, bis zu dem sich Idealisierungen als geschnitten gelten (Fall A). Default = 1e-3
JOINT_PARALLEL_TOL = 1e-4   # [-] Sinus-Grenzwert, unterhalb dem zwei Richtungen als parallel gelten. Default = 1e-4
JOINT_REGULARIZATION = 1e-6 # [-] Gewicht der Nähe zu c_real bei mehrdeutigen kürzesten Verbindungen (parallele Idealisierungen). Default = 1e-6
JOINT_EPSILON_STUDY = (0.025, 0.05, 0.1)  # [m] Sensitivitätsstudie für eps (Phase A + B je Wert neu)

CONTACT_TYPE_TO_K = {"0D_Punkt": "K0", "1D_Linie": "K1", "2D_Fläche": "K2", "3D_Volumen": "K3"}

# Schlüssel der Klassifizierung (Reihenfolge wie in _classify_pair bzw. _classify_joint)
PAIR_CLASS_KEYS = (
    "pair_id", "region_id", "guid_a", "guid_b", "status", "reason", "contact_k",
    "reference_guid", "o_vector", "o_length_m", "o_O1_length_m", "o_O2_length_m",
    "o_norm", "rho", "o_class", "gamma", "ideal_case", "fall_c_fallback",
    "position_a", "position_b", "configuration", "angle_deg", "angle_class"
)
JOINT_CLASS_KEYS = (
    "joint_id", "valence", "valence_class", "h", "p",
    "o_norm_max", "o_norm_mean", "o_norm_sd", "o_class", "o_class_profile", "gamma_max",
    "k_max", "k_profile", "configuration", "pair_configurations", "angle_class",
    "pair_count", "pair_count_evaluated", "pairs", "shortcode"
)

# =========================================================
# EXPERIMENT 1, PHASE A
# =========================================================
class _PhaseASchritt1Geometrie:
    """Geometrische Hilfsmethoden für die Ermittlung der Kontaktregionen."""

    def _contact_from_surfaces(self, tri_a, tri_b, box_min, box_max, epsilon=None):
        """
        Prüft die Dreiecksoberflächen innerhalb der Box auf Abstand <= eps.
        Hinweis: Die 'test_points' aus dem Dateneingang tasten nur Dreieckskanten ab und
        reichen für Flächenkontakte nicht aus; daher werden die Dreiecksflächen hier selbst abgetastet.

        Args:
            tri_a (np.ndarray): Dreiecke von Bauteil a, Shape (n, 3, 3).
            tri_b (np.ndarray): Dreiecke von Bauteil b, Shape (m, 3, 3).
            box_min (np.ndarray): Untere Ecke des lokalen Suchbereichs.
            box_max (np.ndarray): Obere Ecke des lokalen Suchbereichs.
            epsilon (float | None): Kontakttoleranz in [m]; None = JOINT_EPSILON.

        Returns:
            tuple: (points, d_min)
                points (np.ndarray | None): Kontaktpunkte (Vereinigung beider Seiten) oder None.
                d_min (float): Kleinster gefundener Oberflächenabstand in [m].
        """
        eps = JOINT_EPSILON if epsilon is None else float(epsilon)
        # Abtastquellen: Dreiecke in der Box; Abstandsziele: Dreiecke bis eps um die Box
        source_a = self._select_triangles_in_box(tri_a, box_min, box_max)
        source_b = self._select_triangles_in_box(tri_b, box_min, box_max)
        target_a = self._select_triangles_in_box(tri_a, box_min - eps, box_max + eps)
        target_b = self._select_triangles_in_box(tri_b, box_min - eps, box_max + eps)
        if len(source_a) == 0 or len(source_b) == 0:
            return None, float('inf')

        samples_a = self._sample_triangles(source_a, box_min, box_max)
        samples_b = self._sample_triangles(source_b, box_min, box_max)
        d_a = self._point_triangle_distance(samples_a, target_b) if len(samples_a) else np.empty(0)
        d_b = self._point_triangle_distance(samples_b, target_a) if len(samples_b) else np.empty(0)

        d_min = min(d_a.min() if len(d_a) else float('inf'), d_b.min() if len(d_b) else float('inf'))
        contact = np.vstack([samples_a[d_a <= eps], samples_b[d_b <= eps]])
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


class Experiment1PhaseASchritt1(_PhaseASchritt1Geometrie):
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


    def _compute_contact_regions(self, nodes, gt_pairs, epsilon=None):
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
            epsilon (float | None): Kontakttoleranz in [m]; None = JOINT_EPSILON.

        Returns:
            list[dict]: Kontaktregionen mit den Schlüsseln aus CONTACT_REGION_KEYS.
        """
        eps = JOINT_EPSILON if epsilon is None else float(epsilon)
        self.log("", print_console=False) # Leerzeile: Sprung zum nächsten GT-Paar
        self.log(f"{'='*10} EXPERIMENT_1 RBC_Fehlertaxonomie {'='*10}")
        self.log(f"1.A Schritt 1: Bestimme Kontaktregionen für {len(gt_pairs)} GT-Paare (eps = {eps*1000:.1f} mm)...")
        self.log(
            f"1.A Schritt 1: Konfiguration: epsilon={100 * eps:.2f} cm; tau={JOINT_TAU}; "
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
            
            if np.any(axis_gaps > eps):
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
                tri_a, tri_b, overlap_min - eps, overlap_max + eps, epsilon=eps
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
            f"\n GT-Paare={len(gt_pairs)} | "
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


class _BauteilEigenschaften:
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



class Experiment1PhaseASchritt2(_BauteilEigenschaften):
    """Rekonstruiert Soll-Joints aus Kontaktregionen und Ground Truth."""

    log: Callable[..., None]

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
        min_count = min(guid_counts) if guid_counts else "n/a"
        max_count = max(guid_counts) if guid_counts else "n/a"
        
        # Protokoll - Zusammenfassung Schritt 2
        self.log("", print_console=False) # Leerzeile vor Bericht
        self.log(
            f"1.A Schritt 2: Zusammenfassung: "
            f"\nSoll-Joints={len(soll_joints)} | "
            f"Valenz={valenz:.2f} | min_count={min_count} | max_count={max_count}",
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


class Experiment1PhaseBSchritt3(_BauteilEigenschaften):
    """Klassifiziert Soll-Joints anhand der Ergebnisse von Phase A."""

    log: Callable[..., None]

    # Die Klassifizierung verwendet auch Bauteileigenschaften aus Phase A.
    def _classify_soll_joints(self, nodes, gt_pairs, contact_regions, soll_joints, epsilon=None):
        """
        Schritt 3: Klassifiziert jeden Soll-Joint über drei Achsen
        (Valenz, Idealisierungsversatz, Morphologie).

        Versatz o_ab = c_ideal - c_real je Paar; Aggregation je Joint über Beträge
        (Õ_max als Hauptprädiktor), Vektoren werden nicht gemittelt.

        Args:
            nodes (list[dict]): IFC-Knoten mit 'geom_raw', 'entity_processed', 'dimensions'.
            gt_pairs (iterable[tuple[str, str]]): Ground-Truth-Paare (für p(J)).
            contact_regions (list[dict]): Kontaktregionen aus Schritt 1.
            soll_joints (list[dict]): Soll-Joints aus Schritt 2.
            epsilon (float | None): Kontakttoleranz in [m]; None = JOINT_EPSILON.

        Returns:
            list[dict]: Je Joint ein Dict mit den Schlüsseln aus JOINT_CLASS_KEYS.
        """
        eps = JOINT_EPSILON if epsilon is None else float(epsilon)
        
        self.log("", print_console=False) # Leerzeile
        self.log(f"1.B Schritt 3: Klassifiziere {len(soll_joints)} Soll-Joints (eps = {eps*1000:.1f} mm)...")
        self.log(
            f"1.B Schritt 3: Konfiguration: o_tol={JOINT_O_TOL}; rho_axial={JOINT_RHO_AXIAL}; "
            f"rho_lateral={JOINT_RHO_LATERAL}; angle_tol={JOINT_ANGLE_TOL}°; "
            f"end_tol_scale={JOINT_END_TOL_SCALE}; ideal_tol={JOINT_IDEAL_TOL} m; "
            f"delta_scale={JOINT_DELTA_SCALE}", print_console=False
        )
        self.log(f"1.B Schritt 3: Schlüssel je Joint: {list(JOINT_CLASS_KEYS)}", print_console=False)

        nodes_by_guid = {node["guid"]: node for node in nodes}
        regions_by_id = {region["region_id"]: region for region in contact_regions}
        neighbours = {}
        for guid_a, guid_b in {tuple(sorted(pair)) for pair in gt_pairs}:
            neighbours.setdefault(guid_a, set()).add(guid_b)
            neighbours.setdefault(guid_b, set()).add(guid_a)

        ideal_cache = {}
        def ideal_of(guid):
            if guid not in ideal_cache:
                node = nodes_by_guid.get(guid)
                if node is None:
                    ideal_cache[guid] = (None, "GUID nicht in den extrahierten Knoten gefunden")
                else:
                    ideal_cache[guid] = self._reference_idealization(node)
                if ideal_cache[guid][0] is None:
                    self.log(
                        f"WARNUNG 1.B Schritt 3: Referenzidealisierung nicht bestimmbar; "
                        f"GUID={guid}; Grund={ideal_cache[guid][1]}", print_console=False
                    )
            return ideal_cache[guid]

        classified = []
        for joint in soll_joints:
            regions = [regions_by_id[rid] for rid in joint["region_ids"] if rid in regions_by_id]
            pairs = [self._classify_pair(region, ideal_of, eps) for region in regions]
            for pair in pairs:
                if pair["status"] != "bewertet":
                    self.log(
                        f"WARNUNG 1.B Schritt 3: Paar nicht bewertbar; joint_id={joint['joint_id']}; "
                        f"pair_id={pair['pair_id']}; Grund={pair['reason']}", print_console=False
                    )
                elif pair["fall_c_fallback"]:
                    self.log(
                        f"WARNUNG 1.B Schritt 3: Fall-C-Fallback (kein Punkt im Suchradius, Fall B global); "
                        f"joint_id={joint['joint_id']}; pair_id={pair['pair_id']}", print_console=False
                    )
            result = self._classify_joint(joint, pairs, neighbours)
            if result["pair_count_evaluated"] == 0:
                self.log(
                    f"WARNUNG 1.B Schritt 3: Joint ohne bewertbare Paare; joint_id={joint['joint_id']}",
                    print_console=False
                )
            classified.append(result)
            self.log(
                f"1.B Schritt 3: {joint['joint_id']} | {result['shortcode']} | "
                f"Paare={result['pair_count_evaluated']}/{result['pair_count']}", print_console=False
            )

        # Protokoll - Zusammenfassung Schritt 3
        def count(key):
            counts = {}
            for item in classified:
                counts[item[key]] = counts.get(item[key], 0) + 1
            return dict(sorted(counts.items(), key=lambda kv: str(kv[0])))
        
        self.log("", print_console=False) # Leerzeile vor Bericht
        self.log(
            f"1.B Schritt 3: Zusammenfassung: Joints={len(classified)}"
            f"\n    Achse 1 Valenz: {count('valence_class')} | h: {count('h')}"
            f"\n    Achse 2 Versatz: O-Klassen: {count('o_class')} | Γ_max: {count('gamma_max')}"
            f"\n    Achse 3 Morphologie: K_max: {count('k_max')} | Konfiguration: {count('configuration')} | "
            f"Winkel: {count('angle_class')}", print_console=False
        )
        return classified


    def _classify_joint(self, joint, pairs, neighbours):
        """
        Aggregiert die Paarmerkmale zu den Joint-Merkmalen aller drei Achsen.

        Args:
            joint (dict): Soll-Joint (guids, elements, guid_count, joint_id).
            pairs (list[dict]): Paarmerkmale aus _classify_pair.
            neighbours (dict): GUID -> Menge der GT-Nachbarn (für p).

        Returns:
            dict: Joint-Merkmale mit den Schlüsseln aus JOINT_CLASS_KEYS.
        """
        # Achse 1 Anschlussgrad (Valenz)
        guids = set(joint["guids"])
        valence = int(joint["guid_count"])
        # Achse 1 Anschlussklasse
        valence_class = f"V{valence}" if valence <= 4 else "V5+"
        # Achse 2 Referenzidealisierung
        categories = {element.get("category") for element in joint.get("elements", [])} - {None}
        # Achse 1 Dimensionsheterogenität
        h = len(categories) if categories else None

        edges = sum(len(neighbours.get(guid, set()) & guids) for guid in guids) // 2
        max_edges = valence * (valence - 1) / 2
        # Achse 1 Innentopologische Dichte
        p = edges / max_edges if max_edges else None
        
        # Achse 2 Versatzklassen
        evaluated = [pair for pair in pairs if pair["status"] == "bewertet"]
        o_values = np.array([pair["o_norm"] for pair in evaluated], dtype=float)
        profile = {name: 0 for name in ("O0", "O1", "O2", "O3")}
        for pair in evaluated:
            profile[pair["o_class"]] += 1

        o_max = o_mean = o_sd = o_class = gamma_max = None
        if evaluated:
            o_max = float(o_values.max())
            o_mean = float(o_values.mean())
            o_sd = float(o_values.std()) # Populations-SD (ddof=0); bei einem Paar 0
            top = min((pair for pair in evaluated if pair["o_norm"] == o_max), key=lambda pair: pair["pair_id"])
            o_class = top["o_class"]
            gamma_max = max(pair["gamma"] for pair in evaluated)

        # Achse 3 Morphologie (Berührunsmodus)
        k_profile = {name: 0 for name in CONTACT_TYPE_TO_K.values()}
        for pair in pairs:
            if pair["contact_k"] in k_profile:
                k_profile[pair["contact_k"]] += 1
        present = [name for name, number in k_profile.items() if number]
        k_max = max(present) if present else None

        pair_configurations = [pair["configuration"] for pair in evaluated]
        if not evaluated:
            configuration = None
        elif valence == 2:
            configuration = evaluated[0]["configuration"]
        elif valence == 3 and all(
            pair["position_a"] == "Rand" and pair["position_b"] == "Rand" for pair in evaluated
        ):
            configuration = "Y"
        else:
            configuration = "KN"

        # Achse 3 Morphologie (Winkel)
        angle_classes = [pair["angle_class"] for pair in evaluated]
        if not angle_classes:
            angle_class = None
        elif "schiefwinklig" in angle_classes:
            angle_class = "schiefwinklig"
        elif "orthogonal" in angle_classes:
            angle_class = "orthogonal"
        else:
            angle_class = "parallel"

        short = {"orthogonal": "orth", "parallel": "par", "schiefwinklig": "schief"}
        shortcode = (
            f"{valence_class}-h{h if h is not None else 'n/a'}-"
            f"p{p:.2f} | " if p is not None else f"{valence_class}-h{h if h is not None else 'n/a'}-pn/a | "
        )
        shortcode += (
            f"{o_class}-Õ{o_max:.2f}" if o_max is not None else "On/a-Õn/a"
        )
        shortcode += (
            f" | {k_max or 'Kn/a'}-{configuration or 'n/a'}-{short.get(angle_class, 'n/a')}"
        )

        return {
            "joint_id": joint["joint_id"], 
            # Achse 1 (Valenz, Dimensionshetegorenität, innentopologische Dichte)
            "valence": valence, "valence_class": valence_class, "h": h, "p": np.round(p, 6), 
            # Achse 2 (Versatzidealisierungen - Referenzidealsierung, paarweise, normiert, Versatzcharakter, Dimensionssprung)
            "o_norm_max": np.round(o_max, 6), "o_norm_mean": np.round(o_mean, 6), "o_norm_sd": np.round(o_sd, 6),
            "o_class": o_class, "o_class_profile": profile, "gamma_max": gamma_max,
            # Achse 3 (Morphologie - Berührungsmodus, -winkel)
            "k_max": k_max, "k_profile": k_profile, "configuration": configuration,
            "pair_configurations": pair_configurations, "angle_class": angle_class,
            # Metadaten
            "pair_count": len(pairs), "pair_count_evaluated": len(evaluated),
            "pairs": pairs, "shortcode": shortcode,
        }


    def _classify_pair(self, region, ideal_of, eps):
        """
        Berechnet alle Paarmerkmale (Versatz, Γ, Lage, Konfiguration, Winkel) für eine Kontaktregion.

        Bezugsbauteil für die Zerlegung des Versatzes: das dimensionsniedrigere Bauteil;
        bei gleicher Dimension das erste in sortierter GUID-Reihenfolge.
        Bezug 1D (Achse u):   o_O1 = (o·u)u (kollinear), o_O2 = o - o_O1 (lateral).
        Bezug 2D (Normale n): o_O2 = (o·n)n (lateral),   o_O1 = o - o_O2 (kollinear in der Ebene).

        Args:
            region (dict): Kontaktregion (guid_a, guid_b, centroid, contact_type, ...).
            ideal_of (Callable): GUID -> (Idealisierung | None, Grund).
            eps (float): Kontakttoleranz in [m].

        Returns:
            dict: Paarmerkmale mit den Schlüsseln aus PAIR_CLASS_KEYS.
        """
        guid_a, guid_b = sorted((region["guid_a"], region["guid_b"]))
        pair = dict.fromkeys(PAIR_CLASS_KEYS)
        pair.update({
            "pair_id": region.get("pair_id"), "region_id": region["region_id"],
            "guid_a": guid_a, "guid_b": guid_b, "status": "nicht_bewertbar",
            "contact_k": CONTACT_TYPE_TO_K.get(region.get("contact_type")),
            "fall_c_fallback": False,
        })
        ideal_a, reason_a = ideal_of(guid_a)
        ideal_b, reason_b = ideal_of(guid_b)
        if ideal_a is None or ideal_b is None:
            pair["reason"] = "; ".join(
                f"{guid}: {reason}" for guid, ideal, reason
                in ((guid_a, ideal_a, reason_a), (guid_b, ideal_b, reason_b)) if ideal is None
            )
            return pair

        c_real = np.asarray(region["centroid"], dtype=float)
        t_a, t_b = ideal_a["t"], ideal_b["t"]
        r_search = JOINT_DELTA_SCALE * max(t_a, t_b)
        c_ideal, case, fallback = self._ideal_meeting_point(ideal_a, ideal_b, c_real, r_search)

        o_vector = c_ideal - c_real
        o_length = float(np.linalg.norm(o_vector))
        o_norm = o_length / (0.5 * (t_a + t_b))

        if ideal_a["dim"] <= ideal_b["dim"]:
            reference, reference_guid = ideal_a, guid_a
        else:
            reference, reference_guid = ideal_b, guid_b
        o_class, rho, o_axial, o_lateral = self._offset_class(o_vector, o_length, o_norm, reference)

        angle = self._pair_angle(ideal_a, ideal_b)
        if abs(angle - 90.0) <= JOINT_ANGLE_TOL:
            angle_class = "orthogonal"
        elif angle <= JOINT_ANGLE_TOL:
            angle_class = "parallel"
        else:
            angle_class = "schiefwinklig"
        position_a = self._position_class(ideal_a, c_real, eps + JOINT_END_TOL_SCALE * t_a)
        position_b = self._position_class(ideal_b, c_real, eps + JOINT_END_TOL_SCALE * t_b)
        if position_a == "Rand" and position_b == "Rand":
            configuration = "I" if angle <= JOINT_ANGLE_TOL else "L"
        elif "Rand" in (position_a, position_b):
            configuration = "T"
        else:
            configuration = "X"

        pair.update({
            "status": "bewertet", "reason": None, "reference_guid": reference_guid,
            "o_vector": np.round(o_vector, 4).tolist(), "o_length_m": np.round(o_length, 4),
            "o_O1_length_m": np.round(o_axial, 4), "o_O2_length_m": np.round(o_lateral, 4),
            "o_norm": np.round(o_norm, 4), "rho": np.round(rho, 4), "o_class": o_class,
            "gamma": abs(ideal_a["dim"] - ideal_b["dim"]),
            "ideal_case": case, "fall_c_fallback": fallback,
            "position_a": position_a, "position_b": position_b,
            "configuration": configuration, "angle_deg": np.round(angle, 4), "angle_class": angle_class,
        })
        return pair


    def _clip_line_to_rectangle(self, point, direction, rect, t_range):
        """
        Schneidet die Gerade point + t*direction (t in t_range) mit den In-Ebene-Grenzen
        eines Rechtecks (Abstand zur Ebene wird nicht geprüft). Liang-Barsky.

        Returns:
            tuple | None: (t_min, t_max) oder None bei leerem Schnitt.
        """
        lo, hi = t_range
        for axis, half in zip(rect["dirs"], rect["half"]):
            offset = float((point - rect["center"]) @ axis)
            speed = float(direction @ axis)
            limit = half + JOINT_IDEAL_TOL
            if abs(speed) < 1e-12:
                if abs(offset) > limit:
                    return None
            else:
                t1, t2 = (-limit - offset) / speed, (limit - offset) / speed
                lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        return (lo, hi) if lo <= hi + 1e-12 else None


    def _closest_points(self, ideal_a, ideal_b, reference_point):
        """
        Kürzeste Verbindung zweier Idealisierungen (konvexe Mengen: Strecke/Rechteck).

        Gelöst wird das konvexe Problem min ||P_a(x_a) - P_b(x_b)||² unter Boxrestriktionen
        exakt durch Aufzählung aller Aktivmengen (je Parameter: unten, frei, oben; höchstens 3^4)
        mit Kleinste-Quadrate-Lösung der freien Parameter. Bei mehrdeutiger Lösung
        (parallele Idealisierungen) wählt ein Regularisierungsterm (JOINT_REGULARIZATION)
        das Paar, dessen Mittelpunkt am nächsten an reference_point (= c_real) liegt.

        Returns:
            tuple: (p_a, p_b, distance)
        """
        dirs_a, dirs_b = ideal_a["dirs"], ideal_b["dirs"]
        count_a, count_b = len(dirs_a), len(dirs_b)
        half = np.array(list(ideal_a["half"]) + list(ideal_b["half"]), dtype=float)
        size = count_a + count_b
        direction_matrix = np.zeros((3, size))
        mean_matrix = np.zeros((3, size))
        for i, axis in enumerate(dirs_a):
            direction_matrix[:, i] = axis
            mean_matrix[:, i] = 0.5 * axis
        for j, axis in enumerate(dirs_b):
            direction_matrix[:, count_a + j] = -axis
            mean_matrix[:, count_a + j] = 0.5 * axis
        weight = np.sqrt(JOINT_REGULARIZATION)
        matrix = np.vstack([direction_matrix, weight * mean_matrix])
        target = np.concatenate([
            ideal_b["center"] - ideal_a["center"],
            weight * (reference_point - 0.5 * (ideal_a["center"] + ideal_b["center"])),
        ])

        best = None
        for state in itertools.product((-1, 0, 1), repeat=size):
            state = np.asarray(state)
            fixed = state != 0
            z = np.where(fixed, state * half, 0.0)
            free = ~fixed
            if free.any():
                rhs = target - matrix[:, fixed] @ z[fixed]
                z[free] = np.linalg.lstsq(matrix[:, free], rhs, rcond=None)[0]
                if np.any(np.abs(z[free]) > half[free] + 1e-9):
                    continue
            objective = float(np.sum((matrix @ z - target) ** 2))
            if best is None or objective < best[0] - 1e-18:
                best = (objective, z)
        z = best[1]
        p_a = ideal_a["center"] + sum((z[i] * axis for i, axis in enumerate(dirs_a)), np.zeros(3))
        p_b = ideal_b["center"] + sum((z[count_a + j] * axis for j, axis in enumerate(dirs_b)), np.zeros(3))
        return p_a, p_b, float(np.linalg.norm(p_a - p_b))


    def _convex_hull_2d(self, points):
        """Konvexe Hülle (Monotone Chain) lexikografisch sortierter, eindeutiger 2D-Punkte, gegen den Uhrzeigersinn."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        pts = [tuple(p) for p in points]
        lower, upper = [], []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return np.array(lower[:-1] + upper[:-1], dtype=float)


    def _ideal_meeting_point(self, ideal_a, ideal_b, c_real, r_search):
        """
        Bestimmt c_ideal, den Treffpunkt der Referenzidealisierungen.

        Fall A: Idealisierungen schneiden sich (Abstand <= JOINT_IDEAL_TOL) -> Schwerpunkt der Schnittmenge.
        Fall B: kein Schnitt -> Mittelpunkt der kürzesten Verbindung.
        Fall C: liegt das Ergebnis weiter als r_search von c_real, werden beide Idealisierungen
                auf die Kugel um c_real (r_search) beschränkt (Rechtecke näherungsweise auf die
                umschreibende Box der Kreisfläche) und A/B darauf erneut angewandt.
                Bleibt nichts im Radius übrig, gilt Fall B global mit Flag fall_c_fallback.

        Returns:
            tuple: (c_ideal, Fall 'A'|'B'|'C', fall_c_fallback)
        """
        p_a, p_b, distance = self._closest_points(ideal_a, ideal_b, c_real)
        midpoint = 0.5 * (p_a + p_b)
        if distance <= JOINT_IDEAL_TOL:
            point, case = self._intersection_centroid(ideal_a, ideal_b, p_a, p_b), "A"
        else:
            point, case = midpoint, "B"
        if np.linalg.norm(point - c_real) <= r_search:
            return point, case, False

        restricted_a = self._restrict_to_ball(ideal_a, c_real, r_search)
        restricted_b = self._restrict_to_ball(ideal_b, c_real, r_search)
        if restricted_a is None or restricted_b is None:
            return midpoint, "B", True
        q_a, q_b, distance = self._closest_points(restricted_a, restricted_b, c_real)
        if distance <= JOINT_IDEAL_TOL:
            return self._intersection_centroid(restricted_a, restricted_b, q_a, q_b), "C", False
        return 0.5 * (q_a + q_b), "C", False


    def _ideal_intersection_polygon(self, rect_a, rect_b):
        """Schnittpolygon (Sutherland-Hodgman) zweier koplanarer Rechtecke in den Koordinaten von rect_a."""
        e1, e2 = rect_a["dirs"]
        h1, h2 = rect_a["half"]
        subject = [(-h1, -h2), (h1, -h2), (h1, h2), (-h1, h2)]
        clipper = []
        for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corner = rect_b["center"] + s1 * rect_b["half"][0] * rect_b["dirs"][0] \
                + s2 * rect_b["half"][1] * rect_b["dirs"][1]
            local = corner - rect_a["center"]
            clipper.append((float(local @ e1), float(local @ e2)))
        area = sum(
            clipper[i][0] * clipper[(i + 1) % 4][1] - clipper[(i + 1) % 4][0] * clipper[i][1]
            for i in range(4)
        )
        if area < 0:
            clipper.reverse()

        output = subject
        for i in range(len(clipper)):
            start, end = clipper[i], clipper[(i + 1) % len(clipper)]
            def inside(q):
                return (end[0] - start[0]) * (q[1] - start[1]) - (end[1] - start[1]) * (q[0] - start[0]) >= -1e-12
            def intersection(q1, q2):
                d1 = (q2[0] - q1[0], q2[1] - q1[1])
                d2 = (end[0] - start[0], end[1] - start[1])
                denominator = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(denominator) < 1e-18:
                    return q2
                t = ((start[0] - q1[0]) * d2[1] - (start[1] - q1[1]) * d2[0]) / denominator
                return (q1[0] + t * d1[0], q1[1] + t * d1[1])
            previous_points, output = output, []
            if not previous_points:
                break
            s = previous_points[-1]
            for e in previous_points:
                if inside(e):
                    if not inside(s):
                        output.append(intersection(s, e))
                    output.append(e)
                elif inside(s):
                    output.append(intersection(s, e))
                s = e
        return output


    def _intersection_centroid(self, ideal_a, ideal_b, p_a, p_b):
        """
        Schwerpunkt der Schnittmenge zweier sich schneidender Idealisierungen (Fall A):
        Punkt (nicht parallel), Überlappstrecke (kollinear), Strecke in der Ebene (geclippt),
        Schnittgerade zweier Rechtecke (geclippt) bzw. Überlappfläche koplanarer Rechtecke.
        Schlägt die Sonderfallberechnung fehl, gilt der Mittelpunkt der kürzesten Verbindung.
        """
        midpoint = 0.5 * (p_a + p_b)
        if ideal_a["dim"] == 2 and ideal_b["dim"] == 1:
            ideal_a, ideal_b = ideal_b, ideal_a

        if ideal_a["dim"] == 1 and ideal_b["dim"] == 1:
            axis_a, axis_b = ideal_a["dirs"][0], ideal_b["dirs"][0]
            if np.linalg.norm(np.cross(axis_a, axis_b)) > JOINT_PARALLEL_TOL:
                return midpoint
            shift = float((ideal_b["center"] - ideal_a["center"]) @ axis_a)
            low = max(-ideal_a["half"][0], shift - ideal_b["half"][0])
            high = min(ideal_a["half"][0], shift + ideal_b["half"][0])
            if low > high:
                return midpoint
            return ideal_a["center"] + axis_a * 0.5 * (low + high)

        if ideal_a["dim"] == 1 and ideal_b["dim"] == 2:
            axis, normal = ideal_a["dirs"][0], ideal_b["normal"]
            if abs(float(axis @ normal)) > JOINT_PARALLEL_TOL:
                return midpoint
            if abs(float((ideal_a["center"] - ideal_b["center"]) @ normal)) > JOINT_IDEAL_TOL:
                return midpoint
            span = self._clip_line_to_rectangle(
                ideal_a["center"], axis, ideal_b, (-ideal_a["half"][0], ideal_a["half"][0])
            )
            return midpoint if span is None else ideal_a["center"] + axis * 0.5 * (span[0] + span[1])

        normal_a, normal_b = ideal_a["normal"], ideal_b["normal"]
        line = np.cross(normal_a, normal_b)
        sine = float(np.linalg.norm(line))
        if sine > JOINT_PARALLEL_TOL:
            line = line / sine
            system = np.vstack([normal_a, normal_b, line])
            rhs = np.array([
                normal_a @ ideal_a["center"], normal_b @ ideal_b["center"], line @ midpoint
            ])
            origin = np.linalg.solve(system, rhs)
            span_a = self._clip_line_to_rectangle(origin, line, ideal_a, (-np.inf, np.inf))
            span_b = self._clip_line_to_rectangle(origin, line, ideal_b, (-np.inf, np.inf))
            if span_a is None or span_b is None:
                return midpoint
            low, high = max(span_a[0], span_b[0]), min(span_a[1], span_b[1])
            return midpoint if low > high else origin + line * 0.5 * (low + high)

        if abs(float((ideal_b["center"] - ideal_a["center"]) @ normal_a)) > JOINT_IDEAL_TOL:
            return midpoint
        polygon = self._ideal_intersection_polygon(ideal_a, ideal_b)
        if len(polygon) < 3:
            return midpoint
        xs = np.array([q[0] for q in polygon])
        ys = np.array([q[1] for q in polygon])
        xn, yn = np.roll(xs, -1), np.roll(ys, -1)
        cross = xs * yn - xn * ys
        area = 0.5 * float(cross.sum())
        if abs(area) < 1e-12:
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            cx = float(((xs + xn) * cross).sum() / (6.0 * area))
            cy = float(((ys + yn) * cross).sum() / (6.0 * area))
        return ideal_a["center"] + cx * ideal_a["dirs"][0] + cy * ideal_a["dirs"][1]


    def _min_area_rectangle(self, points):
        """
        Minimal umschließendes Rechteck einer 2D-Punktwolke (eine Rechteckseite liegt auf
        einer Kante der konvexen Hülle).

        Returns:
            tuple: (center (2,), axes (2, 2) als Zeilen, half (2,) absteigend sortiert)
                bzw. (None, None, None) bei entarteter Punktwolke.
        """
        unique = np.unique(np.round(points, 6), axis=0)
        if len(unique) < 3:
            return None, None, None
        hull = self._convex_hull_2d(unique)
        if len(hull) < 3:
            return None, None, None
        best = None
        for i in range(len(hull)):
            edge = hull[(i + 1) % len(hull)] - hull[i]
            length = np.linalg.norm(edge)
            if length < 1e-12:
                continue
            e = edge / length
            f = np.array([-e[1], e[0]])
            pe, pf = hull @ e, hull @ f
            area = (pe.max() - pe.min()) * (pf.max() - pf.min())
            if best is None or area < best[0] - 1e-12:
                best = (area, e, f, pe.min(), pe.max(), pf.min(), pf.max())
        if best is None or best[0] <= 1e-12:
            return None, None, None
        _, e, f, e_lo, e_hi, f_lo, f_hi = best
        center = e * 0.5 * (e_lo + e_hi) + f * 0.5 * (f_lo + f_hi)
        half = np.array([0.5 * (e_hi - e_lo), 0.5 * (f_hi - f_lo)])
        axes = np.array([e, f])
        order = np.argsort(half)[::-1]
        return center, axes[order], half[order]


    def _offset_class(self, o_vector, o_length, o_norm, reference):
        """
        Klassifiziert den Versatzcharakter O0-O3 (Reihenfolge verbindlich):
        Õ < JOINT_O_TOL -> O0 (ρ wird nicht berechnet); sonst ρ = |o_O2|/|o|:
        ρ < JOINT_RHO_AXIAL -> O1, ρ > JOINT_RHO_LATERAL -> O2, sonst O3.

        Returns:
            tuple: (Klasse, ρ | None, |o_O1|, |o_O2|)
        """
        if o_norm < JOINT_O_TOL:
            return "O0", None, None, None
        if reference["dim"] == 1:
            axis = reference["dirs"][0]
            o_axial = (o_vector @ axis) * axis
            o_lateral = o_vector - o_axial
        else:
            normal = reference["normal"]
            o_lateral = (o_vector @ normal) * normal
            o_axial = o_vector - o_lateral
        rho = float(np.linalg.norm(o_lateral) / o_length)
        if rho < JOINT_RHO_AXIAL:
            name = "O1"
        elif rho > JOINT_RHO_LATERAL:
            name = "O2"
        else:
            name = "O3"
        return name, rho, float(np.linalg.norm(o_axial)), float(np.linalg.norm(o_lateral))


    def _pair_angle(self, ideal_a, ideal_b):
        """Spitzer Winkel in [0°, 90°]: Achsen (1D–1D), Normalen (2D–2D) bzw. Gerade/Ebene (1D–2D)."""
        if ideal_a["dim"] == 1 and ideal_b["dim"] == 1:
            cosine = abs(float(ideal_a["dirs"][0] @ ideal_b["dirs"][0]))
            return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
        if ideal_a["dim"] == 2 and ideal_b["dim"] == 2:
            cosine = abs(float(ideal_a["normal"] @ ideal_b["normal"]))
            return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
        line, plane = (ideal_a, ideal_b) if ideal_a["dim"] == 1 else (ideal_b, ideal_a)
        sine = abs(float(line["dirs"][0] @ plane["normal"]))
        return float(np.degrees(np.arcsin(np.clip(sine, 0.0, 1.0))))


    def _position_class(self, ideal, point, r_end):
        """'Rand' (Ende bzw. Rand der Idealisierung innerhalb r_end) oder 'innen' für die Lage von point."""
        margins = [
            max(half - abs(float((point - ideal["center"]) @ axis)), 0.0)
            for axis, half in zip(ideal["dirs"], ideal["half"])
        ]
        return "Rand" if min(margins) <= r_end else "innen"


    def _reference_idealization(self, node):
        """
        Referenzidealisierung Ŝ_v: feste, konverterunabhängige Regel ohne Verlängerung/Verschiebung.
        Aus den Vertices wird per PCA ein orientiertes Hüllvolumen (OBB) bestimmt.
          linear: Strecke entlang der Hauptachse u durch den Mittelpunkt des Querschnitts (Min-Area-Rechteck).
          planar: Rechteck senkrecht zur kleinsten Hauptrichtung n; Seiten per Min-Area-Rechteck.
        Entartet (zu wenige Vertices, Querschnitt/Fläche ohne Ausdehnung, Achse nicht eindeutig
        weil axiale Länge < Querschnittsmaß, Platte nicht dünn) -> keine Idealisierung.

        Args:
            node (dict): IFC-Knoten mit 'geom_raw', 'entity_processed', 'dimensions'.

        Returns:
            tuple: (dict | None, Grund | None). Dict: guid, category, dim (1|2), t, center,
                dirs (Richtungen der Aufspannung), half (Halbausdehnungen), normal (nur planar).
        """
        category, thickness, reason = self._element_category_and_thickness(node)
        if category is None or thickness is None:
            return None, reason
        try:
            vertices = np.asarray((node.get("geom_raw") or {})["triangles"], dtype=float).reshape(-1, 3)
        except (KeyError, TypeError, ValueError):
            return None, "Dreiecksgeometrie fehlt oder ist ungültig"
        vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
        vertices = np.unique(np.round(vertices, 6), axis=0)
        if len(vertices) < 4:
            return None, "zu wenige Vertices für die Idealisierung"

        mean = vertices.mean(axis=0)
        centered = vertices - mean
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

        if category == "linear":
            axis = eigenvectors[:, 0]
            basis = eigenvectors[:, 1:3]
            axial = centered @ axis
            low, high = float(axial.min()), float(axial.max())
            center2, _, half2 = self._min_area_rectangle(centered @ basis)
            if center2 is None:
                return None, "Querschnitt ist entartet"
            if (high - low) < 2.0 * float(half2[0]):
                return None, "Bauteilachse nicht eindeutig (axiale Länge kleiner als Querschnittsmaß)"
            center = mean + axis * 0.5 * (low + high) + basis @ center2
            return {
                "guid": node.get("guid"), "category": category, "dim": 1, "t": thickness,
                "center": center, "dirs": [axis], "half": [0.5 * (high - low)], "normal": None,
            }, None

        normal = eigenvectors[:, 2]
        basis = eigenvectors[:, 0:2]
        center2, axes2, half2 = self._min_area_rectangle(centered @ basis)
        if center2 is None:
            return None, "Fläche ist entartet"
        offset = centered @ normal
        low, high = float(offset.min()), float(offset.max())
        if (high - low) >= 2.0 * float(half2[1]):
            return None, "Bauteil ist nicht plattenförmig (Dicke nicht kleiner als kürzeste Seite)"
        center = mean + normal * 0.5 * (low + high) + basis @ center2
        return {
            "guid": node.get("guid"), "category": category, "dim": 2, "t": thickness,
            "center": center, "dirs": [basis @ axes2[0], basis @ axes2[1]],
            "half": [float(half2[0]), float(half2[1])], "normal": normal,
        }, None


    def _restrict_to_ball(self, ideal, center, radius):
        """
        Beschränkt eine Idealisierung auf die Kugel um center. Strecken exakt; Rechtecke
        auf die umschreibende Box der Schnittkreisfläche. Leer -> None.
        """
        local = center - ideal["center"]
        if ideal["dim"] == 1:
            axis = ideal["dirs"][0]
            along = float(local @ axis)
            perpendicular_sq = float(local @ local) - along ** 2
            if perpendicular_sq > radius ** 2:
                return None
            reach = float(np.sqrt(max(radius ** 2 - perpendicular_sq, 0.0)))
            low = max(-ideal["half"][0], along - reach)
            high = min(ideal["half"][0], along + reach)
            if low > high:
                return None
            restricted = dict(ideal)
            restricted["center"] = ideal["center"] + axis * 0.5 * (low + high)
            restricted["half"] = [0.5 * (high - low)]
            return restricted
        height = float(local @ ideal["normal"])
        if abs(height) > radius:
            return None
        reach = float(np.sqrt(max(radius ** 2 - height ** 2, 0.0)))
        shift = np.zeros(3)
        halves = []
        for axis, half in zip(ideal["dirs"], ideal["half"]):
            along = float(local @ axis)
            low, high = max(-half, along - reach), min(half, along + reach)
            if low > high:
                return None
            shift = shift + axis * 0.5 * (low + high)
            halves.append(0.5 * (high - low))
        restricted = dict(ideal)
        restricted["center"] = ideal["center"] + shift
        restricted["half"] = halves
        return restricted





class Experiment1RBCFehlertaxonomie(
    Experiment1PhaseASchritt1, Experiment1PhaseASchritt2, Experiment1PhaseBSchritt3
):
    """Fasst die implementierten Schritte von Experiment 1 als Pipeline-Mixin zusammen."""

    def _run_epsilon_study(self, nodes, gt_pairs, epsilons=None):
        """
        Sensitivitätsstudie: Führt Phase A (Schritt 1 und 2) und Phase B je eps neu aus,
        da eps Kontaktregionen und Joint-Clustering beeinflusst.

        Args:
            nodes (list[dict]): IFC-Knoten.
            gt_pairs (set[tuple]): Ground-Truth-Paare.
            epsilons (iterable[float] | None): Toleranzen in [m]; None = JOINT_EPSILON_STUDY.

        Returns:
            dict[float, dict]: eps -> {'contact_regions', 'soll_joints', 'joint_classes'}.
        """
        results = {}
        for eps in (JOINT_EPSILON_STUDY if epsilons is None else epsilons):
            self.log(f"{'='*10} 1.B eps-Studie: eps = {eps*1000:.1f} mm {'='*10}")
            regions = self._compute_contact_regions(nodes, gt_pairs, epsilon=eps)
            joints = self._build_soll_joints(nodes, gt_pairs, regions)
            classes = self._classify_soll_joints(nodes, gt_pairs, regions, joints, epsilon=eps)
            results[float(eps)] = {
                "contact_regions": regions, "soll_joints": joints, "joint_classes": classes
            }
        return results


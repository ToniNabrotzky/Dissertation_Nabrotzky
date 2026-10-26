Einleitung:
Ein GNN-Modell soll über Kantenklassifikation den statischen Lastfluss in einem Gebäude voraussagen. In diesem Graphen repräsentieren die Bauteile die Knoten und die Kanten (als topologische Beziehungen) die statischen Verbindungen jeweils mit Knoten- bzw. Kantenmerkmalen.

Aufgabe Kantenmerkmale:
Überlege und entwickle mathematische Beschreibungen, mit denen man die topologischen Abhängigkeiten zwischen Bauteilen beschreiben kann. Diese Informationen sollen an den jeweiligen Kanten zwischen den Bauteilen gespeichert werden. Als Inspiration findest du folgend Regeln für die manuelle, expertengestützte Definition des Lastflusses. Greife diese Ideen auf und ergänze sie um eigene Ansätze. Mache kenntlich, welche Idee übernommen und welche von dir selbst entwickelt ist.

| Symbol | Bedeutung |
|---|---|
| $\mathbf{p}_s, \mathbf{p}_e \in \mathbb{R}^3$ | Start-/Endknoten eines Stabes |
| $\mathbf{d} = \mathbf{p}_e - \mathbf{p}_s$ | Richtungsvektor (unnormiert) |
| $\hat{\mathbf{d}} = \mathbf{d}/\lVert\mathbf{d}\rVert$ | Einheits-Richtungsvektor |
| $L = \lVert\mathbf{d}\rVert$ | Stablänge |
| $\mathcal{P} = \{\mathbf{v}_1,\dots,\mathbf{v}_m\}$ | Eckpunkte eines 2D-Elements (Polygon) |
| $\hat{\mathbf{n}}$ | Einheits-Normale einer Fläche |
| $\mathbf{c}$ | Schwerpunkt/Zentroid |
| $A$ | Flächeninhalt |
| $\mathbf{B} = [\mathbf{b}_{\min}, \mathbf{b}_{\max}]$ | Achsparallele Bounding-Box (AABB) |
| $t$ | Bauteildicke bzw. Querschnittsabmessung |




**Start Anhang**

Bauteilübersicht:
1D-Elemente (Stab): Stütze, Balken, Verband
2D-Elemente (Fläche): Decke, Wand


Teil 1: Richtlinien für die CAD-Modellierung (volumetrische Elemente)
Präambel - Annahmen:
- Dieser Teil enthält Regeln, unter welchen Bedingungen Bauteile für die Beschriftung als miteinander verbunden/angeschlossen gelten.
	Schreibweise für Anschlussketten: A-B-C. Jeder "-" trennt einzelne Kettenglieder ab. Gibt es verschiedene Optionen, werden diese pro Kettenglied mit "/" aufgeführt, z.B. Stütze-Decke-Stütze/Wand.
- Jedes Bauteil (Element A) wird einzeln durchlaufen, um dessen benachbarte Bauteile (Kandidaten B) anhand spezifischer Regeln zu überprüfen.
- Ein korrekter Lastfluss findet prinzipiell über die folgenden Wege statt:
	- *Horizontal* zur Aussteifung des Gebäudes oder zur Umleitung von Lasten und 
	- *Vertikal* aufgrund der Schwerkraft bzw. durch beide gleichzeitig bei schrägen Elementen.
- Bauteile verfügen über mehrere relevante Ebenen:
	- Bei 1D-Stabelementen gibt es eine *Hauptachse* entlang der Systemlinie (meist der Schwerpunkt), die als *Richtungsvektor* bezeichnet wird.
	- Bei 2D-Flächenelementen wird eine *Hauptebene* parallel zur Systemfläche aufgespannt (meist durch den Schwerpunkt). Die stärkere Ausdehnung gibt dabei die *Primärachse*, die kürzere Ausdehnung die *Sekundärachse* vor. Die dritte Achse (*Tertiärachse*) zeigt als Normale aus der Ebene heraus. 
- Eine wesentliche Rolle spielt die *konvexe Hülle* der Bauteile, die aus den äußeren Teilflächen eines Elements besteht. Bei diesen wird je nach Neigungsgrad unterschieden zwischen:
	- *Grundflächen* (horizontal),
	- *Mantelflächen* (vertikal) und
	- *Schrägflächen* (sonstige).
- Eine *Projektion* der einzelnen Teilflächen von Element A *entlang ihrer Normalen* "nach außen" ist *Hauptindikator*, ob bzw. wie stark ausgeprägt die nachfolgenden Regeln erfüllt sind.
- Damit ein Anschluss als positiv bewertet werden kann, muss die Projektion von Element A einen Kandidaten B *echt schneiden*. Das heißt, die Projektion muss *teilweise oder komplett in das Nachbarobjekt eindringen* und ein Volumen als internen Schnittraum aufspannen. Hinweis: Bei idealisierten Elementen wäre das gleichbedeutend eine Schnittlinie bzw. -fläche.
	- Echte Schnitte treten in der Regel auf, wenn sich beide Bauteile bereits ohne Projektion berühren (euklidische Distanz = 0) oder wenn sie sich innerhalb eines definierten Suchradius' befinden.
- Wenn die Projektion den Kandidaten B echt schneidet gelten beide Bauteile als *direkt angeschlossen*. Eine *Nebenvoraussetzung* für bestimmte Anschlusskonstellationen ist allerdings, dass sich *kein weiteres drittes Bauteil zwsichen den beiden* (also innerhalb der Projektion) befindet. Falls doch, so ist die aktuelle Verbindung als *indirekt angeschlossen* einzustufen.


1D-Bauteil veritkal (Stütze):
- Stützen dienen primär dem vertikalen Lastabtrag von oberen Elementen in das Fundament. Sie werden hauptsächlich auf Druck (Normalkraft) beansprucht.
- Projektion in Richtungsvektor (meist Grundflächen). 
	- Anschluss mit Decke: Stützen werden direkt an Decken ober- oder unterhalb angeschlossen.
	- Typisch direkte Anschlüsse: Stütze-Decke/Stütze/Wand/Balken/Verband
	- Anschluss mit Stütze/Wand: Es gilt die Nebenvoraussetzung für direkte Verbindungen. 
	- Typisch indirekte Anschlüsse: Stütze-Decke-Stütze/Wand
- keine Projektion senkrecht zu Richtungsvektor (meist Mantelflächen).


1D-Bauteil horizontal (Balken):
- Balken dienen primär dem horizontalen Lastabtrag, um Kräfte umzuleiten. Sie nehmen Kräfte quer zu ihrer Achse auf (Z.B. von Deckenplatten) und werden daher auf Biegung und Querkraft beansprucht.
- Projektion senkrecht zu Richtungsvektor (meist Grundflächen).
	- Anschluss mit Decke: Balken werden an Decken angeschlossen. Diese wirken dann als Unterzug.
	- Typisch direkte Anschlüsse: Balken-Decke/Verband
- Projektion in Richtungsvektor (meist Mantelflächen).
	- Anschluss mit Stäben: Balken werden an benachbarte oder kreuzende Balken, Stützen, Verbände angeschlossen.
	- Anschluss mit Wand: Balken werden an Wände angeschlossen. Dies  kann als T-Stoß, L-Stoß oder fluchtend (in der Hauptebene der Wand) erfolgen.
	- Typisch direkte Anschlüsse: Balken-Balken/Stütze/Verband/Wand
	

1D-Bauteil schräg (Verband):
- Verbände dienen primär der räumlichen Stabilität und Aussteifung des Bauwerkes. Sie nehmen horizontale Kräfte (z.B. Wind) und wirken meist als reine Druck- oder Zugstreben.
- Projektion in Richtungsvektor (meist Mantelflächen oder Schrägflächen).
	- Anschluss mit Stäben: Verbände werden an benachbarte oder kreuzende Balken, Stützen, Verbände angeschlossen.
	- Typisch direkte Anschlüsse: Verband-Balken/Stütze/Verband


2D-Bauteil vertikal (Wand):
- Wände dienen zum Lastabtrag als tragende Wände. Sie nehmen vertikale Lasten in Richtung der Wandebene auf und werden auf Druck beansprucht.
- Wände dienen ebenfalls als aussteifende Wände. Sie nehmen horizontale Lasten (z.B. Wind) in Richtung der Wandebene als Schubwände/Wandscheiben auf, wenn sie parallel zur Windrichtung stehen.
- Projektion in vertikaler Achse der Hauptebene (meist Grundflächen; bei langen Wänden Sekundärachse, bei kurzen Wänden Primärachse).
	- Anschluss mit Decke: Wände werden direkt an Decken ober- oder unterhalb angeschlossen.
	- Typisch direkte Anschlüsse: Wand-Decke/Wand/Stütze
	- Anschluss mit Wand/Stütze: Es gilt die Nebenvoraussetzung für direkte Verbindungen.
	- Typisch indirekte Anschlüsse: Wand-Decke-Wand/Stütze
- Projektion in horizontaler Achse der Hauptebene (meist Mantelflächen; bei langen Wänden Primärachse, bei kurzen Wänden Sekundärachse).
	- Anschluss mit Wänden/Balken: Wände werden an Wände oder Balken angeschlossen. Dies  kann als T-Stoß, L-Stoß oder fluchtend (in der Hauptebene der Wand) erfolgen.
	- Typisch direkte Anschlüsse: Wand-Wand/Balken


2D-Bauteil horizontal (Decke):
- Decken nutzen ihre Plattenwirkung, um Vertikallasten abzutragen, die senkrecht zur senkrecht zur Deckenfläche wirken. Sie wird dadurch auf Biegung und Querkraft beansprucht.
- Decken nutzen ihre Scheibenwirkung, um Horizontallasten abzutragen, die in der Deckenebene wirken. Sie werden deshalb hauptsächlich auf Druck beansprucht.
- Projektion in Tertiärachse (meist Grundflächen).
	- Anschluss mit Wand/Stütze/Balken: Decken werden direkt an Wände/Stütze/Balken oder- oder unterhalb angeschlossen.
	- Typisch direkte Anschlüsse: Decke-Wand/Stütze/Balken
- Projektion in Hauptebene (meist Mantelflächen; Primär- und Sekundärachse).
	- Anschluss mit Decke: Decken werden an benachbarte Decken angeschlossen.



Teil 2: Algorithmische und topologische Logik für Analysemodelle (idealisierte Elemente)
Präambel - Annahmen:
- Dieser Teil enthält Regeln und Prüfbedingungen, wie idealisierte Elemente miteinander verbunden werden sollen.
	- Schreibweise für Anschlussketten: A-B-C. Jeder "-" trennt einzelne Kettenglieder ab. Gibt es verschiedene Optionen, werden diese pro Kettenglied mit "/" aufgeführt, z.B. Stütze-Decke-Stütze/Wand.
- Jedes Bauteil (Element A) wird einzeln durchlaufen, um dessen benachbarte Bauteile (Kandidaten B) anhand spezifischer Regeln zu überprüfen.
- Ein korrekter Lastfluss findet prinzipiell über die folgenden Wege statt:
	- *Horizontal* zur Aussteifung des Gebäudes oder zur Umleitung von Lasten und 
	- *Vertikal* aufgrund der Schwerkraft bzw. durch beide gleichzeitig bei schrägen Elementen.
- Eine wesentliche Rolle spielt der *Richtungsvektor* der Bauteile. Bei 1D-Elementen entspricht dieser der Systemlinie; bei 2D-ELementen entspricht dieser den beiden Achsen der Systemfläche.
	- Die erforderliche Projektion (Verlängerung) dieses Vektors ist *Hauptindikator*, ob bzw. wie stark ausgeprägt die nachfolgenden Regeln erfüllt sind.
- Damit ein Anschluss als positiv bewertet werden kann, muss die Projektion vom Richtungsvektor den Richtungsvektor eines Kandidaten B *echt schneiden*. Das heißt, die Projektion muss entweder *exakt auf oder hinter dem Zielobjekt enden* und eine Linie als interne Schnittlinie aufspannen. Echte Schnitte treten in der Regel auf, wenn sich beide Bauteile bereits ohne Projektion berühren (euklidische Distanz = 0) oder wenn sie sich innerhalb eines definierten Suchradius' befinden.
- Wenn die Projektion den Kandidaten B echt schneidet gelten beide Bauteile als *direkt angeschlossen*. Eine *Nebenvoraussetzung* für bestimmte Anschlusskonstellationen ist allerdings, dass sich *kein weiteres drittes Bauteil zwsichen den beiden* (also innerhalb der Projektion) befindet. Falls doch, so ist die aktuelle Verbindung als *indirekt angeschlossen* einzustufen.


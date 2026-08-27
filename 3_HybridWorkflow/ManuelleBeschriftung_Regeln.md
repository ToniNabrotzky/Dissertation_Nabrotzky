Regeln für die manuelle Modellkonvertierung
Teil 1: Richtlinien der CAD-Modellierung (volumetrische Elemente)
Diese Regeln definieren, wie Bauteile im CAD-System geometrisch platziert werden müssen, um einen korrekten Lastfluss zu gewährleisten.
	1D-Bauteil vertikal (Stütze):
	Vertikaler Anschluss (Decke): Stützen werden an Decken direkt ober- oder unterhalb angeschlossen.
	Vertikaler Anschluss (Stütze): Stützen können direkt an darüber- oder darunterliegende Stützen angeschlossen werden (euklidische Distanz = 0), sofern sich keine anderen Bauteile (z.B. Geschossdecken) dazwischen befinden, in solchen Fällen erfolgt der Lastfluss indirekt über diese Elemente.
	Positionsstabilität: Stützen dürfen nicht horizontal an Wände verschoben oder („bündig“) an diese gesnappt werden. Befindet sich eine Wand exakt über oder unter der Stütze, ist diese Verbindung zulässig.
	Stab-Anschluss: Stützen dürfen an Balken oder Verbände angeschlossen werden. Die Systemachse des anschließenden Stabes muss dabei in Richtung des Start- oder Endknotens der Stütze zeigen bzw. in Richtung der (kurz) verlängerten Achse des ursprünglichen Stabes.
	1D-Bauteil (horizontal):
	Stab-Anschlüsse: Balken werden an benachbarte oder kreuzende Balken, Stützen oder Verbände angeschlossen.
	Wand-Anschluss: Balken können an Wände angeschlossen werden. Dies kann als T-Stoß, L-Stoß oder fluchtend (in der Ebene der Wand) erfolgen.
	Decken-Anschluss (Unterzug): Balken werden an Decken angeschlossen, wenn sie als Unterzug wirken. Dafür muss die Systemachse des Balkens direkt unterhalb der Decke verlaufen.
	1-D Bauteil schräg (Verband):
	Stab-Anschluss: Verbände werden an die Knotenpunkte oder Achsen (verlängert oder innerhalb) durchkreuzender 1D-Bauteil (Stützen, Balken, andere Verbände) angeschlossen.
	2D Bauteil vertikal (Wand):
	Horizontale Verbindung (Wand-Wand): Wände stoßen über ihre vertikalen Flächen aneinander. Dies ist als T-Stoß, L-Stoß oder fluchtend (als direkte Fortsetzung z.B. zur Raumabgrenzung) zulässig.
	Vertikale Verbindung (Wand-Decke/Wand): Wände werden über ihre horizontalen Flächen direkt an darüber- oder darunterliegenden Decken angeschlossen. Es sind auch direkte vertikale Wand-Wand-Verbindungen wie bei den Stützen zulässig, wenn keine Bauteile dazwischen liegen.
	2D Bauteil horizontal Decke):
	Horizontale Verbindungen: Decken werden entlang ihrer Ränder (vertikale Flächen) mit angrenzenden Decken verbunden.
	Vertikale Verbindungen: Stützen, Balken, Verbände und Wände, die sich innerhalb der Projektion der Grundrissfläche der Decken befinden, werden an die Decke angeschlossen. Bei Verbänden muss zusätzlich die Achse die Deckenfläche die Deckenfläche durchstoßen.

Teil 2: Algorithmische und topologische Logik (idealisierte Elemente)
Diese Regeln definieren die Prüfbedingungen für das automatische Verbinden oder die Fehlererkennung in Skripten und Graph-Modellen.
Definition eines „echten Schnitts“ (Überschneidung, Durchstoßen): Zwei Flächen A und B schneiden sich echt, wenn ihre Schnittmenge I=A∩B eine Linie ist und I nicht ausschließlich aus den Begrenzungslinien von A oder B besteht. Eine reine Kantenberührung (euklidische Distanz = 0, aber keine Durchdringung der projizierten Flächen) wird als falsch gewertet. Zwischen Stab und Fläche ist die Schnittmenge I=A∩B ein Punkt, wobei dieser kein Start- oder Endpunkt des Stabes sein darf.
	1D-Bauteil vertikal (Stütze):
	Bedingung Decke: Eine topologische Verbindung zur Decke ist gültig, wenn die Verlängerung des Richtungsvektors der Stütze die Deckenfläche durchstößt.
	Bedingung Stab: Eine Verbindung zu einem anderen 1D-Bauteilist gültig, wenn die Verlängerung des eigenen Richtungsvektors die Verlängerung des Richtungsvektors des Zielbauteils schneidet.
	Ausschluss: Es darf keine Kanten- oder Knotenkopplung (Node-/Edge-Link) generiert werden, die den Vektor der Stütze horizontal zu einer Wandgeometrie translatiert.
	1D-Bauteil horizontal (Balken):
	Bedingung Stab: Eine Verbindung ist gültig, wenn die Verlängerung des eigenen Richtungsvektors die Verlängerung des Richtungsvektors des Zielbauteils schneidet.
	Bedingung Wand: Eine topologische Verbindung existiert, wenn die Verlängerung des Richtungsvektors des Balkens eine Wandfläche durchstößt.
	Bedingung Decke (Unterzug): Ein Balken ist topologisch mit einer Decke verbunden, wenn er das nach unten extrudierte 3D-Volumen der Deckenfläche durchstößt UND die Z-Koordinate der Balkenachse kleine oder gleich der Z-Koordinate der Deckenmittelfläche ist (Z_Balken≤Z_Decke). Die horizontale Distanz zur Deckenprojektion ist in diesem Fall 0.
	1D-Bauteil schräg (Verband):
	Bedingung Stab: Eine Verbindung zu anderen 1D-Bauteilen ist gültig, wenn die Verlängerung des eigenen Richtungsvektors die Verlängerung des Richtungsvektors des Zielbauteils schneidet.
	Bedingung Fläche: Eine Verbindung zu Flächen (Wand/Decke) ist gültig, wenn die Verlängerung des eigenen Richtungsvektors die Fläche durchstößt.
	2D-Bauteil vertikal (Wand): 
	Horizontale Projektion (Wand-Wand): Die vertikalen Begrenzungsflächen der Wand werden entlang der Wand-Längsachse im Raum verschoben. Ein Anschluss ist gültig, wenn diese Prüffläche des eigenen Bauteils die analog verlängerte Prüffläche des Zielbauteils echt schneidet (notwendig zur Erfassung von L-Stößen).
	Vertikale Projektion (Wand-Decke/Wand): Die horizontalen Begrenzungslinien der Wand werden virtuell in positiver und negativer Z-richtung zu einer Prüffläche verlängert. Ein Anschluss ist gültig, wenn die Prüffläche der eigenen Wand die Fläche der Decke oder der Ziel-Wand echt schneidet.
	Topologischer Ausschluss: Im Graphen darf keine direkte Kante zwischen Wand W_i in Ebene Z und Wand W_j in Ebene Z±1 existieren, wenn dadurch andere Bauteile (z.B. eine Geschossdecke) echt geschnitten wird (und sich deren horizontale Grundrissprojektionen im Raum nicht überlappen).
	2D-Bauteil horizontal (Decke):
	Bedingung Decke: Die Begrenzungslinien der eigenen Decke und der Ziel-Decke werden in ihrer jeweiligen Ebene „nach außen“ verschoben (größer skaliert). Ein Anschluss ist gültig, wenn sich diese neu aufgespannten, vergrößerten Flächen echt schneiden und die jeweiligen Z-Koordinaten einen bestimmten Grenzwert nicht weit nicht auseinanderliegen (dies deckt auch Decken mit unterschiedlichen Dicken/Höhenlagen ab).
	Bedingung vertikale Bauteile: Die Deckenfläche wird entlang ihrer Normalen (beidseitig in Z) zu einem Volumenkörper extrudiert. Ein Anschluss zu Wänden, Stützen oder Balken ist gültig, wenn diese das Volumen durchstoßen.

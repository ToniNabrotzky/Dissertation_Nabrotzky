### Import external modules
import logging
# import json
# import os
# import random
# import sys
from pathlib import Path

### Import local modules
import Parametric_Modelling

### SetUp Logging --- filename muss kompletter lokaler Pfad sein, sonst landet es im cwd, was ganz woanders liegen kann.
log = logging.getLogger(__name__)
script_path = Path(__file__).resolve()
script_folder = script_path.parent
log_file_name = "Porotokoll_ParametrischesModellieren.log"
log_file_path = Path(script_folder / log_file_name)

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')  # INFO or DEBUG
# logging.basicConfig(filename= log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')  # INFO or DEBUG


def main():
    log.info("___Intitialisiere Skript___")
    # """Start Tests"""
    # log.info(f"Start - Tests")
    # ## Teste Parameter-Import
    # log.info(Parametric_Modelling.Parameter.teste_method("Test"))

    # log.info(f"Ende - Test")
    # """Ende Tests"""



    """1. ENTWURFSPARAMETER - Start"""
    log.info(f"Start - Entwurfsparameter")
    ### Definiere Parameterwertebereich
    Parameter = Parametric_Modelling.Parameter   
    value_domain = Parameter.Parameterwertbereich_Test
    log.info(f"Parameterwertebereich: {value_domain}")

    ### Erstelle Parameterraum aus Parameterkombinationen
    parameter_space = Parameter.generate_parameter_combinations(value_domain)
    
    ## Protokolliere Ergebnisse
    len_parameter_space = len(parameter_space)
    log.info(f"Parameterraum mit einer Dimensionalität von {len_parameter_space}:")
    for parameter_set in parameter_space[:int(len_parameter_space)]:
        log.info(f"{parameter_set}")
    """1. ENTWURFSPARAMETER - Ende"""
    

    
    """2. BAUWERKSGEOMETRIE - Start"""



    """2. BAUWERKSGEOMETRIE - Ende"""
    log.info("___Skript erfolgreich beendet___")
    




"""Schritte zum parametrischen Modellieren:
Glossar:
    - Parameter: Eine frei definierbare Größe, die Eigenschaften der Geometrie oder des Modells beeinflusst
        o Englisch: Parameter
        o Beispiel: Anzahl der Gitterfelder in X-Richtung, Feldgröße, Geschosshöhe, Plattendicke.
    - Parameterwertbereich / Parameterdomäne: Der erlaubte Wertebereich oder die erlaubte Werteauswahl eines einzelnen Parameters. Domäne ist stärker mathematisch geprägt.
        o Englisch: Parameter Domain / Value Domain
        o Beispiel: Grid_X = {2,3,4}, Floor_Count = {1,2,3}
    - Parameterkombination - Parameterkonfiguration: Eine konkrete Zusammenstellung von Parameterwerten, die eine einzelne Konfiguration innerhalb des Parameterraums definiert.
                           - Parametersatz: Jede Parameterkombination beschreibt einen individuellen Parameteratz, der zur Erzeugung einer spezifischen Geometrieinstanz verwendet wird.
        o Englisch: Geometry Instance
        o Beispiel: (Grid_X=2, Grid_Y=3, Floor_Count=2) ist ein Parametersatz.
    - Parameterraum: Die Gesamtheit aller möglichen Kombinationen der definierten Parameterwerte. (Der Parameterraum umfasst alle Kombinationen der definierten Entwurfsparameter)
        o Englisch: Parameter Space / Design Space
        o Beispiel: Wenn Grid_X = {2,3,4}, Grid_Y = {1,2,3}, und Floor_Count = {1,2}, ergibt sich ein Parameterraum mit insgesamt 18 möglichen Kombinationen.
    - Parameterdimension: Die Anzahl an Parametern, die ein System beschreibt.
        o Englisch: Parameter Dimension / Dimensionality of Parameter Space
        o Beipsiel: 5 Parameter -> 5-dimensionale Parameterräume
    - Geometrieinstanz: Das aus einer Parameterkombination abgeleitete, spezifische virtuelle Gebäudemodell. Jede Geometrieinstanz besitzt individuelle Abmessungen, Anzahl der Bauteile und Positionen.
        o Englisch: Geometry Instance
    - Modellraum: Die Gesamtheit aller Geometrieinstanzen, die aus dem Parameterraum erzeugt wurden. (Die Gesamtheit aller Geometrieinstanzen formt den Modellraum)
        o Englisch: Model Space / Instance Space
        o Beispiel: Wenn der Parameterraum 20 Kombinationen umfasst und für jede ein IFC-Modell erzeugt wird, besteht der Modellraum aus 20 Geometrieinstanzen.
    - Fehlerfall / Störungskombination: Eine absichtlich veränderte Parameter- oder Geometriekonfiguration zur Simulation von fehlerhaften Bauteilen oder Konstruktionen
        o Englisch: Fault Case / Failure Scenario / Pertubation Configuration
        o Englisch: falsche Stützenhöhe oder abweichende Deckenplattenmaße.


1. Entwurfsparameter 
    - Definiere Entwurfsparameter (z.B. Anzahl und Größe der Felder in x und y, Geschossanzahl und -höhe, vorhandene Bauteile, etc.) und weise erlaubten Wertebereich zu (konkrete Werte)
    - Erstelle Parameterkombinationen (Parameterkonfiguration als Betonung der konkreten Zusammensetzung | Parametersatz als Stichprobe)


2. Bauwerksgeometrie
    Class BuildingGeometry:
    - Erstelle Gitterstruktur
        o mit Offset und Rotation
        o Speichere die einzelnen Gitterpunkte als einzelne Objekte ab
        o Merkmale: ID (String mit Geschossnummer und je 2 Ziffern z.B. "000", "100"), Position (x,y,z), Positionstyp (Ecke, Rand, Innen)
    - Erstelle Bauteile (Gründung (Bodenplatte + Streifenfundament), Geschossdecken (Slabs), Stützen, Wände, Balken) mit notwendigen Attributen und Merkmalen
        o Attribute: (ID?), Name, Klasse/Typ, Position, Geschoss (als Index)
        o Merkmale Slab: Abmessungen (in x und y + Höhe/Breite der Stützenprofile), Dicke, Material, Is_Foundation (Boolean), Fehlerattribute (modify_x+/-, modify_y+/-, modify_thickness, Fehlerfall)
        o Merkmale Coloumn: Profil, Höhe, Material, Positionstyp (Ecke, Rand, Innen), Fehlerattribute (Fehlerfall, modify_height)
        o Merkmale Wall: Abmessungen (Länge, Dicke, Höhe), Material, Fehlerattribute (Fehlerfall, modify_thickness, modify_height)
        o Merkmale Beam: Profil, Länge, Material, Fehlerattribute (Fehlerfall, modify_x, modify_y, modify_length)
    - Erstelle Geschosse iterativ über die Geschossanzahl mit notwendigen Attributen und Bauteilen
        o Definiere sinnvolle Regeln wann, was, wo, wie erzeugt wird
    
    Class ErrorFactory:
    - Erzeuge zufällig Modellierungsfehler
    - bei Stützen über:
        o die Höhe
        o Fehlerfälle
    - bei Decken über:
        o die Abmessungen (in x und y)
        o die Dicke
        o Fehlerfälle
    - bei Wänden über:
        o die Dicke
        o die Höhe
        o Fehlerfälle
    - bei Balken über:
        o die Abmessungen (im Querschnitt in x und y)
        o die Länge
        o Fehlerfälle
    
    Class ProfileFactory:
    - Erzeuge Querschnittsprofile:
    - Rechteck:
        o x_dim
        o y_dim
    - Kreis:
        o Radius
    - I-Profil:
        o Höhe
        o Breite
        o t_f
        o t_w


3. IFC-Modell
    - Erzeuge GUID
    - Erzeuge PSet
        o Slab_Common
        o Column_Common
        o Wall_Common
        o Beam_Common
        o Error_Info
    - Erzeuge IFC-Modell

    
4. Export
    - Identifziere und Definiere notwendige Pfade
    - Exportiere Parameterkombination je Index als .txt und .json
    - Exportiere BuildingGeometrie je Index als .txt und .json
    - Exportiere IFC-Modell je Index
"""



if __name__ == "__main__":
    main()
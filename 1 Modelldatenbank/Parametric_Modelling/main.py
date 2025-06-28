### Import external modules
import logging
# import json
import os
# import random
# import sys
from pathlib import Path

### Import local modules
import Parametric_Modelling


### Setup Logging --> filename muss kompletter lokaler Pfad sein, sonst landet es im cwd, was ganz woanders liegen kann.
log = logging.getLogger(__name__)
script_path = Path(__file__).resolve() # --> Absoluter Pfad dieser Datei
script_folder = script_path.parent # --> Absoluter Pfad zum Ordner dieser Datei
print("SCRIPT_PATH: ", script_path) # Demo
print("SCRIPT_FOLDER: ", script_folder) # Demo
log_file_name = "Porotokoll_ParametrischesModellieren.log"
log_file_path = Path(script_folder / log_file_name)

## Speicher Logging-Protokoll im cwd oder lokal
# logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')  # INFO or DEBUG --> Kein Speichern (Protokoll im Terminal)
logging.basicConfig(filename= log_file_path, level=logging.INFO, format='%(asctime)s %(message)s')  # INFO or DEBUG --> Konkretes Speichern (Im Skript-Pfad)


def main():
    """0. LOGGING DOKUMENTATION - START"""
    ### Setup Logging --> filename muss kompletter lokaler Pfad sein, sonst landet es im cwd, was ganz woanders liegen kann.

    # """Start Tests"""
    # log.info(f"Start - Tests")
    # ## Teste Parameter-Import
    # log.info(Parametric_Modelling.Parameter.test_method("Test"))

    # log.info(f"Ende - Test")
    # """Ende Tests"""
    
    log.info("___Intitialisiere Skript___")
    """0. LOGGING DOKUMENTATION - ENDE"""


    """1. ENTWURFSPARAMETER - Start"""
    log.info(f"___Start - Entwurfsparameter")
    ### Definiere Parameterwertebereich
    Parameter = Parametric_Modelling.parameter_factory
    value_domain = Parameter.Parameterwertbereich_Test
    log.info(f"Parameterwertebereich: {value_domain}")

    ### Erstelle Parameterraum aus Parameterkombinationen
    # parameter_space = Parameter.generate_parameter_combinations(value_domain)
    ParameterSpace = Parameter.ParameterSpace(value_domain)
    parameter_space = ParameterSpace.get_parameter_space()
    
    ## Protokolliere Ergebnisse
    parameter_dimensionality = len(ParameterSpace)
    log.info(f"Parameterraum mit einer Dimensionalität von {parameter_dimensionality}:")
    for parameter_set in parameter_space[:int(parameter_dimensionality)]:
        log.info(f"{parameter_set}")
    """1. ENTWURFSPARAMETER - Ende"""

    
    
    """2. BAUWERKSGEOMETRIE - Start"""
    def exkurs_werte_aus_objekten_einlesen():
        print(f"""Werte können auf 3 Arten sinnvoll eingelesen werden:
              Beispielhaft werden die Werte: "grid_x_n, grid_y_n, grid_x_size, grid_y_size, grid_rotation, grid_x_offset, grid_y_offset, floors, floor_height" gebraucht.
              
              1. Aus einem bestimmten Parameter-Objekt sollen Werte ermittelt werden:
              param_set = parameter_space[0]

              grid_parameter = [
                  param_set.grid_x_n,
                  param_set.grid_y_n,
                  ...
                  param_set.floors,
                  param_set.floor_height
              ]

              2. Aus mehreren Parameter-Objekten die Werte über eine Schleife ermitteln:
              all_grid_parameters = [
                  [
                      p.grid_x_n, p.grid_y_n, ...,
                      p.floors, p.floor_height
                  ]
                  for p in parameter_space
              ]

              3. Parametergruppen über Hilfsmethoden aus betroffener Klasse einladen:
              def get_grid_parameters(self):
                  return [
                      self.grid_x_n, self.grid_y_n, ..., self.floors, self.floor_height
                  ]
              
              Dann in main-Skript einfach schreiben:
              grid_parameters = param_set.get_grid_parameters()
              """)
        pass

    ### Erstelle 3D-Gitterstruktur
    Grid = Parametric_Modelling.grid_factory

    """Start - Tests """
    ### Ermittle ein bestimmtes Parameterset
    # print("PARAM_ATTRIBUT grid_x_n: ", parameter_space[0].grid_x_n) # Debug: So gebe ich direkt die Informationen zurück
    param_set = parameter_space[0] #-> Gibt ein spezifisches Klassen-Objekt zurück
    # print("PARAM_SET [0]: ", param_set) # Demo
    log.info(f"Param_Set [0]:  {param_set}:")

    # grid_parameters = param_set.get_grid_parameters() #-> Gibt durch die Klassenfunktion aus dem aktuellem Klassenobjekt die Parameter für das Grid zurück
    # print("GRID_PARAMS: ", grid_parameters) # Debug

    # grid = Grid.GridFactory(*grid_parameters) #-> Gibt Klassen-Objekt zurück
    grid = Grid.GridFactory(param_set) #-> Gibt Klassen-Objekt zurück
    # print("GRID: ", grid) # Demo
    log.info(f"Grid:  {grid}:")

    grid_points = grid.get_points() #-> Gibt Punkte als eine lange Liste aus Dictionaries zurück
    # print("GRID_POINTS: ", grid_points) # Debug

    ## Gitter plotten und exportieren
    # grid.plot_3d_grid(grid_points) # Debug
    """Ende - Tests"""

    ### Erstelle die Bauwerksgeometrie
    Geometry = Parametric_Modelling.geometry_factory

    """Start - Tests """
    building_geometry = Geometry.BuildingGeometry(grid_points, param_set)
    geometry = building_geometry.generate_gemeotry()
    # print("BAUTEILE: ", *geometry, sep="\n \t") # Demo
    log.info("Bauteile:\n\t" + "\n\t".join(str(bauteil) for bauteil in geometry))
    """Ende - Tests"""
    """2. BAUWERKSGEOMETRIE - Ende"""


    """3. IFC - Start"""
    ### Erstelle IFC-Modell
    Ifc = Parametric_Modelling.ifc_factory

    """Start - Tests """
    ifc = Ifc.IfcFactory(grid_points, param_set, geometry)
    ifc_model = ifc.generate_ifc_model()
    # print("IFC_MODEL: ", ifc_model) # Debug
    """Ende - Tests """
    """3. IFC - Ende"""

    
    """4. Export - Start"""
    ### Exportiere Parameter, Bauwerksgeometrie, IFC-Modell und Plotte Grid
    Utils = Parametric_Modelling.utils

    """Start - Tests"""
    path = Utils.PathFactory(script_path, script_folder, value_domain, parameter_space)
    export_path = path.create_export_path()
    # print("EXPORT_PATH: ", export_path ) # Debug

    log.info(f"Exportiere Daten in folgenden Projektordner: {export_path}")
    exporter = Utils.Exporter(grid_points, param_set, geometry, ifc_model)
    param_set_dic = param_set.__dict__
    geometry_dic = geometry
    exporter.export_data_as_json(param_set_dic, "Parameterset", export_path, 0)
    exporter.export_data_as_json(geometry_dic, "BuildingGeometry", export_path, 0)
    exporter.export_data_as_ifc(ifc_model, "Parametrisches Modell", export_path, 0)
    """Ende - Tests"""

    # ...Eigentlicher_Code IFC-Export (entspricht ca. line 803 export_ifc_model())
    # ...
    """4. Export - Ende"""
    log.info("___Skript erfolgreich beendet___")
    return
    

def maina():
    """0. LOGGING DOKUMENTATION - START"""
    ### Setup Logging --> filename muss kompletter lokaler Pfad sein, sonst landet es im cwd, was ganz woanders liegen kann.
    script_path = Path(__file__).resolve() # --> Absoluter Pfad dieser Datei
    script_folder = script_path.parent # --> Absoluter Pfad zum Ordner dieser Datei
    # print("SCRIPT_PATH: ", script_path) # Demo
    # print("SCRIPT_FOLDER: ", script_folder) # Demo

    log = logging.getLogger(__name__)
    log_file_name = "Porotokoll_ParametrischesModellieren.log"
    log_file_path = Path(script_folder / log_file_name)

    # """Start Tests"""
    # log.info(f"Start - Tests")
    # ## Teste Parameter-Import
    # log.info(Parametric_Modelling.Parameter.test_method("Test"))

    # log.info(f"Ende - Test")
    # """Ende Tests"""
    
    log.info("___Intitialisiere Skript___")
    """0. LOGGING DOKUMENTATION - ENDE"""


    """1. ENTWURFSPARAMETER - Start"""
    log.info(f"___Start - Entwurfsparameter")
    ### Definiere Parameterwertebereich
    Parameter = Parametric_Modelling.parameter_factory
    value_domain = Parameter.Parameterwertbereich_Test
    log.info(f"Parameterwertebereich: {value_domain}")

    ### Erstelle Parameterraum aus Parameterkombinationen
    # parameter_space = Parameter.generate_parameter_combinations(value_domain)
    ParameterSpace = Parameter.ParameterSpace(value_domain)
    parameter_space = ParameterSpace.get_parameter_space()
    
    ## Protokolliere Ergebnisse
    parameter_dimensionality = len(ParameterSpace)
    log.info(f"Parameterraum mit einer Dimensionalität von {parameter_dimensionality}:")
    for parameter_set in parameter_space[:int(parameter_dimensionality)]:
        log.info(f"{parameter_set}")
    """1. ENTWURFSPARAMETER - Ende"""

    
    
    """2. BAUWERKSGEOMETRIE - Start"""
    def exkurs_werte_aus_objekten_einlesen():
        print(f"""Werte können auf 3 Arten sinnvoll eingelesen werden:
              Beispielhaft werden die Werte: "grid_x_n, grid_y_n, grid_x_size, grid_y_size, grid_rotation, grid_x_offset, grid_y_offset, floors, floor_height" gebraucht.
              
              1. Aus einem bestimmten Parameter-Objekt sollen Werte ermittelt werden:
              param_set = parameter_space[0]

              grid_parameter = [
                  param_set.grid_x_n,
                  param_set.grid_y_n,
                  ...
                  param_set.floors,
                  param_set.floor_height
              ]

              2. Aus mehreren Parameter-Objekten die Werte über eine Schleife ermitteln:
              all_grid_parameters = [
                  [
                      p.grid_x_n, p.grid_y_n, ...,
                      p.floors, p.floor_height
                  ]
                  for p in parameter_space
              ]

              3. Parametergruppen über Hilfsmethoden aus betroffener Klasse einladen:
              def get_grid_parameters(self):
                  return [
                      self.grid_x_n, self.grid_y_n, ..., self.floors, self.floor_height
                  ]
              
              Dann in main-Skript einfach schreiben:
              grid_parameters = param_set.get_grid_parameters()
              """)
        pass

    ### Erstelle 3D-Gitterstruktur
    Grid = Parametric_Modelling.grid_factory

    """Start - Tests """
    ### Ermittle ein bestimmtes Parameterset
    # print("PARAM_ATTRIBUT grid_x_n: ", parameter_space[0].grid_x_n) # Debug: So gebe ich direkt die Informationen zurück
    param_set = parameter_space[0] #-> Gibt ein spezifisches Klassen-Objekt zurück
    print("PARAM_SET [0]: ", param_set) # Demo

    # grid_parameters = param_set.get_grid_parameters() #-> Gibt durch die Klassenfunktion aus dem aktuellem Klassenobjekt die Parameter für das Grid zurück
    # print("GRID_PARAMS: ", grid_parameters) # Debug

    # grid = Grid.GridFactory(*grid_parameters) #-> Gibt Klassen-Objekt zurück
    grid = Grid.GridFactory(param_set) #-> Gibt Klassen-Objekt zurück
    print("GRID: ", grid) # Demo

    grid_points = grid.get_points() #-> Gibt Punkte als eine lange Liste aus Dictionaries zurück
    # print("GRID_POINTS: ", grid_points) # Debug

    ## Gitter plotten und exportieren
    # grid.plot_3d_grid(grid_points) # Debug
    """Ende - Tests"""

    # ...Eigentlicher_Code Gitterstruktur
    # ...


    ### Erstelle die Bauwerksgeometrie
    Geometry = Parametric_Modelling.geometry_factory

    """Start - Tests """
    geometry = Geometry.BuildingGeometry(grid_points, param_set)
    building_geometry = geometry.generate_gemeotry()
    print("BAUTEILE: ", *building_geometry, sep="\n \t") # Demo
    """Ende - Tests"""
    
    # ...Eigentlicher_Code Bauwerksgeometrie
    # ...
    """2. BAUWERKSGEOMETRIE - Ende"""


    """3. IFC - Start"""
    ### Erstelle IFC-Modell
    Ifc = Parametric_Modelling.ifc_factory

    """Start - Tests """
    ifc = Ifc.IfcFactory(grid_points, param_set, building_geometry)
    ifc_model = ifc.generate_ifc_model()
    print("IFC_MODEL: ", ifc_model) # Debug
    """Ende - Tests """

    # ...Eigentlicher_Code IFC-Modell
    # ...
    """3. IFC - Ende"""

    
    """4. Export - Start"""
    ### Exportiere IFC-Modell
    Utils = Parametric_Modelling.utils

    """Start - Tests"""
    path = Utils.PathFactory(script_path, script_folder)

    exporter = Utils.Exporter(grid_points, param_set, building_geometry, ifc_model)
    """Ende - Tests"""

    # ...Eigentlicher_Code IFC-Export (entspricht ca. line 803 export_ifc_model())
    # ...
    """4. Export - Ende"""

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
        o Beispiel: falsche Stützenhöhe oder abweichende Deckenplattenmaße.

        

Ablaufdiagramm:
[parameter_factory]  -> Erzeugt Parameterraum aus Parameterkombinationen
    [ParameterSpace]  -> Speichert Parameterraum aus Parameterkombinationen (ParameterSpace Objekt)
        |
        v
    [ParameterSet]  -> Speichert spezifische Parameterkombination (ParameterSet Objekt)
        |
        v
[grid_factory]  -> Erzeugt 3D-Gitter
    [GridFactory]  -> Speichert 3D-Gitter (GridFactory Objekt)
        |
        v
[geometry_factory]  -> Erstellt die Bauwerksgeometrie
    [BuildingGeometry]  -> Erzeugt aus Grid Geometrie (Decken, Stützen, Wände, Balken, Fundament)
        |
        v
        [ErrorFactory]  -> Erzeugt aus regelbasierte zufällige Fehlerfälle in der Bauteilmodellierung
        |
        v
    [IfcGeometry]  -> Erstellt IFC-Datei für späteren Export
        [ProfileFactory]  -> Erzeugt Regelquerschnitte
        |
        v
[utils]  -> Hilfsfunktionen wie Plotting, Export, etc.



Strukturaufbau:
/Projektordner/
├── main.py
└── Parametric_Modelling/
    ├── __init__.py
    ├── parameter_factory.py    # ParameterSet und ParameterSpace (kombiniert Parameter und speichert die Kombinationen)
    ├── grid_factory.py         # GridFactory (erzeugt und verwaltet Gitterpunkte)
    ├── geometry_factory.py     # BuildingGeometry (erzeugt Bauteile aus GridFactory)
    ├── ifc_factory.py          # IfcGeometry (Exportiert Geometrie als IFC-Modell)
    └── utils/                  # (Utility (Hilfsfunktionen wie Plotting, JSON-Export, etc.))



Inhaltsbeschreibung:
1. Entwurfsparameter 
    - Definiere Entwurfsparameter (z.B. Anzahl und Größe der Felder in x und y, Geschossanzahl und -höhe, vorhandene Bauteile, etc.) und weise erlaubten Wertebereich zu (konkrete Werte)
    - Erstelle Parameterkombinationen (Parameterkonfiguration als Betonung der konkreten Zusammensetzung | Parametersatz als Stichprobe)


2. Bauwerksgeometrie
    Class GridFactory:
    - Erstelle Gitterstruktur
        o mit Rotation und Offset
        o Speichere die einzelnen Gitterpunkte als einzelne Objekte ab
        o Merkmale: ID (0, 1, .., n)), Position (x,y,z), Positionstyp (Ecke, Rand, Innen), Reihenindex für x und y
    
    Class BuildingFactory:
    - Erstelle Bauteile (Gründung (Bodenplatte + Streifenfundament), Geschossdecken (Slabs), Stützen, Wände, Balken) mit notwendigen Attributen und Merkmalen
        o Attribute: (ID?), Name, Klasse/Typ, Position, Geschoss (als Index)
        o Merkmale Slab: Abmessungen (in x und y + Höhe/Breite der Stützenprofile), Dicke, Material, Is_Foundation (Boolean), Fehlerattribute (modify_x+/-, modify_y+/-, modify_thickness, Fehlerfall)
        o Merkmale Coloumn: Profil, Höhe, Material, Positionstyp (Ecke, Rand, Innen), Fehlerattribute (Fehlerfall, modify_height)
        o Merkmale Wall: Abmessungen (Länge, Dicke, Höhe), Material, Fehlerattribute (Fehlerfall, modify_thickness, modify_height)
        o Merkmale Beam: Profil, Länge, Material, Fehlerattribute (Fehlerfall, modify_x, modify_y, modify_length)
    - Erstelle iterativ Geschosse über die Geschossanzahl mit notwendigen Attributen und Bauteilen
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
    Class IfcFactory:
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




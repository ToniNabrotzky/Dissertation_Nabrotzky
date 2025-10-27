import itertools

"""Start - Tests"""
def test_method(text=""):
    return f"Teste Method mit Text: {text}"


class Klasse:
    def test_class(text=""):
        return f"Teste Class mit Text: {text}"
"""Ende - Tests"""



class ParameterSpace:
     """Erzeugt beim Initialisieren direkt alle möglichen Parameterkombinationen"""
     def __init__(self, value_domain):
        self.value_domain = value_domain
        self.combinations = self.generate_combinations()
        return
     

     def __repr__(self):
        return f"ParameterSpace mit einer Dimensionalität von {self.dimensionality} Kombinationen."
     

     def __len__(self):
         return len(self.combinations)


     """Hauptfunktion"""
     def generate_combinations(self):
        """Erzeugt alle möglichen Kombinationen von Parametern (Parameterraum).

        Erklärung Funktionsweise:
        - value_domain ist ein Dictionary, das Parameter-Namen als Schlüssel und Listen von möglichen Werten als Werte enthält.
        - Mit value_domain.items() erhält man eine Liste von Tupeln: [(key1, values1), (key2, values2), ...]
        - zip(*value_domain.items()) entpackt diese Tupel in zwei separate Tupel-Listen: 
                keys = (key1, key2, ...)
                values = ([values1], [values2], ...)
        - itertools.product(*values) bildet das kartesische Produkt der Werte-Listen, also alle möglichen Kombinationen.
        - Jede dieser Kombinationen wird mit *combo an den Konstruktor ParameterSet übergeben.
        - Die Funktion gibt eine Liste dieser ParameterSet-Objekte zurück.

        Parameter:
        value_domain : dict
                Dictionary mit Parameternamen als Schlüssel und Listen möglicher Werte als Werte.

        Rückgabe:
        list
                Liste von ParameterSet-Objekten mit allen möglichen Parametersätzen.

        Beispiel:
        value_domain = {
                'grid_x_n': [2, 4],
                'grid_y_n': [3, 5],
                'floors': [1, 2]
        }

        # zip(*value_domain.items()) ergibt:
        # keys   = ('grid_x_n', 'grid_y_n', 'floors')
        # values = ([2, 4], [3, 5], [1, 2])

        # itertools.product(*values) erzeugt:
        # (2, 3, 1)
        # (2, 3, 2)
        # (2, 5, 1)
        # (2, 5, 2)
        # (4, 3, 1)
        # (4, 3, 2)
        # (4, 5, 1)
        # (4, 5, 2)

        # Jede Kombination wird mit *combo an ParameterSet übergeben und in eine Liste gepackt.
        """

        print("___Starte Kombination der Parameter___") # Demo

        ## Parameter-Namen (keys) und Listen von möglichen Werten (values) aus dem Dictionary entpacken
        # Der Sternchen-Operator (*) entpackt Werte als separate Argumente
        # zip-Funktion gruppiert die entpackten Argumente aus den Tupeln in keys und values
        keys, values = zip(*self.value_domain.items())
        # print("KEYS: ", keys)
        # print("VALUES: ", values)

        ## Erzeugung aller möglichen Kombinationen der Werte (kartesisches Produkt)
        # ## Variante Liste ohne Index:
        # combinations = [ParameterSet(*combo) for combo in itertools.product(*values)]

        # ## Variante Klasse ohne Index, aber Output mit Index:
        # combinations = []
        # for i, combo in enumerate(itertools.product(*values)):
        #      param_set = {"index": f"{i:05d}"}
        #      param_set.update(dict(zip(keys, combo)))
        #      combinations.append(param_set)

        ## Variante Klasse mit Index:
        combinations = []
        for i, combo in enumerate(itertools.product(*values)):
             index = f"{i:05d}"
            #  print("*COMBO: ", *combo) # Debug
             param_set = ParameterSet(index, *combo)
             combinations.append(param_set)      

        ## Rückgabe der Parameterkombinationen als Liste von ParameterSet-Objekten
        return combinations
     

     """Extra Methoden"""
     def get_parameter_space(self):
        """Gibt die Liste der ParameterSet-Objekte zurück."""
        return self.combinations



class ParameterSet:
    """Speichert eine spezifische Parameterkombination für ein Bauwerk"""
    def __init__(self, index,
                 grid_x_n, grid_y_n, grid_x_size, grid_y_size, grid_rotation, grid_x_offset, grid_y_offset, # 2D-Gitterstruktur
                 floors, floor_height, # 3D-Gitterstruktur
                 foundation_thickness, strip_foundation_profile, # Gründung
                 slab_thickness, beam_profile, wall_prob, wall_thickness, # Horizontale Bauteile
                 column_prob, column_profile, # Vertikale Bauteile
                ):
        ## Index
        self.index = index

        ## Parameter Gitterstruktur
        self.grid_x_n = grid_x_n                # Anzahl der Gitterfelder in X-Richtung
        self.grid_y_n = grid_y_n                # Anzahl der Gitterfelder in Y-Richtung
        self.grid_x_size = grid_x_size          # Feldgröße in X-Richtung [m]
        self.grid_y_size = grid_y_size          # Feldgröße in Y-Richtung [m]
        self.grid_rotation = grid_rotation      # Rotationswinkel des Gitters um den Ursprung [°]
        self.grid_x_offset = grid_x_offset      # Verschiebung in X-Richtung
        self.grid_y_offset = grid_y_offset      # Verschiebung in Y-Richtung
        
        ## Parameter Geschoss
        self.floors = floors                            # Anzahl der Geschosse
        self.floor_height = floor_height                # Geschosshöhe (OKRD bis OKRD) [m]
        
        ## Parameter Bauteil - Gründung 
        self.foundation_thickness = foundation_thickness              # Dicke der Bodenplatte
        self.strip_foundation_profile = strip_foundation_profile      # QS-Profil der Streifenfundamente
        ## Parameter Bauteil - Deckenplatten/-balken
        self.slab_thickness = slab_thickness                          # Dicke der Deckenplatte
        # self.slab_beam_profile = column_profile                            # QS-Profil der Deckenbalken (richtet sich nach Breite und Höhe der Stützen und Deckendicke)
        ## Parameter Bauteil - Balken
        self.beam_profile = beam_profile                              # QS-Profil der Balken
        ## Parameter Bauteil - Wände
        self.wall_prob = wall_prob                                    # Wand-Wahrscheinlichkeit
        self.wall_thickness = wall_thickness                          # Wanddicke        
        ## Parameter Bauteil - Stützen
        self.column_prob = column_prob                                # Stützen-Wahrscheinlichkeit
        self.column_profile = column_profile                          # Stützenprofil mit entsprechenden Parametern

    
    def __repr__(self):
        # return(f"""ParameterSet(index= {self.index}: 
        #        grid_x_n= {self.grid_x_n}, grid_y_n= {self.grid_y_n}, \tgrid_x_size= {self.grid_x_size}m, grid_y_size= {self.grid_y_size}m, \tgrid_rotation= {self.grid_rotation}°, grid_x_offset= {self.grid_x_offset}m, grid_y_offset= {self.grid_y_offset}m,
        #        floors= {self.floors}, floor_height= {self.floor_height}m,""")
        return (f"ParameterSet(index= {self.index}: "
                f"\n \t grid_x_n= {self.grid_x_n}, grid_y_n= {self.grid_y_n},"
                f"\t grid_x_size= {self.grid_x_size}m, grid_y_size= {self.grid_y_size}m,"
                f"\t grid_rotation= {self.grid_rotation}°, grid_x_offset= {self.grid_x_offset}m, grid_y_offset= {self.grid_y_offset}m,"
                f"\n \t floors= {self.floors}, floor_height= {self.floor_height}m,"
                f"\t foundation_thickness= {self.foundation_thickness}, strip_foundation_profile= {self.strip_foundation_profile},"
                f"\t slab_thickness= {self.slab_thickness}," # Bei Deckenbalken, dann
                f"\n \t beam_profile= {self.beam_profile},"   
                f"\t wall_prob= {self.wall_prob}, wall_thickness= {self.wall_thickness},"
                f"\t column_prob= {self.column_prob}, column_profile= {self.column_profile})"
                )
    

#     """Extra Methoden""" # --> NICHT MEHR NOTWENDIG
#     ### Rückgabe von Parametergruppen für spezielle Anwendungen
#     def get_grid_parameters(self):
#         """Gibt die Parameter aus dem aktuellen Set für die Gittererzeugung als Liste zurück""" ##Ändern auch zu Dic?Wäre maximal wegen einheitlicher Form
#         return [
#                 self.index,
#                 self.grid_x_n, self.grid_y_n, self.grid_x_size, self.grid_y_size, 
#                 self.grid_rotation, self.grid_x_offset, self.grid_y_offset,
#                 self.floors, self.floor_height
#                 ]



class ProfileFactory:
    """Speichert ein spezifisches Querschnittsprofil als Rechteck-, Kreis- oder I-Profil"""
    ### Datenbank für I-Profile
    i_profile_database = { # t_w und t_w direkt in [m] wenn mehr als 10mm, sonst in [mm]/1000[mm] für [m]-Wert
        'HEA400': {'h': 0.39, 'b': 0.3, 't_f': 0.019, 't_w': 0.011},
        'HEB400': {'h': 0.4, 'b': 0.3, 't_f': 0.0135, 't_w': 0.024},
        'IPE300': {'h': 0.3, 'b': 0.15, 't_f': 0.0107, 't_w': 0.0071},
        'IPE400': {'h': 0.4, 'b': 0.18, 't_f': 0.0135, 't_w': 0.0086},
    }


    def __init__(self, profile_type, *args):
        self.profile = self.create_profile(profile_type, *args)
        return
    

    def __repr__(self):
        return (f"<Profile type={self.profile}")
    

    """Hauptfunktion"""
    def create_profile(self, profile_type, *args):
        """Erstellt je nach Profiltyp einen spezifischen Querschnitt"""
        if profile_type == 'rectangle':
            x_dim, y_dim = args 
            return {'type': 'rectangle', 'x_dim': x_dim, 'y_dim': y_dim}

        elif profile_type == 'circle':
            (diameter,) = args # --> Sichert ab, dass es genau 1 Element im Tuple gibt. Alternativ geht auch: diameter = args[0] -> Dann nimmt es einfach das erste Element
            return {'type': 'circle', 'radius': diameter, 'x_dim': 2 * diameter, 'y_dim': 2 * diameter}

        elif profile_type == 'i-profile':
            designation = args[0]
            props = ProfileFactory.i_profile_database.get(designation)
            if props is None:
                raise ValueError(f"Profil {designation} nicht in Datenbank vorhanden.")
            return {'type': 'i-profile',
                    'designation': designation,
                    'h': props['h'], 'b': props['b'],
                    't_f': props['t_f'], 't_w': props['t_w'],
                    'x_dim': props['b'], 'y_dim': props['h']}

        else:
            raise ValueError(f"Unbekannter Profiltyp: {profile_type}")
    


"""Vordefinierte Parameterwertbereiche als Dictionaries"""
Parameterwertbereich_Test = {
        ## Parameter Gitterstruktur
        'grid_x_n': [2],
        'grid_y_n': [3],
        'grid_x_size': [5.0], # in [m]
        'grid_y_size': [2,], # in [m]
        'grid_rotation': [0], # in [degree]
        'grid_x_offset': [0.0], # in [m]
        'grid_y_offset': [0.0], # in [m]
        ## Parameter Geschoss
        'floors': [2],
        'floor_height': [2.8], # in [m]
        ## Parameter Bauteil - Gründung
        'foundation_thickness': [0.6], # --> Dicke der Bodenplatte
        'strip_foundation_profile': [ProfileFactory('rectangle', 0.3, 0.8).profile], # --> QS-Profil der Streifenfundamente
        ## Parameter Bauteil - Deckenplatten/-balken
        'slab_thickness': [0.25], # --> Dicke der Deckenplatte
        ## Parameter Bauteil - Balken
        'beam_profile': [ProfileFactory('i-profile', 'IPE300').profile], # --> QS-Profil der Balken
        ## Parameter Bauteil - Wände
        'wall_prob': [0.6], # --> Wand-Wahrscheinlichkeit
        'wall_thickness': [0.24], # --> Wanddicke        
        ## Parameter Bauteil - Stützen
        'column_prob': [0.8], # --> Stützen-Wahrscheinlichkeit
        'column_profile': [ProfileFactory('circle', 0.2).profile], # --> Stützenprofil mit entsprechenden Parametern
}

Parameterwertbereich_Beton = {
        ## Parameter Gitterstruktur
        'grid_x_n': [2, 4],
        'grid_y_n': [3, 5],
        'grid_x_size': [5.0, 8], # in [m]
        'grid_y_size': [7.5, 10.0], # in [m]
        'grid_rotation': [0.0], # in [degree]
        'grid_x_offset': [0.0, 4.0], # in [m]
        'grid_y_offset': [0.0, 3.0], # in [m]
        ## Parameter Geschoss
        'floors': [2, 4, 5],
        'floor_height': [2.8], # in [m]
        # ## Parameter Bauteil - Platten
        # 'slab_thickness': [0.3], # in [m]
        # 'foundation_thickness': [0.6],  # in [m]
        # ## Parameter Bauteil - Stützen
        # 'corner_columns': [True, False],
        # 'edge_columns': [True, False],
        # 'inner_columns': [True, False],
        # # 'column_profile': [ProfileFactory.Rec_20_30] # in [m] --> [('rectangle', x_dim, y_dim)], [('circle', radius)], [('i-profile', h, b, t_f, t_w)]
        # # 'column_profile': [ProfileFactory.IPE400] # in [m] --> [('rectangle', x_dim, y_dim)], [('circle', radius)], [('i-profile', h, b, t_f, t_w)]
}

Parameterwertbereich_Stahl = {
        ## Parameter Gitterstruktur
        'grid_x_n': [1, 2, 4],
        'grid_y_n': [5, 6],
        'grid_x_size': [4.3, 8], # in [m]
        'grid_y_size': [7.5, 11.5], # in [m]
        'grid_rotation': [0.0], # in [degree]
        'grid_x_offset': [0.0, 3.0], # in [m]
        'grid_y_offset': [0.0, 1.8], # in [m]
        ## Parameter Geschoss
        'floors': [2, 4, 5],
        'floor_height': [2.8], # in [m]
}

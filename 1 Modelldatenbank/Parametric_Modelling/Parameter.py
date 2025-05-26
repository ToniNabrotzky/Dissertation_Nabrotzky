import itertools

"""Start - Dokumentation"""
def teste_method(text=""):
    return f"Teste Method mit Text: {text}"


class Klasse:
    def teste_class(text=""):
        return f"Teste Class mit Text: {text}"
"""Ende - Dokumentation"""



class ParameterSet:
    """Speichert eine spezifische Parameterkombination für ein Bauwerk"""
    def __init__(self, index,
                 grid_x_fields, grid_y_fields, grid_x_size, grid_y_size, grid_x_offset, grid_y_offset, grid_rotation, # 2D-Gitterstruktur
                 floors, floor_height, # 3D-Gitterstruktur
                #  slab_thickness, foundation_thickness,
                #  corner_columns, edge_columns, inner_columns, column_profile
                ):
        ## Index
        self.index = index
        ## Parameter Gitterstruktur
        self.grid_x_fields = grid_x_fields        # Anzahl an Feldern in x-Richtung
        self.grid_y_fields = grid_y_fields        # Anzahl an Feldern in y-Richtung
        self.grid_x_size = grid_x_size            # Ausdehnung einer Gitterzelle in x-Richtung
        self.grid_y_size = grid_y_size            # Ausdehnung einer Gitterzelle in y-Richtung
        self.grid_x_offset = grid_x_offset        # Verschiebung in X-Richtung
        self.grid_y_offset = grid_y_offset        # Verschiebung in Y-Richtung
        self.grid_rotation = grid_rotation        # Verschiebung in X-Richtung
        ## Parameter Geschoss
        self.floors = floors                      # Anzahl an Geschossen
        self.floor_height = floor_height          # Geschosshöhe (OKRD bis OKRD)
        # ## Parameter Bauteil - Platten
        # self.slab_thickness = slab_thickness      # Deckendicke
        # self.foundation_thickness = foundation_thickness  # Dicke der Bodenplatte
        # ## Parameter Bauteil - Stützen
        # self.corner_columns = corner_columns      # Eckstützen (True/False)
        # self.edge_columns = edge_columns          # Randstützen (True/False)
        # self.inner_columns = inner_columns        # Innenstützen (True/False)
        # self.column_profile = column_profile      # Stützenprofil mit entsprechenden Parametern

    
    def __repr__(self):
        return (f"ParameterSet(index={self.index}: "
                f"grid_x_fields={self.grid_x_fields}, grid_y_fields={self.grid_y_fields},   "
                f"grid_x_size={self.grid_x_size}, grid_y_size={self.grid_y_size},   "
                f"grid_x_offset={self.grid_x_offset}, grid_y_offset={self.grid_y_offset}, grid_rotation={self.grid_rotation},   "
                f"floors={self.floors}, floor_height={self.floor_height},   "
                # f"\nslab_thickness={self.slab_thickness}, foundation_thickness={self.foundation_thickness},   "
                # f"\tcorner_columns={self.corner_columns}, edge_columns={self.edge_columns}, inner_columns={self.inner_columns}, column_profile={self.column_profile})"
                )



def generate_parameter_combinations(value_domain):
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
                'grid_x_fields': [2, 4],
                'grid_y_fields': [3, 5],
                'floors': [1, 2]
        }

        # zip(*value_domain.items()) ergibt:
        # keys   = ('grid_x_fields', 'grid_y_fields', 'floors')
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

        ## Parameter-Namen (keys) und Listen von möglichen Werten (values) aus dem Dictionary entpacken
        # Der Sternchen-Operator (*) entpackt Werte als separate Argumente
        # zip-Funktion gruppiert die entpackten Argumente aus den Tupeln in keys und values
        keys, values = zip(*value_domain.items())

        ## Erzeugung aller möglichen Kombinationen der Werte (kartesisches Produkt)
        # ## Variante ohne Index:
        # combinations = [ParameterSet(*combo) for combo in itertools.product(*values)]

        # ## Variante Klasse ohne, aber Output mit Index:
        # combinations = []
        # for i, combo in enumerate(itertools.product(*values)):
        #      param_set = {"index": f"{i:05d}"}
        #      param_set.update(dict(zip(keys, combo)))
        #      combinations.append(param_set)

        ## Variante Klasse mit Index:
        combinations = []
        for i, combo in enumerate(itertools.product(*values)):
             index = f"{i:05d}"
             combinations.append(ParameterSet(index, *combo))        

        ## Rückgabe der Parameterkombinationen als Liste von ParameterSet-Objekten
        return combinations



"""Vordefinierte Parameterwertbereiche"""
Parameterwertbereich_Test = {
        ## Parameter Gitterstruktur
        'grid_x_fields': [2, 4],
        'grid_y_fields': [3, 5],
        'grid_x_size': [5.0, 8], # in [m]
        'grid_y_size': [7.5, 10.0], # in [m]
        'grid_x_offset': [0.0, 4.0], # in [m]
        'grid_y_offset': [0.0, 3.0], # in [m]
        'grid_rotation': [0.0], # in [degree] --> Bauteile passen sich Drehung nicht mit an
        ## Parameter Geschoss
        'floors': [2, 4, 5],
        'floor_height': [2.8], # in [m]
}

Parameterwertbereich_Beton = {
        ## Parameter Gitterstruktur
        'grid_x_fields': [2, 4],
        'grid_y_fields': [3, 5],
        'grid_x_size': [5.0, 8], # in [m]
        'grid_y_size': [7.5, 10.0], # in [m]
        'grid_x_offset': [0.0, 4.0], # in [m]
        'grid_y_offset': [0.0, 3.0], # in [m]
        'grid_rotation': [0.0], # in [degree] --> Bauteile passen sich Drehung nicht mit an
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
        'grid_x_fields': [1, 2, 4],
        'grid_y_fields': [5, 6],
        'grid_x_size': [4.3, 8], # in [m]
        'grid_y_size': [7.5, 11.5], # in [m]
        'grid_x_offset': [0.0, 3.0], # in [m]
        'grid_y_offset': [0.0, 1.8], # in [m]
        'grid_rotation': [0.0], # in [degree] --> Bauteile passen sich Drehung nicht mit an
        ## Parameter Geschoss
        'floors': [2, 4, 5],
        'floor_height': [2.8], # in [m]
}

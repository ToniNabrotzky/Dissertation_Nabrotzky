"""Übersicht Import-Varianten
from .Parameter import teste_method --> Spezieller Import (also ausgewählte/gezielte Methoden, Klassen, Variablen)

import Parametric_Modelling.Parameter --> Allgemeiner Import, bei dem man das Modul immer mitangeben muss (z.B. print(Parametric_Modelling.Parameter.teste_method()))
from .Parameter import * --> Allgemeiner Import, bei dem man keine extra Benennung mit angeben muss. Vorsicht geboten! Nur bei sehr kleinen Varianten empfohlen. (z.B. print(Parametric_Modelling.teste_method()))
"""

import Parametric_Modelling.Parameter
import Parametric_Modelling.BuildingGeometry

# from . import Geometry
# from . import Ifc
# from . import Export
"""Übersicht Import-Varianten
from .Parameter import teste_method     --> Spezieller Import (also ausgewählte/gezielte Methoden, Klassen, Variablen)

import Parametric_Modelling.Parameter   --> Allgemeiner Import, bei dem man das Modul immer mitangeben muss (z.B. print(Parametric_Modelling.Parameter.test_method()))
from Parametric_Modelling import Parameter  --> gleichwertige Variante, die stabiler ist

from .Parameter import *                --> Allgemeiner Import, bei dem man keine extra Benennung mit angeben muss. Vorsicht geboten! Nur bei sehr kleinen Varianten empfohlen. (z.B. print(Parametric_Modelling.test_method()))
"""

from Parametric_Modelling import parameter_factory
from Parametric_Modelling import grid_factory
from Parametric_Modelling import geometry_factory
from Parametric_Modelling import ifc_factory
from Parametric_Modelling import utils

# from . import Geometry
# from . import Ifc
# from . import Export
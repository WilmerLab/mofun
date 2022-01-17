import warnings

from mofun.mofun import *
from mofun.atoms import Atoms
from mofun.atomic_masses import ATOMIC_MASSES

# Force warnings.warn() to omit the source code line in the message
warnings_formatwarning = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    warnings_formatwarning(message, category, filename, lineno, line='')

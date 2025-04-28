import inspect
from pathlib import Path
import cblpython as cbl

MODULE_DIR = Path(inspect.getfile(cbl)).parents[2]
DATA_DIR = MODULE_DIR / "data"
MODEL_DIR = MODULE_DIR / "models"

# %%
import logging
import sys
from pathlib import Path

# !%load_ext autoreload
# !%autoreload 2
# !%matplotlib inline
# !%config InlineBackend.figure_format = 'retina'
import seaborn as sns

sns.set_context("poster")
sns.set(rc={"figure.figsize": (16, 9.0)})
sns.set_style("whitegrid")

import pandas as pd

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

import pyvista as pv

# pv.set_jupyter_backend("client")  # Faster rendering for plots

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

# %%
import cblpython as cbl

# %% [markdown]
# **PLEASE** save this file right now using snake_case_convention (helps pylance keep track of your code for refactoring)

def get_centroid_mesh(mesh: pv.DataSet) -> tuple[pv.PointSet, list[float]]:
    """Calculate the centroid of a given mesh and return it as a point set and a list of coordinates.

    Parameters:
        mesh (pv.DataSet): The input mesh for which the centroid is to be calculated.

    Returns:
        tuple[pv.PointSet, list[float]]: A tuple containing:
            - pv.PointSet: A point set representing the centroid.
            - list[float]: A list of coordinates [x, y, z] of the centroid.
    """
    centroid = mesh.center_of_mass()
    return pv.PointSet(centroid), centroid



# %%

mesh_dir = Path(cbl.DATA_DIR / "example" / "meshes")
knee_mesh = pv.MultiBlock()

pl0 = pv.Plotter()
for mesh_file in mesh_dir.glob("*.inp"):
    tissue_mesh = pv.read(mesh_file, file_format="abaqus")
    pl0.add_mesh(
        tissue_mesh,
        show_edges=True,
        color=None,
        opacity=0.5,
        label=mesh_file.stem,
        scalars=None,
    )
    centroid_mesh, _ = get_centroid_mesh(tissue_mesh) # put items you don't need in _
    pl0.add_mesh(
        centroid_mesh,
        render_points_as_spheres=True,
        point_size=10,
        color="red",
        label=f"{mesh_file.stem}_centroid",
    )
    try:
        pl0.remove_scalar_bar()
    except StopIteration:
        pass

    knee_mesh.append(tissue_mesh)

pl0.add_legend()
pl0.show()

# %%
zeroth_mesh_centroid = knee_mesh[0].center_of_mass()
pl = pv.Plotter()
pl.add_mesh(
    pv.PointSet(zeroth_mesh_centroid),
    render_points_as_spheres=True,
    point_size=10,
    color="red",
)
pl.add_mesh(knee_mesh, show_edges=True, multi_colors=True, opacity=0.5)
pl.add_axes()
pl.show()


# %%

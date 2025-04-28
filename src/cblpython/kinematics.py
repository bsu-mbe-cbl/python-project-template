from dataclasses import dataclass
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import pyvista as pv
from trimesh import unitize
import numpy as np

class Triad:
    """A class to represent a 3D coordinate system triad for visualization."""

    def __init__(self, **kwargs):
        """Initialize the Triad with optional Grood-Suntay nodes.

        Args:
            **kwargs: Optional keyword arguments.
                gs_nodes (np.ndarray, optional): Grood-Suntay nodes for the triad.

        """
        self._transformation = np.ones((4, 4))
        self._transformation[:3, :3] = self.rotation
        gs_nodes = kwargs.get("gs_nodes", None)
        if gs_nodes is not None:
            gs_nodes = np.atleast_2d(gs_nodes).T
            origin = gs_nodes[0, 1:]
            ml = origin + gs_nodes[1, 1:]
            si = origin + gs_nodes[2, 1:]
            ap = origin + gs_nodes[3, 1:]
        else:
            origin = np.zeros(3)
            ml = np.array([1, 0, 0])
            si = np.array([0, 1, 0])
            ap = np.array([0, 0, 1])
        self.x_line = pv.Line(origin, ml)
        self.y_line = pv.Line(origin, si)
        self.z_line = pv.Line(origin, ap)
        self._origin_history = np.atleast_2d([self.origin])
        self._triad = pv.MultiBlock([self.x_line, self.y_line, self.z_line])
        # if gs_nodes is not None:
        #     [x.translate(gs_nodes[1:, 0], inplace=True) for x in self._triad]
        self._triad = pv.MultiBlock([self.x_line, self.y_line, self.z_line])
        self._original_triad = self._triad.copy()

    @property
    def triad(self):
        """Return the current triad."""
        return self._triad

    @property
    def origin(self):
        """Return the origin of the triad."""
        return np.squeeze(self._transformation[:3, 3])

    @property
    def path(self):
        """Return the path of the triad's origin history."""
        return pv.utilities.lines_from_points(self._origin_history)

    @property
    def rotation(self):
        """Return the rotation matrix of the triad."""
        return self._transformation[:3, :3]

    @property
    def translation(self):
        """Return the translation vector of the triad."""
        return np.squeeze(self._transformation[:3, 3])

    @property
    def transformation(self):
        """Return the transformation matrix of the triad."""
        return self._transformation

    def translate(self, translation):
        """Translate the triad by a given vector.

        Args:
            translation (np.ndarray): Translation vector.

        """
        translation = np.array(translation)
        for idx in range(self._triad.n_blocks):
            self._triad[idx].translate(translation, inplace=True)
        self._transformation[:3, 3] = self.origin + translation
        self._origin_history = np.vstack([self._origin_history, self.origin])

    def transform(self, transformation):
        """Apply a transformation matrix to the triad.

        Args:
            transformation (np.ndarray): Transformation matrix.

        """
        self._transformation = transformation
        [x.transform(transformation, inplace=True) for x in self._triad]
        self._transformation = transformation @ self._transformation
        self._origin_history = np.vstack([self._origin_history, self.origin])

    def gs_transform(self, transformation):
        """Apply a Grood-Suntay transformation to the triad.

        Args:
            transformation (np.ndarray): Transformation matrix.

        """
        for idx in range(self._triad.n_blocks):
            pts = self._original_triad[idx].points.copy()
            pts = np.hstack([np.ones((pts.shape[0], 1)), pts])
            padded_pts = (transformation) @ pts.copy().T
            self._triad[idx].points = padded_pts.T[:, 1:]

    def add_to_plotter(self, plotter, **kwargs):
        """Add the triad to a PyVista plotter.

        Args:
            plotter (pv.Plotter): PyVista plotter.
            **kwargs: Additional keyword arguments for plotter.add_mesh().

        """
        plotter.add_mesh(self.x_line, color="r", **kwargs)
        plotter.add_mesh(self.y_line, color="g", **kwargs)
        plotter.add_mesh(self.z_line, color="b", **kwargs)

    def update(self):
        """Update the original triad to the current state."""
        self._original_triad = self._triad.copy()


@dataclass
class AbaqusKinematicsSubject:
    """A class to represent the kinematics of a subject using Abaqus simulation results.

    This class:
      1) Reads columns from a DataFrame corresponding to femoral, tibial, and patellar
         origin, medial-lateral, anterior-posterior, and superior-inferior nodes.
      2) Constructs homogeneous coordinate matrices for each bone.
      3) Defines local coordinate systems for each bone (femur, tibia, patella).
      4) Computes transformation matrices from femur to tibia and femur to patella.
      5) Extracts clinical angles (flexion-extension, varus-valgus, int-ext rotation)
         and translations (ML, AP, SI) in degrees/millimeters, following a
         Grood-Suntay-like approach.

    Attributes:
        idStr (str): Identifier string for the subject.
        side (str): Side of the body (left or right).
        results_df (pd.DataFrame): DataFrame containing the simulation results.

    """

    idStr: str
    side: str
    # geometry: DenverGeometry
    results_df: pd.DataFrame

    def __post_init__(self):
        """Post-initialization to set up index attributes for kinematic calculations."""
        # kintff_raw clinical 6do translation and rotation indices
        self._fe_idx = 0
        self._vv_idx = 1
        self._ie_idx = 2
        self._ml_idx = 3
        self._ap_idx = 4
        self._si_idx = 5

    def _reshape_bone_triad_coordinates(self, triad_coordinate_list):  # gsRowToMatrix
        """Reshape the bone coordinates from a flat array to a 3xN matrix.

                Convert a 1D array (length 12) of [x, y, z] repeated for four points
                (origin, ML, AP, SI) into a 4×3 matrix.

        Args:
                triad_coordinate_list (pd.Series or np.ndarray): The grood-suntay
                results.

        Returns:
                    np.ndarray: Reshaped bone coordinates.

        """
        if isinstance(triad_coordinate_list, pd.Series):
            triad_coordinate_list = triad_coordinate_list.values
        return triad_coordinate_list.reshape(3, -1, order="F").T

    def pad_with_ones(self, matrix):
        """Pad a matrix with a column of ones.

        Add a leading column of ones to create a homogeneous coordinate matrix.

        Args:
            matrix (np.ndarray): The input matrix.

        Returns:
            np.ndarray: The padded (4, 3) matrix.

        """
        return np.hstack([np.ones((matrix.shape[0], 1)), matrix])

    def _unitize_vector(self, vector):
        """Normalize a vector to unit length.

        Args:
            vector (np.ndarray): The input vector.

        Returns:
            np.ndarray: The unit vector.

        """
        return unitize(vector)

    def wrap_angle_if_exceeds_90(self, feCalculation, cutoff):
        """Wrap an angle if it exceeds 90 degrees.

        Adjust the flexion angle if it is below a certain cutoff (e.g., -90 degrees),
        to handle possible wrapping.

        Args:
            feCalculation (float): The angle to wrap.
            cutoff (float): The cutoff angle.

        Returns:
            float: The wrapped angle.

        """
        if feCalculation < cutoff:
            feCalculation = feCalculation + np.pi
        else:
            feCalculation = feCalculation
        return feCalculation

    def compute_relative_transformation(self, *, measure_from_frame, relative_to_frame):
        """Compute the transformation matrix from one frame to another.

        Compute the transformation that expresses 'relative_to_frame' coordinates in the
        'measure_from_frame' reference system.

        Args:
            measure_from_frame (np.ndarray): The source frame.
                4x4 homogeneous transform of the starting reference frame.
            relative_to_frame (np.ndarray): The target frame.
                4x4 homogeneous transform of the destination reference frame.

        Returns:
            np.ndarray: The transformation matrix.

        """
        return np.linalg.inv(measure_from_frame) @ relative_to_frame

    def extract_clinical_kinematics(self, shared_frame, return_degrees=True):
        """Extract clinical kinematics from a shared frame.

        Interprets a 4x4 transformation matrix between the femur and another bone
        (tibia or patella) to compute 3 rotational (FE, VV, IE) and 3 translational
        (ML, AP, SI) degrees of freedom.

        Args:
            shared_frame (np.ndarray): The shared frame matrix.
            return_degrees (bool): Whether to return angles in degrees.

        Returns:
            np.ndarray: The clinical kinematics.
                1D array of length 6: [FE, VV, IE, ML, AP, SI]

        """
        fe = -np.arctan2(
            shared_frame[2, 3], shared_frame[3, 3]
        )  # Needs to be positive.
        if "left" in self.side.lower():
            vv = np.arccos(shared_frame[1, 3]) + np.pi / 2
            ie = np.arctan2(shared_frame[1, 2], shared_frame[1, 1])
        else:
            vv = np.arccos(shared_frame[1, 3]) - np.pi / 2
            ie = -np.arctan2(shared_frame[1, 2], shared_frame[1, 1])

        # Translations
        ml = shared_frame[1, 0]
        ap = shared_frame[2, 0] * np.cos(fe) - shared_frame[3, 0] * np.sin(fe)
        si = (
            shared_frame[1, 3] * shared_frame[1, 0]
            + shared_frame[2, 3] * shared_frame[2, 0]
            + shared_frame[3, 3] * shared_frame[3, 0]
        )

        # TODO: Store this in a dataclass
        kin = np.hstack([fe, vv, ie, ml, ap, si])
        if return_degrees:
            kin[0:3] = np.rad2deg(kin[0:3])
        return kin

    def build_local_frame(self, gs_triad_coordinates):
        """Build a global-to-local coordinate transformation matrix for a bone.

        Construct a local 4x4 coordinate system for a given bone (femur/tibia/patella).
        This matrix is used to compute the transformation from the bone's global
        coordinates to its local coordinate system.

        Args:
            gs_triad_coordinates (np.ndarray): The grood-suntay global bone triad
                coordinates.

                A 4x4 matrix [1, x, y, z].T for each of [origin, ML, AP, SI]
                points.

        Returns:
            np.ndarray: The local coordinate system matrix.
                4x4 matrix representing the bone's local coordinate system.

        """
        origin_row = 0
        ml_row = 1
        ap_row = 2  # noqa: F841
        si_row = 3
        loc_cols = np.arange(1, 4)

        origin_loc = gs_triad_coordinates[origin_row, loc_cols]
        # lat_assumed = unitize(gs_bone[ml_row, loc_cols] - origin_loc)
        lat_assumed = unitize(origin_loc - gs_triad_coordinates[ml_row, loc_cols])
        sup_direction = unitize(gs_triad_coordinates[si_row, loc_cols] - origin_loc)
        ap_direction = gs_triad_coordinates[si_row, loc_cols] - origin_loc
        ant_direction = unitize(np.cross(sup_direction, lat_assumed))
        lat_direction = unitize(np.cross(ant_direction, sup_direction))

        lat_assumed = unitize(gs_triad_coordinates[ml_row, loc_cols] - origin_loc)
        # lat_assumed = unitize(origin_loc - gs_bone[ml_row, loc_cols])
        sd = unitize(gs_triad_coordinates[si_row, loc_cols] - origin_loc)
        ad = unitize(np.cross(sd, lat_assumed))
        lat_direction = unitize(np.cross(ad, sd))

        # Just using the GS nodes as unit vectors
        origin_loc = gs_triad_coordinates[origin_row, loc_cols]
        lat_direction = unitize(gs_triad_coordinates[ml_row, loc_cols] - origin_loc)
        ant_direction = unitize(gs_triad_coordinates[ap_row, loc_cols] - origin_loc)
        sup_direction = unitize(gs_triad_coordinates[si_row, loc_cols] - origin_loc)

        local_sys = np.zeros((4, 4))
        local_sys[0, 0] = 1
        local_sys[1:4, :4] = np.vstack(
            [origin_loc, lat_direction, sup_direction, ant_direction]
        ).T
        return local_sys

    def get_gs_kinematics(self):
        """Get the grood-suntay kinematics for the subject.

        Computes Grood-Suntay style kinematics (FE, VV, IE, ML, AP, SI) for each bone
        pair (femur-tibia & femur-patella) across all time steps.

        Returns:
            pd.DataFrame: DataFrame containing the kinematics.
                [femFE, femVV, femIE, femML, femAP, femSI, ...
                    patFE, patVV, patIE, patML, patAP, patSI]

        """
        # extract the bone-specific tables
        fem_df = self._extract_global_bone_axis_data("fem")
        tib_df = self._extract_global_bone_axis_data("tib")
        pat_df = self._extract_global_bone_axis_data("pat")

        # Extract the kinematic data for each joint
        kinematics = np.full((fem_df.shape[0], 12), np.nan)
        for idx in np.arange(fem_df.shape[0]):
            # convert the row to matrix form and prepend a column of ones
            femur_global_coordinate_matrix = self.pad_with_ones(
                self._reshape_bone_triad_coordinates(fem_df.iloc[idx, :].values)
            )
            tibia_global_coordinate_matrix = self.pad_with_ones(
                self._reshape_bone_triad_coordinates(tib_df.iloc[idx, :].values)
            )
            patella_global_coordinate_matrix = self.pad_with_ones(
                self._reshape_bone_triad_coordinates(pat_df.iloc[idx, :].values)
            )

            # Get the local coordinate frames for each bone
            femur_local_transform = self.build_local_frame(
                femur_global_coordinate_matrix
            )
            tibia_local_transform = self.build_local_frame(
                tibia_global_coordinate_matrix
            )
            patella_local_transform = self.build_local_frame(
                patella_global_coordinate_matrix
            )

            # Get the transformation matrices for each bone
            femur_transformed_to_tibia = self.compute_relative_transformation(
                measure_from_frame=femur_local_transform,
                relative_to_frame=tibia_local_transform,
            )
            patella_transformed_to_femur = self.compute_relative_transformation(
                measure_from_frame=patella_local_transform,
                relative_to_frame=femur_local_transform,
            )
            # Clinical kinematics
            femur_kinematics = self.extract_clinical_kinematics(femur_transformed_to_tibia)
            patella_kinematics = self.extract_clinical_kinematics(patella_transformed_to_femur)

            kinematics[idx, :] = np.hstack([femur_kinematics, patella_kinematics])
        kinematics = pd.DataFrame(
            kinematics,
            columns=[
                "fe",
                "vv",
                "ie",
                "ml",
                "ap",
                "si",
                "fe",
                "vv",
                "ie",
                "ml",
                "ap",
                "si",
            ],
        )
        return kinematics

    def calculate_affine_matrices(self):
        """Calculate the affine matrices for the kinematic data."""
        # get the kinematic data
        fem_df = self._extract_global_bone_axis_data("fem")
        tib_df = self._extract_global_bone_axis_data("tib")
        pat_df = self._extract_global_bone_axis_data("pat")
        np.moveaxis(fem_df.iloc[:2, :6].T.values.reshape(-1, 3, 2), [1, 2], [2, 0])

    def _extract_global_bone_axis_data(self, bone: str):
        """Get the ordered bone table from the results DataFrame.

        Retrieve DataFrame columns for the specified bone (fem, tib, pat). Returns
        four node sets in the order of [origin, ML, AP, SI].

        Args:
            bone (str): The bone name to query.
                "fem", "tib", or "pat"

        Returns:
            pd.DataFrame: The ordered bone table.
            Sub-dataframe with columns for origin, ml, ap, si

        """
        # grab the columns containing the bone query
        df = self.results_df
        has_bone = df.columns.str.contains(bone, case=False)
        bone_table = df.loc[:, has_bone]

        # Find the extracted column containing the origin
        has_origin = bone_table.columns.str.contains("orig", case=False)
        orig_cols = bone_table.loc[:, has_origin]

        # Repeat for the medial-lateral, anterior-posterior, and superior-inferior axes
        has_ml = (
            bone_table.columns.str.contains("med", case=False)
            | bone_table.columns.str.contains("lat", case=False)
            | bone_table.columns.str.contains("ml", case=False)
        )

        has_ap = (
            bone_table.columns.str.contains("ant", case=False)
            | bone_table.columns.str.contains("post", case=False)
            | bone_table.columns.str.contains("ap", case=False)
        )
        has_si = (
            bone_table.columns.str.contains("sup", case=False)
            | bone_table.columns.str.contains("inf", case=False)
            | bone_table.columns.str.contains("si", case=False)
        )
        has_xyz = (
            bone_table.columns.str.contains("x", case=False)
            | bone_table.columns.str.contains("y", case=False)
            | bone_table.columns.str.contains("z", case=False)
        )

        ml_cols = bone_table.loc[:, has_xyz & has_ml]
        ap_cols = bone_table.loc[:, has_xyz & has_ap]
        si_cols = bone_table.loc[:, has_xyz & has_si]

        # Return a dataframe with the columns concatenated as origin, ml, ap, si
        return pd.concat([orig_cols, ml_cols, ap_cols, si_cols], axis=1)

    def plot_kinematics(self, show_plot: bool = True):
        """Plot the history of the kinematic data.

        Plot the computed (FE, VV, IE, ML, AP, SI) angles/translations over time.

        Args:
            show_plot (bool): Whether to show the plot.  If True, show the resulting
            plots immediately.

        """
        has_time = self.results_df.columns.str.contains("time", case=False)
        time_series = self.results_df.loc[:, has_time]
        # Change the time series column label to lowercase time
        time_series.columns = ["time"]

        kin_df = self.get_gs_kinematics()
        kin_df = kin_df.iloc[:, :6]
        kin_df = pd.concat([time_series, kin_df], axis=1)
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()

        for i, column in enumerate(kin_df.columns[1:]):
            kin_df.plot(x="time", y=column, ax=axs[i])
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel(column)

        plt.tight_layout()
        if show_plot:
            plt.show()

    def animate_joint(self, data_dir: Path):  # FIXME: This shouldn't be a Path
        """Animate the joint using the kinematic data.

        Demonstration method that uses PyVista to visualize the bones
        through time, translating them according to the transforms.

        Args:
            data_dir (Path): The directory containing the geometry INP files.
            The folder containing the *.inp files (e.g. BONE-FEMUR.inp, etc.)

        """
        # Load the geometry INP files using pyvista
        femur_bone = pv.read(data_dir.joinpath("BONE-FEMUR.inp"), file_format="abaqus")
        tibia_bone = pv.read(data_dir.joinpath("BONE-TIBIA.inp"), file_format="abaqus")
        patella_bone = pv.read(
            data_dir.joinpath("BONE-PATELLA.inp"), file_format="abaqus"
        )

        # get the kinematic data
        fem_df = self._extract_global_bone_axis_data("fem")
        tib_df = self._extract_global_bone_axis_data("tib")
        pat_df = self._extract_global_bone_axis_data("pat")

        # Extract the kinematic data for each joint
        kinematics = np.full((fem_df.shape[0], 12), np.nan)
        tibiofemoral_transforms = np.full((fem_df.shape[0], 4, 4), np.nan)
        patellofemoral_transforms = np.full((fem_df.shape[0], 4, 4), np.nan)
        fem_global = np.full((fem_df.shape[0], 4, 4), np.nan)
        tib_global = np.full((fem_df.shape[0], 4, 4), np.nan)
        pat_global = np.full((fem_df.shape[0], 4, 4), np.nan)
        for idx in np.arange(fem_df.shape[0]):
            # convert the row to matrix form and prepend a column of ones
            fem_gs = self.pad_with_ones(
                self._reshape_bone_triad_coordinates(fem_df.iloc[idx, :])
            )
            tib_gs = self.pad_with_ones(
                self._reshape_bone_triad_coordinates(tib_df.iloc[idx, :])
            )
            pat_gs = self.pad_with_ones(
                self._reshape_bone_triad_coordinates(pat_df.iloc[idx, :])
            )

            # Get the local coordinate frames for each bone
            fem_local = self.build_local_frame(fem_gs)
            tib_local = self.build_local_frame(tib_gs)
            pat_local = self.build_local_frame(pat_gs)

            # Get the transformation matrices for each bone
            femur_rel_tibia = self.compute_relative_transformation(
                measure_from_frame=fem_local, relative_to_frame=tib_local
            )
            femur_rel_patella = self.compute_relative_transformation(
                measure_from_frame=fem_local, relative_to_frame=pat_local
            )
            tibiofemoral_transforms[idx, :, :] = femur_rel_tibia
            patellofemoral_transforms[idx, :, :] = femur_rel_patella
            fem_global[idx, :] = fem_local.copy()
            pat_global[idx, :] = pat_local.copy()
            tib_global[idx, :] = tib_local.copy()

        # Create the plotter
        # TODO: Candidate for inheritance
        pl = pv.Plotter(off_screen=False)
        pl.add_axes(xlabel="x-Med", ylabel="y-Sup", zlabel="z-Ant")
        femur_actor = pl.add_mesh(femur_bone, opacity=0.5)
        tibia_actor = pl.add_mesh(tibia_bone, opacity=0.5)
        patella_actor = pl.add_mesh(patella_bone)
        fem_gs_translation = fem_global[0, 1:4, 0]
        tib_gs_translation = tib_global[0, 1:4, 0]
        pat_gs_translation = pat_global[0, 1:4, 0]
        # tibia_bone.translate(-tib_gs_node_origin, inplace=True)

        # patella_bone.translate(-pat_gs_node_origin, inplace=True)

        femur_axes = Triad(gs_nodes=fem_global[0])
        # femur_axes = Triad()
        femur_axes.add_to_plotter(pl, render_lines_as_tubes=True)
        fem_gs_translation = fem_global[0, 1:4, 0]  # - np.array([0, 30, 0])
        femur_bone.translate(-fem_gs_translation, inplace=True)
        femur_axes.translate(-fem_gs_translation)
        femur_axes.update()

        tibia_bone.translate(-tib_gs_translation, inplace=True)
        patella_bone.translate(-pat_gs_translation, inplace=True)

        # plot the geometry and translate it according to the kinematic data
        pl.show(auto_close=False)
        pts = femur_bone.points.copy()
        pts = np.hstack([np.ones((pts.shape[0], 1)), pts])

        tib_pts = tibia_bone.points.copy()
        tib_pts = np.hstack([np.ones((tib_pts.shape[0], 1)), tib_pts])
        pat_pts = patella_bone.points.copy()
        pat_pts = np.hstack([np.ones((pat_pts.shape[0], 1)), pat_pts])
        loops = 0
        while loops < 4:
            for idx in range(fem_global.shape[0]):
                # tf_transform = tibiofemoral_transforms[idx]
                fem_transform = fem_global[idx]
                tib_transform = tib_global[idx]
                pat_transform = pat_global[idx]
                padded_pts = fem_transform @ pts.copy().T
                femur_bone.points = padded_pts.T[:, 1:]

                padded_pts = (pat_transform) @ pat_pts.copy().T
                patella_bone.points = padded_pts.T[:, 1:]

                padded_pts = (tib_transform) @ tib_pts.copy().T
                tibia_bone.points = padded_pts.T[:, 1:]

                femur_axes.gs_transform(fem_transform)

                # pl.write_frame()
                pl.update()
                # femur_bone.transform(np.linalg.inv(tf_transform))
                # femur_axes.transform(np.linalg.inv(tf_transform))

            for idx in range(fem_global.shape[0]):
                # tf_transform = tibiofemoral_transforms[idx]
                fem_transform = fem_global[::-1][idx]
                tib_transform = tib_global[::-1][idx]
                pat_transform = pat_global[::-1][idx]
                padded_pts = fem_transform @ pts.copy().T
                femur_bone.points = padded_pts.T[:, 1:]

                padded_pts = (pat_transform) @ pat_pts.copy().T
                patella_bone.points = padded_pts.T[:, 1:]

                padded_pts = (tib_transform) @ tib_pts.copy().T
                tibia_bone.points = padded_pts.T[:, 1:]

                femur_axes.gs_transform(fem_transform)

                # pl.write_frame()
                pl.update()
            loops += 1
        pl.close()


def denver_to_pyvista_geometry(geometry):
    """Convert Denver geometry to PyVista PolyData.

    Args:
        geometry (np.ndarray): Denver geometry data.

    Returns:
        pv.PolyData: PyVista PolyData object.

    """
    nodes = geometry[0, 0]["nds"][0, 0][:, 1:]
    faces = geometry[0, 0]["faces"][0, 0] - 1
    faces[:, 0] = 3
    return pv.PolyData(nodes, faces)

import os
import numpy as np
import pymech as pm
import xarray as xr
from xarray.core.utils import Frozen

class NekDataStore(xr.backends.common.AbstractDataStore):
    """Xarray backend store to map a single Nek/Neko element's binary data."""
    
    axes = ("z", "y", "x")

    def __init__(self, field_elem, mesh_elem):
        self.field = field_elem  # Holds variables (velocity, pressure, etc.)
        self.mesh = mesh_elem    # Holds spatial grid coordinates

    def meshgrid_to_dim(self, mesh_array):
        """Extracts 1D coordinate arrays from a 3D meshgrid layout."""
        # Use 5 decimals for float32 to clear machine noise, 8 for float64
        decimals = 5 if mesh_array.dtype == np.float32 else 8
        return np.unique(np.round(mesh_array, decimals))

    def get_dimensions(self):
        return self.axes

    def get_attrs(self):
        attrs = {
            "boundary_conditions": self.field.bcs,
            "curvature": self.field.curv,
            "curvature_type": self.field.ccurv,
        }
        return Frozen(attrs)

    def get_variables(self):
        """Generates the dictionary of xarray Variables for this element."""
        ax = self.axes
        f, m = self.field, self.mesh

        # Base spatial coordinates and mesh fields
        data_vars = {
            ax[2]: self.meshgrid_to_dim(m.pos[0]),  # x-coordinate
            ax[1]: self.meshgrid_to_dim(m.pos[1]),  # y-coordinate
            ax[0]: self.meshgrid_to_dim(m.pos[2]),  # z-coordinate
            "xmesh": xr.Variable(ax, m.pos[0]),
            "ymesh": xr.Variable(ax, m.pos[1]),
            "zmesh": xr.Variable(ax, m.pos[2]),
            "ux": xr.Variable(ax, f.vel[0]),
            "uy": xr.Variable(ax, f.vel[1]),
            "uz": xr.Variable(ax, f.vel[2]),
        }

        # Conditional physical fields
        if f.pres.size:
            data_vars["pressure"] = xr.Variable(ax, f.pres[0])
        if f.temp.size:
            data_vars["temperature"] = xr.Variable(ax, f.temp[0])

        # Dynamic passive scalars
        if f.scal.size:
            for i in range(f.scal.shape[0]):
                data_vars[f"s{i+1:02d}"] = xr.Variable(ax, f.scal[i])

        return Frozen(data_vars)


def open_dataset(path, ref, drop_variables=None, DTYPE='float64'):
    """Interface for converting single Nek/Neko field files into xarray Datasets.

    Parameters
    ----------
    path : str
        Path to the target Neko field file (e.g., 'field0.f00005')
    ref : str
        Path to the reference mesh topology file (usually 'field0.f00000')
    drop_variables : list, optional
        List of variable names to drop from the final dataset.
    DTYPE : str, default='float64'
        Precision layout of the binary file ('float32' or 'float64').
    """
    field_data = pm.readnek(path, dtype=DTYPE)
    if isinstance(field_data, int):
        raise OSError(f"Failed to load target file: {path}")

    mesh_data = pm.readnek(ref, dtype=DTYPE)
    if isinstance(mesh_data, int):
        raise OSError(f"Failed to load reference mesh: {ref}")

    # Build elemental datasets
    elem_stores = [NekDataStore(f, m) for f, m in zip(field_data.elem, mesh_data.elem)]
    
    try:
        elem_dsets = [
            xr.Dataset.load_store(store).set_coords(store.axes) for store in elem_stores
        ]
    except ValueError as err:
        raise NotImplementedError(
            "Dataset parsing failed. This function currently only maps structured "
            "Cartesian box meshes cleanly into 1D coordinate dimensions."
        ) from err

    # Combine separate elements into a single continuous domain
    ds = xr.combine_by_coords(elem_dsets, combine_attrs="drop")
    ds.coords.update({"time": field_data.time})

    return ds.drop_vars(drop_variables) if drop_variables else ds


def comp_nut(les_folder, save=False, output_file="nut_profiles.nc", DTYPE="float32"):
    """Extracts, horizontally averages, and concatenates turbulent viscosity profiles

    from Neko 'les0.f*' field history outputs.
    """
    files = sorted([f for f in os.listdir(les_folder) if f.startswith("les0.f")])
    if not files:
        raise FileNotFoundError(f"No les0.f* files found in {les_folder}")
        
    les0_ref_path = os.path.join(les_folder, files[0])
    nut_list = []

    for file in files:
        full_path = os.path.join(les_folder, file)
        
        # Open using local open_dataset function with matching DTYPE precision
        ds = open_dataset(path=full_path, ref=les0_ref_path, DTYPE=DTYPE)
        
        # Compute horizontal planar mean (LES context: homogeneous x-y directions)
        # Neko outputs turbulent viscosity (nut) inside the temperature slot for SGS fields
        nut_profile = ds.temperature.mean(dim=["x", "y"])
        
        # Expand along actual physical simulation time instead of loop counter integers
        nut_profile = nut_profile.expand_dims(time=[ds.time.values])
        nut_list.append(nut_profile)

    # Combine profiles along the time timeline
    nut_profiles = xr.concat(nut_list, dim="time")
    nut_profiles.name = "nut"

    if save:
        output_path = os.path.join(les_folder, output_file)
        nut_profiles.to_netcdf(output_path)

    return nut_profiles
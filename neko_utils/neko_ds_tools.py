import xarray as xr
import pymech as pm
import numpy as np
from xarray.core.utils import Frozen

class NekDataStore(xr.backends.common.AbstractDataStore):
    """Xarray store for a Nek field element.

    Parameters
    ----------
    elem: :class:`pymech.core.Elem`
        A Nek5000 element.

    """
    axes = ("z", "y", "x")

    def __init__(self, elem, elem2):
        self.elem = elem
        self.elem2 = elem2

    def meshgrid_to_dim(self, mesh):
        """Reverse of np.meshgrid. This method extracts one-dimensional
        coordinates from a cubical array format for every direction.
        """
        dim = np.unique(np.round(mesh, 8))
        return dim

    def get_dimensions(self):
        return self.axes

    def get_attrs(self):
        elem = self.elem
        attrs = {
            "boundary_conditions": elem.bcs,
            "curvature": elem.curv,
            "curvature_type": elem.ccurv,
        }
        return Frozen(attrs)

    def get_variables(self):
        """Generate an xarray dataset from a single element."""
        ax = self.axes
        elem = self.elem
        elem2 = self.elem2

        data_vars = {
            ax[2]: self.meshgrid_to_dim(elem2.pos[0]),  # x
            ax[1]: self.meshgrid_to_dim(elem2.pos[1]),  # y
            ax[0]: self.meshgrid_to_dim(elem2.pos[2]),  # z
            "xmesh": xr.Variable(ax, elem2.pos[0]),
            "ymesh": xr.Variable(ax, elem2.pos[1]),
            "zmesh": xr.Variable(ax, elem2.pos[2]),
            "ux": xr.Variable(ax, elem.vel[0]),
            "uy": xr.Variable(ax, elem.vel[1]),
            "uz": xr.Variable(ax, elem.vel[2]),
        }
        if elem.pres.size:
            data_vars["pressure"] = xr.Variable(ax, elem.pres[0])

        if elem.temp.size:
            data_vars["temperature"] = xr.Variable(ax, elem.temp[0])

        if elem.scal.size:
            data_vars.update(
                {
                    "s{:02d}".format(iscalar + 1): xr.Variable(ax, elem.scal[iscalar])
                    for iscalar in range(elem.scal.shape[0])
                }
            )

        return Frozen(data_vars)

def nek_dataset(path, ref, drop_variables=None):
    """Interface for converting Nek field files into xarray_ datasets.

    Input: path (str) = path to Neko field file *0.f0* (not *0.f00000)
           ref (str) = path to zero-th Neko field file *0.f00000
           drop_variables (bool) = list of variables to drop 

    Output: xarray dataset now retaining coordinate information

    Usage: ds = nek_dataset(path = f"path/to/case/field0.f000{n}",
                             ref = f"path/to/ref/field0.f00000")

    .. _xarray: https://docs.xarray.dev/en/stable/
    """
    field = pm.readnek(path)
    if isinstance(field, int):
        raise OSError(f"Failed to load {path}")

    el = pm.readnek(ref)
    elements = field.elem
    elements2 = el.elem
    elem_stores = [
        NekDataStore(elem, elem2) for elem, elem2 in zip(elements, elements2)
    ]
    try:
        elem_dsets = [
            xr.Dataset.load_store(store).set_coords(store.axes) for store in elem_stores
        ]
    except ValueError as err:
        raise NotImplementedError(
            "Opening dataset failed because you probably tried to open a field file "
            "with an unsupported mesh. "
            "The `pymech.open_dataset` function currently works only with cartesian "
            "box meshes. For more details on this, see "
            "https://github.com/eX-Mech/pymech/issues/31"
        ) from err

    ds = xr.combine_by_coords(elem_dsets, combine_attrs="drop")
    ds.coords.update({"time": field.time})

    if drop_variables:
        ds = ds.drop_vars(drop_variables)

    return ds

# neko_utils/csv_tools.py

import pandas as pd
import xarray as xr
import neko_utils as nk

def csv_to_xr(path, type="fluid", basic=True, height="z",fluid_csv=None, save=False):
    """
    Converts csv Neko file into xarray DataSet.

    Input: 
        path (str) = path to CSV file with Neko data
        basic (bool) = if True, the function assumes the csv file has the "basic"
                       variables from Neko documentation
        height (str) = name of the vertical coordinate in the xarray DataSet to create
        type (str) = "fluid" or "scalar" depending on the type of csv file
        fluid_csv (str) = path to the corresponding fluid csv file when type="scalar" to import velocity/pressure
        save (bool) = if True, saves the xarray DataSet to a netCDF file
         
    Output: xarray DataSet with the time (time) and height (z) as coordinates.
            Higher order statistics have the mean component removed.

    Usage: ds = csv_to_xr(/path/to/csv, basic=True, height="z")
    """

    df = pd.read_csv(path)
    if type=="fluid":
        if basic or (len(df.columns) - 2 == 11):
            vars = [
                "p", "u", "v", "w", "pp", "uu", "vv", "ww", "uv", "uw", "vw"
            ] 
        else:
            vars = [
                "p", "u", "v", "w", "pp", "uu", "vv", "ww", "uv", "uw", "vw", "uuu", "vvv", "www", 
                "uuv", "uuw", "uvv", "uvw", "vvw", "uww", "vww", "uuuu", "vvvv", "wwww", "ppp", 
                "pppp", "pu", "pv", "pw", "pdudx", "pdudy", "pdudz", "pdvdx", "pdvdy", "pdvdz", 
                "pdwdx", "pdwdy", "pdwdz", "e11", "e22", "e33", "e12", "e13", "e23"
            ]
    elif type=="scalar":
        if basic or (len(df.columns) - 2 == 5):
            vars = [
                "s", "us", "vs", "ws", "ss"
            ]
        else:
            vars = [
                "s", "us", "vs", "ws", "ss", "sss", "ssss",
                "uss", "vss", "wss", "uus", "vvs", "wws","uvs", "uws", "vws", "ps",
                "pdsdx", "pdsdy", "pdsdz",
                "udsdx", "udsdy", "udsdz",
                "vdsdx", "vdsdy", "vdsdz",
                "wdsdx", "wdsdy", "wdsdz",
                "sdudx", "sdudy", "sdudz",
                "sdvdx", "sdvdy", "sdvdz",
                "sdwdx", "sdwdy", "sdwdz",
                "ess", "eus", "evs", "ews"
            ]

    col_names = ['time', height] + [var for var in vars]
    df.columns = col_names
    df_grouped = df.groupby(['time', height]).mean().reset_index()

    pivoted_data = {}
    for var in col_names[2:]:
        pivoted_data[var] = df_grouped.pivot(index='time', columns=height, values=var)

    ds = xr.Dataset({
        var: xr.DataArray(pivoted_data[var].values, dims=("time", height), 
                          coords={"time": pivoted_data[var].index, height: pivoted_data[var].columns})
        for var in col_names[2:]
    })

    if type == "fluid":
        fluid_stat = ds
    elif type == "scalar":
        if fluid_csv is None:
            raise TypeError("For scalar csv files, please provide the corresponding fluid csv file path.")
        fluid_stat = nk.csv_to_xr(fluid_csv, type="fluid", basic=basic, height=height)

    for var in vars:  # vars with gradients and e_ij are not treated
        if len(var) > 1 and "d" not in var:
            if len(var) == 2:
                if var[0] != "s":
                    ds[var] -= fluid_stat[var[0]] * ds[var[1]]
                else:
                    ds[var] -= ds[var[0]] * ds[var[1]]
            elif len(var) == 3:
                if var[0] != "s":
                    if var[1] != "s":
                        ds[var] -= fluid_stat[var[0]] * fluid_stat[var[1]] * ds[var[2]]
                    else:
                        ds[var] -= fluid_stat[var[0]] * ds[var[1]] * ds[var[2]]
                else:
                    ds[var] -= ds[var[0]] * ds[var[1]] * ds[var[2]]
            elif len(var) == 4:
                ds[var] -= ds[var[0]] ** 4

    if save:
        ds.to_netcdf(path.replace('.csv', '.nc'))
    return ds

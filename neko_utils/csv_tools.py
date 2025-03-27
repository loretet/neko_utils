# neko_utils/csv_tools.py

import pandas as pd
import xarray as xr

def csv_to_xr(path, basic=True, height="z"):
    """
    Converts csv Neko file into xarray DataSet.

    Input: path (str) = path to CSV file with Neko data
           basic (bool) = if True, the function assumes the csv file has the "basic"
                         variables from Neko documentation
           height (str) = name of the height coordinate in the xarray DataSet to create

    Output: xarray DataSet with the time (time) and height (z) as coordinates.
            Higher order statistics have the mean component removed.

    Usage: ds = csv_to_xr(/path/to/csv, basic=True, height="z")
    """

    df = pd.read_csv(path)
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

    for var in vars:  # vars with gradients and e_ij are not treated
        if len(var) > 1 and "d" not in var:
            if len(var) == 2:
                ds[var] -= ds[var[0]] * ds[var[1]]
            elif len(var) == 3:
                ds[var] -= ds[var[0]] * ds[var[1]] * ds[var[2]]
            elif len(var) == 4:
                ds[var] -= ds[var[0]] ** 4

    return ds

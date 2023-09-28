import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import xarray as xr
import numpy as np
# import datetime

VAR_CODES = {'Temperature': {'codes': ['TEMPS901', 'TEMPS601'], 'units': 'C'},
             'Salinity': {'codes': [], 'units': 'PSS-78'}}


def plot_cast_temp():
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)
    os.chdir(new_dir)

    # Add scatter of cast data
    cast_list = glob.glob('E:\\charles\\mooring_data_page\\e01\\cast_ctd_data\\' + '*.ctd.nc')
    cast_list.sort()

    # Initialize pandas dataframe to hold lat and lon for each file
    cast_coords = pd.DataFrame(columns=['File', 'Latitude', 'Longitude', 'Station'])

    fig, ax = plt.subplots(nrows=3)

    # split the time series into three segments
    subplot_years = [(1978, 1993), (1993, 2008), (2008, 2023)]

    myvar = 'Temperature'
    scale_factor = 0.1

    for i in range(0, len(cast_list)):  # take a subset for testing
        ds = xr.open_dataset(cast_list[i])
        cast_time = pd.Timestamp(ds.time.data)
        cast_year = cast_time.year
        # second arg is to get day of year
        cast_year_decimals = cast_year + cast_time.timetuple().tm_yday / 365
        if cast_year <= subplot_years[0][1]:
            k = 0
        elif subplot_years[1][0] < cast_year <= subplot_years[1][1]:
            k = 1
        elif subplot_years[2][0] < cast_year <= subplot_years[2][1]:
            k = 2
        else:
            print('Need to update year ranges')
        # Find the code for the select var e.g. temperature
        var_name = None
        for code in VAR_CODES[myvar]['codes']:
            if hasattr(ds, code):
                var_name = code
        if var_name is None:
            print('ds', myvar, 'code unknown !')
        # Split time series into 3 subplots, otherwise too long
        # Align the profile with its date by subtracting an adjustment
        ax[k].plot(cast_year_decimals + (ds[var_name].data - ds[var_name].data.max()) * scale_factor,
                   ds.depth.data)

    # Add labelling to the plot
    for k in [0, 1, 2]:
        ax[k].set_xlim(subplot_years[k])
        # xticks = ax[k].get_xticks()
        # print(xticks)
        # # Take 30 to be average month length
        # month - add 1 because e.g. the 6.452th day is in the 7th day
        # xlabels = [
        #     datetime.datetime.fromisoformat('{}-{:02d}-{:02d}'.format(y, m, d)) for y, m, d in
        #     zip(xticks.astype(int),
        #         (xticks % 1 * 365 / 30).astype(int) + 1,
        #         ((xticks % 1 * 365) % 30).astype(int) + 1)  # day
        # ]
        # # Convert xticks to datetime format
        # ax[k].set_xticks(ticks=xticks, labels=xlabels)
        # ax[k].set_xticks(ticks=ax[k].get_xticks(), minor=True)
        ax[k].tick_params(axis='x', which='both', direction='in')
        # ax[k].set_yticks(ticks=ax[k].get_yticks(), minor=True)
        ax[k].tick_params(axis='y', which='both', direction='in')
        ax[k].set_ylabel('Depth (m)')
        ax[k].set_ylim((120, 0))  # Invert the y axis, depth
    # ax[2].set_xlabel('{} ({})'.format(myvar, var_codes[myvar]['units']))
    ax[0].set_title('E01 - Cast CTD {} ({})'.format(myvar, VAR_CODES[myvar]['units']))

    plt.tight_layout()
    plt.savefig('.\\figures\\e01_cast_data_ts_T_all.png', dpi=300)
    plt.close(fig)

    os.chdir(old_dir)
    return


def plot_cast_sst(station: str):
    old_dir = os.getcwd()
    new_dir = os.path.dirname(old_dir)
    os.chdir(new_dir)

    # Add scatter of cast data
    cast_list = glob.glob(
        f'E:\\charles\\mooring_data_page\\{station.lower()}\\cast_ctd_data\\' + '*.ctd.nc'
    )
    cast_list.sort()

    myvar = 'Temperature'

    fig, ax = plt.subplots(nrows=2, sharex=True)

    # Initialize arrays to hold sst values so that they can be plotted as line plot instead of scatter
    year = np.repeat(np.nan, len(cast_list))
    sst = np.repeat(np.nan, len(cast_list))
    depth = np.repeat(np.nan, len(cast_list))

    for i in range(0, len(cast_list)):  # take a subset for testing
        ds = xr.open_dataset(cast_list[i])

        # Get the date in float year format
        cast_time = pd.Timestamp(ds.time.data)
        cast_year = cast_time.year
        # second arg is to get day of year
        cast_year_decimals = cast_year + cast_time.timetuple().tm_yday / 365

        # Find the code for the select var e.g. temperature
        var_name = None
        for code in VAR_CODES[myvar]['codes']:
            if hasattr(ds, code):
                var_name = code
        if var_name is None:
            print('ds', myvar, 'code unknown !')

        year[i] = cast_year_decimals
        sst[i] = ds[var_name].data[0]
        depth[i] = ds.depth.data[0]

        # # Plot the temperature in the first subplot and the depth in the lower subplot
        # ax[0].scatter(cast_year_decimals, ds[var_name].data[0], color='tab:orange',
        #               s=2)
        # ax[1].scatter(cast_year_decimals, ds.depth.data[0], color='tab:blue',
        #               s=2)

    indices_sorted = np.argsort(year)

    ax[0].scatter(year[indices_sorted], sst[indices_sorted], color='tab:orange', s=2,
                  marker='o')
    ax[1].scatter(year[indices_sorted], depth[indices_sorted], color='tab:blue', s=2,
                  marker='o')

    # format the plot
    ax[0].set_ylabel('{} ({})'.format(myvar, VAR_CODES[myvar]['units']))
    ax[1].set_ylabel('Depth (m)')
    ax[1].set_ylim((15, -1))

    for k in [0, 1]:
        ax[k].tick_params(axis='x', which='both', direction='in')
        # ax[k].set_yticks(ticks=ax[k].get_yticks(), minor=True)
        ax[k].tick_params(axis='y', which='both', direction='in')

    ax[0].set_title(station.upper(), loc='left')
    plt.tight_layout()
    plt.savefig(f'.\\{station.lower()}\\figures\\{station.lower()}_ctd_cast_data_sst.png', dpi=300)
    plt.close(fig)

    os.chdir(old_dir)
    return

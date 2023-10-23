import numpy as np
# import xarray as xr
import os
import glob
import ios_shell
from tqdm import trange
import pandas as pd

# ----------------------------osd_data_archive/Mooring_Data--------------------------

# Output a csv file containing the lat, lon, water depth, and years active for every
# mooring station in the osd_data_archive

# Go through all the files in osd_data_archive/Mooring_Data
# starting with just the recoveries going back to 2015

# Make a dictionary where each key is a unique station name
# and the value is a dictionary containing information pertinent to each station

# Take median lat/lon and water depth out of all available files from a station

mooring_dir = 'Y:\\Mooring_Data\\'

# output_df_name = ('C:\\Users\\HourstonH\\Documents\\mooring_tracking\\'
#                   'mooring_station_inventory_2015-present.csv')
output_df_name = ('C:\\Users\\HourstonH\\Documents\\mooring_tracking\\'
                  'mooring_station_inventory_all.csv')

# subdirs don't exist for 2016 and 2017
subdirs_new = [mooring_dir + f'{x}-recoveries\\' for x in np.arange(2015, 2024)]

subdirs_all = glob.glob(mooring_dir + '*\\')

# key: value, where value={latitude: a, longitude: b, water_depth:c, years_active:d}
station_dict = {}
# station_df = pd.DataFrame(
#     columns=['station', "latitude", 'longitude', 'qater_depth', 'years_active']
# )

# for subdir in subdirs:
#     if os.path.exists(subdir):
for subdir in subdirs_all:
    print(subdir)
    # search is not case-sensitive
    mctd_files = glob.glob(subdir + '*\\*.ctd')
    cur_files = glob.glob(subdir + '*\\*.cur')
    adcp_files = glob.glob(subdir + '*\\*.adcp')

    all_files = cur_files + mctd_files
    all_files.sort()

    # Open each file with a parser
    # If station not in station_dict yet, add it
    for i in trange(len(all_files)):
        f = all_files[i]

        # Skip historical files
        if 'History' in f or 'HISTORY' in f:
            continue

        # Get start year, end year from file name
        # Assume name format STN_YYYYMMDD_YYYYMMDD_DEPTHm.suffix
        try:
            year_st = int(os.path.basename(f).split('_')[1][:4])
            year_en = int(os.path.basename(f).split('_')[2][:4])
            years_active = [year_st, year_en] if year_st < year_en else [year_st]
        except IndexError:
            print('Invalid filename for file', f, 'skipping !!')
            continue

        # Read in IOS Shell format file
        try:
            parsed = ios_shell.ShellFile.fromfile(all_files[i])
        except ValueError:
            print('Possible unknown time format: UTC, in file', f,
                  '; skipping file for you to add later !!')
            continue

        station = parsed.location.station
        water_depth = parsed.location.water_depth
        latitude = parsed.location.latitude
        longitude = parsed.location.longitude

        # if station in station_df['station']:
        if station in station_dict.keys():
            # Add active years if not already present
            for year in years_active:
                if year not in station_dict[station]['years_active']:
                    station_dict[station]['years_active'].append(year)
        elif station.upper() in station_dict.keys():
            # Add active years if not already present
            for year in years_active:
                if year not in station_dict[station]['years_active']:
                    station_dict[station]['years_active'].append(year)
        else:
            station_dict[station] = {'latitude': latitude,
                                     'longitude': longitude,
                                     'water_depth': water_depth,
                                     'years_active': years_active}

# Sort the active years
for station in station_dict.keys():
    station_dict[station]['years_active'].sort()

# Convert the dict to a pandas dataframe
station_all = station_dict.keys()
latitude_all = [station_dict[k]['latitude'] for k in station_all]
longitude_all = [station_dict[k]['longitude'] for k in station_all]
water_depth_all = [station_dict[k]['water_depth'] for k in station_all]
years_active_all = [
    '|'.join([str(x) for x in station_dict[k]['years_active']]) for k in station_all
]

df_out = pd.DataFrame({'station': station_all,
                       'latitude': latitude_all,
                       'longitude': longitude_all,
                       'water_depth': water_depth_all,
                       'years_active': years_active_all})

# Sort the dataframe based on the station name
df_out.sort_values(by=['station'], inplace=True)

df_out.to_csv(output_df_name, index=False)

# ----------------------------osd_data_archive/netCDF_Data------------------------------

# # Station and water depth not available in the current meter nc files!
#
# nc_dir = 'Y:\\netCDF_Data\\'
#
# adcp_dir = nc_dir + 'ADCP\\'
# cur_dir = nc_dir + 'CUR\\'
# mctd_dir = nc_dir + 'mCTD\\'
#
# station_dict_nc = {}
#
# for xdir in [cur_dir, mctd_dir, adcp_dir]:
#     nc_list = glob.glob(xdir + '*\\*.nc', recursive=True)
#     nc_list.sort()
#     for f in nc_list:
#         stn = os.path.basename(f).split('_')[0]
#         ds = xr.open_dataset(f)
#         water_depth = None
#         if stn not in station_dict_nc:
#             pass
#         else:
#             pass

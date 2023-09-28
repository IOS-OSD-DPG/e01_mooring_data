import pandas as pd
import xarray as xr
import numpy as np
from gsw import z_from_p
import os

"""
One CUR file could not be parsed by ios_shell python package, so we must use its netCDF data
instead and add it to the dataframe of the rest of the CUR data later
"""

df_vars = ['Record_Number', 'Date', 'Time', 'Temperature', 'Salinity', 'Pressure',
           'Oxygen:Dissolved:SBE', 'Filename']

old_dir = os.getcwd()
new_dir = os.path.dirname(old_dir)
os.chdir(new_dir)

ncfile = 'E:\\charles\\e01_data_page\\data\\nc_alternatives\\e01_20210603_20220520_0035m_L1.cur.nc'

ncdata = xr.open_dataset(ncfile)

dfout = pd.DataFrame()

dfout['Record_Number'] = np.repeat(np.nan, len(ncdata.time.data))
dfout['Date'] = ncdata['time'].data.astype('datetime64[D]').astype(str)
dfout['Time'] = [dt[-8:] for dt in ncdata['time'].data.astype('datetime64[s]').astype(str)]
dfout['Temperature'] = ncdata['TEMPPR01'].data
dfout['Salinity'] = np.repeat(np.nan, len(ncdata.time.data))
dfout['Depth'] = -z_from_p(p=ncdata['PRESPR01'].data, lat=ncdata.latitude.data)
dfout['Oxygen:Dissolved:SBE'] = np.repeat(np.nan, len(ncdata.time.data))
dfout['Filename'] = np.repeat(ncfile, len(ncdata.time.data))

dfout.to_csv('.\\data\\e01_20210603_20220520_0035m_L1.cur.csv', index=False)

# Add to dataframe of all current meter files
cur_file = '.\\data\\e01_cur_data.csv'
df_cur = pd.read_csv(cur_file)
df_cur = pd.concat((df_cur, dfout))
df_cur.to_csv('.\\data\\e01_cur_data_all.csv', index=False)

os.chdir(old_dir)

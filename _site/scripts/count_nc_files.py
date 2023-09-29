import os
import pandas as pd

# Script to count the available current meter and moored CTD netCDF files
# from Water Properties
# Check if any IOS Shell format files don't have a corresponding netCDF file

shell_list_file = 'E:\\charles\\e01_data_page\\wget_file_download_list.csv'
nc_list_file = 'E:\\charles\\e01_data_page\\wget_netcdf_file_download_list.csv'

# Add name for column
shell_df = pd.read_csv(shell_list_file, header=None, names=['File'])
nc_df = pd.read_csv(nc_list_file, header=None, names=['File'])

# Add columns to each df containing the basename without file suffix
extract_basename = lambda x: os.path.basename(x).split('.')[0]

shell_df['basename'] = list(map(extract_basename, shell_df['File']))
nc_df['basename'] = list(map(extract_basename, nc_df['File']))

# Initialize a list to hold ios shell file names without corresponding nc file
missing_nc = []
for i in range(len(shell_df)):
    if shell_df.loc[i, 'basename'] not in nc_df.loc[:, 'basename'].tolist():
        missing_nc.append(shell_df.loc[i, 'File'])

print(len(shell_df))
print(len(nc_df))
print(len(missing_nc))
print(missing_nc)

"""
Two files without a corresponding netCDF file out of 183 CTD and CUR files with prefix E01*
'e01_20190801_20200718_0035m_L1.cur'
'E01_20100502_20100807_0035m.cur'
"""
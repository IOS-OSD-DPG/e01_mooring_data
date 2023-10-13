import ios_shell
import glob
import pandas as pd
import numpy as np
from tqdm import trange
from gsw import z_from_p

"""
Convert IOS shell format files to csv files using ios_shell and pandas
"""

list_of_stations = ['e01', 'a1', 'scott2', 'hak1', 'srn1']


def do_conversion(station: str):
    """
    Convert CUR and CTD ios shell files to csv format and merge them
    :param station:
    :return:
    """
    data_dir = f'E:\\charles\\mooring_data_page\\{station}\\ios_shell_data\\'
    output_dir = data_dir.replace('ios_shell_data', 'csv_data')

    # test_file = data_dir + 'E01A_19800507_19800913_0015m_L2.CUR'
    # par = ios_shell.ShellFile.fromfile(test_file)
    # location = ios_shell.ShellFile.get_location(par)
    # lon, lat = location.values()

    # Vars to keep
    # Can't convert Time:Code to Date and Time vars - calculate from start time and increment instead
    # No Oxygen in current meter files
    df_vars = ['Record_Number', 'Date', 'Time', 'Temperature', 'Salinity', 'Depth',
               'Oxygen:Dissolved:SBE']

    # unique_vars = []

    derived_depth_files = pd.DataFrame(columns=['Filename', 'Depth_is_static'], dtype='object')

    # old_dir = os.getcwd()
    # new_dir = os.path.dirname(old_dir)
    # os.chdir(new_dir)

    # Get both current meter and CTD data
    # Iterate through instruments
    for inst in ['cur', 'ctd']:
        # Don't add files in /definitely_wrong folder
        inst_file_list = glob.glob(data_dir + f'*.{inst}', recursive=False)
        inst_file_list.sort()

        # Initialize a dataframe to contain all instrument data
        inst_df = pd.DataFrame()

        for i in trange(len(inst_file_list)):
            # Read in IOS Shell format file
            try:
                parsed = ios_shell.ShellFile.fromfile(inst_file_list[i])
            except ValueError:
                print('Possible unknown time format: UTC, in file', inst_file_list[i],
                      '; skipping file for you to add later !!')
                continue

            # Get current meter start time and increment from file header

            # Convert the parsed file to pandas dataframe format
            obs_df = ios_shell.ShellFile.to_pandas(parsed)

            # Current meter data checks

            # Only applies to current meter data
            if 'Temperature:High_Res' in obs_df.columns and 'Temperature' not in obs_df.columns:
                obs_df.rename(columns={'Temperature:High_Res': 'Temperature'}, inplace=True)

            # Check if temperature or salinity are in current meter file
            if not any([x in obs_df.columns for x in ['Temperature', 'Salinity']]):
                print('Neither of temperature or salinity in file', inst_file_list[i])
                continue

            # Check for time data; if none then calculate from start time and time increment
            if not any([x in obs_df.columns for x in ['Date', 'Time']]):
                # Create time data using ios_shell
                obs_time = ios_shell.ShellFile.get_obs_time(parsed)
                # Convert to string 'Date' and 'Time columns
                obs_df['Date'] = [dt.strftime('%Y-%m-%d') for dt in obs_time]
                obs_df['Time'] = [dt.strftime('%H:%M:%S') for dt in obs_time]

            # Add static pressure data if no sensor
            # e.g., E01_19790507_19791010_0089m_L2.CUR doesn't have pressure
            if 'Depth' not in obs_df.columns:
                if 'Pressure' not in obs_df.columns:
                    depth_static = parsed.instrument.depth
                    obs_df['Depth'] = np.repeat(depth_static, len(obs_df))
                    # Add file to list for static pressure files
                    derived_depth_files.loc[len(derived_depth_files)] = [
                        inst_file_list[i], True
                    ]
                else:
                    # Calculate depth from time series pressure
                    # Get instrument depth, lat, lon
                    location = ios_shell.ShellFile.get_location(parsed)
                    lon, lat = location.values()
                    depth = -z_from_p(
                        p=obs_df.loc[:, 'Pressure'].to_numpy(),
                        lat=lat
                    )
                    obs_df['Depth'] = depth
                    # Add file to list for time series-depth-derived pressure files
                    derived_depth_files.loc[len(derived_depth_files)] = [
                        inst_file_list[i], False
                    ]

            # # Add to list of unique vars
            # for col in t.columns:
            #     if col not in unique_vars:
            #         unique_vars.append(col)

            # if 'Time:Code' in t.columns:
            #     print(inst_file_list[i])
            #     print(t.loc[:5, 'Time:Code'])

            # Remove not-needed columns, such as current meter direction and speed
            vars_to_drop = ['Direction: Geog(to)', 'Speed', 'Density', 'Speed:Sound',
                            'Speed:East', 'Speed:North', 'Speed:Up', 'Amplitude:Beam1',
                            'Amplitude:Beam2', 'Amplitude:Beam3', 'Heading', 'Pitch',
                            'Roll', 'Speed:Sound:1', 'Speed:Sound:2', 'Pressure:2',
                            'Speed:Current', 'Conductivity', 'Reference', 'Time:Code']
            # Ignore error messages if any vars are not present in a file
            obs_df.drop(columns=vars_to_drop, inplace=True, errors='ignore')

            # Reorder the columns to be the same order for every file
            # Add nan variables if some required ones are missing
            # No Oxygen in CUR files
            for var in df_vars:
                if var not in obs_df.columns:
                    obs_df[var] = np.repeat(np.nan, len(obs_df))
            # Reorder the columns
            obs_df = obs_df[df_vars]

            # Replace misc pad values with pandas nan
            sal_pads = [2.233, -99]
            temp_pads = [32.767, -99]
            oxy_pads = [-99]
            temp_mask = (
                (obs_df['Temperature'].values == temp_pads[0]) |
                (obs_df['Temperature'].values <= temp_pads[1])
            )
            sal_mask = (
                (obs_df['Salinity'].values == sal_pads[0]) |
                (obs_df['Salinity'].values <= sal_pads[1])
            )
            oxy_mask = (
                    obs_df['Oxygen:Dissolved:SBE'].values <= oxy_pads[0]
            )
            obs_df.loc[temp_mask, 'Temperature'] = pd.NA
            obs_df.loc[sal_mask, 'Salinity'] = pd.NA
            obs_df.loc[oxy_mask, 'Oxygen:Dissolved:SBE'] = pd.NA

            # Add file name as a column to the dataframe
            obs_df['Filename'] = np.repeat(inst_file_list[i], len(obs_df))

            # obs_df.to_csv(output_dir + os.path.basename(inst_file_list[i]) + '.csv', index=False)

            # Append the data to one big dataframe?
            inst_df = pd.concat((inst_df, obs_df), ignore_index=True)

        # Save ctd dataframe to csv file
        inst_df.to_csv(output_dir + f'{station}_{inst}_data.csv', index=False)
        print(f'{station}_{inst}_data.csv')

    # save list of static pressure files to csv
    derived_depth_files.to_csv(output_dir + f'{station}_cur_ctd_derived_depth.csv', index=False)
    return


# os.chdir(old_dir)

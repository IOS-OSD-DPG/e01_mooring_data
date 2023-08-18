import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import numpy as np
import datetime

VARS = ['Temperature', 'Salinity', 'Oxygen:Dissolved:SBE']
BIN_DEPTHS = [35, 75, 95]  # Bin the data to +/- 5m around each bin centre


def add_datetime(df: pd.DataFrame):
    """
    Add column containing datetime-format dates to pandas dataframe containing the observations
    :param df:
    :return:
    """
    # Need to replace the slash in YYYY/mm/dd format with a dash to comply with ISO format
    # so YYYY-mm-dd
    df['Date'] = [x.replace('/', '-') for x in df['Date']]
    df['Datetime'] = [
        datetime.datetime.fromisoformat(d + ' ' + t)
        for d, t in zip(df['Date'], df['Time'])
    ]
    return df


def plot_annual_samp_freq(df: pd.DataFrame, var: str, output_dir: str):
    """
    Plot histogram of annual counts of observations
    :param output_dir:
    :param df:
    :param var:
    :return:
    """
    # Make a mask for the specific variable and use data in it for the plot
    var_mask = ~df.loc[:, var].isna()
    # https://stackoverflow.com/questions/42379818/correct-way-to-set-new-column-in-pandas-dataframe-to-avoid-settingwithcopywarnin
    df_masked = df.loc[var_mask, :].copy()
    df_masked.reset_index(drop=True, inplace=True)

    # Get number of files - have both .CUR and .cur, and .CTD and .ctd file suffixes
    ctd_mask = [x.lower().endswith('.ctd') for x in df_masked.loc[:, 'Filename']]
    cur_mask = [x.lower().endswith('.cur') for x in df_masked.loc[:, 'Filename']]
    num_ctd_files = len(np.unique(df_masked.loc[ctd_mask, 'Filename']))
    num_cur_files = len(np.unique(df_masked.loc[cur_mask, 'Filename']))

    # Access datetime.datetime properties
    # df_masked['Month'] = [x.month for x in df_masked.loc[:, 'Datetime'].copy()]
    df_masked['Year'] = [x.year for x in df_masked.loc[:, 'Datetime'].copy()]
    min_year = df_masked['Year'].min()
    max_year = df_masked['Year'].max()
    num_bins = max_year - min_year + 1

    # Create a new 2d array to hold the data
    # Need to pad whichever record is shorter to match the length of the other one
    # between the current meter record and the ctd record
    arr_length = sum(ctd_mask) if sum(ctd_mask) > sum(cur_mask) else sum(cur_mask)
    padded_cur_arr = np.append(df_masked.loc[cur_mask, 'Year'].to_numpy(),
                               np.repeat(np.nan, arr_length - sum(cur_mask)))
    padded_ctd_arr = np.append(df_masked.loc[ctd_mask, 'Year'].to_numpy(),
                               np.repeat(np.nan, arr_length - sum(ctd_mask)))
    year_array = np.vstack((padded_cur_arr, padded_ctd_arr)).T  # Need transpose to get 2 columns for histogram

    # # Manually assign y-axis ticks to have only whole number ticks
    # num_yticks = df_masked['Year'].max()

    # if num_yticks < 10:
    #     yticks = np.arange(num_yticks + 1)

    plt.clf()  # Clear any active plots
    fig, ax = plt.subplots()  # Create a new figure and axis instance

    # Plot ctd data on top of cur data in different colours
    # ax.hist(df_masked.loc[:, 'Year'], bins=num_bins, align='left',
    #         label='Number of CUR files: {}\nNumber of CTD files: {}'.format(num_cur_files,
    #                                                                         num_ctd_files))
    ax.hist(year_array, bins=num_bins, align='left', stacked=True, histtype='bar',
            label=[f'Number of CUR files: {num_cur_files}',
                   f'Number of CTD files: {num_ctd_files}'],
            color=['orange', 'b'])

    ax.minorticks_on()
    ax.set_ylabel('Number of Measurements')
    plt.legend()

    plt.tight_layout()
    var = var.split(':')[0]  # Remove colons from oxygen
    plt.savefig(os.path.join(output_dir, f'e01_{var}_annual_sampling_counts.png'))
    plt.close(fig)
    return


def plot_monthly_samp_freq(df: pd.DataFrame, var: str, output_dir: str):
    """
    Plot monthly numbers of observations.
    Credit: James Hannah
    :param output_dir: path to folder for outputs
    :param df: pandas dataframe containing the raw data
    :param var: name of variable to use
    :return:
    """
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Make a mask for the specific variable and use data in it for the plot
    var_mask = ~df.loc[:, var].isna()
    # https://stackoverflow.com/questions/42379818/correct-way-to-set-new-column-in-pandas-dataframe-to-avoid-settingwithcopywarnin
    df_masked = df.loc[var_mask, :].copy()
    df_masked.reset_index(drop=True, inplace=True)

    # Access datetime.datetime properties
    df_masked['Month'] = [x.month for x in df_masked.loc[:, 'Datetime'].copy()]
    df_masked['Year'] = [x.year for x in df_masked.loc[:, 'Datetime'].copy()]
    min_year = df_masked['Year'].min()
    max_year = df_masked['Year'].max()
    year_range = max_year - min_year + 1

    # Initialize array to hold heatmap data
    monthly_counts = np.zeros(
        shape=(year_range, len(months)), dtype='int')

    # Populate the above array
    for i in range(year_range):
        for j in range(len(months)):
            monthly_counts[i, j] = sum(
                (df_masked['Year'].values == min_year + i) &
                (df_masked['Month'].values == j + 1))

    # Max counts for setting limit of plot colour bar
    max_counts = np.max(monthly_counts)

    plt.clf()  # Close any open active plots
    matplotlib.rc("axes", titlesize=25)
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)
    # Adjust figsize since oxy has shorted time series than temp and sal
    if var in ['Temperature', 'Salinity']:
        figsize = (40, 10)
    elif var == 'Oxygen:Dissolved:SBE':
        figsize = (9, 10)
    plt.figure(figsize=figsize, constrained_layout=True)

    # Display data as an image, i.e., on a 2D regular raster.
    plt.imshow(monthly_counts.T, vmin=0, vmax=max_counts, cmap="Blues")
    plt.yticks(ticks=range(12), labels=months)
    plt.xticks(
        ticks=range(0, year_range, 2),
        labels=range(min_year, max_year + 1, 2),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    # Add observation count to each cell in the grid
    for i in range(year_range):
        for j in range(len(months)):
            plt.text(
                i,  # position to place the text
                j,  # "                        "
                monthly_counts[i, j],  # the text (number of profiles)
                ha="center",
                va="center",
                color="k",
                fontsize="large",
            )

    plt.title(var, loc='left')
    plt.tight_layout()
    plt.colorbar()
    var = var.split(':')[0]
    output_file = os.path.join(output_dir, f'e01_{var}_monthly_sampling_counts.png')
    plt.savefig(output_file)
    plt.close()

    # Reset values
    matplotlib.rcdefaults()
    plt.axis("auto")

    return


def plot_raw_time_series(df: pd.DataFrame, output_dir: str):
    """
    Plot raw time series of temperature, salinity, and oxygen at selected depths
    Bin the data to 10m bins centred on the depths specified in BIN_DEPTHS
    Differentiate by colour the data from current meters and from CTDs
    :param df:
    :param output_dir:
    :return:
    """
    # Create masks based on instrument type
    # Get number of files - have both .CUR and .cur, and .CTD and .ctd file suffixes
    ctd_mask = np.array([x.lower().endswith('.ctd') for x in df.loc[:, 'Filename']])
    cur_mask = np.array([x.lower().endswith('.cur') for x in df.loc[:, 'Filename']])

    units = ['C', 'PSS-78', 'mL/L']
    y_axis_limits = [(4, 19), (26, 38), (0, 7.5)]
    for depth in BIN_DEPTHS:
        # Make a mask to capture data within 5 vertical meters of each bin depth
        depth_mask = (df.loc[:, 'Depth'].to_numpy() >= depth - 5) & (df.loc[:, 'Depth'].to_numpy() <= depth + 5)
        if depth == 75:  # No oxygen at this level for all time
            num_subplots = 2
            figsize = (10, 7)
        else:
            num_subplots = 3
            figsize = (10, 10)
        # Make plot with 3 subplots
        fig, ax = plt.subplots(num_subplots, figsize=figsize, sharex=True)
        for i, var in enumerate(VARS[:num_subplots]):
            # Plot CUR data
            ax[i].scatter(df.loc[cur_mask & depth_mask, 'Datetime'].to_numpy(),
                          df.loc[cur_mask & depth_mask, var].to_numpy(),
                          marker='.', s=2, c='orange', label='CUR')
            # Plot CTD data in a different colour
            ax[i].scatter(df.loc[ctd_mask & depth_mask, 'Datetime'].to_numpy(),
                          df.loc[ctd_mask & depth_mask, var].to_numpy(),
                          marker='.', s=2, c='b', label='CTD')
            ax[i].legend(loc='upper left', scatterpoints=3)  # Increase number of marker points in the legend
            ax[i].set_ylim(y_axis_limits[i])
            ax[i].set_ylabel(f'{var} ({units[i]})')
            ax[i].set_title(var.split(':')[0])
            # Make ticks point inward and on all sides
            ax[i].tick_params(which='major', direction='in',
                              bottom=True, top=True, left=True, right=True)
            ax[i].tick_params(which='minor', direction='in',
                              bottom=True, top=True, left=True, right=True)

        plt.suptitle(f'Station E01 - {depth} m')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'e01_raw_tso_{depth}m.png'))
        plt.close(fig)
    return


def plot_monthly_means(df: pd.DataFrame, output_dir: str):
    return


def plot_daily_means(df: pd.DataFrame, output_dir: str):
    """
    Calculate 1990-2020 climatology, which is 30 years because no data from 2007
    :return:
    """
    days_of_year = np.arange(1, 365 + 1)
    # Get integer day of year
    # df['Day_of_year'] = [datetime.datetime.timetuple(dt).tm_yday for dt in df.loc[:, 'Datetime']]

    obs_dates, indices = np.unique(df['Date'], return_index=True)
    obs_days_of_year = np.array([x.timetuple().tm_yday for x in df.loc[indices, 'Datetime']])

    for depth in BIN_DEPTHS:
        # Make a mask to capture data within 5 vertical meters of each bin depth
        depth_mask = (df.loc[:, 'Depth'].to_numpy() >= depth - 5) & (df.loc[:, 'Depth'].to_numpy() <= depth + 5)

        # Exclude oxygen for this work
        num_subplots = 2
        figsize = (10, 7)

        # Make plots with 3 subplots

        # Daily mean figure
        fig_dm, ax_dm = plt.subplots(num_subplots, figsize=figsize, sharex=True)

        # Daily climatology figure
        fig_dc, ax_dc = plt.subplots(num_subplots, figsize=figsize, sharex=True)

        # Daily anom figure
        fig_da, ax_da = plt.subplots(num_subplots, figsize=figsize, sharex=True)

        for k, var in enumerate(VARS[:num_subplots]):
            # Compute average for every day
            daily_means = np.zeros(len(obs_dates))

            for i, date in enumerate(obs_dates):
                daily_means[i] = df.loc[df['Date'] == date, var].mean()

            # Plot daily means
            ax_dm[k].scatter(df.loc[indices, 'Datetime'], daily_means, marker='.', s=2)
            ax_dm[k].set_title(var.split(':')[0])
            # todo add more plot formatting here

            # Compute the average over all time for each day in 1-365
            var_clim = np.zeros(len(days_of_year))
            for dy in days_of_year:
                # dy_mask = df.loc[:, 'Day_of_year'] == dy
                var_clim[dy - 1] = np.nanmean(daily_means[obs_days_of_year == dy])

    return


def plot_monthly_clim():
    return


def plot_daily_anom():
    return


def plot_monthly_anom():
    return


def run_plot(
        do_monthly_avail: bool = False,
        do_annual_avail: bool = False,
        do_raw_ts: bool = False,
        do_daily_mean_ts: bool = False,
        do_monthly_mean_ts: bool = False,
        do_daily_clim: bool = False,
        do_monthly_clim: bool = False,
        do_daily_anom: bool = False,
        do_monthly_anom: bool = False
):
    old_dir = os.getcwd()
    if os.path.basename(old_dir) == 'scripts':
        new_dir = os.path.dirname(old_dir)
    elif os.path.basename(old_dir) == 'e01_mooring_data':
        new_dir = old_dir
        old_dir = os.path.join(new_dir, 'scripts')
    os.chdir(new_dir)
    output_dir = os.path.join(new_dir, 'figures')

    file_list = ['.\\data\\e01_cur_data_all.csv', '.\\data\\e01_ctd_data.csv']
    df_all = pd.concat((pd.read_csv(file_list[0]), pd.read_csv(file_list[1])))
    df_dt = add_datetime(df_all)

    if do_monthly_avail:
        for var in VARS:
            plot_monthly_samp_freq(df_dt, var, output_dir)

    if do_annual_avail:
        for var in VARS:
            plot_annual_samp_freq(df_dt, var, output_dir)

    if do_raw_ts:
        plot_raw_time_series(df_dt, output_dir)

    # Reset the current directory
    os.chdir(old_dir)

    return

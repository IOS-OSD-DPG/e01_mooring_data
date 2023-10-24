"""
Using E01, Satellite SST around E01, and E01 SST from CTD profiles..

Purpose: applying a common MHW analysis to each time series.
Then we can see whether the surface MHW have a subsurface
expression. Subsurface will see El Nino and not much else.

method: do the MHW analysis to Satellite SST, CTD SST, and
T time series @ three different depths for a comparison of
the signals/linkages.
"""

import marineHeatWaves as mhw
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob
import pandas as pd
import xarray as xr
from scipy.signal import correlate

day2s = 1 * 24 * 60 * 60
YLIM = (-5, 20)  # standardize ylimits for temperature
station = 'E01'
data_dir = "E:\\charles\\mooring_data_page\\e01\\"


def plot_mhw_thresh(dates, temp, clim: dict, datatype: str, mhws: dict = None):
    """
    Plot the MHW threshold and the seasonal cycle on top of the data
    :param mhws:
    :param dates:
    :param temp:
    :param clim:
    :param datatype:
    :return:
    """
    # ev = np.argmax(mhws['intensity_max'])  # Find largest event

    plt.figure(figsize=(14, 7))
    # plt.subplot(2, 1, 1)
    # Plot temperature, seasonal cycle, and threshold
    plt.plot(dates, temp, 'k-', label='Daily mean temperature')
    plt.plot(dates, clim['thresh'], 'g-', label='Threshold')
    plt.plot(dates, clim['seas'], 'b-', label='Seasonal cycle')
    plt.suptitle('E01 MHW analysis - {}'.format(datatype.replace('_', ' ')))
    # plt.xlim(time_ord[0], time_ord[-1])
    plt.ylim(YLIM)
    if datatype == 'Mooring_35m' and mhws is not None:
        imax_ge_3_loc = np.where(np.array(mhws['intensity_max']) >= 3)[0]
        if len(imax_ge_3_loc) > 0:
            for k in imax_ge_3_loc:
                plt.axvline(x=mhws['date_start'][k], c='red', alpha=0.7)
        # xmin = pd.to_datetime('1995-01-01')
        # xmax = pd.to_datetime('1999-01-01')
        # # xmin = pd.to_datetime('2010-01-01')
        # # xmax = pd.to_datetime('2023-01-01')
        # plt.xlim((xmin, xmax))
    elif datatype == 'Mooring_75m' and mhws is not None:
        imax_ge_3_loc = np.where(np.array(mhws['intensity_max']) >= 3)[0]
        if len(imax_ge_3_loc) > 0:
            for k in imax_ge_3_loc:
                plt.axvline(x=mhws['date_start'][k], c='red', alpha=0.7)
        # # xmin = pd.to_datetime('1996-01-01')
        # # xmax = pd.to_datetime('2001-01-01')
        # xmin = pd.to_datetime('2014-01-01')
        # xmax = pd.to_datetime('2023-01-01')
        # plt.xlim((xmin, xmax))
    # elif datatype == 'Mooring_92m':
    #     xmin = pd.to_datetime('2012-01-01')
    #     xmax = pd.to_datetime('2022-10-01')
    #     plt.xlim((xmin, xmax))
    plt.ylabel(r'Temperature [$^\circ$C]')
    plt.legend(loc='upper left')

    # plt.subplot(2, 1, 2)  # A zoomed-in version of the upper subplot
    # # Find indices for all ten MHWs before and after event of interest and shade accordingly
    # for ev0 in np.arange(ev - 10, ev + 11, 1):
    #     t1 = np.where(time_ord == mhws['time_start'][ev0])[0][0]
    #     t2 = np.where(time_ord == mhws['time_end'][ev0])[0][0]
    #     plt.fill_between(dates[t1:t2 + 1], temp[t1:t2 + 1], clim['thresh'][t1:t2 + 1],
    #                      color=(1, 0.6, 0.5))
    # # Find indices for MHW of interest and shade accordingly
    # t1 = np.where(time_ord == mhws['time_start'][ev])[0][0]
    # t2 = np.where(time_ord == mhws['time_end'][ev])[0][0]
    # plt.fill_between(dates[t1:t2 + 1], temp[t1:t2 + 1], clim['thresh'][t1:t2 + 1],
    #                  color='r')
    # # Plot SST, seasonal cycle, threshold, shade MHWs with main event in red
    # plt.plot(dates, temp, 'k-', linewidth=2, label='Daily mean temperature')
    # plt.plot(dates, clim['thresh'], 'g-', linewidth=2, label='Threshold')
    # plt.plot(dates, clim['seas'], 'b-', linewidth=2, label='Seasonal cycle')
    # plt.legend(loc='upper left')
    # plt.xlim(mhws['time_start'][ev] - 150, mhws['time_end'][ev] + 150)
    # plt.ylim(clim['seas'].min() - 1, clim['seas'].max() + mhws['intensity_max'][ev] + 0.5)
    # plt.ylabel(r'SST [$^\circ$C]')

    # Save figure
    plt.tight_layout()
    plt.savefig(f'e01_mhw_thresh_{datatype.lower()}.png')
    # plt.savefig(f'e01_mhw_thresh_{datatype.lower()}_{xmin.year}-{xmax.year}.png')
    plt.close()
    return


def plot_mhw_stats_dist(mhws: dict, datatype: str):
    ev = np.argmax(mhws['intensity_max'])  # Find largest event

    plt.figure(figsize=(15, 7))
    # Duration
    plt.subplot(2, 2, 1)
    evMax = np.argmax(mhws['duration'])
    plt.bar(range(mhws['n_events']), mhws['duration'], width=0.6,
            color=(0.7, 0.7, 0.7))
    plt.bar(evMax, mhws['duration'][evMax], width=0.6, color=(1, 0.5, 0.5))
    plt.bar(ev, mhws['duration'][ev], width=0.6, edgecolor=(1, 0., 0.),
            color='none')
    plt.xlim(0, mhws['n_events'])
    plt.ylabel('[days]')
    plt.title('Duration')
    # Maximum intensity
    plt.subplot(2, 2, 2)
    evMax = np.argmax(mhws['intensity_max'])
    plt.bar(range(mhws['n_events']), mhws['intensity_max'], width=0.6,
            color=(0.7, 0.7, 0.7))
    plt.bar(evMax, mhws['intensity_max'][evMax], width=0.6, color=(1, 0.5, 0.5))
    plt.bar(ev, mhws['intensity_max'][ev], width=0.6, edgecolor=(1, 0., 0.),
            color='none')
    plt.xlim(0, mhws['n_events'])
    plt.ylabel(r'[$^\circ$C]')
    plt.title('Maximum Intensity')
    # Mean intensity
    plt.subplot(2, 2, 4)
    evMax = np.argmax(mhws['intensity_mean'])
    plt.bar(range(mhws['n_events']), mhws['intensity_mean'], width=0.6,
            color=(0.7, 0.7, 0.7))
    plt.bar(evMax, mhws['intensity_mean'][evMax], width=0.6, color=(1, 0.5, 0.5))
    plt.bar(ev, mhws['intensity_mean'][ev], width=0.6, edgecolor=(1, 0., 0.),
            color='none')
    plt.xlim(0, mhws['n_events'])
    plt.title('Mean Intensity')
    plt.ylabel(r'[$^\circ$C]')
    plt.xlabel('MHW event number')
    # Cumulative intensity
    plt.subplot(2, 2, 3)
    evMax = np.argmax(mhws['intensity_cumulative'])
    plt.bar(range(mhws['n_events']), mhws['intensity_cumulative'], width=0.6,
            color=(0.7, 0.7, 0.7))
    plt.bar(evMax, mhws['intensity_cumulative'][evMax], width=0.6, color=(1, 0.5, 0.5))
    plt.bar(ev, mhws['intensity_cumulative'][ev], width=0.6, edgecolor=(1, 0., 0.),
            color='none')
    plt.xlim(0, mhws['n_events'])
    plt.title(r'Cumulative Intensity')
    plt.ylabel(r'[$^\circ$C$\times$days]')
    plt.xlabel('MHW event number')

    plt.suptitle('E01 MHW stats - {}'.format(datatype.replace('_', ' ')))

    # Save fig
    plt.tight_layout()
    plt.savefig(f'e01_mhw_stats_dist_{datatype.lower()}.png')
    plt.close()

    return


def print_mhw_stats(mhws, obs_depth):
    """
    Print marine heat wave statistics
    :param obs_depth:
    :param mhws:
    :return:
    """
    print('Depth:', obs_depth)
    print('N events:', mhws['n_events'])
    ev = np.argmax(mhws['intensity_max'])  # Find largest event
    print('Maximum intensity:', mhws['intensity_max'][ev], 'deg. C')
    print('Average intensity:', mhws['intensity_mean'][ev], 'deg. C')
    print('Cumulative intensity:', mhws['intensity_cumulative'][ev], 'deg. C-days')
    print('Duration:', mhws['duration'][ev], 'days')
    print('Start date:', mhws['date_start'][ev].strftime("%d %B %Y"))
    print('End date:', mhws['date_end'][ev].strftime("%d %B %Y"))
    print()
    return


def plot_90p_diff(dates, diff: np.ndarray, datatype: str):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, diff, c='tab:blue')
    # add horizontal line at y=0
    plt.axhline(y=0, c='k', alpha=0.7)
    plt.ylim((-5, 5))
    plt.ylabel(r'Temperature diff. [$^\circ$C]')
    plt.suptitle('E01 Daily Mean T Minus 90th p. Thresh. - {}'.format(datatype.replace('_', ' ')))
    plt.tight_layout()
    plt.savefig(f'e01_mhw_mean_minus_thresh_{datatype.lower()}.png')
    plt.close()
    return


# --------------------Run mhw analysis on E01 cast data-----------------------


def run_cast():
    cast_files = glob.glob(data_dir + 'cast_ctd_data\\*.ctd.nc')
    cast_files.sort()

    ordinal_time = np.repeat(0, len(cast_files))
    date_time = np.repeat(pd.NaT, len(cast_files))
    sst = np.repeat(np.nan, len(cast_files))
    depth_cast = np.repeat(np.nan, len(cast_files))

    for i in range(len(cast_files)):
        ds = xr.open_dataset(cast_files[i])

        temp_var = None
        for code in ['TEMPS901', 'TEMPS601']:
            if hasattr(ds, code):
                temp_var = code

        ordinal_time[i] = int(pd.Timestamp(ds.time.values).toordinal())
        date_time[i] = pd.Timestamp(ds.time.values)
        sst[i] = ds[temp_var].data[0]
        depth_cast[i] = ds.depth.data[0]

    # sort the data by time
    indices_sorted = np.argsort(ordinal_time)
    ordinal_time = ordinal_time[indices_sorted]
    date_time = date_time[indices_sorted]
    sst = sst[indices_sorted]
    depth_cast = depth_cast[indices_sorted]

    # Run the analysis not by hand
    mhws_cast, clim_cast = mhw.detect(ordinal_time, sst)

    # print_mhw_stats(mhws_cast, 'surface')

    plot_mhw_thresh(date_time, sst, clim_cast, 'Cast_SST')

    # plot_mhw_stats_dist(mhws_cast, 'Cast_SST')

    # ------------------------Monthly means--------------------------------
    months = np.arange(1, 12 + 1)
    date_year = np.array([x.year for x in date_time])
    date_month = np.array([x.month for x in date_time])
    min_year = min(date_time).year
    max_year = max(date_time).year
    # initialize 2d array with shape (num_years, num_months)
    monthly_means = np.repeat(np.nan, (max_year - min_year + 1) * len(months)).reshape(
        max_year - min_year + 1, len(months)
    )

    for i in range(max_year - min_year + 1):
        for j in range(len(months)):
            mask = np.array([
                (date_year == min_year + i) & (date_month == months[j])
            ]).flatten()
            monthly_means[i, j] = np.nanmean(sst[mask])

    # Do analysis by hand
    monthly_clim = np.repeat(np.nan, len(months))
    p90 = np.repeat(np.nan, len(months))
    map_clim = np.repeat(np.nan, len(monthly_means.flatten())).reshape(monthly_means.shape)
    map_p90 = np.repeat(np.nan, len(monthly_means.flatten())).reshape(monthly_means.shape)
    for j in range(len(months)):
        monthly_clim[j] = np.nanmean(monthly_means[:, j])
        p90[j] = np.nanpercentile(monthly_means[:, j], 90)
        # map back
        map_clim[:, j] = monthly_clim[j]
        map_p90[:, j] = p90[j]

    # Run analysis by hand
    date_plot = np.repeat(pd.NaT, len(monthly_means.flatten())).reshape(monthly_means.shape)
    for i in range(max_year - min_year + 1):
        for j in range(len(months)):
            date_plot[i, j] = pd.Timestamp(min_year + i, j + 1, 15)

    # Remove NaNs?
    nan_mask = np.isnan(monthly_means.flatten())
    # monthly_means_final = monthly_means.flatten()[~nan_mask]
    # map_p90_final = map_p90.flatten()[~nan_mask]
    # map_clim_final = map_clim.flatten()[~nan_mask]
    # date_plot_final = date_plot.flatten()[~nan_mask]

    plt.figure(figsize=(14, 7))
    plt.plot(date_plot.flatten()[~np.isnan(monthly_means.flatten())],
             monthly_means.flatten()[~np.isnan(monthly_means.flatten())], 'k-', label='Monthly mean SST')
    plt.plot(date_plot.flatten()[~np.isnan(map_p90.flatten())],
             map_p90.flatten()[~np.isnan(map_p90.flatten())], 'g-', label='Threshold')
    plt.plot(date_plot.flatten()[~np.isnan(map_clim.flatten())],
             map_clim.flatten()[~np.isnan(map_clim.flatten())], 'b-', label='Monthly clim.')
    plt.suptitle('E01 MHW analysis - Cast CTD SST')
    plt.ylim(YLIM)
    plt.ylabel(r'SST [$^\circ$C]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('e01_mhw_thresh_cast_ctd_sst_no_nans_v2.png')
    plt.close()
    return


# -------------------Run mhw analysis on E01 mooring data----------------------

def run_mooring():
    bin_depths = [35, 75, 92]

    # convert datetime to integer type for the mhw package to interpret

    # Use daily mean data
    daily_means_file = 'C:/Users/HourstonH/mooring_data_page/e01/data/e01_daily_mean_TS_data.csv'
    df_daily_means = pd.read_csv(daily_means_file)
    df_daily_means['Datetime'] = [pd.to_datetime(x) for x in df_daily_means['Datetime']]
    df_daily_means['Ordinal_time'] = [x.toordinal() for x in df_daily_means['Datetime']]

    print_txt_stats = False
    do_plot1 = False
    do_plot_stats = False
    print_ge_3C_dates = False
    take_90p_diff = True

    for d in bin_depths[:2]:
        # Run the analysis
        # t: Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
        mhws_d, clim_d = mhw.detect(t=df_daily_means['Ordinal_time'].values,
                                    temp=df_daily_means[f'Temperature_{d}m'].values)

        # get start month-year for all big events for E01 at 35m and 75m

        if do_plot1:
            plot_mhw_thresh(
                df_daily_means['Datetime'].values,
                df_daily_means.loc[:, f'Temperature_{d}m'].values,
                clim_d,
                f'Mooring_{d}m',
                mhws_d
            )

        if do_plot_stats:
            plot_mhw_stats_dist(mhws_d, datatype=f'Mooring_{d}m')

        if print_txt_stats:
            # Print statistics
            print_mhw_stats(mhws_d, d)

        if print_ge_3C_dates:
            # Print start and end dates for marine heat waves with max intensity >= 3
            imax_gt_3_loc = np.where(np.array(mhws_d['intensity_max']) >= 3)[0]
            if len(imax_gt_3_loc) > 0:
                print('Depth:', d, 'm')
                for k in imax_gt_3_loc:
                    print(mhws_d['date_start'][k], mhws_d['date_end'][k])

        if take_90p_diff:
            diff = df_daily_means.loc[:, f'Temperature_{d}m'].values - clim_d['thresh']

            plot_90p_diff(df_daily_means['Datetime'].values, diff, datatype=f'Mooring_{d}m')

    # # Find when threshold is at its largest in the year - it is nov. 5th
    # idx_max = np.where(clim_d['thresh'] == max(clim_d['thresh']))[0]
    # dates_max_thresh = df_daily_means.loc[idx_max, 'Datetime']

    return


# --------------Run mhw analysis on satellite E01 data from Andrea---------------

# did not work on this monthly data -- need to calculate by hand


def run_daily_hires_sat():
    satellite_file = data_dir + 'satellite_data\\jplMURSST41_30d1_1b52_2af4_mean_3x3.csv'
    df = pd.read_csv(satellite_file)
    # convert time to datetime format
    df['Datetime'] = pd.to_datetime(df['time'])
    # convert datetime to ordinal time format
    df['Ordinal_time'] = [pd.Timestamp(x).toordinal() for x in df['Datetime'].values]
    # run the analysis
    mhws, clim = mhw.detect(t=df['Ordinal_time'].values, temp=df['mean_sst'].values)
    # Plot the analysis
    datatype = 'HiRes_Satellite_SST'
    plot_mhw_thresh(dates=df['Datetime'].values, temp=df['mean_sst'], clim=clim,
                    datatype=datatype, mhws=mhws)
    plot_mhw_stats_dist(mhws, datatype)
    plot_90p_diff(dates=df['Datetime'].values,
                  diff=df['mean_sst'].values-clim['thresh'],
                  datatype=datatype)
    return


def run_monthly_satellite():
    satellite_file = data_dir + 'satellite_data\\E01_monthly_avhrr_4km_SST.csv'

    df_sat = pd.read_csv(satellite_file)

    # Get required time formats
    df_sat['day'] = np.repeat(15, len(df_sat))  # midmonth
    df_sat['Datetime'] = pd.to_datetime(df_sat[['year', 'month', 'day']])
    # df_sat['Ordinal_time'] = [pd.Timestamp(x).toordinal() for x in df_sat['Datetime']]

    # mhws_sat, clim_sat = mhw.detect(df_sat.loc[:, 'Ordinal_time'].values,
    #                                 df_sat.loc[:, 'mean_SST'].values)

    # Compute climatology for whole time series
    month = np.arange(1, 12 + 1)
    monthly_clim_sat = np.zeros(len(month))

    # Compute 90th percentile for each day of year
    monthly_p90_sat = np.zeros(len(month))

    df_sat['monthly_clim'] = np.repeat(np.nan, len(df_sat))
    df_sat['90p'] = np.repeat(np.nan, len(df_sat))

    for i in range(len(month)):
        mask = df_sat['month'] == i + 1
        monthly_clim_sat[i] = df_sat.loc[mask, 'mean_SST'].mean()
        monthly_p90_sat[i] = np.nanpercentile(df_sat.loc[mask, 'mean_SST'], q=90)
        # Map the climatology and 90p values to the whole time series
        df_sat.loc[mask, 'monthly_clim'] = monthly_clim_sat[i]
        df_sat.loc[mask, '90p'] = monthly_p90_sat[i]

    # plot the data
    # Have mean, median, sd, min, and max sst for each month
    plt.figure(figsize=(14, 7))
    plt.plot(df_sat['Datetime'].values, df_sat['mean_SST'].values, 'k-', label='Monthly mean SST')
    plt.plot(df_sat['Datetime'].values, df_sat['90p'].values, 'g-', label='Threshold')
    plt.plot(df_sat['Datetime'].values, df_sat['monthly_clim'].values, 'b-', label='Monthly clim.')
    plt.suptitle('E01 MHW analysis - Satellite SST')
    plt.ylim(YLIM)
    plt.ylabel(r'SST [$^\circ$C]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('e01_mhw_thresh_sat_sst.png')
    plt.close()
    return


def do_plots_for_charles(t_sat, t_35m, t_75m, T_sat, T_35m, T_75m, are_diff: bool, do_zoom: bool):
    """
    Do plots for Charles
    :param do_zoom:
    :param are_diff:
    :param t_sat:
    :param t_35m:
    :param t_75m:
    :param T_sat:
    :param T_35m:
    :param T_75m:
    :return:
    """

    plt.figure(figsize=(14, 7))
    plt.plot(t_sat, T_sat, 'b-', label='Satellite SST')
    plt.plot(t_35m, T_35m, 'r-', label='Mooring 35m')
    plt.plot(t_75m, T_75m, 'y-', label='Mooring 75m')
    plt.legend(loc='upper left')
    # Zoom in on the last few years
    if do_zoom:
        left = pd.to_datetime('2020-01-01')
        right = pd.to_datetime('2023-12-31')
        plt.xlim(left=left, right=right)
    if are_diff:
        plt.ylim((-10, 10))
        plt.axhline(y=0, c='k')  # Add horizontal line at zero
        plt.ylabel(r'T minus 90th p. [$^\circ$C]')
        plt.title('E01 Temperature minus 90th percentile')
        img_name = 'E01_3_depths_T_minus_90th_p.png'
    else:
        plt.ylabel(r'Temperature [$^\circ$C]')
        plt.title('E01 Temperature')
        img_name = 'E01_3_depths_T.png'

    if do_zoom:
        img_name = img_name.replace('.png', f'_{left.year}-present.png')
    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()

    return


def charles_subplots(t_sat, t_35m, t_75m, T_sat, T_35m, T_75m, are_diff: bool):
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(14, 10))

    linewidth = 0.5
    ax[0].plot(t_sat, T_sat, 'b-', label='Satellite SST', linewidth=linewidth)
    ax[1].plot(t_35m, T_35m, 'r-', label='Mooring 35m', linewidth=linewidth)
    ax[2].plot(t_75m, T_75m, 'y-', label='Mooring 75m', linewidth=linewidth)

    left = pd.to_datetime('1989-10-01')  # just before 1990
    right = pd.to_datetime('2024-01-01')
    plt.xlim(left=left, right=right)

    mean_ts_ylim = YLIM
    diff_ts_ylim = (-18, 18)

    ts_ylim = diff_ts_ylim if are_diff else mean_ts_ylim

    for i in range(3):
        ax[i].set_ylim(ts_ylim)
        if are_diff:
            ax[i].legend(loc='upper right')
            ax[i].axhline(y=0, c='k')  # Add horizontal line at zero
        else:
            ax[i].legend(loc='upper left')
        ax[i].set_ylabel(r'T minus 90th p. [$^\circ$C]')
        ax[i].tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)

    if are_diff:
        plt.suptitle('E01 Temperature minus 90th percentile')
        img_name = 'E01_3_depths_T_minus_90th_p_separate.png'
    else:
        plt.suptitle('E01 Temperature')
        img_name = 'E01_3_depths_T_separate.png'

    plt.tight_layout()
    plt.savefig(img_name)
    plt.close()
    return


def main_plots_for_charles(
        plot_daily_means: bool = False,
        plot_diffs_together: bool = False,
        plot_separate: bool = False,
        do_zoom: bool = False
):
    # For 1990 to present, SST, 35 m, 75 m

    # # HiRes satellite data
    #
    # satellite_file = data_dir + 'satellite_data\\jplMURSST41_30d1_1b52_2af4_mean_3x3.csv'
    # df_sat = pd.read_csv(satellite_file)
    # # convert time to datetime format
    # df_sat['Datetime'] = pd.to_datetime(df_sat['time'])
    # # convert datetime to ordinal time format
    # df_sat['Ordinal_time'] = [pd.Timestamp(x).toordinal() for x in df_sat['Datetime'].values]
    #
    # mask_1990_sat = df_sat['Datetime'].dt.year >= 1990

    satellite_file = data_dir + 'satellite_data\\nceiPH53sstn1day_18b4_ef55_4075_QC.nc'
    ds_sat = xr.open_dataset(satellite_file)
    ds_sat['datetime'] = ('time', pd.to_datetime(ds_sat.time.data))

    # # run the analysis
    # mhws_sat, clim_sat = mhw.detect(
    #     t=df_sat.loc[mask_1990_sat, 'Ordinal_time'].values,
    #     temp=df_sat.loc[mask_1990_sat, 'mean_sst'].values
    # )

    # mhws_sat, clim_sat = mhw.detect(t=df_sat['Ordinal_time'].values, temp=df_sat['mean_sst'].values)

    # Mooring data

    # convert datetime to integer type for the mhw package to interpret
    mooring_file = 'C:/Users/HourstonH/mooring_data_page/e01/data/e01_daily_mean_TS_data.csv'
    df_moor = pd.read_csv(mooring_file)
    df_moor['Datetime'] = [pd.to_datetime(x) for x in df_moor['Datetime']]
    # df_moor['Datetime'] = pd.to_datetime(df_moor['Datetime'], utc=True)
    df_moor['Ordinal_time'] = [x.toordinal() for x in df_moor['Datetime']]

    mask_1990_mr = [x.year >= 1990 for x in df_moor['Datetime'].values]
    # mask_1990_mr = df_moor['Datetime'].dt.year >= 1990

    # # Run the analysis
    # # t: Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
    # mhws_35, clim_35 = mhw.detect(
    #     t=df_moor.loc[mask_1990_mr, 'Ordinal_time'].values,
    #     temp=df_moor.loc[mask_1990_mr, 'Temperature_35m'].values
    # )
    #
    # mhws_75, clim_75 = mhw.detect(
    #     t=df_moor.loc[mask_1990_mr, 'Ordinal_time'].values,
    #     temp=df_moor.loc[mask_1990_mr, 'Temperature_75m'].values
    # )

    # Plot the daily mean data
    if plot_daily_means:
        # do_plots_for_charles(
        #     t_sat=df_sat.loc[mask_1990_sat, 'Datetime'].values,
        #     t_35m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
        #     t_75m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
        #     T_sat=df_sat.loc[mask_1990_sat, 'mean_sst'].values,
        #     T_35m=df_moor.loc[mask_1990_mr, 'Temperature_35m'].values,
        #     T_75m=df_moor.loc[mask_1990_mr, 'Temperature_75m'].values,
        #     are_diff=False,
        #     do_zoom=do_zoom
        # )
        do_plots_for_charles(
            t_sat=ds_sat.datetime.data,
            t_35m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            t_75m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            T_sat=ds_sat.sst.data,
            T_35m=df_moor.loc[mask_1990_mr, 'Temperature_35m'].values,
            T_75m=df_moor.loc[mask_1990_mr, 'Temperature_75m'].values,
            are_diff=False,
            do_zoom=do_zoom
        )

    # Compute the daily climatology for 1990-present
    # clim_st_sat, clim_en_sat = [2002, 2022]
    # clim_st_mr, clim_en_mr = [1990, 2022]

    # clim_sat = np.repeat(np.nan, 365)
    # clim_moor = np.repeat(np.nan, 365)

    # Initialize because we have to smooth later
    p90_sat = np.repeat(np.nan, 365)
    p90_moor_35m = np.repeat(np.nan, 365)
    p90_moor_75m = np.repeat(np.nan, 365)

    ds_sat['day_of_year'] = ('time', np.array([pd.to_datetime(x).day_of_year for x in ds_sat.time.data]))
    df_moor['day_of_year'] = [x.day_of_year for x in df_moor.loc[:, 'Datetime']]

    # df_sat['clim_value'] = np.repeat(np.nan, len(df_sat))
    # df_moor['clim_value'] = np.repeat(np.nan, len(df_moor))
    ds_sat['p90_value'] = ('time', np.repeat(np.nan, len(ds_sat.time.data)))
    df_moor['p90_value_35m'] = np.repeat(np.nan, len(df_moor))
    df_moor['p90_value_75m'] = np.repeat(np.nan, len(df_moor))

    for i in range(len(p90_sat)):
        # clim_sat[i] = df_sat.loc[df_sat['day_of_year'] == i + 1, 'mean_sst'].mean()
        # clim_moor[i] = df_moor.loc[df_moor['day_of_year'] == i + 1, 'mean_sst'].mean()

        p90_sat[i] = np.nanpercentile(ds_sat.sst.data[ds_sat.day_of_year.data == i + 1], 90)
        p90_moor_35m[i] = np.nanpercentile(
            df_moor.loc[df_moor['day_of_year'] == i + 1, 'Temperature_35m'],
            90
        )
        p90_moor_75m[i] = np.nanpercentile(
            df_moor.loc[df_moor['day_of_year'] == i + 1, 'Temperature_75m'],
            90
        )

    # Smooth the 90th percentile (default true with mhw.detect, using 31 width)
    smoothPercentileWidth = 31
    p90_sat_smooth = mhw.runavg(p90_sat, smoothPercentileWidth)
    p90_moor_smooth_35m = mhw.runavg(p90_moor_35m, smoothPercentileWidth)
    p90_moor_smooth_75m = mhw.runavg(p90_moor_75m, smoothPercentileWidth)

    for i in range(len(p90_sat)):
        # df_sat.loc[df_sat['day_of_year'] == i + 1, 'clim_value'] = clim_sat[i]
        # df_moor.loc[df_moor['day_of_year'] == i + 1, 'clim_value'] = clim_moor[i]
        ds_sat.p90_value.data[ds_sat.day_of_year.data == i + 1] = p90_sat_smooth[i]
        df_moor.loc[df_moor['day_of_year'] == i + 1, 'p90_value_35m'] = p90_moor_smooth_35m[i]
        df_moor.loc[df_moor['day_of_year'] == i + 1, 'p90_value_75m'] = p90_moor_smooth_75m[i]

    # Take the difference between the mean values and the smoothed 90th percentile

    # Plot the daily mean data minus 90th percentile
    ds_sat['mean_minus_p90'] = ('time', ds_sat.sst.data - ds_sat.p90_value.data)
    # df_sat['diff'] = np.repeat(np.nan, len(df_sat))
    # df_sat.loc[mask_1990_sat, 'diff'] = (
    #         df_sat.loc[mask_1990_sat, 'mean_sst'].values - df_sat.loc[mask_1990_sat, 'p90_value'].values
    # )
    df_moor['diff_35m'] = np.repeat(np.nan, len(df_moor))
    df_moor.loc[mask_1990_mr, 'diff_35m'] = (
            df_moor.loc[mask_1990_mr, 'Temperature_35m'].values - df_moor.loc[mask_1990_mr, 'p90_value_35m'].values
    )
    df_moor['diff_75m'] = np.repeat(np.nan, len(df_moor))
    df_moor.loc[mask_1990_mr, 'diff_75m'] = (
            df_moor.loc[mask_1990_mr, 'Temperature_75m'].values - df_moor.loc[mask_1990_mr, 'p90_value_75m'].values
    )

    if plot_diffs_together:
        do_plots_for_charles(
            t_sat=ds_sat.datetime.data,
            t_35m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            t_75m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            T_sat=ds_sat.mean_minus_p90.data,
            T_35m=df_moor.loc[mask_1990_mr, 'diff_35m'].values,
            T_75m=df_moor.loc[mask_1990_mr, 'diff_75m'].values,
            are_diff=True,
            do_zoom=do_zoom
        )

    if plot_separate:
        charles_subplots(
            t_sat=ds_sat.datetime.data,
            t_35m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            t_75m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            T_sat=ds_sat.sst.data,
            T_35m=df_moor.loc[mask_1990_mr, 'Temperature_35m'].values,
            T_75m=df_moor.loc[mask_1990_mr, 'Temperature_75m'].values,
            are_diff=False
        )
        charles_subplots(
            t_sat=ds_sat.datetime.data,
            t_35m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            t_75m=df_moor.loc[mask_1990_mr, 'Datetime'].values,
            T_sat=ds_sat.mean_minus_p90.data,
            T_35m=df_moor.loc[mask_1990_mr, 'diff_35m'].values,
            T_75m=df_moor.loc[mask_1990_mr, 'diff_75m'].values,
            are_diff=True
        )

    return


# ---------------------------Correlation----------------------------

# Use scipy.signal.correlation (better for arrays >1e5 length) or numpy.correlation


# ------------------------Old mooring unaveraged code-----------------

# mooring_files = [data_dir + f'csv_data\\{station.lower()}_cur_data_all.csv',
#                  data_dir + f'csv_data\\{station.lower()}_ctd_data.csv']
#
# # Only keep current meter data before 2007 since 2008 is when CTD data start
# cur_data_all = pd.read_csv(mooring_files[0])
# cur_data_pre2007 = cur_data_all.loc[cur_data_all['Date'].to_numpy() < '2007', :]
# df_merged = pd.concat((cur_data_pre2007, pd.read_csv(mooring_files[1])))
#
# # Reset the index in the dataframe
# df_merged.reset_index(drop=True, inplace=True)
#
# # Format Date correctly
# df_merged['Date'] = [x.replace('/', '-') for x in df_merged['Date']]
# df_merged['Timestamp'] = [
#         int(datetime.fromisoformat(d + ' ' + t).timestamp())
#         for d, t in zip(df_merged['Date'], df_merged['Time'])
#     ]

# # Add time zone
# for i in range(len(df_merged)):
#     try:
#         df_merged.loc[i, 'Datetime'] = df_merged.loc[i, 'Datetime'].timestamp()
#     except TypeError:
#         df_merged.loc[i, 'Datetime'] = df_merged.loc[i, 'Datetime'].timestamp()
#
# df_merged['s_since_epoch'] = [
#     (dt - datetime.utcfromtimestamp(0)).days * day2s + (dt - datetime.utcfromtimestamp(0)).seconds
#     for dt in df_merged['Datetime']
# ]

# # .toordinal(): Return proleptic Gregorian ordinal.  January 1 of year 1 is day 1.
# t = np.array([dt.toordinal() for dt in df_merged['Datetime']])

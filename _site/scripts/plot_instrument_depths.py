import matplotlib.pyplot as plt
import glob
import os
import numpy as np

# Preliminary plot
# Evaluate where the "standard" instrument depths are overall for each station

# Increase a1 max depth to account for case where mooring was hit and displaced to deep water
station_dict = {
    'e01': {'max_depth': 120, 'bin_depths': [35, 75, 95], 'bin_size': 10},
    'a1': {'max_depth': 620, 'bin_depths': [35, 100, 180, 300, 400, 490], 'bin_size': 10},
    'scott2': {'max_depth': 300, 'bin_depths': [40, 100, 150, 200, 280], 'bin_size': 20}
}

old_dir = os.getcwd()
parent_dir = os.path.dirname(old_dir)

for station in station_dict.keys():
    # station = 'a1'
    ybot = station_dict[station]['max_depth']

    shell_dir = f'E:\\charles\\mooring_data_page\\{station}\\ios_shell_data\\'
    output_dir = os.path.join(parent_dir, station.lower(), 'figures')
    files = glob.glob(shell_dir + '*.*')
    depths = np.array(
        [int(os.path.basename(x).split('_')[3].split('.')[0][:-1]) for x in files]
    )
    dep_years = np.array(
        [int(os.path.basename(x).split('_')[1][:4]) for x in files]
    )

    # Make masks for CUR vs CTD data
    # Capture both .ctd and .CTD if existing
    is_CTD = np.array([x.lower().endswith('.ctd') for x in files])

    fig, ax = plt.subplots()
    ax.scatter(dep_years[is_CTD], depths[is_CTD], label='CTD', markercolor='blue')
    ax.scatter(dep_years[~is_CTD], depths[~is_CTD], label='CUR', markercolor='orange')
    ax.legend(loc='upper left', ncols=2)

    # Add horizontal lines to show bin size and location
    half_bin_size = station_dict[station]['bin_size'] / 2
    for sbin in station_dict[station]['bin_depths']:
        ax.axhline(y=sbin - half_bin_size, color='lightgrey', linestyle='-', alpha=0.5)
        ax.axhline(y=sbin + half_bin_size, color='lightgrey', linestyle='-', alpha=0.5)

    ax.set_ylim((ybot, 0))
    ax.set_ylabel('Depth (m)')
    ax.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
    plt.title(station.upper(), loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{station}_cur_ctd_depths.png'))
    plt.close()

os.chdir(old_dir)

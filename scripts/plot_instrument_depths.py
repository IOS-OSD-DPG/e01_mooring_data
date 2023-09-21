import matplotlib.pyplot as plt
import glob
import os

# Preliminary plot
# Evaluate where the "standard" instrument depths are overall for each station

max_depths = {'e01': 120, 'a1': 520, 'scott2': 300}

for station in max_depths.keys():
    # station = 'a1'
    ybot = max_depths[station]

    data_dir = f'E:\\charles\\mooring_data_page\\{station}\\ios_shell_data\\'
    output_dir = data_dir.replace('ios_shell_data', 'csv_data')
    files = glob.glob(data_dir + '*.*')
    depths = [int(os.path.basename(x).split('_')[3].split('.')[0][:-1]) for x in files]
    dep_years = [int(os.path.basename(x).split('_')[1][:4]) for x in files]

    plt.scatter(dep_years, depths)
    plt.ylim((ybot, 0))
    plt.ylabel('Depth (m)')
    plt.title(station.upper(), loc='left')
    plt.tight_layout()
    plt.savefig(output_dir + f'{station}_cur_ctd_depths.png')
    plt.close()

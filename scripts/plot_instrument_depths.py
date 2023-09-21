import matplotlib.pyplot as plt
import glob
import os

# Preliminary plot
# Evaluate where the "standard" instrument depths are overall for each station

max_depths = {'e01': 120, 'a1': 520, 'scott2': 300}

old_dir = os.getcwd()
parent_dir = os.path.dirname(old_dir)

for station in max_depths.keys():
    # station = 'a1'
    ybot = max_depths[station]

    data_dir = f'E:\\charles\\mooring_data_page\\{station}\\ios_shell_data\\'
    output_dir = os.path.join(parent_dir, station.lower(), 'figures')
    files = glob.glob(data_dir + '*.*')
    depths = [int(os.path.basename(x).split('_')[3].split('.')[0][:-1]) for x in files]
    dep_years = [int(os.path.basename(x).split('_')[1][:4]) for x in files]

    fig, ax = plt.subplots()
    ax.scatter(dep_years, depths)
    ax.set_ylim((ybot, 0))
    ax.set_ylabel('Depth (m)')
    ax.tick_params(which='major', direction='in', bottom=True, top=True, left=True, right=True)
    plt.title(station.upper(), loc='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{station}_cur_ctd_depths.png'))
    plt.close()

os.chdir(old_dir)

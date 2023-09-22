from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Change working directory
old_wd = os.getcwd()
new_wd = os.path.dirname(old_wd)
os.chdir(new_wd)

nominal_coordinates_file = '.\\map\\nominal_station_coordinates.csv'
coordinates_df = pd.read_csv(nominal_coordinates_file, index_col=[0])

# Initialize plot
fig, ax = plt.subplots()

# Set up basemap
left_lon, bot_lat, right_lon, top_lat = [-133, 47, -120, 54.5]

m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))

m.drawcoastlines(linewidth=0.2)
m.drawparallels(np.arange(bot_lat, top_lat, 1), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(left_lon, right_lon, 2), labels=[0, 0, 0, 1])
# m.drawparallels(np.arange(bot_lat, top_lat, 0.3), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(left_lon, right_lon, 1), labels=[0, 0, 0, 1])
# m.drawmapboundary(fill_color='white')
m.fillcontinents(color='0.8')

# Convert data coordinates into plot coordinates
x, y = m(coordinates_df['Longitude'].values, coordinates_df['Latitude'].values)

# Use zorder to plot the points on top of the continents
m.scatter(x, y, marker='*', color='r', s=20, zorder=5)

# Label each point
pad = ' '
for i in range(len(coordinates_df)):
    ax.annotate(pad + coordinates_df.index[i], (x[i], y[i]), fontsize=8)

# Plot formatting
plot_name = '.\\map\\mooring_map.png'
plt.tight_layout()
plt.savefig(plot_name, dpi=300)  # Save at lower quality than dpi=400
plt.close(fig)

# Reset working directory
os.chdir(old_wd)

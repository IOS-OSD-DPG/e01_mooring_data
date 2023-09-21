from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os


# Change working directory
old_wd = os.getcwd()
new_wd = os.path.dirname(old_wd)
os.chdir(new_wd)

nominal_e01_coordinates = (-126.60352, 49.28833)
# -126 deg 36.2112 minutes
# 49 deg 17.2998 minutes

# Initialize plot
fig, ax = plt.subplots()

# Set up basemap
left_lon, bot_lat, right_lon, top_lat = [-133, 48, -120, 55]

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
x, y = m(nominal_e01_coordinates[0], nominal_e01_coordinates[1])

# Use zorder to plot the points on top of the continents
m.scatter(x, y, marker='*', color='r', s=20, zorder=5)

# Plot formatting
plot_name = '.\\figures\\e01_map.png'
plt.tight_layout()
plt.savefig(plot_name, dpi=300)  # Save at lower quality than dpi=400
plt.close(fig)

# Reset working directory
os.chdir(old_wd)

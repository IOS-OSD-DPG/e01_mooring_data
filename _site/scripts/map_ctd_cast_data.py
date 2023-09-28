from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import xarray as xr

# Change working directory
old_wd = os.getcwd()
new_wd = os.path.dirname(old_wd)
os.chdir(new_wd)

nominal_e01_coordinates = (-126.60352, 49.28833)

# Initialize plot
fig, ax = plt.subplots()

# Set up basemap
left_lon, bot_lat, right_lon, top_lat = [-127, 49, -126, 50]

m = Basemap(llcrnrlon=left_lon, llcrnrlat=bot_lat,
            urcrnrlon=right_lon, urcrnrlat=top_lat,
            projection='lcc',
            resolution='h', lat_0=0.5 * (bot_lat + top_lat),
            lon_0=0.5 * (left_lon + right_lon))

m.drawcoastlines(linewidth=0.2)
m.drawparallels(np.arange(bot_lat, top_lat, 0.2), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(left_lon, right_lon, 0.5), labels=[0, 0, 0, 1])
# m.drawparallels(np.arange(bot_lat, top_lat, 0.3), labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(left_lon, right_lon, 1), labels=[0, 0, 0, 1])
# m.drawmapboundary(fill_color='white')
m.fillcontinents(color='0.8')

# Convert data coordinates into plot coordinates
x, y = m(nominal_e01_coordinates[0], nominal_e01_coordinates[1])

# Use zorder to plot the points on top of the continents
m.scatter(x, y, marker='*', color='r', s=20, zorder=5)

# Add scatter of cast data
cast_list = glob.glob('E:\\charles\\e01_data_page\\cast_ctd_data\\' + '*.ctd.nc')
cast_list.sort()

# Initialize pandas dataframe to hold lat and lon for each file
cast_coords = pd.DataFrame(columns=['File', 'Latitude', 'Longitude', 'Station'])

for i in range(len(cast_list)):
    ds = xr.open_dataset(cast_list[i])
    cast_coords.loc[len(cast_coords)] = [cast_list[i], ds.latitude.data,
                                         ds.longitude.data]

# Convert data coordinates into plot coordinates
xx, yy = m(cast_coords.loc[:, 'Longitude'].values,
           cast_coords.loc[:, 'Latitude'].values)

m.scatter(xx, yy, marker='o', color='b', s=10, zorder=4.9)

# Plot formatting
plot_name = '.\\figures\\e01_cast_data_map.png'
plt.tight_layout()
plt.savefig(plot_name, dpi=300)  # Save at lower quality than dpi=400
plt.close(fig)

# Reset working directory
os.chdir(old_wd)

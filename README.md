# e01_mooring_data
Data page for DFO mooring station E01 in BC. This site is under construction and all figures are provisional.
The webpage can be found [here](https://ios-osd-dpg.github.io/e01_mooring_data/), or by navigating to Deployments > github-pages.

The following types of plots of temperature (T), salinity (S) and oxygen (O) data are presented:
* Histograms of yearly measurement counts
* Tables of monthly measurement counts
* Raw time series
* Daily mean time series
* Daily climatologies for the 30-year period 1990-2020 (no data from 2007)
* Daily mean anomaly time series
* Monthly mean time series (underway)
* Monthly climatologies for the 30-year period 1990-2020 (underway)
* Monthly mean anomaly time series (underway)

### Information for updating the webpage:

The raw data were obtained from [Water Properties](https://www.waterproperties.ca/) in IOS Shell-type format or netCDF format. The search criteria used were:
* filename = "e01*"
* file suffixes = "ctd", "cur"

A netCDF-format file was only used for one case where the corresponding IOS Shell file could not be parsed using the ios_shell Python package. 

The raw data were too large to save within this repository.

#### Scripts
count_nc_files.py: Compare the number of IOS Shell-format files with the number of netCDF file versions available from the "wget" CSV file download lists from Water Properties. This showed that there were two IOS Shell files without a netCDF version.

convert_cur_ctd_from_shell.py: Convert IOS Shell-format files (*.CUR and *.CTD) to CSV format. This didn't work for one file.

convert_nc_to_csv.py: Convert netCDF-format files to csv format. This script is for the one file that couldn't be converted from IOS Shell, since it had a netCDF version.

map_e01_location.py: Plot the nominal location of station E01 on a map.

plot_e01_moored_data.py: Script containing the rest of the plotting functions, all callable by the function called `run_plot()`.

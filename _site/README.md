# osd_mooring_data
Data page for DFO mooring stations E01, SCOTT2, and A1 in BC. This site is under construction and all figures are provisional.
The webpage can be found [here](https://ios-osd-dpg.github.io/mooring_data_page/), or by navigating to Deployments > github-pages.

The following quantities are plotted:
* Instrument depths over time
* Yearly & monthly measurement counts
* Raw time series separated by instrument type (CTD and current meter)
* Daily & monthly mean time series
* Daily & monthly climatologies
* Daily & monthly mean anomaly time series

### Information for updating the webpage

The raw data were obtained from [Water Properties](https://www.waterproperties.ca/) in IOS Shell-type format or netCDF format. The search criteria used were:
* filename = "e01*"
* file suffixes = "ctd", "cur" ("cur" searches for files with *.CUR and *.cur suffixes)

A netCDF-format file was only used for one case where the corresponding IOS Shell file could not be parsed using the ios_shell Python package. 

The raw data were too large to save within this repository, so they were stored locally. Only daily mean data are saved within this repository.

#### Scripts
count_nc_files.py: Compare the number of IOS Shell-format files with the number of netCDF file versions available from the "wget" CSV file download lists from Water Properties. This showed that there were two IOS Shell files without a netCDF version.

convert_cur_ctd_from_shell.py: Convert IOS Shell-format files (*.CUR and *.CTD) to CSV format. This didn't work for one file.

convert_nc_to_csv.py: Convert netCDF-format files to csv format. This script is for the one file that couldn't be converted from IOS Shell, since it had a netCDF version.

map_e01_location.py: Plot the nominal location of station E01 on a map.

plot_e01_moored_data.py: Script containing the rest of the plotting functions, all callable by the function called `run_plot()`.

#### Viewing Changes Locally
The site is generated using [Jekyll](https://jekyllrb.com), which is the default static site generator for [GitHub Pages](https://pages.github.com).
Because the site is generated using templating, opening `index.html` directly in a browser is not going to produce great results, and will look nothing like the finished product.
In order to view your changes locally, you need to install Jekyll and then run `jekyll serve` from the root of the project (the same directory as this readme).
This will process the template and compile the site, and allow you to view it.

Running `jekyll serve` should result in something like the following:

	$ jekyll serve
	...
		Server address: http://127.0.0.1:4000
	  Server running... press ctrl-c to stop.

If you see this, just type `http://127.0.0.1:4000` into your browser and hit enter.

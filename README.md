# larva-locater

#### Active development takes place in the Test branch :)

# analysis.ipynb

Active work with the plotting and cleanup scripts, to be consulted for examples of how to run these scripts.

# cleanup.py

Used to pre-process the output of Tgrabs+Trex tracking into a Pandas dataframe, for use in plotting. Currently also supports linear interpolation of missing data, grouping single larvae artificially and calculating WIP metrics such as polarization or Directed Hausdorff Distance between a pair of conditions.

# plotting.py

Has a function for most common representations of larval tracking data and some novel ones such as time-to-odour. Full documentation is alongside each function and they are generally standalone given a dataframe processed by cleanup.py.

 

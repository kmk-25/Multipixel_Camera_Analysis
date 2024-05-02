# Multipixel_Camera_Analysis


### To import package, use pip install -e [path_to_folder]/Multipixel_Camera_Analysis

This package is designed to help work with camera data stored in h5 files.  It lets you: 
- Calculate the fourier transform for each individual pixel of a camera data stream (recorded as arrays in an h5 file), 
- Calculate the relative phase at any given 
- Given any arbitrary map, calculate the sum of each image in the camera datastream multiplied by that map
- Calculate the psd of a datastream of a single value
- Generate a new h5 file with a normalized subset of pixels from an existing camera datastream
- Plot the transfer function for x and y frequency combs, and normalize to force units

### For example usage, see jupyter notebook Exampleusage.ipynb
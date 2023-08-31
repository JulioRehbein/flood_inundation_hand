import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import earthpy.plot as eplt
import earthpy.spatial as espat
import floodtools as ft


demfile = os.path.join("dem_merged","merged.tif")
import numpy as np

#Calculates the area of grid cells if you don't have an area mask and only have the latitudes and longitudes of the centres of the cells. The function should work in situations both with equal grid spacing globally
# and with variable grid spacing.
#The input parameters are numpy arrays or xarray dataarrays of the latitudes of the center of cells and the longitudes of the center of cells in degrees
#Modified version of an answer on this post https://gis.stackexchange.com/questions/232813/easiest-way-to-create-an-area-raster so that the function works with variable grid spacing as well as equal grid spacing


def area_grid (latitudes,longitudes):
    latitudes=np.array(latitudes)
    # Switch to radians
    lats = np.deg2rad(latitudes)
    r_sq = 6371000**2 #(radius earth in metres)^2
    n = int(np.size(longitudes)) 
    area = r_sq*np.ones(n)[:, None]*np.deg2rad(np.diff(longitudes)[1])*(
                np.sin(lats+0.5*np.deg2rad(np.append(np.diff(latitudes)[0],np.diff(latitudes)))) - np.sin(lats-0.5*np.deg2rad(np.append(np.diff(latitudes)[0],np.diff(latitudes))))) #lat and lon represents middle of grid cell
    return area.T
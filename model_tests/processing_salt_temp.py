## This file saves salt and temperature fields from 34 members of the CESM large ensemble into lists where each list member is regridded to the 1x1 grid needed


# modules needed


import netCDF4
import xarray as xr
import numpy as np
import copy
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter #see https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
import xesmf as xe

## Below we open salt fields
#Open salt fields pre 2005
#files can be downloaded from https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.cesmLE.ocn.proc.monthly_ave.SALT/file.html
salt=[]
for i in range(2,36):
    if i<10:
        f="/scratch/abf376/CESM_data/SALT/b.e11.B20TRC5CNBDRD.f09_g16.00%d.pop.h.SALT.192001-200512.nc" % i
    else:
        f="/scratch/abf376/CESM_data/SALT/b.e11.B20TRC5CNBDRD.f09_g16.0%d.pop.h.SALT.192001-200512.nc" % i
    s=xr.open_dataset(f)['SALT']
    salt.append(s[:,0,:,:])
    print(i)


#Open salt fields post 2005
salt_2005on=[]
for i in range(2,36):
    if i<10:
        f="/scratch/abf376/CESM_data/SALT/b.e11.BRCP85C5CNBDRD.f09_g16.00%d.pop.h.SALT.200601-208012.nc" % i
    elif i<34:
        f="/scratch/abf376/CESM_data/SALT/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.pop.h.SALT.200601-208012.nc" % i
    else:
        f="/scratch/abf376/CESM_data/SALT/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.pop.h.SALT.200601-210012.nc" % i
    s=xr.open_dataset(f)['SALT']
    salt_2005on.append(s[:,0,:,:])

## Open temperature fields
#files can be downloaded from https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.cesmLE.ocn.proc.monthly_ave.TEMP.html
#Open temp fields pre 2005
temp=[]
for i in range(2,36):
    if i<10:
        f="/scratch/abf376/CESM_data/TEMP/b.e11.B20TRC5CNBDRD.f09_g16.00%d.pop.h.TEMP.192001-200512.nc" % i
    else:
        f="/scratch/abf376/CESM_data/TEMP/b.e11.B20TRC5CNBDRD.f09_g16.0%d.pop.h.TEMP.192001-200512.nc" % i
    s=xr.open_dataset(f)['TEMP']
    temp.append(s[:,0,:,:])

#Open temp fields post 2005
temp_2005on=[]
for i in range(2,36):
    if i<10:
        f="/scratch/abf376/CESM_data/TEMP/b.e11.BRCP85C5CNBDRD.f09_g16.00%d.pop.h.TEMP.200601-208012.nc" % i
    elif i<34:
        f="/scratch/abf376/CESM_data/TEMP/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.pop.h.TEMP.200601-208012.nc" % i
    else:
        f="/scratch/abf376/CESM_data/TEMP/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.pop.h.TEMP.200601-210012.nc" % i
    s=xr.open_dataset(f)['TEMP']
    temp_2005on.append(s[:,0,:,:])

# Make a regridder object using xesmf package

ds_out = xe.util.grid_global(1, 1)
regridder_tocesm = xe.Regridder(salt_flux[0], ds_out, "bilinear",periodic=True)

##Regrid all the above fields
#Regrid salt pre 2005
regridded_salt=[]
for i in range(0,34):
    s = regridder_tocesm(salt[i])
    regridded_salt.append(s)

#Regrid salt post 2005
regridded_salt_2005on=[]
for i in range(0,34):
    s = regridder_tocesm(salt_2005on[i])
    regridded_salt_2005on.append(s)

#Regrid temp pre 2005
regridded_temp=[]
for i in range(0,34):
    s = regridder_tocesm(temp[i])
    regridded_temp.append(s)

#Regrid temp post 2005
regridded_temp_2005on=[]
for i in range(0,34):
    s = regridder_tocesm(temp_2005on[i])
    regridded_temp_2005on.append(s)

#Pickle the files
import pickle
with open("regridded_salt_1920to2005_historical", "wb") as fp:   #Pickling
    pickle.dump(regridded_salt, fp)

with open("regridded_salt_2006to2080_rcp8.5", "wb") as fp:   #Pickling
    pickle.dump(regridded_salt_2005on, fp)

with open("regridded_temp", "wb") as fp:   #Pickling
    pickle.dump(regridded_temp, fp)

with open("regridded_temp_2006to2080_rcp8.5", "wb") as fp:   #Pickling
    pickle.dump(regridded_temp_2005on, fp)
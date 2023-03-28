#This file saves salt fluxes from 34 members of the CESM large ensemble into regridded lists

# modules needed


import netCDF4
import xarray as xr
import numpy as np
import copy
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter #see https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
import xesmf as xe


# #Open salt flux members before 2005
#Files can be downloaded from https://www.earthsystemgrid.org/dataset/ucar.cgd.ccsm4.cesmLE.ocn.proc.monthly_ave.SFWF/file.html if you make an account and login

#files of the form b.e11.B20TRC5CNBDRD.f09_g16.!.pop.h.SFWF.192001-200512.nc where ! is the realization numberwith 00 before single digit numbers and 0 before double digit numbers
salt_flux=[]
for i in range(2,36):
    if i<10:
        f="/scratch/abf376/CESM_data/salt_surface_flux/b.e11.B20TRC5CNBDRD.f09_g16.00%d.pop.h.SFWF.192001-200512.nc" % i
    else:
        f="/scratch/abf376/CESM_data/salt_surface_flux/b.e11.B20TRC5CNBDRD.f09_g16.0%d.pop.h.SFWF.192001-200512.nc" % i
    s=xr.open_dataset(f)['SFWF']
    salt_flux.append(s[:,:,:])


# Open salt flux members after 2005
salt_flux_2005on=[]
for i in range(2,36):
    if i<10:
        f="/scratch/abf376/CESM_data/salt_surface_flux/b.e11.BRCP85C5CNBDRD.f09_g16.00%d.pop.h.SFWF.200601-208012.nc" % i
    elif i<34:
        f="/scratch/abf376/CESM_data/salt_surface_flux/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.pop.h.SFWF.200601-208012.nc" % i
    else:
        f="/scratch/abf376/CESM_data/salt_surface_flux/b.e11.BRCP85C5CNBDRD.f09_g16.0%d.pop.h.SFWF.200601-210012.nc" % i
    s=xr.open_dataset(f)['SFWF']


# Make a regridder object using xesmf package

ds_out = xe.util.grid_global(1, 1)
regridder_tocesm = xe.Regridder(salt_flux[0], ds_out, "bilinear",periodic=True)


# Regrid the salt_flux list containing 1920 to 2005 salt fluxes


#regrid the pre2005 fluxes (1920-2005)
regridded=[]
for i in range(0,34):
    s = regridder_tocesm(salt_flux[i])
    regridded.append(s)


# Regrid the post2005 fluxes
regridded_2005on=[]
for i in range(0,34):
    s = regridder_tocesm(salt_flux_2005on[i])
    regridded_2005on.append(s)


# Pickle the fluxes


import pickle
with open("regridded_salt_flux_historical", "wb") as fp:   #Pickling
    pickle.dump(regridded, fp)
    
with open("regridded_salt_flux_2005on", "wb") as fp:   #Pickling
    pickle.dump(regridded_2005on, fp)


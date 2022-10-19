############
#The functions in this document are used to regrid the FAFMIP functions. The first part of each function puts it on the same grid as the salt field that is given (from the dataset that we will apply linear response theory to)
#The second part of each function finds the mean salinity over time in each region that was identified by fitting a GMM to the input data salt field. Thus, these functions need to be applied after clustering of the salinity from the dataset that will have linear response theory applied to it.

#The first two functions in this file act on the ocean only FAFMIP data - specifically MITgcm, HadOM3, ACCESS-OM2, and MOM5. The first function acts on salt and the second function acts on temperature.

#The last two functions in this file act on the coupled FAFMIP data - which is generally not used in this project. They are exactly comparable to the first two functions except they take in data from the coupled FAFMIP models.
###############

import scipy.io
import netCDF4
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, glob 
import imageio
from matplotlib import animation
import copy
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter #see https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
import certifi
import ssl
import math
ssl._create_default_https_context = ssl._create_unverified_context
from cmip_basins.basins import generate_basin_codes
import cmip_basins.gfdl as gfdl
import cmip_basins.cmip6 as cmip6
from scipy import stats
#import gsw
from xgcm import Grid
import statsmodels.api as sm
import matplotlib.ticker as ticker
from matplotlib.axes._secondary_axes import SecondaryAxis
#import percentiles_function
import xesmf as xe
from area_grid import *

def regridded_fafmip(salt,area,a2,n):
    #----------------------------------------------------
    #This function has input of:
        # - salt: salt field over the period of time of interest from the dataset of interest
        # - area: The area grid of the salt field of interest
        # - a2: the index locations in the vector of salinities x=np.linspace(31,38,10000) where each Gaussian starts and stops. Note this is output from the function clusters in clustering_tools.py!
        # - n: the number of clusters
    # This function has output of:
        # Timeseries of salt for each ocean only FAFMIP model (HadOM3, MITgcm, ACCESS-OM2, MOM5) for each individual perturbation experiment (stress, heat, water) in each cluster based on the clustering of the salt field of interest
    #-----------------------------------------------------
    ## OCEAN ONLY FAFMIP, SALT:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-all/so_yr_ACCESS-OM2_FAF-all_01-70.nc'
    salt_all=xr.open_dataset(f)['salt']
    salt_all=salt_all.where(salt_all !=9.969209968386869e+36) #get rid of weird values

    regridder_accesstocesm2 = xe.Regridder(salt_all[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_access = regridder_accesstocesm2(salt_all[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-all/so_yr_MOM5_FAF-all_01-70.nc'
    salt_all_mom=xr.open_dataset(f)['salt']
    salt_all_mom=salt_all_mom.where(salt_all_mom !=9.969209968386869e+36) #get rid of weird values

    regridder_momtocesm2 = xe.Regridder(salt_all_mom[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mom = regridder_momtocesm2(salt_all_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-all/so_yr_HadOM3_FAF-all_01-70.nc'
    salt_all=xr.open_dataset(f)['salt'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_all=salt_all.where(salt_all !=9.969209968386869e+36)
    salt_adjusted_all=salt_all*1000+35
    salt_adjusted_all=salt_adjusted_all.where(salt_adjusted_all>6)
    lat = xr.open_dataset(f)['latitude']
    lon = xr.open_dataset(f)['longitude']

    regridder_hadtocesm2 = xe.Regridder(salt_adjusted_all[:,0,:,:].where(salt_adjusted_all.latitude<65), salt.where(salt.latitude<65), "bilinear",periodic=True)
    regrid_surface_had = regridder_hadtocesm2(salt_adjusted_all[:,0,:,:].where(salt_adjusted_all.latitude<65))

    
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-stress/so_yr_ACCESS-OM2_FAF-stress_01-70.nc'
    salt_stress=xr.open_dataset(f)['salt']
    salt_stress=salt_stress.where(salt_stress !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_stress = regridder_accesstocesm2(salt_stress[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-stress/so_yr_MOM5_FAF-stress_01-70.nc'
    salt_stress_mom=xr.open_dataset(f)['salt']
    salt_stress_mom=salt_stress_mom.where(salt_stress_mom !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mom_stress = regridder_momtocesm2(salt_stress_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-stress/so_yr_HadOM3_FAF-stress_01-70.nc'
    salt_stress=xr.open_dataset(f)['salt'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_stress=salt_stress.where(salt_stress !=9.969209968386869e+36)
    salt_adjusted_stress=salt_stress*1000+35
    salt_adjusted_stress=salt_adjusted_stress.where(salt_adjusted_stress>6)

    regrid_surface_had_stress = regridder_hadtocesm2(salt_adjusted_stress[:,0,:,:].where(salt_adjusted_stress.latitude<65))


    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-stress/SALT_yr_faf-stress_CanESM2_10000-10100.nc'
    salt_stress_mit=xr.open_dataset(f)['SALT']
    salt_stress_mit=salt_stress_mit.where(salt_stress_mit !=9.969209968386869e+36) #get rid of weird values

    regridder_mittocesm2 = xe.Regridder(salt_stress_mit[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mit_stress = regridder_mittocesm2(salt_stress_mit[:,0,:,:])
    
    
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-water/so_yr_ACCESS-OM2_FAF-water_01-70.nc'
    salt_water=xr.open_dataset(f)['salt']
    salt_water=salt_water.where(salt_water !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_water = regridder_accesstocesm2(salt_water[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-water/so_yr_MOM5_FAF-water_01-70.nc'
    salt_water_mom=xr.open_dataset(f)['salt']
    salt_water_mom=salt_water_mom.where(salt_water_mom !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mom_water = regridder_momtocesm2(salt_water_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-water/so_yr_HadOM3_FAF-water_01-70.nc'
    salt_water=xr.open_dataset(f)['salt'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_water=salt_water.where(salt_water !=9.969209968386869e+36)
    salt_adjusted_water=salt_water*1000+35
    salt_adjusted_water=salt_adjusted_water.where(salt_adjusted_water>6)

    regrid_surface_had_water = regridder_hadtocesm2(salt_adjusted_water[:,0,:,:].where(salt_adjusted_water.latitude<65))


    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-water/SALT_yr_faf-water_CanESM2_10000-10100.nc'
    salt_water_mit=xr.open_dataset(f)['SALT']
    salt_water_mit=salt_water_mit.where(salt_water_mit !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_mit_water = regridder_mittocesm2(salt_water_mit[:,0,:,:])

    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-heat/so_yr_ACCESS-OM2_FAF-heat_01-70.nc'
    salt_heat=xr.open_dataset(f)['salt']
    salt_heat=salt_heat.where(salt_heat !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_heat = regridder_accesstocesm2(salt_heat[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-heat/so_yr_MOM5_FAF-heat_01-70.nc'
    salt_heat_mom=xr.open_dataset(f)['salt']
    salt_heat_mom=salt_heat_mom.where(salt_heat_mom !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mom_heat = regridder_momtocesm2(salt_heat_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-heat/so_yr_HadOM3_FAF-heat_01-70.nc'
    salt_heat=xr.open_dataset(f)['salt'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_heat=salt_heat.where(salt_heat !=9.969209968386869e+36)
    salt_adjusted_heat=salt_heat*1000+35
    salt_adjusted_heat=salt_adjusted_heat.where(salt_adjusted_heat>6)

    regrid_surface_had_heat = regridder_hadtocesm2(salt_adjusted_heat[:,0,:,:].where(salt_adjusted_heat.latitude<65))

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-heat/SALT_yr_faf-heat_CanESM2_10000-10100.nc'
    salt_heat_mit=xr.open_dataset(f)['SALT']
    salt_heat_mit=salt_heat_mit.where(salt_heat_mit !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_mit_heat = regridder_mittocesm2(salt_heat_mit[:,0,:,:])

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()

    x=np.linspace(31,38,10000)
    #let's find the change in salinity in each of these regions over the 70 years
    s=(salt[0:36,:,:].mean('time')).where(salt.latitude<65)
    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_access_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_access_water[j,:,:]).where(regrid_surface_access_water.latitude<65)
        for i in range(0,n):
            salt_access_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_had_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_water[j,:,:])
        for i in range(0,n):
            salt_had_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)



    salt_mom_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom_water[j,:,:])
        for i in range(0,n):
            salt_mom_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    salt_mit_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_water[j,:,:])
        for i in range(0,n):
            salt_mit_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_access_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_access_heat[j,:,:]).where(regrid_surface_access_heat.latitude<65)
        for i in range(0,n):
            salt_access_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_had_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_heat[j,:,:])
        for i in range(0,n):
            salt_had_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)



    salt_mom_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom_heat[j,:,:])
        for i in range(0,n):
            salt_mom_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    salt_mit_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_heat[j,:,:])
        for i in range(0,n):
            salt_mit_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_access_stress=np.empty([70,n])
    for j in range(0,67): #this is only 67 long
        s_new=(regrid_surface_access_stress[j,:,:]).where(regrid_surface_access_stress.latitude<65)
        for i in range(0,n):
            salt_access_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_had_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_stress[j,:,:])
        for i in range(0,n):
            salt_had_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)



    salt_mom_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom_stress[j,:,:])
        for i in range(0,n):
            salt_mom_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    salt_mit_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_stress[j,:,:])
        for i in range(0,n):
            salt_mit_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    return salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water


def regridded_fafmip_all_control(salt,area,a2,n):
    #----------------------------------------------------
    #This function has input of:
        # - salt: salt field over the period of time of interest from the dataset of interest
        # - area: The area grid of the salt field of interest
        # - a2: the index locations in the vector of salinities x=np.linspace(31,38,10000) where each Gaussian starts and stops. Note this is output from the function clusters in clustering_tools.py!
        # - n: the number of clusters
    # This function has output of:
        # Timeseries of salt for each ocean only FAFMIP model (HadOM3, MITgcm, ACCESS-OM2, MOM5) for each individual perturbation experiment (stress, heat, water) in each cluster based on the clustering of the salt field of interest
    #-----------------------------------------------------
    ## OCEAN ONLY FAFMIP, SALT:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-all/so_yr_ACCESS-OM2_FAF-all_01-70.nc'
    salt_all=xr.open_dataset(f)['salt']
    salt_all=salt_all.where(salt_all !=9.969209968386869e+36) #get rid of weird values

    regridder_accesstocesm2 = xe.Regridder(salt_all[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_access = regridder_accesstocesm2(salt_all[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-all/so_yr_MOM5_FAF-all_01-70.nc'
    salt_all_mom=xr.open_dataset(f)['salt']
    salt_all_mom=salt_all_mom.where(salt_all_mom !=9.969209968386869e+36) #get rid of weird values

    regridder_momtocesm2 = xe.Regridder(salt_all_mom[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mom = regridder_momtocesm2(salt_all_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-all/so_yr_HadOM3_FAF-all_01-70.nc'
    salt_all=xr.open_dataset(f)['salt'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_all=salt_all.where(salt_all !=9.969209968386869e+36)
    salt_adjusted_all=salt_all*1000+35
    salt_adjusted_all=salt_adjusted_all.where(salt_adjusted_all>6)
    lat = xr.open_dataset(f)['latitude']
    lon = xr.open_dataset(f)['longitude']
    
    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-all/SALT_yr_faf-all_CanESM2_10000-10100.nc'
    salt_all_mit=xr.open_dataset(f)['SALT']
    salt_all_mit=salt_all_mit.where(salt_all_mit !=9.969209968386869e+36) #get rid of weird values

    regridder_mittocesm2 = xe.Regridder(salt_all_mit[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mit = regridder_mittocesm2(salt_all_mit[:,0,:,:])
    

    regridder_hadtocesm2 = xe.Regridder(salt_adjusted_all[:,0,:,:].where(salt_adjusted_all.latitude<65), salt.where(salt.latitude<65), "bilinear",periodic=True)
    regrid_surface_had = regridder_hadtocesm2(salt_adjusted_all[:,0,:,:].where(salt_adjusted_all.latitude<65))


    ## OCEAN ONLY FAFMIP, CONTROL:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-control/so_yr_ACCESS-OM2_FAF-control_01-70.nc'
    salt_con=xr.open_dataset(f)['salt']
    salt_con=salt_con.where(salt_con !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_con = regridder_accesstocesm2(salt_con[:,0,:,:])


    f='/scratch/abf376/FAFMIP/HadOM3/FAF-control/so_yr_HadOM3_FAF-control_01-70.nc'
    salt_con=xr.open_dataset(f)['sea_water_salinity'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_con=salt_con.where(salt_con !=9.969209968386869e+36)
    lat = xr.open_dataset(f)['latitude']
    lon = xr.open_dataset(f)['longitude']

    regrid_surface_had_con = regridder_hadtocesm2(salt_con[:,0,:,:].where(salt_con.latitude<65))

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-control/SALT_yr_flux-only_CanESM2_10000-10100.nc'
    salt_con_mit=xr.open_dataset(f)['SALT']
    salt_con_mit=salt_con_mit.where(salt_con_mit !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mit_con = regridder_mittocesm2(salt_con_mit[:,0,:,:])


    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()

    x=np.linspace(31,38,10000)
    #let's find the change in salinity in each of these regions over the 70 years
    s=(salt[0:36,:,:].mean('time')).where(salt.latitude<65)

    salt_access_all=np.empty([70,n])
    for j in range(0,70): 
        s_new=(regrid_surface_access[j,:,:]).where(regrid_surface_access.latitude<65)
        for i in range(0,n):
            salt_access_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_had_all=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had[j,:,:])
        for i in range(0,n):
            salt_had_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_mom_all=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom[j,:,:])
        for i in range(0,n):
            salt_mom_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    salt_mit_all=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit[j,:,:])
        for i in range(0,n):
            salt_mit_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    #CONTROL
    salt_access_con=np.empty([70,n])
    for j in range(0,70): 
        s_new=(regrid_surface_access_con[j,:,:]).where(regrid_surface_access_con.latitude<65)
        for i in range(0,n):
            salt_access_con[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_had_con=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_con[j,:,:])
        for i in range(0,n):
            salt_had_con[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    salt_mit_con=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_con[j,:,:])
        for i in range(0,n):
            salt_mit_con[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    return salt_mit_all,salt_mom_all,salt_had_all,salt_access_all, salt_access_con, salt_had_con, salt_mit_con

def regridded_fafmip_temp(salt,area,a2,n):
    #----------------------------------------------------
    #This function has input of:
        # - salt: salt field over the period of time of interest from the dataset of interest
        # - area: The area grid of the salt field of interest
        # - a2: The vector categorizing salinity ranges for each cluster (output from the GMM functions which are run on the salt field of interest)
        # - n: the number of clusters
    # This function has output of:
        # Timeseries of surface temperature in each ocean only FAFMIP model (HadOM3, MITgcm, ACCESS-OM2, MOM5) for each individual perturbation experiment (stress, heat, water) in each cluster based on the clustering of the salt field of interest
    #NOTE: This function takes the same input as the function above (regridded_fafmip_salt), but returns time series of surface temperature rather than surface salinity
    #-----------------------------------------------------

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    ## OCEAN ONLY FAFMIP, GET REGRIDDERS:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-all/so_yr_ACCESS-OM2_FAF-all_01-70.nc'
    salt_all=xr.open_dataset(f)['salt']
    salt_all=salt_all.where(salt_all !=9.969209968386869e+36) #get rid of weird values

    regridder_accesstocesm2 = xe.Regridder(salt_all[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_access = regridder_accesstocesm2(salt_all[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-all/so_yr_MOM5_FAF-all_01-70.nc'
    salt_all_mom=xr.open_dataset(f)['salt']
    salt_all_mom=salt_all_mom.where(salt_all_mom !=9.969209968386869e+36) #get rid of weird values

    regridder_momtocesm2 = xe.Regridder(salt_all_mom[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mom = regridder_momtocesm2(salt_all_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-all/so_yr_HadOM3_FAF-all_01-70.nc'
    salt_all=xr.open_dataset(f)['salt'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    salt_all=salt_all.where(salt_all !=9.969209968386869e+36)
    salt_adjusted_all=salt_all*1000+35
    salt_adjusted_all=salt_adjusted_all.where(salt_adjusted_all>6)
    lat = xr.open_dataset(f)['latitude']
    lon = xr.open_dataset(f)['longitude']

    regridder_hadtocesm2 = xe.Regridder(salt_adjusted_all[:,0,:,:].where(salt_adjusted_all.latitude<65), salt.where(salt.latitude<65), "bilinear",periodic=True)
    regrid_surface_had = regridder_hadtocesm2(salt_adjusted_all[:,0,:,:].where(salt_adjusted_all.latitude<65))

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-stress/SALT_yr_faf-stress_CanESM2_10000-10100.nc'
    salt_stress_mit=xr.open_dataset(f)['SALT']
    salt_stress_mit=salt_stress_mit.where(salt_stress_mit !=9.969209968386869e+36) #get rid of weird values

    regridder_mittocesm2 = xe.Regridder(salt_stress_mit[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mit_stress = regridder_mittocesm2(salt_stress_mit[:,0,:,:])
    
    ## OCEAN ONLY FAFMIP, TEMP:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-stress/thetao_yr_ACCESS-OM2_FAF-stress_01-70.nc'
    temp_stress=xr.open_dataset(f)['temp']
    temp_stress=temp_stress.where(temp_stress !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_stress_temp = regridder_accesstocesm2(temp_stress[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-stress/thetao_yr_MOM5_FAF-stress_01-70.nc'
    temp_stress_mom=xr.open_dataset(f)['temp']
    temp_stress_mom=temp_stress_mom.where(temp_stress_mom !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mom_stress_temp = regridder_momtocesm2(temp_stress_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-heat/thetao_yr_ACCESS-OM2_FAF-heat_01-70.nc'
    temp_heat=xr.open_dataset(f)['temp']
    temp_heat=temp_heat.where(temp_heat !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_heat_temp = regridder_accesstocesm2(temp_heat[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-heat/thetao_yr_MOM5_FAF-heat_01-70.nc'
    temp_heat_mom=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_heat_mom=temp_heat_mom.where(temp_heat_mom !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mom_heat_temp = regridder_momtocesm2(temp_heat_mom[:,0,:,:])

    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-water/thetao_yr_ACCESS-OM2_FAF-water_01-70.nc'
    temp_water=xr.open_dataset(f)['temp']
    temp_water=temp_water.where(temp_water !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_water_temp = regridder_accesstocesm2(temp_water[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MOM5/FAF-water/thetao_yr_MOM5_FAF-water_01-70.nc'
    temp_water_mom=xr.open_dataset(f)['temp']
    temp_water_mom=temp_water_mom.where(temp_water_mom !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_mom_water_temp = regridder_momtocesm2(temp_water_mom[:,0,:,:])


    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-heat/THETA_yr_faf-heat_CanESM2_10000-10100.nc'
    temp_heat_mit=xr.open_dataset(f)['THETA']
    temp_heat_mit=temp_heat_mit.where(temp_heat_mit !=9.969209968386869e+36) #get rid of weird values
    temp_heat_mit=temp_heat_mit.where(temp_heat_mit>-30) #get rid of weird values
    regrid_surface_mit_heat_temp = regridder_mittocesm2(temp_heat_mit[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-water/THETA_yr_faf-water_CanESM2_10000-10100.nc'
    temp_water_mit=xr.open_dataset(f)['THETA']
    temp_water_mit=temp_water_mit.where(temp_water_mit !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_mit_water_temp = regridder_mittocesm2(temp_water_mit[:,0,:,:])

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-stress/THETA_yr_faf-stress_CanESM2_10000-10100.nc'
    temp_stress_mit=xr.open_dataset(f)['THETA']
    temp_stress_mit=temp_stress_mit.where(temp_stress_mit !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_mit_stress_temp = regridder_mittocesm2(temp_stress_mit[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-heat/thetao_yr_HadOM3_FAF-heat_01-70.nc'
    temp_heat_had=xr.open_dataset(f)['temp']
    temp_heat_had=temp_heat_had.where(temp_heat_had !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_had_heat_temp = regridder_hadtocesm2(temp_heat_had[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-water/thetao_yr_HadOM3_FAF-water_01-70.nc'
    temp_water_had=xr.open_dataset(f)['temp']
    temp_water_had=temp_water_had.where(temp_water_had !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_had_water_temp = regridder_hadtocesm2(temp_water_had[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadOM3/FAF-stress/thetao_yr_HadOM3_FAF-stress_01-70.nc'
    temp_stress_had=xr.open_dataset(f)['temp']
    temp_stress_had=temp_stress_had.where(temp_stress_had !=9.969209968386869e+36) #get rid of weird values
    regrid_surface_had_stress_temp = regridder_hadtocesm2(temp_stress_had[:,0,:,:])

    x=np.linspace(31,38,10000)
    #let's find the change in salinity in each of these regions over the 70 years
    s=(salt[0:36,:,:].mean('time')).where(salt.latitude<65)
    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    
    ##TEMP WATER

    #we have a 50 year time series, we want to find the mean temp at each region defined by the first year at each of these years
    temp_access_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_access_water_temp[j,:,:]).where(regrid_surface_access_water_temp.latitude<65)
        for i in range(0,n):
            temp_access_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)



    temp_mom_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom_water_temp[j,:,:]).where(regrid_surface_mom_water_temp.latitude<65)
        for i in range(0,n):
            temp_mom_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_mit_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_water_temp[j,:,:])
        for i in range(0,n):
            temp_mit_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_had_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_water_temp[j,:,:])
        for i in range(0,n):
            temp_had_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    ##TEMP STRESS

    #we have a 50 year time series, we want to find the mean temp at each region defined by the first year at each of these years
    temp_access_stress=np.empty([67,n])
    for j in range(0,67):
        s_new=(regrid_surface_access_stress_temp[j,:,:]).where(regrid_surface_access_stress_temp.latitude<65)
        for i in range(0,n):
            temp_access_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)



    temp_mom_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom_stress_temp[j,:,:]).where(regrid_surface_mom_stress_temp.latitude<65)
        for i in range(0,n):
            temp_mom_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_mit_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_stress_temp[j,:,:])
        for i in range(0,n):
            temp_mit_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_had_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_stress_temp[j,:,:])
        for i in range(0,n):
            temp_had_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    ##TEMP HEAT
    #we have a 50 year time series, we want to find the mean temp at each region defined by the first year at each of these years
    temp_access_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_access_heat_temp[j,:,:]).where(regrid_surface_access_heat_temp.latitude<65)
        for i in range(0,n):
            temp_access_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)



    temp_mom_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mom_heat_temp[j,:,:]).where(regrid_surface_mom_heat_temp.latitude<65)
        for i in range(0,n):
            temp_mom_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_mit_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_heat_temp[j,:,:])
        for i in range(0,n):
            temp_mit_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_had_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_heat_temp[j,:,:])
        for i in range(0,n):
            temp_had_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    return temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water
    
def regridded_fafmip_temp_all_control(salt,area,a2,n):
    #----------------------------------------------------
    #This function has input of:
        # - salt: salt field over the period of time of interest from the dataset of interest
        # - area: The area grid of the salt field of interest
        # - a2: The vector categorizing salinity ranges for each cluster (output from the GMM functions which are run on the salt field of interest)
        # - n: the number of clusters
    # This function has output of:
        # Timeseries of surface temperature in each ocean only FAFMIP model (HadOM3, MITgcm, ACCESS-OM2, MOM5) for each individual perturbation experiment (stress, heat, water) in each cluster based on the clustering of the salt field of interest
    #NOTE: This function takes the same input as the function above (regridded_fafmip_salt), but returns time series of surface temperature rather than surface salinity
    #-----------------------------------------------------

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    ## OCEAN ONLY FAFMIP, GET REGRIDDERS:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-all/thetao_yr_ACCESS-OM2_FAF-all_01-70.nc'
    temp_all=xr.open_dataset(f)['temp']
    temp_all=temp_all.where(temp_all !=9.969209968386869e+36) #get rid of weird values

    regridder_accesstocesm2 = xe.Regridder(temp_all[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_access = regridder_accesstocesm2(temp_all[:,0,:,:])


    f='/scratch/abf376/FAFMIP/HadOM3/FAF-all/thetao_yr_HadOM3_FAF-all_01-70.nc'
    temp_all=xr.open_dataset(f)['temp'] #function of time depth latitude and longitude. long_name: salinity (ocean) (psu-35)/1000
    temp_all=temp_all.where(temp_all !=9.969209968386869e+36)
    lat = xr.open_dataset(f)['latitude']
    lon = xr.open_dataset(f)['longitude']

    regridder_hadtocesm2 = xe.Regridder(temp_all[:,0,:,:].where(temp_all.latitude<65), salt.where(salt.latitude<65), "bilinear",periodic=True)
    regrid_surface_had = regridder_hadtocesm2(temp_all[:,0,:,:].where(temp_all.latitude<65))

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-all/THETA_yr_faf-all_CanESM2_10000-10100.nc'
    temp_all_mit=xr.open_dataset(f)['THETA']
    temp_all_mit=temp_all_mit.where(temp_all_mit !=9.969209968386869e+36) #get rid of weird values

    regridder_mittocesm2 = xe.Regridder(temp_all_mit[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_surface_mit = regridder_mittocesm2(temp_all_mit[:,0,:,:])


    ## OCEAN ONLY FAFMIP, CONTROL:
    f='/scratch/abf376/FAFMIP/ACCESS-OM2/FAF-control/thetao_yr_ACCESS-OM2_FAF-control_01-70.nc'
    temp_con=xr.open_dataset(f)['temp']
    temp_con=temp_con.where(temp_con !=9.969209968386869e+36) #get rid of weird values

    regrid_surface_access_con = regridder_accesstocesm2(temp_con[:,0,:,:])


    f='/scratch/abf376/FAFMIP/HadOM3/FAF-control/thetao_yr_HadOM3_FAF-control_01-70.nc'
    temp_con=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_con=temp_con.where(temp_con !=9.969209968386869e+36)
    temp_con=temp_con-273.15
    temp_con=temp_con.where(temp_con>-20)
    lat = xr.open_dataset(f)['latitude']
    lon = xr.open_dataset(f)['longitude']

    regrid_surface_had_con = regridder_hadtocesm2(temp_con[:,0,:,:].where(temp_con.latitude<65))

    f='/scratch/abf376/FAFMIP/MITGCM_v2/FAF-control/THETA_yr_flux-only_CanESM2_10000-10100.nc'
    temp_con_mit=xr.open_dataset(f)['THETA']
    temp_con_mit=temp_con_mit.where(temp_con_mit !=9.969209968386869e+36) #get rid of weird values
    temp_con_mit=temp_con_mit.where(temp_con_mit>-30)

    regrid_surface_mit_con = regridder_mittocesm2(temp_con_mit[:,0,:,:])
    
    
    x=np.linspace(31,38,10000)
    #let's find the change in salinity in each of these regions over the 70 years
    s=(salt[0:36,:,:].mean('time')).where(salt.latitude<65)
   
    ##TEMP ALL
    temp_access_all=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_access[j,:,:]).where(regrid_surface_access.latitude<65)
        for i in range(0,n):
            temp_access_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_mit_all=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit[j,:,:])
        for i in range(0,n):
            temp_mit_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_had_all=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had[j,:,:])
        for i in range(0,n):
            temp_had_all[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    ##TEMP CONTROL
    temp_access_con=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_access_con[j,:,:]).where(regrid_surface_access_con.latitude<65)
        for i in range(0,n):
            temp_access_con[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_mit_con=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_mit_con[j,:,:])
        for i in range(0,n):
            temp_mit_con[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    temp_had_con=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_surface_had_con[j,:,:])
        for i in range(0,n):
            temp_had_con[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    return temp_mit_all,temp_had_all,temp_access_all, temp_access_con, temp_mit_con, temp_had_con


def regridded_fafmip_coupled(salt,area,a2,n):
    #----------------------------------------------------
    #This function has input of:
        # - salt: salt field over the period of time of interest from the dataset of interest
        # - area: The area grid of the salt field of interest
        # - a2: The vector categorizing salinity ranges for each cluster (output from the GMM functions which are run on the salt field of interest)
        # - n: the number of clusters
    # This function has output of:
        # Timeseries of coupled FAFMIP models (HadGEM2, MPI-ESM, GFDL-ESM2M) for each individual perturbation experiment (stress, heat, water) in each cluster based on the clustering of the salt field of interest
    #NOTE: This function is equivalent to regridded_fafmip_salt, but returns time series of surface salinity from coupled FAFMIP models rather than ocean only FAFMIP models
    #-----------------------------------------------------

    ## COUPLED FAFMIP, SALT:
    ##WATER
    f='/scratch/abf376/FAFMIP/MPI-ESM/FAF-water/so_yr_MPI-ESM-LR-remap_FAF-water_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_water_mpi=xr.open_dataset(f)['sea_water_salinity']
    salt_water_mpi=salt_water_mpi.where(salt_water_mpi !=9.969209968386869e+36) #get rid of weird values

    regridder_mpitocesm2 = xe.Regridder(salt_water_mpi[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_water_mpi = regridder_mpitocesm2(salt_water_mpi[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadGEM2/FAF-water/so_yr_HadGEM2-ES_FAF-water_r1i1p1_185912-192911.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_water_hadgem=xr.open_dataset(f)['sea_water_salinity']
    salt_water_hadgem=salt_water_hadgem.where(salt_water_hadgem !=9.969209968386869e+36) #get rid of weird values
    salt_water_hadgem=salt_water_hadgem

    regridder_hadgemtocesm2 = xe.Regridder(salt_water_hadgem[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_water_hadgem = regridder_hadgemtocesm2(salt_water_hadgem[:,0,:,:])

    f='/scratch/abf376/FAFMIP/GFDL-ESM2M/FAF-water/so_yr_GFDL-ESM2M_FAF-water_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_water_gfdl=xr.open_dataset(f)['sea_water_salinity']
    salt_water_gfdl=salt_water_gfdl.where(salt_water_gfdl !=9.969209968386869e+36) #get rid of weird values
    salt_water_gfdl=salt_water_gfdl

    regridder_gfdltocesm2 = xe.Regridder(salt_water_gfdl[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_water_gfdl = regridder_gfdltocesm2(salt_water_gfdl[:,0,:,:])

    ##STRESS
    f='/scratch/abf376/FAFMIP/MPI-ESM/FAF-stress/so_yr_MPI-ESM-LR-remap_FAF-stress_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_stress_mpi=xr.open_dataset(f)['sea_water_salinity']
    salt_stress_mpi=salt_stress_mpi.where(salt_stress_mpi !=9.969209968386869e+36) #get rid of weird values

    regrid_stress_mpi = regridder_mpitocesm2(salt_stress_mpi[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadGEM2/FAF-stress/so_yr_HadGEM2-ES_FAF-stress_r1i1p1_185912-192911.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_stress_hadgem=xr.open_dataset(f)['sea_water_salinity']
    salt_stress_hadgem=salt_stress_hadgem.where(salt_stress_hadgem !=9.969209968386869e+36) #get rid of weird values
    salt_stress_hadgem=salt_stress_hadgem

    regrid_stress_hadgem = regridder_hadgemtocesm2(salt_stress_hadgem[:,0,:,:])

    f='/scratch/abf376/FAFMIP/GFDL-ESM2M/FAF-stress/so_yr_GFDL-ESM2M_FAF-stress_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_stress_gfdl=xr.open_dataset(f)['sea_water_salinity']
    salt_stress_gfdl=salt_stress_gfdl.where(salt_stress_gfdl !=9.969209968386869e+36) #get rid of weird values
    salt_stress_gfdl=salt_stress_gfdl

    regrid_stress_gfdl = regridder_gfdltocesm2(salt_stress_gfdl[:,0,:,:])

    ##HEAT
    f='/scratch/abf376/FAFMIP/MPI-ESM/FAF-heat/so_yr_MPI-ESM-LR-remap_FAF-heat_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_heat_mpi=xr.open_dataset(f)['sea_water_salinity']
    salt_heat_mpi=salt_heat_mpi.where(salt_heat_mpi !=9.969209968386869e+36) #get rid of weird values

    regrid_heat_mpi = regridder_mpitocesm2(salt_heat_mpi[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadGEM2/FAF-heat/so_yr_HadGEM2-ES_FAF-heat_r1i1p1_185912-192911.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_heat_hadgem=xr.open_dataset(f)['sea_water_salinity']
    salt_heat_hadgem=salt_heat_hadgem.where(salt_heat_hadgem !=9.969209968386869e+36) #get rid of weird values
    salt_heat_hadgem=salt_heat_hadgem

    regrid_heat_hadgem = regridder_hadgemtocesm2(salt_heat_hadgem[:,0,:,:])

    f='/scratch/abf376/FAFMIP/GFDL-ESM2M/FAF-heat/so_yr_GFDL-ESM2M_FAF-heat_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    salt_heat_gfdl=xr.open_dataset(f)['sea_water_salinity']
    salt_heat_gfdl=salt_heat_gfdl.where(salt_heat_gfdl !=9.969209968386869e+36) #get rid of weird values
    salt_heat_gfdl=salt_heat_gfdl

    regrid_heat_gfdl = regridder_gfdltocesm2(salt_heat_gfdl[:,0,:,:])

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()

    ############################ WATER
    x=np.linspace(31,38,1000)
    #let's find the change in salinity in each of these regions over the 70 years
    s=(salt[0:36,:,:].mean('time')).where(salt.latitude<65)
    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_hadgem_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_water_hadgem[j,:,:]).where(regrid_water_hadgem.latitude<65)
        for i in range(0,n):
            salt_hadgem_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_gfdl_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_water_gfdl[j,:,:])
        for i in range(0,n):
            salt_gfdl_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_mpi_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_water_mpi[j,:,:])
        for i in range(0,n):
            salt_mpi_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    ########################### HEAT
    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_hadgem_heat=np.empty([69,n])
    for j in range(0,69):
        s_new=(regrid_heat_hadgem[j,:,:]).where(regrid_heat_hadgem.latitude<65)
        for i in range(0,n):
            salt_hadgem_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_gfdl_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_heat_gfdl[j,:,:])
        for i in range(0,n):
            salt_gfdl_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_mpi_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_heat_mpi[j,:,:])
        for i in range(0,n):
            salt_mpi_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    ####################### STRESS
    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_hadgem_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_stress_hadgem[j,:,:]).where(regrid_stress_hadgem.latitude<65)
        for i in range(0,n):
            salt_hadgem_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_gfdl_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_stress_gfdl[j,:,:])
        for i in range(0,n):
            salt_gfdl_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    salt_mpi_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_stress_mpi[j,:,:])
        for i in range(0,n):
            salt_mpi_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    return salt_gfdl_stress,salt_mpi_stress,salt_hadgem_stress,salt_gfdl_heat,salt_mpi_heat,salt_hadgem_heat,salt_gfdl_water,salt_mpi_water,salt_hadgem_water

def regridded_fafmip_coupled_temp(salt,area,a2,n):
    #----------------------------------------------------
    #This function has input of:
        # - salt: salt field over the period of time of interest from the dataset of interest
        # - area: The area grid of the salt field of interest
        # - a2: The vector categorizing salinity ranges for each cluster (output from the GMM functions which are run on the salt field of interest)
        # - n: the number of clusters
    # This function has output of:
        # Timeseries of coupled FAFMIP models (HadGEM2, MPI-ESM, GFDL-ESM2M) for each individual perturbation experiment (stress, heat, water) in each cluster based on the clustering of the salt field of interest
    #NOTE: This function is equivalent to regridded_fafmip_temp, but returns time series of surface temperature from coupled FAFMIP models rather than ocean only FAFMIP models
    #----------------------------------------------------

    ## COUPLED FAFMIP, temp:
    ##WATER
    f='/scratch/abf376/FAFMIP/MPI-ESM/FAF-water/thetao_yr_MPI-ESM-LR-remap_FAF-water_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_water_mpi=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_water_mpi=temp_water_mpi.where(temp_water_mpi !=9.969209968386869e+36) #get rid of weird values

    regridder_mpitocesm2 = xe.Regridder(temp_water_mpi[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_water_mpi = regridder_mpitocesm2(temp_water_mpi[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadGEM2/FAF-water/thetao_yr_HadGEM2-ES_FAF-water_r1i1p1_185912-192911.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_water_hadgem=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_water_hadgem=temp_water_hadgem.where(temp_water_hadgem !=9.969209968386869e+36) #get rid of weird values
    temp_water_hadgem=temp_water_hadgem

    regridder_hadgemtocesm2 = xe.Regridder(temp_water_hadgem[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_water_hadgem = regridder_hadgemtocesm2(temp_water_hadgem[:,0,:,:])

    f='/scratch/abf376/FAFMIP/GFDL-ESM2M/FAF-water/thetao_yr_GFDL-ESM2M_FAF-water_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_water_gfdl=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_water_gfdl=temp_water_gfdl.where(temp_water_gfdl !=9.969209968386869e+36) #get rid of weird values
    temp_water_gfdl=temp_water_gfdl

    regridder_gfdltocesm2 = xe.Regridder(temp_water_gfdl[:,0,:,:], salt, "bilinear",periodic=True)
    regrid_water_gfdl = regridder_gfdltocesm2(temp_water_gfdl[:,0,:,:])

    ##STRESS
    f='/scratch/abf376/FAFMIP/MPI-ESM/FAF-stress/thetao_yr_MPI-ESM-LR-remap_FAF-stress_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_stress_mpi=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_stress_mpi=temp_stress_mpi.where(temp_stress_mpi !=9.969209968386869e+36) #get rid of weird values

    regrid_stress_mpi = regridder_mpitocesm2(temp_stress_mpi[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadGEM2/FAF-stress/thetao_yr_HadGEM2-ES_FAF-stress_r1i1p1_185912-192911.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_stress_hadgem=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_stress_hadgem=temp_stress_hadgem.where(temp_stress_hadgem !=9.969209968386869e+36) #get rid of weird values
    temp_stress_hadgem=temp_stress_hadgem

    regrid_stress_hadgem = regridder_hadgemtocesm2(temp_stress_hadgem[:,0,:,:])

    f='/scratch/abf376/FAFMIP/GFDL-ESM2M/FAF-stress/thetao_yr_GFDL-ESM2M_FAF-stress_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_stress_gfdl=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_stress_gfdl=temp_stress_gfdl.where(temp_stress_gfdl !=9.969209968386869e+36) #get rid of weird values
    temp_stress_gfdl=temp_stress_gfdl

    regrid_stress_gfdl = regridder_gfdltocesm2(temp_stress_gfdl[:,0,:,:])

    ##HEAT
    f='/scratch/abf376/FAFMIP/MPI-ESM/FAF-heat/thetao_yr_MPI-ESM-LR-remap_FAF-heat_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_heat_mpi=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_heat_mpi=temp_heat_mpi.where(temp_heat_mpi !=9.969209968386869e+36) #get rid of weird values

    regrid_heat_mpi = regridder_mpitocesm2(temp_heat_mpi[:,0,:,:])

    f='/scratch/abf376/FAFMIP/HadGEM2/FAF-heat/thetao_yr_HadGEM2-ES_FAF-heat_r1i1p1_185912-192911.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_heat_hadgem=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_heat_hadgem=temp_heat_hadgem.where(temp_heat_hadgem !=9.969209968386869e+36) #get rid of weird values
    temp_heat_hadgem=temp_heat_hadgem

    regrid_heat_hadgem = regridder_hadgemtocesm2(temp_heat_hadgem[:,0,:,:])

    f='/scratch/abf376/FAFMIP/GFDL-ESM2M/FAF-heat/thetao_yr_GFDL-ESM2M_FAF-heat_r1i1p1.nc'
    file2read = netCDF4.Dataset(f,'r') #use this line if you want to see the descriptions
    #print(file2read.variables)
    temp_heat_gfdl=xr.open_dataset(f)['sea_water_potential_temperature']
    temp_heat_gfdl=temp_heat_gfdl.where(temp_heat_gfdl !=9.969209968386869e+36) #get rid of weird values
    temp_heat_gfdl=temp_heat_gfdl

    regrid_heat_gfdl = regridder_gfdltocesm2(temp_heat_gfdl[:,0,:,:])

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()

    ############################ WATER
    x=np.linspace(31,38,1000)
    #let's find the change in salinity in each of these regions over the 70 years
    s=(salt[0:36,:,:].mean('time')).where(salt.latitude<65)
    #we have a 50 year time series, we want to find the mean temp at each region defined by the first year at each of these years
    temp_hadgem_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_water_hadgem[j,:,:]).where(regrid_water_hadgem.latitude<65)
        for i in range(0,n):
            temp_hadgem_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_gfdl_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_water_gfdl[j,:,:])
        for i in range(0,n):
            temp_gfdl_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_mpi_water=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_water_mpi[j,:,:])
        for i in range(0,n):
            temp_mpi_water[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    ########################### HEAT
    #we have a 50 year time series, we want to find the mean temp at each region defined by the first year at each of these years
    temp_hadgem_heat=np.empty([69,n])
    for j in range(0,69):
        s_new=(regrid_heat_hadgem[j,:,:]).where(regrid_heat_hadgem.latitude<65)
        for i in range(0,n):
            temp_hadgem_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_gfdl_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_heat_gfdl[j,:,:])
        for i in range(0,n):
            temp_gfdl_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_mpi_heat=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_heat_mpi[j,:,:])
        for i in range(0,n):
            temp_mpi_heat[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    ####################### STRESS
    #we have a 50 year time series, we want to find the mean temp at each region defined by the first year at each of these years
    temp_hadgem_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_stress_hadgem[j,:,:]).where(regrid_stress_hadgem.latitude<65)
        for i in range(0,n):
            temp_hadgem_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_gfdl_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_stress_gfdl[j,:,:])
        for i in range(0,n):
            temp_gfdl_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)


    temp_mpi_stress=np.empty([70,n])
    for j in range(0,70):
        s_new=(regrid_stress_mpi[j,:,:])
        for i in range(0,n):
            temp_mpi_stress[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)

    return temp_gfdl_stress,temp_mpi_stress,temp_hadgem_stress,temp_gfdl_heat,temp_mpi_heat,temp_hadgem_heat,temp_gfdl_water,temp_mpi_water,temp_hadgem_water
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
import percentiles_function
import xesmf as xe
from area_grid import *

def clusters(gm,salt,title,n):
    #this gives distinct clusters for salinity
    x=np.linspace(31,38,10000)
    a,a1=np.unique(gm.predict(x.reshape(-1,1)),return_index=True)
    np.sort(a1)
    a,a1=np.unique(gm.predict(x.reshape(-1,1)),return_index=True)
    np.sort(a1)
    P=copy.deepcopy(gm.predict(x.reshape(-1,1)))
    #for i in range(0,n):
    #    if i<5:
    #        P[np.sort(a1)[i]:np.sort(a1)[i+1]]=i
    #    else:
    #        P[np.sort(a1)[i]:len(P)-1]=i
    a2=np.append(np.sort(a1),len(x)-1)

    ## This chooses it so that it is distinct features rather than within 1 standard deviation
    s=salt
    #make an object that can hold where each gaussian (mean +/- one standard deviation) is spatially located
    y_disjoint = xr.DataArray(
        data=np.empty((180, 360,n)), dims=["latitude","longitude","gaussian"],coords=dict(latitude=s.latitude,longitude=s.longitude,gaussian=np.linspace(1,n,n)))
    for i in range(0,n):
        y_disjoint[:,:,i]=xr.where((s<(x[a2[i+1]]))&(s>(x[a2[i]])),i+1,0)

    if n==6:
        colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59']
    elif n==4:
        colorsList = ['#f6eff7','#d0d1e6','#1c9099','#016c59']
    elif n==9:
        colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#225ea8','#253494','#081d58']
    elif n==15:
        colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#225ea8','#253494','#081d58','#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59']
    elif n==5:
        colorsList = ['#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59']
    elif n==8:
        colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#253494','#081d58']
    elif n==7:
        colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#253494']
    o=y_disjoint.sum('gaussian').where(y_disjoint.sum('gaussian')>0)
    CustomCmap = matplotlib.colors.ListedColormap(colorsList)
    fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()},figsize=(10,8),dpi=70) #this is from cartopy https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html
    p=(o.where(y_disjoint[:,:,1].latitude<65)).plot(cbar_kwargs={'shrink':0.75,'orientation':'horizontal','extend':'both','pad':0.05},ax=ax,cmap=CustomCmap,alpha=1) #you have to set a colormap here because plotting xarray infers from the 
    ax.coastlines(color='grey',lw=0.5)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    ax.set_title(title)
    fig.tight_layout()

    return y_disjoint,a2

def regridded_fafmip(salt,area,a2,n):
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



def regridded_fafmip_temp(salt,area,a2,n):

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
    temp_heat_mit=temp_heat_mit.where(temp_heat_mit >0) #get rid of weird values
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

def regridded_fafmip_coupled(salt,area,a2,n):
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

def linear_response(salt,temp,salt_mean,n):
    from percentiles_function import GMM_timedep
    area=area_grid(latitudes=np.array(salt[0,:,:].latitude),longitudes=salt[0,:,:].longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt[0,:,:].latitude,salt[0,:,:].longitude])

    mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:12,:,:].mean('time')).where(salt_mean.latitude<65),n,'First year, 2015, SSP8.5') 
    y,a2=clusters(gm,salt,'Location of each Gaussian, categorized by first time point of SSP8.5 run, CESM2',n)

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    s=(salt_mean[0:12,:,:].mean('time')).where(salt_mean.latitude<65)
    x=np.linspace(31,38,1000)
    #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
    salt_cesm2=np.empty([50,n])
    temp_cesm2=np.empty([50,n])
    for j in range(0,50):
        s_new=(salt[j*12:(j+1)*12,:,:].mean('time')).where(salt.latitude<65)
        t_new=(temp[j*12:(j+1)*12,:,:].mean('time')).where(temp.latitude<65)
        for i in range(0,n):
            salt_cesm2[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)
            temp_cesm2[j,i]=area_weighted_disjoint(area,i,s,t_new,x,a2)

    change_cesm2=salt_cesm2[:,:]-np.mean(salt_cesm2[0:1,:],0)
    change_temp_cesm2=temp_cesm2[:,:]-np.mean(temp_cesm2[0:1,:],0)


    temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water=regridded_fafmip_temp(salt,area,a2,n)
    salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water=regridded_fafmip(salt,area,a2,n)


    start_yr=14
    end_yr=19

    df=np.zeros([50,end_yr-start_yr,3,3])
    df2=np.zeros([50,end_yr-start_yr,3,3])
    df4=np.zeros([50,end_yr-start_yr,3,3])
    dist=np.zeros([50,end_yr-start_yr,3])
    for p in range(0,3):

        access1mit0had2=p

        start_yr=14
        end_yr=19
        for start in range(start_yr,end_yr):

            if access1mit0had2==0:
                change_water=salt_mit_water[:,:]-salt_mit_water[0,:]
                change_heat=salt_mit_heat[:,:]-salt_mit_heat[0,:]
                change_stress=salt_mit_stress[:,:]-salt_mit_stress[0,:]
                change_water_temp=temp_mit_water[:,:]-temp_mit_water[0,:]
                change_heat_temp=temp_mit_heat[:,:]-temp_mit_heat[0,:]
                change_stress_temp=temp_mit_stress[:,:]-temp_mit_stress[0,:]
            elif access1mit0had2==1:
                change_water=salt_access_water[:,:]-salt_access_water[0,:]
                change_heat=salt_access_heat[:,:]-salt_access_heat[0,:]
                change_stress=salt_access_stress[:,:]-salt_access_stress[0,:]
                change_water_temp=temp_access_water[:,:]-temp_access_water[0,:]
                change_heat_temp=temp_access_heat[:,:]-temp_access_heat[0,:]
                change_stress_temp=temp_access_stress[:,:]-temp_access_stress[0,:]
            elif access1mit0had2==2:
                change_water=salt_had_water[:,:]-salt_had_water[0,:]
                change_heat=salt_had_heat[:,:]-salt_had_heat[0,:]
                change_stress=salt_had_stress[:,:]-salt_had_stress[0,:]
                change_water_temp=temp_had_water[:,:]-temp_had_water[0,:]
                change_heat_temp=temp_had_heat[:,:]-temp_had_heat[0,:]
                change_stress_temp=temp_had_stress[:,:]-temp_had_stress[0,:]

            #a=np.max(change_temp_cesm2)/np.max(change_cesm2)
            a=np.linalg.norm(change_temp_cesm2)/np.linalg.norm(change_cesm2)

            change2_water=np.concatenate((change_water[start:71],change_water_temp[start:71]/a),axis=1)
            change2_heat=np.concatenate((change_heat[start:71],change_heat_temp[start:71]/a),axis=1)
            if p==1:
                change2_stress=np.concatenate((change_stress[start:67],change_stress_temp[start:71]/a),axis=1)
            elif p==2:
                change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
            elif p==0:
                change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)

            change_cesm2_stack=np.concatenate((change_cesm2,change_temp_cesm2/a),axis=1)
            
            for i in range(0,49):
                sum=np.zeros(2*n)
                for j in range(0,i):
                    B=np.concatenate((np.matrix(change2_water[i-j,:]).T,np.matrix(change2_heat[i-j,:]).T,np.matrix(change2_stress[i-j,:]).T),axis=1)
                    sum=((B)*np.matrix(df[j,start-start_yr,:,p]).T).T+sum
                RHS=change_cesm2_stack[i+1,:].T-sum
                A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T,np.matrix(change2_stress[0,:]).T),axis=1)
                #A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T),axis=1)
                #df[i,start-start_yr,:,p]=(np.linalg.inv(np.matmul(A.T,A))*np.matmul(A.T,np.matrix(RHS).T)).reshape(3)
                #dist[i,start-start_yr,p] = np.linalg.norm(np.matmul(A,df[i,start-start_yr,:,p])-RHS)
                x, residuals, rank, s=np.linalg.lstsq(A,np.matrix(RHS).T,rcond = -1)
                df[i,start-start_yr,:,p]=x.reshape(3)


            for k in range(0,3):
                df2[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum())-(df[:,start-start_yr,k,p].cumsum())[0] #subtract off so starts from 0
                df4[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum()) #don't subtract off so starts at 0
            df3=np.mean(df2,axis=1) #mean over the start years where we subtracted off
            df5=np.mean(df4,axis=1) #mean over the start years where we didn't make start from 0
    df3_mean=np.mean(df3,axis=2)
    df5_mean=np.mean(df5,axis=2)

    fig,ax=plt.subplots(figsize=(10,6))
    plt.plot(df3_mean[:,0])
    plt.plot(df3[:,0,0],':')
    plt.plot(df3[:,0,1],':')
    plt.plot(df3[:,0,2],':')
    plt.title('Freshwater flux forcing (compared to FAFMIP, mean ACCESS-OM2 and MOM5, mean of start years 14-19). Intercept not subtracted')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.legend(['Mean MITgcm and ACCESS-OM2','MITgcm','ACCESS-OM2','HadOM3'])
    p=scipy.stats.linregress(np.linspace(0,50,50), y=df3_mean[:,0], alternative='two-sided')
    plt.plot(np.linspace(0,50,50),p.slope*np.linspace(0,50,50))
    plt.plot(np.linspace(0,50,50),(p.slope+p.stderr)*np.linspace(0,50,50))
    plt.plot(np.linspace(0,50,50),(p.slope-p.stderr)*np.linspace(0,50,50))

    change_water_1=p.slope*50
    change_water_upper=(p.slope+p.stderr)*50
    change_water_lower=(p.slope-p.stderr)*50

    fig,ax=plt.subplots(figsize=(12,6))
    plt.plot(df3_mean[:,1])
    plt.plot(df3[:,1,0],':')
    plt.plot(df3[:,1,1],':')
    plt.plot(df3[:,1,2],':')
    plt.title('Heat flux forcing which affects salinity field (compared to FAFMIP, mean ACCESS-OM2 and MOM5, mean of start years 14-19). Intercept subtracted')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.legend(['Mean MITgcm and ACCESS-OM2','MITgcm','ACCESS-OM2','HadOM3'])
    p=scipy.stats.linregress(np.linspace(0,50,50), y=df3_mean[:,1], alternative='two-sided')
    plt.plot(np.linspace(0,50,50),p.slope*np.linspace(0,50,50))
    plt.plot(np.linspace(0,50,50),(p.slope+p.stderr)*np.linspace(0,50,50))
    plt.plot(np.linspace(0,50,50),(p.slope-p.stderr)*np.linspace(0,50,50))

    change_heat_1=p.slope*50
    change_heat_upper=(p.slope+p.stderr)*50
    change_heat_lower=(p.slope-p.stderr)*50

    fig,ax=plt.subplots(figsize=(10,6))
    plt.plot(df3_mean[:,2])
    plt.plot(df3[:,2,0],':')
    plt.plot(df3[:,2,1],':')
    plt.plot(df3[:,2,2],':')
    plt.title('Wind stress change which affects salinity field (compared to FAFMIP, mean ACCESS-OM2 and MOM5, mean of start years 14-19). Intercept subtracted')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.legend(['Mean MOM5 and ACCESS-OM2','MOM5','ACCESS-OM2','HadOM3'])

    return change_water_1, change_water_upper, change_water_lower,change_heat_1,change_heat_upper,change_heat_lower
    
def linear_response_list(salt,temp,salt_mean,n,a2):
#input a list of salt and temp
    from percentiles_function import GMM_timedep
    area=area_grid(latitudes=np.array(salt_mean.latitude),longitudes=salt_mean.longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_mean.latitude,salt_mean.longitude])

    #mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65),n,'First year, 2015, SSP8.5') 
    #y,a2=clusters(gm,salt_mean[0:36,:,:].mean('time'),'Location of each Gaussian, categorized by first time point of SSP8.5 run, CESM2',n)

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    change_water_1=np.empty([len(salt)])
    change_water_upper=np.empty([len(salt)])
    change_water_lower=np.empty([len(salt)])

    change_heat_1=np.empty([len(salt)])
    change_heat_upper=np.empty([len(salt)])
    change_heat_lower=np.empty([len(salt)])
    
    temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water=regridded_fafmip_temp(salt[0],area,a2,n)
    salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water=regridded_fafmip(salt[0],area,a2,n)

    for q in range(0,len(salt)):
        s=(salt_mean[0:12,:,:].mean('time')).where(salt_mean.latitude<65)
        x=np.linspace(31,38,10000)
        nn=int(np.size(salt_mean[:,0,0])/12)
        #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
        salt_cesm2=np.empty([nn,n,len(salt)])
        temp_cesm2=np.empty([nn,n,len(salt)])
        for j in range(0,nn):
            s_new=((salt[q])[j*12:(j+1)*12,:,:].mean('time')).where(salt_mean.latitude<65)
            t_new=((temp[q])[j*12:(j+1)*12,:,:].mean('time')).where(salt_mean.latitude<65)
            for i in range(0,n):
                salt_cesm2[j,i,q]=area_weighted_disjoint(area,i,s,s_new,x,a2)
                temp_cesm2[j,i,q]=area_weighted_disjoint(area,i,s,t_new,x,a2)

        change_cesm=salt_cesm2[:,:,q]-np.mean(salt_cesm2[0:1,:,q],0)
        change_temp_cesm=temp_cesm2[:,:,q]-np.mean(temp_cesm2[0:1,:,q],0)


        start_yr=15
        end_yr=19
        
        #df=np.zeros([50,end_yr-start_yr,3,2])
        #df2=np.zeros([50,end_yr-start_yr,3,2])
        #df4=np.zeros([50,end_yr-start_yr,3,2])
        #dist=np.zeros([50,end_yr-start_yr,3])
        df=np.zeros([nn,end_yr-start_yr,3,3])
        df2=np.zeros([nn,end_yr-start_yr,3,3])
        df4=np.zeros([nn,end_yr-start_yr,3,3])
        dist=np.zeros([nn,end_yr-start_yr,3])
        for p in range(0,3):
        
            access1mit0had2=p
        
            start_yr=15
            end_yr=19
            for start in range(start_yr,end_yr):
        
                if access1mit0had2==0:
                    change_water=salt_mit_water[:,:]-salt_mit_water[0,:]
                    change_heat=salt_mit_heat[:,:]-salt_mit_heat[0,:]
                    change_stress=salt_mit_stress[:,:]-salt_mit_stress[0,:]
                    change_water_temp=temp_mit_water[:,:]-temp_mit_water[0,:]
                    change_heat_temp=temp_mit_heat[:,:]-temp_mit_heat[0,:]
                    change_stress_temp=temp_mit_stress[:,:]-temp_mit_stress[0,:]
                elif access1mit0had2==1:
                    change_water=salt_access_water[:,:]-salt_access_water[0,:]
                    change_heat=salt_access_heat[:,:]-salt_access_heat[0,:]
                    change_stress=salt_access_stress[:,:]-salt_access_stress[0,:]
                    change_water_temp=temp_access_water[:,:]-temp_access_water[0,:]
                    change_heat_temp=temp_access_heat[:,:]-temp_access_heat[0,:]
                    change_stress_temp=temp_access_stress[:,:]-temp_access_stress[0,:]
                elif access1mit0had2==2:
                    change_water=salt_had_water[:,:]-salt_had_water[0,:]
                    change_heat=salt_had_heat[:,:]-salt_had_heat[0,:]
                    change_stress=salt_had_stress[:,:]-salt_had_stress[0,:]
                    change_water_temp=temp_had_water[:,:]-temp_had_water[0,:]
                    change_heat_temp=temp_had_heat[:,:]-temp_had_heat[0,:]
                    change_stress_temp=temp_had_stress[:,:]-temp_had_stress[0,:]
                    
                if p==2:
                   da = xr.DataArray(change_water)
                   change_water=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat)
                   change_heat=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress)
                   change_stress=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_water_temp)
                   change_water_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat_temp)
                   change_heat_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress_temp)
                   change_stress_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
        
                #a=np.max(change_temp_cesm2)/np.max(change_cesm2)
                a=np.linalg.norm(change_temp_cesm)/np.linalg.norm(change_cesm)
        
                change2_water=np.concatenate((change_water[start:71],change_water_temp[start:71]/a),axis=1)
                change2_heat=np.concatenate((change_heat[start:71],change_heat_temp[start:71]/a),axis=1)
                if p==1:
                    change2_stress=np.concatenate((change_stress[start:67],change_stress_temp[start:71]/a),axis=1)
                elif p==2:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
                elif p==0:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
        
                change_cesm2_stack=np.concatenate((change_cesm,change_temp_cesm/a),axis=1)
                
                for i in range(0,nn-1):
                    sum=np.zeros(2*n)
                    for j in range(0,i):
                        B=np.concatenate((np.matrix(change2_water[i-j,:]).T,np.matrix(change2_heat[i-j,:]).T,np.matrix(change2_stress[i-j,:]).T),axis=1)
                        sum=((B)*np.matrix(df[j,start-start_yr,:,p]).T).T+sum
                    RHS=change_cesm2_stack[i+1,:].T-sum
                    A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T,np.matrix(change2_stress[0,:]).T),axis=1)
                    #A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T),axis=1)
                    #df[i,start-start_yr,:,p]=(np.linalg.inv(np.matmul(A.T,A))*np.matmul(A.T,np.matrix(RHS).T)).reshape(3)
                    #dist[i,start-start_yr,p] = np.linalg.norm(np.matmul(A,df[i,start-start_yr,:,p])-RHS)
                    x, residuals, rank, s=np.linalg.lstsq(A,np.matrix(RHS).T,rcond = -1)
                    df[i,start-start_yr,:,p]=x.reshape(3)
        
        
                for k in range(0,3):
                    df2[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum())-(df[:,start-start_yr,k,p].cumsum())[0] #subtract off so starts from 0
                    df4[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum()) #don't subtract off so starts at 0
                df3=np.mean(df2,axis=1) #mean over the start years where we subtracted off
                df5=np.mean(df4,axis=1) #mean over the start years where we didn't make start from 0
        df3_mean=np.mean(df3,axis=2)
        df5_mean=np.mean(df5,axis=2)

        p=scipy.stats.linregress(np.linspace(0,nn-1,nn), y=df3_mean[:,0], alternative='two-sided')

        change_water_1[q]=p.slope*nn+p.intercept
        change_water_upper[q]=(p.slope+p.stderr)*nn + p.intercept
        change_water_lower[q]=(p.slope-p.stderr)*nn + p.intercept

        p=scipy.stats.linregress(np.linspace(0,nn-1,nn), y=df3_mean[:,1], alternative='two-sided')

        change_heat_1[q]=p.slope*nn+p.intercept ##### CHANGE HERE
        change_heat_upper[q]=(p.slope+p.stderr)*nn + p.intercept
        change_heat_lower[q]=(p.slope-p.stderr)*nn + p.intercept
    return change_water_1, change_water_upper, change_water_lower,change_heat_1,change_heat_upper,change_heat_lower
    
    
def linear_response_list2(salt,temp,salt_mean,n,a2): #the previous function uses the linear trend and this just takes last 5 years minus first 5 years.
#input a list of salt and temp
    from percentiles_function import GMM_timedep
    area=area_grid(latitudes=np.array(salt_mean.latitude),longitudes=salt_mean.longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_mean.latitude,salt_mean.longitude])

    #mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65),n,'First year, 2015, SSP8.5') 
    #y,a2=clusters(gm,salt_mean[0:36,:,:].mean('time'),'Location of each Gaussian, categorized by first time point of SSP8.5 run, CESM2',n)

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    change_water_1=np.empty([len(salt)])
    change_water_upper=np.empty([len(salt)])
    change_water_lower=np.empty([len(salt)])

    change_heat_1=np.empty([len(salt)])
    change_heat_upper=np.empty([len(salt)])
    change_heat_lower=np.empty([len(salt)])
    
    temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water=regridded_fafmip_temp(salt[0],area,a2,n)
    salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water=regridded_fafmip(salt[0],area,a2,n)

    for q in range(0,len(salt)):
        s=(salt_mean[0:12,:,:].mean('time')).where(salt_mean.latitude<65)
        x=np.linspace(31,38,10000)
        nn=int(np.size(salt_mean[:,0,0])/12)
        #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
        salt_cesm2=np.empty([nn,n,len(salt)])
        temp_cesm2=np.empty([nn,n,len(salt)])
        for j in range(0,nn):
            s_new=((salt[q])[j*12:(j+1)*12,:,:].mean('time')).where(salt_mean.latitude<65)
            t_new=((temp[q])[j*12:(j+1)*12,:,:].mean('time')).where(salt_mean.latitude<65)
            for i in range(0,n):
                salt_cesm2[j,i,q]=area_weighted_disjoint(area,i,s,s_new,x,a2)
                temp_cesm2[j,i,q]=area_weighted_disjoint(area,i,s,t_new,x,a2)

        change_cesm=salt_cesm2[:,:,q]-np.mean(salt_cesm2[0:1,:,q],0)
        change_temp_cesm=temp_cesm2[:,:,q]-np.mean(temp_cesm2[0:1,:,q],0)
        nn=int(np.size(salt_mean[:,0,0])/12)


        start_yr=15
        end_yr=19
        
        #df=np.zeros([50,end_yr-start_yr,3,2])
        #df2=np.zeros([50,end_yr-start_yr,3,2])
        #df4=np.zeros([50,end_yr-start_yr,3,2])
        #dist=np.zeros([50,end_yr-start_yr,3])
        df=np.zeros([nn,end_yr-start_yr,3,3])
        df2=np.zeros([nn,end_yr-start_yr,3,3])
        df4=np.zeros([nn,end_yr-start_yr,3,3])
        dist=np.zeros([nn,end_yr-start_yr,3])
        for p in range(0,3):
        
            access1mit0had2=p
        
            start_yr=15
            end_yr=19
            for start in range(start_yr,end_yr):
        
                if access1mit0had2==0:
                    change_water=salt_mit_water[:,:]-salt_mit_water[0,:]
                    change_heat=salt_mit_heat[:,:]-salt_mit_heat[0,:]
                    change_stress=salt_mit_stress[:,:]-salt_mit_stress[0,:]
                    change_water_temp=temp_mit_water[:,:]-temp_mit_water[0,:]
                    change_heat_temp=temp_mit_heat[:,:]-temp_mit_heat[0,:]
                    change_stress_temp=temp_mit_stress[:,:]-temp_mit_stress[0,:]
                elif access1mit0had2==1:
                    change_water=salt_access_water[:,:]-salt_access_water[0,:]
                    change_heat=salt_access_heat[:,:]-salt_access_heat[0,:]
                    change_stress=salt_access_stress[:,:]-salt_access_stress[0,:]
                    change_water_temp=temp_access_water[:,:]-temp_access_water[0,:]
                    change_heat_temp=temp_access_heat[:,:]-temp_access_heat[0,:]
                    change_stress_temp=temp_access_stress[:,:]-temp_access_stress[0,:]
                elif access1mit0had2==2:
                    change_water=salt_had_water[:,:]-salt_had_water[0,:]
                    change_heat=salt_had_heat[:,:]-salt_had_heat[0,:]
                    change_stress=salt_had_stress[:,:]-salt_had_stress[0,:]
                    change_water_temp=temp_had_water[:,:]-temp_had_water[0,:]
                    change_heat_temp=temp_had_heat[:,:]-temp_had_heat[0,:]
                    change_stress_temp=temp_had_stress[:,:]-temp_had_stress[0,:]
                    
                if p==2:
                   da = xr.DataArray(change_water)
                   change_water=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat)
                   change_heat=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress)
                   change_stress=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_water_temp)
                   change_water_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat_temp)
                   change_heat_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress_temp)
                   change_stress_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
        
                #a=np.max(change_temp_cesm2)/np.max(change_cesm2)
                a=np.linalg.norm(change_temp_cesm)/np.linalg.norm(change_cesm)
        
                change2_water=np.concatenate((change_water[start:71],change_water_temp[start:71]/a),axis=1)
                change2_heat=np.concatenate((change_heat[start:71],change_heat_temp[start:71]/a),axis=1)
                if p==1:
                    change2_stress=np.concatenate((change_stress[start:67],change_stress_temp[start:71]/a),axis=1)
                elif p==2:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
                elif p==0:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
        
                change_cesm2_stack=np.concatenate((change_cesm,change_temp_cesm/a),axis=1)
                
                for i in range(0,nn-1):
                    sum=np.zeros(2*n)
                    for j in range(0,i):
                        B=np.concatenate((np.matrix(change2_water[i-j,:]).T,np.matrix(change2_heat[i-j,:]).T,np.matrix(change2_stress[i-j,:]).T),axis=1)
                        sum=((B)*np.matrix(df[j,start-start_yr,:,p]).T).T+sum
                    RHS=change_cesm2_stack[i+1,:].T-sum
                    A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T,np.matrix(change2_stress[0,:]).T),axis=1)
                    #A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T),axis=1)
                    #df[i,start-start_yr,:,p]=(np.linalg.inv(np.matmul(A.T,A))*np.matmul(A.T,np.matrix(RHS).T)).reshape(3)
                    #dist[i,start-start_yr,p] = np.linalg.norm(np.matmul(A,df[i,start-start_yr,:,p])-RHS)
                    x, residuals, rank, s=np.linalg.lstsq(A,np.matrix(RHS).T,rcond = -1)
                    df[i,start-start_yr,:,p]=x.reshape(3)
        
        
                for k in range(0,3):
                    df2[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum())-(df[:,start-start_yr,k,p].cumsum())[0] #subtract off so starts from 0
                    df4[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum()) #don't subtract off so starts at 0
                df3=np.mean(df2,axis=1) #mean over the start years where we subtracted off
                df5=np.mean(df4,axis=1) #mean over the start years where we didn't make start from 0
        df3_mean=np.mean(df3,axis=2)
        df5_mean=np.mean(df5,axis=2)

        change_water_1[q]=df3_mean[40:45,0].mean()-df3_mean[0:5,0].mean()
        change_heat_1[q]=df3_mean[40:45,1].mean()-df3_mean[0:5,1].mean()
    return change_water_1, change_heat_1
    
    
def linear_response_list_bootstrap(salt,temp,salt_mean,n,a2,obs=0): #this function is similar to linear_response_list but takes input that's already collocated in clusters (rather than a lat lon grid). takes last 5 years minus first 5 years.
#input a list of salt and temp
    from percentiles_function import GMM_timedep
    area=area_grid(latitudes=np.array(salt_mean.latitude),longitudes=salt_mean.longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_mean.latitude,salt_mean.longitude])

    #mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65),n,'First year, 2015, SSP8.5') 
    #y,a2=clusters(gm,salt_mean[0:36,:,:].mean('time'),'Location of each Gaussian, categorized by first time point of SSP8.5 run, CESM2',n)

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    change_water_1=np.empty([len(salt)])
    change_water_upper=np.empty([len(salt)])
    change_water_lower=np.empty([len(salt)])

    change_heat_1=np.empty([len(salt)])
    change_heat_upper=np.empty([len(salt)])
    change_heat_lower=np.empty([len(salt)])
    if obs==0:
        nn=int(np.size(salt_mean[:,0,0])/12)
    elif obs==1:
        nn=int(np.size(salt_mean[:,0,0]))
    
    temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water=regridded_fafmip_temp(salt_mean,area,a2,n)
    salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water=regridded_fafmip(salt_mean,area,a2,n)
    
    for q in range(0,len(salt)):
        change_cesm=salt[q]
        change_temp_cesm=temp[q]

        start_yr=15
        end_yr=19
        
        #df=np.zeros([50,end_yr-start_yr,3,2])
        #df2=np.zeros([50,end_yr-start_yr,3,2])
        #df4=np.zeros([50,end_yr-start_yr,3,2])
        #dist=np.zeros([50,end_yr-start_yr,3])
        df=np.zeros([nn,end_yr-start_yr,3,3])
        df2=np.zeros([nn,end_yr-start_yr,3,3])
        df4=np.zeros([nn,end_yr-start_yr,3,3])
        dist=np.zeros([nn,end_yr-start_yr,3])
        for p in range(0,3):
        
            access1mit0had2=p
        
            start_yr=15
            end_yr=19
            for start in range(start_yr,end_yr):
        
                if access1mit0had2==0:
                    change_water=salt_mit_water[:,:]-salt_mit_water[0,:]
                    change_heat=salt_mit_heat[:,:]-salt_mit_heat[0,:]
                    change_stress=salt_mit_stress[:,:]-salt_mit_stress[0,:]
                    change_water_temp=temp_mit_water[:,:]-temp_mit_water[0,:]
                    change_heat_temp=temp_mit_heat[:,:]-temp_mit_heat[0,:]
                    change_stress_temp=temp_mit_stress[:,:]-temp_mit_stress[0,:]
                elif access1mit0had2==1:
                    change_water=salt_access_water[:,:]-salt_access_water[0,:]
                    change_heat=salt_access_heat[:,:]-salt_access_heat[0,:]
                    change_stress=salt_access_stress[:,:]-salt_access_stress[0,:]
                    change_water_temp=temp_access_water[:,:]-temp_access_water[0,:]
                    change_heat_temp=temp_access_heat[:,:]-temp_access_heat[0,:]
                    change_stress_temp=temp_access_stress[:,:]-temp_access_stress[0,:]
                elif access1mit0had2==2:
                    change_water=salt_had_water[:,:]-salt_had_water[0,:]
                    change_heat=salt_had_heat[:,:]-salt_had_heat[0,:]
                    change_stress=salt_had_stress[:,:]-salt_had_stress[0,:]
                    change_water_temp=temp_had_water[:,:]-temp_had_water[0,:]
                    change_heat_temp=temp_had_heat[:,:]-temp_had_heat[0,:]
                    change_stress_temp=temp_had_stress[:,:]-temp_had_stress[0,:]
                    
                if p==2:
                   da = xr.DataArray(change_water)
                   change_water=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat)
                   change_heat=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress)
                   change_stress=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_water_temp)
                   change_water_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat_temp)
                   change_heat_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress_temp)
                   change_stress_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
        
                #a=np.max(change_temp_cesm2)/np.max(change_cesm2)
                a=np.linalg.norm(change_temp_cesm)/np.linalg.norm(change_cesm)
        
                change2_water=np.concatenate((change_water[start:71],change_water_temp[start:71]/a),axis=1)
                change2_heat=np.concatenate((change_heat[start:71],change_heat_temp[start:71]/a),axis=1)
                if p==1:
                    change2_stress=np.concatenate((change_stress[start:67],change_stress_temp[start:71]/a),axis=1)
                elif p==2:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
                elif p==0:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
        
                change_cesm2_stack=np.concatenate((change_cesm,change_temp_cesm/a),axis=1)
                
                for i in range(0,nn-1):
                    sum=np.zeros(2*n)
                    for j in range(0,i):
                        B=np.concatenate((np.matrix(change2_water[i-j,:]).T,np.matrix(change2_heat[i-j,:]).T,np.matrix(change2_stress[i-j,:]).T),axis=1)
                        sum=((B)*np.matrix(df[j,start-start_yr,:,p]).T).T+sum
                    RHS=change_cesm2_stack[i+1,:].T-sum
                    A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T,np.matrix(change2_stress[0,:]).T),axis=1)
                    #A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T),axis=1)
                    #df[i,start-start_yr,:,p]=(np.linalg.inv(np.matmul(A.T,A))*np.matmul(A.T,np.matrix(RHS).T)).reshape(3)
                    #dist[i,start-start_yr,p] = np.linalg.norm(np.matmul(A,df[i,start-start_yr,:,p])-RHS)
                    x, residuals, rank, s=np.linalg.lstsq(A,np.matrix(RHS).T,rcond = -1)
                    df[i,start-start_yr,:,p]=x.reshape(3)
        
        
                for k in range(0,3):
                    df2[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum())-(df[:,start-start_yr,k,p].cumsum())[0] #subtract off so starts from 0
                    df4[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum()) #don't subtract off so starts at 0
                df3=np.mean(df2,axis=1) #mean over the start years where we subtracted off
                df5=np.mean(df4,axis=1) #mean over the start years where we didn't make start from 0
        df3_mean=np.mean(df3,axis=2)
        df5_mean=np.mean(df5,axis=2)

        change_water_1[q]=df3_mean[40:45,0].mean()-df3_mean[0:5,0].mean()
        change_heat_1[q]=df3_mean[40:45,1].mean()-df3_mean[0:5,1].mean()
    return change_water_1, change_heat_1
    
    
def linear_response_list_bootstrap_acceptnan(salt,temp,salt_mean,n,a2,obs=0): #this function is similar to linear_response_list but takes input that's already collocated in clusters (rather than a lat lon grid). takes last 5 years minus first 5 years.
#input a list of salt and temp
#this version will accept input that has nan in some of the clusters. Can be used when we've removed a cluster due to insignificant linear trend
    from percentiles_function import GMM_timedep
    area=area_grid(latitudes=np.array(salt_mean.latitude),longitudes=salt_mean.longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_mean.latitude,salt_mean.longitude])

    #mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65),n,'First year, 2015, SSP8.5') 
    #y,a2=clusters(gm,salt_mean[0:36,:,:].mean('time'),'Location of each Gaussian, categorized by first time point of SSP8.5 run, CESM2',n)
    
    ## identify where the nans are
    signif_salt=np.where(np.abs(salt[0][0,:])>0,np.ones(6),np.nan) #this returns an array of size 6 which identifies which regions are significant for salt
    signif_temp=np.where(np.abs(salt[0][0,:])>0,np.ones(6),np.nan) #this returns an array of size 6 which identifies which regions are significant for temp
    
    ## Above, we drop the region for temperature and salinity (both observables) if the trend is insignificant for salt. The trend is always significant for temperature.
    

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    change_water_1=np.empty([len(salt)])
    change_water_upper=np.empty([len(salt)])
    change_water_lower=np.empty([len(salt)])

    change_heat_1=np.empty([len(salt)])
    change_heat_upper=np.empty([len(salt)])
    change_heat_lower=np.empty([len(salt)])
    if obs==0:
        nn=int(np.size(salt_mean[:,0,0])/12)
    elif obs==1:
        nn=int(np.size(salt_mean[:,0,0]))
    
    temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water=regridded_fafmip_temp(salt_mean,area,a2,n)
    salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water=regridded_fafmip(salt_mean,area,a2,n)
    
    for q in range(0,len(salt)):
        change_cesm=salt[q][~np.isnan(salt[q])].reshape(-1, np.size(signif_salt[~np.isnan(signif_salt)])) #drop the nan parts of salt[q]
        change_temp_cesm=temp[q][~np.isnan(temp[q])].reshape(-1, np.size(signif_temp[~np.isnan(signif_temp)])) #drop the nan parts of temp[q]
        #change_cesm=salt[q].dropna()
        #change_temp_cesm=salt[q].dropna()

        start_yr=15
        end_yr=19
        
        #df=np.zeros([50,end_yr-start_yr,3,2])
        #df2=np.zeros([50,end_yr-start_yr,3,2])
        #df4=np.zeros([50,end_yr-start_yr,3,2])
        #dist=np.zeros([50,end_yr-start_yr,3])
        df=np.zeros([nn,end_yr-start_yr,3,3])
        df2=np.zeros([nn,end_yr-start_yr,3,3])
        df4=np.zeros([nn,end_yr-start_yr,3,3])
        dist=np.zeros([nn,end_yr-start_yr,3])
        for p in range(0,3):
        
            access1mit0had2=p
        
            start_yr=15
            end_yr=19
            for start in range(start_yr,end_yr):
        
                if access1mit0had2==0:
                    change_water=salt_mit_water[:,:]-salt_mit_water[0,:]
                    change_heat=salt_mit_heat[:,:]-salt_mit_heat[0,:]
                    change_stress=salt_mit_stress[:,:]-salt_mit_stress[0,:]
                    change_water_temp=temp_mit_water[:,:]-temp_mit_water[0,:]
                    change_heat_temp=temp_mit_heat[:,:]-temp_mit_heat[0,:]
                    change_stress_temp=temp_mit_stress[:,:]-temp_mit_stress[0,:]
                elif access1mit0had2==1:
                    change_water=salt_access_water[:,:]-salt_access_water[0,:]
                    change_heat=salt_access_heat[:,:]-salt_access_heat[0,:]
                    change_stress=salt_access_stress[:,:]-salt_access_stress[0,:]
                    change_water_temp=temp_access_water[:,:]-temp_access_water[0,:]
                    change_heat_temp=temp_access_heat[:,:]-temp_access_heat[0,:]
                    change_stress_temp=temp_access_stress[:,:]-temp_access_stress[0,:]
                elif access1mit0had2==2:
                    change_water=salt_had_water[:,:]-salt_had_water[0,:]
                    change_heat=salt_had_heat[:,:]-salt_had_heat[0,:]
                    change_stress=salt_had_stress[:,:]-salt_had_stress[0,:]
                    change_water_temp=temp_had_water[:,:]-temp_had_water[0,:]
                    change_heat_temp=temp_had_heat[:,:]-temp_had_heat[0,:]
                    change_stress_temp=temp_had_stress[:,:]-temp_had_stress[0,:]
                    
                if p==2:
                   da = xr.DataArray(change_water)
                   change_water=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat)
                   change_heat=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress)
                   change_stress=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_water_temp)
                   change_water_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_heat_temp)
                   change_heat_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                   da = xr.DataArray(change_stress_temp)
                   change_stress_temp=da.rolling(dim_0=4,center=True).mean().dropna('dim_0')
                
                ## multiply the fafmip stuff by whether to include it due to significant from input and then drop the nan
                change_water=signif_salt*change_water
                da=xr.DataArray(change_water)
                change_water=da.dropna('dim_1')
                change_heat=signif_salt*change_heat
                da=xr.DataArray(change_heat)
                change_heat=da.dropna('dim_1')
                change_stress=signif_salt*change_stress
                da=xr.DataArray(change_stress)
                change_stress=da.dropna('dim_1')

                change_water_temp=signif_temp*change_water_temp
                da=xr.DataArray(change_water_temp)
                change_water_temp=da.dropna('dim_1')
                change_heat_temp=signif_temp*change_heat_temp
                da=xr.DataArray(change_heat_temp)
                change_heat_temp=da.dropna('dim_1')
                change_stress_temp=signif_temp*change_stress_temp
                da=xr.DataArray(change_stress_temp)
                change_stress_temp=da.dropna('dim_1')
                
                a=np.linalg.norm(change_temp_cesm)/np.linalg.norm(change_cesm)
        
                change2_water=np.concatenate((change_water[start:71],change_water_temp[start:71]/a),axis=1)
                change2_heat=np.concatenate((change_heat[start:71],change_heat_temp[start:71]/a),axis=1)

                if p==1:
                    change2_stress=np.concatenate((change_stress[start:67],change_stress_temp[start:71]/a),axis=1)
                elif p==2:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
                elif p==0:
                    change2_stress=np.concatenate((change_stress[start:71],change_stress_temp[start:71]/a),axis=1)
        
                change_cesm2_stack=np.concatenate((change_cesm,change_temp_cesm/a),axis=1)
                
                for i in range(0,nn-1):
                    sum=np.zeros(np.size(signif_salt[~np.isnan(signif_salt)])+np.size(signif_temp[~np.isnan(signif_temp)]))
                    for j in range(0,i):
                        B=np.concatenate((np.matrix(change2_water[i-j,:]).T,np.matrix(change2_heat[i-j,:]).T,np.matrix(change2_stress[i-j,:]).T),axis=1)
                        sum=((B)*np.matrix(df[j,start-start_yr,:,p]).T).T+sum
                    RHS=change_cesm2_stack[i+1,:].T-sum
                    A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T,np.matrix(change2_stress[0,:]).T),axis=1)
                 # #  #A=np.concatenate((np.matrix(change2_water[0,:]).T,np.matrix(change2_heat[0,:]).T),axis=1)
                 # #  #df[i,start-start_yr,:,p]=(np.linalg.inv(np.matmul(A.T,A))*np.matmul(A.T,np.matrix(RHS).T)).reshape(3)
                 # #  #dist[i,start-start_yr,p] = np.linalg.norm(np.matmul(A,df[i,start-start_yr,:,p])-RHS)
                    x, residuals, rank, s=np.linalg.lstsq(A,np.matrix(RHS).T,rcond = -1)
                    df[i,start-start_yr,:,p]=x.reshape(3)
        
        
                for k in range(0,3):
                    df2[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum())-(df[:,start-start_yr,k,p].cumsum())[0] #subtract off so starts from 0
                    df4[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum()) #don't subtract off so starts at 0
                df3=np.mean(df2,axis=1) #mean over the start years where we subtracted off
                df5=np.mean(df4,axis=1) #mean over the start years where we didn't make start from 0
        df3_mean=np.mean(df3,axis=2)
        df5_mean=np.mean(df5,axis=2)

        change_water_1[q]=df3_mean[40:45,0].mean()-df3_mean[0:5,0].mean()
        change_heat_1[q]=df3_mean[40:45,1].mean()-df3_mean[0:5,1].mean()
    return change_water_1, change_heat_1, signif_salt
    #return change_water, signif_salt
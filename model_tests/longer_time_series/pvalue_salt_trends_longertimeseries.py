#This file calculates the p values of the linear fits to each region for each CESM ensemble member
#This could have been combined with run_bootstrap_cesm_ensemble.py, but the ideas were run separately and we didn't want need to rerun both to recombine
#However, a lot of the code has similarities to runbootstrap_cesm_ensemble.py. First we categorize each region, then find the linear trend (and associated p value) and block bootstrap and then invert the confidence interval of that (to find p value)
# Linear response theory is not applied to the artificial ensembles here though.

# Load in needed packages


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
from scipy import stats
from xgcm import Grid
import statsmodels.api as sm
import matplotlib.ticker as ticker
from matplotlib.axes._secondary_axes import SecondaryAxis
import xesmf as xe
import recombinator



#Empty arrays to store block bootstrapped p values
pvalue_store=np.empty([34,6]) #p value from original linear fit
pvalue_bootstrap_store=np.empty([34,6]) #p value from block bootstrapped confidence interval

#Open the salt and temperature fields and area matrix
#These files were generated in processing_salt_temp.py. That file needs to be run first and then the paths below must match where the files are saved
np.random.seed(0)
import pickle
with open("/scratch/abf376/regridded_salt_2006to2080_rcp8.5", "rb") as fp:   # Unpickling
    regridded_salt_2005on= pickle.load(fp)

with open("/scratch/abf376/regridded_temp_2006to2080_rcp8.5", "rb") as fp:   # Unpickling
    regridded_temp_2005on= pickle.load(fp)

salt_2005on_list=[] #make list of salt post 2005 and give it correct coordinates
for i in range(0,34):
    s=regridded_salt_2005on[i].rename({'y': 'latitude','x': 'longitude'})
    s=s.assign_coords(latitude=s.lat[:,0],longitude=s.lon[0,:])
    salt_2005on_list.append(s)
    
salt_list=[] #combine salt lists from the two time periods and then cut to 1975 to 2025
for i in range(0,34):
    s=xr.concat([salt_2005on_list[i]],dim="time")
    salt_list.append(s[5*12:12*55,:,:]) #cut to 1975 to 2025
    
#same as above but for temperature
temp_2005on_list=[]
for i in range(0,34):
    s=regridded_temp_2005on[i].rename({'y': 'latitude','x': 'longitude'})
    s=s.assign_coords(latitude=s.lat[:,0],longitude=s.lon[0,:])
    temp_2005on_list.append(s)
    
temp_list=[]
for i in range(0,34):
    s=xr.concat([temp_2005on_list[i]],dim="time")
    temp_list.append(s[5*12:12*55,:,:]) #cut to 1975 to 2025

#define an area matrix
import sys
sys.path.append('/scratch/abf376/freshwater_flux_linresp/tools')
from area_grid import *

area=area_grid(latitudes=np.array(salt_list[0].latitude),longitudes=salt_list[0].longitude)
area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_list[0].latitude,salt_list[0].longitude])
x=np.linspace(31,38,10000)

#Define the way of area weighting
def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
    return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()

np.random.seed(0)
for real in range(0,34):

    n=6 
    from clustering_tools import GMM_timedep
    mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_list[real][0*12:5*12,:,:].mean('time')).where(salt_list[real][0,:,:].latitude<65),n,'1975 to 1980 Historical') #take mean over first year

    from clustering_tools import clusters
    y,a2=clusters(gm,salt_list[real][0*12:5*12,:,:].mean('time'),'Location of each Gaussian, categorized by years 1975-1980 (mean), CESM',n)

    s=(salt_list[real][0*12:5*12,:,:].mean('time')).where(salt_list[real][0*12:5*12,:,:].latitude<65)


    salt_cesm_member=np.empty([50,n])
    temp_cesm_member=np.empty([50,n])
    for j in range(0,50):
        s_new=(salt_list[real][j*12:(j+1)*12,:,:].mean('time')).where(salt_list[real].latitude<65) #update aug 24 is changing this from salt_list[0] to salt_list[real]
        t_new=(temp_list[real][j*12:(j+1)*12,:,:].mean('time')).where(temp_list[real].latitude<65) #update aug 24 is changing this from temp_list[0] to temp_list[real]
        for i in range(0,n):
            salt_cesm_member[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)
            temp_cesm_member[j,i]=area_weighted_disjoint(area,i,s,t_new,x,a2)
    change_cesm_member=salt_cesm_member[:,:]-np.mean(salt_cesm_member[0:2,:],0)
    change_cesm_temp_member=temp_cesm_member[:,:]-np.mean(temp_cesm_member[0:2,:],0)

    change_cesm_member=change_cesm_member[0:45,:]
    change_cesm_temp_member=change_cesm_temp_member[0:45,:]

    #Find the trend for each region and then block bootstrap
    trend=np.empty([45,6])
    pvalue=np.empty([6])
    for i in range(0,6):
        p=scipy.stats.linregress(np.linspace(0,44,45), y=change_cesm_member[:,i], alternative='two-sided')
        trend[:,i]=p.intercept+p.slope*np.linspace(0,44,45)
        pvalue_store[real,i]=p.pvalue

    #Find the trend for each region and then block bootstrap
    trend_temp=np.empty([45,6])
    for i in range(0,6):
        p=scipy.stats.linregress(np.linspace(0,44,45), y=change_cesm_temp_member[:,i], alternative='two-sided')
        trend_temp[:,i]=p.intercept+p.slope*np.linspace(0,44,45)

    from recombinator.block_bootstrap import circular_block_bootstrap

    # number of replications for bootstraps (number of resampled time-series to generate)
    B = 3000

    y_star_cb \
        = circular_block_bootstrap(np.concatenate([change_cesm_member-trend,change_cesm_temp_member-trend_temp],axis=1), 
                                   block_length=2, 
                                   replications=B, replace=True)
    bootstrap_salt=y_star_cb[:,:,0:6]
    bootstrap_temp=y_star_cb[:,:,6:12]
    #so we now have 50 "members" from performing block bootstrapping with 1 member that we had. Let's put them now in a list
    salt_list_bootstrap=[]
    for i in range(0,B):
        salt_list_bootstrap.append(trend+bootstrap_salt[i,:,:])

    temp_list_bootstrap=[]
    for i in range(0,B):
        temp_list_bootstrap.append(trend_temp+bootstrap_temp[i,:,:])

    bootstrapped_slope=np.empty([B,6])
    for i in range(0,B):
        for j in range(0,6):
            p=scipy.stats.linregress(np.linspace(0,44,45), y=salt_list_bootstrap[i][:,j], alternative='two-sided')
            bootstrapped_slope[i,j]=p.slope

    for j in range(0,6):
        z=bootstrapped_slope[:,j].mean()/bootstrapped_slope[:,j].std()
        pvalue_bootstrap_store[real,j] = scipy.stats.norm.sf(abs(z)) #we use this form because we expect the null hypothesis to be less than values

    print(real)

        
#pickle the results
with open("pvalue_linear_trends_2011to2055", "wb") as fp:   #Pickling the pvalues for each region for each ensemble member from the linear trend
    pickle.dump(pvalue_store, fp)
with open("pvalue_from_bootstrap_2011to2055", "wb") as fp:   #Pickling the pvalues for each region for each ensemble member from block bootstrapping
    pickle.dump(pvalue_bootstrap_store, fp)
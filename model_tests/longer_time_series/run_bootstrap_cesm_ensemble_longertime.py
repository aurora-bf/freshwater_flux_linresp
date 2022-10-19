# This file iterates through each CESM ensemble member and for each applies the method by first fitting a GMM, then creating an artificial ensemble using block bootstrapping, then applying linear response theory.
#Note: This file is essentially the same as run_bootstrap_notrenduncertainty.py in not cleaned up version of file

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



# Define function which applies the method to each ensemble member 


def ensemble_member_blockbootstrap(l):
    np.random.seed(0)
    real=l #realization in the list

    n=6 #number of clusters
    #cluster the salinity field
    from clustering_tools import GMM_timedep
    mean_con,sigma_con,weights_con,gm=GMM_timedep((salt_list[real][0*12:3*12,:,:].mean('time')).where(salt_list[real][0,:,:].latitude<65),n,'2011 to 2014') #take mean over first year

    from clustering_tools import clusters
    y,a2=clusters(gm,salt_list[real][0*12:3*12,:,:].mean('time'),'Location of each Gaussian, categorized by years 2011 to 2014 (mean), CESM',n)

    s=(salt_list[real][0*12:3*12,:,:].mean('time')).where(salt_list[real][0*12:3*12,:,:].latitude<65)

    #find the mean salinity and temperature at each time in each of the clusters found above    
    salt_cesm_member=np.empty([50,n])
    temp_cesm_member=np.empty([50,n])
    for j in range(0,50):
        s_new=(salt_list[real][j*12:(j+1)*12,:,:].mean('time')).where(salt_list[real].latitude<65) #update aug 24 is changing this from salt_list[0] to salt_list[real]
        t_new=(temp_list[real][j*12:(j+1)*12,:,:].mean('time')).where(temp_list[real].latitude<65) #update aug 24 is changing this from temp_list[0] to temp_list[real]
        for i in range(0,n):
            salt_cesm_member[j,i]=area_weighted_disjoint(area,i,s,s_new,x,a2)
            temp_cesm_member[j,i]=area_weighted_disjoint(area,i,s,t_new,x,a2)
            
    ## THIS PART IS ONLY NEEDED IF YOU'RE DOING AREA WEIGHTED        
    def area_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    ones_grid=salt_list[real][0,:,:].where(salt_list[real][0,:,:]==1, other=1)
    area_cluster=np.empty(6)
    for i in range(0,n):
        area_cluster[i]=area_disjoint(area,i,s,ones_grid,x,a2)
    ##

    #find the change in salinity and temperature in each cluster
    change_cesm_member=salt_cesm_member[:,:]-np.mean(salt_cesm_member[0:2,:],0)
    change_cesm_temp_member=temp_cesm_member[:,:]-np.mean(temp_cesm_member[0:2,:],0)

    #cut to the 1975 to 2019 period
    change_cesm_member=change_cesm_member[0:45,:]
    change_cesm_temp_member=change_cesm_temp_member[0:45,:]
    
    #Find the trend for each region (salinity)
    trend=np.empty([45,6])
    pvalue=np.empty([6])
    for i in range(0,6):
        p=scipy.stats.linregress(np.linspace(0,44,45), y=change_cesm_member[:,i], alternative='two-sided')
        trend[:,i]=p.intercept+p.slope*np.linspace(0,44,45)
        pvalue[i]=p.pvalue
        
    #Find the trend for each region (temperature)
    trend_temp=np.empty([45,6])
    for i in range(0,6):
        p=scipy.stats.linregress(np.linspace(0,44,45), y=change_cesm_temp_member[:,i], alternative='two-sided')
        trend_temp[:,i]=p.intercept+p.slope*np.linspace(0,44,45)
        
    from recombinator.block_bootstrap import circular_block_bootstrap

    # number of replications for bootstraps (number of resampled time-series to generate)
    B = 3000

    #perform block bootstrapping
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

    #Apply linear response theory to each member of the artificial ensemble which was created for individual ensemble member l
    n=6
    from linear_response_tools import linear_response_list_bootstrap
    #change_water1, change_heat1=linear_response_list_bootstrap(salt_list_bootstrap,temp_list_bootstrap,salt_list[real][0:45*12,:,:],n,a2) SWITCH TO THIS LINE IF YOU DON'T WANT AREA WEIGHTED
    change_water1, change_heat1=linear_response_list_bootstrap(salt_list_bootstrap,temp_list_bootstrap,salt_list[real][0:45*12,:,:],n,a2,weighted=1,area_cluster=area_cluster)
    
    std=change_water1.std()
    mean=change_water1.mean()
    return std,mean,change_water1


#Define the way of area weighting
def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
    return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()


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
    
salt_list=[] 
for i in range(0,34):
    s=salt_2005on_list[i]
    salt_list.append(s[5*12:12*55,:,:]) #cut to 2011 to 2055
    
#same as above but for temperature
temp_2005on_list=[]
for i in range(0,34):
    s=regridded_temp_2005on[i].rename({'y': 'latitude','x': 'longitude'})
    s=s.assign_coords(latitude=s.lat[:,0],longitude=s.lon[0,:])
    temp_2005on_list.append(s)
    
temp_list=[]
for i in range(0,34):
    s=temp_2005on_list[i]
    temp_list.append(s[5*12:12*55,:,:]) #cut to 2011 to 2055

#define an area matrix
import sys
sys.path.append('/scratch/abf376/freshwater_flux_linresp/tools')
from area_grid import *

area=area_grid(latitudes=np.array(salt_list[0].latitude),longitudes=salt_list[0].longitude)
area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_list[0].latitude,salt_list[0].longitude])

x=np.linspace(31,38,10000)




#empty arrays we will store in
mean_member2=np.empty(34) #store the means of applying linear response theory to each ensemble member
std_member2=np.empty(34) #store the standard deviations of applying linear response theory to each ensemble member
change_water=np.empty([34,3000])



#iterate the above function over the 34 members
for i in range(0,34): 
    std_member2[i],mean_member2[i],change_water[i,:]=ensemble_member_blockbootstrap(i)
    print(i)

#pickle the results
with open("bootstrap_std_3000_2011to2050_weighted", "wb") as fp:   #Pickling
    pickle.dump(std_member2, fp)
with open("bootstrap_mean_3000_2011to2050_weighted", "wb") as fp:   #Pickling
    pickle.dump(mean_member2, fp)
with open("bootstrap_change_water_3000_2011to2050_weighted","wb") as fp:
    pickle.dump(change_water, fp)
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cmip_basins.gfdl as gfdl
import cmip_basins.cmip6 as cmip6
from cmip_basins.basins import generate_basin_codes
from scipy import stats
from xgcm import Grid
import cartopy.crs as ccrs
import cartopy as cart
from cmip_basins.basins import generate_basin_codes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter #see https://scitools.org.uk/cartopy/docs/v0.15/examples/tick_labels.html
import matplotlib.cm as cm
from area_grid import *
import sys
import scipy
sys.path.insert(1,'/Users/aurora/Documents/GitHub/pattern_amplification')
#from dip_test import dip
import sklearn.mixture
from sklearn.mixture import GaussianMixture

def rename_models(salt_heat,salt_control,salt_water,salt_all,salt_stress,model):
    ## FUNCTION DESCRIPTION_____________________________________
    #This function the dimensions of surface variables to a standard form so that they can be run through the PA function.
    # It renames all latitude coordinates to 'latitude', all longitude coordinates to 'longitude' and all time coordinates to 'time'.
    # Apply this function to data before running it through PA if the names do not already follow this convention.
    # The model parameter allows you to specify FAFMIP data that came from MITgcm or MOM5/ACCESS-OM2 so that they can be renamed appropriately 
    #_____________________________________________________
    if model==0: #mitgcm
        salt_heat=salt_heat.rename({'Y': 'latitude','X': 'longitude','T':'time'})
        salt_all=salt_all.rename({'Y': 'latitude','X': 'longitude','T':'time'})
        salt_control=salt_control.rename({'Y': 'latitude','X': 'longitude'})
        salt_water=salt_water.rename({'Y': 'latitude','X': 'longitude','T':'time'})
        salt_stress=salt_stress.rename({'Y':'latitude','X':'longitude','T':'time'})
    elif model==1: #mom5 or accessom2
        salt_heat=salt_heat.rename({'yt_ocean': 'latitude','xt_ocean': 'longitude'})
        salt_all=salt_all.rename({'yt_ocean': 'latitude','xt_ocean': 'longitude'})
        salt_control=salt_control.rename({'yt_ocean': 'latitude','xt_ocean': 'longitude'})
        salt_water=salt_water.rename({'yt_ocean': 'latitude','xt_ocean': 'longitude'})
        salt_stress=salt_stress.rename({'yt_ocean':'latitude','xt_ocean':'longitude'})
    return salt_heat, salt_control, salt_water, salt_all, salt_stress


def pdf(salt_heat,salt_control,salt_water,salt_all,model):
#This function shows the probability distribution function of surface salinity in the last decade of the control and forced experiments.
#It also computes some metrics for example it outputs the earth movers distance between the control distribution and the heat, water, and faf-all distributions. 
#It also performs the dip test for unimodality on the pdfs 

    Lx_div12=20
    x=np.linspace(31,38,12*Lx_div12+1) # salinities on the x axis that we using for (discretized) CDF

    lon = np.array(salt_heat.longitude)
    lat = np.array(salt_heat.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                             persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    #take mean of last 10years of data
    salt_con_surface=salt_control[60:71,0,:,:].mean('time')
    salt_con_surface=salt_con_surface.where(salt_con_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_heat_surface=salt_heat[60:71,0,:,:].mean('time')
    salt_heat_surface=salt_heat_surface.where(salt_heat_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_water_surface=salt_water[60:71,0,:,:].mean('time')
    salt_water_surface=salt_water_surface.where(salt_water_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_all_surface=salt_all[60:71,0,:,:].mean('time')
    salt_all_surface=salt_all_surface.where(salt_all_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))


    # getting data of the histogram for FAFWATER
    count, bins_count = np.histogram(salt_water_surface, bins=x,weights=area)
    pdf_water = count / sum(count)
    cdf_water = np.cumsum(pdf_water)

    # getting data of the histogram for FAFHEAT
    count, bins_count = np.histogram(salt_heat_surface, bins=x,weights=area)
    pdf_heat = count / sum(count)
    cdf_heat = np.cumsum(pdf_heat)

    # getting data of the histogram for CONTROL
    count, bins_count = np.histogram(salt_con_surface, bins=x,weights=area)
    pdf_con = count / sum(count)
    cdf_con = np.cumsum(pdf_con)

    # getting data of the histogram for faf-all
    count, bins_count = np.histogram(salt_all_surface, bins=x,weights=area)
    pdf_all = count / sum(count)
    cdf_all = np.cumsum(pdf_all)

    #get the means
    ma = np.ma.MaskedArray(salt_con_surface.where(salt_con_surface>np.min(x)).where(salt_con_surface<np.max(x)), mask=np.isnan(salt_con_surface.where(salt_con_surface>np.min(x)).where(salt_con_surface<np.max(x))))
    mean_con=np.ma.average(ma, weights=area)

    ma = np.ma.MaskedArray(salt_heat_surface.where(salt_heat_surface>np.min(x)).where(salt_heat_surface<np.max(x)), mask=np.isnan(salt_heat_surface.where(salt_heat_surface>np.min(x)).where(salt_heat_surface<np.max(x))))
    mean_heat=np.ma.average(ma, weights=area)

    ma = np.ma.MaskedArray(salt_water_surface.where(salt_water_surface>np.min(x)).where(salt_water_surface<np.max(x)), mask=np.isnan(salt_water_surface.where(salt_water_surface>np.min(x)).where(salt_water_surface<np.max(x))))
    mean_water=np.ma.average(ma, weights=area)

    ma = np.ma.MaskedArray(salt_all_surface.where(salt_all_surface>np.min(x)).where(salt_all_surface<np.max(x)), mask=np.isnan(salt_all_surface.where(salt_all_surface>np.min(x)).where(salt_all_surface<np.max(x))))
    mean_all=np.ma.average(ma, weights=area)

    # plotting PDF and CDF
    fig, ax = plt.subplots()
    plt.plot(bins_count[1:], cdf_con, color="red", label="control")
    plt.plot(bins_count[1:], cdf_heat, label="faf-heat")
    plt.plot(bins_count[1:], cdf_water, label="faf-water")
    plt.plot(bins_count[1:], cdf_all, label="faf-all")
    ax.set_xlabel('Salinity')
    ax.set_ylabel('Area')
    str2= 'Cumulative distribution function - Surface salinity, '
    str2+=model
    ax.set_title(str2)
    ax.invert_xaxis()
    ax.legend()

    # plotting PDF and CDF
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(bins_count[1:], pdf_con, color="red", label="control")
    plt.plot(bins_count[1:], pdf_heat, label="faf-heat")
    plt.plot(bins_count[1:], pdf_water, label="faf-water")
    plt.plot(bins_count[1:], pdf_all, color="tab:green",label='faf-all')
    plt.axvline(mean_con,color="tab:red",linestyle='--',label="control mean")
    plt.axvline(mean_heat,color="tab:blue",linestyle='--',label="faf-heat mean")
    plt.axvline(mean_water,color="tab:orange",linestyle='--',label="faf-water mean")
    plt.axvline(mean_all,color="tab:green",linestyle='--',label="faf-all mean")
    ax.set_xlabel('Salinity')
    ax.set_ylabel('Area')
    str= 'Probability distribution function - Surface salinity, '
    str+=model
    ax.set_title(str)
    ax.invert_xaxis()
    ax.legend()

    #we can also compute some metrics to do with the pdfs

    #compute the Earth movers distance - a measure of the difference between pdfs
    distance_water_control=scipy.stats.wasserstein_distance(bins_count[1:],bins_count[1:],pdf_con,pdf_water)
    distance_heat_control=scipy.stats.wasserstein_distance(bins_count[1:],bins_count[1:],pdf_con,pdf_heat)
    distance_all_control=scipy.stats.wasserstein_distance(bins_count[1:],bins_count[1:],pdf_con,pdf_all)

    dip_heat=dip(pdf_heat,p=0.99,num_bins=240)
    dip_con=dip(pdf_con,p=0.99,num_bins=240)
    dip_water=dip(pdf_water,p=0.99,num_bins=240)
    dip_all=dip(pdf_all,p=0.99,num_bins=240)


    return distance_heat_control,distance_water_control, distance_all_control,dip_heat, dip_con, dip_water,dip_all

def pdf_input2d(salt_heat,salt_control,salt_water,salt_all,model):
#This function shows the probability distribution function of surface salinity in the last decade of the control and forced experiments.
#It also computes some metrics for example it outputs the earth movers distance between the control distribution and the heat, water, and faf-all distributions. 
#It also performs the dip test for unimodality on the pdfs 

    Lx_div12=20
    x=np.linspace(31,38,12*Lx_div12+1) # salinities on the x axis that we using for (discretized) CDF

    lon = np.array(salt_heat.longitude)
    lat = np.array(salt_heat.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                             persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    #take mean of last 10years of data
    salt_con_surface=salt_control[60:71,:,:].mean('time')
    salt_con_surface=salt_con_surface.where(salt_con_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_heat_surface=salt_heat[60:71,:,:].mean('time')
    salt_heat_surface=salt_heat_surface.where(salt_heat_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_water_surface=salt_water[60:71,:,:].mean('time')
    salt_water_surface=salt_water_surface.where(salt_water_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_all_surface=salt_all[60:71,:,:].mean('time')
    salt_all_surface=salt_all_surface.where(salt_all_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))


    # getting data of the histogram for FAFWATER
    count, bins_count = np.histogram(salt_water_surface, bins=x,weights=area)
    pdf_water = count / sum(count)
    cdf_water = np.cumsum(pdf_water)

    # getting data of the histogram for FAFHEAT
    count, bins_count = np.histogram(salt_heat_surface, bins=x,weights=area)
    pdf_heat = count / sum(count)
    cdf_heat = np.cumsum(pdf_heat)

    # getting data of the histogram for CONTROL
    count, bins_count = np.histogram(salt_con_surface, bins=x,weights=area)
    pdf_con = count / sum(count)
    cdf_con = np.cumsum(pdf_con)

    # getting data of the histogram for faf-all
    count, bins_count = np.histogram(salt_all_surface, bins=x,weights=area)
    pdf_all = count / sum(count)
    cdf_all = np.cumsum(pdf_all)

    #get the means
    ma = np.ma.MaskedArray(salt_con_surface.where(salt_con_surface>np.min(x)).where(salt_con_surface<np.max(x)), mask=np.isnan(salt_con_surface.where(salt_con_surface>np.min(x)).where(salt_con_surface<np.max(x))))
    mean_con=np.ma.average(ma, weights=area)

    ma = np.ma.MaskedArray(salt_heat_surface.where(salt_heat_surface>np.min(x)).where(salt_heat_surface<np.max(x)), mask=np.isnan(salt_heat_surface.where(salt_heat_surface>np.min(x)).where(salt_heat_surface<np.max(x))))
    mean_heat=np.ma.average(ma, weights=area)

    ma = np.ma.MaskedArray(salt_water_surface.where(salt_water_surface>np.min(x)).where(salt_water_surface<np.max(x)), mask=np.isnan(salt_water_surface.where(salt_water_surface>np.min(x)).where(salt_water_surface<np.max(x))))
    mean_water=np.ma.average(ma, weights=area)

    ma = np.ma.MaskedArray(salt_all_surface.where(salt_all_surface>np.min(x)).where(salt_all_surface<np.max(x)), mask=np.isnan(salt_all_surface.where(salt_all_surface>np.min(x)).where(salt_all_surface<np.max(x))))
    mean_all=np.ma.average(ma, weights=area)

    # plotting PDF and CDF
    fig, ax = plt.subplots()
    plt.plot(bins_count[1:], cdf_con, color="red", label="control")
    plt.plot(bins_count[1:], cdf_heat, label="faf-heat")
    plt.plot(bins_count[1:], cdf_water, label="faf-water")
    plt.plot(bins_count[1:], cdf_all, label="faf-all")
    ax.set_xlabel('Salinity')
    ax.set_ylabel('Area')
    str2= 'Cumulative distribution function - Surface salinity, '
    str2+=model
    ax.set_title(str2)
    ax.invert_xaxis()
    ax.legend()

    # plotting PDF and CDF
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(bins_count[1:], pdf_con, color="red", label="control")
    plt.plot(bins_count[1:], pdf_heat, label="faf-heat")
    plt.plot(bins_count[1:], pdf_water, label="faf-water")
    plt.plot(bins_count[1:], pdf_all, color="tab:green",label='faf-all')
    plt.axvline(mean_con,color="tab:red",linestyle='--',label="control mean")
    plt.axvline(mean_heat,color="tab:blue",linestyle='--',label="faf-heat mean")
    plt.axvline(mean_water,color="tab:orange",linestyle='--',label="faf-water mean")
    plt.axvline(mean_all,color="tab:green",linestyle='--',label="faf-all mean")
    ax.set_xlabel('Salinity')
    ax.set_ylabel('Area')
    str= 'Probability distribution function - Surface salinity, '
    str+=model
    ax.set_title(str)
    ax.invert_xaxis()
    ax.legend()

    #we can also compute some metrics to do with the pdfs

    #compute the Earth movers distance - a measure of the difference between pdfs
    distance_water_control=scipy.stats.wasserstein_distance(bins_count[1:],bins_count[1:],pdf_con,pdf_water)
    distance_heat_control=scipy.stats.wasserstein_distance(bins_count[1:],bins_count[1:],pdf_con,pdf_heat)
    distance_all_control=scipy.stats.wasserstein_distance(bins_count[1:],bins_count[1:],pdf_con,pdf_all)

    dip_heat=dip(pdf_heat,p=0.99,num_bins=240)
    dip_con=dip(pdf_con,p=0.99,num_bins=240)
    dip_water=dip(pdf_water,p=0.99,num_bins=240)
    dip_all=dip(pdf_all,p=0.99,num_bins=240)


    return distance_heat_control,distance_water_control, distance_all_control,dip_heat, dip_con, dip_water,dip_all


def percentiles(salt_heat,salt_control,salt_water,salt_all,model,coarse=0):
# This function can be used to generate plots of the change in the minimum salinity associated with each percentile between the last decade of a forced experiment and the last decade of control
#default is to not coarse grain it

    lon = np.array(salt_heat.longitude)
    lat = np.array(salt_heat.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                             persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    def weighted_percentile_of_score(a, weights, score, kind='weak'): #from https://stackoverflow.com/questions/48252282/weighted-version-of-scipy-percentileofscore
        npa = np.array(a)
        npw = np.array(weights)

        indx = npa <= score
        weak = 100 * sum(npw[indx]) / sum(weights)
        if kind == 'weak':
            return weak

    def find_percentiles_weighted(surface_data,weights):
        A=surface_data.stack(z=("latitude", "longitude"))
        B=weights.stack(z=("latitude", "longitude"))
        B=B[A.notnull()]
        A=A[A.notnull()] #.sortby(A[A.notnull()]) #so this is an array that has all not null values from max to min


        def percentiles_np(data,weights):
            return [weighted_percentile_of_score(data,weights,i,kind='weak') for i in data]

        p=(xr.apply_ufunc(percentiles_np,A,B)).unstack() #this allows us to apply a function to an xarray that expects a numpy array
        return p

        #take mean of last 10years of data
    salt_con_surface=salt_control[60:71,0,:,:].mean('time')
    salt_con_surface=salt_con_surface.where(salt_con_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_heat_surface=salt_heat[60:71,0,:,:].mean('time')
    salt_heat_surface=salt_heat_surface.where(salt_heat_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_water_surface=salt_water[60:71,0,:,:].mean('time')
    salt_water_surface=salt_water_surface.where(salt_water_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_all_surface=salt_all[60:71,0,:,:].mean('time')
    salt_all_surface=salt_all_surface.where(salt_all_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    percentiles_con=find_percentiles_weighted(salt_con_surface,area)
    percentiles_heat=find_percentiles_weighted(salt_heat_surface,area)
    percentiles_water=find_percentiles_weighted(salt_water_surface,area)
    percentiles_all=find_percentiles_weighted(salt_all_surface,area)

    if coarse==0: #don't find in smaller bins and then coarse grain
        #the bins are far too small, let's make them of 1% size for now. use the ceiling function
        index=np.ceil(percentiles_con)
        index_heat=np.ceil(percentiles_heat)
        #now we want to plot this so the y axis is the percentiles and then the x axis is the change in psu between experiments
    
        D=np.empty(100)

        for i in np.linspace(1,100,100):
            i=np.int(i)-1
            D[i]=salt_heat_surface.where(index_heat==i+1).min()-salt_con_surface.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(D,np.linspace(1,100,100))
        plt.axvline(D.mean(),color="tab:red")
        plt.xlabel('$\mathcal{S}^p(p,t_1)-\mathcal{S}^p(p,t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
        plt.ylabel('Percentile (p)')
        str='Change of last decade of faf-heat vs control ($t_0$ is last decade control and $t_1$ is last decade faf-heat), '
        str+=model
        plt.title(str)
    


        #the bins are far too small, let's make them of 1% size for now. use the ceiling function
        index_water=np.ceil(percentiles_water)

        D=np.empty(100)

        for i in np.linspace(1,100,100):
            i=np.int(i)-1
            D[i]=salt_water_surface.where(index_water==i+1).min()-salt_con_surface.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(D,np.linspace(1,100,100))
        plt.axvline(D.mean(),color="tab:red")
        plt.xlabel('$\mathcal{S}^p(p,t_1)-\mathcal{S}^p(p,t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
        plt.ylabel('Percentile (p)')
        str='Change of last decade of faf-water vs control ($t_0$ is last decade control and $t_1$ is last decade faf-water), '
        str+=model
        plt.title(str)

        #the bins are far too small, let's make them of 1% size for now. use the ceiling function
        index_all=np.ceil(percentiles_all)

        for i in np.linspace(1,100,100):
            i=np.int(i)-1
            D[i]=salt_all_surface.where(index_all==i+1).min()-salt_con_surface.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(D,np.linspace(1,100,100))
        plt.axvline(D.mean(),color="tab:red")
        plt.xlabel('$\mathcal{S}^p(p,t_1)-\mathcal{S}^p(p,t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
        plt.ylabel('Percentile (p)')
        str='Change of last decade of faf-all vs control ($t_0$ is last decade control and $t_1$ is last decade faf-all), '
        str+=model
        plt.title(str)

    if coarse==1:
        #the bins are far too small, let's make them of 1% size for now. use the ceiling function
        index_pt1=np.round(percentiles_con,1)
        index_heat_pt1=np.round(percentiles_heat,1)
        #now we want to plot this so the y axis is the percentiles and then the x axis is the change in psu between experiments

        def area_weighted(area_hadom3,index,salt_surface):
            return ((salt_surface*area_hadom3).where(index==i+1)).sum()/((area_hadom3.where(index==i+1)).sum())

    
        D_pt1=np.empty(1000)

        for i in np.linspace(1,1000,1000):
            i=np.int(i)-1
            D_pt1[i]=salt_heat_surface.where(index_heat_pt1==0.1*i+1).min()-salt_con_surface.where(index_pt1==0.1*i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        D_approx=np.empty(100)
        for i in range(0,100):
            D_approx[i]=np.nanmean(D_pt1[10*(i):10*(i)+11])
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(D_approx[0:100],np.linspace(1,100,100))
        plt.axvline(D_approx[0:100].mean(),color="tab:red")
        plt.xlabel('$\mathcal{S}_c^p(t_1)-\mathcal{S}_c^p(t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
        plt.ylabel('Percentile (p)')
        str='Change of last decade of faf-heat vs control ($t_0$ is last decade control and $t_1$ is last decade faf-heat), '
        str+=model
        plt.title(str)
    


        #the bins are far too small, let's make them of 1% size for now. use the ceiling function
        index_water_pt1=np.round(percentiles_water,1)

        for i in np.linspace(1,1000,1000):
            i=np.int(i)-1
            D_pt1[i]=salt_water_surface.where(index_water_pt1==0.1*i+1).min()-salt_con_surface.where(index_pt1==0.1*i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        D_approx=np.empty(100)
        for i in range(0,100):
            D_approx[i]=np.nanmean(D_pt1[10*(i):10*(i)+11])
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(D_approx[0:100],np.linspace(1,100,100))
        plt.axvline(D_approx[0:100].mean(),color="tab:red")
        plt.xlabel('$\mathcal{S}_c^p(t_1)-\mathcal{S}_c^p(t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
        plt.ylabel('Percentile (p)')
        str='Change of last decade of faf-water vs control ($t_0$ is last decade control and $t_1$ is last decade faf-water), '
        str+=model
        plt.title(str)

        #the bins are far too small, let's make them of 1% size for now. use the ceiling function
        index_all_pt1=np.round(percentiles_all,1)

        for i in np.linspace(1,1000,1000):
            i=np.int(i)-1
            D_pt1[i]=salt_all_surface.where(index_all_pt1==0.1*i+1).min()-salt_con_surface.where(index_pt1==0.1*i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        D_approx=np.empty(100)
        for i in range(0,100):
            D_approx[i]=np.nanmean(D_pt1[10*(i):10*(i)+11])
        fig, ax = plt.subplots(figsize=(10,6))
        plt.scatter(D_approx[0:100],np.linspace(1,100,100))
        plt.axvline(D_approx[0:100].mean(),color="tab:red")
        plt.xlabel('$\mathcal{S}_c^p(t_1)-\mathcal{S}_c^p(t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
        plt.ylabel('Percentile (p)')
        str='Change of last decade of faf-all vs control ($t_0$ is last decade control and $t_1$ is last decade faf-all), '
        str+=model
        plt.title(str)

# def percentiles_input2d(salt_heat,salt_control,salt_water,salt_all,salt_heat_reference,salt_control_reference,salt_water_reference,salt_all_reference,model,coarse=0):
# # This function can be used to generate plots of the change in the minimum salinity associated with each percentile between the last decade of a forced experiment and the last decade of control
# #default is to not coarse grain it

# #The first set of salts here (e.g. salt_heat, salt_control etc) are the salt field that we want to categorize. The second set (e.g. salt_heat_reference, salt_control_reference etc) are the salt field that we find the percentiles relative to. T
# #The idea is that we could, for example, cateogirze interior mean salinity change relative to surface percentiles

#     lon = np.array(salt_heat.longitude)
#     lat = np.array(salt_heat.latitude)

#     #first get the area grid
#     area=area_grid(latitudes=np.array(lat),longitudes=lon)
#     area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

#     #get the locations of basins
#     grid = xr.Dataset()
#     grid["lon"] = xr.DataArray(lon, dims=("lon"))
#     grid["lat"] = xr.DataArray(lat, dims=("lat"))
#     codes = generate_basin_codes(grid, lon="lon", lat="lat",
#                              persian=True, style='cmip6')
#     grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
#     codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

#     def weighted_percentile_of_score(a, weights, score, kind='weak'): #from https://stackoverflow.com/questions/48252282/weighted-version-of-scipy-percentileofscore
#         npa = np.array(a)
#         npw = np.array(weights)

#         indx = npa <= score
#         weak = 100 * sum(npw[indx]) / sum(weights)
#         if kind == 'weak':
#             return weak

#     def find_percentiles_weighted(surface_data,weights):
#         A=surface_data.stack(z=("latitude", "longitude"))
#         B=weights.stack(z=("latitude", "longitude"))
#         B=B[A.notnull()]
#         A=A[A.notnull()] #.sortby(A[A.notnull()]) #so this is an array that has all not null values from max to min


#         def percentiles_np(data,weights):
#             return [weighted_percentile_of_score(data,weights,i,kind='weak') for i in data]

#         p=(xr.apply_ufunc(percentiles_np,A,B)).unstack() #this allows us to apply a function to an xarray that expects a numpy array
#         return p

#         #take mean of last 10years of data
#     salt_con_reference_mean=salt_control_reference[60:71,:,:].mean('time')
#     salt_con_reference_mean=salt_con_reference_mean.where(salt_con_reference_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_con_mean=salt_control[60:71,:,:].mean('time')
#     salt_con_mean=salt_con_mean.where(salt_con_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_heat_reference_mean=salt_heat_reference[60:71,:,:].mean('time')
#     salt_heat_reference_mean=salt_heat_reference_mean.where(salt_heat_reference_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_heat_mean=salt_heat[60:71,:,:].mean('time')
#     salt_heat_mean=salt_heat_mean.where(salt_heat_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_water_reference_mean=salt_water_reference[60:71,:,:].mean('time')
#     salt_water_reference_mean=salt_water_reference_mean.where(salt_water_reference_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_water_mean=salt_water[60:71,:,:].mean('time')
#     salt_water_mean=salt_water_mean.where(salt_water_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_all_reference_mean=salt_all_reference[60:71,:,:].mean('time')
#     salt_all_reference_mean=salt_all_reference_mean.where(salt_all_reference_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     salt_all_mean=salt_all[60:71,:,:].mean('time')
#     salt_all_mean=salt_all_mean.where(salt_all_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

#     percentiles_con=find_percentiles_weighted(salt_con_reference_mean,area)
#     percentiles_heat=find_percentiles_weighted(salt_heat_reference_mean,area)
#     percentiles_water=find_percentiles_weighted(salt_water_reference_mean,area)
#     percentiles_all=find_percentiles_weighted(salt_all_reference_mean,area)

#     if coarse==0: #don't find in smaller bins and then coarse grain
#         #the bins are far too small, let's make them of 1% size for now. use the ceiling function
#         index=np.ceil(percentiles_con)
#         index_heat=np.ceil(percentiles_heat)
#         #now we want to plot this so the y axis is the percentiles and then the x axis is the change in psu between experiments
    
#         D=np.empty(100)

#         for i in np.linspace(1,100,100):
#             i=np.int(i)-1
#             D[i]=salt_heat_mean.where(index_heat==i+1).min()-salt_con_mean.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
#         fig, ax = plt.subplots(figsize=(10,6))
#         plt.scatter(D,np.linspace(1,100,100))
#         plt.axvline(D.mean(),color="tab:red")
#         plt.xlabel('$\mathcal{S}^p(p,t_1)-\mathcal{S}^p(p,t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
#         plt.ylabel('Percentile (p)')
#         str='Change of last decade of faf-heat vs control ($t_0$ is last decade control and $t_1$ is last decade faf-heat), '
#         str+=model
#         plt.title(str)
    


#         #the bins are far too small, let's make them of 1% size for now. use the ceiling function
#         index_water=np.ceil(percentiles_water)

#         D=np.empty(100)

#         for i in np.linspace(1,100,100):
#             i=np.int(i)-1
#             D[i]=salt_water_mean.where(index_water==i+1).min()-salt_con_mean.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
#         fig, ax = plt.subplots(figsize=(10,6))
#         plt.scatter(D,np.linspace(1,100,100))
#         plt.axvline(D.mean(),color="tab:red")
#         plt.xlabel('$\mathcal{S}^p(p,t_1)-\mathcal{S}^p(p,t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
#         plt.ylabel('Percentile (p)')
#         str='Change of last decade of faf-water vs control ($t_0$ is last decade control and $t_1$ is last decade faf-water), '
#         str+=model
#         plt.title(str)

#         #the bins are far too small, let's make them of 1% size for now. use the ceiling function
#         index_all=np.ceil(percentiles_all)

#         for i in np.linspace(1,100,100):
#             i=np.int(i)-1
#             D[i]=salt_all_mean.where(index_all==i+1).min()-salt_con_mean.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
#         fig, ax = plt.subplots(figsize=(10,6))
#         plt.scatter(D,np.linspace(1,100,100))
#         plt.axvline(D.mean(),color="tab:red")
#         plt.xlabel('$\mathcal{S}^p(p,t_1)-\mathcal{S}^p(p,t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
#         plt.ylabel('Percentile (p)')
#         str='Change of last decade of faf-all vs control ($t_0$ is last decade control and $t_1$ is last decade faf-all), '
#         str+=model
#         plt.title(str)
def percentiles_input2d(salt_forced,salt_control,salt_reference,field,experiment,model):
# This function can be used to generate plots of the change in the minimum salinity associated with each percentile between the last decade of a forced experiment and the last decade of control
#default is to not coarse grain it

#The first set of salts here (e.g. salt_forced, salt_control etc) are the salt field that we want to categorize. The second set (salt_reference ) are the salt field that we find the percentiles relative to. 
#The idea is that we could, for example, cateogirze interior mean salinity change relative to surface percentiles

    lon = np.array(salt_forced.longitude)
    lat = np.array(salt_forced.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                             persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    def weighted_percentile_of_score(a, weights, score, kind='weak'): #from https://stackoverflow.com/questions/48252282/weighted-version-of-scipy-percentileofscore
        npa = np.array(a)
        npw = np.array(weights)

        indx = npa <= score
        weak = 100 * sum(npw[indx]) / sum(weights)
        if kind == 'weak':
            return weak

    def find_percentiles_weighted(surface_data,weights):
        A=surface_data.stack(z=("latitude", "longitude"))
        B=weights.stack(z=("latitude", "longitude"))
        B=B[A.notnull()]
        A=A[A.notnull()] #.sortby(A[A.notnull()]) #so this is an array that has all not null values from max to min


        def percentiles_np(data,weights):
            return [weighted_percentile_of_score(data,weights,i,kind='weak') for i in data]

        p=(xr.apply_ufunc(percentiles_np,A,B)).unstack() #this allows us to apply a function to an xarray that expects a numpy array
        return p

        #take mean of last 10years of data
    salt_con_reference_mean=salt_reference[60:71,:,:].mean('time')
    salt_con_reference_mean=salt_con_reference_mean.where(salt_con_reference_mean.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_con_mean=salt_control.where(salt_control.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    salt_forced_mean=salt_forced.where(salt_forced.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    percentiles_con=find_percentiles_weighted(salt_con_reference_mean,area)

    #the bins are far too small, let's make them of 1% size for now. use the ceiling function
    index=np.ceil(percentiles_con)

    #now we want to plot this so the y axis is the percentiles and then the x axis is the change in psu between experiments

    D=np.empty(100)

    def area_weighted(area_hadom3,index,salt_surface):
        return ((salt_surface*area_hadom3).where(index==i+1)).sum()/((area_hadom3.where(index==i+1)).sum())

    for i in np.linspace(1,100,100):
        i=np.int(i)-1
        D[i]=area_weighted(area,index,salt_forced_mean-salt_con_mean) #this is how the minimum salinity (bounding salinity) in a percentile is changing  
    fig, ax = plt.subplots(figsize=(10,6))
    plt.scatter(D,np.linspace(1,100,100))
    plt.axvline(D.mean(),color="tab:red")
    plt.xlabel('Mean change within a percentile')
    plt.ylabel('Percentile (p)')
    str='Mean change of '
    str+=field
    str+=' in percentiles characterized by the control surface field between '
    str+=experiment
    str+=' and control, '
    str+=model
    plt.title(str)

    return D
    
        





def percentiles_compute(salt_con,salt_field,coarse=0):
    lon = np.array(salt_con.longitude)
    lat = np.array(salt_con.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                             persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    def weighted_percentile_of_score(a, weights, score, kind='weak'): #from https://stackoverflow.com/questions/48252282/weighted-version-of-scipy-percentileofscore
        npa = np.array(a)
        npw = np.array(weights)

        indx = npa <= score
        weak = 100 * sum(npw[indx]) / sum(weights)
        if kind == 'weak':
            return weak

    def find_percentiles_weighted(surface_data,weights):
        A=surface_data.stack(z=("latitude", "longitude"))
        B=weights.stack(z=("latitude", "longitude"))
        B=B[A.notnull()]
        A=A[A.notnull()] #.sortby(A[A.notnull()]) #so this is an array that has all not null values from max to min


        def percentiles_np(data,weights):
            return [weighted_percentile_of_score(data,weights,i,kind='weak') for i in data]

        p=(xr.apply_ufunc(percentiles_np,A,B)).unstack() #this allows us to apply a function to an xarray that expects a numpy array
        return p

        #take mean of last 10years of data
    salt_field_surface=salt_field[60:71,0,:,:].mean('time')
    salt_field_surface=salt_field_surface.where(salt_field_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5)).where(salt_field_surface>29)

    salt_con_surface=salt_con[60:71,0,:,:].mean('time')
    salt_con_surface=salt_con_surface.where(salt_con_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5)).where(salt_con_surface>29)


    percentiles_con=find_percentiles_weighted(salt_con_surface,area)
    percentiles_field=find_percentiles_weighted(salt_field_surface,area)



    def area_weighted(area_hadom3,index,salt_surface):
        return ((salt_surface*area_hadom3).where(index==i+1)).sum()/((area_hadom3.where(index==i+1)).sum())


    if coarse==1:
        #the bins are far too small, let's make them of .1% size for now. use the ceiling function
        index_pt1=np.round(percentiles_con,1)
        index_field_pt1=np.round(percentiles_field,1)
        D_pt1=np.empty(1000)
        for i in np.linspace(1,1000,1000):
            i=np.int(i)-1
            D_pt1[i]=salt_field_surface.where(index_field_pt1==0.1*i+1).min()-salt_con_surface.where(index_pt1==0.1*i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing  
        D=np.empty(100)
        for i in range(0,100):
            D[i]=np.nanmean(D_pt1[10*(i):10*(i)+11])

    if coarse==0:
        index=np.ceil(percentiles_con)
        index_field=np.ceil(percentiles_field)
        D=np.empty(100)
        for i in np.linspace(1,100,100):
            i=np.int(i)-1
            #D[i]=area_weighted(area_hadom3,index,salt_heat_surface)-area_weighted(area_hadom3,index,salt_con_surface) #the mean change in a percentile according to the control definition of percentile
            D[i]=salt_field_surface.where(index_field==i+1).min()-salt_con_surface.where(index==i+1).min() #this is how the minimum salinity (bounding salinity) in a percentile is changing

    return D

def plot_percentile_change(D_field,experiment):

    fig, ax = plt.subplots(figsize=(10,6))
    plt.scatter(D_field[0:100],np.linspace(1,100,100))
    plt.axvline(D_field[0:100].mean(),color="tab:blue")
    plt.xlabel('$S^p(t_1)-S^p(t_0)$ (psu)') #where t_1 is mean of last decade of faf-heat and t_0 is mean of last decade of faf-control
    plt.ylabel('Percentile (p)')
    str='Change of last decade of '
    str+=experiment
    str+='vs control ($t_0$ is last decade control and $t_1$ is last decade '
    str+=experiment
    str+=')'
    plt.title(str)


#Now we include some functions to model the pdfs using Gaussian Mixture Models. This helps us conclude what is happening with the distribution when we say changes in percentile space

def GMM(salt_experiment,k,experiment):
    Lx_div12=20
    x=np.linspace(31,38,12*Lx_div12+1) # salinities on the x axis that we using for (discretized) CDF

    lon = np.array(salt_experiment.longitude)
    lat = np.array(salt_experiment.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                                persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    #take mean of last 10years of data
    salt_experiment_surface=salt_experiment[60:71,0,:,:].mean('time')
    salt_experiment_surface=salt_experiment_surface.where(salt_experiment_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    # getting data of the histogram for CONTROL
    count, bins_count = np.histogram(salt_experiment_surface, bins=x,weights=area)
    pdf_experiment = count / sum(count)

    ###
    n=100000
    X=np.zeros(n)
    for i in range(0,n):
        X[i]=np.random.choice(bins_count[1:], p=pdf_experiment)
    X2=X.reshape(-1,1)
    gm = GaussianMixture(n_components=k, tol=0.001,n_init=40).fit(X2) #rather than fit to the pdf, we want to fit to random numbers sampled from the pdf

    ######
    fig, ax = plt.subplots()
    # Compute PDF of whole mixture
    x=np.linspace(31,38,n) 
    logprob = gm.score_samples(x.reshape(-1, 1)) #model outputs log probabilities
    pdf = np.exp(logprob)

    # Plot PDF of whole model
    # x.plot(x, pdf, '-k', label='Mixture PDF')


    # Compute PDF for each component
    responsibilities = gm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]



    # getting data of the histogram
    count_a, bins_count_a = np.histogram(X, bins=x)

    area = sum(np.diff(bins_count_a)*count_a)

    weights=np.empty(k)
    for i in range(0,k):
        weights[i]=gm.weights_[i]
    
    # finding the PDF of the histogram using count values
    X2 = count_a / sum(count_a)

    fig, ax = plt.subplots()
    plt.plot(x[1:],X2)
    ax.set_xlabel('Salinity (psu)')
    ax.set_ylabel('Area')
    str3='PDF of surface salinity in '
    str3+=experiment
    plt.title(str3)
    print(X2.sum())

    fig, ax = plt.subplots()
    plt.plot(x[1:],X2,alpha=0.3)

    # Plot PDF of each component
    ax.plot(x, pdf_individual*weights/area, '--', label='Component PDF')
    # Plot PDF of whole model
    ax.plot(x, pdf_individual.dot(weights)/area, '-k', label='Mixture PDF')
    str2='Gaussian Mixture Model with '
    str2+=str(k)
    str2+=' components in '
    str2+=experiment
    plt.title(str2)


    ax.set_xlabel('Salinity (psu)')
    ax.set_ylabel('Area')

    #Turn means and covariances into proper numpy arrays
    means=np.concatenate(gm.means_, axis=0 )
    sigma=np.concatenate(np.concatenate(np.sqrt(gm.covariances_)),axis=0)
    weights=gm.weights_

    means=[x for _,x in sorted(zip(means,means))]
    sigma=[x for _,x in sorted(zip(means,sigma))]
    weights=[x for _,x in sorted(zip(means,gm.weights_))]

    return means, sigma, weights, gm

def GMM_timedep(salt_experiment,k,experiment,precise=0):
    np.random.seed(0)
    plt.rcParams['figure.dpi'] = 200
    Lx_div12=20
    x=np.linspace(31,38,12*Lx_div12+1) # salinities on the x axis that we using for (discretized) CDF

    lon = np.array(salt_experiment.longitude)
    lat = np.array(salt_experiment.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                                persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    salt_experiment_surface=salt_experiment
    salt_experiment_surface=salt_experiment_surface.where(salt_experiment_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    # getting data of the histogram for CONTROL
    count, bins_count = np.histogram(salt_experiment_surface, bins=x,weights=area)
    pdf_experiment = count / sum(count)

    ###
    n=100000
    X=np.zeros(n)
    for i in range(0,n):
        X[i]=np.random.choice(bins_count[1:], p=pdf_experiment)
    X2=X.reshape(-1,1)
    if precise==1:
        gm = GaussianMixture(n_components=k, tol=1E-4, max_iter=10000,n_init=1000,random_state=0).fit(X2) #rather than fit to the pdf, we want to fit to random numbers sampled from the pdf #previously had 500 ninit and 1E-3 tol. didn't have max iter before
    elif precise==0:
        gm = GaussianMixture(n_components=k, tol=1E-3, n_init=40,random_state=0).fit(X2)
    
    ######
    fig, ax = plt.subplots()
    # Compute PDF of whole mixture
    x=np.linspace(31,38,n) 
    logprob = gm.score_samples(x.reshape(-1, 1)) #model outputs log probabilities
    pdf = np.exp(logprob)

    # Plot PDF of whole model
    # x.plot(x, pdf, '-k', label='Mixture PDF')


    # Compute PDF for each component
    responsibilities = gm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]



    # getting data of the histogram
    count_a, bins_count_a = np.histogram(X, bins=x)

    area = sum(np.diff(bins_count_a)*count_a)

    weights=np.empty(k)
    for i in range(0,k):
        weights[i]=gm.weights_[i]
    
    # finding the PDF of the histogram using count values
    X2 = count_a / sum(count_a)

    fig, ax = plt.subplots()
    plt.plot(x[1:],X2)
    ax.set_xlabel('Salinity (psu)')
    ax.set_ylabel('Area')
    str3='PDF of surface salinity in '
    str3+=experiment
    plt.title(str3)
    print(X2.sum())

    fig, ax = plt.subplots()
    plt.plot(x[1:],X2,alpha=0.3)

    # Plot PDF of each component
    #ax.plot(x, pdf_individual*weights/area, '--', label='Component PDF')
    ax.plot(x, pdf_individual*weights/area, '--', label='Component PDF')
    # Plot PDF of whole model
    ax.plot(x, pdf_individual.dot(weights)/area, '-k', label='Mixture PDF')
    str2='Gaussian Mixture Model with '
    str2+=str(k)
    str2+=' components in '
    str2+=experiment
    plt.title(str2)


    ax.set_xlabel('Salinity (psu)')
    ax.set_ylabel('Area')

    #Turn means and covariances into proper numpy arrays
    means=np.concatenate(gm.means_, axis=0 )
    sigma=np.concatenate(np.concatenate(np.sqrt(gm.covariances_)),axis=0)
    weights=gm.weights_

    means=[x for _,x in sorted(zip(means,means))]
    sigma=[x for _,x in sorted(zip(means,sigma))]
    weights=[x for _,x in sorted(zip(means,gm.weights_))]

    return means, sigma, weights, gm

def AIC_BIC(salt_experiment,experiment):
    Lx_div12=20
    x=np.linspace(31,38,12*Lx_div12+1) # salinities on the x axis that we using for (discretized) CDF

    lon = np.array(salt_experiment.longitude)
    lat = np.array(salt_experiment.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                                persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    #take mean of last 10years of data
    salt_experiment_surface=salt_experiment[60:71,0,:,:].mean('time')
    salt_experiment_surface=salt_experiment_surface.where(salt_experiment_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    # getting data of the histogram for CONTROL
    count, bins_count = np.histogram(salt_experiment_surface, bins=x,weights=area)
    pdf_experiment = count / sum(count)

    ###
    n=100000
    X=np.zeros(n)
    for i in range(0,n):
        X[i]=np.random.choice(bins_count[1:], p=pdf_experiment)
    X2=X.reshape(-1,1)

    # Fit models with 1-10 components
    k_arr = np.arange(10) + 1
    models = [
    GaussianMixture(n_components=k1, tol=0.001,n_init=40).fit(X2) #rather than fit to the pdf, we want to fit to random numbers sampled from the pdf
    for k1 in k_arr
    ]

    fig, ax = plt.subplots()
    # Compute metrics to determine best hyperparameter
    AIC = [m.aic(X2) for m in models]
    BIC = [m.bic(X2) for m in models]
    # Plot these metrics
    plt.plot(k_arr, AIC, label='AIC')
    plt.plot(k_arr, BIC, label='BIC')
    plt.xlabel('Number of Components ($k$)')
    str='AIC and BIC for '
    str+=experiment
    plt.title(str)
    plt.legend()

    fig, ax = plt.subplots()
    # Compute metrics to determine best hyperparameter
    AIC = [m.aic(X2) for m in models]
    BIC = [m.bic(X2) for m in models]
    # Plot these metrics
    plt.plot(k_arr, np.gradient(AIC), label='Gradient of AIC')
    plt.plot(k_arr, np.gradient(BIC), label='Gradient of BIC')
    plt.xlabel('Number of Components ($k$)')
    str='Gradient of AIC and BIC for '
    str+=experiment
    plt.title(str)
    plt.legend()

def AIC_BIC_timedep(salt_experiment,experiment):
    Lx_div12=20
    x=np.linspace(31,38,12*Lx_div12+1) # salinities on the x axis that we using for (discretized) CDF

    lon = np.array(salt_experiment.longitude)
    lat = np.array(salt_experiment.latitude)

    #first get the area grid
    area=area_grid(latitudes=np.array(lat),longitudes=lon)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[lat,lon])

    #get the locations of basins
    grid = xr.Dataset()
    grid["lon"] = xr.DataArray(lon, dims=("lon"))
    grid["lat"] = xr.DataArray(lat, dims=("lat"))
    codes = generate_basin_codes(grid, lon="lon", lat="lat",
                                persian=True, style='cmip6')
    grid=grid.rename({'lat': 'latitude','lon': 'longitude'})
    codes=codes.rename({'lat': 'latitude','lon': 'longitude'})

    salt_experiment_surface=salt_experiment
    salt_experiment_surface=salt_experiment_surface.where(salt_experiment_surface.latitude<65).where((codes==1) | (codes==2)| (codes==3)| (codes==5))

    # getting data of the histogram for CONTROL
    count, bins_count = np.histogram(salt_experiment_surface, bins=x,weights=area)
    pdf_experiment = count / sum(count)

    ###
    n=100000
    X=np.zeros(n)
    for i in range(0,n):
        X[i]=np.random.choice(bins_count[1:], p=pdf_experiment)
    X2=X.reshape(-1,1)

    # Fit models with 1-15 components
    k_arr = np.arange(15) + 1
    models = [
    GaussianMixture(n_components=k1, tol=0.001,n_init=40).fit(X2) #rather than fit to the pdf, we want to fit to random numbers sampled from the pdf
    for k1 in k_arr
    ]

    fig, ax = plt.subplots()
    # Compute metrics to determine best hyperparameter
    AIC = [m.aic(X2) for m in models]
    BIC = [m.bic(X2) for m in models]
    # Plot these metrics
    plt.plot(k_arr, AIC, label='AIC')
    plt.plot(k_arr, BIC, label='BIC')
    plt.xlabel('Number of Components ($k$)')
    str='AIC and BIC for '
    str+=experiment
    plt.title(str)
    plt.legend()

    fig, ax = plt.subplots()
    # Compute metrics to determine best hyperparameter
    AIC = [m.aic(X2) for m in models]
    BIC = [m.bic(X2) for m in models]
    # Plot these metrics
    plt.plot(k_arr, np.gradient(AIC), label='Gradient of AIC')
    plt.plot(k_arr, np.gradient(BIC), label='Gradient of BIC')
    plt.xlabel('Number of Components ($k$)')
    str='Gradient of AIC and BIC for '
    str+=experiment
    plt.title(str)
    plt.legend()
############
#The functions in this document are used to perform and then show the clustering of the surface salinity distribution. 
#The first function GMM_timedep takes in the salt field and number of clusters and fits a Gaussian mixture model to the surface salinity distribution. The _timedep ending to the function refers to the fact that the user should pass in a field that is already averaged over the appropriate times (rather than the function receiving a timeseries of salt and using a particular time slice for the fit)
#The second function computes the AIC (Akaike information criterion) and BIC (Bayesian information criterion) metrics for the salt field that is given. This helps to choose the appropriate number of clusters.
#The last function categorizes each surface point into clusters using the highest probability that that point's salinity falls into a given Gaussian from previous functions. It then plots the surface as clustered.
###############

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

def GMM_timedep(salt_experiment,k,experiment,precise=0):
    #----------------------------------------------------
    #This function has input of:
        # - salt_experiment: salt field with a time mean over the period of time of interest from the dataset of interest
        # - k: number of clusters
        # - experiment: A string describing where the salt field came from that is placed on the plot of the distribution
        # - (OPTIONAL) precise: By default the GMM is found by 40 initial conditions and iterates with a tolerance of 1E-3. Optionally, if you set precise=1, it will instead use 1000 initial conditions and a tolerance of 1E-4. This greatly increases the time for the function to run.
    # This function has output of:
        # - means: means of each gaussian from the fit
        # - sigma: standard deviation of each Gaussian from the fit
        # - weights: weights of each Gaussian from the fit
        # - gm: The gaussian mixture model fit to the appropriate data
        # This function also outputs plots of the PDF of the surface salinity both with and without the GMM fit to it.
    #-----------------------------------------------------
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


def AIC_BIC_timedep(salt_experiment,experiment):
    #----------------------------------------------------
    #This function has input of:
        # - salt_experiment: salt field with a time mean over the period of time of interest from the dataset of interest
        # - experiment: A string describing where the salt field came from that is placed on the plot
        # By default, this function uses 40 initial conditions and a tolerance of 1E-3 for fitting the GMMs
    # This function outputs a plot of the AIC and BIC metrics for different numbers of clusters
    #-----------------------------------------------------
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

def clusters(gm,salt,title,k,matching_paper=0):
    #----------------------------------------------------
    #This function has input of:
        # - gm: output from GMM_timedep above. This it the gaussian mixture model fit
        # - salt_experiment: salt field with a time mean over the period of time of interest from the dataset of interest
        # - title: title of the map showing clusters (string)
        # - k: number of clusters
        # - (OPTIONAL) matching_paper. THere are two colour schemes that can be used here. By default if matching_paper not supplied it is one option. We also have the option to give matching_paper=1 which uses another colour scheme that we are going to use for the paper so all the plots can match
    # This function has output of:
        # - y_disjoint: an array with all lat and lon and then the location of each gaussian marked using the gaussian dimension
        #- a2: the index locations in the vector of salinities x=np.linspace(31,38,10000) where each Gaussian starts and stops
    #-----------------------------------------------------
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
    for i in range(0,k):
        y_disjoint[:,:,i]=xr.where((s<(x[a2[i+1]]))&(s>(x[a2[i]])),i+1,0)

    if matching_paper==0:
        if n==6:
            colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59']
        elif n==4:
            colorsList = ['#f6eff7','#d0d1e6','#1c9099','#016c59']
        elif n==9:
            colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#225ea8','#253494','#081d58']
        elif n==5:
            colorsList = ['#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59']
        elif n==8:
            colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#253494','#081d58']
        elif n==7:
            colorsList = ['#f6eff7','#d0d1e6','#a6bddb','#67a9cf','#1c9099','#016c59','#253494']
    elif matching_paper==1:
        colorsList=['#ffffcc','#c7e9b4','#7fcdbb','#41b6c4','#2c7fb8','#253494'] #these colours aren't ideal but we use for paper so that the Gaussian mixture plot, plots of salinity and this plot can have consistent colours for each region.
    o=y_disjoint.sum('gaussian').where(y_disjoint.sum('gaussian')>0)
    CustomCmap = matplotlib.colors.ListedColormap(colorsList)
    #CustomCmap.set_bad(color='w')
    fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()},figsize=(10,8),dpi=120) #this is from cartopy https://rabernat.github.io/research_computing_2018/maps-with-cartopy.html
    #p=(o.where(y_disjoint[:,:,1].latitude<65)).plot(cbar_kwargs={'shrink':0.75,'orientation':'horizontal','extend':'both','pad':0.08,'spacing':'proportional'},ax=ax,cmap=CustomCmap,alpha=1) #you have to set a colormap here because plotting xarray infers from the 
    p=(o.where(y_disjoint[:,:,1].latitude<65)).plot(vmin=1,vmax=6,ax=ax,cmap=CustomCmap,alpha=1,add_colorbar=False) #you have to set a colormap here because plotting xarray infers from the 

    cbar = plt.colorbar(p,orientation='horizontal',extend='neither',pad=0.1,shrink=0.85)
    tick_locs = (np.arange(n) + 1.7)*(k-1)/k
    cbar.set_ticks(tick_locs)
    
    # set tick labels (as before)
    cbar.set_ticklabels(np.arange(n)+1)
    
    
    ax.coastlines(color='grey',lw=0.5)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k')
    ax.set_title(title)
    fig.tight_layout()

    return y_disjoint,a2
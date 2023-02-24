############
#The functions in this document are used to perform linear response theory and solve for freshwater fluxes as a proportion of FAFMIP perturbations taking input of time series of salt and temperature in regions identified by clustering functions in clustering_tools.py
#The first function takes a single salt and temp field, the second function takes a list of salt and temp fields and reclusters for each one, and the last funciton takes a list of salt and temp fields that are bootstrapped (and so already clustered, not in lat and lon coordinates)
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

def linear_response(salt,temp,salt_mean,n):
    #----------------------------------------------------
    #Note: this function is different than linear_response in older versions of this work as it characterizes the proportion of FAFMIP response as last 5 years minus first 5 years rather than fitting a linear trend 
    #This function has input of:
        # - salt: The surface salt field that we are applying linear response theory to. Input in time,latitude,longitude format. input over the period of time that you want analyzed
        # - temp: The surface temp field that we are applying linear response theory to. Input in time,latitude,longitude format. input with time cut to period of time that you want analyzed
        # - salt_mean: salt field that the clusters should be made according to (in most cases this should be the same as salt field)
        # - n: number of clusters
    # This function has output of:
        # - change_water1: Output of proportion of FAFMIP freshwater flux over period of interest. Calculated as mean of last 5 years minus first 5 years
        # - change_heat1: Output of proportion of FAFMIP heat flux over period of interest. Calculated as mean of last 5 years minus first 5 years
        # Plots of timeseries of proportion of FAFMIP freshwater flux, heat flux, wind stress 
    #-----------------------------------------------------
    from clustering_tools import GMM_timedep
    from clustering_tools import clusters
    from fafmip_gridding_tools import regridded_fafmip
    from fafmip_gridding_tools import regridded_fafmip_temp  
    area=area_grid(latitudes=np.array(salt[0,:,:].latitude),longitudes=salt[0,:,:].longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt[0,:,:].latitude,salt[0,:,:].longitude])

    mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65),n, 'categorize clusters') 
    y,a2=clusters(gm,salt_mean[0:36,:,:].mean('time'),'Location of each Gaussian',n,matching_paper=1)

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    s=(salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65)
    x=np.linspace(31,38,10000)
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


    start_yr=15
    end_yr=19

    df=np.zeros([50,end_yr-start_yr,3,3])
    df2=np.zeros([50,end_yr-start_yr,3,3])
    df4=np.zeros([50,end_yr-start_yr,3,3])
    dist=np.zeros([50,end_yr-start_yr,3])
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

    fig,ax=plt.subplots(figsize=(10,6))
    plt.plot(df3_mean[:,2])
    plt.plot(df3[:,2,0],':')
    plt.plot(df3[:,2,1],':')
    plt.plot(df3[:,2,2],':')
    plt.title('Wind stress change which affects salinity field (compared to FAFMIP, mean ACCESS-OM2 and MOM5, mean of start years 14-19). Intercept subtracted')
    plt.xlabel('Time')
    plt.ylabel('F(t)')
    plt.legend(['Mean MOM5 and ACCESS-OM2','MOM5','ACCESS-OM2','HadOM3'])

    change_water_1=df3_mean[40:45,0].mean()-df3_mean[0:5,0].mean()
    change_heat_1=df3_mean[40:45,1].mean()-df3_mean[0:5,1].mean()

    return change_water_1, change_heat_1


def linear_response_list(salt,temp,n): 
   #----------------------------------------------------
    #Note: this function is different than the older linear_response_list2 in older versions of this work as it properly reclusters each member of the list
    #This function takes in salt and temperature fields as a list and then clusters them and applies linear response theory to each one
    #This function has input of:
        # - salt: List of the surface salt fields that we are applying linear response theory to. Each list member input in time,latitude,longitude format. input over the period of time that you want analyzed
        # - temp: List of the surface temp fields that we are applying linear response theory to. Each list member input in time,latitude,longitude format. input with time cut to period of time that you want analyzed
        # - n: number of clusters
    # This function has output of:
        # - change_water1: List of output of proportion of FAFMIP freshwater flux over period of interest. Calculated as mean of last 5 years minus first 5 years
        # - change_heat1: List of output of proportion of FAFMIP heat flux over period of interest. Calculated as mean of last 5 years minus first 5 years
        # Plots of timeseries of proportion of FAFMIP freshwater flux, heat flux, wind stress 
    #-----------------------------------------------------
#input a list of salt and temp
    from clustering_tools import GMM_timedep
    from clustering_tools import clusters
    from fafmip_gridding_tools import regridded_fafmip
    from fafmip_gridding_tools import regridded_fafmip_temp   
    area=area_grid(latitudes=np.array(salt[0].latitude),longitudes=salt[0].longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt[0].latitude,salt[0].longitude])


    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    change_water_1=np.empty([len(salt)])

    change_heat_1=np.empty([len(salt)])

    nn=int(np.size(salt[0][:,0,0])/12)

    for q in range(0,len(salt)):
        mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt[q][0:36,:,:].mean('time')).where(salt[0].latitude<65),n,'distribution',plot=1) 
        y,a2=clusters(gm,salt[q][0:36,:,:].mean('time'),'Location of each Gaussian',n,plot=1)

        temp_mit_stress,temp_mom_stress,temp_had_stress,temp_access_stress,temp_mit_heat,temp_mom_heat,temp_had_heat,temp_access_heat,temp_mit_water,temp_mom_water,temp_had_water,temp_access_water=regridded_fafmip_temp(salt[q],area,a2,n)
        salt_mit_stress,salt_mom_stress,salt_had_stress,salt_access_stress,salt_mit_heat,salt_mom_heat,salt_had_heat,salt_access_heat,salt_mit_water,salt_mom_water,salt_had_water,salt_access_water=regridded_fafmip(salt[q],area,a2,n)


        s=(salt[q][0:36,:,:].mean('time')).where(salt[q].latitude<65) #this was 12 and i changed to 36 on 4 aug
        x=np.linspace(31,38,10000)
        #we have a 50 year time series, we want to find the mean salt at each region defined by the first year at each of these years
        salt_cesm2=np.empty([nn,n,len(salt)])
        temp_cesm2=np.empty([nn,n,len(salt)])
        for j in range(0,nn):
            s_new=((salt[q])[j*12:(j+1)*12,:,:].mean('time')).where(salt[0].latitude<65)
            t_new=((temp[q])[j*12:(j+1)*12,:,:].mean('time')).where(salt[0].latitude<65)
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

        change_water_1[q]=df3_mean[40:45,0].mean()-df3_mean[0:5,0].mean()
        change_heat_1[q]=df3_mean[40:45,1].mean()-df3_mean[0:5,1].mean()
    return change_water_1, change_heat_1
    
    
def linear_response_list_bootstrap(salt,temp,salt_mean,n,a2,obs=0,weighted=0,area_cluster=np.ones(6),returns=2): 
   #----------------------------------------------------
    #This function takes in salt and temperature fields as a bootstrapped list list. Thus, the preprocessing of clustering and then bootstrapping around has already been done
    #This function has input of:
        # - salt: List of the surface salt fields that we are applying linear response theory to. Each list member input in time,latitude,longitude format. input over the period of time that you want analyzed
        # - temp: List of the surface temp fields that we are applying linear response theory to. Each list member input in time,latitude,longitude format. input with time cut to period of time that you want analyzed
        # - a2: the index locations in the vector of salinities x=np.linspace(31,38,10000) where each Gaussian starts and stops from clustering of the realization
        # - salt_mean: give it the salt field that was clustered on so that it can put the fafmip data in the correct clusters
        # - n: number of clusters
        # - obs=0, 0 if data is monthly and 1 if data is yearly
        # - weighted determines whether to solve the linear response problem weighted by the area of the cluster
        # - area_cluster is an input of the area of each cluster (vector of size n)
        # - returns=2 is default and gives a list of output of the proportion of freshwater flues and heat fluxes over the period of interest. If returns=4, we also include two more lists that have the same thing but before we took the mdean across ocean models -- thus it can be used to get a sense of spread in response depending which ocean model is used as the response function. The paper uses returns=2
    # This function has output of:
        # - change_water1: List of output of proportion of FAFMIP freshwater flux over period of interest. Calculated as mean of last 5 years minus first 5 years
        # - change_heat1: List of output of proportion of FAFMIP heat flux over period of interest. Calculated as mean of last 5 years minus first 5 years
        # - if returns=4, the same as above plus 2 more outputs that include change_water1 and change_heat1 before means are taken across ocean models as response functions
        # Plots of timeseries of proportion of FAFMIP freshwater flux, heat flux, wind stress 
    #-----------------------------------------------------
#input a list of salt and temp
    from clustering_tools import GMM_timedep
    from clustering_tools import clusters
    from fafmip_gridding_tools import regridded_fafmip
    from fafmip_gridding_tools import regridded_fafmip_temp  
    area=area_grid(latitudes=np.array(salt_mean.latitude),longitudes=salt_mean.longitude)
    area=xr.DataArray(area,dims=["latitude","longitude"],coords=[salt_mean.latitude,salt_mean.longitude])

    #mean_had_con,sigma_had_con,weights_had_con,gm=GMM_timedep((salt_mean[0:36,:,:].mean('time')).where(salt_mean.latitude<65),n,'First year, 2015, SSP8.5') 
    #y,a2=clusters(gm,salt_mean[0:36,:,:].mean('time'),'Location of each Gaussian, categorized by first time point of SSP8.5 run, CESM2',n)

    def area_weighted_disjoint(area,i,salt_surface,thing_to_weight,x,a2):
        return ((thing_to_weight*area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()/((area).where(salt_surface>(x[a2[i]])).where(salt_surface<(x[a2[i+1]]))).sum()
    
    change_water_1=np.empty([len(salt)])


    change_heat_1=np.empty([len(salt)])
    change_water_modelspread=np.empty([len(salt),3])
    change_heat_modelspread=np.empty([len(salt),3])

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
                    if weighted==0:
                        x, residuals, rank, s=np.linalg.lstsq(A,np.matrix(RHS).T,rcond = -1)
                    elif weighted==1:
                        W = np.concatenate([area_cluster,area_cluster])
                        W = np.sqrt(np.diag(W))
                        Aw = np.dot(W,A)
                        Bw = np.dot(np.matrix(RHS),W)
                        x, residuals, rank, s= np.linalg.lstsq(Aw, Bw.T,rcond = -1)
                    df[i,start-start_yr,:,p]=x.reshape(3)
        
        
                for k in range(0,3):
                    df2[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum())-(df[:,start-start_yr,k,p].cumsum())[0] #subtract off so starts from 0
                    df4[:,start-start_yr,k,p]=(df[:,start-start_yr,k,p].cumsum()) #don't subtract off so starts at 0
                df3=np.mean(df2,axis=1) #mean over the start years where we subtracted off
                df5=np.mean(df4,axis=1) #mean over the start years where we didn't make start from 0
            change_water_modelspread[q,p]=df3[40:45,0,p].mean()-df3[0:5,0,p].mean() #also include all versions from different ocean models
            change_heat_modelspread[q,p]=df3[40:45,1,p].mean()-df3[0:5,1,p].mean()
        df3_mean=np.mean(df3,axis=2)
        df5_mean=np.mean(df5,axis=2)

        change_water_1[q]=df3_mean[40:45,0].mean()-df3_mean[0:5,0].mean()
        change_heat_1[q]=df3_mean[40:45,1].mean()-df3_mean[0:5,1].mean()
        

    if returns==2:
        return change_water_1, change_heat_1
    if returns==4:
        return change_water_1, change_heat_1, change_water_modelspread.ravel(), change_heat_modelspread.ravel()
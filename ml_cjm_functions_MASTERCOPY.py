#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
#
# Use "module load scitools/experimental-current" at the command line before running this.

# Import some modules
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from scipy import stats

# Define some functions

def initialise():
    f=open("expt_name_for_picking_up.txt", "r")
    a=f.readline()
    f.close()
    return {'expt_name':a};

def read_in_for_one_subdomain(domain,date_range,regions_to_process,setup,monthyear,ndays):
    # N.B. The files being read in have data for only one subdomain (out of 64) to reduce file size 
    #      but they do contain data for all 99 regions, and for range of heights and for multiple days.
    nc_file   = '/data/nwp1/frme/ML/'+monthyear+'/'+date_range+'-ml_aggregate_data-subdomain'+str(domain)+'.nc'
    fh        = Dataset(nc_file, mode='r')
    # Using original varibale names
    theta     = fh.variables['liquid_ice_static_potential_temperature_K'][:]
    theta_adv = fh.variables['adv_flux_thetali_Ks-1'][:]
    qt        = fh.variables['total_humidity_qvqclqcfqrainqgraupel_kgkg-1'][:]
    qt_adv    = fh.variables['adv_flux_qtotal_kgkg-1s-1'][:]
    # Using netcdf compliant variable names
    # theta     = fh.variables['liquid_ice_static_potential_temperature'][:]
    # theta_adv = fh.variables['net_advective_flux_liquid_ice_static_potential_temperature'][:]
    # qt        = fh.variables['total_specific_humidity'][:]
    # qt_adv    = fh.variables['net_advective_flux_total_specific_humidity'][:]
    toa_sw    = fh.variables['toa_incoming_shortwave_flux'][:]
    shf       = fh.variables['surface_upward_sensible_heat_flux'][:]
    lhf       = fh.variables['surface_upward_latent_heat_flux'][:]
    rain      = fh.variables['stratiform_rainfall_flux'][:]
    snow      = fh.variables['stratiform_snowfall_flux'][:]
    bcf       = fh.variables['bulk_cloud_fraction'][:]
    lon       = fh.variables['LonRegionCentreDeg'][:]
    hours     = fh.variables['HoursSince2016_01_01_00Z'][:]
    fh.close()
    #
    tmp1,tmp2,tmp3,nr=theta.shape
    #
    #Trim down data to lowest 50 of the 70 levels
    nz        = setup['nz_trim']
    theta     = theta[:,0:nz,:,:]
    qt        = qt[:,0:nz,:,:]
    # Read in to one level higher than needed in case we apply vertical averaging.
    theta_adv = theta_adv[:,0:nz+1,:,:]
    qt_adv    = qt_adv[:,0:nz+1,:,:]
    # Data has been output every 2 hours, so there are 12 data samples per day
    tsperday=12
    #Trim down data to first ndays days (so will work on all months apart fro February).
    nt=(ndays-1)*tsperday
    shift=setup['shift']
    # Start reading in the data from tsperday-1, so effectively ignore whole 02Z-22Z of first day 
    # and keep from 00Z on second day onwards.
    # Also only consider a sub set of the regions.
    # If shift=1, then the "next" variable is 2 hours later, if shift=2 then it is 4 hours later etc.
    theta_now     = theta    [tsperday-1       :(tsperday-1)+nt,      :,:,regions_to_process]
    theta_next    = theta    [tsperday-1+shift :(tsperday-1)+nt+shift,:,:,regions_to_process]
    qt_now        = qt       [tsperday-1       :(tsperday-1)+nt,      :,:,regions_to_process]
    qt_next       = qt       [tsperday-1+shift :(tsperday-1)+nt+shift,:,:,regions_to_process]
    if setup['smooth_adv']==1:
        # For smoothing adv increments
        theta_tmp = theta_adv*0.0
        qt_tmp    = qt_adv*0.0
        for r_ind in np.arange(0,nr,1):
            for k_ind in np.arange(1,nz,1):
                for t_ind in np.arange(1,tmp1,1):
                    theta_tmp[t_ind,k_ind,0,r_ind]=np.mean(theta_adv[t_ind-1:t_ind,k_ind-1:k_ind+1,0,r_ind])
                    qt_tmp   [t_ind,k_ind,0,r_ind]=np.mean(   qt_adv[t_ind-1:t_ind,k_ind-1:k_ind+1,0,r_ind])
        theta_adv = theta_tmp[tsperday-1:(tsperday-1)+nt,0:nz,:,regions_to_process]
        qt_adv    = qt_tmp   [tsperday-1:(tsperday-1)+nt,0:nz,:,regions_to_process]
    else:
    # Use the advection increments as they are (but now ignore top level that we did not need to read in).
        theta_adv = theta_adv[tsperday-1:(tsperday-1)+nt,0:nz,:,regions_to_process]
        qt_adv    = qt_adv   [tsperday-1:(tsperday-1)+nt,0:nz,:,regions_to_process]
    #end if
    #
    rain_p1       = rain     [tsperday-1+1:  (tsperday-1)+nt+1,  :,:,regions_to_process]
    snow_p1       = snow     [tsperday-1+1:  (tsperday-1)+nt+1,  :,:,regions_to_process]
    ppn           = rain_p1 + snow_p1 
    #
    rain_p2       = rain     [tsperday-1+2:(tsperday-1)+nt+2,:,:,regions_to_process]
    snow_p2       = snow     [tsperday-1+2:(tsperday-1)+nt+2,:,:,regions_to_process]
    ppn_p2        = rain_p2 + snow_p2
    #
    rain_p3       = rain     [tsperday-1+3:(tsperday-1)+nt+3,:,:,regions_to_process]
    snow_p3       = snow     [tsperday-1+3:(tsperday-1)+nt+3,:,:,regions_to_process]
    ppn_p3        = rain_p3 + snow_p3
    #
    rain_p4       = rain     [tsperday-1+4:(tsperday-1)+nt+4,:,:,regions_to_process]
    snow_p4       = snow     [tsperday-1+4:(tsperday-1)+nt+4,:,:,regions_to_process]
    ppn_p4        = rain_p4 + snow_p4
    #
    rain_p5       = rain     [tsperday-1+5:(tsperday-1)+nt+5,:,:,regions_to_process]
    snow_p5       = snow     [tsperday-1+5:(tsperday-1)+nt+5,:,:,regions_to_process]
    ppn_p5        = rain_p5 + snow_p5
    #
    rain_p6       = rain     [tsperday-1+6:(tsperday-1)+nt+6,:,:,regions_to_process]
    snow_p6       = snow     [tsperday-1+6:(tsperday-1)+nt+6,:,:,regions_to_process]
    ppn_p6        = rain_p6 + snow_p6
    #
    if shift==2:
        ppn=(ppn+ppn_p2)/2
    if shift==3:
        ppn=(ppn+ppn_p2+ppn_p3)/3
    if shift==4:
        ppn=(ppn+ppn_p2+ppn_p3+ppn_p4)/4
    if shift==5:
        ppn=(ppn+ppn_p2+ppn_p3+ppn_p4+ppn_p5)/5
    if shift==6:
        ppn=(ppn+ppn_p2+ppn_p3+ppn_p4+ppn_p5+ppn_p6)/6
    #
    toa_sw_now    = toa_sw   [tsperday-1:(tsperday-1)+nt,  :,:,regions_to_process]
    shf_now       = shf      [tsperday-1:(tsperday-1)+nt,  :,:,regions_to_process]
    lhf_now       = lhf      [tsperday-1:(tsperday-1)+nt,  :,:,regions_to_process]
    toa_sw_before = toa_sw   [tsperday-2:(tsperday-1)+nt-1,:,:,regions_to_process]
    shf_before    = shf      [tsperday-2:(tsperday-1)+nt-1,:,:,regions_to_process]
    lhf_before    = lhf      [tsperday-2:(tsperday-1)+nt-1,:,:,regions_to_process]
    #
    theta_adv=theta_adv*(2.0*60.0*60.0)
    qt_adv=qt_adv*(2.0*60.0*60.0)
    # Calculate change over each time interval
    theta_diff = theta_next - theta_now
    qt_diff    =    qt_next -    qt_now
    #
    # Infer physics increment
    theta_phys = theta_diff - theta_adv
    qt_phys    =    qt_diff -    qt_adv
    #
    # Calculate change in surface and TOA forcings
    toa_sw_diff= toa_sw_now - toa_sw_before
    shf_diff   = shf_now    - shf_before
    lhf_diff   = lhf_now    - lhf_before
    #
    theta_plus_adv = theta_now + theta_adv
    qt_plus_adv    =    qt_now +    qt_adv
    #
    hours_now     = hours    [tsperday-1       :(tsperday-1)+nt]
    dh=24.0*lon/360.0
    cos_term_now=np.empty([nt,99])+np.nan
    sin_term_now=np.empty([nt,99])+np.nan
    cos_term_end=np.empty([nt,99])+np.nan
    sin_term_end=np.empty([nt,99])+np.nan
    for r_ind in np.arange(0,nr,1):
        for t_ind in np.arange(0,nt,1):
            cos_term_now[t_ind,r_ind]=np.cos(2.0*np.pi*(hours_now[t_ind]+dh[r_ind])/24.0)
            sin_term_now[t_ind,r_ind]=np.sin(2.0*np.pi*(hours_now[t_ind]+dh[r_ind])/24.0)
            cos_term_end[t_ind,r_ind]=np.cos(2.0*np.pi*(hours_now[t_ind]+dh[r_ind]+(shift*2.0))/24.0)
            sin_term_end[t_ind,r_ind]=np.sin(2.0*np.pi*(hours_now[t_ind]+dh[r_ind]+(shift*2.0))/24.0)
    cos_term_now=cos_term_now[:,regions_to_process]
    sin_term_now=sin_term_now[:,regions_to_process]
    cos_term_end=cos_term_end[:,regions_to_process]
    sin_term_end=sin_term_end[:,regions_to_process]
    return {'theta_now':theta_now, 'qt_now':qt_now, 'theta_next':theta_next, 'qt_next':qt_next, 'theta_adv':theta_adv, 'qt_adv':qt_adv, 'theta_phys':theta_phys, 'qt_phys':qt_phys, 'toa_sw':toa_sw_now, 'shf':shf_now, 'lhf':lhf_now, 'rain':rain, 'snow':snow, 'ppn':ppn, 'toa_sw_diff':toa_sw_diff, 'shf_diff':shf_diff, 'lhf_diff':lhf_diff,'theta_plus_adv':theta_plus_adv, 'qt_plus_adv':qt_plus_adv, 'bcf':bcf, 'theta_diff':theta_diff, 'qt_diff':qt_diff, 'cos_term_now':cos_term_now, 'sin_term_now':sin_term_now, 'cos_term_end':cos_term_end, 'sin_term_end':sin_term_end};

def calc_profiles_mean_and_std(data):
    nt,nz,tmp3,nr=data['theta_now'].shape
    # Find average theta profile (mean over time and over region, no need to mean over sub-domains as there is only 1 being read in at a time.)
    theta_mean_prof = np.squeeze(np.mean(np.mean(data['theta_now'],axis=3),axis=0))
    qt_mean_prof    = np.squeeze(np.mean(np.mean(data['qt_now'],   axis=3),axis=0))
    #
    # For the standard deviation, calculate it separately at each level
    theta_std_prof = np.zeros([nz])
    qt_std_prof    = np.zeros([nz])
    # Also calculate profile of standard deviation for adv and phys increments (not sure this will be needed but let's try).
    theta_adv_std_prof  = np.zeros([nz])
    qt_adv_std_prof     = np.zeros([nz])
    theta_phys_std_prof = np.zeros([nz])
    qt_phys_std_prof    = np.zeros([nz])
    # Unfold 2d array manually to make it clearer what is going on.
    for k in np.arange(0,nz,1):
        tmp                    = np.reshape(data['theta_now'][:,k,:,:],(nt*nr,1))
        theta_std_prof[k]      = np.std(tmp)
        tmp                    = np.reshape(   data['qt_now'][:,k,:,:],(nt*nr,1))
        qt_std_prof[k]         = np.std(tmp)
        #
        tmp                    = np.reshape(data['theta_adv'][:,k,:,:],(nt*nr,1))
        theta_adv_std_prof[k]  = np.std(tmp)
        tmp                    = np.reshape(   data['qt_adv'][:,k,:,:],(nt*nr,1))
        qt_adv_std_prof[k]     = np.std(tmp)
        #
        tmp                    = np.reshape(data['theta_phys'][:,k,:,:],(nt*nr,1))
        theta_phys_std_prof[k] = np.std(tmp)
        tmp                    = np.reshape(   data['qt_phys'][:,k,:,:],(nt*nr,1)) 
        qt_phys_std_prof[k]    = np.std(tmp)
    # end
    #
    shf_std    = np.std(np.reshape(data['shf'],(nt*nr,1)))
    #    print('shf_std=',shf_std)
    lhf_std    = np.std(np.reshape(data['lhf'],(nt*nr,1)))
    return {'theta_mean_prof':theta_mean_prof, 'qt_mean_prof':qt_mean_prof, 'theta_std_prof':theta_std_prof, 'qt_std_prof':qt_std_prof, 'theta_adv_std_prof':theta_adv_std_prof, 'qt_adv_std_prof':qt_adv_std_prof, 'theta_phys_std_prof':theta_phys_std_prof, 'qt_phys_std_prof':qt_phys_std_prof,'shf_std':shf_std, 'lhf_std':lhf_std};

def calc_profiles_max_min_range(data):
    # Calculate profiles of maximum and minimum values (these will be an option for rescaling variables).
    theta_min_prof = np.squeeze(np.amin(np.amin(data['theta_now'],axis=3),axis=0))
    theta_max_prof = np.squeeze(np.amax(np.amax(data['theta_now'],axis=3),axis=0))
    qt_min_prof    = np.squeeze(np.amin(np.amin(data['qt_now'],   axis=3),axis=0))
    qt_max_prof    = np.squeeze(np.amax(np.amax(data['qt_now'],   axis=3),axis=0))
    #
    theta_adv_maxabs_prof = np.squeeze(np.amax(np.amax(np.abs(data['theta_adv']),axis=3),axis=0))
    qt_adv_maxabs_prof = np.squeeze(np.amax(np.amax(np.abs(data['qt_adv']),axis=3),axis=0))
    #
    theta_phys_maxabs_prof = np.squeeze(np.amax(np.amax(np.abs(data['theta_phys']),axis=3),axis=0))
    qt_phys_maxabs_prof = np.squeeze(np.amax(np.amax(np.abs(data['qt_phys']),axis=3),axis=0))
    shf_max    = np.amax(np.abs(data['shf']))
    lhf_max    = np.amax(np.abs(data['lhf']))
    toa_sw_max = np.amax(data['toa_sw'])
    ppn_max    = np.amax(data['ppn'])
    return {'theta_min_prof':theta_min_prof, 'theta_max_prof':theta_max_prof, 'qt_min_prof':qt_min_prof, 'qt_max_prof':qt_max_prof, 'theta_adv_maxabs_prof':theta_adv_maxabs_prof, 'qt_adv_maxabs_prof':qt_adv_maxabs_prof, 'theta_phys_maxabs_prof':theta_phys_maxabs_prof, 'qt_phys_maxabs_prof':qt_phys_maxabs_prof,'ppn_max':ppn_max, 'toa_sw_max':toa_sw_max, 'shf_max':shf_max, 'lhf_max':lhf_max};

def calc_profiles_centiles(data,setup):
    nt,nz,tmp3,nr=data['theta_now'].shape
    #
    theta_2p5_prof      = np.zeros([nz])
    theta_975_prof      = np.zeros([nz])
    qt_2p5_prof         = np.zeros([nz])
    qt_975_prof         = np.zeros([nz])
    theta_adv_2p5_prof  = np.zeros([nz])
    theta_adv_975_prof  = np.zeros([nz])
    qt_adv_2p5_prof     = np.zeros([nz])
    qt_adv_975_prof     = np.zeros([nz])
    theta_phys_2p5_prof = np.zeros([nz])
    theta_phys_975_prof = np.zeros([nz])
    qt_phys_2p5_prof    = np.zeros([nz])
    qt_phys_975_prof    = np.zeros([nz])
    #
    theta_diff_2p5_prof = np.zeros([nz])
    theta_diff_975_prof = np.zeros([nz])
    qt_diff_2p5_prof    = np.zeros([nz])
    qt_diff_975_prof    = np.zeros([nz])
    #
    tmp                  = np.reshape(data['shf'][:,:,:,:],(nt*nr,1))
    shf_2p5=np.percentile(tmp,setup['clip_centile'])
    shf_975=np.percentile(tmp,(100.0-setup['clip_centile']))
    tmp                  = np.reshape(data['lhf'][:,:,:,:],(nt*nr,1))
    lhf_2p5=np.percentile(tmp,setup['clip_centile'])
    lhf_975=np.percentile(tmp,(100.0-setup['clip_centile']))
    #
    # Unfold 2d array manually to make it clearer what is going on.
    for k in np.arange(0,nz,1):
        tmp                    = np.reshape(data['theta_now'][:,k,:,:],(nt*nr,1))
        theta_2p5_prof[k]      = np.percentile(tmp,setup['clip_centile'])
        theta_975_prof[k]      = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['qt_now'][:,k,:,:],(nt*nr,1))
        qt_2p5_prof[k]         = np.percentile(tmp,setup['clip_centile'])
        qt_975_prof[k]         = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['theta_adv'][:,k,:,:],(nt*nr,1))
        theta_adv_2p5_prof[k]  = np.percentile(tmp,setup['clip_centile'])
        theta_adv_975_prof[k]  = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['qt_adv'][:,k,:,:],(nt*nr,1))
        qt_adv_2p5_prof[k]     = np.percentile(tmp,setup['clip_centile'])
        qt_adv_975_prof[k]     = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['theta_phys'][:,k,:,:],(nt*nr,1))
        theta_phys_2p5_prof[k] = np.percentile(tmp,setup['clip_centile'])
        theta_phys_975_prof[k] = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['qt_phys'][:,k,:,:],(nt*nr,1))
        qt_phys_2p5_prof[k]    = np.percentile(tmp,setup['clip_centile'])
        qt_phys_975_prof[k]    = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['theta_diff'][:,k,:,:],(nt*nr,1))
        theta_diff_2p5_prof[k] = np.percentile(tmp,setup['clip_centile'])
        theta_diff_975_prof[k] = np.percentile(tmp,(100.0-setup['clip_centile']))
        tmp                    = np.reshape(data['qt_diff'][:,k,:,:],(nt*nr,1))
        qt_diff_2p5_prof[k]    = np.percentile(tmp,setup['clip_centile'])
        qt_diff_975_prof[k]    = np.percentile(tmp,(100.0-setup['clip_centile']))
    return {'theta_2p5_prof':theta_2p5_prof, 'theta_975_prof':theta_975_prof, 'qt_2p5_prof':qt_2p5_prof, 'qt_975_prof':qt_975_prof, 'theta_adv_2p5_prof':theta_adv_2p5_prof, 'theta_adv_975_prof':theta_adv_975_prof, 'qt_adv_2p5_prof':qt_adv_2p5_prof, 'qt_adv_975_prof':qt_adv_975_prof, 'theta_phys_2p5_prof':theta_phys_2p5_prof, 'theta_phys_975_prof':theta_phys_975_prof, 'qt_phys_2p5_prof':qt_phys_2p5_prof, 'qt_phys_975_prof':qt_phys_975_prof, 'shf_2p5':shf_2p5, 'shf_975':shf_975, 'lhf_2p5':lhf_2p5, 'lhf_975':lhf_975, 'theta_diff_2p5_prof':theta_diff_2p5_prof, 'theta_diff_975_prof':theta_diff_975_prof, 'qt_diff_2p5_prof':qt_diff_2p5_prof, 'qt_diff_975_prof':qt_diff_975_prof};

def normalise(setup,data,range_prof,mean_prof,centile_prof):
    # Prior to ingesting into machine learning algorithm,the variables need to be "rescaled", "normalised" or "standardised".
    # Often this is described as "subtracting the mean and dividing by the standard deviation". 
    # But this can be done separately at each level, or using data for all levels at once. 
    # Also the "width" that one divide by could be the standard deviation, or the range (maximum minus minimum) 
    # or it could be the 2.5 to 97.5 centile range. And this could be done level by level or for the whole column. Lots of options!
    nt,nz,tmp3,nr=data['theta_now'].shape
    # Average over the levels.
    theta_std     = np.mean(mean_prof['theta_std_prof'])
    qt_std        = np.mean(mean_prof['qt_std_prof'])
    theta_adv_std = np.mean(mean_prof['theta_adv_std_prof'])
    qt_adv_std    = np.mean(mean_prof['qt_adv_std_prof'])
    theta_phys_std = np.mean(mean_prof['theta_phys_std_prof'])
    qt_phys_std    = np.mean(mean_prof['qt_phys_std_prof'])
    #
    # Initialise arrays of correct size
    theta_now_norm  = data['theta_now']*0.0
    qt_now_norm     = data['theta_now']*0.0
    theta_adv_norm  = data['theta_now']*0.0
    qt_adv_norm     = data['theta_now']*0.0
    theta_phys_norm = data['theta_now']*0.0
    qt_phys_norm    = data['theta_now']*0.0
    theta_next_norm  = data['theta_now']*0.0
    qt_next_norm     = data['theta_now']*0.0
    #
    theta_plus_adv_norm  = data['theta_now']*0.0
    qt_plus_adv_norm     = data['theta_now']*0.0
    #
    theta_diff_norm      = data['theta_now']*0.0
    qt_diff_norm         = data['theta_now']*0.0
    #
    # Normalise / Standardise
    #    
    # Normalising by std, leads to some values still at +10
    halfrange=np.amax([centile_prof['shf_975'],-centile_prof['shf_2p5']])
    shf_norm=data['shf']/halfrange
    shf_diff_norm=2.0*data['shf_diff']/halfrange
    #
    halfrange=np.amax([centile_prof['lhf_975'],-centile_prof['lhf_2p5']])
    lhf_norm=data['lhf']/halfrange
    lhf_diff_norm=2.0*data['lhf_diff']/halfrange
    #
    toa_sw_norm=data['toa_sw']/range_prof['toa_sw_max']
    toa_sw_diff_norm=2.0*data['toa_sw_diff']/range_prof['toa_sw_max']
    if setup['ppn_option']==0:
        ppn_norm=data['ppn']*0.0
    elif setup['ppn_option']==1:
        ppn_norm=(data['ppn']/(0.5*range_prof['ppn_max']))-1.0
    elif setup['ppn_option']==2:
        # Ensure ppn is on scale of 0 1 (even though using tanh output layer with range form -1 to 1).
        ppn_norm=data['ppn']/range_prof['ppn_max']
    # end if
    #
    # Do theta_now
    #
    if setup['flag_theta_now']==0:
        # Do nothing
        for k in np.arange(0,nz,1):
            theta_now_norm[:,k,0,:]=data['theta_now'][:,k,0,:]
    elif setup['flag_theta_now']==1:
        # Subtract mean at each level, and divide by std at each level
        for k in np.arange(0,nz,1):
            theta_now_norm[:,k,0,:]=(data['theta_now'][:,k,0,:]-mean_prof['theta_mean_prof'][k])/mean_prof['theta_std_prof'][k]
    elif setup['flag_theta_now']==2:
        # Subtract mean at each level, and divide by vertical average of std
        for k in np.arange(0,nz,1):
            theta_now_norm[:,k,0,:]=(data['theta_now'][:,k,0,:]-mean_prof['theta_mean_prof'][k])/theta_std
    elif setup['flag_theta_now']==10:
        for k in np.arange(0,nz,1):
            halfrange=np.amax([centile_prof['theta_975_prof'][k]-mean_prof['theta_mean_prof'][k],mean_prof['theta_mean_prof'][k]-centile_prof['theta_2p5_prof'][k]])
            theta_now_norm     [:,k,0,:]=(data['theta_now']     [:,k,0,:]-mean_prof['theta_mean_prof'][k])/halfrange
            theta_plus_adv_norm[:,k,0,:]=(data['theta_plus_adv'][:,k,0,:]-mean_prof['theta_mean_prof'][k])/halfrange
    else:
        print('You have not selected a valid option')
    #
    # Do theta_next
    #
    if setup['flag_theta_next']==0:
        # Do nothing
        for k in np.arange(0,nz,1):
            theta_next_norm[:,k,0,:]=data['theta_next'][:,k,0,:]
    elif setup['flag_theta_next']==1:
        # Subtract mean at each level, and divide by std at each level
        for k in np.arange(0,nz,1):
            theta_next_norm[:,k,0,:]=(data['theta_next'][:,k,0,:]-mean_prof['theta_mean_prof'][k])/mean_prof['theta_std_prof'][k]
    elif setup['flag_theta_next']==2:
        # Subtract mean at each level, and divide by vertical average of std
        for k in np.arange(0,nz,1):
            theta_next_norm[:,k,0,:]=(data['theta_next'][:,k,0,:]-mean_prof['theta_mean_prof'][k])/theta_std
    elif setup['flag_theta_next']==10:
        for k in np.arange(0,nz,1):
            halfrange=np.amax([centile_prof['theta_975_prof'][k]-mean_prof['theta_mean_prof'][k],mean_prof['theta_mean_prof'][k]-centile_prof['theta_2p5_prof'][k]])
            theta_next_norm[:,k,0,:]=(data['theta_next'][:,k,0,:]-mean_prof['theta_mean_prof'][k])/halfrange
    else:
        print('You have not selected a valid option')
    #
    # Do qt_now
    #
    if setup['flag_qt_now']==0:
        # Do nothing
        for k in np.arange(0,nz,1):
            qt_now_norm[:,k,0,:]=data['qt_now'][:,k,0,:]
    elif setup['flag_qt_now']==1:
        # Subtract mean at each level, and divide by std at each level
        for k in np.arange(0,nz,1):
            qt_now_norm[:,k,0,:]=(data['qt_now'][:,k,0,:]-mean_prof['qt_mean_prof'][k])/mean_prof['qt_std_prof'][k]
    elif setup['flag_qt_now']==2:
        # Subtract mean at each level, and divide by vertical average of std
        for k in np.arange(0,nz,1):
            qt_now_norm[:,k,0,:]=(data['qt_now'][:,k,0,:]-mean_prof['qt_mean_prof'][k])/qt_std
    elif setup['flag_qt_now']==10:
        for k in np.arange(0,nz,1):
            halfrange=np.amax([centile_prof['qt_975_prof'][k]-mean_prof['qt_mean_prof'][k],mean_prof['qt_mean_prof'][k]-centile_prof['qt_2p5_prof'][k]])
            qt_now_norm     [:,k,0,:]=(data['qt_now']     [:,k,0,:]-mean_prof['qt_mean_prof'][k])/halfrange
            qt_plus_adv_norm[:,k,0,:]=(data['qt_plus_adv'][:,k,0,:]-mean_prof['qt_mean_prof'][k])/halfrange
    else:
        print('You have not selected a valid option')
    #
    # Do qt_next
    #
    if setup['flag_qt_next']==0:
        # Subtract mean at each level, and divide by std at each level
        for k in np.arange(0,nz,1):
            qt_next_norm[:,k,0,:]=data['qt_next'][:,k,0,:]
    elif setup['flag_qt_next']==1:
        # Subtract mean at each level, and divide by std at each level
        for k in np.arange(0,nz,1):
            qt_next_norm[:,k,0,:]=(data['qt_next'][:,k,0,:]-mean_prof['qt_mean_prof'][k])/mean_prof['qt_std_prof'][k]
    elif setup['flag_qt_next']==2:
        # Subtract mean at each level, and divide by vertical average of std
        for k in np.arange(0,nz,1):
            qt_next_norm[:,k,0,:]=(data['qt_next'][:,k,0,:]-mean_prof['qt_mean_prof'][k])/qt_std
    elif setup['flag_qt_next']==10:
        for k in np.arange(0,nz,1):
            halfrange=np.amax([centile_prof['qt_975_prof'][k]-mean_prof['qt_mean_prof'][k],mean_prof['qt_mean_prof'][k]-centile_prof['qt_2p5_prof'][k]])
            qt_next_norm[:,k,0,:]=(data['qt_next'][:,k,0,:]-mean_prof['qt_mean_prof'][k])/halfrange
    else:
        print('You have not selected a valid option')
    #
    # Do theta incr (these should have a mean of zero on average anyway).
    #
    if setup['flag_delta_theta']==0:
        # Do nothing
        for k in np.arange(0,nz,1):
            theta_adv_norm[:,k,0,:]=data['theta_adv'][:,k,0,:]
            theta_phys_norm[:,k,0,:]=data['theta_phys'][:,k,0,:]
    elif setup['flag_delta_theta']==1:
        # Divide by std at each level
        for k in np.arange(0,nz,1):
            theta_adv_norm[:,k,0,:]=data['theta_adv'][:,k,0,:]/mean_prof['theta_adv_std_prof'][k]
            theta_phys_norm[:,k,0,:]=data['theta_phys'][:,k,0,:]/mean_prof['theta_phys_std_prof'][k]
    elif setup['flag_delta_theta']==2:
        # Divide by vertical average of std
        for k in np.arange(0,nz,1):
            theta_adv_norm[:,k,0,:]=data['theta_adv'][:,k,0,:]/theta_adv_std
            theta_phys_norm[:,k,0,:]=data['theta_phys'][:,k,0,:]/ theta_phys_std
    elif setup['flag_delta_theta']==10:
        for k in np.arange(0,nz,1):
            halfrange=np.amax([centile_prof['theta_adv_975_prof'][k],-centile_prof['theta_adv_2p5_prof'][k]])
            theta_adv_norm[:,k,0,:]=data['theta_adv'][:,k,0,:]/halfrange
            halfrange=np.amax([centile_prof['theta_phys_975_prof'][k],-centile_prof['theta_phys_2p5_prof'][k]])
            theta_phys_norm[:,k,0,:]=data['theta_phys'][:,k,0,:]/halfrange
    else:
        print('You have not selected a valid option')
    #
    # Do qt incr (these should have a mean of zero on average anyway).
    #
    if setup['flag_delta_qt']==0:
        # Do nothing
        for k in np.arange(0,nz,1):
            qt_adv_norm   [:,k,0,:]=   data['qt_adv'][:,k,0,:]
            qt_phys_norm   [:,k,0,:]=   data['qt_phys'][:,k,0,:]
    if setup['flag_delta_qt']==1:
        # Divide by std at each level
        for k in np.arange(0,nz,1):
            qt_adv_norm   [:,k,0,:]=   data['qt_adv'][:,k,0,:]/mean_prof['qt_adv_std_prof'][k]
            qt_phys_norm   [:,k,0,:]=   data['qt_phys'][:,k,0,:]/mean_prof['qt_phys_std_prof'][k]
    elif setup['flag_delta_qt']==2:
        # Divide by vertical average of std
        for k in np.arange(0,nz,1):
            qt_adv_norm   [:,k,0,:]=   data['qt_adv'][:,k,0,:]/qt_adv_std
            qt_phys_norm   [:,k,0,:]=   data['qt_phys'][:,k,0,:]/qt_phys_std
    elif setup['flag_delta_qt']==10:
        for k in np.arange(0,nz,1):
            halfrange=np.amax([centile_prof['qt_adv_975_prof'][k],-centile_prof['qt_adv_2p5_prof'][k]])
            qt_adv_norm[:,k,0,:]=data['qt_adv'][:,k,0,:]/halfrange
            halfrange=np.amax([centile_prof['qt_phys_975_prof'][k],-centile_prof['qt_phys_2p5_prof'][k]])
            qt_phys_norm[:,k,0,:]=data['qt_phys'][:,k,0,:]/halfrange
    else:
        print('You have not selected a valid option')
    #
    for k in np.arange(0,nz,1):
        halfrange=np.amax([centile_prof['qt_diff_975_prof'][k],-centile_prof['qt_diff_2p5_prof'][k]])
        qt_diff_norm[:,k,0,:]=data['qt_diff'][:,k,0,:]/halfrange
        halfrange=np.amax([centile_prof['theta_diff_975_prof'][k],-centile_prof['theta_diff_2p5_prof'][k]])
        theta_diff_norm[:,k,0,:]=data['theta_diff'][:,k,0,:]/halfrange
    #
    return {'theta_now_norm':theta_now_norm,'qt_now_norm':qt_now_norm,'theta_adv_norm':theta_adv_norm,'qt_adv_norm':qt_adv_norm,'theta_phys_norm':theta_phys_norm,'qt_phys_norm':qt_phys_norm,'shf_norm':shf_norm,'lhf_norm':lhf_norm,'toa_sw_norm':toa_sw_norm, 'theta_next_norm':theta_next_norm,'qt_next_norm':qt_next_norm, 'theta_next_norm':theta_next_norm, 'qt_next_norm':qt_next_norm, 'ppn_norm':ppn_norm, 'shf_diff_norm':shf_diff_norm, 'lhf_diff_norm':lhf_diff_norm, 'toa_sw_diff_norm':toa_sw_diff_norm, 'theta_plus_adv_norm':theta_plus_adv_norm, 'qt_plus_adv_norm':qt_plus_adv_norm, 'theta_diff_norm':theta_diff_norm, 'qt_diff_norm':qt_diff_norm};

def extricate(setup,norms,data,i_flag_add_adv):
    # Version used for learning thermodynamic profiles at next timestep (or their physics increments) from current thermodynamic profile,
    # some advection information and some surface and TOA fields (and their time derivatives).
    # This is where we take all the arrays of normalised data and package them together into a series of vectors each of which contains
    # all the required inputs at the start of the vector and all the outputs at the end. 
    # The final array then contain n samples each of which is one of these vectors.
    nt,nz,tmp3,nr=data['theta_now'].shape
    #
    # Have option of only doing calculation every k_step levels
    nz_to_use=nz/setup['k_step']
    tmp4=np.empty([int(6*nz_to_use)+11,nt*nr])
    k_indices=setup['k_indices']
    #
    # Construct massive vector for ML
    for r in np.arange(0,nr,1):
        for t in np.arange(0,nt,1):
            # Extract required (possibly sub-sampled) columns of data.
            theta_col          =norms['theta_now_norm'][t,k_indices,0,r]
            qt_col             =norms['qt_now_norm']   [t,k_indices,0,r]
            theta_adv_col      =norms['theta_adv_norm'][t,k_indices,0,r]
            qt_adv_col         =norms['qt_adv_norm']   [t,k_indices,0,r]
            theta_phys_col     =norms['theta_phys_norm'][t,k_indices,0,r]
            qt_phys_col        =norms['qt_phys_norm']   [t,k_indices,0,r]
            theta_next_col     =norms['theta_next_norm'][t,k_indices,0,r]
            qt_next_col        =norms['qt_next_norm']   [t,k_indices,0,r]
            theta_plus_adv_col =norms['theta_plus_adv_norm'][t,k_indices,0,r]
            qt_plus_adv_col    =norms['qt_plus_adv_norm']   [t,k_indices,0,r]
            theta_diff_col     =norms['theta_diff_norm'][t,k_indices,0,r]
            qt_diff_col        =norms['qt_diff_norm']   [t,k_indices,0,r]
            # Create a really long vector with all the inputs...
            #     1st and 2nd set of data are theta and q
            tmp4[int(0*nz_to_use):int(1*nz_to_use),int((r*nt)+t)]=theta_col
            tmp4[int(1*nz_to_use):int(2*nz_to_use),int((r*nt)+t)]=qt_col
            if i_flag_add_adv==1:
                # The 3rd and 4th set of data is (theta+dtheta_adv) and (qt+dqt_adv))
                # Use this option for *training* the model.
                tmp4[int(2*nz_to_use):int(3*nz_to_use),int((r*nt)+t)]=theta_plus_adv_col
                tmp4[int(3*nz_to_use):int(4*nz_to_use),int((r*nt)+t)]=qt_plus_adv_col
            else:
                # The 3rd and 4th set of data is (      dtheta_adv) and (   dqt_adv))
                # Use this option for *evaluating* the model recursively (as need to use *updated* theta and q).
                tmp4[int(2*nz_to_use):int(3*nz_to_use),int((r*nt)+t)]=theta_adv_col
                tmp4[int(3*nz_to_use):int(4*nz_to_use),int((r*nt)+t)]=qt_adv_col
            # end if
            tmp4[int(4*nz_to_use),int((r*nt)+t)]                     =norms['shf_norm'][t,0,0,r]
            tmp4[int(4*nz_to_use)+1,int((r*nt)+t)]                   =norms['lhf_norm'][t,0,0,r]
            tmp4[int(4*nz_to_use)+2,int((r*nt)+t)]                   =norms['toa_sw_norm'][t,0,0,r]
            if setup['include_flux_deriv']==1:
                tmp4[int(4*nz_to_use)+3,int((r*nt)+t)]               =norms['shf_diff_norm'][t,0,0,r]
                tmp4[int(4*nz_to_use)+4,int((r*nt)+t)]               =norms['lhf_diff_norm'][t,0,0,r]
                tmp4[int(4*nz_to_use)+5,int((r*nt)+t)]               =norms['toa_sw_diff_norm'][t,0,0,r]
            else:
                tmp4[int(4*nz_to_use)+3,int((r*nt)+t)]               =norms['shf_diff_norm'][t,0,0,r]*0.0
                tmp4[int(4*nz_to_use)+4,int((r*nt)+t)]               =norms['lhf_diff_norm'][t,0,0,r]*0.0
                tmp4[int(4*nz_to_use)+5,int((r*nt)+t)]               =norms['toa_sw_diff_norm'][t,0,0,r]*0.0
            if setup['include_cos_and_sin_terms']==1:
                tmp4[int(4*nz_to_use)+6,int((r*nt)+t)]               =data['cos_term_now'][t,r]
                tmp4[int(4*nz_to_use)+7,int((r*nt)+t)]               =data['sin_term_now'][t,r]
                tmp4[int(4*nz_to_use)+8,int((r*nt)+t)]               =data['cos_term_end'][t,r]
                tmp4[int(4*nz_to_use)+9,int((r*nt)+t)]               =data['sin_term_end'][t,r]
            else:
                tmp4[int(4*nz_to_use)+6,int((r*nt)+t)]               =data['cos_term_now'][t,r]*0.0
                tmp4[int(4*nz_to_use)+7,int((r*nt)+t)]               =data['sin_term_now'][t,r]*0.0
                tmp4[int(4*nz_to_use)+8,int((r*nt)+t)]               =data['cos_term_end'][t,r]*0.0
                tmp4[int(4*nz_to_use)+9,int((r*nt)+t)]               =data['sin_term_end'][t,r]*0.0
            # ...and the outputs we are trying to learn.
            if setup['what_to_predict']==0:
                # Either try to learn theta and q at next timestep...
                tmp4[int(4*nz_to_use)+10:int(5*nz_to_use)+10,int((r*nt)+t)]=theta_next_col
                tmp4[int(5*nz_to_use)+10:int(6*nz_to_use)+10,int((r*nt)+t)]=qt_next_col
            if setup['what_to_predict']==1:
                # ... or try to learn the physics increments.
                tmp4[int(4*nz_to_use)+6:int(5*nz_to_use)+6,int((r*nt)+t)]=theta_phys_col
                tmp4[int(5*nz_to_use)+6:int(6*nz_to_use)+6,int((r*nt)+t)]=qt_phys_col
            if setup['what_to_predict']==2:
                # ... or try to learn the total increments.
                tmp4[int(4*nz_to_use)+6:int(5*nz_to_use)+6,int((r*nt)+t)]=theta_diff_col
                tmp4[int(5*nz_to_use)+6:int(6*nz_to_use)+6,int((r*nt)+t)]=qt_diff_col
            # Try to learn precipitation.
            tmp4[int(6*nz_to_use)+10,int((r*nt)+t)]=norms['ppn_norm'][t,0,0,r]
    #
    return tmp4;

def extricate_bcf(setup,norms,data,i_flag_add_adv):
    # Version used for learning cloud fraction from theta and q.
    # This is where we take all the arrays of normalised data and package them together into a series of vectors each of which contains
    # all the required inputs at the start of the vector and all the outputs at the end. 
    # The final array then contain n samples each of which is one of these vectors.
    #
    nt,nz,tmp3,nr=data['theta_now'].shape
    #
    # Have option of only doing calculation every k_step levels
    k_step=2
    nz_to_use=nz/k_step
    tmp4=np.empty([int(3*nz_to_use),nt*nr])
    k_indices=setup['k_indices']
    #
    # Construct massive vector for ML
    for r in np.arange(0,nr,1):
        for t in np.arange(0,nt,1):
            # Extract required (possibly sub-sampled) columns of data.
            theta_col     =norms['theta_now_norm'][t,k_indices,0,r]
            qt_col        =norms['qt_now_norm']   [t,k_indices,0,r]
            bcf_col       =data ['bcf']           [t,k_indices,0,r]
            #     1st and 2nd set of data are theta and q
            tmp4[int(0*nz_to_use):int(1*nz_to_use),int((r*nt)+t)]=theta_col
            tmp4[int(1*nz_to_use):int(2*nz_to_use),int((r*nt)+t)]=qt_col
            # Try to learn bulk cloud fraction
            tmp4[int(2*nz_to_use):int(3*nz_to_use),int((r*nt)+t)]=bcf_col
    #
    return tmp4;

def write_out_profiles(range_prof,mean_prof,centiles_prof,domain,date_range):
    # Used as part of preprocessing
    fileout='ml_profiles/'+date_range+'.theta_mean_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['theta_mean_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_std_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['theta_std_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_adv_std_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['theta_adv_std_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_phys_std_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['theta_phys_std_prof'], fmt='%10.7f')
    #
    fileout='ml_profiles/'+date_range+'.qt_mean_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['qt_mean_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_std_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['qt_std_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_adv_std_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['qt_adv_std_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_phys_std_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, mean_prof['qt_phys_std_prof'], fmt='%10.7f')
    #
    fileout='ml_profiles/'+date_range+'.theta_min_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['theta_min_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_max_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['theta_max_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_adv_maxabs_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['theta_adv_maxabs_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_phys_maxabs_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['theta_phys_maxabs_prof'], fmt='%10.7f')
    #
    fileout='ml_profiles/'+date_range+'.qt_min_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['qt_min_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_max_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['qt_max_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_adv_maxabs_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['qt_adv_maxabs_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_phys_maxabs_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, range_prof['qt_phys_maxabs_prof'], fmt='%10.7f')
    #
    fileout='ml_profiles/'+date_range+'.shf_std.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*mean_prof['shf_std'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.lhf_std.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*mean_prof['lhf_std'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.ppn_max.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*range_prof['ppn_max'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.toa_sw_max.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*range_prof['toa_sw_max'], fmt='%10.3f')
    #
    fileout='ml_profiles/'+date_range+'.theta_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['theta_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['theta_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['qt_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['qt_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_adv_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['theta_adv_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_adv_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['theta_adv_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_adv_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['qt_adv_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_adv_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['qt_adv_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_phys_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['theta_phys_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_phys_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['theta_phys_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_phys_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['qt_phys_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_phys_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, centiles_prof['qt_phys_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.shf_max.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*range_prof['shf_max'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.lhf_max.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*range_prof['lhf_max'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.shf_2p5.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['shf_2p5'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.shf_975.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['shf_975'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.lhf_2p5.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['lhf_2p5'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.lhf_975.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['lhf_975'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_diff_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['theta_diff_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.theta_diff_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['theta_diff_975_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_diff_2p5_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['qt_diff_2p5_prof'], fmt='%10.7f')
    fileout='ml_profiles/'+date_range+'.qt_diff_975_prof.domain'+str(domain)+'.txt'
    np.savetxt(fileout, np.ones((1,1))*centiles_prof['qt_diff_975_prof'], fmt='%10.7f')
    outcome=1
    return outcome;

def read_in_mean_profiles(date_range):
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_mean_prof.avg_over_domains.txt'
    theta_mean_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_std_prof.avg_over_domains.txt'
    theta_std_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_adv_std_prof.avg_over_domains.txt'
    theta_adv_std_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_phys_std_prof.avg_over_domains.txt'
    theta_phys_std_prof=np.loadtxt(filein)
    #
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_mean_prof.avg_over_domains.txt'
    qt_mean_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_std_prof.avg_over_domains.txt'
    qt_std_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_adv_std_prof.avg_over_domains.txt'
    qt_adv_std_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_phys_std_prof.avg_over_domains.txt'
    qt_phys_std_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.shf_std.avg_over_domains.txt'
    shf_std=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.lhf_std.avg_over_domains.txt'
    lhf_std=np.loadtxt(filein)
    return {'theta_mean_prof':theta_mean_prof, 'qt_mean_prof':qt_mean_prof, 'theta_std_prof':theta_std_prof, 'qt_std_prof':qt_std_prof, 'theta_adv_std_prof':theta_adv_std_prof, 'qt_adv_std_prof':qt_adv_std_prof, 'theta_phys_std_prof':theta_phys_std_prof, 'qt_phys_std_prof':qt_phys_std_prof, 'shf_std':shf_std, 'lhf_std':lhf_std};

def read_in_range_profiles(date_range):
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_min_prof.avg_over_domains.txt'
    theta_min_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_max_prof.avg_over_domains.txt'
    theta_max_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_adv_maxabs_prof.avg_over_domains.txt'
    theta_adv_maxabs_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_phys_maxabs_prof.avg_over_domains.txt'
    theta_phys_maxabs_prof=np.loadtxt(filein)
    #
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_min_prof.avg_over_domains.txt'
    qt_min_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_max_prof.avg_over_domains.txt'
    qt_max_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_adv_maxabs_prof.avg_over_domains.txt'
    qt_adv_maxabs_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_phys_maxabs_prof.avg_over_domains.txt'
    qt_phys_maxabs_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.ppn_max.avg_over_domains.txt'
    ppn_max=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.toa_sw_max.avg_over_domains.txt'
    toa_sw_max=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.shf_max.avg_over_domains.txt'
    shf_max=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.lhf_max.avg_over_domains.txt'
    lhf_max=np.loadtxt(filein)
    return {'theta_min_prof':theta_min_prof, 'theta_max_prof':theta_max_prof, 'qt_min_prof':qt_min_prof, 'qt_max_prof':qt_max_prof, 'theta_adv_maxabs_prof':theta_adv_maxabs_prof, 'qt_adv_maxabs_prof':qt_adv_maxabs_prof, 'theta_phys_maxabs_prof':theta_phys_maxabs_prof, 'qt_phys_maxabs_prof':qt_phys_maxabs_prof, 'ppn_max':ppn_max, 'toa_sw_max':toa_sw_max, 'shf_max':shf_max, 'lhf_max':lhf_max};

def read_in_centile_profiles(date_range):
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_2p5_prof.avg_over_domains.txt'
    theta_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_975_prof.avg_over_domains.txt'
    theta_975_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_adv_2p5_prof.avg_over_domains.txt'
    theta_adv_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_adv_975_prof.avg_over_domains.txt'
    theta_adv_975_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_phys_2p5_prof.avg_over_domains.txt'
    theta_phys_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_phys_975_prof.avg_over_domains.txt'
    theta_phys_975_prof=np.loadtxt(filein)
    #
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_2p5_prof.avg_over_domains.txt'
    qt_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_975_prof.avg_over_domains.txt'
    qt_975_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_adv_2p5_prof.avg_over_domains.txt'
    qt_adv_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_adv_975_prof.avg_over_domains.txt'
    qt_adv_975_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_phys_2p5_prof.avg_over_domains.txt'
    qt_phys_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_phys_975_prof.avg_over_domains.txt'
    qt_phys_975_prof=np.loadtxt(filein)

    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.shf_2p5.avg_over_domains.txt'
    shf_2p5=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.shf_975.avg_over_domains.txt'
    shf_975=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.lhf_2p5.avg_over_domains.txt'
    lhf_2p5=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.lhf_975.avg_over_domains.txt'
    lhf_975=np.loadtxt(filein)

    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_diff_2p5_prof.avg_over_domains.txt'
    theta_diff_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_diff_975_prof.avg_over_domains.txt'
    theta_diff_975_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_diff_2p5_prof.avg_over_domains.txt'
    qt_diff_2p5_prof=np.loadtxt(filein)
    filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_diff_975_prof.avg_over_domains.txt'
    qt_diff_975_prof=np.loadtxt(filein)
    return {'theta_2p5_prof':theta_2p5_prof,'theta_975_prof':theta_975_prof,'theta_adv_2p5_prof':theta_adv_2p5_prof,'theta_adv_975_prof':theta_adv_975_prof,'theta_phys_2p5_prof':theta_phys_2p5_prof,'theta_phys_975_prof':theta_phys_975_prof, 'qt_2p5_prof':qt_2p5_prof,'qt_975_prof':qt_975_prof, 'qt_adv_2p5_prof':qt_adv_2p5_prof, 'qt_adv_975_prof':qt_adv_975_prof, 'qt_phys_2p5_prof':qt_phys_2p5_prof,'qt_phys_975_prof':qt_phys_975_prof, 'shf_2p5':shf_2p5, 'shf_975':shf_975, 'lhf_2p5':lhf_2p5, 'lhf_975':lhf_975, 'theta_diff_2p5_prof':theta_diff_2p5_prof, 'theta_diff_975_prof':theta_diff_975_prof, 'qt_diff_2p5_prof':qt_diff_2p5_prof, 'qt_diff_975_prof':qt_diff_975_prof};

def plot_centiles(datain,setup):
    # For plotting profiles to check how the normalisations have behaved.
    lev=setup['k_indices']
    fig, axs = plt.subplots(4, 2)
    axs[0, 0].plot(np.amax(datain[:,0:25],axis=0),lev)
    axs[0, 0].plot(np.amin(datain[:,0:25],axis=0),lev)
    axs[0, 0].plot(np.percentile(datain[:,0:25],2.5,axis=0),lev)
    axs[0, 0].plot(np.percentile(datain[:,0:25],50,axis=0),lev)
    axs[0, 0].plot(np.percentile(datain[:,0:25],97.5,axis=0),lev)
    axs[0, 0].plot(np.mean(datain[:,0:25],axis=0),lev)
    #
    axs[0, 1].plot(np.amax(datain[:,25:50],axis=0),lev)
    axs[0, 1].plot(np.amin(datain[:,25:50],axis=0),lev)
    axs[0, 1].plot(np.percentile(datain[:,25:50],2.5,axis=0),lev)
    axs[0, 1].plot(np.percentile(datain[:,25:50],50,axis=0),lev)
    axs[0, 1].plot(np.percentile(datain[:,25:50],97.5,axis=0),lev)
    axs[0, 1].plot(np.mean(datain[:,25:50],axis=0),lev)
    #
    axs[1, 0].plot(np.amax(datain[:,50:75],axis=0),lev)
    axs[1, 0].plot(np.amin(datain[:,50:75],axis=0),lev)
    axs[1, 0].plot(np.percentile(datain[:,50:75],2.5,axis=0),lev)
    axs[1, 0].plot(np.percentile(datain[:,50:75],50,axis=0),lev)
    axs[1, 0].plot(np.percentile(datain[:,50:75],97.5,axis=0),lev)
    axs[1, 0].plot(np.mean(datain[:,50:75],axis=0),lev)
    #
    axs[1, 1].plot(np.amax(datain[:,75:100],axis=0),lev)
    axs[1, 1].plot(np.amin(datain[:,75:100],axis=0),lev)
    axs[1, 1].plot(np.percentile(datain[:,75:100],2.5,axis=0),lev)
    axs[1, 1].plot(np.percentile(datain[:,75:100],50,axis=0),lev)
    axs[1, 1].plot(np.percentile(datain[:,75:100],97.5,axis=0),lev)
    axs[1, 1].plot(np.mean(datain[:,75:100],axis=0),lev)
    #
    axs[2, 0].plot(np.amax(datain[:,108:133],axis=0),lev)
    axs[2, 0].plot(np.amin(datain[:,108:133],axis=0),lev)
    axs[2, 0].plot(np.percentile(datain[:,108:133],2.5,axis=0),lev)
    axs[2, 0].plot(np.percentile(datain[:,108:133],50,axis=0),lev)
    axs[2, 0].plot(np.percentile(datain[:,108:133],97.5,axis=0),lev)
    axs[2, 0].plot(np.mean(datain[:,108:133],axis=0),lev)
    #
    axs[2, 1].plot(np.amax(datain[:,133:158],axis=0),lev)
    axs[2, 1].plot(np.amin(datain[:,133:158],axis=0),lev)
    axs[2, 1].plot(np.percentile(datain[:,133:158],2.5,axis=0),lev)
    axs[2, 1].plot(np.percentile(datain[:,133:158],50,axis=0),lev)
    axs[2, 1].plot(np.percentile(datain[:,133:158],97.5,axis=0),lev)
    axs[2, 1].plot(np.mean(datain[:,133:158],axis=0),lev)
    indices=[100,101,102,103,104,105,106,107,156]
    axs[3, 0].plot(np.amax(datain[:,indices],axis=0),indices)
    axs[3, 0].plot(np.amin(datain[:,indices],axis=0),indices)
    axs[3, 0].plot(np.percentile(datain[:,indices],2.5,axis=0),indices)
    axs[3, 0].plot(np.percentile(datain[:,indices],50,axis=0),indices)
    axs[3, 0].plot(np.percentile(datain[:,indices],97.5,axis=0),indices)
    axs[3, 0].plot(np.mean(datain[:,indices],axis=0),indices)
    plt.show()
    outcome=1
    return outcome;

def plot_in_and_out(tmp4,tmp10):
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    im=ax0.pcolormesh(tmp4[np.arange(0,3000,10),:])
    fig.colorbar(im, ax=ax0)
    im=ax1.pcolormesh(tmp10[np.arange(0,3000,10),:])
    fig.colorbar(im, ax=ax1)
    plt.show()
    #
    fig, (ax0, ax1) = plt.subplots(ncols=2)
    im=ax0.pcolormesh(X[0:1000,:])
    fig.colorbar(im, ax=ax0)
    im=ax1.pcolormesh(y[0:1000,:])
    fig.colorbar(im, ax=ax1)
    plt.show()
    outcome=1
    return outcome;

def process_series_of_domains(date_range,setup,regions_to_process,domains_to_do,i_flag_add_adv,monthyear,ndays):
    for d in np.arange(0,len(domains_to_do)):
        domain       = domains_to_do[d]
        data         = read_in_for_one_subdomain(domain,date_range,regions_to_process,setup,monthyear,ndays)
        # N.B. we are intentionally normalising on the profiles only calculated for one month. 
        # i.e. we don't want different norm in each month!
        mean_prof    = read_in_mean_profiles(   '20151231-20160131')
        range_prof   = read_in_range_profiles(  '20151231-20160131')
        centile_prof = read_in_centile_profiles('20151231-20160131')
        norms        = normalise(setup,data,range_prof,mean_prof,centile_prof)
        output_1dom  = extricate(setup,norms,data,i_flag_add_adv)
        if d==0:
            output_alldom=output_1dom
        else:
            output_alldom=np.append(output_alldom,output_1dom,axis=1)
        # endif
    # end d
    print('Size of output_alldom=',output_alldom.shape)
    # === Set anything outside of +/- clip_limit to +/- clip_limit ===
    output_alldom=np.clip(output_alldom,-setup['clip_limit'],setup['clip_limit'])
    # ====================================================
    output_alldom=np.transpose(output_alldom)
    print('Domain ',str(domains_to_do),' done')
    return {'output_alldom':output_alldom, 'norms':norms};

def process_series_of_domains_for_cloud_fraction(date_range,setup,regions_to_process,domains_to_do,i_flag_add_adv,monthyear,ndays):
    for d in np.arange(0,len(domains_to_do)):
        domain       = domains_to_do[d]
        data         = read_in_for_one_subdomain(domain,date_range,regions_to_process,setup,monthyear,ndays)
        # N.B. we are intentionally normalising on the profiles only calculated for one month. i.e. we don't want different norm in each month!
        mean_prof    = read_in_mean_profiles('20151231-20160131')
        range_prof   = read_in_range_profiles('20151231-20160131')
        centile_prof = read_in_centile_profiles('20151231-20160131')
        norms        = normalise(setup,data,range_prof,mean_prof,centile_prof)
        output_1dom  = extricate_bcf(setup,norms,data,i_flag_add_adv)
        if d==0:
            output_alldom=output_1dom
        else:
            output_alldom=np.append(output_alldom,output_1dom,axis=1)
        # endif
    # end d
    print('Size of output_alldom=',output_alldom.shape)
    # === Set anything outside of +/- clip_limit to +/- clip_limit ===
    output_alldom=np.clip(output_alldom,-setup['clip_limit'],setup['clip_limit'])
    # ====================================================
    output_alldom=np.transpose(output_alldom)
    print('Domain ',str(domains_to_do),' done')
    return {'output_alldom':output_alldom, 'norms':norms};


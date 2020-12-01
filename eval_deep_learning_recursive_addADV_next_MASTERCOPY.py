#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
#
# Code for recursive single column model evaluation
#
# Use "module load scitools/experimental-current" at the command line before running this.

# Import some modules
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from numpy import loadtxt

from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
#from keras.models import load_weights
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam

from keras.metrics import RootMeanSquaredError

from scipy import stats

from ml_namelist_MASTERCOPY import setup_parameters
from ml_cjm_functions_MASTERCOPY import process_series_of_domains
from ml_cjm_functions_MASTERCOPY import read_in_range_profiles
from ml_cjm_functions_MASTERCOPY import read_in_centile_profiles
from ml_cjm_functions_MASTERCOPY import read_in_mean_profiles
from matplotlib.ticker import MultipleLocator

# Start of functions

def unnormalise_adv_incr(processed):
    date_range='20151231-20160131'
    centile_prof   = read_in_centile_profiles(date_range)
    adv_theta_norm = processed['output_alldom'][:,50:75]
    adv_qt_norm    = processed['output_alldom'][:,75:100]
    adv_theta_raw  = adv_theta_norm*np.nan
    adv_qt_raw     = adv_theta_norm*np.nan
    nz=50
    k_step=2
    k_indices=np.arange(0,nz,k_step)
    for k in np.arange(0,len(k_indices),1):
        halfrange=np.amax([centile_prof['theta_adv_975_prof'][k_indices[k]],-centile_prof['theta_adv_2p5_prof'][k_indices[k]]])
        adv_theta_raw[:,k]=adv_theta_norm[:,k]*halfrange
        halfrange=np.amax([centile_prof['qt_adv_975_prof'][k_indices[k]],-centile_prof['qt_adv_2p5_prof'][k_indices[k]]])
        adv_qt_raw[:,k]=adv_qt_norm[:,k]*halfrange
    return {'adv_theta_raw':adv_theta_raw, 'adv_qt_raw':adv_qt_raw};

def unnormalise_theta_and_q(theta_norm,qt_norm,setup):
    date_range='20151231-20160131'
    centile_prof = read_in_centile_profiles(date_range)
    mean_prof    = read_in_mean_profiles(date_range)
    theta_raw    = theta_norm*np.nan
    qt_raw       = theta_norm*np.nan
    k_indices=setup['k_indices']
    for k in np.arange(0,len(k_indices),1):
        halfrange=np.amax([centile_prof['theta_975_prof'][k_indices[k]]-mean_prof['theta_mean_prof'][k_indices[k]],mean_prof['theta_mean_prof'][k_indices[k]]-centile_prof['theta_2p5_prof'][k_indices[k]]])
        theta_raw[k]=(theta_norm[k]*halfrange)+mean_prof['theta_mean_prof'][k_indices[k]]
        halfrange=np.amax([centile_prof['qt_975_prof'][k_indices[k]]-mean_prof['qt_mean_prof'][k_indices[k]],mean_prof['qt_mean_prof'][k_indices[k]]-centile_prof['qt_2p5_prof'][k_indices[k]]])
        qt_raw[k]=(qt_norm[k]*halfrange)+mean_prof['qt_mean_prof'][k_indices[k]]
    return {'theta_raw':theta_raw, 'qt_raw':qt_raw};

def normalise_field_plus_adv_incr(theta_plus_adv_raw,qt_plus_adv_raw,setup):
    date_range='20151231-20160131'
    centile_prof        = read_in_centile_profiles(date_range)
    mean_prof           = read_in_mean_profiles(date_range)
    theta_plus_adv_norm = theta_plus_adv_raw*np.nan
    qt_plus_adv_norm    =    qt_plus_adv_raw*np.nan
    k_indices=setup['k_indices']
    for k in np.arange(0,len(k_indices),1):
        halfrange=np.amax([centile_prof['theta_975_prof'][k_indices[k]]-mean_prof['theta_mean_prof'][k_indices[k]],mean_prof['theta_mean_prof'][k_indices[k]]-centile_prof['theta_2p5_prof'][k_indices[k]]])
        theta_plus_adv_norm[k]=(theta_plus_adv_raw[k]-mean_prof['theta_mean_prof'][k_indices[k]])/halfrange
        #
        halfrange=np.amax([centile_prof['qt_975_prof'][k_indices[k]]-mean_prof['qt_mean_prof'][k_indices[k]],mean_prof['qt_mean_prof'][k_indices[k]]-centile_prof['qt_2p5_prof'][k_indices[k]]])
        qt_plus_adv_norm[k]=(qt_plus_adv_raw[k]-mean_prof['qt_mean_prof'][k_indices[k]])/halfrange
    return {'theta_plus_adv_norm':theta_plus_adv_norm, 'qt_plus_adv_norm':qt_plus_adv_norm};

def recursive_model(processed,starttimestep,adv_incr_raw,ndays,setup,day1_ctl_theta_gm,day1_ctl_qt_gm,day2_ctl_theta_gm,day2_ctl_qt_gm,my_counter,best_so_far):
    id_2h ='expt154'
    id_4h ='expt155'
    id_6h ='expt156'
    id_8h ='expt157'
    id_10h='expt158'
    id_12h='expt159'
    i_option=0
    time_window='expt154_155_156abcdef'
    #
    json_file=open(id_2h+'/expt_a/ml_model.json', 'r')
    loaded_model_json=json_file.read()
    json_file.close()
    model2a=model_from_json(loaded_model_json)
    model2a.load_weights(id_2h+'/expt_a/ml_lastest_saved_weights.h5')
    model2b=model_from_json(loaded_model_json)
    model2b.load_weights(id_2h+'/expt_b/ml_lastest_saved_weights.h5')
    model2c=model_from_json(loaded_model_json)
    model2c.load_weights(id_2h+'/expt_c/ml_lastest_saved_weights.h5')
    model2d=model_from_json(loaded_model_json)
    model2d.load_weights(id_2h+'/expt_d/ml_lastest_saved_weights.h5')
    model2e=model_from_json(loaded_model_json)
    model2e.load_weights(id_2h+'/expt_e/ml_lastest_saved_weights.h5')
    model2f=model_from_json(loaded_model_json)
    model2f.load_weights(id_2h+'/expt_f/ml_lastest_saved_weights.h5')
    #
    model4a= model_from_json(loaded_model_json)
    model4a.load_weights(id_4h+'/expt_a/ml_lastest_saved_weights.h5')
    model4b= model_from_json(loaded_model_json)
    model4b.load_weights(id_4h+'/expt_b/ml_lastest_saved_weights.h5')
    model4c= model_from_json(loaded_model_json)
    model4c.load_weights(id_4h+'/expt_c/ml_lastest_saved_weights.h5')
    model4d= model_from_json(loaded_model_json)
    model4d.load_weights(id_4h+'/expt_d/ml_lastest_saved_weights.h5')
    model4e= model_from_json(loaded_model_json)
    model4e.load_weights(id_4h+'/expt_e/ml_lastest_saved_weights.h5')
    model4f= model_from_json(loaded_model_json)
    model4f.load_weights(id_4h+'/expt_f/ml_lastest_saved_weights.h5')
    #
    model6a= model_from_json(loaded_model_json)
    model6a.load_weights(id_6h+'/expt_a/ml_lastest_saved_weights.h5')
    model6b= model_from_json(loaded_model_json)
    model6b.load_weights(id_6h+'/expt_b/ml_lastest_saved_weights.h5')
    model6c= model_from_json(loaded_model_json)
    model6c.load_weights(id_6h+'/expt_c/ml_lastest_saved_weights.h5')
    model6d= model_from_json(loaded_model_json)
    model6d.load_weights(id_6h+'/expt_d/ml_lastest_saved_weights.h5')
    model6e= model_from_json(loaded_model_json)
    model6e.load_weights(id_6h+'/expt_e/ml_lastest_saved_weights.h5')
    model6f= model_from_json(loaded_model_json)
    model6f.load_weights(id_6h+'/expt_f/ml_lastest_saved_weights.h5')
    #
    model8a= model_from_json(loaded_model_json)
    model8a.load_weights(id_8h+'/expt_a/ml_lastest_saved_weights.h5')
    model8b= model_from_json(loaded_model_json)
    model8b.load_weights(id_8h+'/expt_b/ml_lastest_saved_weights.h5')
    model8c= model_from_json(loaded_model_json)
    model8c.load_weights(id_8h+'/expt_c/ml_lastest_saved_weights.h5')
    model8d= model_from_json(loaded_model_json)
    model8d.load_weights(id_8h+'/expt_d/ml_lastest_saved_weights.h5')
    model8e= model_from_json(loaded_model_json)
    model8e.load_weights(id_8h+'/expt_e/ml_lastest_saved_weights.h5')
    model8f= model_from_json(loaded_model_json)
    model8f.load_weights(id_8h+'/expt_f/ml_lastest_saved_weights.h5')
    #
    model10a = model_from_json(loaded_model_json)
    model10a.load_weights(id_10h+'/expt_a/ml_lastest_saved_weights.h5')
    model10b = model_from_json(loaded_model_json)
    model10b.load_weights(id_10h+'/expt_b/ml_lastest_saved_weights.h5')
    model10c = model_from_json(loaded_model_json)
    model10c.load_weights(id_10h+'/expt_c/ml_lastest_saved_weights.h5')
    model10d = model_from_json(loaded_model_json)
    model10d.load_weights(id_10h+'/expt_d/ml_lastest_saved_weights.h5')
    model10e = model_from_json(loaded_model_json)
    model10e.load_weights(id_10h+'/expt_e/ml_lastest_saved_weights.h5')
    model10f = model_from_json(loaded_model_json)
    model10f.load_weights(id_10h+'/expt_f/ml_lastest_saved_weights.h5')
    #
    model12a = model_from_json(loaded_model_json)
    model12a.load_weights(id_12h+'/expt_a/ml_lastest_saved_weights.h5')
    model12b = model_from_json(loaded_model_json)
    model12b.load_weights(id_12h+'/expt_b/ml_lastest_saved_weights.h5')
    model12c = model_from_json(loaded_model_json)
    model12c.load_weights(id_12h+'/expt_c/ml_lastest_saved_weights.h5')
    model12d = model_from_json(loaded_model_json)
    model12d.load_weights(id_12h+'/expt_d/ml_lastest_saved_weights.h5')
    model12e = model_from_json(loaded_model_json)
    model12e.load_weights(id_12h+'/expt_e/ml_lastest_saved_weights.h5')
    model12f = model_from_json(loaded_model_json)
    model12f.load_weights(id_12h+'/expt_f/ml_lastest_saved_weights.h5')
    #
    opt = Adam(learning_rate=1.0e-6)
    #
    model2a.compile(optimizer=opt, loss='mse')
    model2b.compile(optimizer=opt, loss='mse')
    model2c.compile(optimizer=opt, loss='mse')
    model2d.compile(optimizer=opt, loss='mse')
    model2e.compile(optimizer=opt, loss='mse')
    model2f.compile(optimizer=opt, loss='mse')
    #
    model4a.compile(optimizer=opt, loss='mse')
    model4b.compile(optimizer=opt, loss='mse')
    model4c.compile(optimizer=opt, loss='mse')
    model4d.compile(optimizer=opt, loss='mse')
    model4e.compile(optimizer=opt, loss='mse')
    model4f.compile(optimizer=opt, loss='mse')
    #
    model6a.compile(optimizer=opt, loss='mse')
    model6b.compile(optimizer=opt, loss='mse')
    model6c.compile(optimizer=opt, loss='mse')
    model6d.compile(optimizer=opt, loss='mse')
    model6e.compile(optimizer=opt, loss='mse')
    model6f.compile(optimizer=opt, loss='mse')
    #
    model8a.compile(optimizer=opt, loss='mse')
    model8b.compile(optimizer=opt, loss='mse')
    model8c.compile(optimizer=opt, loss='mse')
    model8d.compile(optimizer=opt, loss='mse')
    model8e.compile(optimizer=opt, loss='mse')
    model8f.compile(optimizer=opt, loss='mse')
    #
    model10a.compile(optimizer=opt, loss='mse')
    model10b.compile(optimizer=opt, loss='mse')
    model10c.compile(optimizer=opt, loss='mse')
    model10d.compile(optimizer=opt, loss='mse')
    model10e.compile(optimizer=opt, loss='mse')
    model10f.compile(optimizer=opt, loss='mse')
    #
    model12a.compile(optimizer=opt, loss='mse')
    model12b.compile(optimizer=opt, loss='mse')
    model12c.compile(optimizer=opt, loss='mse')
    model12d.compile(optimizer=opt, loss='mse')
    model12e.compile(optimizer=opt, loss='mse')
    model12f.compile(optimizer=opt, loss='mse')
    #
    # Option to read in 3 different ML, but not really tested yet.
    # Currently pointing to same file, put in future could point to 3 different files, learnt using different time windows.
    big_data=processed['output_alldom']
    # The first 110 elements are the inputs, the next 51 are what we are trying to predict (last output is precip).
    inputs_now  =big_data[:,0:110]
    truth_next  =big_data[:,110:161]
    #
    adv_theta_raw=adv_incr_raw['adv_theta_raw']
    adv_qt_raw=adv_incr_raw['adv_qt_raw']
    #
    tsperday=12
    #    ndays=10
    nt_here=(ndays*tsperday)+1
    #
    recursive_theta_norm=np.zeros((nt_here,25))+np.nan
    recursive_qt_norm   =np.zeros((nt_here,25))+np.nan
    recursive_ppn_norm  =np.zeros((nt_here,1))+np.nan
    recursive_ppn_norm1 =np.zeros((nt_here,1))+np.nan
    recursive_ppn_norm2 =np.zeros((nt_here,1))+np.nan
    recursive_ppn_norm3 =np.zeros((nt_here,1))+np.nan
    recursive_ppn_norm4 =np.zeros((nt_here,1))+np.nan
    recursive_ppn_norm5 =np.zeros((nt_here,1))+np.nan
    recursive_ppn_norm6 =np.zeros((nt_here,1))+np.nan
    #
    theta_norm  =inputs_now[starttimestep,0:25]
    qt_norm     =inputs_now[starttimestep,25:50]
    nz   = theta_norm.shape[0]
    timestep=0
    recursive_theta_norm[timestep,:]=theta_norm
    recursive_qt_norm   [timestep,:]=qt_norm
    recursive_ppn_norm  [timestep,0]=np.nan
    truth_theta_norm=inputs_now[starttimestep:starttimestep+nt_here,0:25]
    truth_qt_norm   =inputs_now[starttimestep:starttimestep+nt_here,25:50]
    truth_ppn_norm  =np.append(np.nan,truth_next[starttimestep:starttimestep+nt_here,50])
    #
    # Initialise arrays storing future increments that will be used 
    # on subsequent timestep (when they are present increments).
    np1=[0,0]
    np2=[0]
    wp1=[0,0]
    wp2=[0]
    #
    #    np1=[0,0,0,0,0]
    #    np2=[0,0,0,0]
    #    np3=[0,0,0]
    #    np4=[0,0]
    #    np5=[0]
    #    wp1=[0,0,0,0,0]
    #    wp2=[0,0,0,0]
    #    wp3=[0,0,0]
    #    wp4=[0,0]
    #    wp5=[0]
    #
    ppn1=[0,0]
    ppn2=[0]
    ########
    for timestep in np.arange(1,nt_here,1):
        #
        raw=unnormalise_theta_and_q(theta_norm,qt_norm,setup)
        theta_plus_adv_raw=raw['theta_raw']+adv_theta_raw[starttimestep+timestep-1,:]
        qt_plus_adv_raw=raw['qt_raw']+adv_qt_raw[starttimestep+timestep-1,:]
        norm=normalise_field_plus_adv_incr(theta_plus_adv_raw,qt_plus_adv_raw,setup)
        #
        if 1==2:
            lev=np.arange(0,50,2)
            fig, axs = plt.subplots(2, 2)
            im=axs[0,0].plot(raw['theta_raw'],lev)
            im=axs[0,0].plot(theta_plus_adv_raw,lev)
            im=axs[0,1].plot(raw['qt_raw'],lev)
            im=axs[0,1].plot(qt_plus_adv_raw,lev)
            im=axs[1,0].plot(theta_norm,lev)
            im=axs[1,0].plot(norm['theta_plus_adv_norm'],lev)
            im=axs[1,1].plot(qt_norm,lev)
            im=axs[1,1].plot(norm['qt_plus_adv_norm'],lev)
            plt.show()
        # end if
        #
        input_vector=np.zeros((2,110))+np.nan
        input_vector[:,  0: 25]=theta_norm
        input_vector[:, 25: 50]=qt_norm
        input_vector[:, 50: 75]=norm['theta_plus_adv_norm']
        input_vector[:, 75:100]=norm['qt_plus_adv_norm']
        input_vector[:,100:110]=inputs_now[starttimestep+timestep-1,100:110]
        #
        output_vector2a = model2a.predict(input_vector)
        output_vector2b = model2b.predict(input_vector)
        output_vector2c = model2c.predict(input_vector)
        output_vector2d = model2d.predict(input_vector)
        output_vector2e = model2e.predict(input_vector)
        output_vector2f = model2f.predict(input_vector)
        #
        output_vector4a = model4a.predict(input_vector)
        output_vector4b = model4b.predict(input_vector)
        output_vector4c = model4c.predict(input_vector)
        output_vector4d = model4d.predict(input_vector)
        output_vector4e = model4e.predict(input_vector)
        output_vector4f = model4f.predict(input_vector)
        #
        output_vector6a = model6a.predict(input_vector)
        output_vector6b = model6b.predict(input_vector)
        output_vector6c = model6c.predict(input_vector)
        output_vector6d = model6d.predict(input_vector)
        output_vector6e = model6e.predict(input_vector)
        output_vector6f = model6f.predict(input_vector)
        #
        output_vector8a = model8a.predict(input_vector)
        output_vector8b = model8b.predict(input_vector)
        output_vector8c = model8c.predict(input_vector)
        output_vector8d = model8d.predict(input_vector)
        output_vector8e = model8e.predict(input_vector)
        output_vector8f = model8f.predict(input_vector)
        #
        output_vector10a = model10a.predict(input_vector)
        output_vector10b = model10b.predict(input_vector)
        output_vector10c = model10c.predict(input_vector)
        output_vector10d = model10d.predict(input_vector)
        output_vector10e = model10e.predict(input_vector)
        output_vector10f = model10e.predict(input_vector)
        #
        output_vector12a = model12a.predict(input_vector)
        output_vector12b = model12b.predict(input_vector)
        output_vector12c = model12c.predict(input_vector)
        output_vector12d = model12d.predict(input_vector)
        output_vector12e = model12e.predict(input_vector)
        output_vector12f = model12e.predict(input_vector)
        #
        diff2a =(output_vector2a[:,0:50]-input_vector[:,0:50])
        diff2b =(output_vector2b[:,0:50]-input_vector[:,0:50])
        diff2c =(output_vector2c[:,0:50]-input_vector[:,0:50])
        diff2d =(output_vector2d[:,0:50]-input_vector[:,0:50])
        diff2e =(output_vector2e[:,0:50]-input_vector[:,0:50])
        diff2f =(output_vector2f[:,0:50]-input_vector[:,0:50])
        #
        diff4a =(output_vector4a[:,0:50]-input_vector[:,0:50])
        diff4b =(output_vector4b[:,0:50]-input_vector[:,0:50])
        diff4c =(output_vector4c[:,0:50]-input_vector[:,0:50])
        diff4d =(output_vector4d[:,0:50]-input_vector[:,0:50])
        diff4e =(output_vector4e[:,0:50]-input_vector[:,0:50])
        diff4f =(output_vector4f[:,0:50]-input_vector[:,0:50])
        #
        diff6a =(output_vector6a[:,0:50]-input_vector[:,0:50])
        diff6b =(output_vector6b[:,0:50]-input_vector[:,0:50])
        diff6c =(output_vector6c[:,0:50]-input_vector[:,0:50])
        diff6d =(output_vector6d[:,0:50]-input_vector[:,0:50])
        diff6e =(output_vector6e[:,0:50]-input_vector[:,0:50])
        diff6f =(output_vector6f[:,0:50]-input_vector[:,0:50])
        #
        diff8a =(output_vector8a[:,0:50]-input_vector[:,0:50])
        diff8b =(output_vector8b[:,0:50]-input_vector[:,0:50])
        diff8c =(output_vector8c[:,0:50]-input_vector[:,0:50])
        diff8d =(output_vector8d[:,0:50]-input_vector[:,0:50])
        diff8e =(output_vector8e[:,0:50]-input_vector[:,0:50])
        diff8f =(output_vector8f[:,0:50]-input_vector[:,0:50])
        #
        diff10a =(output_vector10a[:,0:50]-input_vector[:,0:50])
        diff10b =(output_vector10b[:,0:50]-input_vector[:,0:50])
        diff10c =(output_vector10c[:,0:50]-input_vector[:,0:50])
        diff10d =(output_vector10d[:,0:50]-input_vector[:,0:50])
        diff10e =(output_vector10e[:,0:50]-input_vector[:,0:50])
        diff10f =(output_vector10f[:,0:50]-input_vector[:,0:50])
        #
        diff12a =(output_vector12a[:,0:50]-input_vector[:,0:50])
        diff12b =(output_vector12b[:,0:50]-input_vector[:,0:50])
        diff12c =(output_vector12c[:,0:50]-input_vector[:,0:50])
        diff12d =(output_vector12d[:,0:50]-input_vector[:,0:50])
        diff12e =(output_vector12e[:,0:50]-input_vector[:,0:50])
        diff12f =(output_vector12f[:,0:50]-input_vector[:,0:50])
        #
        #Average the various instances together
        diff2=(diff2a+diff2b+diff2c+diff2d+diff2e+diff2f)/6.0
        diff4=(diff4a+diff4b+diff4c+diff4d+diff4e+diff4f)/6.0
        diff6=(diff6a+diff6b+diff6c+diff6d+diff6e+diff6f)/6.0
        diff8=(diff8a+diff8b+diff8c+diff8d+diff8e+diff8f)/6.0
        diff10=(diff10a+diff10b+diff10c+diff10d+diff10e+diff10f)/6.0
        diff12=(diff12a+diff12b+diff12c+diff12d+diff12e+diff12f)/6.0
        #
        #Now rescale those increments that are longer than two hours back to a 2-hour increment
        diff4 =diff4/2
        diff6 =diff6/3
        diff8 =diff8/4
        diff10=diff10/5
        diff12=diff12/6
        #
        weights=np.append(np.append([1,1,1],wp1),wp2)
        #weights=np.append(np.append(np.append(np.append(np.append([1,1,1,1,1,1],wp1),wp2),wp3),wp4),wp5)

        # Use only one increment or a hybrid increment based on learning over different time windows.
        if i_option==0:
            combined_diff=diff2
        if i_option==1:
            combined_diff=(diff2+diff4+diff6)/3
        if i_option==2:
            combined_diff=(diff2+diff4+diff6+np1[0]+np1[1]+np2)/np.sum(weights)
        if i_option==3:
            combined_diff=(diff2+diff4+diff6+diff8+diff10+diff12)/6

        # Now do precip
        #combined_ppn=(output_vector2[0,50]+output_vector4[0,50]+output_vector6[0,50]+ppn1[0]+ppn1[1]+ppn2)/np.sum(weights)
        #combined_ppn=(output_vector2[0,50]+output_vector4[0,50]+output_vector6[0,50])/3.0
        combined_ppn=output_vector2a[0,50]
        #
        #Shuffle all data back through memory arrays:
        # ... increments
        np2=np1[0]
        np1=[diff6,diff4]
        # ... and weigths
        wp2=wp1[0]
        wp1=[1,1]
        #
        # ... increments
        #        np5=np4[0]
        #        np4=[np3[0],np3[1]]
        #        np3=[np2[0],np2[1],np2[2]]
        #        np2=[np1[0],np1[1],np1[2],np1[3]]
        #        np1=[diff12,diff10,diff8,diff6,diff4]
        #np1=[     0,     0,    0,diff6,diff4]
        ##np1=[     0,     0,    0,    0,    0]
        # ... and weigths
        #        wp5=wp4[0]
        #        wp4=[wp3[0],wp3[1]]
        #        wp3=[wp2[0],wp2[1],wp2[2]]
        #        wp2=[wp1[0],wp1[1],wp1[2],wp1[3]]
        #        wp1=[1,1,1,1,1]
        ##wp1=[0,0,0,1,1]
        ##wp1=[0,0,0,0,0]
        #
        ppn2=ppn1[0]
        ppn1=[output_vector6a[0,50],output_vector4a[0,50]]
        #
        theta_norm = input_vector[0, 0:25]+combined_diff[0, 0:25]
        qt_norm    = input_vector[0,25:50]+combined_diff[0,25:50]
        #
        recursive_theta_norm[timestep,:]=theta_norm
        recursive_qt_norm   [timestep,:]=qt_norm
        recursive_ppn_norm  [timestep,0]=combined_ppn
        # Store others so can compare them
        recursive_ppn_norm1 [timestep,0]=output_vector2a[0,50]
        recursive_ppn_norm2 [timestep,0]=output_vector4a[0,50]
        recursive_ppn_norm3 [timestep,0]=output_vector6a[0,50]
        recursive_ppn_norm4 [timestep,0]=ppn1[0]
        recursive_ppn_norm5 [timestep,0]=ppn1[1]
        recursive_ppn_norm6 [timestep,0]=ppn2
    #
    #plt.show()
    #
    clim_theta_norm=np.zeros([nt_here,nz])
    clim_qt_norm=np.zeros([nt_here,nz])
    #print('clim_theta_norm.shape=',clim_theta_norm.shape)
    #
    persist_theta_norm=np.zeros([nt_here,nz])*np.nan
    persist_qt_norm=np.zeros([nt_here,nz])*np.nan

    local_clim_theta=np.zeros([nt_here,nz])*np.nan
    local_clim_qt   =np.zeros([nt_here,nz])*np.nan
    #
    for timestep in np.arange(0,nt_here,1):
        persist_theta_norm[timestep,:]=inputs_now[starttimestep,0:25]
        persist_qt_norm[timestep,:]=inputs_now[starttimestep,25:50]
        local_clim_theta[timestep,:]=np.mean(inputs_now[tsperday-1:-1, 0:25],axis=0)
        local_clim_qt   [timestep,:]=np.mean(inputs_now[tsperday-1:-1,25:50],axis=0)
    #
    rmse_theta_clim       = np.sqrt(np.mean(      (clim_theta_norm-truth_theta_norm)**2, axis=1) )
    rmse_theta_persist    = np.sqrt(np.mean(   (persist_theta_norm-truth_theta_norm)**2, axis=1) )
    rmse_theta_nn         = np.sqrt(np.mean( (recursive_theta_norm-truth_theta_norm)**2, axis=1) )
    rmse_theta_gm1        = np.sqrt(np.mean(    (day1_ctl_theta_gm-truth_theta_norm)**2, axis=1) )
    rmse_theta_gm2        = np.sqrt(np.mean(    (day2_ctl_theta_gm-truth_theta_norm)**2, axis=1) )
    rmse_qt_clim          = np.sqrt(np.mean(         (clim_qt_norm-truth_qt_norm)**2,    axis=1) )
    rmse_qt_persist       = np.sqrt(np.mean(      (persist_qt_norm-truth_qt_norm)**2,    axis=1) )
    rmse_qt_nn            = np.sqrt(np.mean(    (recursive_qt_norm-truth_qt_norm)**2,    axis=1) )
    rmse_qt_gm1           = np.sqrt(np.mean(       (day1_ctl_qt_gm-truth_qt_norm)**2,    axis=1) )
    rmse_qt_gm2           = np.sqrt(np.mean(       (day2_ctl_qt_gm-truth_qt_norm)**2,    axis=1) )
    rmse_theta_clim_local = np.sqrt(np.mean(     (local_clim_theta-truth_theta_norm)**2, axis=1) )
    rmse_qt_clim_local    = np.sqrt(np.mean(        (local_clim_qt-truth_qt_norm)**2,    axis=1) )

    overall_rmse=np.mean(rmse_theta_nn+rmse_qt_nn)

    print(my_counter,overall_rmse)
    my_counter=my_counter+1
    #    print('my_counter=',my_counter)

    if my_counter==41:
        # Write out this data which is the median performer.
        fileout='best_theta_nn.dat'
        np.savetxt(fileout, recursive_theta_norm, fmt='%10.7f')
        fileout='best_theta_truth.dat'
        np.savetxt(fileout, truth_theta_norm, fmt='%10.7f')
        fileout='best_qt_nn.dat'
        np.savetxt(fileout, recursive_qt_norm, fmt='%10.7f')
        fileout='best_qt_truth.dat'
        np.savetxt(fileout, truth_qt_norm, fmt='%10.7f')

    if 1==2:
        cmap = plt.get_cmap('bwr')
        lev=setup['k_indices']
        fig, axs = plt.subplots(2, 5)
        im=axs[0,0].pcolormesh(np.transpose(truth_theta_norm),vmin=-1,vmax=1,cmap=cmap)
        im=axs[0,1].pcolormesh(np.transpose(recursive_theta_norm),vmin=-1,vmax=1,cmap=cmap)
        im=axs[0,2].pcolormesh(np.transpose(persist_theta_norm),vmin=-1,vmax=1,cmap=cmap)
        im=axs[0,3].pcolormesh(np.transpose(day1_ctl_theta_gm),vmin=-1,vmax=1,cmap=cmap)
        #
        im=axs[0,4].plot(rmse_theta_clim,'k-', label='Climatology')
        im=axs[0,4].plot(rmse_theta_persist,'b-', label='Persistence')
        im=axs[0,4].plot(rmse_theta_nn,'r-', label='NN')
        #
        im=axs[1,0].pcolormesh(np.transpose(truth_qt_norm),vmin=-1,vmax=1,cmap=cmap)
        im=axs[1,1].pcolormesh(np.transpose(recursive_qt_norm),vmin=-1,vmax=1,cmap=cmap)
        im=axs[1,2].pcolormesh(np.transpose(persist_qt_norm),vmin=-1,vmax=1,cmap=cmap)
        im=axs[1,3].pcolormesh(np.transpose(day1_ctl_qt_gm),vmin=-1,vmax=1,cmap=cmap)
        #
        im=axs[1,4].plot(rmse_qt_clim,'k-', label='Climatology')
        im=axs[1,4].plot(rmse_qt_persist,'b-', label='Persistence')
        im=axs[1,4].plot(rmse_qt_nn,'r-', label='NN')
        im=axs[1,4].legend()
        plt.show()
    #
    tmp=np.reshape(truth_theta_norm[0:24,:],(12,2,nz))
    mean_diurnal_cycle_theta_truth=np.mean(tmp,axis=1)
    tmp=np.reshape(truth_qt_norm[0:24,:],(12,2,nz))
    mean_diurnal_cycle_qt_truth=np.mean(tmp,axis=1)
    tmp=np.reshape(day1_ctl_theta_gm[0:24,:],(12,2,nz))
    mean_diurnal_cycle_theta_gm=np.mean(tmp,axis=1)
    tmp=np.reshape(day1_ctl_qt_gm[0:24,:],(12,2,nz))
    mean_diurnal_cycle_qt_gm=np.mean(tmp,axis=1)
    tmp=np.reshape(recursive_theta_norm[0:24,:],(12,2,nz))
    mean_diurnal_cycle_theta_nn=np.mean(tmp,axis=1)
    tmp=np.reshape(recursive_qt_norm[0:24,:],(12,2,nz))
    mean_diurnal_cycle_qt_nn=np.mean(tmp,axis=1)
    #
    return {'rmse_theta_clim':rmse_theta_clim,'rmse_theta_persist':rmse_theta_persist,'rmse_theta_nn':rmse_theta_nn,'rmse_qt_clim':rmse_qt_clim,'rmse_qt_persist':rmse_qt_persist,'rmse_qt_nn':rmse_qt_nn, 'rmse_theta_gm1':rmse_theta_gm1, 'rmse_qt_gm1':rmse_qt_gm1, 'rmse_theta_gm2':rmse_theta_gm2, 'rmse_qt_gm2':rmse_qt_gm2, 'mean_diurnal_cycle_theta_truth':mean_diurnal_cycle_theta_truth, 'mean_diurnal_cycle_qt_truth':mean_diurnal_cycle_qt_truth, 'mean_diurnal_cycle_theta_gm':mean_diurnal_cycle_theta_gm, 'mean_diurnal_cycle_qt_gm':mean_diurnal_cycle_qt_gm, 'mean_diurnal_cycle_theta_nn':mean_diurnal_cycle_theta_nn, 'mean_diurnal_cycle_qt_nn':mean_diurnal_cycle_qt_nn, 'rmse_theta_clim_local':rmse_theta_clim_local, 'rmse_qt_clim_local':rmse_qt_clim_local, 'time_window':time_window, 'my_counter':my_counter, 'best_so_far':best_so_far};

# End of functions

def main():
    
    my_counter=1
    best_so_far=0.0

    for dd in np.arange(0,3,1):

        setup=setup_parameters()

        months_to_do=['20170531-20170630']
        monthyearpaths=['JUN2017']

        days_to_do_in_each_month=[31]

        month=0
        date_range=months_to_do[month]
        monthyear=monthyearpaths[month]
        ndays=days_to_do_in_each_month[month]

        domains_to_do=[1]

        i_flag_add_adv=0
    
        if dd==0:
            region_to_do=[6-1]
        if dd==1:
            region_to_do=[74-1]
        if dd==2:
            region_to_do=[67-1]
    
        print('region_to_do=',region_to_do)
    
        processed=process_series_of_domains(date_range,setup,region_to_do,domains_to_do,i_flag_add_adv,monthyear,ndays)

        if 1==2:
            cmap = plt.get_cmap('bwr')
            fig, axs = plt.subplots(3, 2)
            im=axs[0,0].pcolormesh(np.transpose(processed1['output_alldom'][:,50:75]),vmin=-1,vmax=1,cmap=cmap)
            im=axs[0,1].pcolormesh(np.transpose(processed1['output_alldom'][:,75:100]),vmin=-1,vmax=1,cmap=cmap)
            im=axs[1,0].pcolormesh(np.transpose(processed3['output_alldom'][:,50:75]),vmin=-1,vmax=1,cmap=cmap)
            im=axs[1,1].pcolormesh(np.transpose(processed3['output_alldom'][:,75:100]),vmin=-1,vmax=1,cmap=cmap)
            im=axs[2,0].pcolormesh(np.transpose(processed3['output_alldom'][:,50:75])-np.transpose(processed1['output_alldom'][:,50:75]),vmin=-1,vmax=1,cmap=cmap)
            im=axs[2,1].pcolormesh(np.transpose(processed3['output_alldom'][:,75:100])-np.transpose(processed1['output_alldom'][:,75:100]),vmin=-1,vmax=1,cmap=cmap)
            plt.show()

        if 1==2:
            cmap = plt.get_cmap('bwr')
            fig, axs = plt.subplots(3, 2)
            im=axs[0,0].plot(np.transpose(processed['output_alldom'][:,156]))
            plt.show()

        adv_incr_raw=unnormalise_adv_incr(processed)

        tsperday=12

        nstartdates=25

        # Lets do some n-day runs.
        ndays=5
        nt=(ndays*tsperday)+1

        keep_rmse_theta_clim=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_theta_persist=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_theta_nn=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_theta_gm1=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_theta_gm2=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_qt_clim=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_qt_persist=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_qt_nn=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_qt_gm1=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_qt_gm2=np.zeros([nt,nstartdates])*np.nan

        keep_rmse_theta_clim_local=np.zeros([nt,nstartdates])*np.nan
        keep_rmse_qt_clim_local=np.zeros([nt,nstartdates])*np.nan

        #Keep mean diurnal cycles (dc)
        nz=25
        keep_dc_theta_truth=np.zeros([tsperday,nz,nstartdates])*np.nan
        keep_dc_qt_truth   =np.zeros([tsperday,nz,nstartdates])*np.nan
        keep_dc_theta_gm   =np.zeros([tsperday,nz,nstartdates])*np.nan
        keep_dc_qt_gm      =np.zeros([tsperday,nz,nstartdates])*np.nan
        keep_dc_theta_nn   =np.zeros([tsperday,nz,nstartdates])*np.nan
        keep_dc_qt_nn      =np.zeros([tsperday,nz,nstartdates])*np.nan
        ############# GET HOLD OF GLOBAL MODEL DATA AND RESCALE IT ####
        if dd==0:
            nc_file    = '/scratch/frme/ML/u-bx951/step3alt_single_columns/0N0E_subregion_1.air_potential_temperature.nc'
            fh         = Dataset(nc_file, mode='r')
            day1_theta = fh.variables['air_potential_temperature'][:]
            fh.close()
            nc_file    = '/scratch/frme/ML/u-bx951/step3alt_single_columns/0N0E_subregion_1.specific_humidity.nc'
            fh         = Dataset(nc_file, mode='r')
            day1_qt    = fh.variables['specific_humidity'][:]
            fh.close()
        if dd==1:
            nc_file    = '/scratch/frme/ML/u-bx951/step3alt_single_columns/40N25W_subregion_1.air_potential_temperature.nc'
            fh         = Dataset(nc_file, mode='r')
            day1_theta = fh.variables['air_potential_temperature'][:]
            fh.close()
            nc_file    = '/scratch/frme/ML/u-bx951/step3alt_single_columns/40N25W_subregion_1.specific_humidity.nc'
            fh         = Dataset(nc_file, mode='r')
            day1_qt    = fh.variables['specific_humidity'][:]
            fh.close()
        if dd==2:
            nc_file    = '/scratch/frme/ML/u-bx951/step3alt_single_columns/40S0E_subregion_1.air_potential_temperature.nc'
            fh         = Dataset(nc_file, mode='r')
            day1_theta = fh.variables['air_potential_temperature'][:]
            fh.close()
            nc_file    = '/scratch/frme/ML/u-bx951/step3alt_single_columns/40S0E_subregion_1.specific_humidity.nc'
            fh         = Dataset(nc_file, mode='r')
            day1_qt    = fh.variables['specific_humidity'][:]
            fh.close()
    
        print(day1_theta.shape)

        date_range='20151231-20160131'
        filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_mean_prof.avg_over_domains.txt'
        theta_mean_prof=np.loadtxt(filein)
        filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_2p5_prof.avg_over_domains.txt'
        theta_2p5_prof=np.loadtxt(filein)
        filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.theta_975_prof.avg_over_domains.txt'
        theta_975_prof=np.loadtxt(filein)
        filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_mean_prof.avg_over_domains.txt'
        qt_mean_prof=np.loadtxt(filein)
        filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_2p5_prof.avg_over_domains.txt'
        qt_2p5_prof=np.loadtxt(filein)
        filein='/home/h01/frme/cyrilmorcrette-projects/python/ml_supermean_profiles/'+date_range+'.qt_975_prof.avg_over_domains.txt'
        qt_975_prof=np.loadtxt(filein)

        day1_theta = day1_theta[0:50,:]
        day1_qt    = day1_qt   [0:50,:]

        tmp=day1_theta.shape
        tmp_nz=tmp[0]
        day1_theta_gm=np.copy(day1_theta)*0.0
        day1_qt_gm=np.copy(day1_qt)*0.0
        for k in np.arange(0,tmp_nz,1):
            halfrange=np.amax([theta_975_prof[k]-theta_mean_prof[k],theta_mean_prof[k]-theta_2p5_prof[k]])
            day1_theta_gm[k,:]=(day1_theta[k,:]-theta_mean_prof[k])/halfrange
            halfrange=np.amax([qt_975_prof[k]-qt_mean_prof[k],qt_mean_prof[k]-qt_2p5_prof[k]])
            day1_qt_gm     [k,:]=(day1_qt[k,:]-qt_mean_prof[k])/halfrange
        day1_theta_gm=day1_theta_gm[setup['k_indices'],:]
        day1_qt_gm=day1_qt_gm[setup['k_indices'],:]
        ##########################################

        ts_in_gm_data=61
        for startday in np.arange(0,nstartdates,1):
            starttimestep=startday*tsperday
            start_ts_ctl=startday*ts_in_gm_data
            day1_ctl_theta_gm=np.transpose(day1_theta_gm[:,start_ts_ctl:start_ts_ctl+nt])
            day1_ctl_qt_gm=np.transpose(day1_qt_gm[:,start_ts_ctl:start_ts_ctl+nt])
            day2_ctl_theta_gm=day1_ctl_theta_gm
            day2_ctl_qt_gm=day1_ctl_qt_gm
            # Do the business!
            outcome=recursive_model(processed,starttimestep,adv_incr_raw,ndays,setup,day1_ctl_theta_gm,day1_ctl_qt_gm,day2_ctl_theta_gm,day2_ctl_qt_gm,my_counter,best_so_far)
            #
            my_counter=outcome['my_counter']
            best_so_far=outcome['best_so_far']
            #
            keep_rmse_theta_clim[:,startday]=outcome['rmse_theta_clim']
            keep_rmse_theta_persist[:,startday]=outcome['rmse_theta_persist']
            keep_rmse_theta_nn[:,startday]=outcome['rmse_theta_nn']
            keep_rmse_theta_gm1[:,startday]=outcome['rmse_theta_gm1']
            keep_rmse_theta_gm2[:,startday]=outcome['rmse_theta_gm2']
            keep_rmse_qt_clim[:,startday]=outcome['rmse_qt_clim']
            keep_rmse_qt_persist[:,startday]=outcome['rmse_qt_persist']
            keep_rmse_qt_nn[:,startday]=outcome['rmse_qt_nn']
            keep_rmse_qt_gm1[:,startday]=outcome['rmse_qt_gm1']
            keep_rmse_qt_gm2[:,startday]=outcome['rmse_qt_gm2']
            #
            keep_rmse_theta_clim_local[:,startday]=outcome['rmse_theta_clim_local']
            keep_rmse_qt_clim_local[:,startday]=outcome['rmse_qt_clim_local']
            #
            keep_dc_theta_truth[:,:,startday]=outcome['mean_diurnal_cycle_theta_truth']
            keep_dc_qt_truth   [:,:,startday]=outcome['mean_diurnal_cycle_qt_truth']
            keep_dc_theta_gm[:,:,startday]=outcome['mean_diurnal_cycle_theta_gm']
            keep_dc_qt_gm   [:,:,startday]=outcome['mean_diurnal_cycle_qt_gm']
            keep_dc_theta_nn[:,:,startday]=outcome['mean_diurnal_cycle_theta_nn']
            keep_dc_qt_nn   [:,:,startday]=outcome['mean_diurnal_cycle_qt_nn']
        #end 

        rmse_theta_clim=np.mean(keep_rmse_theta_clim,axis=1)
        rmse_theta_persist=np.mean(keep_rmse_theta_persist,axis=1)
        rmse_theta_nn=np.mean(keep_rmse_theta_nn,axis=1)
        rmse_theta_gm1=np.mean(keep_rmse_theta_gm1,axis=1)
        rmse_theta_gm2=np.mean(keep_rmse_theta_gm2,axis=1)
        rmse_qt_clim=np.mean(keep_rmse_qt_clim,axis=1)
        rmse_qt_persist=np.mean(keep_rmse_qt_persist,axis=1)
        rmse_qt_nn=np.mean(keep_rmse_qt_nn,axis=1)
        rmse_qt_gm1=np.mean(keep_rmse_qt_gm1,axis=1)
        rmse_qt_gm2=np.mean(keep_rmse_qt_gm2,axis=1)

        rmse_theta_clim_local=np.mean(keep_rmse_theta_clim_local,axis=1)
        rmse_qt_clim_local=np.mean(keep_rmse_qt_clim_local,axis=1)

        stde_rmse_theta_clim=np.std(keep_rmse_theta_clim,axis=1)/np.sqrt(nstartdates)
        stde_rmse_theta_persist=np.std(keep_rmse_theta_persist,axis=1)/np.sqrt(nstartdates)
        stde_rmse_theta_nn=np.std(keep_rmse_theta_nn,axis=1)/np.sqrt(nstartdates)
        stde_rmse_theta_gm1=np.std(keep_rmse_theta_gm1,axis=1)/np.sqrt(nstartdates)
        stde_rmse_theta_gm2=np.std(keep_rmse_theta_gm2,axis=1)/np.sqrt(nstartdates)
        stde_rmse_qt_clim=np.std(keep_rmse_qt_clim,axis=1)/np.sqrt(nstartdates)
        stde_rmse_qt_persist=np.std(keep_rmse_qt_persist,axis=1)/np.sqrt(nstartdates)
        stde_rmse_qt_nn=np.std(keep_rmse_qt_nn,axis=1)/np.sqrt(nstartdates)
        stde_rmse_qt_gm1=np.std(keep_rmse_qt_gm1,axis=1)/np.sqrt(nstartdates)
        stde_rmse_qt_gm2=np.std(keep_rmse_qt_gm2,axis=1)/np.sqrt(nstartdates)

        time=np.arange(0,nt)*2.0/24.0

        if 1==2:
            fig, axs = plt.subplots(2, 2)
            im=axs[0,0].plot(time,rmse_theta_clim,'k-', label='Climatology')
            im=axs[0,0].plot(time,rmse_theta_persist,'b-', label='Persist')
            im=axs[0,0].plot(time,rmse_theta_nn,'r-', label='NN')
            im=axs[0,0].plot(time,rmse_theta_gm1,'c-', label='GM1')
            im=axs[0,0].plot([time[0],time[-1]],[rmse_theta_gm1[0],rmse_theta_gm1[-1]],'g--')
            im=axs[0,0].plot(time,rmse_theta_clim-stde_rmse_theta_clim,'k:', label='Climatology')
            im=axs[0,0].plot(time,rmse_theta_clim+stde_rmse_theta_clim,'k:', label='Climatology')
            im=axs[0,0].plot(time,rmse_theta_persist-stde_rmse_theta_persist,'b:', label='Persist')
            im=axs[0,0].plot(time,rmse_theta_persist+stde_rmse_theta_persist,'b:', label='Persist')
            im=axs[0,0].plot(time,rmse_theta_nn-stde_rmse_theta_nn,'r:', label='NN')
            im=axs[0,0].plot(time,rmse_theta_nn+stde_rmse_theta_nn,'r:', label='NN')
            im=axs[0,0].plot(time,rmse_theta_gm1-stde_rmse_theta_gm1,'c:', label='GM1')
            im=axs[0,0].plot(time,rmse_theta_gm1+stde_rmse_theta_gm1,'c:', label='GM1')
            #
            im=axs[0,1].plot(time,rmse_qt_clim,'k-', label='Climatology')
            im=axs[0,1].plot(time,rmse_qt_persist,'b-', label='Persist')
            im=axs[0,1].plot(time,rmse_qt_nn,'r-', label='NN')
            im=axs[0,1].plot(time,rmse_qt_gm1,'c-', label='GM1')
            im=axs[0,1].plot(time,rmse_qt_clim-stde_rmse_qt_clim,'k:', label='Climatology')
            im=axs[0,1].plot(time,rmse_qt_clim+stde_rmse_qt_clim,'k:', label='Climatology')
            im=axs[0,1].plot(time,rmse_qt_persist-stde_rmse_qt_persist,'b:', label='Persist')
            im=axs[0,1].plot(time,rmse_qt_persist+stde_rmse_qt_persist,'b:', label='Persist')
            im=axs[0,1].plot(time,rmse_qt_nn-stde_rmse_qt_nn,'r:', label='NN')
            im=axs[0,1].plot(time,rmse_qt_nn+stde_rmse_qt_nn,'r:', label='NN')
            im=axs[0,1].plot(time,rmse_qt_gm1-stde_rmse_qt_gm1,'c:', label='GM1')
            im=axs[0,1].plot(time,rmse_qt_gm1+stde_rmse_qt_gm1,'c:', label='GM1')
            #
            im=axs[1,0].plot(time,rmse_theta_clim,'k-', label='Climatology')
            im=axs[1,0].plot(time,rmse_theta_persist,'b-', label='Persist')
            im=axs[1,0].plot(time,rmse_theta_nn,'r-', label='NN')
            im=axs[1,0].plot(time,rmse_theta_gm1,'c-', label='GM1')
            im=axs[1,0].legend()
            im=axs[1,0].set_xlabel('Time [days]')
            im=axs[1,0].set_ylabel('Theta RMSE')
            #
            im=axs[1,1].plot(time,rmse_qt_clim,'k-', label='Climatology')
            im=axs[1,1].plot(time,rmse_qt_persist,'b-', label='Persist')
            im=axs[1,1].plot(time,rmse_qt_nn,'r-', label='NN')
            im=axs[1,1].plot(time,rmse_qt_gm1,'c-', label='GM1')
            im=axs[1,1].legend()
            im=axs[1,1].set_xlabel('Time [days]')
            im=axs[1,1].set_ylabel('qt RMSE')
            plt.show()

        print('keep_dc_theta_truth=',keep_dc_theta_truth.shape)

        supermean_dc_theta_truth=np.mean(keep_dc_theta_truth,axis=2)
        supermean_dc_qt_truth   =np.mean(keep_dc_qt_truth,axis=2)
        supermean_dc_theta_gm   =np.mean(keep_dc_theta_gm,axis=2)
        supermean_dc_qt_gm      =np.mean(keep_dc_qt_gm,axis=2)
        supermean_dc_theta_nn   =np.mean(keep_dc_theta_nn,axis=2)
        supermean_dc_qt_nn      =np.mean(keep_dc_qt_nn,axis=2)
        print('supermean_dc_qt_nn=',supermean_dc_qt_nn.shape)

        my_max_t=np.max([np.max(np.abs(supermean_dc_theta_truth)),np.max(np.abs(supermean_dc_theta_gm)),np.max(np.abs(supermean_dc_theta_nn))])
        my_max_q=np.max([np.max(np.abs(supermean_dc_qt_truth)),   np.max(np.abs(supermean_dc_qt_gm)),   np.max(np.abs(supermean_dc_qt_nn))])
        print('my_max_t,my_max_q=',my_max_t,my_max_q)

        time_here=np.arange(0,24,2)
        lev=setup['k_indices']+1

        if 1==2:
            cmap = plt.get_cmap('bwr')
            fig, axs = plt.subplots(3, 2)
            im=axs[0,0].pcolormesh(time_here,lev,np.transpose(supermean_dc_theta_gm),   vmin=-my_max_t,vmax=my_max_t,cmap=cmap)
            im=axs[0,1].pcolormesh(time_here,lev,np.transpose(supermean_dc_qt_gm),      vmin=-my_max_q,vmax=my_max_q,cmap=cmap)
            im=axs[0,0].set_title('GM theta')
            im=axs[0,1].set_title('GM qt')
            im=axs[1,0].pcolormesh(time_here,lev,np.transpose(supermean_dc_theta_truth),vmin=-my_max_t,vmax=my_max_t,cmap=cmap)
            im=axs[1,1].pcolormesh(time_here,lev,np.transpose(supermean_dc_qt_truth),   vmin=-my_max_q,vmax=my_max_q,cmap=cmap)
            im=axs[1,0].set_title('Truth theta')
            im=axs[1,1].set_title('Truth qt')
            im=axs[2,0].pcolormesh(time_here,lev,np.transpose(supermean_dc_theta_nn),   vmin=-my_max_t,vmax=my_max_t,cmap=cmap)
            im=axs[2,1].pcolormesh(time_here,lev,np.transpose(supermean_dc_qt_nn),      vmin=-my_max_q,vmax=my_max_q,cmap=cmap)
            im=axs[2,0].set_title('NN theta')
            im=axs[2,1].set_title('NN qt')
            im=axs[2,0].set_xlabel('Time [hours]')
            im=axs[2,0].set_ylabel('Height [model levels]')
            im=axs[2,1].set_xlabel('Time [hours]')
            im=axs[2,1].set_ylabel('Height [model levels]')
            im=axs[0,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[0,1].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[1,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[1,1].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[2,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[2,1].xaxis.set_major_locator(MultipleLocator(3))
            plt.show()

        anom_supermean_dc_theta_gm=np.copy(supermean_dc_theta_gm)*0.0
        anom_supermean_dc_theta_nn=np.copy(supermean_dc_theta_gm)*0.0
        anom_supermean_dc_theta_truth=np.copy(supermean_dc_theta_gm)*0.0

        anom_supermean_dc_qt_gm=np.copy(supermean_dc_theta_gm)*0.0
        anom_supermean_dc_qt_nn=np.copy(supermean_dc_theta_gm)*0.0
        anom_supermean_dc_qt_truth=np.copy(supermean_dc_theta_gm)*0.0

        print('tmp_nz=',tmp_nz)

        for k in np.arange(0,nz,1):
            anom_supermean_dc_theta_gm[:,k]=supermean_dc_theta_gm[:,k]-np.mean(supermean_dc_theta_truth[:,k])
            anom_supermean_dc_theta_nn[:,k]=supermean_dc_theta_nn[:,k]-np.mean(supermean_dc_theta_truth[:,k])
            anom_supermean_dc_theta_truth[:,k]=supermean_dc_theta_truth[:,k]-np.mean(supermean_dc_theta_truth[:,k])
            anom_supermean_dc_qt_gm[:,k]=supermean_dc_qt_gm[:,k]-np.mean(supermean_dc_qt_truth[:,k])
            anom_supermean_dc_qt_nn[:,k]=supermean_dc_qt_nn[:,k]-np.mean(supermean_dc_qt_truth[:,k])
            anom_supermean_dc_qt_truth[:,k]=supermean_dc_qt_truth[:,k]-np.mean(supermean_dc_qt_truth[:,k])

        my_max_t=np.max([np.max(np.abs(anom_supermean_dc_theta_truth)),np.max(np.abs(anom_supermean_dc_theta_gm)),np.max(np.abs(anom_supermean_dc_theta_nn))])
        my_max_q=np.max([np.max(np.abs(anom_supermean_dc_qt_truth)),   np.max(np.abs(anom_supermean_dc_qt_gm)),   np.max(np.abs(anom_supermean_dc_qt_nn))])
        print('my_max_t,my_max_q=',my_max_t,my_max_q)

        if 1==2:
            cmap = plt.get_cmap('bwr')
            fig, axs = plt.subplots(3, 2)
            im=axs[0,0].pcolormesh(time_here,lev,np.transpose(anom_supermean_dc_theta_gm),   vmin=-my_max_t,vmax=my_max_t,cmap=cmap)
            im=axs[0,1].pcolormesh(time_here,lev,np.transpose(anom_supermean_dc_qt_gm),      vmin=-my_max_q,vmax=my_max_q,cmap=cmap)
            im=axs[0,0].set_title('GM theta')
            im=axs[0,1].set_title('GM qt')
            im=axs[1,0].pcolormesh(time_here,lev,np.transpose(anom_supermean_dc_theta_truth),vmin=-my_max_t,vmax=my_max_t,cmap=cmap)
            im=axs[1,1].pcolormesh(time_here,lev,np.transpose(anom_supermean_dc_qt_truth),   vmin=-my_max_q,vmax=my_max_q,cmap=cmap)
            im=axs[1,0].set_title('Truth theta')
            im=axs[1,1].set_title('Truth qt')
            im=axs[2,0].pcolormesh(time_here,lev,np.transpose(anom_supermean_dc_theta_nn),   vmin=-my_max_t,vmax=my_max_t,cmap=cmap)
            im=axs[2,1].pcolormesh(time_here,lev,np.transpose(anom_supermean_dc_qt_nn),      vmin=-my_max_q,vmax=my_max_q,cmap=cmap)
            im=axs[2,0].set_title('NN theta')
            im=axs[2,1].set_title('NN qt')
            im=axs[2,0].set_xlabel('Time [hours]')
            im=axs[2,0].set_ylabel('Height [model levels]')
            im=axs[2,1].set_xlabel('Time [hours]')
            im=axs[2,1].set_ylabel('Height [model levels]')
            im=axs[0,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[0,1].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[1,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[1,1].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[2,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[2,1].xaxis.set_major_locator(MultipleLocator(3))
            plt.show()

        gm_theta_error=np.sqrt(np.mean((supermean_dc_theta_gm-supermean_dc_theta_truth)**2.0, axis=1))
        nn_theta_error=np.sqrt(np.mean((supermean_dc_theta_nn-supermean_dc_theta_truth)**2.0, axis=1))
        gm_qt_error=np.sqrt(np.mean((supermean_dc_qt_gm-supermean_dc_qt_truth)**2.0, axis=1))
        nn_qt_error=np.sqrt(np.mean((supermean_dc_qt_nn-supermean_dc_qt_truth)**2.0, axis=1))

        print('gm_theta_error.shape=',gm_theta_error.shape)

        if 1==2:
            fig, axs = plt.subplots(2, 2)
            im=axs[0,0].plot(time_here,gm_theta_error,'k-')
            im=axs[0,0].plot(time_here,nn_theta_error,'r--')
            im=axs[0,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[0,0].set_xlabel('Time [hours]')
            im=axs[0,1].plot(time_here,gm_qt_error,'k-')
            im=axs[0,1].plot(time_here,nn_qt_error,'r--')
            im=axs[0,1].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[0,1].set_xlabel('Time [hours]')
            z=np.polyfit(time_here,gm_theta_error,1)
            gm_theta_error=gm_theta_error-((time_here*z[0])+z[1])
            im=axs[1,0].plot(time_here,gm_theta_error,'k-')
            z=np.polyfit(time_here,nn_theta_error,1)
            nn_theta_error=nn_theta_error-((time_here*z[0])+z[1])
            im=axs[1,0].plot(time_here,nn_theta_error,'r--')
            z=np.polyfit(time_here,gm_qt_error,1)
            gm_qt_error=gm_qt_error-((time_here*z[0])+z[1])
            im=axs[1,1].plot(time_here,gm_qt_error,'k-')
            z=np.polyfit(time_here,nn_qt_error,1)
            nn_qt_error=nn_qt_error-((time_here*z[0])+z[1])
            im=axs[1,1].plot(time_here,nn_qt_error,'r--')
            im=axs[1,0].xaxis.set_major_locator(MultipleLocator(3))
            im=axs[1,1].xaxis.set_major_locator(MultipleLocator(3))
            plt.show()

        time_window=outcome['time_window']
        expt_descr='ml_eval_output/'+time_window+'_r'+str(region_to_do[0]+1)+'_d'+str(domains_to_do[0])
        ext='_ml_rmse_with_lead_time.txt'

        fileout=expt_descr+'_rmse_theta_clim'+ext
        np.savetxt(fileout, rmse_theta_clim, fmt='%10.7f')
        fileout=expt_descr+'_rmse_theta_persist'+ext
        np.savetxt(fileout, rmse_theta_persist, fmt='%10.7f')
        fileout=expt_descr+'_rmse_theta_nn'+ext
        np.savetxt(fileout, rmse_theta_nn, fmt='%10.7f')
        fileout=expt_descr+'_rmse_theta_gm1'+ext
        np.savetxt(fileout, rmse_theta_gm1, fmt='%10.7f')
        fileout=expt_descr+'_rmse_theta_gm2'+ext
        np.savetxt(fileout, rmse_theta_gm2, fmt='%10.7f')
        fileout=expt_descr+'_rmse_qt_clim'+ext
        np.savetxt(fileout, rmse_qt_clim, fmt='%10.7f')
        fileout=expt_descr+'_rmse_qt_persist'+ext
        np.savetxt(fileout, rmse_qt_persist, fmt='%10.7f')
        fileout=expt_descr+'_rmse_qt_nn'+ext
        np.savetxt(fileout, rmse_qt_nn, fmt='%10.7f')
        fileout=expt_descr+'_rmse_qt_gm1'+ext
        np.savetxt(fileout, rmse_qt_gm1, fmt='%10.7f')
        fileout=expt_descr+'_rmse_qt_gm2'+ext
        np.savetxt(fileout, rmse_qt_gm2, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_theta_clim'+ext
        np.savetxt(fileout, stde_rmse_theta_clim, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_theta_persist'+ext
        np.savetxt(fileout, stde_rmse_theta_persist, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_theta_nn'+ext
        np.savetxt(fileout, stde_rmse_theta_nn, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_theta_gm1'+ext
        np.savetxt(fileout, stde_rmse_theta_gm1, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_theta_gm2'+ext
        np.savetxt(fileout, stde_rmse_theta_gm2, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_qt_clim'+ext
        np.savetxt(fileout, stde_rmse_qt_clim, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_qt_persist'+ext
        np.savetxt(fileout, stde_rmse_qt_persist, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_qt_nn'+ext
        np.savetxt(fileout, stde_rmse_qt_nn, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_qt_gm1'+ext
        np.savetxt(fileout, stde_rmse_qt_gm1, fmt='%10.7f')
        fileout=expt_descr+'_stde_rmse_qt_gm2'+ext
        np.savetxt(fileout, stde_rmse_qt_gm2, fmt='%10.7f')
        fileout=expt_descr+'_rmse_theta_clim_local'+ext
        np.savetxt(fileout, rmse_theta_clim_local, fmt='%10.7f')
        fileout=expt_descr+'_rmse_qt_clim_local'+ext
        np.savetxt(fileout, rmse_qt_clim_local, fmt='%10.7f')

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()



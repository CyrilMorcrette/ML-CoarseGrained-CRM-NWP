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
from keras.models import load_model
from ml_namelist_MASTERCOPY import setup_parameters
from ml_cjm_functions_MASTERCOPY import process_series_of_domains
from ml_cjm_functions_MASTERCOPY import read_in_range_profiles
from ml_cjm_functions_MASTERCOPY import read_in_centile_profiles
from ml_cjm_functions_MASTERCOPY import read_in_mean_profiles

from keras.models import model_from_json

setup=setup_parameters()

json_file=open('expt166/expt_a/ml_model.json', 'r')
loaded_model_json=json_file.read()
json_file.close()
model2a=model_from_json(loaded_model_json)
model2a.load_weights('expt166/expt_a/ml_lastest_saved_weights.h5')
model2b=model_from_json(loaded_model_json)
model2b.load_weights('expt166/expt_b/ml_lastest_saved_weights.h5')
model2c=model_from_json(loaded_model_json)
model2c.load_weights('expt166/expt_c/ml_lastest_saved_weights.h5')
model2d=model_from_json(loaded_model_json)
model2d.load_weights('expt166/expt_d/ml_lastest_saved_weights.h5')
model2e=model_from_json(loaded_model_json)
model2e.load_weights('expt166/expt_e/ml_lastest_saved_weights.h5')
model2f=model_from_json(loaded_model_json)
model2f.load_weights('expt166/expt_f/ml_lastest_saved_weights.h5')

model4a=model_from_json(loaded_model_json)
model4a.load_weights('expt167/expt_a/ml_lastest_saved_weights.h5')
model4b=model_from_json(loaded_model_json)
model4b.load_weights('expt167/expt_b/ml_lastest_saved_weights.h5')
model4c=model_from_json(loaded_model_json)
model4c.load_weights('expt167/expt_c/ml_lastest_saved_weights.h5')
model4d=model_from_json(loaded_model_json)
model4d.load_weights('expt167/expt_d/ml_lastest_saved_weights.h5')
model4e=model_from_json(loaded_model_json)
model4e.load_weights('expt167/expt_e/ml_lastest_saved_weights.h5')
model4f=model_from_json(loaded_model_json)
model4f.load_weights('expt167/expt_f/ml_lastest_saved_weights.h5')

model6a=model_from_json(loaded_model_json)
model6a.load_weights('expt168/expt_a/ml_lastest_saved_weights.h5')
model6b=model_from_json(loaded_model_json)
model6b.load_weights('expt168/expt_b/ml_lastest_saved_weights.h5')
model6c=model_from_json(loaded_model_json)
model6c.load_weights('expt168/expt_c/ml_lastest_saved_weights.h5')
model6d=model_from_json(loaded_model_json)
model6d.load_weights('expt168/expt_d/ml_lastest_saved_weights.h5')
model6e=model_from_json(loaded_model_json)
model6e.load_weights('expt168/expt_e/ml_lastest_saved_weights.h5')
model6f=model_from_json(loaded_model_json)
model6f.load_weights('expt168/expt_f/ml_lastest_saved_weights.h5')

lev=setup['k_indices']

in_perturb_zero   =np.zeros([25,110])

# Make sure we do a copy
in_perturb_theta=np.copy(in_perturb_zero)
in_perturb_q    =np.copy(in_perturb_zero)

delta_theta=0.05
delta_q=delta_theta
for i in np.arange(0,25):
    in_perturb_theta[i,i+50]  =delta_theta
    in_perturb_q    [i,i+75]  =delta_q

#i=0
#in_perturb_theta[i,i+50]=delta_theta
#in_perturb_theta[i,i+50+1]=delta_theta
#in_perturb_q    [i,i+75]=delta_q
#in_perturb_q    [i,i+75+1]=delta_q

#i=24
#in_perturb_theta[i,i+50]=delta_theta
#in_perturb_theta[i,i+50-1]=delta_theta
#in_perturb_q    [i,i+75]=delta_q
#in_perturb_q    [i,i+75-1]=delta_q

cmax=delta_theta
if 1==1:
    cmap = plt.get_cmap('bwr')   
    fig, axs = plt.subplots(2, 2)
    im=axs[0,0].pcolormesh(np.transpose(in_perturb_theta),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,1].pcolormesh(np.transpose(in_perturb_q),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,0].pcolormesh(np.transpose(in_perturb_zero),vmin=-cmax,vmax=cmax,cmap=cmap)
    plt.show()

filein='../matlab/standard_atm_p_on_l70_levels_1_to_50.dat'
p_on_lev=np.loadtxt(filein, delimiter=",")
p_on_lev=p_on_lev[lev]

out2a_perturb_theta=model2a.predict(in_perturb_theta)
out2a_perturb_q    =model2a.predict(in_perturb_q    )
out2a_perturb_zero =model2a.predict(in_perturb_zero )
out2b_perturb_theta=model2b.predict(in_perturb_theta)
out2b_perturb_q    =model2b.predict(in_perturb_q    )
out2b_perturb_zero =model2b.predict(in_perturb_zero )
out2c_perturb_theta=model2c.predict(in_perturb_theta)
out2c_perturb_q    =model2c.predict(in_perturb_q    )
out2c_perturb_zero =model2c.predict(in_perturb_zero )
out2d_perturb_theta=model2d.predict(in_perturb_theta)
out2d_perturb_q    =model2d.predict(in_perturb_q    )
out2d_perturb_zero =model2d.predict(in_perturb_zero )
out2e_perturb_theta=model2e.predict(in_perturb_theta)
out2e_perturb_q    =model2e.predict(in_perturb_q    )
out2e_perturb_zero =model2e.predict(in_perturb_zero )
out2f_perturb_theta=model2f.predict(in_perturb_theta)
out2f_perturb_q    =model2f.predict(in_perturb_q    )
out2f_perturb_zero =model2f.predict(in_perturb_zero )

out2_perturb_theta=(out2a_perturb_theta+out2b_perturb_theta+out2c_perturb_theta+out2d_perturb_theta+out2e_perturb_theta+out2f_perturb_theta)/6.0
out2_perturb_q    =(out2a_perturb_q+out2b_perturb_q+out2c_perturb_q+out2d_perturb_q+out2e_perturb_q+out2f_perturb_q)/6.0
out2_perturb_zero =(out2a_perturb_zero+out2b_perturb_zero+out2c_perturb_zero+out2d_perturb_zero+out2e_perturb_zero+out2f_perturb_zero)/6.0

cmax=delta_theta*0.2

%cmax=delta_theta

if 1==1:
    fig, axs = plt.subplots(2, 2)
    im=axs[0,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(out2_perturb_theta[:, 0:25]-out2_perturb_zero[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,0].set_xlim(1020, 75)
    im=axs[0,0].set_ylim(1020, 75)
    im=axs[0,0].set_aspect('equal', 'box')
    im=axs[0,0].plot([0,1100],[0,1100],'k:')
    im=axs[0,0].set_xlabel('p_in [hPa]')
    im=axs[0,0].set_ylabel('p_out [hPa]')
    im=axs[0,0].set_title('a) ' r'$\theta$' ' response to ' r'$\theta$')
    im=axs[0,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(out2_perturb_theta[:,25:50]-out2_perturb_zero[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,1].set_xlim(1020, 75)
    im=axs[0,1].set_ylim(1020, 75)
    im=axs[0,1].set_aspect('equal', 'box')
    im=axs[0,1].plot([0,1100],[0,1100],'k:')
    im=axs[0,1].set_xlabel('p_in [hPa]')
    im=axs[0,1].set_ylabel('p_out [hPa]')
    im=axs[0,1].set_title('b) q response to ' r'$\theta$')
    im=axs[1,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(out2_perturb_q    [:, 0:25]-out2_perturb_zero[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,0].set_xlim(1020, 75)
    im=axs[1,0].set_ylim(1020, 75)
    im=axs[1,0].set_aspect('equal', 'box')
    im=axs[1,0].plot([0,1100],[0,1100],'k:')
    im=axs[1,0].set_xlabel('p_in [hPa]')
    im=axs[1,0].set_ylabel('p_out [hPa]')
    im=axs[1,0].set_title('c) ' r'$\theta$' ' response to q')
    im=axs[1,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(out2_perturb_q    [:,25:50]-out2_perturb_zero[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,1].set_xlim(1020, 75)
    im=axs[1,1].set_ylim(1020, 75)
    im=axs[1,1].set_aspect('equal', 'box')
    im=axs[1,1].plot([0,1100],[0,1100],'k:')
    im=axs[1,1].set_xlabel('p_in [hPa]')
    im=axs[1,1].set_ylabel('p_out [hPa]')
    im=axs[1,1].set_title('d) q response to q')
    plt.show()

out4a_perturb_theta=model4a.predict(in_perturb_theta)
out4a_perturb_q    =model4a.predict(in_perturb_q    )
out4a_perturb_zero =model4a.predict(in_perturb_zero )
out4b_perturb_theta=model4b.predict(in_perturb_theta)
out4b_perturb_q    =model4b.predict(in_perturb_q    )
out4b_perturb_zero =model4b.predict(in_perturb_zero )
out4c_perturb_theta=model4c.predict(in_perturb_theta)
out4c_perturb_q    =model4c.predict(in_perturb_q    )
out4c_perturb_zero =model4c.predict(in_perturb_zero )
out4d_perturb_theta=model4d.predict(in_perturb_theta)
out4d_perturb_q    =model4d.predict(in_perturb_q    )
out4d_perturb_zero =model4d.predict(in_perturb_zero )
out4e_perturb_theta=model4e.predict(in_perturb_theta)
out4e_perturb_q    =model4e.predict(in_perturb_q    )
out4e_perturb_zero =model4e.predict(in_perturb_zero )
out4f_perturb_theta=model4f.predict(in_perturb_theta)
out4f_perturb_q    =model4f.predict(in_perturb_q    )
out4f_perturb_zero =model4f.predict(in_perturb_zero )

out4_perturb_theta=(out4a_perturb_theta+out4b_perturb_theta+out4c_perturb_theta+out4d_perturb_theta+out4e_perturb_theta+out4f_perturb_theta)/6.0
out4_perturb_q    =(out4a_perturb_q+out4b_perturb_q+out4c_perturb_q+out4d_perturb_q+out4e_perturb_q+out4f_perturb_q)/6.0
out4_perturb_zero =(out4a_perturb_zero+out4b_perturb_zero+out4c_perturb_zero+out4d_perturb_zero+out4e_perturb_zero+out4f_perturb_zero)/6.0

if 1==1:
    fig, axs = plt.subplots(2, 2)
    im=axs[0,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(out4_perturb_theta[:, 0:25]-out4_perturb_zero[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,0].set_xlim(1020, 75)
    im=axs[0,0].set_ylim(1020, 75)
    im=axs[0,0].set_aspect('equal', 'box')
    im=axs[0,0].plot([0,1100],[0,1100],'k:')
    im=axs[0,0].set_xlabel('p_in [hPa]')
    im=axs[0,0].set_ylabel('p_out [hPa]')
    im=axs[0,0].set_title('a) ' r'$\theta$' ' response to ' r'$\theta$')
    im=axs[0,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(out4_perturb_theta[:,25:50]-out4_perturb_zero[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,1].set_xlim(1020, 75)
    im=axs[0,1].set_ylim(1020, 75)
    im=axs[0,1].set_aspect('equal', 'box')
    im=axs[0,1].plot([0,1100],[0,1100],'k:')
    im=axs[0,1].set_xlabel('p_in [hPa]')
    im=axs[0,1].set_ylabel('p_out [hPa]')
    im=axs[0,1].set_title('b) q response to ' r'$\theta$')
    im=axs[1,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(out4_perturb_q    [:, 0:25]-out4_perturb_zero[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,0].set_xlim(1020, 75)
    im=axs[1,0].set_ylim(1020, 75)
    im=axs[1,0].set_aspect('equal', 'box')
    im=axs[1,0].plot([0,1100],[0,1100],'k:')
    im=axs[1,0].set_xlabel('p_in [hPa]')
    im=axs[1,0].set_ylabel('p_out [hPa]')
    im=axs[1,0].set_title('c) ' r'$\theta$' ' response to q')
    im=axs[1,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(out4_perturb_q    [:,25:50]-out4_perturb_zero[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,1].set_xlim(1020, 75)
    im=axs[1,1].set_ylim(1020, 75)
    im=axs[1,1].set_aspect('equal', 'box')
    im=axs[1,1].plot([0,1100],[0,1100],'k:')
    im=axs[1,1].set_xlabel('p_in [hPa]')
    im=axs[1,1].set_ylabel('p_out [hPa]')
    im=axs[1,1].set_title('d) q response to q')
    plt.show()


out6a_perturb_theta=model6a.predict(in_perturb_theta)
out6a_perturb_q    =model6a.predict(in_perturb_q    )
out6a_perturb_zero =model6a.predict(in_perturb_zero )
out6b_perturb_theta=model6b.predict(in_perturb_theta)
out6b_perturb_q    =model6b.predict(in_perturb_q    )
out6b_perturb_zero =model6b.predict(in_perturb_zero )
out6c_perturb_theta=model6c.predict(in_perturb_theta)
out6c_perturb_q    =model6c.predict(in_perturb_q    )
out6c_perturb_zero =model6c.predict(in_perturb_zero )
out6d_perturb_theta=model6d.predict(in_perturb_theta)
out6d_perturb_q    =model6d.predict(in_perturb_q    )
out6d_perturb_zero =model6d.predict(in_perturb_zero )
out6e_perturb_theta=model6e.predict(in_perturb_theta)
out6e_perturb_q    =model6e.predict(in_perturb_q    )
out6e_perturb_zero =model6e.predict(in_perturb_zero )
out6f_perturb_theta=model6f.predict(in_perturb_theta)
out6f_perturb_q    =model6f.predict(in_perturb_q    )
out6f_perturb_zero =model6f.predict(in_perturb_zero )

out6_perturb_theta=(out6a_perturb_theta+out6b_perturb_theta+out6c_perturb_theta+out6d_perturb_theta+out6e_perturb_theta+out6f_perturb_theta)/6.0
out6_perturb_q    =(out6a_perturb_q+out6b_perturb_q+out6c_perturb_q+out6d_perturb_q+out6e_perturb_q+out6f_perturb_q)/6.0
out6_perturb_zero =(out6a_perturb_zero+out6b_perturb_zero+out6c_perturb_zero+out6d_perturb_zero+out6e_perturb_zero+out6f_perturb_zero)/6.0

if 1==1:
    fig, axs = plt.subplots(2, 2)
    im=axs[0,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(out6_perturb_theta[:, 0:25]-out6_perturb_zero[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,0].set_xlim(1020, 75)
    im=axs[0,0].set_ylim(1020, 75)
    im=axs[0,0].set_aspect('equal', 'box')
    im=axs[0,0].plot([0,1100],[0,1100],'k:')
    im=axs[0,0].set_xlabel('p_in [hPa]')
    im=axs[0,0].set_ylabel('p_out [hPa]')
    im=axs[0,0].set_title('a) ' r'$\theta$' ' response to ' r'$\theta$')
    im=axs[0,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(out6_perturb_theta[:,25:50]-out6_perturb_zero[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,1].set_xlim(1020, 75)
    im=axs[0,1].set_ylim(1020, 75)
    im=axs[0,1].set_aspect('equal', 'box')
    im=axs[0,1].plot([0,1100],[0,1100],'k:')
    im=axs[0,1].set_xlabel('p_in [hPa]')
    im=axs[0,1].set_ylabel('p_out [hPa]')
    im=axs[0,1].set_title('b) q response to ' r'$\theta$')
    im=axs[1,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(out6_perturb_q    [:, 0:25]-out6_perturb_zero[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,0].set_xlim(1020, 75)
    im=axs[1,0].set_ylim(1020, 75)
    im=axs[1,0].set_aspect('equal', 'box')
    im=axs[1,0].plot([0,1100],[0,1100],'k:')
    im=axs[1,0].set_xlabel('p_in [hPa]')
    im=axs[1,0].set_ylabel('p_out [hPa]')
    im=axs[1,0].set_title('c) ' r'$\theta$' ' response to q')
    im=axs[1,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(out6_perturb_q    [:,25:50]-out6_perturb_zero[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,1].set_xlim(1020, 75)
    im=axs[1,1].set_ylim(1020, 75)
    im=axs[1,1].set_aspect('equal', 'box')
    im=axs[1,1].plot([0,1100],[0,1100],'k:')
    im=axs[1,1].set_xlabel('p_in [hPa]')
    im=axs[1,1].set_ylabel('p_out [hPa]')
    im=axs[1,1].set_title('d) q response to q')
    plt.show()





delta2_theta=out2_perturb_theta-out2_perturb_zero
delta2_q    =out2_perturb_q    -out2_perturb_zero
delta4_theta=(out4_perturb_theta-out4_perturb_zero)/2
delta4_q    =(out4_perturb_q    -out4_perturb_zero)/2
delta6_theta=(out6_perturb_theta-out6_perturb_zero)/3
delta6_q    =(out6_perturb_q    -out6_perturb_zero)/3

combined_theta=(delta2_theta+delta4_theta+delta6_theta)/3
combined_q=(delta2_q+delta4_q+delta6_q)/3

combined_theta=combined_theta/delta_theta
combined_q=combined_q/delta_q

cmax=0.2

if 1==1:
    fig, axs = plt.subplots(2, 2)
    im=axs[0,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(combined_theta[:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,0].set_xlim(1020, 75)
    im=axs[0,0].set_ylim(1020, 75)
    im=axs[0,0].set_aspect('equal', 'box')
    im=axs[0,0].plot([0,1100],[0,1100],'k:')
    im=axs[0,0].set_xlabel('p_in [hPa]')
    im=axs[0,0].set_ylabel('p_out [hPa]')
    im=axs[0,0].set_title('a) ' r'$\theta$' ' response to ' r'$\theta$')
    im=axs[0,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(combined_theta[:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[0,1].set_xlim(1020, 75)
    im=axs[0,1].set_ylim(1020, 75)
    im=axs[0,1].set_aspect('equal', 'box')
    im=axs[0,1].plot([0,1100],[0,1100],'k:')
    im=axs[0,1].set_xlabel('p_in [hPa]')
    im=axs[0,1].set_ylabel('p_out [hPa]')
    im=axs[0,1].set_title('b) q response to ' r'$\theta$')
    im=axs[1,0].pcolormesh(p_on_lev,p_on_lev,np.transpose(combined_q    [:, 0:25]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,0].set_xlim(1020, 75)
    im=axs[1,0].set_ylim(1020, 75)
    im=axs[1,0].set_aspect('equal', 'box')
    im=axs[1,0].plot([0,1100],[0,1100],'k:')
    im=axs[1,0].set_xlabel('p_in [hPa]')
    im=axs[1,0].set_ylabel('p_out [hPa]')
    im=axs[1,0].set_title('c) ' r'$\theta$' ' response to q')
    im=axs[1,1].pcolormesh(p_on_lev,p_on_lev,np.transpose(combined_q    [:,25:50]),vmin=-cmax,vmax=cmax,cmap=cmap)
    im=axs[1,1].set_xlim(1020, 75)
    im=axs[1,1].set_ylim(1020, 75)
    im=axs[1,1].set_aspect('equal', 'box')
    im=axs[1,1].plot([0,1100],[0,1100],'k:')
    im=axs[1,1].set_xlabel('p_in [hPa]')
    im=axs[1,1].set_ylabel('p_out [hPa]')
    im=axs[1,1].set_title('d) q response to q')
    plt.show()

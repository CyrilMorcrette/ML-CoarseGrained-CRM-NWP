#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
#
# Use "module load scitools/experimental-current" at the command line before running this.
##################################################
# Prior to running this, need to use:
# calc_coarse_grained_mean_profiles.py 
#    to calculate profiles of max/min/centiles for each of the 64 subdomain (but sampling all 99 regions and a whole month).
# calc_coarse_grained_supermean_profiles.py
#    to mean/max/min across the 64 subdomain and hence find profiles to use for rescaling.
##################################################
# After running this use:
# eval_deep_learning_recursive_addADV_next_MASTERCOPY.py
#    to apply the ML algo recursively and compare to climatology and persistence.
##################################################

# Import some modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from netCDF4 import Dataset
from numpy import loadtxt

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
from keras.layers import Dropout
from keras.constraints import maxnorm

from scipy import stats

from keras.callbacks import LearningRateScheduler

from ml_cjm_functions import plot_centiles
from ml_cjm_functions import process_series_of_domains
from ml_cjm_functions import plot_in_and_out

# The line below gets modified by sed command to become
#from ml_namelist_MASTERCOPY import setup_parameters
from ml_namelist import setup_parameters

# Start of functions

def my_learning_rate(superloop):
    setup=setup_parameters()
    initial_learning_rate = setup['initial_learning_rate']
    # Multiply the learning rate by decay_rate every decay_step superloops.
    decay_rate=setup['decay_rate']
    decay_steps=setup['decay_steps']
    lrate=initial_learning_rate * decay_rate ** np.floor(float(superloop) / decay_steps)
    return lrate;

def myprintout(datain):
    print('checking learning_rate',datain)
    outcome=1
    return outcome;

# End of functions

def main():

    # Read in namelist type information
    setup=setup_parameters()

    # Print out to check that what was run was what it should have been
    print('n_nodes=',setup['n_nodes'])
    print('n_layers=',setup['n_layers'])
    print('my_alpha=',setup['my_alpha'])
    print('minibatch=',setup['minibatch'])
    print('n_superloops=',setup['n_superloops'])
    print('initial_learning_rate=',setup['initial_learning_rate'])
    print('months_to_train_on=',setup['months_to_train_on'])
    print('shift=',setup['shift'])
    print('include_flux_deriv=',setup['include_flux_deriv'])
    print('include_cos_and_sin_terms=',setup['include_cos_and_sin_terms'])
    print('what_to_predict=',setup['what_to_predict'])
    print('i_flag_add_adv=',setup['i_flag_add_adv'])

    # I think last layer should be tanh so that we get values from -1 to 1
    # Hence ensure the thing we want to predict has been normalised to range -1 to 1
    #
    # Define the keras model
    n_nodes=setup['n_nodes']
    n_layers=setup['n_layers']

    my_alpha=setup['my_alpha']
    minibatch=setup['minibatch']
    n_superloops=setup['n_superloops']

    model = Sequential()

    #    model.add(Dense(n_nodes, input_dim=110, kernel_constraint=maxnorm(setup['my_max_norm'])))
    #    model.add(LeakyReLU(alpha=my_alpha))
    #    model.add(Dropout(setup['dropout_rate']))

    model.add(Dense(n_nodes, input_dim=110))
    model.add(LeakyReLU(alpha=my_alpha))

    for internal_layers in np.arange(1,n_layers,1):
        #        model.add(Dense(n_nodes, kernel_constraint=maxnorm(setup['my_max_norm'])))
        #        model.add(LeakyReLU(alpha=my_alpha))
        #        model.add(Dropout(setup['dropout_rate']))
        model.add(Dense(n_nodes))
        model.add(LeakyReLU(alpha=my_alpha))

    model.add(Dense(51, activation='tanh'))

    initial_learning_rate = my_learning_rate(0.0)

    opt = Adam(learning_rate=initial_learning_rate)

    #opt = SGD(lr=initial_learning_rate, momentum=0.9)

    model.summary()
    #plot_model(model, 'model.png', show_shapes=True)
    model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Write things out
    model_json=model.to_json()
    with open("ml_model.json", "w") as json_file:
        json_file.write(model_json)
        json_file.close()
    #
    months_to_do=['20151231-20160131','20160331-20160430','20160630-20160731','20160930-20161031','20161130-20161231','20160229-20160331']
    monthyearpaths=['JAN2016','APR2016','JUL2016','OCT2016','DEC2016','MAR2016']
    days_to_do_in_each_month=[31,30,31,31,31,31]
    # Only use 5 out of the 6 months (k-fold)
    tmp_n=np.size(setup['months_to_train_on'])
    tmp_months_to_do=months_to_do[setup['months_to_train_on'][0]]
    tmp_monthyearpaths=monthyearpaths[setup['months_to_train_on'][0]]
    tmp_days_to_do_in_each_month=days_to_do_in_each_month[setup['months_to_train_on'][0]]
    for mm in np.arange(1,tmp_n,1):
        tmp_months_to_do=np.append(tmp_months_to_do,months_to_do[setup['months_to_train_on'][mm]])
        tmp_monthyearpaths=np.append(tmp_monthyearpaths,monthyearpaths[setup['months_to_train_on'][mm]])
        tmp_days_to_do_in_each_month=np.append(tmp_days_to_do_in_each_month,days_to_do_in_each_month[setup['months_to_train_on'][mm]])
    months_to_do=tmp_months_to_do
    monthyearpaths=tmp_monthyearpaths
    days_to_do_in_each_month=tmp_days_to_do_in_each_month
    print('Will train using:',monthyearpaths)
    #
    try_a_restart=0
    if try_a_restart==1:
        filein='ml_for_restarting.txt'
        restart_data=np.loadtxt(filein)
        start_loop=int(restart_data[0])
        model.load_weights('ml_lastest_saved_weights.h5')
        # Read in arrays to store the rmse as one loops through multiple sets of multiples epochs.
        filein='ml_timeseries_rmse_train.txt'
        keep_rmse_train=np.loadtxt(filein)
        filein='ml_timeseries_rmse_test.txt'
        keep_rmse_test=np.loadtxt(filein)
        filein='ml_timeseries_lrate.txt'
        keep_lrate=np.loadtxt(filein)
    else:
        start_loop=0
        start_offset=1
        start_month=0
        # Set up arrays to store the rmse as one loops through multiple sets of multiples epochs.
        keep_rmse_train=np.empty([1,1])+np.nan
        keep_rmse_test =np.empty([1,1])+np.nan
        keep_lrate=np.empty([1,1])+np.nan
    # end if
    for superloop in np.arange(start_loop,n_superloops,1):
        check_lrs = my_learning_rate(superloop)
        print('check_lrs=',check_lrs)
        if try_a_restart==1 and superloop == start_loop:
            start_offset=int(restart_data[1])
        else:
            tmp=setup['shift']-1
            start_offset=(tmp-3*(tmp//3))+1
        for offset in np.arange(start_offset,64+1,setup['subdomain_skip']):
            # Consider each of the 64 sub-domains one at a time.
            if try_a_restart==1 and offset == start_offset:
                start_month=int(restart_data[2])
            else:
                start_month=0
            #            
            # Try shuffling months
            #
            tmp_a=np.arange(0,len(months_to_do),1)
            tmp_b=np.random.choice(tmp_a,size=len(months_to_do),replace=False)
            for month in tmp_b:
            #for month in np.arange(start_month,len(months_to_do),1):
            #for month in setup['months_to_train_on']:
                date_range=months_to_do[month]
                monthyear=monthyearpaths[month]
                ndays=days_to_do_in_each_month[month]
                print('Superloop=',superloop,'Sub-domain',offset,'Using data for',ndays,'days in',monthyear,'from',date_range)
                # Can not go through all 64 subdomains in one go, but could do several at once.
                # e.g. domains_to_do=np.arange(3,64,3)+offset
                domains_to_do=[0]+offset
                i_flag_add_adv=setup['i_flag_add_adv']
                processed=process_series_of_domains(date_range,setup,setup['regions_to_train_on'],domains_to_do,i_flag_add_adv,monthyear,ndays)
                if offset==99:
                    # Plot some profiles of min/max, 25/50/75 centiles
                    outcome=plot_centiles(processed['output_alldom'],setup)
                    cmap = plt.get_cmap('bwr')
                    plt.pcolormesh(processed['output_alldom'][0:3000,:],vmin=-1,vmax=1,cmap=cmap)
                    plt.show()
                    plt.plot(processed['output_alldom'][0:100,:])
                    plt.show()
                # Copy ordered data into array called shuffled, N.B. it is not shuffled yet.
                shuffled_big_data=processed['output_alldom']
                for repetitions in np.arange(0,1,1):
                    print('Shuffling data for the ',str(repetitions+1),' time.')
                    # This shuffles the original data on the first pass 
                    # and reshuffles the already shuffled data on subsequent passes.
                    shuffled_big_data=np.random.permutation(shuffled_big_data)
                    # Must shuffle array and then take X and y from it 
                    # (rather than take X and Y and then shuffle) to make sure things line up.
                    #
                    # Take the input from the first portion of the array and the ouput from the latter portion.
                    X=shuffled_big_data[:,0:110]
                    y=shuffled_big_data[:,110:161]
                    if 1==2:
                        outcome=plot_in_and_out(processed['output_alldom'],shuffled_big_data)
                    # 
                    n_samples,tmp=X.shape
                    print(n_samples,tmp)
                    # Separate data into a training set and a test set.
                    n_train=int(setup['train_fraction']*n_samples)
                    trainX, testX = X[:n_train, :], X[n_train:, :]
                    trainy, testy = y[:n_train], y[n_train:]
                    print('n_train=',n_train)
                    # For restarts
                    fileout='ml_for_restarting.txt'
                    dataout=[superloop,offset,month]
                    np.savetxt(fileout, np.ones((1,1))*dataout, fmt='%10.7f')
                    # Fit the Keras model on the dataset
                    # There are 12 2-hour timesteps per day and 99 regions (12*99=1188), so:
                    #   a batch size of order(  100) means roughly  1         point per region.
                    #   a batch size of order( 1000) means roughly  1 diurnal cycle per region.
                    #   a batch size of order(10000) means roughly 10 diurnal cycle per region.
                    lrate = LearningRateScheduler(my_learning_rate)
                    history=model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1, batch_size=minibatch, callbacks=[lrate])
                    #
                    #keep_history_train=np.append(keep_history_train,history.history['loss'])
                    #keep_history_test=np.append(keep_history_test,history.history['val_loss'])
                    #keep_history_lrate=np.append(keep_history_lrate,lrate)
                    keep_rmse_train=np.append(keep_rmse_train,history.history['root_mean_squared_error'])
                    keep_rmse_test =np.append(keep_rmse_test,history.history['val_root_mean_squared_error'])
                    keep_lrate=np.append(keep_lrate,check_lrs)
                    fileout='ml_timeseries_rmse_train.txt'
                    np.savetxt(fileout, np.ones((1,1))*keep_rmse_train, fmt='%10.7f')
                    fileout='ml_timeseries_rmse_test.txt'
                    np.savetxt(fileout, np.ones((1,1))*keep_rmse_test, fmt='%10.7f')
                    fileout='ml_timeseries_lrate.txt'
                    np.savetxt(fileout, np.ones((1,1))*keep_lrate, fmt='%10.7f')
                    # Save weights after every pass
                    # model.save_weights("ml_lastest_saved_weights.h5")
                # end repetitions (so have gone through this subset of the data many times).
            # end month
        # end offset (have now gone through all 64 subdomains).
        model.save_weights("ml_lastest_saved_weights.h5")
    # end superloop

    if 1==2:
        # plot loss during training
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(history.history['loss'], label='train')
        axs[0, 0].plot(history.history['val_loss'], label='test')
        axs[0, 0].legend()
        axs[0, 1].plot(keep_history_train, label='train')
        axs[0, 1].plot(keep_history_test, label='test')
        axs[0, 1].legend()
        axs[1, 0].plot(keep_history_lrate)
        plt.show()

    print('All training completed successfully!')

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()



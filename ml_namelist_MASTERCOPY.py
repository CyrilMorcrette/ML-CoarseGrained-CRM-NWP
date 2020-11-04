#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
import numpy as np

# This first function is effectively a namelist.

def setup_parameters():
    # A place to define parameters etc. A namelist of sorts.
    #
    # For defining the neural network
    n_nodes=256
    n_layers=9
    my_alpha=0.3
    # For training
    minibatch=1000
    n_superloops=20
    # Learning rate
    initial_learning_rate = 1.0e-5
    # Multiply the learning rate by decay_rate every decay_step superloops.
    decay_rate=1.0
    decay_steps=100.0
    #
    dropout_rate=0.0
    my_max_norm=2.0
    months_to_train_on=[0,1,2,3,4,5]
    #
    # Whether to consider data 1 2-hour window away or 2(4h), 3(6h) etc
    shift=1
    # Wheter to consider all 64 subdomains (skip=1) or every other one (skip=2) to speed things up a bit.
    subdomain_skip=1
    #
    tsperday=12
    #
    flag_theta_now=10
    flag_qt_now=10
    #
    flag_theta_next=10
    flag_qt_next=10
    #
    flag_delta_theta=10
    flag_delta_qt=10
    #
    train_fraction=0.7
    clip_limit=1
    # All regions
    regions_all=np.arange(1,99+1)-1
    # All regions apart from 6 and 67 and 74
    regions_to_train_on=np.append(np.append(np.append(np.arange(1,6),np.arange(7,67)),np.arange(68,74)),np.arange(75,100))-1
    # Just regions 6 and 74
    regions_to_eval_on=np.append(6,74)-1
    regions_to_eval_on=[6-1]
    regions_to_eval_on=[74-1]
    # If data is normaly distributed, 95% will lay within +/- 2 sigma. Hence clip data below and above (100-95)/2.
    #    clip_centile=2.5
    # If data is normaly distributed, 99.7% will lay within +/- 3 sigma. Hence clip data below and above (100-99.7)/2.
    clip_centile=0.15
    # Flag to smooth the advection increments 0=off 1=on
    smooth_adv=0
    # Only consider data up to level...
    nz_trim=50
    # Sub-sample data in the vertical. Only consider data every k_step level.
    k_step=2
    k_indices=np.arange(1,nz_trim,k_step)
    #
    include_flux_deriv=0
    include_cos_and_sin_terms=0
    ppn_option=0
    what_to_predict=0
    i_flag_add_adv=1
    return {'tsperday':tsperday, 'flag_theta_now':flag_theta_now, 'flag_qt_now':flag_qt_now,'flag_theta_next':flag_theta_next, 'flag_qt_next':flag_qt_next, 'flag_delta_theta':flag_delta_theta, 'flag_delta_qt':flag_delta_qt, 'train_fraction':train_fraction, 'clip_limit':clip_limit,'regions_to_train_on':regions_to_train_on, 'regions_to_eval_on':regions_to_eval_on,'regions_all':regions_all, 'clip_centile':clip_centile, 'smooth_adv':smooth_adv, 'nz_trim':nz_trim, 'k_step':k_step, 'k_indices':k_indices, 'initial_learning_rate':initial_learning_rate, 'decay_rate':decay_rate, 'decay_steps':decay_steps, 'n_nodes':n_nodes, 'n_layers':n_layers, 'my_alpha':my_alpha, 'minibatch':minibatch, 'n_superloops':n_superloops, 'shift':shift, 'subdomain_skip':subdomain_skip, 'include_flux_deriv':include_flux_deriv, 'include_cos_and_sin_terms':include_cos_and_sin_terms, 'ppn_option':ppn_option, 'dropout_rate':dropout_rate, 'my_max_norm':my_max_norm, 'months_to_train_on':months_to_train_on, 'what_to_predict':what_to_predict, 'i_flag_add_adv':i_flag_add_adv};

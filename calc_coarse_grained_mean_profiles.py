#!/usr/bin/env python	
#
# Cyril Morcrette (2020), Met Office, UK
#
# For use in averaging profiles for use in normalisation/standardization
#
# Use "module load scitools/experimental-current" at the command line before running this.

import numpy as np

from ml_cjm_functions_MASTERCOPY import read_in_for_one_subdomain
from ml_cjm_functions_MASTERCOPY import calc_profiles_max_min_range
from ml_cjm_functions_MASTERCOPY import calc_profiles_mean_and_std
from ml_cjm_functions_MASTERCOPY import calc_profiles_centiles
from ml_cjm_functions_MASTERCOPY import write_out_profiles
from ml_cjm_functions_MASTERCOPY import setup_parameters

setup=setup_parameters()

months_to_do=['20151231-20160131']
ndays=31
monthyear='JAN2016'

for month in np.arange(0,len(months_to_do),1):
    date_range=months_to_do[month]
    for offset in np.arange(0,-4,-1):
        domains_to_do=np.arange(4,64+1,4)+offset
        for d in np.arange(0,len(domains_to_do)):
            domain        = domains_to_do[d]
            data          = read_in_for_one_subdomain(domain,date_range,setup['regions_all'],setup,monthyear,ndays)
            range_prof    = calc_profiles_max_min_range(data)
            mean_prof     = calc_profiles_mean_and_std(data)
            centiles_prof = calc_profiles_centiles(data,setup)
            outcome       = write_out_profiles(range_prof,mean_prof,centiles_prof,domain,date_range)
        #end
    #end
#end


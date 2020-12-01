#!/usr/bin/env python	
#
# Cyril Morcrette (2019), Met Office, UK
#
# Use "module load scitools/default-current" at the command line before running this.

#
# Use  retrieve_ml_global.py                              Python
# Then extract_ml_lamlike_region_from_global.py           Python
# Then new_area_ml_global_model_for_continuous_curtains.m Matlab
#

# Import some modules

import subprocess
import numpy as np
from datetime import timedelta, date

# Define some functions

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def generate_filename_in(date,vt,ext,stream,analysis_time):
    vtstr=str(vt)
    if vt < 100:
        vtstr='0'+str(vt) 
    if vt < 10:
        vtstr='00'+str(vt)
    filename=date.strftime("%Y%m%d")+analysis_time+'_glm_p_GLOBE_'+stream+'_'+vtstr+ext
    print(filename)
    #
    return filename;

def retrieve_a_file(date,vt,roseid,analysis_time,stream):
    # Currently hard-coded for user "frme"
    tmppath='/scratch/frme/ML/'+roseid+'/step1_um_output/'
    moopath='moose:/devfc/'+roseid+'/field.pp/'
    filename=generate_filename_in(date,vt,'.pp',stream,analysis_time)
    # moo retrieval will not overwrite an existing file.
    # So remove any existing occurrence of the file we are trying to retrieve
    stalefile=tmppath+filename
    subprocess.call(["rm", stalefile])
    fullname=moopath+filename
    subprocess.call(["moo","get", fullname, tmppath])
    outcome=1
    return outcome;

# End of functions

########################################################
# Next few lines are things you will need to change
########################################################
roseid='u-bx951'

# Probably worth ignoring the first 24 hours of the simulation
# so set start_date to 1 day after start of the actual run.
start_date = date(2017, 5, 31)
end_date   = date(2017, 6, 30)

###############################################################
# Below here are things you probably will not need to change.
###############################################################
# There is a new output file every hour.
# The timestamps on the files is T000 to T011.
# The simulation has a new analysis coming in every 12 hours
# So for every day there is a T0000Z and a T1200Z and for each there is T000 to T011.
# For each of those times there are 3 files containing:
#   a) lots of 2d fields (surface and radiative fluxes, screen temperature, precip, MSLP etc)
#   b) 3d fields of temperature, qv, qcl, qcf, cfl, cff, bcf, qrain
#   c) 3d field of advective increments to b).

list_analysis_time=['T0000Z']
list_stream=['a','b','c']
list_stream=['b','c']
list_stream=['b']

for single_date in daterange(start_date, end_date):
    for ana_time in np.arange(0,len(list_analysis_time),1):
        # Some of the runs start at 00Z, some at 12Z
        analysis_time=list_analysis_time[ana_time]
        for vt in np.arange(0,120,2):
            for stream_number in np.arange(0,len(list_stream),1):
                # Too much data to all be in one file (even for a given hourly timestamp)
                # so data has been split up into several "streams".
                stream=list_stream[stream_number]
                outcome=retrieve_a_file(single_date,vt,roseid,analysis_time,stream)



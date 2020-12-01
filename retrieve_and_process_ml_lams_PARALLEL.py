#!/usr/bin/env python	
#
# Cyril Morcrette (2019), Met Office, UK
#
# Use "module load scitools/default-current" at the command line before running this.

# Import some modules
import subprocess
import numpy as np
from datetime import date
import iris
import iris.analysis

# Import some of my own functions
from cjm_functions import get_lat_lon_hecto_nest
from cjm_functions import daterange
from cjm_functions import retrieve_all_files_for_a_day
from cjm_functions import name_of_hecto_lam
from cjm_functions import extract_fields_for_advective_tendencies
from cjm_functions import process_ml_lam_file
from cjm_functions import all_ml_processing_for_a_region

def main():

    ########################################################
    # Next few lines are things you will need to change
    ########################################################
    roseid='u-bw210'
    #roseid='u-bv805'
    #roseid='u-bw155'

    analysis_time='T0000Z'

    name_str='ra1m'

    # The name of all the domains needs to be added here. May be easiest to just type these out by hand. 
    # Or could try linking to a file stored somewhere with the full list.
    result=get_lat_lon_hecto_nest('/home/h01/frme/ml_lams_latlon_aqua_only.dat')
    reg_lat=result['reg_lat']
    reg_lon=result['reg_lon']

    ###############################################################
    # Below here are things you probably will not need to change.
    ###############################################################
    # There is a new output file every 2 hours.
    # The timestamps on the files is T000 to T022.
    # The simulation has a new analysis coming in every 24 hours
    # So for every day there is a T0000Z is T000, T002, T004 ... to T022.
    # The T000 file contains 02Z data, the T002 file contains 04Z data etc.
    # The T000 file also contains 00Z data.
    # For each of those times there are 5 files containing:
    #   a) lots of 2d fields (surface and radiative fluxes, screen temperature, MSLP etc)
    #   b) 2d fields of precip, cloud, and total column moisture
    #   c) 3d fields of theta, qv, qcl, qcf, cfl, cff, bcf, qrain, qgraupel
    #   d) 3d field of advective increments to c).
    #   e) 3d fields of u, v, and w
    #   f) 3d fields of density and pressure.
    # A combination of c, e and f are used to calculate advective tendencies.
    #
    # List of the surface and TOA fields to extract, coarse-grain and write out (these 2d fields are technically 3d because of time dimension)
    #
    # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 3 of these variables.
    # 1 207 Incoming SW Rad Flux (TOA)
    # 1 208 Outgoing SW Rad Flux (TOA)
    # 1 235 Total Downward surface SW flux
    # 2 205 Outgoing LW rad (TOA)
    # 3 217 Surface Sensible Heat Flux
    # 3 234 Surface Latent Heat Flux
    # 4 203 Large-scale rainfall rate
    # 4 204 Large-scale snowfall rate
    # 9 217 Total Cloud Amount
    list_stash_sec= [  1,  1,  1,  2,  3,  3,  4,  4,  9]
    list_stash_code=[207,208,235,205,217,234,203,204,217]
    list_stream=    ['a','a','a','a','a','a','b','b','b']
    #
    # List of 3d fields to extract, coarse-grain and write out (these 3d fields are technically 4d because of time dimension)
    #
    # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 3 of these variables.
    # 0 266 Bulk Cloud Fraction
    list_stash_sec_3d =[  0]
    list_stash_code_3d=[266]
    list_stream_3d    =['c']

    # Although we will be ignoring the first 24 hours of the simulation
    # set start_date as start of simulation to have data at 00Z on 1 Jan
    start_date = date(2017,  6, 23)
    end_date   = date(2017,  6, 30)

    for single_date in daterange(start_date, end_date):

        outcome=retrieve_all_files_for_a_day(single_date,roseid,1)

        inargs = [dict(zip(['single_date','roseid','name_str','analysis_time','region_number','list_stream_3d','list_stash_sec_3d','list_stash_code_3d','list_stream','list_stash_sec','list_stash_code','reg_lat','reg_lon'], [single_date,roseid,name_str,analysis_time,region_number,list_stream_3d,list_stash_sec_3d,list_stash_code_3d,list_stream,list_stash_sec,list_stash_code,reg_lat,reg_lon])) for region_number in np.arange(0,len(reg_lat),1)]

        parallelise(MainWrapper, processes=6)(inargs) # Parallel

        outcome=retrieve_all_files_for_a_day(single_date,roseid,0)
    
def MainWrapper(inarg):
    all_ml_processing_for_a_region(**inarg)

#-----------------------------------------------------------------------------
def parallelise(f, processes=None):
    """
     Wrapper to parallelise any function
     Example:
     somefunc_star_parallel = parallelise(somefunc_star_linear)
     inargs = itertools.izip(np.arange(N), itertools.repeat(2))
     parallel_result = somefunc_star_parallel(inargs)
     """
    import multiprocessing as mp
    if processes is None: processes = max(1, mp.cpu_count() - 1)
    if processes <= 0: processes = 1
    def easy_parallise(f, sequence):
        pool = mp.Pool(processes=processes)
        result = pool.map_async(f, sequence).get()
        pool.close()
        pool.join()
        return result
    from functools import partial
    return partial(easy_parallise, f)

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    main()


    
    
    
    
    
    

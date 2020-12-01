#!/usr/bin/env python	
#
# Cyril Morcrette (2019), Met Office, UK
#
# Use "module load scitools/default-current" at the command line before running this.

# Python
# Use  retrieve_ml_global.py
# Then extract_ml_lamlike_region_from_global.py
# Matlab
# Then new_area_ml_global_model_for_continuous_curtains.m
#

# Import some modules

import subprocess
import numpy as np
from datetime import timedelta, date
from cjm_functions import make_stash_string
import iris
import iris.quickplot
import iris.analysis
import matplotlib.pyplot as plt
from iris.analysis.cartography import cosine_latitude_weights

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def generate_filename_in(single_date,vt,ext,stream,analysis_time):
    vtstr=str(vt)
    if vt < 100:
        vtstr='0'+str(vt) 
    if vt < 10:
        vtstr='00'+str(vt)
    filename='step1_um_output/'+single_date.strftime("%Y%m%d")+analysis_time+'_glm_p_GLOBE_'+stream+'_'+vtstr+ext
    print(filename)
    #
    return filename;

def generate_filename_out(date,vt,ext,analysis_time,region,stashnumber):
    vtstr=str(vt)
    if vt < 100:
        vtstr='0'+str(vt) 
    if vt < 10:
        vtstr='00'+str(vt)
    filename='step2_individual_regions/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_glm_'+vtstr+'_'+stashnumber+ext
    print(filename)
    #
    return filename;

def read_a_file(date,vt,roseid,analysis_time,stream,region,lat,lon,stash_sec,stash_code):
    # Currently hard-coded for user "frme"
    tmppath='/scratch/frme/ML/'+roseid+'/'
    filename=generate_filename_in(date,vt,'.pp',stream,analysis_time)
    filein=tmppath+filename
    # Read in data
    result = make_stash_string(stash_sec,stash_code)
    fieldin = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    # There will be several time instances all will get processed and all written out.
    #
    # LAM domains are 360x360, but we are only interested in central 240x240
    # LAMs use 1.5 km gridlength. So domain of interest is 240x1.5 i.e. 360 km x 360 km.
    # So we want to extract +/- 180 km around the central latitude.
    # Given a degree of latitude is ~111km, extract 2 degrees
    dlat=2.0
    latmin=lat-dlat
    latmax=lat+dlat
    # Find what one degree of longitude is in km at this latitude.
    #    dx_north=2.0*np.pi*re*np.cos(latmax*np.pi/180)/360
    #    dx_south=2.0*np.pi*re*np.cos(latmin*np.pi/180)/360
    #    dx=np.amin([[dx_north,dx_south]])
    # Hence 180 km means extracting +/- dlon
    # Here we extract a rectangle, but use the pole-most latitude to calculate size of region to extract.
    #
    # The pole-most latitude we are to consider is around 82N (and S), 
    # where extracting a 180km region at that latitude
    # means going 12 degrees of longitude west and east.
    dlon=12.0
    lonmin=lon-dlon
    lonmax=lon+dlon
    #
    subregion=fieldin.intersection(latitude=(latmin,latmax),longitude=(lonmin,lonmax))
    #
    filenameout=generate_filename_out(date,vt,'.nc',analysis_time,region,result['stashstr_fout'])
    fileout=tmppath+filenameout
    print(fileout)
    iris.save(subregion, fileout)
    outcome=1
    return outcome;

#############################

roseid='u-bl012'
roseid='u-bx951'

#list_analysis_time=['T0000Z','T1200Z']
list_analysis_time=['T0000Z']

list_stash_sec=[0,0,0,0,0]
list_stash_code=[4,10,254,12,272]
list_stream=['b','b','b','b','b']

start_date = date(2017, 5, 31)
end_date   = date(2017, 6, 30)

# Read in the coords written out by maps_nesting_domains_aqua_only.m
reg_lat=np.empty(0, int)
reg_lon=np.empty(0, int)
with open('/home/h01/frme/ml_lams_latlon_aqua_only.dat', 'r') as filestream:
    print(filestream)
    for line in filestream:
        currentline = line.split(",")
        reg_lat=np.append(reg_lat,int(currentline[1]))
        reg_lon=np.append(reg_lon,int(currentline[2]))
# End definition of lat/lon for all the regions.

#Having done that, actually just extract the Gulf of Guinea domain at (0N, 0E) (domain number 6-1)
#reg_lat=[0]
#reg_lon=[0]

#Having done that, actually just extract the Azores domain at (40N, 25W) (domain number 74-1)
#reg_lat=[40]
#reg_lon=[-25]

#Having done that, actually just extract a different South Atlantic domain at (40S, 0E) (domain number 67-1)
reg_lat=[-40]
reg_lon=[0]

for single_date in daterange(start_date, end_date):
    for ana_time in np.arange(0,1,1):
        # All runs now start at 00Z (but it used to be that some of the runs start at 00Z, some at 12Z).
        analysis_time=list_analysis_time[ana_time]
        for vt in np.arange(0,120,2):
            # vt=validity time (i.e. T+?) range from T+0 to T+11 for sims that have analyses at 00Z and 12Z.
            for region_number in np.arange(0,len(reg_lat),1):
                # Loop over the various LAMs (this could be all 98 of them).
                if reg_lat[region_number]>=0:
                    lat_letter='N'
                else:
                    lat_letter='S'
                if reg_lon[region_number]<0:
                    lon_letter='W'
                else:
                    lon_letter='E'
                region=str(np.abs(reg_lat[region_number]))+lat_letter+str(np.abs(reg_lon[region_number]))+lon_letter
                for stream_number in np.arange(0,len(list_stream),1):
                    # Too much data to all be in one file (even for a given hourly timestamp)
                    # so data has been split up into several "streams".
                    stash_sec=list_stash_sec[stream_number]
                    stash_code=list_stash_code[stream_number]
                    stream=list_stream[stream_number]
                    outcome=read_a_file(single_date,vt,roseid,analysis_time,stream,region,reg_lat[region_number],reg_lon[region_number],stash_sec,stash_code)



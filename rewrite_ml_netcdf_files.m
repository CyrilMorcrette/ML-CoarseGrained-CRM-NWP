clear

load_L70_80km

names_out_2d={'toa_incoming_shortwave_flux','surface_upward_sensible_heat_flux','surface_upward_latent_heat_flux','stratiform_rainfall_flux','stratiform_snowfall_flux'};
names_out_2d_NEW={'toa_incoming_shortwave_flux','surface_upward_sensible_heat_flux','surface_upward_latent_heat_flux','stratiform_rainfall_flux','stratiform_snowfall_flux'};
names_out_2d_UNITS={'W m-2','W m-2','W m-2','kg m-2 s-1','kg m-2 s-1'};

names_out_3d={'liquid_ice_static_potential_temperature_K','total_humidity_qvqclqcfqrainqgraupel_kgkg-1','adv_flux_thetali_Ks-1','adv_flux_qtotal_kgkg-1s-1'};
names_out_3d_NEW={'air_potential_temperature','specific_humidity','net_flux_air_potential_temperature','net_flux_specific_humidity'};
names_out_3d_LONG={'liquid_ice_static_potential_temperature','total_specific_humidity','net_advective_flux_liquid_ice_static_potential_temperature','net_advective_flux_total_specific_humidity'};
names_out_3d_UNITS={'K','kg kg-1','K s-1','kg kg-1 s-1'}

months={'APR2016','DEC2016','JAN2016','JUL2016','JUN2017','MAR2016','OCT2016'}
datestr={'20160331-20160430','20161130-20161231','20151231-20160131','20160630-20160731','20170531-20170630','20160229-20160331','20160930-20161031'}

nregion=99;
nsubdomain=64;

for mm=1:length(months)
  for dd=1:nsubdomain
    filein=['/data/nwp1/frme/ML/',months{mm},'/',datestr{mm},'-ml_aggregate_data-subdomain',num2str(dd),'.nc']
    fileout=['/data/nwp1/frme/for_BADC/',months{mm},'/',datestr{mm},'-ml_aggregate_data-subdomain',num2str(dd),'.nc']
    cmd=['touch ',fileout];
    unix(cmd);
    % Remove pre-existing file
    cmd=['rm ',fileout];
    unix(cmd);
    %
    for ff=1:length(names_out_3d)
      data=ncread(filein,names_out_3d{ff});
      [nx,ny,nz,nt]=size(data);
      nccreate(fileout,names_out_3d_NEW{ff},'Dimensions',{'Region',nregion,'Subdomain',1,'Height',nz,'Time',nt},'ChunkSize',[nregion,1,nz,nt],'FillValue',-999)
      ncwrite(fileout,names_out_3d_NEW{ff},data,[1,1,1,1])
      ncwriteatt(fileout,names_out_3d_NEW{ff},'units',names_out_3d_UNITS{ff})
      ncwriteatt(fileout,names_out_3d_NEW{ff},'long_name',names_out_3d_LONG{ff})
    end
    %
    for ff=1:length(names_out_2d)
      data=ncread(filein,names_out_2d{ff});
      [nx,ny,tmp,nt]=size(data);
      nccreate(fileout,names_out_2d_NEW{ff},'Dimensions',{'Region',nregion,'Subdomain',1,'Height',nz,'Time',nt},'ChunkSize',[nregion,1,1,nt],'FillValue',-999)
      ncwrite(fileout,names_out_2d_NEW{ff},data,[1,1,1,1])
      ncwriteatt(fileout,names_out_2d_NEW{ff},'units',names_out_2d_UNITS{ff})
    end

    field='LatRegionCentreDeg';
    data=ncread(filein,field);
    nccreate(fileout,field,'Dimensions',{'Region',nregion},'ChunkSize',[nregion],'FillValue',-999)
    ncwrite(fileout,field,data,[1])
    ncwriteatt(fileout,field,'units','degrees')

    field='LonRegionCentreDeg';
    data=ncread(filein,field);
    nccreate(fileout,field,'Dimensions',{'Region',nregion},'ChunkSize',[nregion],'FillValue',-999)
    ncwrite(fileout,field,data,[1])
    ncwriteatt(fileout,field,'units','degrees')

    field='DeltaXkmSubDomainWRTRegionCentre';
    data=ncread(filein,field);
    nccreate(fileout,field,'Dimensions',{'Subdomains',nsubdomain},'ChunkSize',[nsubdomain],'FillValue',-999)
    ncwrite(fileout,field,data,[1])
    ncwriteatt(fileout,field,'units','km')

    field='DeltaYkmSubDomainWRTRegionCentre';
    data=ncread(filein,field);
    nccreate(fileout,field,'Dimensions',{'Subdomains',nsubdomain},'ChunkSize',[nsubdomain],'FillValue',-999)
    ncwrite(fileout,field,data,[1])
    ncwriteatt(fileout,field,'units','km')

    field='HoursSince2016_01_01_00Z';
    data=ncread(filein,field);
    [nt,tmp]=size(data);
    nccreate(fileout,field,'Dimensions',{'Time',nt},'ChunkSize',[nt],'FillValue',-999)
    ncwrite(fileout,field,data,[1])
    ncwriteatt(fileout,field,'units','hours')

    field='Height';
    data=height_theta_levels;
    [tmp,nz]=size(data);
    nccreate(fileout,field,'Dimensions',{'Height',nz},'ChunkSize',[nz],'FillValue',-999)
    ncwrite(fileout,field,data,[1])
    ncwriteatt(fileout,field,'units','m')

    ncwriteatt(fileout,'/','Conventions','CF-1.0');
    ncwriteatt(fileout,'/','title','1.5km convection-permitting model data coarse-grained to 45 km scale');
    ncwriteatt(fileout,'/','institution','Met Office, United Kingdom');
ncwriteatt(fileout,'/','source','Model data generated using Met Office Unified Model rose suite-id u-bw210. This is a nesting suite that runs an N512 global forecast and 99 embedded limited-area models each using a convection-permitting grid-length of 1.5km. The LAMs are each 360x360 grid points. The outer region is deemed to be a spin-up region and is ignored. The central 240x240 is then coarse-grained onto a 45km scale using 30x30 horizontal averaging to produce a 8x8=64 grid of spatially averaged data. Each file contains data from only one of these 64 subdomains, but data from every one the 99 regions around the globe. The nesting simulations are free-running within each LAM, but the driving model is re-initialised every 00Z using operational atmospheric analyses. All 99 regions are wholly over the sea. For ease of comparison with global model output, the LAMs have been run using the global model L70 vertical level set with the top level at 80 km altitude.');
    ncwriteatt(fileout,'/','history',date);
    ncwriteatt(fileout,'/','references','Global driving model is using configuration known as GA6 (Walters et al 2017, doi.org/10.5194/gmd-10-1487-2017). Individual LAMs are using configuration known as RA1 (Bush et al 2019, doi: 10.5194/gmd-2019-130). The nesting suite is documented by Webster et al 2008, doi: 10.1002/asl.172). The data was used for a paper entitled -Machine learning of coarse-grained convection-permitting numerical weather prediction model thermodynamic tendencies- by Cyril Morcrette, code for which is available from https://github.com/CyrilMorcrette/ML_CoarseGrained_CRM_NWP');
    ncwriteatt(fileout,'/','comments','The potential temperature is the liquid ice static potential temperature found from the dry-bulb potential temperature by adjusting for latent cooling once all condensate is evaporated. The humidity is the total humidity found from the sum of water vapour, liquid cloud condensate, frozen ice cloud condensate, liquid rain precipitation and frozen graupel precipitation. The advective fluxes are found from centred-difference vector(u) dot grad(field) using the 3-dimensional winds, taking account of the staggered levels that various variables are held on. All data are provided every 2-hours.');

  end

end





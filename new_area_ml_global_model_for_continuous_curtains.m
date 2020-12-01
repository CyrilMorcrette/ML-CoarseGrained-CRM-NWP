clear
i_actually_do_stuff=1;

roseid='u-bl012';
roseid='u-bx951';
user='frme';

stashs={'00004','00010','00254','00012','00272'};
names={'air_potential_temperature','specific_humidity','mass_fraction_of_cloud_liquid_water_in_air','mass_fraction_of_cloud_ice_in_air','mass_fraction_of_rain_in_air'};


nz=70;

% Size of the region that was extract for each of the 98 by extract_ml_lamlike_region_from_global.py
lat_range=2;
lon_range=12;

%Earth radius
re=6371229.0;

% Read in the lat/lon coordinates produced by maps_nesting_domains_aqua_only.m
data=dlmread('/home/h01/frme/ml_lams_latlon_aqua_only.dat');
number=data(:,1);
reg_lat=data(:,2);
reg_lon=data(:,3);

% Only consider one region for testing the code.
reg_lat=reg_lat(1);
reg_lon=reg_lon(1);

% Gulf of Guinea
reg_lat=0;
reg_lon=0;

% Azores domain
%reg_lat=40;
%reg_lon=-25;

% South Atlantic
reg_lat=-40;
reg_lon=0;

start_date=datetime(2017,5,31);


%end_date=datetime(2016,3,31);

ana_times={'0000'}
ana_time=ana_times{1};

onedeglat=2.0*pi*re/360.0;

% Number of subdomains that inner LAM is being chopped into in both east-west and north-south direction
% 8 will mean 8x8.
number=8;

% The 360x360 LAM has a 240x240 central region away from spin-up boundary.
% With dx=1.5, this 240x240 is 360km x 360km. So width of each region to extract and average over is 360km/8=45km
dx_extract=(360/number)*1e3;
dy_extract=dx_extract;

for ss=1:length(stashs)
  % Loop over temperature and humidity etc
  stash_str=stashs{ss};
  variable_name=names{ss};

  date=start_date;

  for phi=1:length(reg_lon)
    % Loop over all the LAM domain
    LAT=reg_lat(phi);
    LON=reg_lon(phi);

    if LAT>=0
      lat_bit=[num2str(abs(LAT)),'N'];
    else
      lat_bit=[num2str(abs(LAT)),'S'];
    end

    if LON<0
      lon_bit=[num2str(abs(LON)),'W'];
    else
      lon_bit=[num2str(abs(LON)),'E'];
    end

    region_name=[lat_bit,lon_bit];

%!!!!!!!!!!!!!!!! HARD WIRED FOR
    domain_counter=1;
% as only actually need to do this once
%!!!!!!!!!!!!!!!!

    for i=-(number*0.5)%:(number*0.5)-1
      for j=-(number*0.5)%:(number*0.5)-1

        % Loop over the say 8x8 domains

        fileout=['/scratch/',user,'/ML/',roseid,'/step3alt_single_columns/',region_name,'_subregion_',num2str(domain_counter),'.',variable_name,'.nc'];

        % Remove pre-existing file
        cmd=['rm ',fileout];
        unix(cmd);
        % and create file from scratch.
        nccreate(fileout,variable_name,'Dimensions',{'Time',inf,'Height',nz},'ChunkSize',[1 nz])

        time_counter=1;
        for day=1:31
	  yyyy=date.Year;
          mm=date.Month;
          if mm<10
            mm_str=['0',num2str(mm)];
          else
            mm_str=[num2str(mm)];
          end

          dd=date.Day;
          if dd<10
            dd_str=['0',num2str(dd)];
          else
            dd_str=[num2str(dd)];
          end

          datestr=[num2str(yyyy),mm_str,dd_str];

          for hh=0:2:118

            time_str=num2str(hh);
            if hh<100
              time_str=['0',num2str(hh)];
            end
            if hh<10
              time_str=['00',num2str(hh)];
            end

            % Construct filename
            filein=['/scratch/',user,'/ML/',roseid,'/step2_individual_regions/',datestr,'T',ana_time,'Z_',region_name,'_glm_',time_str,'_',stash_str,'.nc'];
            % Read in data
            data_full=ncread(filein,names{ss});

            [nx ny nz nt]=size(data_full);

            for t=1:nt
              datain=data_full(:,:,:,t);

              dlon=(2*lon_range)/(nx);
              lon=[-lon_range+(0.5*dlon):dlon:lon_range-(0.5*dlon)]+LON;
              dlat=(2*lat_range)/(ny);
              lat=[-lat_range+(0.5*dlat):dlat:lat_range-(0.5*dlat)]+LAT;

              if i_actually_do_stuff==1
    	        % Initialise a vector to store the profile of data
		store_profile=nan(1,70);
		% Find the west and east hand limits (in km)
		startx_km=i*dx_extract;
		endx_km=(i+1)*dx_extract;
		% Fine the south and north limits (in km).
		starty_km=j*dy_extract;
		endy_km=(j+1)*dy_extract;
		% For the mean latitude of this sub-domain,
		% find how far 1 degree of longitude is.
		mean_lat=LAT+((j+0.5)*dx_extract/onedeglat);
		onedeglon=2.0*pi*cos(mean_lat*pi/180)*re/360.0;
		% Hence convert distance in km to distance in degrees.
		startx_deg=LON+(startx_km/onedeglon);
		endx_deg=LON+(endx_km/onedeglon);
		starty_deg=LAT+(starty_km/onedeglat);
		endy_deg=LAT+(endy_km/onedeglat);

		newx=[startx_deg:0.01:endx_deg];
		newy=[starty_deg:0.01:endy_deg];
		[X Y]=meshgrid(newx,newy);

		for k=1:70
		  tmp=datain(:,:,k)';
		  %'
		  interp_field=interp2(lon,lat,tmp,X,Y,'nearest');
		  store_profile(k)=mean(mean(interp_field));
		end % k

		ncwrite(fileout,variable_name,store_profile,[time_counter,1])

		time_counter=time_counter+1;

              end % i_actually_do_stuff

            end

          end

	  date=date+1;

        end

      end

    end
  end
end





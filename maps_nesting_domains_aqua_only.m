clear

latbands=[-80:10:80];

nperlatband=[2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2];

figure(1)
hold off

orog=ncread('/data/nwp1/frme/ml/n512_landseamask.nc','lsm');
n=512

orog=[orog;orog];

orog=orog(n+1:3*n,:);

dlon=360/(2*n);
lon=[-180:dlon:180-dlon];
dlat=180/(1.5*n);
lat=[-90+(0.5*dlat):dlat:90-(0.5*dlat)];
contour(lon,lat,orog',[0.1 0.1],'k-');%'
hold on

all_lat=[];
all_lon=[];

land_and_sea=0;

if land_and_sea==1

  for j=1:length(latbands)

    delta_lon=round(360/nperlatband(j));
    domains_lon=[0:delta_lon:180]+(mod(j,2)*0.5*delta_lon);
    domains_lon(domains_lon>180)=[];

    if domains_lon(1)==0
      domains_lon=[sort(-domains_lon(2:length(domains_lon))),domains_lon];
    else
      domains_lon=[sort(-domains_lon),domains_lon];
    end

    domains_lat=latbands(j)*ones(1,length(domains_lon));

    all_lat=[all_lat,domains_lat];
    all_lon=[all_lon,domains_lon];

  end

else

  some_lat=[0,0,0,0,0,0,0,0,0,0];
  some_lon=[-160,-130,-100,-30,-15,0,50,70,88,160];

  all_lat=[some_lat];
  all_lon=[some_lon];

  some_lat=[-10,-10,-10,-10,-10,-10,-10,-10,-10,-10];
  some_lon=[-170,-140,-120,-90,-30,-15,5,60,88,170];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[10,10,10,10,10,10,10,10,10,10];
  some_lon=[-170,-140,-120,-100,-50,-30,60,88,145,160];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[-20,-20,-20,-20,-20,-20,-20,-20];
  some_lon=[-160,-130,-100,-30,0,55,80,105];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[20,20,21,20,20,20,20,20];
  some_lon=[-170,-145,-115,-55,-30,65,135,170];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[-30,-30,-30,-30,-30,-30,-30,-30];
  some_lon=[-160,-130,-100,-40,-15,10,60,88];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[30,30,30,29,30,30,30,30];
  some_lon=[-170,-150,-130,-65,-45,-25,145,170];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[-40,-40,-40,-40,-40,-40,-40];
  some_lon=[-160,-130,-100,-50,0,50,100];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[40,40,40,40,40,40,40];
  some_lon=[-160,-140,-65,-45,-25,150,170];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[-50,-50,-50,-50,-50,-50];
  some_lon=[-150,-90,-30,30,88,150];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[50,50,50,50,50,50];
  some_lon=[-160,-140,-45,-25,149,170];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[-60,-60,-60,-60,-60];
  some_lon=[-140,-70,0,70,140];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[60,60];
  some_lon=[-35,-15];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[-70,-70];
  some_lon=[-160,-40];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[70];
  some_lon=[0];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

  some_lat=[80];
  some_lon=[-150];

  all_lat=[all_lat,some_lat];
  all_lon=[all_lon,some_lon];

end

[tmp nd]=size(all_lon)

re=6371000;
deglon=(2*pi*cos(60*pi/180)*re)/360;
nx=360;
dx=nx/2;
L=dx*1500;

for i=1:length(all_lon)
  deglon=(2*pi*cos(abs(all_lat(i)*pi/180))*re)/360;
  dlon=L/deglon;
  deglat=(2*pi*re)/360;
  dlat=L/deglat;
  plot(all_lon(i)+[-dlon,dlon,dlon,-dlon,-dlon],all_lat(i)+[-dlat,-dlat,dlat,dlat,-dlat],'b-');
end

nx=240;
dx=nx/2;
L=dx*1500;

for i=1:length(all_lon)
  deglon=(2*pi*cos(abs(all_lat(i)*pi/180))*re)/360;
  dlon=L/deglon;
  deglat=(2*pi*re)/360;
  dlat=L/deglat;
  plot(all_lon(i)+[-dlon,dlon,dlon,-dlon,-dlon],all_lat(i)+[-dlat,-dlat,dlat,dlat,-dlat],'c-');
end

axis([-180 180 -90 90])
grid off
set(gca,'TickDir','out')

number=[1:nd];

xlabel('Longitude')
ylabel('Latitude')

text(-177,80,'a)')

filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/ml_lam_domain_ocean_only_map.eps'];
print('-depsc2','-r300',filename)
filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/ml_lam_domain_ocean_only_map.png'];
print('-dpng','-r300',filename)
filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/ml_lam_domain_ocean_only_map.pdf'];
print('-dpdf','-r300',filename)

dataout=[number',all_lat',all_lon'];
dlmwrite('/home/h01/frme/ml_lams_latlon_aqua_only.dat',dataout)

figure(2);hold off
latbins=[-85:10:85];
h=histcounts(all_lat,latbins)
b=barh([-80:10:80],h,'c')
axis([0 10.5 -90 90])
set(gca,'TickDir','out')
xlabel('Number of domains in each latitude band')
ylabel('Latitude')

if 1==2
figure(3);hold off
contour(lon,lat,orog',[0.1 0.1],'k-');%'
hold on
for k=1:length(all_lon)
  cmd=[num2str(all_lat(k)),',',num2str(all_lon(k))];
  t1=text(all_lon(k)-15,all_lat(k),cmd);
set(t1,'FontSize',10)
end
end

%figure(4);
hold off
lat_tens=[-80:10:80];
count=zeros(1,length(lat_tens));
for k=1:length(all_lon)

[tmp ind]=min(abs(all_lat(k)-lat_tens));

 cmd=['(',num2str(all_lat(k)),',',num2str(all_lon(k)),')'];
  t1=text(count(ind)+0.05,lat_tens(ind),cmd);
set(t1,'FontSize',6)
count(ind)=count(ind)+1;
hold on
end
axis([0 10.25 -90 90])
set(gca,'TickDir','out')

%	text(1.25,80,'b)')

filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/ml_lam_domain_ocean_only_bar_chart_latlon_coords.eps'];
print('-depsc2','-r300',filename)
filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/ml_lam_domain_ocean_only_bar_chart_latlon_coords.png'];
print('-dpng','-r300',filename)
filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/ml_lam_domain_ocean_only_bar_chart_latlon_coords.pdf'];
print('-dpdf','-r300',filename)

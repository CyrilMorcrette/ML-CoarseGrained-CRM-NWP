clear

lsm=ncread('/data/nwp1/frme/ml/test_data/landseamask.nc','lsm');
[nx ny]=size(lsm);

n=nx/2;
orog=lsm;

dlon=360/(2*n);
lon=[0:dlon:360-dlon];
dlat=180/(1.5*n);
lat=[-90+(0.5*dlat):dlat:90-(0.5*dlat)];

figure(2)
[cc,hh]=contour(lon,lat,orog',[0.1 0.1],'k-');
hold on

t1p5=ncread('/data/nwp1/frme/ml/test_data/20160115T0000Z.Tp12.t1p5.nc','temp');

t1p5(lsm>0.99)=NaN;

re =  6371;  % km
domeLat =  0;       % degrees
domeLon = 0;       % degrees
domeAlt = 0;         % km

[x,y,z] = sphere(nx-1);

new_dlat=180/(nx-1);

new_lat=[-90:new_dlat:90];

new_t1p5=nan(nx,nx);
for i=1:nx
  new_t1p5(i,:)=interp1(lat,t1p5(i,:),new_lat);
end

tmp_t1p5(257:1024,:)=new_t1p5(1:768,:);
tmp_t1p5(1:256,:)=new_t1p5(769:1024,:);

new_t1p5=tmp_t1p5;

xEast  = re * x;
yNorth = re * y;
zUp    = re * z;

views=[[-30,20];[190,20];[80,-20];[210,-20]];

for multi=1:4
  figure(1);
  subplot(2,2,multi)
  hold off
  surf(xEast, yNorth, zUp,new_t1p5','FaceColor','white','FaceAlpha',1.0)

  shading flat
  axis equal

  caxis([242 302])
  colormap(jet(30))

  xlabel('x')
  ylabel('y')
  zlabel('z')

  coast_lon=cc(1,:);
  coast_lat=cc(2,:);

  for i=1:length(coast_lon)
    if coast_lon(i)<0.2
      coast_lon(i)=NaN;
      coast_lat(i)=NaN;
    end
    if coast_lon(i)>359.8
      coast_lon(i)=NaN;
      coast_lat(i)=NaN;
    end
  end

  %  figure(5)
  %  h = plot(coast_lon, coast_lat, 'r.');
  %  axis([0 360 -90 90])

  [globe_coast_x,globe_coast_y,globe_coast_z]=lat_lon_alt_to_x_y_z(coast_lat,coast_lon,re);

  figure(1)
  subplot(2,2,multi)
  hold on
  plot3(globe_coast_x,globe_coast_y,globe_coast_z,'k-')

  % Read in the lat/lon coordinates produced by maps_nesting_domains_aqua_only.m
  data=dlmread('/home/h01/frme/ml_lams_latlon_aqua_only.dat');
  number=data(:,1);
  reg_lat=data(:,2);
  reg_lon=data(:,3);

  for ii=1:length(reg_lon)
    x0=reg_lon(ii);
    y0=reg_lat(ii);

    %for the inner 240 domain (with dx=1.5km) that is a 360 km x 360 km region
    % so we want to know what dy is for half that (180km)

    dy=180/(2*pi*re/360);
    dx=180/(2*pi*re*cos(y0*pi/180)/360);

    %don't bother with base
    %plot3([x1,x2,x2,x1,x1],-[re,re,re,re,re],[y1,y1,y2,y2,y1],'r-')

    albedo=0.6;
    top=re+(80*10);

    % top
    box_lon=x0+[-dx,dx,dx,-dx,-dx];
    box_lat=y0+[-dy,-dy,dy,dy,-dy];
    alt=top;
    [x,y,z]=lat_lon_alt_to_x_y_z(box_lat,box_lon,alt);
    p=[x(1),y(1),z(1)];
    q=[x(2),y(2),z(2)];
    r=[x(3),y(3),z(3)];
    pq=q-p;
    pr=r-p;
    norm=cross(pq,pr);
    f1=fill3(x,y,z,'r-');
    set(f1,'FaceColor',[1 1 1]*0.5)

    %west
    box_lon=x0+[-dx,-dx,-dx,-dx,-dx];
    box_lat=y0+[-dy,-dy,dy,dy,-dy];
    alt=[re,top,top,re,re];
    [x,y,z]=lat_lon_alt_to_x_y_z(box_lat,box_lon,alt);
    p=[x(1),y(1),z(1)];
    q=[x(2),y(2),z(2)];
    r=[x(3),y(3),z(3)];
    pq=q-p;
    pr=r-p;
    norm=cross(pq,pr);
    f1=fill3(x,y,z,'r-');
    set(f1,'FaceColor',[1 1 1]*0.5)

    %east
    box_lon=x0+[dx,dx,dx,dx,dx];
    box_lat=y0+[-dy,-dy,dy,dy,-dy];
    alt=[re,top,top,re,re];
    [x,y,z]=lat_lon_alt_to_x_y_z(box_lat,box_lon,alt);
    p=[x(1),y(1),z(1)];
    q=[x(2),y(2),z(2)];
    r=[x(3),y(3),z(3)];
    pq=q-p;
    pr=r-p;
    norm=cross(pq,pr);
    f1=fill3(x,y,z,'r-');%s
    set(f1,'FaceColor',[1 1 1]*0.5)

    %north
    box_lon=x0+[-dx,-dx,dx,dx,-dx];
    box_lat=y0+[dy,dy,dy,dy,dy];
    alt=[re,top,top,re,re];
    [x,y,z]=lat_lon_alt_to_x_y_z(box_lat,box_lon,alt);
    p=[x(1),y(1),z(1)];
    q=[x(2),y(2),z(2)];
    r=[x(3),y(3),z(3)];
    pq=q-p;
    pr=r-p;
    norm=cross(pq,pr);
    f1=fill3(x,y,z,'r-');%
    set(f1,'FaceColor',[1 1 1]*0.5)

    %south
    box_lon=x0+[-dx,-dx,dx,dx,-dx];
    box_lat=y0+[-dy,-dy,-dy,-dy,-dy];
    alt=[re,top,top,re,re];
    [x,y,z]=lat_lon_alt_to_x_y_z(box_lat,box_lon,alt);
    p=[x(1),y(1),z(1)];
    q=[x(2),y(2),z(2)];
    r=[x(3),y(3),z(3)];
    pq=q-p;
    pr=r-p;
    norm=cross(pq,pr);
    f1=fill3(x,y,z,'r-');
    set(f1,'FaceColor',[1 1 1]*0.5)

end

grid off
axis off
set(gca,'View',views(multi,:))
ax = gca;               % get the current axis
ax.Clipping = 'off';    % turn clipping off

zoom(1.55)

if multi==4
  filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/virus_',num2str(multi),'_globes.eps'];
  print('-depsc2','-r600',filename)
  filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/virus_',num2str(multi),'_globes.png'];
  print('-dpng','-r600',filename)
  filename=['/home/h01/frme/cyrilmorcrette-projects/matlab/virus_',num2str(multi),'_globes.pdf'];  
  print('-dpdf','-r600',filename)
end

%		 close(multi)

end


clc;
close all

%% simple relaxation test
% in a single compartment, compares simulated T2 relaxation with
% exponential decay
clear all


Nx = 40;
Ny = 40;
sim.dt = .0005; % time step, ms
sim.dx = .1;    % spatial step, um
sim.map = ones(Nx,Ny);
sim.M0 = 1;
sim.diff = 0; 
sim.T2 = 60;
sim.P = 2e10;
te = 2;
t = sim.dt:sim.dt:te;
sim.Gx = zeros(size(t));

fprintf('Relaxation test 1: T2 = %.3f, TE = %.3f...\n',sim.T2,te);
[sig,Cx,Cy] = dwmr2d(sim);
for n=2:20
    [sig(n),Cx,Cy] = dwmr2d(sim,Cx,Cy);
end
fprintf('RMSE: %.3e\n',norm(exp(-te*(1:20)/sim.T2)-sig));

sim.T2 = 30;
fprintf('Relaxation test 2: T2 = %.3f, TE = %.3f...\n',sim.T2,te);
[sig,Cx,Cy] = dwmr2d(sim);
for n=2:20
    [sig(n),Cx,Cy] = dwmr2d(sim,Cx,Cy);
end
fprintf('RMSE: %.3e\n',norm(exp(-te*(1:20)/sim.T2)-sig));
fprintf('\n');

%% Diffusion test
% starting from a point source, compares the distribution of diffusing
% signal with analytical gaussian diffusion
clear all

Nx = 120;
Ny = 120;
sim.dt = .0005; % time step, ms
sim.dx = .5;    % spatial step, um
sim.map = ones(Nx,Ny);
sim.M0 = 1;
sim.diff = 3; 
sim.T2 = inf;
sim.P = 2e10;
te = 4;
t = sim.dt:sim.dt:te;
sim.Ax = zeros(size(t));
Cxn = zeros(size(sim.map));
Cxn(end/2+1,end/2+1) = 1;

fprintf('Diffusion distribution test 1: D = %.3f, TE = %.3f...\n',sim.diff,te);
[~,Cx] = dwmr2d(sim,Cxn,zeros(size(Cxn)));
[ix, iy] = meshgrid(sim.dx*(-Nx/2:Nx/2-1),sim.dx*(-Ny/2:Ny/2-1));
stdv = sqrt(2*sim.diff*te);
ana = 1/(stdv^2*2*pi)*exp(-ix.^2/2/stdv^2).*exp(-iy.^2/2/stdv^2)*sim.dx^2; % analytic solution
err = norm(Cx(:)-ana(:));
fprintf('RMSE: %.4e\n',err);

sim.diff = .5;
fprintf('Diffusion distribution test 2: D = %.3f, TE = %.3f...\n',sim.diff,te);
[~,Cx] = dwmr2d(sim,Cxn,zeros(size(Cxn)));
[ix, iy] = meshgrid(sim.dx*(-Nx/2:Nx/2-1),sim.dx*(-Ny/2:Ny/2-1));
stdv = sqrt(2*sim.diff*te);
ana = 1/(stdv^2*2*pi)*exp(-ix.^2/2/stdv^2).*exp(-iy.^2/2/stdv^2)*sim.dx^2;
err = norm(Cx(:)-ana(:));
fprintf('RMSE: %.4e\n',err);
fprintf('\n');

%% diffusion gradient test
% simulates diffusive signal decay, compared to the prescribed diffusion
% coefficient
clear all

Nx = 120;
Ny = 120;
sim.dt = .0005; % time step, ms
sim.dx = .1;    % spatial step, um
sim.map = ones(Nx,Ny);
sim.M0 = 1;
sim.diff = 3.0; 
sim.T2 = inf;
sim.P = 2e10;
te = 4;
t = sim.dt:sim.dt:te;
bigDelta = 2;      % ms
littleDelta = 1;    % m
Gdiff = 20; % G/cm
Gx = zeros(size(t));
Gx(t>=0 & t<=littleDelta) = 1;
Gx(t>bigDelta & t<bigDelta+littleDelta) = -1;

fprintf('Diffusion gradient test 1: D = %.3f, diff time = %.3f...\n',sim.diff,bigDelta+littleDelta/3);
for n=1:20
    [sim.Ax b(n)] = GxtoAx(Gx*Gdiff*n/20,sim.dt,sim.dx);
    sig(n) = dwmr2d(sim);
end
fprintf('RMSE: %e\n',norm(exp(-b*sim.diff)-sig));

sim.diff = 1.0;
te = 15;
t = sim.dt:sim.dt:te;
bigDelta = 11;      % ms
littleDelta = 3;    % m
Gdiff = 25; % G/cm
Gx = zeros(size(t));
Gx(t>=0 & t<=littleDelta) = Gdiff;
Gx(t>bigDelta & t<bigDelta+littleDelta) = -Gdiff;
fprintf('Diffusion gradient test 1: D = %.3f, diff time = %.3f...\n',sim.diff,bigDelta+littleDelta/3);
for n=1:20
    [sim.Ax b(n)] = GxtoAx(Gx*Gdiff*n/20,sim.dt,sim.dx);
    sig(n) = dwmr2d(sim);
end
fprintf('RMSE: %e\n',norm(exp(-b*sim.diff)-sig));
fprintf('\n');

%% water density test
% simulates the exchnage of water between compartments with different
% densities
clear all

Nx = 40;
Ny = 40;
width = 10;
sim.dt = .0005; % time step, ms
sim.dx = .5;    % spatial step, um
sim.map = ones(Nx,Ny);
sim.map((end/2+2):(end/2+width+2),:) = 2;
sim.M0 = [1 .2];
sim.diff = [1 1]; 
sim.T2 = [inf inf];
sim.P = 1e30;
te = 2;
t = sim.dt:sim.dt:te;
sim.Ax = zeros(size(t));

fprintf('Water density test 1...\n');
Cx1 = zeros(size(sim.map));
Cx1(end/2,end/2) = 1;
[~,Cx,Cy] = dwmr2d(sim,Cx1,zeros(size(Cx1)));
sig = sum(sum(sim.M0(sim.map).*Cx));
for n=2:60
    [~,Cx,Cy] = dwmr2d(sim,Cx,Cy);
    sig(n) = sum(sum(sim.M0(sim.map).*Cx));
end
ana = sum(sum(sim.M0(sim.map).*Cx1));
fprintf('RMSE: %e\n',norm(ana-sig));


sim.map = ones(Nx,Ny);
sim.map(:,(end/2+2):(end/2+width+2)) = 2;
fprintf('Water density test 2...\n');
Cx1 = zeros(size(sim.map));
Cx1(end/2,end/2) = 1;
[~,Cx,Cy] = dwmr2d(sim,Cx1,zeros(size(Cx1)));
sig = sum(sum(sim.M0(sim.map).*Cx));
for n=2:60
    [~,Cx,Cy] = dwmr2d(sim,Cx,Cy);
    sig(n) = sum(sum(sim.M0(sim.map).*Cx));
end
ana = sum(sum(sim.M0(sim.map).*Cx1));
fprintf('RMSE: %e\n',norm(ana-sig));
fprintf('\n');



%% Permeability test
% compares relaxing signal in two compartments with distinct T2
% relaxation-times with a bi-exponential (no exchange) or two compartment
% model
clear all

Nx = 40;
Ny = 40;
width = 20;
sim.dt = .0002; % time step, ms
sim.dx = .1;    % spatial step, um
sim.map = ones(Nx,Ny);
sim.map(end/2+1:(end/2+width),:) = 2;
sim.M0 = [1 .5];
sim.diff = [6 6]; 
sim.T2 = [60 12];
sim.P = 1e-30;
te = 6;
t = sim.dt:sim.dt:te;
sim.Ax = zeros(size(t));

fprintf('No Exchange test...\n');
[sig,Cx,Cy] = dwmr2d(sim);
for n=2:20
    [sig(n),Cx,Cy] = dwmr2d(sim,Cx,Cy);
end
ana = (sim.M0(1)*sum(sim.map(:)==1)*exp(-te*(1:20)/sim.T2(1)) + sim.M0(2)*sum(sim.map(:)==2)*exp(-te*(1:20)/sim.T2(2)))/sum(sim.M0(sim.map(:)));
fprintf('RMSE: %e\n',norm(ana-sig));


fprintf('Exchange test...\n');
sim.P = 1e-1;
[sig,Cx,Cy] = dwmr2d(sim);
for n=2:20
    [sig(n),Cx,Cy] = dwmr2d(sim,Cx,Cy);
end
M0r = sim.M0(1)*sum(sim.map(:)==1)/sum(sim.map(:)==2)/sim.M0(2);
k = sim.P/(sim.dx*width);
K = [-k k*M0r;k -k*M0r];
L = [1/sim.T2(1) 0;0 1/sim.T2(2)];
ana = zeros(length(sig),2);
for n=1:length(sig)
    ana(n,:) = expm((K-L)*(te*n))*([sim.M0(1)*sum(sim.map(:)==1);sim.M0(2)*sum(sim.map(:)==2)]/sum(sim.M0(sim.map(:))));
end
fprintf('RMSE: %e\n',norm(sum(ana,2)-sig'));
fprintf('\n');

%% restricted diffusion test
% compares restricted diffusion to the tanner & stejskal analytical form
clear all

Nx = 120;
Ny = 120;
sim.dt = .0005; % time step, ms
sim.dx = .2;    % spatial step, um
sim.map = ones(Nx,Ny);
sim.map(:,end/2+1:end) = 2;
sim.M0 = [1 1];
sim.diff = [3.0 3.0]; 
sim.T2 = [inf inf];
sim.P = 1e-30;
te = 10;
t = sim.dt:sim.dt:te;
Gdiff = .2/sim.dt; % G/cm
N = 20;
Gx = zeros(size(t));
Gx(2) = Gdiff/sim.dt;
Gx(end-1) = -Gdiff/sim.dt;

fprintf('Restricted Diffusion test 1: D = %.3f, diff time = %.3f...\n',sim.diff(1),te);
for n=1:N
    [sim.Ax b(n)] = GxtoAx(Gx*n/N,sim.dt,sim.dx);
    sig(n) = dwmr2d(sim);
end
L = Nx/2*sim.dx;
m=1:1000;
for n=1:N
    gGL =  26.7513*Gdiff*L/10000 * n/N;
    ana(n) = 2*(1-cos(gGL))/gGL^2 + 4*gGL^2*sum(exp(-m.^2*pi^2*sim.diff(1)*(te-2*sim.dt)/L/L).*(1-(-1).^m*cos(gGL))./(gGL^2-(m*pi).^2).^2);
end
fprintf('RMSE: %.3e\n',norm(ana-sig));
fprintf('\n');


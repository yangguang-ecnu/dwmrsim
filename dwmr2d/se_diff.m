clear all

%% discretization
Nx = 40; % number for gridpoints in 1st direction
Ny = 40;
sim.dt = .0005; % time step, ms
sim.dx = .1;    % spatial step, um

%% example simulation map - a single axon (square packed via periodic bc)
% axon = 3, myelin - 2, extra-axonal = 1
iR = 1;     % inner radius of axon
oR = 1.65;  % outer radius of axon
sim.map = ones(Nx,Ny);
iNx = repmat((1:Nx)'-Nx/2,1,Ny);
iNy = repmat((1:Ny)-Ny/2,Nx,1);
sim.map(sqrt(iNx.^2+iNy.^2)*sim.dx<oR) = 2;
sim.map(sqrt(iNx.^2+iNy.^2)*sim.dx<iR) = 3;

%% simulation parameters
% map integer value indexes vector characteristics [extra==1, myelin==2, axon==3]
sim.M0 = [1 .5 1];      % water density
sim.diff = [3 .0001 3]; % diffusion coefficent, um^2/ms
sim.T2 = [60 12 60];    % T2 relaxation time, ms
sim.P = 2e10;     % permeability between compartments, um/ms
te = 30;                %  acquisition, or "echo", time

%% setup diffusion gradient 
% example: array gradient amplitude 
Gstr = 0:5:25;  % G/cm
bigDelta = 16;      % ms
littleDelta = 3;    % ms
t = sim.dt:sim.dt:te;
Gdiff = zeros(size(t));
Gdiff(t>=0 & t<=littleDelta) = 1;
Gdiff(t>bigDelta & t<bigDelta+littleDelta) = -1;

%% diffusion simulation
fprintf('%3.0f%% done...',0);
for n=1:length(Gstr)
    [sim.Ax b(n)] = GxtoAx(Gstr(n)*Gdiff,sim.dt,sim.dx);
    
    sig(n) = dwmr2d(sim);
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b%3.0f%% done...',n/length(Gstr)*100);
end
fprintf('\b\b\b\b\b\b\b\b\b\b\b\b%3.0f%% done...\n',100);

%% fit dw signal to exponential decay
b0 = [.5 1];
ub = [ 1 3];
lb = [ 0 0];
beta = lsqnonlin(@(x) x(1).*exp(-b.*x(2))-double(sig),b0,lb,ub,optimset('Display','off'));
plot(b,sig,'.',b,beta(1)*exp(-b*beta(2)),'-','LineWidth',2);
xlabel('b-value (ms/um^2)','FontSize',16)
ylabel('Signal','FontSize',16)
disp(['ADC = ' num2str(beta(2))]);
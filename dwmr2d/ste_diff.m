%% an example on implementing a stimulated echo diffusion experiment
% assumes T1 is negligible during mixing time
clear all

%% discretization
Nx = 40;
Ny = 40;
sim.dt = .002; % time step, ms
sim.dx = .2;    % spatial step, um

%% example simulation map
sim.map = ones(Nx,Ny);
sim.M0 = 1;      % water density
sim.diff = 3; % diffusion coefficent, um^2/ms
T2 = 60;
sim.P = .02;     % permeability between compartments, um/ms

%% setup ste diffusion experiment , arraying both gradient amplitude and gradient seperation
Gstr = 1.875:1.875:30;  % G/cm
bigDelta = [12 30 60 100 150];      % ms
littleDelta = 4;    % ms
te = 18;    %  acquisition, or "echo", time
tm = bigDelta-te/2;
t = sim.dt:sim.dt:te+tm;

gamma = 4257.6*1e-7;
q = gamma*Gstr*littleDelta;

%% 
fprintf('%3.0f%% done...',0);
for m=1:length(Gstr)
    %waveform for this Gstr
    Gdiff = zeros(size(t));
    Gdiff(t>=0 & t<=littleDelta) = Gstr(m);
    Gdiff(t>bigDelta(1) & t<bigDelta(1)+littleDelta) = -Gstr(m);
    Ax = GxtoAx(Gdiff,sim.dt,sim.dx);
    Ax1 = Ax(t<te/2);
    Ax2 = Ax1(end);
    Ax3 = Ax(max(t)-t<te/2);

    % first te/2 time
    sim.T2 = T2;
    sim.Ax = Ax1;
    [s1 Cx Cy] = dwmr2d(sim);
    tmd = diff([0 tm]);
    % mixing time, tms
    for n=1:length(bigDelta)
        sim.T2 = inf;
        sim.Ax = Ax2*ones(tmd(n)/sim.dt);
        [s1 Cx Cy] = dwmr2d(sim,Cx,Cy);
        % last te/2
        sim.T2 = T2;
        sim.Ax = Ax3;
        dw_sig(n,m) = dwmr2d(sim,Cx,Cy);
    end
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b%3.0f%% done...',m/length(Gstr)*100);
end
fprintf('\b\b\b\b\b\b\b\b\b\b\b\b%3.0f%% done...\n',100);

%% display
plot(q,dw_sig,'.-')
legend(cellstr(num2str(bigDelta', 'bigDelta=%-d')))
xlabel('q')
ylabel('signal')
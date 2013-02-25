function [sig Cxo Cyo] = dwmr2d(sim,varargin)
% 
% [sig b Cxo Cyo] = dwmr2d(sim,Cxi,Cyi) 
%   a 2d diffusion-weighed MR signal simulation
%   input: sim is a structure including:
%       sim.map     = 2d map of integer compartment numbers detailing simulation geometry 
%       sim.M0      = vector of water density within indexed compartments
%       sim.diff    = vector of water diffusion coefficient within indexed compartments
%       sim.T2      = vector of water T2 relaxation time-constants within indexed compartments
%       sim.P       = scalar permeability between different compartments
%       sim.dx      = spatial step
%       sim.dt      = time step
%       sim.Ax      = applied diffusion gradient waveform (G/cm, spaced by sim.dt)
%       Cxi, Cyi    = normalized signal to initialize simulation geometry, optional
%
%   returns:
%       sig         = normalized signal value 
%                     i.e. with no diffusion or T2 weighting, sig=1
%                     assumes the gradient moment (Ax) is 0
%       Cxo, Cyo    = real and imaginary signal component density at end of simulation
%                     again, assumes Ax(end) = 0

%first a little error checking
if (max(sim.map(:))>length(sim.M0) || max(sim.map(:))>length(sim.diff) || max(sim.map(:))>length(sim.T2))
    error('Not enough compartments in M0, diff or T2 vectors')
end
if (sim.dt/sim.dx/sim.dx*max(sim.diff)> .2)
    error('Diffusion: dt is too high');
end
if (sim.dt/min(sim.T2) > .001)
    error('T2: dt is too high');
end

% simulation dimension
BlockMax = 256;
Nx = size(sim.map,1);
Ny = size(sim.map,2);

% setup array for T2 decay
T2 = gpuArray(single(sim.dt./sim.T2(sim.map)));

% setup arrays for diffusion 
map = zeros(size(sim.map)+2,'single');
map(2:end-1,2:end-1) = sim.map;
map(1,:) = map(end-1,:);
map(end,:) = map(2,:);
map(:,1) = map(:,end-1);
map(:,end) = map(:,2);
diffu = zeros(Nx,Ny,'single');
diffd = zeros(Nx,Ny,'single');
diffl = zeros(Nx,Ny,'single');
diffr = zeros(Nx,Ny,'single');
ix = 2:Nx+1;
iy = 2:Ny+1;
diffl(:,:) = sim.dt/sim.dx/sim.dx./(.5./sim.diff(map(ix,iy))+.5./sim.diff(map(ix+1,iy)) + 1/sim.dx*(map(ix,iy)~=map(ix+1,iy))./sim.P).*min(1,sim.M0(map(ix+1,iy))./sim.M0(map(ix,iy)));
diffr(:,:) = sim.dt/sim.dx/sim.dx./(.5./sim.diff(map(ix,iy))+.5./sim.diff(map(ix-1,iy)) + 1/sim.dx*(map(ix,iy)~=map(ix-1,iy))./sim.P).*min(1,sim.M0(map(ix-1,iy))./sim.M0(map(ix,iy)));
diffu(:,:) = sim.dt/sim.dx/sim.dx./(.5./sim.diff(map(ix,iy))+.5./sim.diff(map(ix,iy+1)) + 1/sim.dx*(map(ix,iy)~=map(ix,iy+1))./sim.P).*min(1,sim.M0(map(ix,iy+1))./sim.M0(map(ix,iy)));
diffd(:,:) = sim.dt/sim.dx/sim.dx./(.5./sim.diff(map(ix,iy))+.5./sim.diff(map(ix,iy-1)) + 1/sim.dx*(map(ix,iy)~=map(ix,iy-1))./sim.P).*min(1,sim.M0(map(ix,iy-1))./sim.M0(map(ix,iy)));
diffl_d = gpuArray(diffl);
diffr_d = gpuArray(diffr);
diffu_d = gpuArray(diffu);
diffd_d = gpuArray(diffd);

if isfield(sim,'Ax')
    Ax = sim.Ax;
else
    % gradient waveform
    Ax = GxtoAx(sim.Gx,sim.dt,sim.dx);
end

% initalize arrays
if length(varargin)==2
    Cxo = single(varargin{1});
    Cyo = single(varargin{2});
else
    Cxo = ones(Nx,Ny,'single');
    Cyo = zeros(Nx,Ny,'single');
end
C = zeros(Nx,Ny,'single');
Cxo_d = gpuArray(Cxo);

Cxn_d = gpuArray(C);
Cyo_d = gpuArray(Cyo);
Cyn_d = gpuArray(C);

% cuda code
fun = parallel.gpu.CUDAKernel('dwmr2dp.ptx', 'dwmr2dp.cu');
fun.ThreadBlockSize = [min(Nx,BlockMax) 1 1];
fun.GridSize = [ceil(Nx/BlockMax) Ny];
fun.SharedMemorySize = 0;

% run simulation
for n=1:length(Ax)
    [Cxo_d Cyo_d Cxn_d Cyn_d] = feval(fun, Cxn_d, Cyn_d, Cxo_d, Cyo_d, diffu_d, diffd_d, diffl_d, diffr_d, T2, Ax(n), Nx);
end

M0 = sim.M0(sim.map);
sig = abs(sum(gather(Cxo_d(:) + 1i*Cyo_d(:)).*M0(:)))/sum(M0(:));
Cxo = gather(Cxo_d);
Cyo = gather(Cyo_d);
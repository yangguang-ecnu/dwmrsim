function [Ax b] = GxtoAx(Gx,dt,dx)
% [Ax b] = GxtoAx(Gx,dt,dx)
%      Gx = gradient waveform in G/cm
%      dt,dx = time and spacial step in ms,um
%      Ax = unit appropriate integral of Gx for intput into dwmr2d
%      b = b-value for diffusion gradient waveform Gx (assuming Gx integrates to 0)

Gx_um = Gx/10e3; % Gauss/µm
gm = 26.7513; % rad/ms/Gauss
b = sum(cumsum(gm*Gx_um).^2)*dt.^3; % ms/µm^2
Ax = gm*cumsum(Gx_um)*dt*dx; % rad µm
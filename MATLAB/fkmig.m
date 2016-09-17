function [migSIG,param] = fkmig(SIG,param)

%FKMIG   f-k migration for plane wave imaging
%   MIGSIG = FKMIG(SIG,PARAM) performs a f-k migration of the signals
%   stored in the array SIG. MIGSIG contains the migrated signals. PARAM is
%   a structure that contains all the parameter values required for the
%   migration (see below for more details).
%
%   [MIGSIG,PARAM] = FKMIG(SIG,PARAM) also returns the complete list of
%   parameters:
%   PARAM.x and PARAM.z correspond to the coordinates at which the migrated
%   signals are returned. The x-axis is parallel to the transducer and
%   pointing from element #1 to element #N (PARAM.x = 0 at the CENTER of
%   the transducer). The z-axis is PERPENDICULAR to the transducer and
%   pointing downward (PARAM.z = 0 at the level of the transducer).
%
%   Important details on FKMIG:
%   --------------------------
%   1) The signals - typically RF signals - in SIG must be acquired using a
%      PLANE WAVE configuration with a linear array as used in ultrafast
%      ultrasound imaging. If SIG is 2-D, each column corresponds to a
%      single RF signal over time, with the FIRST COLUMN corresponding to
%      the FIRST ELEMENT.
%   2) Several consecutive acquisitions of same size can be stored in a 3-D
%      array for multi-angle compounding. In that case, the migrated output
%      corresponds to the average of the individual migrated signals.
%   3) The steering angle is positive (PARAM.TXangle > 0) if the 1st
%      element is the first to transmit. The steering angle is negative
%      (PARAM.TXangle < 0) if the last element is the first to transmit.
%   4) A bandpass filter can be used to smooth the migrated signals (see
%      PARAM.passband below).
%
%   PARAM is a structure that contains the following fields:
%   -------------------------------------------------------
%   1) PARAM.fs: sample frequency (in Hz, REQUIRED)
%   2) PARAM.pitch: pitch of the linear transducer (in m, REQUIRED)
%   3) PARAM.TXangle: steering (transmit) angles (in rad, default = 0)
%            One must have numel(PARAM.TXangle) = size(SIG,3).
%            PARAM.TXangle can also be a scalar.
%   4) PARAM.c: longitudinal velocity (in m/s, default = 1540 m/s)
%   5) PARAM.t0: acquisition start time (in s, default = 0)
%   6) PARAM.p: p-point sinc interpolation (default: p = 0)
%            A sinc interpolator is used, as described in Harlan, 1982,
%            "Avoiding interpolation artifacts in Stolt migration".
%            If PARAM.p = 0, then a linear interpolation is used (default).
%   7) PARAM.passband: Passband of the bandpass filter (default = [0,1]).
%            Must be a 2-element vector [w1,w2], where 0<w1<w2<1.0, with
%            1.0 corresponding to half the sample rate (i.e. param.fs/2).
%            A 5th order bandpass Butterworth filter is used.
%            Example: If you measured the RF signals at a sample rate of 20
%            MHz with a linear array whose passband is 3-7 MHz, you may use
%            PARAM.passband = [3e6,7e6]/(20e6/2) = [0.3 0.7].
%   8) PARAM.ncol: Number of columns in migSIG (default = original number).
%            The output migSIG is returned with PARAM.ncol columns by
%            interpolation in the Fourier transform along kx.
%
%   IMPORTANT NOTE: FKMIG does not use the transmit time delays as input
%   parameters. The transmit delays are determimed from the specified speed
%   of sound (PARAM.c) and the steering angle (PARAM.TXangle). 
%
%   Reference
%   --------- 
%   Garcia et al., Stolt's f-k migration for plane wave ultrasound imaging.
%   IEEE Trans Ultrason Ferroelectr Freq Control, 2013;60:1853-1867.
%   <a
%   href="matlab:web('http://www.biomecardio.com/pageshtm/publi/ieeeuffc13.pdf')">Paper here</a>
%
%   See also LUMIG, TXMIG, RFMIG
%   
%   -- Damien Garcia & Louis Le Tarnec -- 2011/08, revised 2013/11/18
%   website: <a
%   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>

[nt,nx,Nframes] = size(SIG);

%----- Input parameters -----
%-- 1) Speed of sound
if ~isfield(param,'c')
    param.c = 1540; % longitudinal velocity in m/s
end
c = param.c;
%-- 2) Sample frequency
if ~isfield(param,'fs')
    error(['A sample frequency (fs) ',...
        'must be specified as a structure field.'])
end
%-- 3) Acquisition start time
if ~isfield(param,'t0') % in s
    param.t0 = zeros(1,Nframes); % acquisition time start in s
end
if isscalar(param.t0)
    param.t0 = param.t0*ones(1,Nframes);
end
assert(numel(param.t0)==Nframes,...
    'PARAM.t0 must be a scalar or a vector of length number_of_frames.')
%-- 4) Pitch
if ~isfield(param,'pitch') % in m
    error('A pitch value (PARAM.pitch) is required.')
end
%-- 5) Transmit angle
if ~isfield(param,'TXangle')
    param.TXangle = 0; % in rad
end
if isscalar(param.TXangle)
    param.TXangle = param.TXangle*ones(1,Nframes);
end
assert(numel(param.TXangle)==Nframes,...
    'PARAM.TXangle must be a scalar or a vector of length number_of_frames.')
%-- 6) Interpolation option
if ~isfield(param,'p')
    param.p = 0;
end
assert(param.p>=0 & param.p==round(param.p),...
    ['PARAM.p must be a positive integer (p-point sinc interpolation)',...
    ' or zero (linear interpolation).'])
%-- 7) Passband filter
if isfield(param,'passband')
    wn = param.passband;
else
    wn = [0 1];
end
assert(numel(wn)==2 & wn(1)<wn(2) & wn(1)>=0 & wn(2)<=1 ,...
    'PARAM.passband must be a 2-element vector [w1,w2] with 0<w1<w2<1.')
%-- 8) Number of columns (FFT interpolation)
if ~isfield(param,'ncol')
    param.ncol = nx;
end
%----- end of Input parameters -----

SIG = double(SIG);

%-- Temporal shift
t0 = param.t0;
ntshift = max(round(t0*param.fs));

%-- Zero-padding before FFTs
%- time direction
if param.p
    % slight 0-padding with sinc interpolation
    ntFFT = round(1.5*nt)+ntshift;
else
    % extensive 0-padding is required with linear interpolation
    ntFFT = 4*nt+ntshift;
end
if rem(ntFFT,2)==1 % ntFFT must be even
    ntFFT = ntFFT+1;
end
%- x-direction
factor = 1.5;
nxFFT = ceil(factor*nx); % in order to avoid lateral edge effects
if rem(nxFFT,2)==1 % nxFFT must be even
    nxFFT = nxFFT+1;
end
dx = round(factor*(param.ncol-nx)); % will be used for FFT interpolation

%-- Create the filter if a passband is given
if ~isequal(wn(:),[0;1])
    [b,a] = butter(5,wn);
    hf = abs(freqz(b,a,ntFFT/2+1));
    hf = hf/max(hf);
else
    hf = 1;
end

f0 = (0:ntFFT/2)'*param.fs/ntFFT;
kx = [0:nxFFT/2 -nxFFT/2+1:-1]/param.pitch/nxFFT;
[kx,f] = meshgrid(kx,f0);

migSIG = zeros(ntFFT,nxFFT);

%-- Temporal FFT
SIG = fft(SIG,ntFFT);
% The signal is real: only the positive temporal frequencies are kept:
SIG(ntFFT/2+2:ntFFT,:,:) = [];

for k = 1:Nframes

    SIGk = SIG(:,:,k);
    sinA = sin(param.TXangle(k));
    cosA = cos(param.TXangle(k));
        
    %-- ERM velocity
    v = c/sqrt(1+cosA+sinA^2);
                              
    %-- Compensate for steering angle and/or depth start 
    if sinA~=0 || t0(k)~=0
        dt = sinA*((nx-1)*(param.TXangle(k)<0)-(0:nx-1))*...
            param.pitch/c; % steering angle
        tmp = bsxfun(@times,f0,dt+t0(k)); % depth start
        SIGk = SIGk.*exp(-2*1i*pi*tmp);
    end
    
    %-- Spatial FFT
    SIGk = fft(SIGk,nxFFT,2);

    %-- Note: we choose kz = 2*f/c (i.e. z = c*t/2);
    C = (1+cosA+sinA^2)/(1+cosA)^1.5;
    fkz = v*sqrt(kx.^2+4*f.^2/c^2*C^2);

    %-- Remove evanescent parts
    isevanescent = abs(f)./abs(kx)<c;
    SIGk(isevanescent) = 0;
    
    %-- Interpolation in the frequency domain: f -> fkz
    if param.p
        %- p-point sinc interpolation
        SIGk = interpSINC(param.fs/ntFFT,SIGk,fkz(:,1:nxFFT/2+1),param.p,([0 1]+wn)/2);
    else
        %- linear interpolation
        SIGk = interpLIN(param.fs/ntFFT,SIGk,fkz,([0 1]+wn)/2);
    end
    
    %-- Obliquity factor
    SIGk = SIGk.*f./fkz;
    SIGk(1) = 0;
    
    %-- Filtering
    if hf~=1, SIGk = bsxfun(@times,SIGk,hf); end
    
    %-- f-IFFT
    SIGk = [SIGk; conj([SIGk(ntFFT/2:-1:2,1) SIGk(ntFFT/2:-1:2,end:-1:2)])]; %#ok
    SIGk = ifft(SIGk);
    
    if sinA~=0 %-- Compensate for steering angle
        C = sinA/(2-cosA);
        dx = -C*(0:ntFFT-1)/param.fs*c/2;
        tmp = bsxfun(@times,kx(1,:),dx');
        SIGk = SIGk.*exp(-2*1i*pi*tmp);
    end

    %-- Compounding
    migSIG = ((k-1)*migSIG + SIGk)/k;

end

%-- Interpolation in the kx direction
if param.ncol>nx
    nyqst = nxFFT/2+1;
    migSIG = [migSIG(:,1:nyqst),...
        zeros(ntFFT,dx) migSIG(:,nyqst+1:nxFFT)];
    migSIG(:,nyqst) = migSIG(:,nyqst)/2;
    migSIG(:,nyqst+dx) = migSIG(:,nyqst);
    migSIG = migSIG*param.ncol/nx;
    nx = param.ncol;
end

%-- Final migrated signal
migSIG = ifft(migSIG,[],2,'symmetric'); % kx-IFFT
migSIG = migSIG((1:nt)+ntshift,1:nx);

%-- Grid coordinates
if nargin>1
    param.x = ((0:nx-1)-(nx-1)/2)*param.pitch;
    param.z = ((0:nt-1)+ntshift)*c/2/param.fs;
end

end

function yi = interpLIN(dx,y,xi,wn)
    % -- Linear interpolation along columns
    siz = size(y);
    yi = zeros(siz);
    n1 = round(wn(1)*(siz(1)-1)+1);
    n2 = round(wn(2)*(siz(1)-1)+1);
    
    % -- Classical interpolation
    idx = xi/dx + 1;
    I = idx>(siz(1)-1);
    idx(I) = 1; % arbitrary junk index
    idxf = floor(idx);
    for k = 1:siz(2)
        idxfk = idxf(n1:n2,k);
        idxk = idxfk-idx(n1:n2,k);
        yi(n1:n2,k) = y(idxfk,k).*(idxk+1) - y(idxfk+1,k).*idxk;
    end
    yi(I) = 0;
end

%{
function yi = interpLIN(dx,y,xi)
    % -- Linear interpolation along columns
    siz = size(xi);
    idx = xi/dx + 1;
    I = idx>=siz(1);
    idx(I) = 1; % arbitrary junk index
    idx = bsxfun(@plus,idx,(0:siz(2)-1)*siz(1));
    idxf = floor(idx);
    idx = idxf-idx;
    yi = y(idxf).*(idx+1) - y(idxf+1).*idx;
    yi(I) = 0;
end
%}

function yi = interpSINC(dx,y,xi,p,wn)
% -- Sinc interpolation --
% Harlan, 1982, "Avoiding interpolation artifacts in Stolt migration"
%                cf. equation (4) page 109
% --
% A p-point sinc interpolator is used
% --

%-- SINC approximation around 0:
% sinc0(x) ~ sinc(x) for x in [-.5,.5]
% sinc0(1) = 1 and sinc0(0.5) = sinc(0.5)
sinc0 = @(x) 1+(8/pi-4)*x.^2;

siz = size(y);
yi = zeros(siz);
n1 = round(wn(1)*(siz(1)-1)+1);
n2 = round(wn(2)*(siz(1)-1)+1);

xi = xi/dx + 1; % n + delta_n
idxr = round(xi); % n
delta_n = xi-idxr;
SINmat = sin(pi*delta_n);
EXPmat = cos(pi*delta_n)-1i*SINmat;

mp = [floor(-p/2+1):-1 1:floor(p/2)];
for k = 2:siz(2)/2+1
    yk1 = y(:,k);
    yk2 = y(:,end-k+2);
    idxrk = idxr(n1:n2,k);
    delta_nk = delta_n(n1:n2,k);
    for m = mp
        C = 1./(delta_nk-m);
        % C = (1-abs(delta_nk-m)/floor(p/2+1))./(delta_nk-m); % Triangle tapered
        % C = 1./tan(pi*(delta_nk-m)/p); % Muir's function tapered (see Fomel 2001)
        idxk = idxrk+m;
        
        % idxk = mod(idxk-1,siz(1))+1;
        I = idxk<1 |idxk>siz(1);
        C(I) = 0; idxk(I) = 1; % arbitrary junk index (since C(I)=0)
        
        yi(n1:n2,k) = yi(n1:n2,k) + C.*yk1(idxk);
        if k<siz(2)/2+1
            yi(n1:n2,end-k+2) = yi(n1:n2,end-k+2) + C.*yk2(idxk);
        end
    end
    yi(n1:n2,k) = 1/pi*SINmat(n1:n2,k).*yi(n1:n2,k);
    if k<siz(2)/2+1
        yi(n1:n2,end-k+2) = 1/pi*SINmat(n1:n2,k).*yi(n1:n2,end-k+2);
    end
    
    % idem for m = 0
    % ---
    % idxrk = mod(idxrk-1,siz(1))+1;
    I = idxrk>siz(1);
    idxrk(I) = 1; % arbitrary junk index
    delta_nk(I) = (4-8/pi)^-.5; % so that sinc0(delta_nk(I)) = 0
    
    S = sinc0(delta_nk);
    yi(n1:n2,k) = yi(n1:n2,k) + yk1(idxrk).*S;
    if k<siz(2)/2+1
        yi(n1:n2,end-k+2) = yi(n1:n2,end-k+2) + yk2(idxrk).*S;
    end
    
    yi(n1:n2,k) = yi(n1:n2,k).*EXPmat(n1:n2,k);
    if k<siz(2)/2+1
        yi(n1:n2,end-k+2) = yi(n1:n2,end-k+2).*EXPmat(n1:n2,k);
    end
    
end
end

%{
function yi = interpBSPLINE3(dx,y,xi)
% -- B-spline 3 --
% Thevenaz, 2000, "Interpolation revisited"

siz = size(y);
yi = zeros(siz);
xi = xi/dx + 1;
idxf = floor(xi); % n

% y = filter([0 6],[1 4 1],y);
% w{1} = (1-xi).^3/6;
% w{2} = 2/3 - xi.^2.*(1-xi/2);
% w{3} = 1/6 + xi.*(1+xi.*(1-xi))/2;
% w{4} = xi.^3/6;

w{1} = (-.5+(1-xi/2).*xi).*xi;
w{2} = 1+(-5/2+3/2*xi).*xi.^2;
w{3} = (.5+(2-1.5*xi).*xi).*xi;
w{4} = (-.5+xi/2).*xi.^2;

for k = 2:siz(2)/2+1
    yk1 = y(:,k);
    yk2 = y(:,end-k+2);
    idxfk = idxf(:,k);
    for m = -1:2
        idxk = idxfk+m;
        
        idxk = mod(idxk-1,siz(1))+1;
        % I = idxk<1 |idxk>siz(1);
        % C(I) = 0; idxk(I) = 1; % arbitrary junk index (since C(I)=0)
        
        yi(:,k) = yi(:,k) + w{m+2}(idxk).*yk1(idxk);
        if k<siz(2)/2+1
            yi(:,end-k+2) = yi(:,end-k+2) + w{m+2}(idxk).*yk2(idxk);
        end
    end
    
end
end
%}

%% ---- Simple academic code (horizontal plane waves only) ----
%{
function migRF = ezfkmig(RF,fs,pitch)

[nt0,nx0] = size(RF);

% Zero-padding
nt = 2^(nextpow2(nt0)+1);
nx = 2*nx0;

% Exploding Reflector Model velocity
c = 1540; % propagation velocity (m/s) 
ERMv = c/sqrt(2);

% FFT
fftRF = fftshift(fft2(RF,nt,nx));

% Linear interpolation
f = (-nt/2:nt/2-1)*fs/nt;
kx = (-nx/2:nx/2-1)/pitch/nx;
[kx,f] = meshgrid(kx,f);
fkz = ERMv*sign(f).*sqrt(kx.^2+f.^2/ERMv^2);
fftRF = interp2(kx,f,fftRF,kx,fkz,'linear',0);

% Jacobian (optional)
kz = (-nt/2:nt/2-1)'/ERMv/fs/nt;
fftRF = bsxfun(@times,fftRF,kz)./(fkz+eps);

% IFFT & Migrated RF
migRF = ifft2(ifftshift(fftRF),'symmetric');
migRF = migRF(1:nt0,1:nx0);
%}
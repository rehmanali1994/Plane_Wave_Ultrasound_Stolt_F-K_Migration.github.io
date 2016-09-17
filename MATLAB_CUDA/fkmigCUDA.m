function [migSIG,param] = fkmigCUDA(SIG,param)

%FKMIGCUDA   f-k migration for plane wave imaging (CUDA wrapper)
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

migSIG = zeros(nt, nx);

for idx = 1:Nframes
    % Save signals to a file
    SIGk = SIG(:,:,idx)';
    SIGkflat = SIGk(:);
    fid_img = fopen('SIG.txt','w');
    fprintf(fid_img,'%f\n', SIGkflat);
    
    % Run (maybe compile too if necessary) CUDA code that does f-k migration
    if isunix
        if ~exist('fkmigCUDA.out','file')
            error('Run the following in terminal: nvcc fkmigCUDA.cu -o fkmigCUDA.out -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft');
        end
        system(['./fkmigCUDA.out SIG.txt ', num2str(nt), ' ', num2str(nx), ...
            ' ', num2str(param.fs), ' ', num2str(param.pitch), ' ', ...
            num2str(param.TXangle(idx)), ' ', num2str(param.c), ' ', ...
            num2str(param.t0(idx)), ' migSIG.txt']);
    elseif ispc
        system(['fkmigCUDA.exe SIG.txt ', num2str(nt), ' ', num2str(nx), ...
            ' ', num2str(param.fs), ' ', num2str(param.pitch), ' ', ...
            num2str(param.TXangle(idx)), ' ', num2str(param.c), ' ', ...
            num2str(param.t0(idx)), ' migSIG.txt']);
    end
        
    % Load migrated image from file
    migSIGk = reshape(dlmread('migSIG.txt'), nx, nt)'; 
    migSIG = migSIGk/Nframes;
    
    % Delete the text files created in this process
    if isunix
        system('rm SIG.txt');
        system('rm migSIG.txt');
    elseif ispc
        system('del SIG.txt');
        system('del migSIG.txt');
    end
end
    
%-- Grid coordinates
if nargin>1
    param.x = ((0:nx-1)-(nx-1)/2)*param.pitch;
    param.z = ((0:nt-1))*param.c/2/param.fs;
end

end

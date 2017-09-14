% ---------------------------------------------------------------
% Stolt's f-k migration for plane wave ultrasound imaging
% Garcia D, Le Tarnec L, Muth S, Montagnon E, Por�e J, Cloutier G
% IEEE UFFC
% RUBIC & LBUM, CRCHUM, Montreal
% 03/2013
% ---------------------------------------------------------------

% ----------------------------------
% 403GS LE GAMMEX phantom.
% Verasonics scanner.
% 5 MHz linear transducer, ATL L7-4.
% ----------------------------------

% ----------------------------------------
% Enter "help fkmig" and "help ezfkmig" in
% the Matlab command window for details.
% ----------------------------------------

h = helpdlg({'Enter "help fkmig" and "help ezfkmig" in',...
    'the Matlab command window for details.'},'Help');
uiwait(h)

%% Load the RF data
% The RF data have been converted to 8 bits due to size limit (up to 1.5
% MB) of the zipped files in the Supplementary materials.
load ../RFdata1
load ../RFdata2
RF1 = double(RF1); RF1 = RF1-mean2(RF1);
RF2 = double(RF2); RF2 = RF2-mean2(RF2);

%% Example #1: Nylon fibers
h = helpdlg('Example #1: Gammex phantom, Nylon fibers','Example 1');
uiwait(h);
%---
% f-k migration
migRF1 = fkmigCUDAwrapper(RF1,param1);
% real envelope
im1 = abs(hilbert(migRF1)).^.7;
% figure #1
x = (0:127)*param1.pitch;
z = (0:size(RF1,1)-1)/param1.fs*param1.c/2;
f1 = figure(1);
imagesc(x,z,im1)
axis equal tight
colormap gray
title('Example #1: \it{f-k} migration with FKMIG','FontWeight','bold')
xlabel('7 angles: -1.5^{o}:0.5^{o}:1.5^{o}','FontWeight','bold')
ylabel('depth (m)')

%% Example #2: Circular targets
h = helpdlg('Example #2: Gammex phantom, Circular targets','Example 2');
uiwait(h);
%---
% f-k migration
migRF2 = fkmigCUDAwrapper(RF2,param2);
% real envelope
im2 = sqrt(abs(hilbert(migRF2)));
% figure #2
x = (0:127)*param2.pitch;
z = (0:size(RF2,1)-1)/param2.fs*param2.c/2;
f2 = figure(2);
imagesc(x,z,im2)
axis equal tight
colormap gray
title('Example #2: \it{f-k} migration with FKMIG','FontWeight','bold')
xlabel('7 angles: -1.5^{o}:0.5^{o}:1.5^{o}','FontWeight','bold')
ylabel('depth (m)')

%% Example #3: EZFKMIG
h = helpdlg('Example #3: before vs. after migration','Example 3');
uiwait(h);
%---
% Note: EZFKMIG works with horizontal plane waves only.
RF3 = RF1(:,:,4); % RF data with angle = 0
param3 = param1; param3.TXangle = 0;
% f-k migration
migRF3 = fkmigCUDAwrapper(RF3,param3);
% real envelope
im3 = abs(hilbert(RF3)).^.7;
im3_mig = abs(hilbert(migRF3)).^.7;
% figure #3
x = (0:127)*param3.pitch;
z = (0:size(RF3,1)-1)/param3.fs*param3.c/2;
f3 = figure(3);
% before migration
subplot(121)
imagesc(x,z,im3)
axis equal tight
colormap gray
title('Example #3: Before migration','FontWeight','bold')
xlabel('1 angle: 0^{o}','FontWeight','bold')
ylabel('depth (m)')
% after migration
subplot(122)
imagesc(x,z,im3_mig)
axis equal tight
title('\it{f-k} migration with EZFKMIG','FontWeight','bold')
xlabel('1 angle: 0^{o}','FontWeight','bold')
ylabel('depth (m)')

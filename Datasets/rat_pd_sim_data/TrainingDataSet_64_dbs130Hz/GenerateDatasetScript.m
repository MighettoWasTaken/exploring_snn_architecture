% Call the function from another file

% IT is how many iterations have been run, or which trial number is being run. It's used for saving the data generated as a .mat file.
% pd is a binary input which defines the state of the brain/neural data, where 0 is a normal/healthy condition, and 1 is a Parkinsonian model
% corstim is also a binary input which defines whether or not cortical stimulation should be applied. Note, cortical stimulation is not the same as deep brain stimulation, and can be turned off (0) while still applying DBS.
% pick_dbs_freq is the index of the desired stimulation frequency in an array of 0 to 200, with a spacing of 5. Note, MATLAB starts indexing at 1, not 0.

%function [] = simulate_network_model(IT,pd,corstim,pick_dbs_freq)
 
%IT - iteration number (trial no)
%pd - 0(normal/healthy condition), 1(Parkinson's disease(PD) condition)
%corstim (cortical stimulation) - 0(off), 1(on)
%pick_dbs_freq - choose appropriate DBS frequency (0, no stimulatoin; 1,
%5hz; 2, 10hz
%%% YU add: Example simulate_network_model(1,1,0,2)  ----> output 1pd5  

clear; clc;

a = 0; % IT
b = 0; % pd
c = 0; % corstim
% d = 1; % pick_dbs_freq
d = 27;
dataset_size = 64;

for i = 1:dataset_size
    a = i;
    if a>dataset_size/2
        b = 1;
    else
        b = 0; %first dataset_size/2 sample will have no pd, later 25 will have pd
    end
        fprintf('i = %d\n', i)
        simulate_network_model(a,b,c,d);
        % return;
end


%% New network to optimize dbs parameters
%  Want to use simulate_network_model() to give us data to feedback into 
%  the model to train a network that minimizes Amplitude & DBS freq while 
%  reducing alpha-beta oscillations
%  - come up with objective function how to do this
%   - can be related to a loss function that involves freq, amplitude,
%   phase, etc.
%
% Spike data:
% - TH_APs, which is the spike times of the action potentials (APs) of the thalamus (TH)
% - STN_APs, which is the spike times of the APs of the subthalamic nucleus(STN, or SN)
% - GPe_APs, which is the spike times of the APs of the globus pallidus externus (GPe, or Ge)
% - GPi_APs, which is the spike times of the APs of the globus pallidus interna (GPi, or Gi)
% - Striat_APs_indr, which is the spike times of the APs of the indirect striatum
% - Striat_APs_dr, which is the spike times of the APs of the direct striatum
% - Cor_APs, which is the spike times of the APs of the cortex
%
% Alpha-beta oscillation:
% - gpi_alpha_beta_area: single decimal value
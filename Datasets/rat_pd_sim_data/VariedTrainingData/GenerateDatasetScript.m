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


%a = 0;   
%b = 0;
%c = 0;
%d = 1;
%pd_on_size = 50;
%pd_off_size = 410;

%for i = 1:(pd_on_size + pd_off_size)
%    a = i;
    
%    if a <= pd_on_size
%        b = 0; % first dataset_size/2 sample will have no pd
%    else
%        b = 1;
        
%        % After the first 50 iterations, increment d every 10 iterations
%        if mod((a - pd_on_size), 10) == 1 & a > pd_on_size + 10
%            d = d + 1;
%        end
%    end
    
%    simulate_network_model(a, b, c, d);
%end


%simulate_network_model(0,0,0,1);
%simulate_network_model(0,0,0,41);

for i = 451:459
    simulate_network_model(i, 1, 0, 41);
end
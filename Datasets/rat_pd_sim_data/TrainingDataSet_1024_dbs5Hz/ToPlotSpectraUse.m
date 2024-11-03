%
 
params.Fs = 40000; %Hz
params.fpass = [1 50];
params.tapers = [3 5];
params.trialave = 1;

[S1,f1]= mtspectrumpt(GPi_APs, params);   % can load GPi_APs or STN_APs under differnt conditions: 1, normal state 2, PD state 3. PD with stimulaiton %% can use  GPe_APs for neurons in GPe; STN_APs for neurons in STN


figure(1)

S1_s=smooth(S1,20,'lowess');
semilogy(f1,S1_s,'k')

figure (2)
plot(f1,10*log10(S1_s),'k')

set(gca,'FontSize',8)
ylabel('log(power','FontSize',8)
xlabel('Freq (Hz)','FontSize',8)
xlim([0 50])
ax=gca;
ax.XTick=([0 10 20 30 40 50]);

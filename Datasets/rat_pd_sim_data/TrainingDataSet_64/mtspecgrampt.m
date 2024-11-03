function [S,t,f]=mtspecgrampt(data,movingwin,params,fscorr)

if nargin < 2; error('Need data and window parameters'); end;
if nargin < 3; params=[]; end;

[tapers,pad,Fs,fpass,err,trialave,params]=getparams(params);%
if length(params.tapers)==3 & movingwin(1)~=params.tapers(2);
    error('Duration of data in params.tapers is inconsistent with movingwin(1), modify params.tapers(2) to proceed')
end

data=change_row_to_column(data);%
if isstruct(data); Ch=length(data); end;
if nargin < 4 || isempty(fscorr); fscorr=0; end;
if nargout > 4 && err(1)==0; error('Cannot compute errors with err(1)=0'); end;

[mintime,maxtime]=minmaxsptimes(data);%
tn=(mintime+movingwin(1)/2:movingwin(2):maxtime-movingwin(1)/2);
Nwin=round(Fs*movingwin(1)); % number of samples in window
nfft=max(2^(nextpow2(Nwin)+pad),Nwin);
f=getfgrid(Fs,nfft,fpass); Nf=length(f);%
params.tapers=dpsschk(tapers,Nwin,Fs); % check tapers
nw=length(tn);

if trialave
    S = zeros(nw,Nf);
    R = zeros(nw,1);
    if nargout==4; Serr=zeros(2,nw,Nf); end;
else
    S = zeros(nw,Nf,Ch);
    R = zeros(nw,Ch);
    if nargout==4; Serr=zeros(2,nw,Nf,Ch); end;
end

for n=1:nw;
   t=linspace(tn(n)-movingwin(1)/2,tn(n)+movingwin(1)/2,Nwin);
   datawin=extractdatapt(data,[t(1) t(end)]);%
   if nargout==5;
     [s,f,r,serr]=mtspectrumpt(datawin,params,fscorr,t);%
     Serr(1,n,:,:)=squeeze(serr(1,:,:));
     Serr(2,n,:,:)=squeeze(serr(2,:,:));
   else
     [s,f,r]=mtspectrumpt(datawin,params,fscorr,t);
   end;
   S(n,:,:)=s;
   R(n,:)=r;
end;
t=tn;
S=squeeze(S); R=squeeze(R); if nargout==5; Serr=squeeze(Serr);end
end

function [tapers,pad,Fs,fpass,err,trialave,params]=getparams(params)
if ~isfield(params,'tapers') || isempty(params.tapers);  %If the tapers don't exist
     display('tapers unspecified, defaulting to params.tapers=[3 5]');
     params.tapers=[3 5];
end;
if ~isempty(params) && length(params.tapers)==3 
    % Compute timebandwidth product
    TW = params.tapers(2)*params.tapers(1);
    % Compute number of tapers
    K  = floor(2*TW - params.tapers(3));
    params.tapers = [TW  K];
end

if ~isfield(params,'pad') || isempty(params.pad);
    params.pad=0;
end;
if ~isfield(params,'Fs') || isempty(params.Fs);
    params.Fs=1;
end;
if ~isfield(params,'fpass') || isempty(params.fpass);
    params.fpass=[0 params.Fs/2];
end;
if ~isfield(params,'err') || isempty(params.err);
    params.err=0;
end;
if ~isfield(params,'trialave') || isempty(params.trialave);
    params.trialave=0;
end;

tapers=params.tapers;
pad=params.pad;
Fs=params.Fs;
fpass=params.fpass;
err=params.err;
trialave=params.trialave;
end

function data=change_row_to_column(data)
dtmp=[];
if isstruct(data);
   C=length(data);
   if C==1;
      fnames=fieldnames(data);
      eval(['dtmp=data.' fnames{1} ';'])
      data=dtmp(:);
   end
else
  [N,C]=size(data);
  if N==1 || C==1;
    data=data(:);
  end;
end;
end

function [mintime, maxtime]=minmaxsptimes(data)
dtmp='';
if isstruct(data)
   data=reshape(data,numel(data),1);
   C=size(data,1);
   fnames=fieldnames(data);
   mintime=zeros(1,C); maxtime=zeros(1,C);
   for ch=1:C
     eval(['dtmp=data(ch).' fnames{1} ';'])
     if ~isempty(dtmp)
        maxtime(ch)=max(dtmp);
        mintime(ch)=min(dtmp);
     else
        mintime(ch)=NaN;
        maxtime(ch)=NaN;
     end
   end;
   maxtime=max(maxtime); % maximum time
   mintime=min(mintime); % minimum time
else
     dtmp=data;
     if ~isempty(dtmp)
        maxtime=max(dtmp);
        mintime=min(dtmp);
     else
        mintime=NaN;
        maxtime=NaN;
     end
end
if mintime < 0 
   error('Minimum spike time is negative'); 
end
end

function [f,findx]=getfgrid(Fs,nfft,fpass)
if nargin < 3; error('Need all arguments'); end;
df=Fs/nfft;
f=0:df:Fs; % all possible frequencies
f=f(1:nfft);
if length(fpass)~=1;
   findx=find(f>=fpass(1) & f<=fpass(end));
else
   [fmin,findx]=min(abs(f-fpass));
   clear fmin
end;
f=f(findx);
end

function [tapers,eigs]=dpsschk(tapers,N,Fs)
if nargin < 3; error('Need all arguments'); end
sz=size(tapers);
if sz(1)==1 && sz(2)==2;
    [tapers,eigs]=dpss(N,tapers(1),tapers(2));
    tapers = tapers*sqrt(Fs);
elseif N~=sz(1);
    error('seems to be an error in your dpss calculation; the number of time points is different from the length of the tapers');
end;
end

function data=extractdatapt(data,t,offset)
if nargin < 2; error('Need data and times'); end;
if t(1) < 0 || t(2)<=t(1);
    error('times cannot be negative and t(2) has to greater than t(1)');
end;
if nargin < 3 || isempty(offset); offset=0; end;
if isstruct(data); 
    C=length(data);
elseif min(size(data))~=1; 
    error('Can only accept single vector data unless it is a struct array'); 
else
    C=1;
    data=change_row_to_column(data);
end;
%fnames=fieldnames(data);
d2(1:C)=struct('times',[]);
for c=1:C,
    if isstruct(data)
       fnames=fieldnames(data);
       eval(['dtmp=data(c).' fnames{1} ';'])
    else
       dtmp=data(:);
    end
%     eval(['dtmp=data(c).' fnames{1} ';' ])
    sp=dtmp(dtmp>=t(1) & dtmp<t(2));
    if offset==1; d2(c).times=sp-t(1); 
    else d2(c).times=sp;end
end;
data=d2;
end

function [S,f,R,Serr]=mtspectrumpt(data,params,fscorr,t)
if nargin < 1; error('Need data'); end;
if nargin < 2; params=[]; end;
[tapers,pad,Fs,fpass,err,trialave,params]=getparams(params);
clear params
data=change_row_to_column(data);
if nargout > 3 && err(1)==0; error('cannot compute error bars with err(1)=0; change params and run again'); end;
if nargin < 3 || isempty(fscorr); fscorr=0;end;
if nargin < 4 || isempty(t);
   [mintime,maxtime]=minmaxsptimes(data);
   dt=1/Fs; % sampling time
   t=mintime-dt:dt:maxtime+dt; % time grid for prolates
end;
N=length(t); % number of points in grid for dpss
nfft=max(2^(nextpow2(N)+pad),N); % number of points in fft of prolates
[f,findx]=getfgrid(Fs,nfft,fpass); % get frequency grid for evaluation
tapers=dpsschk(tapers,N,Fs); % check tapers
[J,Msp,Nsp]=mtfftpt(data,tapers,nfft,t,f,findx); % mt fft for point process times
S=squeeze(mean(conj(J).*J,2));
if trialave; S=squeeze(mean(S,2));Msp=mean(Msp);end;
R=Msp*Fs;
if nargout==4;
   if fscorr==1;
      Serr=specerr(S,J,err,trialave,Nsp);
   else
      Serr=specerr(S,J,err,trialave);
   end;
end;
end

function [J,Msp,Nsp]=mtfftpt(data,tapers,nfft,t,f,findx)
% Multi-taper fourier transform for point process given as times
%
% Usage:
% [J,Msp,Nsp]=mtfftpt (data,tapers,nfft,t,f,findx) - all arguments required
% Input: 
%       data        (struct array of times with dimension channels/trials; 
%                   also takes in 1d array of spike times as a column vector) 
%       tapers      (precalculated tapers from dpss) 
%       nfft        (length of padded data) 
%       t           (time points at which tapers are calculated)
%       f           (frequencies of evaluation)
%       findx       (index corresponding to frequencies f) 
% Output:
%       J (fft in form frequency index x taper index x channels/trials)
%       Msp (number of spikes per sample in each channel)
%       Nsp (number of spikes in each channel)
if nargin < 6; error('Need all input arguments'); end;
if isstruct(data); C=length(data); else C=1; end% number of channels
K=size(tapers,2); % number of tapers
nfreq=length(f); % number of frequencies
if nfreq~=length(findx); error('frequency information (last two arguments) inconsistent'); end;
H=fft(tapers,nfft,1);  % fft of tapers
H=H(findx,:); % restrict fft of tapers to required frequencies
w=2*pi*f; % angular frequencies at which ft is to be evaluated
Nsp=zeros(1,C); Msp=zeros(1,C);
for ch=1:C;
  if isstruct(data);
     fnames=fieldnames(data);
     eval(['dtmp=data(ch).' fnames{1} ';'])
     indx=find(dtmp>=min(t)&dtmp<=max(t));
     if ~isempty(indx); dtmp=dtmp(indx);
     end;
  else
     dtmp=data;
     indx=find(dtmp>=min(t)&dtmp<=max(t));
     if ~isempty(indx); dtmp=dtmp(indx);
     end;
  end;
  Nsp(ch)=length(dtmp);
  Msp(ch)=Nsp(ch)/length(t);
  if Msp(ch)~=0;
      data_proj=interp1(t',tapers,dtmp);
      exponential=exp(-i*w'*(dtmp-t(1))');
      J(:,:,ch)=exponential*data_proj-H*Msp(ch);
  else
      J(1:nfreq,1:K,ch)=0;
  end;
end;
end
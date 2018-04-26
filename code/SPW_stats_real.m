% compute statistics of SPWs from MEA measurements

% directory where mat files are stored
directory = 'C:\Users\user\Documents\BCCN\1st Rotation\spwr_MEA\';
files = dir(strcat(directory,'*.mat'));

IEI = [];
for q = 1:length(files) 
    load(strcat(directory,files(q).name));
    IEI = cat(1,IEI,diff(data_struct.TimeOfEvent));
end

% fit distribution to data. lognormal works best. then gamma. weibull is awful
% birnbaumsaunders : bad tails, bad fit, tail is like normal distr
% burr (3 params, good tails, perfect) ratio of polynomials
% generalized extreme value (3 params, not perfect tails) quite good but
% burr better with 3 params, (much) faster than exponential
% inverse gaussian (no tails), not well fit, better than birnbaum
% log-logistic (kinda good tails, worse than burr), very good fit, ratio of polynomials
% stable (4 params, perfect fit, perfect tails)
% log-normal : worst fit and tails than loglogistic, still good, relatively slow tails
dist = 'lognormal'; % 'gamma', 'weibull', 'lognormal'
f = fitdist(IEI,dist);
x = 0:.1:5;
y = pdf(f,x);
figure
plot(x,y)
hold on

% remove extreme values for better plot
% if th = 15, 4 rejected (8525,8546,8558,8561, so they most possibly are in the same recording)
% if th = 10, 16 rejected (2957,8478,8493,8505,8510,8512,8520,8522,8525,8526,8543,8546,8558,8560,8561,8566)
% again, probably same recording
% if th = 5, 115 rejected, from 10 recordings, mostly from 6 main recordings
% so, extreme values are recording specific and not important!

th = 5;
IEI(IEI>th) = [];
histogram(IEI,200,'Normalization','pdf')
xlabel('IEI (sec)')
title('PDF of IEI for SPW events')

% fit after extreme values are removed is much better

% overall: stable too many params, huge tails
% burr : many params, essentially the same with log-logistic
% log-logistic : very good fit, because tails are probably artificial it doesnt have good tails
% log-normal : not very good fit, tails are very good if we consider that extreme values are artificial

IEI_r = round(IEI,2);
num = 1;
i = 1;
while i<size(IEI_r,1)
    temp = round(IEI_r(i)/.01);
    events(num) = true;
    num = num + temp;
    i = i + 1;
end

[Pxx,F] = periodogram(single(events),rectwin(length(events)),10000,100);
figure
semilogy(F,Pxx)
xlabel('Frequency [Hz]')
ylabel('Spectral density [1/Hz]')
title('Event spectral density')
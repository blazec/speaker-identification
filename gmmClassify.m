dir_train = 'Training';
dir_test = 'Testing';

M = 9;
max_iter = 10;
epsilon = .001;
gmms = gmmTrain(dir_train, max_iter, epsilon, M);

testMfccFiles = dir([dir_test, filesep, '*.mfcc']);
numTestFiles = length(testMfccFiles);
predictions = cell(numTestFiles, 1);

% for accuracy purposes
correctSpeakers = {'MMRP0','MPGH0','MKLW0','FSAH0','FVFB0','FJSP0','MTPF0','MRDD0','MRSO0','MKLS0','FETB0','FMEM0','FCJF0','MWAR0','MTJS0'};
numCorrect = 0;

disp('Writing .lik files');

for iTestFile=1:numTestFiles
    
   filename = testMfccFiles(iTestFile).name;
   unknNameMatch = regexp(filename, 'unkn_[0-9]{1,2}', 'match');
   unknName = unknNameMatch{1};
   
   % get num of mfcc file to check for accuracy
   fileNumMatch = regexp(unknName, '[0-9]{1,2}', 'match');
   fileNum = str2num(fileNumMatch{1});
   
   X = load(strcat(dir_test, filesep, filename));
   numGmms = length(gmms);
   logLikelihoods = zeros(numGmms, 1);
   
   for iGmm=1:numGmms
       [P,L] = ComputeLikelihood(X, gmms{iGmm}, M);
       logLikelihoods(iGmm) = L;
   end
   
   % Get top 5 likelihoods
   [likelihoods_sorted, ordered_indices] = sort(logLikelihoods, 'descend');
   
   fid = fopen(strcat(unknName, '.lik'), 'w');
   fprintf(fid, 'Speaker Likelihood\n');
   for i=1:5
       s = sprintf('%s %f\n', gmms{ordered_indices(i)}.name, likelihoods_sorted(i));
       fprintf(fid, s);
   end
   fclose(fid);
   
   if fileNum <= 15
       if strcmp(gmms{ordered_indices(1)}.name, correctSpeakers{fileNum})
           numCorrect = numCorrect + 1;
       else
           fprintf('Correct speaker: %s\n', correctSpeakers{fileNum});
           fprintf('Got: %s\n', gmms{ordered_indices(1)}.name);
       end
   end
   
   
end

disp('Writing done');
disp(numCorrect);
accuracy = (numCorrect / 15) * 100;
fprintf('Accuracy: %f%%\n',  accuracy);
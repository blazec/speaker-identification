warning('off', 'MATLAB:nargchk:deprecated')
dir_train = 'Training';
% phonemes.(phoneme) = cell array of {mfcc filename, mffcc file row number for start of phoneme utterance,
%                                       mfcc file row number for end of phoneme utterance}
phonemes = struct();
addpath(genpath('FullBNT-1.0.7'));
% Get speaker directories
dirsAll = dir(dir_train);
dirsAll = dirsAll([dirsAll.isdir]);
% dirs is without '.', '..', and non-directory files
dirs = cell(length(dirsAll) - 2, 1);
iDirName = 1;

for l=1:length(dirsAll)
    if strcmp(dirsAll(l).name, '.') || strcmp(dirsAll(l).name, '..') || ~dirsAll(l).isdir
        continue
    end
    dirs{iDirName} = dirsAll(l).name;
    iDirName = iDirName + 1;
end

for iSpeaker=1:length(dirs)
    name = dirs{iSpeaker};
    phnFiles = dir([dir_train, filesep, name, filesep, '*.phn']);
    
    for iPhnFile=1:length(phnFiles)
        phnFileName = phnFiles(iPhnFile).name;
        phnFileTitle = regexp(phnFileName, '\w*[0-9]{1,}', 'match');
        phnFileTitle = phnFileTitle{1};
        phnFilePath = strcat(dir_train, filesep, name, filesep, phnFileName);
        fID = fopen(phnFilePath);
        phnContents = textscan(fID, '%s');
        fclose(fID);
        mfccFile = strcat(dir_train, filesep, name, filesep, phnFileTitle, '.mfcc');
        
        % loop over phn file lines
        iPhn = 1;
        
        while iPhn <= length(phnContents{1})
            startIdx = iPhn;
            endIdx = iPhn + 1;
            phonemeIdx = iPhn + 2;
            start = str2double(phnContents{1}(startIdx)) / 128;
            end_ = str2double(phnContents{1}(endIdx)) / 128;
            phoneme = phnContents{1}{phonemeIdx};            
            if strcmp(phoneme, 'h#')
                phoneme = 'h_';
            end
            
            if ~isfield(phonemes, phoneme)
                phonemes.(phoneme) = {mfccFile, start + 1, end_ + 1};
            else
                phonemes.(phoneme) = [phonemes.(phoneme); {mfccFile, start + 1, end_ + 1}];
            end
            
            iPhn = iPhn + 3;
        end
        
    end
    
end

mfccMatch = struct();
HMMs = struct();
phonemeNames = fieldnames(phonemes);

% Train all phonemes
for iPhnName=1:numel(phonemeNames)
  currPhonemeInfo = phonemes.(phonemeNames{iPhnName});
  currPhonemeName = phonemeNames{iPhnName};
  
  mfccMatch.(currPhonemeName) = {};
  
  iUtterance = 1;
  for utteranceInfo = currPhonemeInfo'
    % Get mfcc file rows
    mfccFile = utteranceInfo{1};
    fID = fopen(mfccFile, 'r');
    mfccRows = textscan(fID, '%s', 'delimiter', '/n');
    fclose(fID);
    start = utteranceInfo{2};
    end_ = min(utteranceInfo{3}, length(mfccRows{1})+1);
    
    mfccMatrix = zeros(14, end_ - start);
    
    % Get the corresponding mfcc vectors for the utterance of the phoneme
    for iMfccRow = start:(end_-1)
      mfccRow = mfccRows{1}{iMfccRow};
      mfccRow = cellfun(@str2double, strsplit(' ', strtrim(mfccRow)));
      mfccMatrix(:, iMfccRow - start + 1) = mfccRow';
    end
    
    mfccMatch.(currPhonemeName){iUtterance} = mfccMatrix;

    iUtterance = iUtterance + 1;
  end
  
  disp(sprintf('Training HMM for phoneme %s', currPhonemeName));
  HMMinit = initHMM(mfccMatch.(currPhonemeName), 8, 3, 'kmeans');
  [HMMphoneme, LLphoneme] = trainHMM(HMMinit, mfccMatch.(currPhonemeName), 10);

  HMMs.(currPhonemeName).HMM = HMMphoneme;
  HMMs.(currPhonemeName).LL = LLphoneme;
end

save('HMMs.mat', 'HMMs');
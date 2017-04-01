function gmms = gmmTrain( dir_train, max_iter, epsilon, M )
% gmmTain
%
%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture
    
    gmms = {};
    

    
    dirsAll = dir(dir_train);
    dirsAll = dirsAll([dirsAll.isdir]);
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
        % each speaker
        name = dirs{iSpeaker};
    
        % Create matrix of mfcc vectors
        mfccFiles = dir([dir_train, filesep, name, filesep, '*.mfcc']);
        X = [];
        for iMfcc=1:length(mfccFiles)
            X = [X; dlmread([dir_train, filesep, name, filesep, mfccFiles(iMfcc).name])];
        end
        
        % Initialize gmm
        gmm = struct();
        gmm.name = name;
        gmm.weights = zeros(1,M) + (1 / M);
        
        numSamples = size(X,1);
        randomizedIndices = randperm(numSamples);
        randomSample = randomizedIndices(1:M);
        means = zeros(M,size(X,2));
        for iRs=1:length(randomSample)
            means(iRs,:) = X(randomSample(iRs),:);
        end 
        
        gmm.means = means';
        gmm.cov = repmat(eye(size(X, 2)), 1, 1, M);
        
        i = 0;
        prevL = -Inf;
        improvement = Inf;
        
        while i <= max_iter && improvement >= epsilon
            [P, L] = ComputeLikelihood(X, gmm, M);
            gmm = UpdateParameters(gmm, X, P, M);
            improvement = L - prevL;
            prevL = L;
            i = i + 1;
        end
       
        gmms{iSpeaker} = gmm;
    end
    
end


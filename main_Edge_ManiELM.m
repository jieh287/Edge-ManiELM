%% Main script for Edge-ManiELM
clear; clc; close all;

addpath(genpath(pwd));

%% Load dataset
% The data matrix should be arranged as [features; target].
% Each column corresponds to one sample.

dataFile  = 'data.mat';
coordFile = 'data_geo.mat';

data = load_first_numeric_matrix(dataFile);

X = data(1:end-1, :)';     % Samples x features
y = data(end, :)';         % Samples x 1

numSamples  = size(X, 1);
numFeatures = size(X, 2);

coord = load_sample_coordinates(coordFile, numSamples);

fprintf('Loaded dataset: %d samples, %d features.\n', numSamples, numFeatures);

%% Parameter settings
numHiddenNodes = 22;
numKnots       = 10;
numFolds       = 5;

activationFunc = 'kan';
problemType    = 0;       % 0 = regression, 1 = classification

opts = struct();

opts.maxEpoch    = 100;
opts.lr          = 0.01;
opts.lambda      = 1e-4;
opts.weightDecay = 1e-5;

% Fixed Edge-ManiELM configuration
opts.splineOrder = 8;      
opts.knotMode    = 1;      
opts.useSmooth   = 1;      
opts.baseMode    = 3;     
opts.baseDecay   = 1e-5;

opts.smoothLambda = 1e-4;

% Spatial-spectral graph settings
opts.useGraph       = 1;
opts.graphK         = 5;
opts.graphUseGeo    = 1;
opts.graphUseSpec   = 1;
opts.graphLambda    = 1e-1;
opts.graphNormalize = 1;
opts.graphCoord     = [];

opts.verbose = 0;
opts.seed    = [];

%% Build K-fold indices
% No fixed random seed is used here.

foldID = make_fold_index(numSamples, numFolds);

%% Metric storage
% Test : RMSE, r, MAE, RPD, RPIQ, MaxRelErr, MinRelErr
% Train: RMSE, r, MAE, RPD, RPIQ, MaxRelErr, MinRelErr

foldMetrics = zeros(numFolds, 14);
foldModels  = cell(numFolds, 1);

%% Cross-validation

for fold = 1:numFolds

    fprintf('\n========== Fold %d / %d ==========\n', fold, numFolds);

    testMask  = (foldID == fold);
    trainMask = ~testMask;

    XTrain = X(trainMask, :);
    yTrain = y(trainMask, :);

    XTest = X(testMask, :);
    yTest = y(testMask, :);

    coordTrain = coord(trainMask, :)';

    %% Normalize data
    % Edge-ManiELM internally uses column-wise samples.

    [PTrain, inputPS] = mapminmax(XTrain', -1, 1);
    PTest = mapminmax('apply', XTest', inputPS);

    [TTrain, outputPS] = mapminmax(yTrain', -1, 1);

    %% Train Edge-ManiELM

    optsFold = opts;
    optsFold.graphCoord = coordTrain;

    [KAN, LW, TF, TYPE, trainInfo] = elmtrain_Edge( ...
        PTrain, ...
        TTrain, ...
        numHiddenNodes, ...
        activationFunc, ...
        problemType, ...
        numKnots, ...
        optsFold);

    %% Prediction

    yTestNormPred  = elmpredict_Edge(PTest,  KAN, LW, TF, TYPE);
    yTrainNormPred = elmpredict_Edge(PTrain, KAN, LW, TF, TYPE);

    %% Reverse normalization

    yTestPred  = mapminmax('reverse', yTestNormPred,  outputPS)';
    yTrainPred = mapminmax('reverse', yTrainNormPred, outputPS)';

    %% Evaluation

    testScores  = Metrics(yTest,  yTestPred);
    trainScores = Metrics(yTrain, yTrainPred);

    foldMetrics(fold, :) = [ ...
        testScores.RMSE, ...
        testScores.r, ...
        testScores.MAE, ...
        testScores.RPD, ...
        testScores.RPIQ, ...
        testScores.MaxRelErr, ...
        testScores.MinRelErr, ...
        trainScores.RMSE, ...
        trainScores.r, ...
        trainScores.MAE, ...
        trainScores.RPD, ...
        trainScores.RPIQ, ...
        trainScores.MaxRelErr, ...
        trainScores.MinRelErr];

    %% Save fold model
    % This is useful when checking a specific fold later.
    model = struct();
    model.KAN = KAN;
    model.LW = LW;
    model.TF = TF;
    model.TYPE = TYPE;
    model.inputPS = inputPS;
    model.outputPS = outputPS;
    model.trainInfo = trainInfo;
    model.fold = fold;

    foldModels{fold} = model;

    fprintf('Test : RMSE = %.4f, r = %.4f, MAE = %.4f\n', ...
        testScores.RMSE, testScores.r, testScores.MAE);

    fprintf('Train: RMSE = %.4f, r = %.4f, MAE = %.4f\n', ...
        trainScores.RMSE, trainScores.r, trainScores.MAE);
end

%% Mean cross-validation results

meanMetrics = mean(foldMetrics, 1);

testRMSE = meanMetrics(1);
testR    = meanMetrics(2);
testMAE  = meanMetrics(3);
testRPD  = meanMetrics(4);
testRPIQ = meanMetrics(5);

trainRMSE = meanMetrics(8);
trainR    = meanMetrics(9);
trainMAE  = meanMetrics(10);
trainRPD  = meanMetrics(11);
trainRPIQ = meanMetrics(12);

testR2 = testR ^ 2;

results_train = [trainRMSE, trainRPIQ, trainMAE, trainR];
results_test  = [testRMSE,  testRPIQ,  testMAE,  testR];

fprintf('\n========== Mean Cross-Validation Results ==========\n');

fprintf('Train: RMSE = %.4f, r = %.4f, MAE = %.4f, RPD = %.4f, RPIQ = %.4f\n', ...
    trainRMSE, trainR, trainMAE, trainRPD, trainRPIQ);

fprintf('Test : RMSE = %.4f, r = %.4f, MAE = %.4f, RPD = %.4f, RPIQ = %.4f, R2 = %.4f\n', ...
    testRMSE, testR, testMAE, testRPD, testRPIQ, testR2);

disp('results_train = [RMSE, RPIQ, MAE, r]');
disp(results_train);

disp('results_test = [RMSE, RPIQ, MAE, r]');
disp(results_test);

%% Optional save
% save('Edge_ManiELM_cv_summary.mat', 'foldMetrics', 'meanMetrics', 'foldModels');

%% Local functions

function data = load_first_numeric_matrix(fileName)

    S = load(fileName);
    names = fieldnames(S);

    data = [];

    for i = 1:numel(names)
        candidate = S.(names{i});
        if isnumeric(candidate) && ismatrix(candidate)
            data = candidate;
            break;
        end
    end

    if isempty(data)
        error('No numeric matrix was found in %s.', fileName);
    end
end


function coord = load_sample_coordinates(fileName, numSamples)

    S = load(fileName);

    if isfield(S, 'data_zuobiao')
        coord0 = S.data_zuobiao;
    else
        names = fieldnames(S);
        coord0 = S.(names{1});
    end

    if size(coord0, 1) == 2 && size(coord0, 2) == numSamples
        coord = coord0';
    elseif size(coord0, 2) == 2 && size(coord0, 1) == numSamples
        coord = coord0;
    else
        error('Coordinate data must be N x 2 or 2 x N.');
    end
end


function foldID = make_fold_index(numSamples, numFolds)

    order = randperm(numSamples);
    foldID = zeros(numSamples, 1);

    for i = 1:numSamples
        foldID(order(i)) = mod(i - 1, numFolds) + 1;
    end
end
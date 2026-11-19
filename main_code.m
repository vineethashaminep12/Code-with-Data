clc
clear
close all
warning off
addpath(genpath('.\Sub_code\'))
addpath(genpath('.\MatFiles\'))
%% Reading Dataset and preprocessing
 PP = dir('.\Dataset\');
for KK = 3
    PP1 = ['.\Dataset\',PP(KK).name];
    P1 = readtable(PP1);
    P_ar = readmatrix(PP1);
    %Missing value imputation
    P_ms=fillmissing(P_ar, 'constant', mean(P_ar, 'omitnan'));
    % Min-Max normalization for each column
    P_min = min(P_ms);    
    P_max = max(P_ms);     
    P_normalized = (P_ms - P_min) ./ (P_max - P_min);  
    %% Battery Usage Pattern Analysis
    %SOC and energy consumption calculation and kmeans clustering and feature extraction
    ime = P_normalized(:, 7);
    batteryCapacity = 100; 
    openCircuitVoltage = 12; 
    fullChargeVoltage = 14.4; 
    voltageSOC = @(V) (V - openCircuitVoltage) / (fullChargeVoltage - openCircuitVoltage) * 100;
    Char_vol= P_normalized(:,5);
    initialSOC = 50;
    Disch_vol=P_normalized(:,4);
    dischargeTime=P_normalized(:,2);
    chargeTime=P_normalized(:,8);
    numSamples = size(P_normalized, 1);
    SOH = zeros(numSamples, 1);
    chargeCurrent = zeros(numSamples, 1);
    dischargeCurrent = zeros(numSamples, 1);
    reminiscentEnergy = zeros(numSamples, 1);
    chargeRate = zeros(numSamples, 1);
    dischargeRate = zeros(numSamples, 1);
    selfDischargeRate = zeros(numSamples, 1);
    ambientTemperature = P_normalized(:, 9); 
    usableCapacity = batteryCapacity * 0.9;    
    energyConsumption = zeros(numSamples, 1);
    frequency = zeros(numSamples, 1);
    durations = zeros(numSamples, 1);
    % balancingTime = zeros(numSamples, 1); 
    balancingCurrentThreshold = 1;
    voltage = zeros(numSamples, 1);
    temperature = zeros(numSamples, 1);
    internalResistance = zeros(numSamples, 1);
    chargingCycle = zeros(numSamples, 1);
    dischargingCycle = zeros(numSamples, 1);
    chargeThreshold = 12; 
    dischargeThreshold = 10;
    A=10;
    B=20;
    for i=1:size(P_normalized,1)
        SOC(i) = calculateBatterySOC(Char_vol(i), chargeTime(i), dischargeTime(i), initialSOC, batteryCapacity, openCircuitVoltage, fullChargeVoltage);
        %energy consumption 
        energyConsumption(i) = Disch_vol(i) * dischargeTime(i) * batteryCapacity;
        if i > 1
            patternDegradation(i) = SOC(i) - SOC(i - 1);                 
        end
    end
     % clustering
    data = [SOC', energyConsumption]; 
    numClusters = 3; 
    maxIter = 100; 
    [clusters, centroids] = PF_Kmeans(data, numClusters, maxIter);
    % visualizing the clusters
    figure;
    gscatter(data(:,1), data(:,2), clusters); 
    hold on;
    plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
    title('Penrose Fossum K-means Clustering');
    xlabel('SOC (%)');
    ylabel('Energy Consumption (Wh)');
    grid on;
    %% Feature Extraction
    % all_features= Feat_Extract(clusters,SOC, SOH, chargeCurrent, dischargeCurrent, ...
    %                reminiscentEnergy, energyConsumption, chargeRate, ...
    %                dischargeRate, selfDischargeRate, ambientTemperature, ...
    %                patternDegradation,frequency,durations,...
    %                voltage,temperature,internalResistance,chargingCycle,dischargingCycle,P_normalized);
%%  
load features
inputs = features;
targets = P_ar(:, 9); 
nanIndices = isnan(inputs);
inputs(isnan(inputs)) = rand(sum(nanIndices(:)), 1);
minInputs = min(inputs, [], 1);
maxInputs = max(inputs, [], 1);
normInputs = (inputs - minInputs) ./ (maxInputs - minInputs);
nanNormIndices = isnan(normInputs);
normInputs(isnan(normInputs)) =rand(sum(nanNormIndices(:)), 1);
minTarget = min(targets);
maxTarget = max(targets);
normTargets = (targets - minTarget) / (maxTarget - minTarget);

% Split the data into training and testing sets
trainRatio = 0.8;
numTrain = round(trainRatio * size(normInputs, 1));

trainInputs = normInputs(1:numTrain, :); 
trainTargets = normTargets(1:numTrain); 
testInputs = normInputs(numTrain+1:end, :); 
testTargets = normTargets(numTrain+1:end);

trainInputsCell = cell(size(trainInputs, 1), 1);
for i = 1:size(trainInputs, 1)
    trainInputsCell{i} = trainInputs(i, :)'; 
end

testInputsCell = cell(size(testInputs, 1), 1);
for i = 1:size(testInputs, 1)
    testInputsCell{i} = testInputs(i, :)'; 
end
trainTargetsCell = num2cell(trainTargets);  
testTargetsCell = num2cell(testTargets);
trainTargetsCategorical = categorical(trainTargets);  
testTargetsCategorical = categorical(testTargets);

% Define RNN architecture with E-swish activation
inputSize = size(trainInputs, 2);
numHiddenUnits = 50;

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    ESwishLayer('e-swish', 1.5)         
    HuberRegressionLayer('huber', 1)     
];

% Define training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 20, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'Plots', 'training-progress');
% EswH_RNNnet = trainNetwork(trainInputsCell, trainTargets, layers, options);

end
if ~exist('Plant', 'var')
    load('Plant.mat');  
end
open_system('main');
%% 
load outtt.mat
load EswH_RNNnet 
signalNames={'avgSoh','avgSoc','avgTemperature','avgVoltage'};
timetableStruct = struct();
for i = 1:length(signalNames)
    time = out.get(signalNames{i}).time;
    time_as_duration = duration(time, 0, 0); 
    signalData = out.get(signalNames{i}).data; 
    signalData_reshaped = squeeze(signalData);
    timetableStruct.(signalNames{i}) = timetable(time_as_duration, signalData_reshaped);
end
simulink_features = [];
for i = 1:length(signalNames)   
    currentTimetable = timetableStruct.(signalNames{i});
    signal_values = currentTimetable{:, 1};
    signal_mean = mean(signal_values, 'omitnan');
    simulink_features = [simulink_features; signal_mean];
end
for i = 1:length(timeseriesSignals)
    ts = out.get(timeseriesSignals{i});
    ts_data = ts.Data;  
    ts_mean = mean(ts_data, 'omitnan');
    simulink_features = [simulink_features; ts_mean];
end
simulink_features_cell = {simulink_features};
predictedRUL_norm = predict(EswH_RNNnet, simulink_features_cell);
disp('Predicted RUL simulink:');
disp(predictedRUL_sim);

 
% Load dataset
data = readtable('AirQualityUCI.csv');

% Convert column to numeric (IMPORTANT FIX) series = str2double(data{:,3});

% Replace invalid values (-200) series(series == -200) = NaN;

% Remove missing values series = series(~isnan(series));

% Normalize
series = normalize(series);

% Create sequences X = series(1:end-1); Y = series(2:end);

XTrain = num2cell(X'); YTrain = num2cell(Y');

% LSTM network layers = [
sequenceInputLayer(1) lstmLayer(50) fullyConnectedLayer(1) regressionLayer
];

% Training options
options = trainingOptions('adam',
... 'MaxEpochs', 50, ... 'Plots','training-progress');
 
% Train
net = trainNetwork(XTrain, YTrain, layers, options);

% Predict
YPred = predict(net, XTrain);

YPred = cell2mat(YPred); YTrue = cell2mat(YTrain);

% Plot results figure; plot(YTrue, 'b'); hold on; plot(YPred, 'r');
legend('Actual','Predicted'); title('LSTM Time Series Prediction');

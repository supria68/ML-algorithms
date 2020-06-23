%% Linear regression with multiple variables
% The example helps in understanding the variance of cost function for changes in alpha

clear; close all; clc;

% Load data
data = load('ex1data2.txt'); % housing area, number of bed rooms Vs housing price
X = data(:,1:2); % features
y = data(:,3); % target
m = length(y) % size of training set

% Normalize the features
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

% Adding first column of 1's
X = [ones(m,1) X];

% Gradient descent algorithm

alpha = [0.3, 0.1, 0.03, 0.01] ; % lets choose 4 values of alpha

num_iters = 400;
figure;
plotColor = 'brgkmcyw';
theta = zeros(3, 1);
for i = 1:length(alpha),
  [theta, J_history] = gradientDescentMulti(X, y, theta, alpha(i), num_iters);
end
for m = 1:4,
  plot(1:numel(J_history(i)), J_history(i), plotColor(i), 'LineWidth', 2);
  hold on;
end
xlabel('Number of iterations');
ylabel('Cost J');




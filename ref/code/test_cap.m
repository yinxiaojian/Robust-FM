
% train metric on training data and evaluate on evaluation data. Finally evaluate on testing data. 
% Zhouyuan Huo 07/02/2016

clear; clc;
feature getpid

load('toy_2.mat');

dimensionality = 100;
target_rank = 15
k = dimensionality - target_rank;

%% initialization 
mu = 0;
M = zeros(dimensionality);
pars.initial_step = 1;
pars.max_iter = intmax;

disp('Training metric without regularization');
M_train = ML_cap(M, training,  mu, pars, k );

mu_values = [10.^([-2:2])];

disp('Training metric with capped norm regularization (and cross validating the regularization parameter)');
best_M_cap = 0;
best_accuracy_cap = 0;
for mu = mu_values
    [M_cap] = ML_cap(M_train, training,  mu, pars, k);
    acc_cap = evaluate_metric(M_cap, validation);
    if acc_cap >= best_accuracy_cap
        best_M_cap = M_cap;
        best_accuracy_cap = acc_cap;
        best_mu = mu;        
    end
end
disp('End of training');
fprintf('capped norm regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n', evaluate_metric(best_M_cap, test),leading_eigenvalues( best_M_cap ));
best_mu

noise = 0;

% add noise to test the robust
if noise > 0
    [out_train_X, out_train_Y] = add_noise(train_X, train_Y, noise);
else
    out_train_X = train_X;
    out_train_Y = train_Y;
end

% load data
training.train_X = out_train_X;
training.train_Y = out_train_Y;

validation.test_X = test_X;
validation.test_Y = test_Y;

% pack paras
pars.task = 'binary-classification';
% pars.task = 'multi-classification';
pars.iter_num = 1;
pars.epoch = 10;
pars.minibatch = 10;

% initial model
[~, p] = size(train_X);
class_num = max(train_Y);

%% ridge classifier
rng('default');
pars.reg = 1e-2;
pars.w0 = zeros(class_num, 1);
pars.W = zeros(class_num,p);

pars.learning_rate = 1e2;
pars.t0 = 1e5;

disp('Training ridge classifier...')
[model_ridge, metric_ridge] = ridge(training, validation, pars);
%% svm
rng('default');
pars.reg = 1e-3;
pars.w0 = zeros(class_num, 1);
pars.W = zeros(class_num,p);

pars.learning_rate = 1e2;
pars.t0 = 1e5;

disp('Training SVM...')
[model_svm, metric_svm] = svm(training, validation, pars);

%% fm
rng('default');

pars.reg = 1e-3;
pars.factors_num =10;

if strcmp(pars.task, 'binary-classification')
    pars.w0 = 0;
    pars.W = zeros(1,p);
    pars.V = 0.1*randn(p,pars.factors_num);
end

if strcmp(pars.task, 'multi-classification')
    pars.w0 = zeros(class_num, 1);
    pars.W = zeros(class_num,p);
    pars.V = 0.1*randn(class_num,p,pars.factors_num);
end


pars.learning_rate = 1e3;
pars.t0 = 1e5;

disp('Training FM...')
[model_fm, metric_fm] = fm(training, validation, pars);

%% capped norm
rng('default');
disp('Training with capped norm...')
pars.alpha = 1e-3;
pars.beta = 1e-3;

pars.epsilon1 = 1e-2;
pars.epsilon2 = 5;
pars.epsilon3 = 0.5;

if strcmp(pars.task, 'binary-classification')
    pars.w0 = 0;
    pars.W = zeros(1,p);
    pars.Z = zeros(p,p);
end

if strcmp(pars.task, 'multi-classification')
    pars.w0 = zeros(class_num, 1);
    pars.W = zeros(class_num,p);
    pars.Z = zeros(class_num,p,p);
end

pars.truncated_k = 10;

pars.learning_rate = 1e3;
pars.t0 = 1e5;

% pars.w0 = model_no_capped.w0;
% pars.W = model_no_capped.W;
% pars.Z = model_no_capped.Z; 

[model_capped, metric_capped] = capped_fm(training, validation, pars);

%% capped norm with increnmental svd
rng('default');
disp('Training with capped norm via inc SVD...')
pars.alpha = 1e-3;
pars.beta = 1e-3;

pars.epsilon1 = 1e-3;
pars.epsilon2 = 5;
pars.epsilon3 = 1e-3;
pars.minibatch = 20;
pars.epoch = 50;

pars.w0 = zeros(1);
pars.W = zeros(1,p);
pars.Z = zeros(p, p);

pars.truncated_k = 10;

pars.learning_rate = 1e3;
pars.t0 = 1e5;

% pars.w0 = model_no_capped.w0;
% pars.W = model_no_capped.W;
% pars.Z = model_no_capped.Z; 

[model_capped_inc, metric_capped_inc] = capped_fm_inc(training, validation, pars);
%% plot
% SVM
plot(metric_svm.loss_fm_test(1,:),'g--o','DisplayName','svm');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;
%%
% FM
plot(metric_fm.loss_fm_test(1,:),'b--o','DisplayName','fm');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('hinge loss');
grid on;
hold on;

%%
% robust FM
a = [0.3454, 0.3031, 0.2851, 0.2753, 0.2637, 0.2589, 0.2542, 0.2489, 0.2467, 0.2466];
% plot(metric_capped.loss_fm_test(1,:),'r--o','DisplayName','robust-fm');
plot(a,'r--o','DisplayName','robust-fm');
legend('-DynamicLegend');

%%
% rank on hinge loss for FM

%% w8a
rank = [2, 5, 10, 20, 30, 40, 50];
loss = [0.0300, 0.0269, 0.0245, 0.0237, 0.0236, 0.0241, 0.0247];

robust_x = [0, 50];
robust_loss = [0.0190, 0.0190];
xlabel('rank');
ylabel('hinge loss');
grid on;
hold on;
plot(rank, loss, 'b--o', 'DisplayName', 'fm');
legend('-DynamicLegend');
plot(robust_x, robust_loss, 'r--', 'DisplayName', 'robust-fm');
legend('-DynamicLegend');

%% protein
rank = [2, 5, 10, 20, 30, 40, 50];
loss = [0.4868, 0.4807, 0.4777, 0.4780, 0.4830, 0.4860, 0.4892];

robust_x = [0, 50];
robust_loss = [0.4671, 0.4671];
xlabel('rank');
ylabel('hinge loss');
grid on;
hold on;
plot(rank, loss, 'b--o', 'DisplayName', 'fm');
legend('-DynamicLegend');
plot(robust_x, robust_loss, 'r--', 'DisplayName', 'robust-fm');
legend('-DynamicLegend');

%% connect-4
rank = [2, 5, 10, 20, 30, 40, 50];
loss = [0.3319, 0.3006, 0.2787, 0.2713, 0.2700, 0.2760, 0.2892];

robust_x = [0, 50];
robust_loss = [0.2466, 0.2466];
xlabel('rank');
ylabel('hinge loss');
grid on;
hold on;
plot(rank, loss, 'b--o', 'DisplayName', 'fm');
legend('-DynamicLegend');
plot(robust_x, robust_loss, 'r--', 'DisplayName', 'robust-fm');
legend('-DynamicLegend');

%% noise on RMSE

% w8a
rank = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
loss = [0.0245, 0.0349, 0.3006, 0.2787, 0.2713, 0.2700];

robust_x = [0, 50];
robust_loss = [0.2466, 0.2466];
xlabel('rank');
ylabel('hinge loss');
grid on;
hold on;
plot(rank, loss, 'b--o', 'DisplayName', 'fm');
legend('-DynamicLegend');
plot(robust_x, robust_loss, 'r--', 'DisplayName', 'robust-fm');
legend('-DynamicLegend');
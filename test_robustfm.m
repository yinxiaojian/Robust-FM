% load data
training.train_X = train_X;
training.train_Y = train_Y;

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


pars.learning_rate = 1e2;
pars.t0 = 1e5;

disp('Training FM...')
[model_fm, metric_fm] = fm(training, validation, pars);

%% capped norm
rng('default');
disp('Training with capped norm...')
pars.alpha = 1e-3;
pars.beta = 1e-3;

pars.epsilon1 = 1e-2;
pars.epsilon2 = 10;
pars.epsilon3 = 1e-1;

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

pars.truncated_k = 5;

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

pars.epsilon1 = 1e-2;
pars.epsilon2 = 5;
pars.epsilon3 = 1e-3;
pars.minibatch = 10;

pars.w0 = zeros(1);
pars.W = zeros(1,p);
pars.Z = zeros(p, p);

pars.truncated_k = 5;

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

plot(metric_capped.loss_fm_test(1,:),'r--o','DisplayName','robust-fm');
legend('-DynamicLegend');
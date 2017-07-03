% standard factorization machine
% solved by SGD
% by Faramita @ZJU 2017.6.24

rng('default');

task = 'regression';

[num_sample, p] = size(train_X);

% parameters
iter_num = 1;
learning_rate = 1e4;
t0 = 1e5;
reg = 1e-4;

factors_num = 10;
  
epoch = 15;

loss_fm_test = zeros(iter_num, epoch);
loss_fm_train = zeros(iter_num, epoch);
accuracy_fm = zeros(iter_num, epoch);

for i=1:iter_num
    
    tic;
    
    w0 = 0;
    W = zeros(1,p);
    V = 0.1*randn(p,factors_num);
    
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);

    for t=1:epoch
        
        loss = 0;
        for j=1:num_sample
                
            X = X_train(j,:);
            y = Y_train(j,:);

            nz_idx = find(X);

            tmp = sum(repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:));
            factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(V(nz_idx,:).^2))))/2;
            y_predict = w0 + W(nz_idx)*X(nz_idx)' + factor_part;

            idx = (t-1)*num_sample + j;
            % SGD update
            if strcmp(task, 'classification')
                err = sigmf(y*y_predict,[1,0]);
                loss = loss - log(err);
                
                w0_ = learning_rate / (idx + t0) * ((err-1)*y);
                w0 = w0 - w0_;
                W_ = learning_rate / (idx + t0) * ((err-1)*y*X(nz_idx) + 2 * reg * W(nz_idx));
                W(nz_idx) = W(nz_idx) - W_;
                V_ = learning_rate / (idx + t0) * ((err-1)*y*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*V(nz_idx,:),length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:))) + 2 * reg * V(nz_idx,:));
                V(nz_idx,:) = V(nz_idx,:) - V_;
            end
            
            if strcmp(task, 'regression')
                err = y_predict - y;
                loss = loss + err^2;
                
                w0_ = learning_rate / (idx + t0) * (2 * err);
                w0 = w0 - w0_;
                W_ = learning_rate / (idx + t0) * (2 * err *X(nz_idx) + 2 * reg * W(nz_idx));
                W(nz_idx) = W(nz_idx) - W_;
                V_ = learning_rate / (idx + t0) * (2 * err *(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*V(nz_idx,:),length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:))) + 2 * reg * V(nz_idx,:));
                V(nz_idx,:) = V(nz_idx,:) - V_;
            end
            
        end
        
        loss_fm_train(i,t) = loss / num_sample;
        fprintf('[iter %d epoch %2d]---train loss:%.4f\t',i, t, loss_fm_train(i,t));
    
        % validate
        loss = 0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        for k=1:num_sample_test

            X = test_X(k,:);
            y = test_Y(k,:);
            nz_idx = find(X);

            tmp = sum(repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:)) ;
            factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(V(nz_idx,:).^2))))/2;
            y_predict = w0 + W(nz_idx)*X(nz_idx)' + factor_part;

            if strcmp(task, 'classification')
                err = sigmf(y*y_predict,[1,0]);
                loss = loss - log(err);

                if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                    correct_num = correct_num + 1;
                end
            end

            if strcmp(task, 'regression')
                err = y_predict - y;
                loss = loss + err^2;
            end

        end

        loss_fm_test(i,t) = loss / num_sample_test;
        fprintf('test loss:%.4f\t', loss_fm_test(i,t));
        if strcmp(task, 'classification')
            accuracy_fm(i,t) = correct_num/num_sample_test;
            fprintf('test accuracy:%.4f', accuracy_fm(i,t));
        end

        fprintf('\n');
    
    end
end


%%
% plot
plot(mse_fm_sgd,'DisplayName','FM');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE');
grid on; 
hold on;  

%%
plot(rmse_fm_test(1,:) ,'k--o','DisplayName','FM');
legend('-DynamicLegend');
% title('Learning Curve on Test Dataset')
hold on;
% plot(rmse_fm_test,'DisplayName','FM\_Test');  
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('logloss');
% legend('FM_Train','FM_Test');
% title('FM\_SGD');
grid on;
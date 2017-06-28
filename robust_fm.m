% robust factorization machine
% solved by SGD
% by Faramita @ZJU 2017.6.24

rng('default');

task = 'classification';

[num_sample, p] = size(train_X);

% parameters
iter_num = 1;
learning_rate = 1e4;
t0 = 1e5;

alpha = 1e-1;
beta = 1e-1;

epoch = 15;

% capped trace norm threshold
epsilon1 = 5;
epsilon2 = 1e2;

loss_fm_test = zeros(iter_num, epoch);
loss_fm_train = zeros(iter_num, epoch);
accuracy_fm = zeros(iter_num, epoch);

for i=1:iter_num
    
    tic;
    
    w0 = 0;
    W = zeros(1,p);
    
    Z = generateSPDmatrix(p);
    
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);

    for t=1:epoch
        
        loss = 0;
        for j=1:num_sample
                
            X = X_train(j,:);
            y = Y_train(j,:);

%             nz_idx = find(X);
            y_predict = w0 + W*X' + sum(sum(X'*X.*Z))/2;

            idx = (t-1)*num_sample + j;
            
            % SGD update
            if strcmp(task, 'classification')
                
                % log loss
                err = sigmf(y*y_predict,[1,0]);
                loss = loss - log(err);
                
                % compute d
                bias = y_predict - y;
                
%                 if abs(bias) < epsilon1
%                     d = 1/(bias^2);
%                 else
%                     d = 0;
%                 end

                d = 1;
                
                w0_ = learning_rate / (idx + t0) * (d * (err-1)*y);
                w0 = w0 - w0_;
                W_ = learning_rate / (idx + t0) * (d * (err-1)*y*X + alpha * W);
                W = W - W_;
                
                % truncated SVD
%                 [U,S,V] = truncated_svd(Z, epsilon2);
%                 Z_ = learning_rate / (idx + t0) * (d * (err-1)*y.*(X'*X)+beta * U' * V .* Z);
                Z_ = learning_rate / (idx + t0) * (d * (err-1)*y.*(X'*X));
                Z = Z- Z_;
            end
            
            if strcmp(task, 'regression')
                err = y_predict - y;
                loss = loss + err^2;
                
                w0_ = learning_rate / (idx + t0) * (2 * err);
                w0 = w0 - w0_;
                W_ = learning_rate / (idx + t0) * (2 * err * X(nz_idx) + 2 * alpha * W(nz_idx));
                W = W - W_;
%                 V_ = learning_rate / (idx + t0) * (2 * err *y*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*V(nz_idx,:),length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:))) + 2 * alpha * V(nz_idx,:));
%                 V(nz_idx,:) = V(nz_idx,:) - V_;
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
%             nz_idx = find(X);

            y_predict = w0 + W*X' + sum(sum(X'*X.*Z))/2;

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
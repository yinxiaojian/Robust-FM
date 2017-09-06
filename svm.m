function [ model, metric ] = svm( training, validation, pars )
%FM Summary of this function goes here
%   Detailed explanation goes here

    task = pars.task;
    train_X = training.train_X;
    train_Y = training.train_Y;
    
    test_X = validation.test_X;
    test_Y = validation.test_Y;

    [num_sample, ~] = size(train_X);

    % parameters
    iter_num = pars.iter_num;
    learning_rate = pars.learning_rate;
    reg = pars.reg;
    t0 = pars.t0;
    
%     factors_num = pars.factors_num;

    epoch = pars.epoch;

    class_num = max(train_Y);

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);

    for i=1:iter_num

        tic;

        w0 = pars.w0;
        W = pars.W;
%         V = pars.V;
        
        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);

        for t=1:epoch

            loss = 0;
            for j=1:num_sample

                X = X_train(j,:);
                if strcmp(task, 'binary-classification')
                    y = Y_train(j,:);
                end

                if strcmp(task, 'multi-classification')
                    y = -ones(1, class_num);
                    y(Y_train(j,:)) = 1;
                end

                nz_idx = find(X);
 
%                 tmp = sum(repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:));
%                 factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(V(nz_idx,:).^2))))/2;
%                 y_predict = w0 + W(nz_idx)*X(nz_idx)' + factor_part;
                

                idx = (t-1)*num_sample + j;
                
                if strcmp(task, 'binary-classification')
                    y_predict = W(nz_idx)*X(nz_idx)' + w0;
                end

                if strcmp(task, 'multi-classification')
                    y_predict = zeros(1, class_num);
                    for u = 1:class_num
                        y_predict(u) = w0(u) + W(u,nz_idx)*X(nz_idx)';
                    end
                end
                % SGD update
                if strcmp(task, 'binary-classification')
                    err = max(0, 1-y*y_predict);
                    loss = loss + err;
                    
                    if err > 0
                        w0_ = learning_rate / (idx + t0) * (-y);
                        w0 = w0 - w0_;
                        W_ = learning_rate / (idx + t0) * (-y*X(nz_idx) + 2 * reg * W(nz_idx));
                        W(nz_idx) = W(nz_idx) - W_;
                    end
                end

                if strcmp(task, 'multi-classification')
                    err = max(0, 1-y.*y_predict);
                    loss = loss + sum(err);
                    
                    for u=1:class_num
                        if err(u) > 0
                            w0_ = learning_rate / (idx + t0) * (-y(u));
                            w0(u) = w0(u) - w0_;
                            W_ = learning_rate / (idx + t0) * (-y(u)*X(nz_idx) + 2 * reg * W(u,nz_idx));
                            W(u,nz_idx) = W(u,nz_idx) - W_;
                        end
                    end
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
                if strcmp(task, 'binary-classification')
                    y = test_Y(k,:);
                end

                if strcmp(task, 'multi-classification')
                    y = -ones(1, class_num);
                    y(test_Y(k,:)) = 1;
                end

                nz_idx = find(X);
%                 tmp = sum(repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:)) ;
%                 factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(V(nz_idx,:).^2))))/2;
%                 y_predict = w0 + W(nz_idx)*X(nz_idx)' + factor_part;

                if strcmp(task, 'binary-classification')
                    y_predict = W*X' + w0;
                    err = max(0, 1-y_predict*y);
                    loss = loss + err;

                    if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                        correct_num = correct_num + 1;
                    end
                end

                if strcmp(task, 'multi-classification')
                    y_predict = zeros(1, class_num);
                    for u = 1:class_num
                        y_predict(u) = w0(u) + W(u,nz_idx)*X(nz_idx)';
                    end

                    err = max(0, 1-y.*y_predict);
                    loss = loss + sum(err);

                    [~, label] = max(y_predict);
                    
                    % accuracy
                    if label == test_Y(k,:)
                        correct_num = correct_num + 1;
                    end
                end
            end

            loss_fm_test(i,t) = loss / num_sample_test;
            fprintf('test loss:%.4f\t', loss_fm_test(i,t));
            accuracy_fm(i,t) = correct_num/num_sample_test;
            fprintf('\ttest accuracy:%.4f', accuracy_fm(i,t));

            fprintf('\n');

        end
        
        toc;
    end
    
    % pack output
    % model
    model.w0 = w0;
    model.W = W;
    
    % metric
    metric.loss_fm_train = loss_fm_train;
    metric.loss_fm_test = loss_fm_test;
    metric.loss_fm_accuracy = accuracy_fm;

end
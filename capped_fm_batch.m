function [ model, metric ] = capped_fm_batch( training, validation, pars)
%CAPPED_FM Summary of this function goes here
% 
%   Detailed explanation goes here
    task = pars.task;
    train_X = training.train_X;
    train_Y = training.train_Y;
    
    test_X = validation.test_X;
    test_Y = validation.test_Y;

    [num_sample, p] = size(train_X);

    % parameters
    iter_num = pars.iter_num;
    learning_rate = pars.learning_rate;
    t0 = pars.t0;

    alpha = pars.alpha;
    beta = pars.beta;

    epoch = pars.epoch;
    minibatch = pars.minibatch;
    truncated_k = pars.truncated_k;

    % capped trace norm threshold
    epsilon1 = pars.epsilon1;
    epsilon2 = pars.epsilon2;
    epsilon3 = pars.epsilon3;

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);
    rank_fm = zeros(iter_num, epoch);
    outlier_fm = zeros(iter_num, epoch);
    noise_fm = zeros(iter_num, epoch);
    obj_fm = zeros(iter_num, epoch);

    rng('default');
    for i=1:iter_num

        tic;
    
        w0 = pars.w0;
        W = pars.W;
        Z = pars.Z;

        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);

        for t=1:epoch

            loss = 0;
            rank = 0;
            outlier = 0;
            noise = 0;
            
            for b=1:num_sample/minibatch + 1
                g_1 = 0;
                g_2 = zeros(1, p);
                g_3 = zeros(p, p);
                
                if b*minibatch > num_sample
                    batch_end = num_sample;
                else
                    batch_end = b*minibatch;
                end
                for j=(b-1)*minibatch + 1: batch_end

                    X = X_train(j,:);
                    y = Y_train(j,:);
                    nz_idx = find(X);

                    y_predict = w0 + W(nz_idx)*X(nz_idx)' + sum(sum(X(nz_idx)'*X(nz_idx).*Z(nz_idx,nz_idx)));
    %                 y_predict = w0 + W(nz_idx)*X(nz_idx)';
    %                 y_predict = sum(sum(X(nz_idx)'*X(nz_idx).*Z(nz_idx,nz_idx)));
                    err = y_predict - y;
                    loss = loss + err^2;
                    g_1 = g_1 + 2 * err;
                    g_2(nz_idx) = g_2(nz_idx) + 2 * err *X(nz_idx);
                    g_3(nz_idx, nz_idx) = g_3(nz_idx, nz_idx) + 2 * err * (X(nz_idx)'*X(nz_idx));
                end
                
                % batch update
                w0_ = learning_rate / t0 / minibatch * g_1;
                w0 = w0 - w0_;
                W_ = learning_rate / t0 / minibatch * (g_2 + alpha * W);
                W = W - W_;

                % truncated SVD
                [U, ~, ~] = truncated_svd_fix(Z, truncated_k);
                rank = rank + truncated_k;
                Z_ = learning_rate / t0  * 1e2 * (g_3 / minibatch + beta * (eye(p) - U*U') .* Z);
                Z = Z - Z_;
                % project on PSD cone!
                Z = psd_cone(Z);
                
            end

            

            loss_fm_train(i,t) = (loss / num_sample);
            if strcmp(task, 'regression')
                loss_fm_train(i, t) = loss_fm_train(i, t)^0.5;
            end
            rank_fm(i, t) = rank;
            outlier_fm(i,t) = outlier/num_sample;
            noise_fm(i, t) = noise/num_sample;
            
            fprintf('[iter %d epoch %2d]---train loss:%.4f\t',i, t, loss_fm_train(i,t));  
            % validate
            loss = 0;
            correct_num = 0;
            [num_sample_test, ~] = size(test_X);
            for k=1:num_sample_test

                X = test_X(k,:);
                y = test_Y(k,:);
                nz_idx = find(X);
                y_predict = w0 + W(nz_idx)*X(nz_idx)' + sum(sum(X(nz_idx)'*X(nz_idx).*Z(nz_idx,nz_idx)));
%                 y_predict = w0 + W(nz_idx)*X(nz_idx)';
%                 y_predict = sum(sum(X(nz_idx)'*X(nz_idx).*Z(nz_idx,nz_idx)));
                if strcmp(task, 'regression')
                    err = y_predict - y;
                    loss = loss + err^2;
                end
                if strcmp(task, 'binary-classification')
                    err = max(0, 1-y_predict*y);
                    loss = loss + err;

                    if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                        correct_num = correct_num + 1;
                    end
                end

            end

            loss_fm_test(i,t) = loss / num_sample_test;
            if strcmp(task, 'regression')
                loss_fm_test(i,t) = loss_fm_test(i,t)^0.5;
            end
            
            fprintf('test loss:%.4f\taverage rank:%7.4f\toutlier percentage:%.4f\noise percentage:%.4f\tobj:%.4f\t', loss_fm_test(i,t), rank_fm(i,t), outlier_fm(i,t), noise_fm(i,t), obj_fm(i,t));
%             fprintf('test loss:%.4f\t', loss_fm_test(i,t));
            
            if strcmp(task, 'binary_classification')
                accuracy_fm(i,t) = correct_num/num_sample_test;
                fprintf('test accuracy:%.4f', accuracy_fm(i,t));
            end

            fprintf('\n');

        end
        
        toc;
    end
    
    % pack output
    % model
    model.w0 = w0;
    model.W = W;
    model.Z = Z;
    
    % metric
    metric.loss_fm_train = loss_fm_train;
    metric.loss_fm_test = loss_fm_test;
    metric.loss_fm_accuracy = accuracy_fm;
    

end
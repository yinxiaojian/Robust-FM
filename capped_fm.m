function [ model, metric ] = capped_fm( training, validation, pars)
%CAPPED_FM Summary of this function goes here
% 
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
    t0 = pars.t0;

    alpha = pars.alpha;
    beta = pars.beta;

    epoch = pars.epoch;

    % capped trace norm threshold
    epsilon1 = pars.epsilon1;
    epsilon2 = pars.epsilon2;

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);
    rank_fm = zeros(iter_num, epoch);
    outlier_fm = zeros(iter_num, epoch);
    obj_fm = zeros(iter_num, epoch);

    for i=1:iter_num

        tic;

%         w0 = 0;
%         W = zeros(1,p);

    %     Z = generateSPDmatrix(p);
%         Z = zeros(p);
    
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
            obj = 0.0;
            
            for j=1:num_sample

                X = X_train(j,:);
                y = Y_train(j,:);
                y_predict = w0 + W*X' + sum(sum(X'*X.*Z));

                idx = (t-1)*num_sample + j;

                % SGD update
                if strcmp(task, 'classification')

                    % log loss
%                     err = sigmf(y*y_predict,[1,0]);
%                     loss = loss - log(err);
% 
%                     w0_ = learning_rate / (idx + t0) * (d * (err-1)*y);
%                     w0 = w0 - w0_;
%                     W_ = learning_rate / (idx + t0) * (d * (err-1)*y*X + alpha * W);
%                     W = W - W_;
% 
%                     % truncated SVD
%                     [U,S,V] = truncated_svd(Z, epsilon2);
%                     Z_ = learning_rate / (idx + t0) * (d * (err-1)*y.*(X'*X)+beta * U' * V .* Z);
%     %                 Z_ = learning_rate / (idx + t0) * (d * (err-1)*y.*(X'*X));
%                     Z = Z - Z_;
                end

                if strcmp(task, 'regression')
                    err = y_predict - y;
                    loss = loss + err^2;
                    

                    % capped norm
                    if beta ~= 0
                        
                        if abs(err) < epsilon1
                            d = 1;
                        else
                            d = 0;
                            outlier = outlier + 1;
                        end
%                         d = 1;

                        if d ~=0
                            w0_ = learning_rate / (idx + t0) * (d * 2 * err);
                            w0 = w0 - w0_;
                            W_ = learning_rate / (idx + t0) * (d * 2 * err *X + alpha * W);
                            W = W - W_;
                            

                            % truncated SVD
                            [U,S,r] = truncated_svd(Z, epsilon2);
                            rank = rank + r;
                            
                            obj = obj + err^2 + alpha/2*(W*W')+beta/2*trace(U*(Z*Z')*U');
                            
                            Z_ = learning_rate / (idx + t0) * (d * 2 * err *(X'*X)+beta * (U'*U) .* Z);
                            Z = Z - Z_;

                            % project on PSD cone!
                            [UU, D, VV] = svd(Z);
                            d = real(diag(D));
                            d(d < 0) = 0;
                            Z =(UU * diag(d) * VV');
                            
                        end
                        
                    % no capped norm    
                    else
                        w0_ = learning_rate / (idx + t0) * (2 * err);
                        w0 = w0 - w0_;
                        W_ = learning_rate / (idx + t0) * (2 * err *X + alpha * W);
                        W = W - W_;

                        Z_ = learning_rate / (idx + t0) * (2 * err .*(X'*X));
                        Z = Z - Z_;
                    end
                    
                end

            end

            loss_fm_train(i,t) = loss / num_sample;
            rank_fm(i, t) = rank/num_sample;
            outlier_fm(i,t) = outlier/num_sample;
            obj_fm(i,t) = obj/num_sample;
            
            fprintf('[iter %d epoch %2d]---train loss:%.4f\t',i, t, loss_fm_train(i,t));  

            % validate
            loss = 0;
            correct_num = 0;
            [num_sample_test, ~] = size(test_X);
            for k=1:num_sample_test

                X = test_X(k,:);
                y = test_Y(k,:);
    %             nz_idx = find(X);

                y_predict = w0 + W*X' + sum(sum(X'*X.*Z));

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
            if beta~=0
                fprintf('test loss:%.4f\taverage rank:%.4f\toutlier percentage:%.4f\tobj:%.4f', loss_fm_test(i,t), rank_fm(i,t), outlier_fm(i,t), obj_fm(i,t));
            else
                fprintf('test loss:%.4f\t', loss_fm_test(i,t));
            end
            
            if strcmp(task, 'classification')
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


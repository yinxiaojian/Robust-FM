function [ model, metric ] = capped_fm_inc( training, validation, pars)
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
    minibatch = pars.minibatch;

    % capped trace norm threshold
    epsilon1 = pars.epsilon1;
    epsilon2 = pars.epsilon2;
    epsilon3 = pars.epsilon3;
    
    truncated_k = pars.truncated_k;

    loss_fm_test = zeros(iter_num, epoch);
    loss_fm_train = zeros(iter_num, epoch);
    accuracy_fm = zeros(iter_num, epoch);
    rank_fm = zeros(iter_num, epoch);
    outlier_fm = zeros(iter_num, epoch);
    noise_fm = zeros(iter_num, epoch);
    obj_fm = zeros(iter_num, epoch);

    for i=1:iter_num

        tic;
    
        w0 = pars.w0;
        W = pars.W;
        Z = pars.Z;

        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);
        
        % we need use truncated svd to calculate initial svd
        first = 1;

        for t=1:epoch

            loss = 0;
            rank = 0;
            outlier = 0;
            noise = 0;
            obj = 0.0;
            
            batch_round = floor(num_sample/minibatch);
            
            for j=1:batch_round

                g_1 = 0;
                g_2 = 0;
                g_3 = 0;
                A = 0;
                
                skip = 1;
                
                for jj = (j-1) * minibatch + 1 : j * minibatch
                    X = X_train(jj,:);
                    y = Y_train(jj,:);
                    y_predict = w0 + W*X' + sum(sum(X'*X.*Z));
                    
                    if strcmp(task, 'binary-classification')

                        % hinge loss
                        err = max(0, 1-y*y_predict);
                        loss = loss + err;
                        

                        if err > epsilon1 && err < epsilon1 + epsilon2
                            d = 1/2/(err - epsilon1);
%                             obj = obj + d*(err-epsilon1)^2; 
                            
                            g_1 = g_1 - y;
                            g_2 = g_2 - y*X;
                            g_3 = g_3 - y*(X'*X);
                            
                            if A == 0
                                A = X';
                            else
                                A = [A X'];
                            end
                    
                            skip = 0;
                        elseif err <= epsilon1
%                             d = 0;
                            noise = noise + 1;
                        else
%                             d = 0;
                            outlier = outlier + 1;
                        end
                        
                    end
                    
                end
                
                % batch sgd
                idx = (t-1)*batch_round + j;
                % SGD update
                if strcmp(task, 'binary-classification')
                    % capped norm
                    if beta ~= 0
                        
                        if ~skip
                            w0_ = learning_rate / (idx + t0)*(g_1);
                            w0 = w0 - w0_;
                            W_ = learning_rate / (idx + t0) * (g_2 + alpha * W);
                            W = W - W_;
                            
                            % truncated SVD
                            [U,~,r] = truncated_svd(Z, epsilon3);
                            % [U,~,~] = truncated_svd(Z, epsilon3);
                            rank = rank + r;
%                             if first
%                                 [U, S,~] = truncated_svd(Z, truncated_k);
% %                                 first = 0;
% %                             else
% %                                 [U, S] = incremental_svd(Z, A, U, S, learning_rate / (idx + t0));
%                             end

                            
                            
%                             [P, r] = svdsecon(Z, epsilon3);
%                             rank = rank + r;
                            
%                             obj = obj + alpha/2*(W*W')+beta/2*trace(U'*(Z*Z')*U);
                            
%                             P = U*U';
%                             tmp = size(P,1);
                            Z_ = learning_rate / (idx + t0) * (g_3+beta * (U' * U) .* Z);
%                             Z_ = learning_rate / (idx + t0) * (-y*(X'*X)+beta * P .* Z);
                            Z = Z - Z_;

                            % project on PSD cone!
                            Z = psd_cone(Z);
%                             if first==1
%                                 Z = psd_cone(Z);
%                                 first = 0;
%                             else
%                                 S(S<0) = 0;
%                                 Z = U * S * U';
%                             end
                            
%                             if first == 0
%                                 A = [A U*sqrt(S)*U'];
% %                                 [U, S] = incremental_svd(Z, A, U, S, learning_rate / (idx + t0));
%                                 [U, S] = Incsvd(U, S, -sqrt(learning_rate / (idx + t0))*A);
%                             end
                            
                        end
                    end

                end
            end

            loss_fm_train(i,t) = loss / num_sample;
            rank_fm(i, t) = rank/(batch_round);
            outlier_fm(i,t) = outlier/num_sample;
            noise_fm(i, t) = noise/num_sample;
            obj_fm(i,t) = obj/(num_sample-outlier);
            
            fprintf('[iter %d epoch %2d]---train loss:%.4f\t',i, t, loss_fm_train(i,t));  

            % validate
            loss = 0;
            correct_num = 0;
            [num_sample_test, ~] = size(test_X);
            for k=1:num_sample_test

                X = test_X(k,:);
                y = test_Y(k,:);

                y_predict = w0 + W*X' + sum(sum(X'*X.*Z));

                if strcmp(task, 'binary-classification')
%                     err = sigmf(y*y_predict,[1,0]);
                    err = max(0, 1-y_predict*y);
                    loss = loss + err;

                    if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                        correct_num = correct_num + 1;
                    end
                end

            end

            loss_fm_test(i,t) = loss / num_sample_test;
            if beta~=0
                fprintf('test loss:%.4f\taverage rank:%7.4f\toutlier percentage:%.4f\noise percentage:%.4f\tobj:%.4f\t', loss_fm_test(i,t), rank_fm(i,t), outlier_fm(i,t), noise_fm(i,t), obj_fm(i,t));
            else
                fprintf('test loss:%.4f\t', loss_fm_test(i,t));
            end
            
            accuracy_fm(i,t) = correct_num/num_sample_test;
            fprintf('test accuracy:%.4f', accuracy_fm(i,t));

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


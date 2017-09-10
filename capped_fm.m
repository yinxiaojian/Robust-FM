function [ model, metric ] = capped_fm( training, validation, pars)
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
    class_num = max(train_Y);
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
            obj = 0.0;
            
            for j=1:num_sample

                X = X_train(j,:);
                if strcmp(task, 'binary-classification')
                    y = Y_train(j,:);
                end

                if strcmp(task, 'multi-classification')
                    y = -ones(1, class_num);
                    y(Y_train(j,:)) = 1;
                end


                if strcmp(task, 'binary-classification')
                    y_predict = w0 + W*X' + sum(sum(X'*X.*Z));
                end

                if strcmp(task, 'multi-classification')
                    y_predict = zeros(1, class_num);
                    for u = 1:class_num
                        y_predict(u) = w0(u) + W(u,:)*X';
                    end
                end

                idx = (t-1)*num_sample + j;
                
                

                % SGD update
                if strcmp(task, 'binary-classification')
                    
                    % hinge loss
                    err = max(0, 1-y*y_predict);
                    loss = loss + err;
                        
                    if err > epsilon1 && err < epsilon1 + epsilon2
                        d = 1/2/(err - epsilon1);
                    elseif err <= epsilon1
                        d = 0;
                        noise = noise + 1;
                    else
                        d = 0;
                        outlier = outlier + 1;
                    end
%                         d = 1;

                    if d ~=0
                        w0_ = learning_rate / (idx + t0)*(-y);
                        w0 = w0 - w0_;
                        W_ = learning_rate / (idx + t0) * (-y*X + alpha * W);
                        W = W - W_;
                        

                        % truncated SVD
                        [U,~,r] = truncated_svd(Z, epsilon3);
                        rank = rank + r;
%                         [U,~,~] = truncated_svd(Z, epsilon3);
%                         [U, ~, ~] = truncated_svd_fix(Z, truncated_k);
%                         rank = rank + truncated_k;
                        
%                         obj = obj + d*(err-epsilon1)^2 + alpha/2*(W*W')+beta/2*trace(U*(Z*Z')*U');
                        
                        Z_ = learning_rate / (idx + t0) * (-y*(X'*X)+beta * (U'*U) .* Z);
                        Z = Z - Z_;

                        % project on PSD cone!
                        Z = psd_cone(Z);
                        
                    end

                end

                if strcmp(task, 'multi-classification')
                    
                    % hinge loss
                    err = max(0, 1-y.*y_predict);
                    loss = loss + sum(err);
                    
                    for u=1:class_num
                        if err(u) > epsilon1 && err(u) < epsilon1 + epsilon2
                            d = 1/2/(err(u) - epsilon1);
                        elseif err <= epsilon1
                            d = 0;
                            noise = noise + 1;
                        else
                            d = 0;
                            outlier = outlier + 1;
                        end
    %                         d = 1;

                        if d ~=0
                            w0_ = learning_rate / (idx + t0)*(-y(u));
                            w0(u) = w0(u) - w0_;
                            W_ = learning_rate / (idx + t0) * (-y(u)*X + alpha * W(u,:));
                            W(u,:) = W(u,:) - W_;
                            

                            % truncated SVD
%                             [U,~,r] = truncated_svd(squeeze(Z(u,:,:)), epsilon3);
%                             rank = rank + r;
                            [U,~,~] = truncated_svd_fix(squeeze(Z(u,:,:)), truncated_k);
                            rank = rank + truncated_k;
                            
                            % obj = obj + d*(err-epsilon1)^2 + alpha/2*(W*W')+beta/2*trace(U*(Z*Z')*U');
                            
                            Z_ = learning_rate / (idx + t0) * (-y(u)*(X'*X)+beta * (U*U') .* squeeze(Z(u,:,:)));
                            Z(u,:,:) = squeeze(Z(u,:,:)) - Z_;

                            % project on PSD cone!
%                             Z(u,:,:) = psd_cone(squeeze(Z(u,:,:)));
                            
                        end    
                    end
                    

                end

            end

            loss_fm_train(i,t) = loss / num_sample;
            rank_fm(i, t) = rank/(num_sample-outlier-noise);
            outlier_fm(i,t) = outlier/num_sample;
            noise_fm(i, t) = noise/num_sample;
            obj_fm(i,t) = obj/(num_sample-outlier-noise);
            
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

                if strcmp(task, 'binary-classification')
                    y_predict = w0 + W*X' + sum(sum(X'*X.*Z));
                    err = max(0, 1-y_predict*y);
                    loss = loss + err;

                    if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                        correct_num = correct_num + 1;
                    end
                end

                if strcmp(task, 'multi-classification')
                    y_predict = zeros(1, class_num);
                    for u = 1:class_num
                        y_predict(u) = w0(u) + W(u,:)*X';
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
% This code is modified from code of paper: Fantope Regularization in Metric Learning.  url: http://www-poleia.lip6.fr/~lawm/projects/cvpr2014/
% Zhouyuan Huo 07/02/2016

%% capped norm: capped_norm(M) =  trace(M^TWM)    
function [ M ] = ML_cap( initial_matrix, training,  mu,  pars, k, validation )
    % compute analysis result in the optimization. 
    
    M = initial_matrix;
    
    nb_training_constraints = size(training.smaller,1);
    C = 1 / nb_training_constraints;
    old_active_constraints = false(nb_training_constraints,1);
    Kgradient = 0;
    obj = intmax;
    best_obj = intmax;
    best_M = M;
    max_iter = 200;
    step_size = pars.initial_step;
    best_acc_val = 0;
    iter = 0;
    
    while iter < max_iter
        iter = iter + 1;
        
        if iter < 10 && mu ~= 0
            epsilon = compute_eps_from_M(M, k);       
    	    W = compute_W_from_M(M,epsilon);
    	elseif mu ~= 0 
            W = compute_W_from_M(M,epsilon);
        end
        
        if (iter > pars.max_iter)
            break;
        end
        

        D_ij =sum((training.larger * M) .* training.larger,2);
        D_kl =sum((training.smaller * M) .* training.smaller,2);
        relative_distances = 1 - D_ij + D_kl;
        bool_distances = relative_distances > 0;
        nb_active_constraints = sum(bool_distances);
        new_active_constraints = (~old_active_constraints & bool_distances);
        new_inactive_constraints = (old_active_constraints & ~bool_distances);
        
        
        old_obj = obj;
        obj = C * sum(relative_distances(bool_distances));
        
        if mu
            obj = obj + 0.5*mu*trace(M' * W * M) ;
        end
        
        obj = real(obj);
        if (abs(old_obj - obj) < 1e-11)
            break;
        end 
        
        if (old_obj < obj)
            step_size = step_size * 0.9;
        else 
            step_size = step_size * 1.05;
        end

        if (best_obj > obj)
            best_obj = obj;
            best_M = M;
        end      
        
		fprintf('iter %d, obj %f\n', iter, obj);
        nb_new_active_constraitns = sum(new_active_constraints) + sum(new_inactive_constraints);
        
        if nb_new_active_constraitns
            if nb_active_constraints < nb_new_active_constraitns
                g1 = training.larger(bool_distances,:);
                g2 = training.smaller(bool_distances,:);
                Kgradient = C.* (g2'*g2 - g1'*g1);
            else
                g3 = training.larger(new_active_constraints,:);
                g4 = training.smaller(new_active_constraints,:);
                g5 = training.larger(new_inactive_constraints,:);
                g6 = training.smaller(new_inactive_constraints,:);
                Kgradient = Kgradient + C.* (g4'*g4 - g3'*g3  - g6'*g6 + g5'*g5);
            end
        end
                
	if mu 
            gradient_M = mu*W*M + Kgradient;
	else
	    gradient_M = Kgradient;
    	end
        M = M - step_size * gradient_M;
        
        % project on PSD cone!
        [UU, D, VV] = svd(M);
        d = real(diag(D));
        d(d < 0) = 0;
        M =(UU * diag(d) * VV');
        
    end
    M = best_M;
end

function [W] = compute_W_from_M(M,epsilon)
    [U,S,~] = svd(M);
    sigmas = diag(S);
    index = find(sigmas <= epsilon);
    new_S = zeros(length(sigmas),1);
    new_S(index) = 0.5*(sigmas(index).^2+eps).^(-0.5);
    W = U*diag(new_S)*U';
end

function epsilon = compute_eps_from_M(M,k)
    [~,D,~] = svd(M);
    [sigmas,~] = sort(diag(D),'ascend');
    epsilon = sigmas(k);
end


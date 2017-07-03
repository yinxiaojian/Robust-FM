% This code is from paper: Fantope Regularization in Metric Learning. 
% url: http://www-poleia.lip6.fr/~lawm/projects/cvpr2014/

function [ T, X, training_constraints, validation_constraints, test_constraints ] = create_toy_dataset( dimensionality, target_rank, number_of_samples, training_size, validation_size, test_size )
    tic;
    % creating samples
    X = rand(number_of_samples, dimensionality);
    
    % creating target distance matrix T
    L = (rand(target_rank) - 0.5) * 10;
    A = L'*L;
    [V,D] = eig(A);
    D = randg(target_rank,1) + 10;
    D = diag(D);
    A = V * D * V';

    L = sqrt(D) * V';
    L = [L, zeros(target_rank, dimensionality - target_rank)];
    L_t = L';
    T = zeros(dimensionality);
    T(1:target_rank, 1:target_rank) = A;
    
    %{
     L = [10*(rand(target_rank)-0.5), zeros(target_rank,dimensionality-target_rank)];
     L_t = L';
     T = L'*L;
    %}
    
    % creating training, validation and test constraints
    
    constraints = zeros(training_size + validation_size + test_size,4);
    for i=1:length(constraints)
        t = randperm(number_of_samples);
        t = t(1:4);
        dist_1 = sum(((X(t(1),:)-X(t(2),:)) * L_t).^2);
        dist_2 = sum(((X(t(3),:)-X(t(4),:)) * L_t).^2);
        if dist_1 < dist_2
            constraints(i,:) = t(1:4);
        else
            constraints(i,:) = t(4:-1:1);
        end
    end
    training_constraints = constraints(1:training_size,:);
    
    validation_constraints = constraints((training_size+1):(training_size+validation_size),:);
    
    test_constraints = constraints((training_size+validation_size+1):end,:);
    
    toc;
end


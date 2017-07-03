% This code is from paper: Fantope Regularization in Metric Learning. 
% url: http://www-poleia.lip6.fr/~lawm/projects/cvpr2014/

function [ training, validation, test] = preprocess_toy( X, training_constraints, validation_constraints, test_constraints )

    training.smaller = X(training_constraints(:,1),:) - X(training_constraints(:,2),:);
    training.larger = X(training_constraints(:,3),:) - X(training_constraints(:,4),:);

    validation.smaller = X(validation_constraints(:,1),:) - X(validation_constraints(:,2),:);
    validation.larger = X(validation_constraints(:,3),:) - X(validation_constraints(:,4),:);

    test.smaller = X(test_constraints(:,1),:) - X(test_constraints(:,2),:);
    test.larger = X(test_constraints(:,3),:) - X(test_constraints(:,4),:);

end


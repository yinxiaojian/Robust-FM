%%%%% Setup to create the toy dataset

dimensionality = 100;
target_rank = 10;
k = dimensionality - target_rank;
number_of_samples = 1e5;
training_size = 1e4;
validation_size = 1e4;
test_size = 1e4;

fprintf('Creating toy dataset:\n- %d random samples\n- %d training constraints\n- %d validation constraints\n- %d test constraints\n- %dx%d dimensional groundtruth distance matrix T with rank(T) = %d\n', number_of_samples, training_size, validation_size, test_size, dimensionality, dimensionality, target_rank);
[ T, X, training_constraints, validation_constraints, test_constraints ] = create_toy_dataset( dimensionality, target_rank, number_of_samples, training_size, validation_size, test_size );

disp('Preprocessing data');
[ training, validation, test] = preprocess_toy( X, training_constraints, validation_constraints, test_constraints);


save('toy_2.mat','training','test','validation');

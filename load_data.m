% movie length 100k
ml_100k_training = 'data/movielens/training_data_100k';
ml_100k_test = 'data/movielens/test_data_100k';

% movie length 1m
ml_1m_training = 'data/movielens/training_data_1m';
ml_1m_test = 'data/movielens/test_data_1m';

% movie length 10m
ml_10m_training = 'data/movielens/training_data_10m';
ml_10m_test = 'data/movielens/test_data_10m';

% amazon video
amazon_video_train = 'data/amazon/training_data_video';
amazon_video_test = 'data/amazon/test_data_video'; 

% netflix subset
% netflix_train_5K5K = 'data/netflix/training_data5K5K';
% netflix_test_5K5K = 'data/netflix/test_data5K5K';

netflix_train_5K5K = 'data/netflix_half_half/train_data';
netflix_test_5K5K = 'data/netflix_half_half/test_data';

% RCV1 dataset
rcv1_train = 'data/rcv1/train_data';
rcv1_test = 'data/rcv1/test_data';

% Covtype
covtype_train = 'data/covtype/train_data';
covtype_test = 'data/covtype/test_data';

% real-sim
realsim_train = 'data/real_sim/train_data';
realsim_test = 'data/real_sim/test_data';

%% classification task
% adult
adult_train = 'data/adult/training_data';
adult_test = 'data/adult/test_data';

banana_train = 'data/banana/train_data';
banana_test = 'data/banana/test_data';

magic04_train = 'data/magic04/train_data';
magic04_test = 'data/magic04/test_data';

phishing_train = 'data/phishing/train_data';
phishing_test = 'data/phishing/test_data';

ijcnn_train = 'data/ijcnn/train_data';
ijcnn_test = 'data/ijcnn/test_data';

w8a_train = 'data/w8a/train_data';
w8a_test = 'data/w8a/test_data';

a9a_train = 'data/a9a/train_data';
a9a_test = 'data/a9a/test_data';

a7a_train = 'data/a7a/train_data';
a7a_test = 'data/a7a/test_data';

%% multiclassification task
letter_train = 'data/letter/train_data';
letter_test = 'data/letter/test_data';

mnist_train = 'data/mnist/train_data';
mnist_test = 'data/mnist/test_data';

usps_train = 'data/usps/train_data';
usps_test = 'data/usps/test_data';

protein_train = 'data/protein/train_data';
protein_test = 'data/protein/test_data';

connect4_train = 'data/connect-4/train_data';
connect4_test = 'data/connect-4/test_data';

%% regression task
ONP_train = 'data/OnlineNewsPopularity/train_data';
ONP_test = 'data/OnlineNewsPopularity/test_data';
%% 
% training_data = ml_100k_training;
% test_data = ml_100k_test;
% 
% training_data = ml_1m_training;
% test_data = ml_1m_test;

% training_data = amazon_video_train;
% test_data = amazon_video_test;

% training_data = netflix_train_5K5K;
% test_data = netflix_test_5K5K;

% training_data = adult_train;
% test_data = adult_test;

% training_data = banana_train;
% test_data = banana_test;

% training_data = magic04_train;
% test_data = magic04_test;

% training_data = phishing_train;
% test_data = phishing_test;

% training_data = ijcnn_train;
% test_data = ijcnn_test;

% training_data = rcv1_train;
% test_data = rcv1_test;

% training_data = letter_train;
% test_data = letter_test;

% training_data = mnist_train;
% test_data = mnist_test;
% 
% training_data = usps_train;
% test_data = usps_test;

% training_data = protein_train;
% test_data = protein_test;

% training_data = covtype_train;
% test_data = covtype_test;

% training_data = realsim_train;
% test_data = realsim_test;

% training_data = ONP_train;
% test_data = ONP_test;

% training_data = w8a_train;
% test_data = w8a_test;

% training_data = a9a_train;
% test_data = a9a_test;

% training_data = a7a_train;
% test_data = a7a_test;

training_data = connect4_train;
test_data = connect4_test;

load(training_data);
load(test_data); 
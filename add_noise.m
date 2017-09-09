function [ out_train_X, out_train_Y ] = add_noise( in_train_X, in_train_Y, noise_ratio)
%ADD_NOISE 此处显示有关此函数的摘要
%   此处显示详细说明
    out_train_X = in_train_X;
    out_train_Y = in_train_Y;
    [num_sample, ~] = size(in_train_X);
    num_flip = ceil(num_sample*noise_ratio);
    
    re_idx = randperm(num_sample);
    out_train_Y(re_idx(1:num_flip)) = -out_train_Y(re_idx(1:num_flip));

end


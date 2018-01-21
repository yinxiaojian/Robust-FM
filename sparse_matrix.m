function [ sparse_matrix ] = sparse_matrix( X )
%SPARSE_MATRIX Summary of this function goes here
%   Detailed explanation goes here

    [num_sample,~] = size(X);
    row = zeros(num_sample*2,1);
    column = zeros(num_sample*2,1);
    value = ones(num_sample*2,1);
    for j = 1:num_sample
        row(2*j-1) = j;
        row(2*j) = j;
        column(2*j-1) = X(j,1);
        column(2*j) = X(j,2);
    end
    sparse_matrix = sparse(row,column, value);

end


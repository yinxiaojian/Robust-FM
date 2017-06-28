function [ U,S,V ] = truncated_svd( data, epsilon )
%TRUNCATED_SVD
%   data:       matrix to be applied truncated svd
%   epsilon:    threshold to truncated eig values
[U, S, V] = svd(data);

singular_values = diag(S);
U = U(singular_values<=epsilon,:);
V = V(singular_values<=epsilon,:);
end


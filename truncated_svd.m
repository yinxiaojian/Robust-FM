function [ P ] = truncated_svd( data, epsilon )
%TRUNCATED_SVD
%   data:       matrix to be applied truncated svd
%   epsilon:    threshold to truncated eig values
[U, S, ~] = svd(data);

singular_values = diag(S);

index = find(singular_values <= epsilon);
new_S = zeros(length(singular_values),1);
new_S(index) = 0.5*(singular_values(index).^2+eps).^(-0.5);
P = U*diag(new_S)*U';
end
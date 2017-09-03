function [ U,S,d ] = truncated_svd( data, epsilon )
%TRUNCATED_SVD
%   data:       matrix to be applied truncated svd
%   epsilon:    threshold to truncated eig values
[U, S, ~] = svd(data);

singular_values = diag(S);

index = singular_values <= epsilon;
new_S = zeros(length(singular_values),1);
d = length(data) - sum(index);

new_S(index) = singular_values(index);
% P = U*diag(new_S)*U';
S = diag(new_S);
U = U(index,:);

% U = U(singular_values <= epsilon,:);
% P = U'*U;

end
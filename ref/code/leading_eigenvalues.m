function [ k ] = leading_eigenvalues( M )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [~, D] = eig(M);
    a = sort(diag(D),'descend');
    k = length(M);
    for i=1:(length(a)-1)
        if a(i) > 5 * a(i+1)
            k = i;
            break;
        end
    end

end


function [ U , S ] = incremental_svd( Z, A, U_, S_, eta)
%INCREMENTAL_SVD 此处显示有关此函数的摘要
%   此处显示详细说明
    A = sqrt(eta)*A;
    [d, k] = size(U_);  
    tmp = U_*U_';
    P = null((eye(d)-tmp)*A);
    
    if ~isempty(P)
        R_A = P'*(eye(d)-tmp)*A;
    
    
        [u, ~] = size(R_A);
        a = [S_ zeros(k,u);zeros(u, k+u)];
        b = [U_'*A;R_A];
        K = a-b*b';
        [U_K, S_K, ~] = svd(K);
        U = [U_ P]*U_K;
        S = S_K;
        
    else
        a = S_;
        b = U_'*A;
        K = a-b*b';
        [U_K, S_K, ~] = svd(K);
        U = U_ * U_K;
        S = S_K;
        
    end
    
    
    
    
end


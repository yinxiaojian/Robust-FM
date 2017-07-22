function [ P ] = psd_cone( U )
%PSD Summary of this function goes here
%   Detailed explanation goes here

    [UU, D, VV] = svd(U);
    d = real(diag(D));
    d(d < 0) = 0;
    P =(UU * diag(d) * VV');

end


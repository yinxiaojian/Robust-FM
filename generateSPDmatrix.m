function Z = generateSPDmatrix( n )
%GENERATESPDMATRIX generate a n*n symetric positive matrix
%   此处显示详细说明

Z = 0.1*randn(n,n);
Z = 0.5 * (Z + Z');
Z = Z + n * eye(n);
% Z(logical(eye(size(Z)))) = 0;

end


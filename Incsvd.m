function [U1,S1]=Incsvd(U,S,A)
%注意需要进行特征值分解，而不是奇异值分解
%本算法的主要功能是实现U1*S1*U1'=U*S*U'+AA'
%如果U=0,只需要对AA'进行特征值分解，考虑到AA'形成的矩阵较大，所以对A进行奇异值分解：A=U2*S2*U2',AA'=U2*S2*S2'*U2',S1=S2*S2'，这边把0项都去掉了，节省存储空间，U2=U2(:,size(S2*S2'))
%如果U!=0,按照IncSVD的思想，先计算U'*A,因为有的时候回出现UU'I的情况，而计算机在进行运算时I-UU'不会等于0，这时会造成较大的麻烦，需要将这种情况排除，
%①如果UU'=I,此时I-UU'=0,这个时候无需计算RA
%②如果UU'!=I,计算A-U(U'A)，得到P,然后计算[U'A RA]'*[U'A RA]

flag=0;
%计算P和RA 
if U==0
    [U1 S2 V1]=svd(full(A),'econ');
    S1=S2'*S2;
    %U1=U1(:,1:size(S1,1));
    return;
else
    UA=U'*A; %之所以这样做是多矩阵乘法加速的原因
end

Ra=A-U*UA;
if(Ra<0.0000001)  %有的时候UU'=I时，计算机给出的结果tmp并不是0，这边就是为了排除这种情况
    if sum(sum(U'*U-eye(size(U,2))))<0.0000001
        flag=1;
    end
end
if flag==0
    if issparse(Ra)~=0
        Ra=full(Ra);
    end
    P=orth(Ra); 
    
    RA=P'*Ra;
    Ktmp=[UA;RA]*[UA;RA]';
else
    Ktmp=UA*UA';
end
 
%此处利用公式(6)计算K
[rowlen,collen]=size(Ktmp); 
[Srowlen,Scollen]=size(S);
tmpS=[[S zeros(Srowlen,collen-Scollen)];zeros(rowlen-Srowlen,Scollen),zeros(collen-Scollen)];
K=tmpS+Ktmp;

%这边的U2,S2,V2分别是论文中的U',S'和V'
[U2,S2]=eig(K);
if flag==0 
    U1=[U P]*U2; 
else
    U1=U*U2;
end
S1=S2;
end
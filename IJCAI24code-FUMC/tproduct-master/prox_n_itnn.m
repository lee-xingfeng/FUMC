function [X,tnn,trank] = prox_n_itnn(Y, tao)
    % process improved tensor nuclear norm
    [n1,n2,n3] = size(Y);
    n12 = min(n1,n2);
  
    % get core matrix
    lambda1 = 1/sqrt(max(n12,n3));
%     lambda1 = 0.001;
    [U, S, V] = n_tsvd(Y,'econ');%change by lxf 2022.12.5
    reshapeS = zeros(n12, n3);
    for i = 1:n12
        reshapeS(i,:) = S(i,i,:);
    end
  
    [A,D,B] = svd(reshapeS, 'econ');
    VT=B';
    D = diag(D);
    ind = find(D > lambda1);
    D = diag(D(ind) - lambda1);
    L = A(:,ind) * D * VT(ind,:);
    T = zeros(n12,n12,n3);
    for i = 1:n12
        T(i,i,:) = L(i,:);
    end
    T = tprod(tprod(U,T), tran(V));
   %% change by lxf 2022.12.5 
    %prox_n_itnn:% min_X rho*||X||_*+0.5*||X-Y||_F^2,Canyi Lu,https://github.com/canyilu/tproduct.
%     tao=gamma/mu;
    [X,tnn,trank] = prox_tnn(T, tao);
end


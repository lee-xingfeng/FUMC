function [S_ind] = construct_anchorgraph(X,m,numclass)

    numview = length(X);

    anchor  = m/numclass;
    similar_value=1/anchor;
    numdiag =  similar_value;
    numdim = anchor;
    a = toeplitz([numdiag,similar_value*ones(1,numdim-1)]); %
    b = repmat({a},numclass,1); 
    for i = 1:numview
        S_ind{i} = blkdiag(b{:});
    end
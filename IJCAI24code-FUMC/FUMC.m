function [UU] = FUMC(best_view,X,numclass,d,numanchor,alpha,beta,gamma)


%% initialize
maxIter = 50 ; % the number of iterations
m = numanchor;
numview = length(X);
numsample = size(X{1},2);

  

%% construct G 
[S_ind]=construct_anchorgraph(X,m,numclass);

for i = 1:numview
    P{i} = zeros(d,m); 
    P_old{i} = zeros(d,m);
   di = size(X{i},1); 
   Q{i} = zeros(di,d);
   %initialize
   Z{i}=zeros(m,numsample);% m  * n
   Z_old = Z;
   W{i} = zeros(m,m);
   W_old{i} = zeros(m,m);
   H{i}=zeros(m,numsample);% m  * n
   J{i}=zeros(m,numsample);% m  * n
   DS{i} = diag(sum(S_ind{i},2));
   LS{i} = DS{i} - S_ind{i};
end
   
opt.disp = 0;

mu = 10e-5; max_mu = 10e10; pho_mu = 2;
flag = 1;
iter = 0;

%%
while flag
    iter = iter + 1; 
        %% optimize Z  ok1     
             % Z_best
         temp_am{best_view} = (gamma+0.5*mu)*ones(1,numsample); 
         temp_Zv = 0;
         G{best_view} = H{best_view} - J{best_view}/mu;
        for p = 1:numview
            if p == best_view
                continue;
            else
                 temp_Zv = temp_Zv + 0.5*alpha*LS{p}*W_old{p}*Z_old{p};
            end
        end 
         temp_Z{best_view} = gamma*P{best_view}'*Q{best_view}'*X{best_view}+0.5*mu*G{best_view}-temp_Zv;
         for ii=1:numsample
            idx = 1:numanchor;
            Z_column = temp_Z{best_view}(idx,ii)./(temp_am{best_view}(ii)); 
            Z{best_view}(idx,ii) = EProjSimplex_new(Z_column');
         end   
              % Z_others ok1
    for v=1:numview 
      if v == best_view
          continue;
      else
        temp_am{v} = (gamma+0.5*mu)*ones(1,numsample); 
        G{v} = H{v} - J{v}/mu;
        temp_Z{v} = gamma*P{v}'*Q{v}'*X{v}+0.5*mu*W{v}'*G{v}-0.5*alpha*W{v}'*LS{v}'*Z{best_view};%
         for ii=1:numsample
            idx = 1:numanchor;
            Z_column = temp_Z{v}(idx,ii)./(temp_am{v}(ii)); 
            Z{v}(idx,ii) = EProjSimplex_new(Z_column');
         end
      end
    end
    Z_old = Z;   

%      %% optimize H 
         for i = 1:numview
             if i == best_view
             WZ{i} = Z{i}';
             Y{i} = J{i}';            
             else
             WZ{i} = (W{i}*Z{i})';
             Y{i} = J{i}';
             end
         end
    Z_tensor = cat(3, WZ{:,:});
    J_tensor = cat(3, Y{:,:});
    Ten = Z_tensor+J_tensor/mu;
    Ten=shiftdim(Ten, 1);
    [H_tensor,~,~] = prox_n_itnn(Ten,1/mu);
    H_tensor = shiftdim(H_tensor, 2);
    
    %% optimize Q{v} ok1
    for v = 1:numview
        temp_Q = X{v}*Z{v}'*P{v}';      
        [U,~,V] = svd(temp_Q,'econ');
        Q{v} = U*V';
    end

    %% optimize P{v} ok1
        temp_Pv = 0;
        for p = 1:numview
            if p == best_view
                continue;
            else
                temp_Pv = temp_Pv + beta*P_old{p}*LS{p}';
            end
        end        
    temp_P = gamma*Q{best_view}' * X{best_view} * Z{best_view}' - temp_Pv;
    [Unew,~,Vnew] = svd(temp_P,'econ');
    P{best_view} = Unew*Vnew';

% P_others ok1
    for v = 1:numview
        if v == best_view
          continue;
        else   
    temp_P = gamma*Q{v}' * X{v} * Z{v}' - beta*P{best_view}*LS{v};
    [Unew,~,Vnew] = svd(temp_P,'econ');
    P{v} = Unew*Vnew';
        end  
    end
    P_old = P;

    %% optimize W{v} ok1
    for v = 1:numview 
        if v==best_view
            continue
        else
        G{v} = H{v}-J{v}/mu;
        temp_W = 0.5*mu*G{v}*Z{v}' + alpha*LS{v}'*Z{best_view}*Z{v}';
        [Unew,~,Vnew] = svd(temp_W,'econ');
        W{v} = Unew*Vnew';
        end
    end 
    W_old = W;

     %% solve  Y_tensor and  penalty parameters        
    J_tensor = J_tensor + mu*(Z_tensor - H_tensor);
    mu = min(mu*pho_mu, max_mu);
    for v = 1:numview
        H{v} = H_tensor(:,:,v)';
        J{v} = J_tensor(:,:,v)';
    end
    %% obtain indicator from Z
    S=0;
    for v = 1:numview
        if v == best_view
            S = S + Z{v};
           temp_WZ{v} = Z{v}; 
        else
            S = S + W{v}*Z{v};
            temp_WZ{v} = W{v}*Z{v};
        end
    end
    S = S/numview;
    [UU,~,V]=svd(S','econ');
    ts{iter} = UU(:,1:numclass);
        term_ZHtensor = 0;
    for v = 1:numview
        term_ZHtensor = term_ZHtensor +  norm(Z{v} - H{v},'fro')^2;
    end
    term_QXPS = 0;
    for v = 1:numview
        term_QXPS = term_QXPS +  norm(Q{v}' * X{v}-P{v} * Z{v},'fro')^2;
    end

    obj(iter) = term_ZHtensor+term_QXPS;

    if (iter>1)
            obj2(iter) = abs((obj(iter-1)-obj(iter))/(obj(iter-1)));
    end
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(S','econ');
        UU = UU(:,1:numclass);
        flag = 0;
    end
end




    
         
         
    

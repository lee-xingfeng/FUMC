function [datas,Yi,mapping] = shuffle_data_part( raw_data,Y)
% x is dv*n
n_views = length(raw_data);
for i =1 :n_views
    raw_data{i}= raw_data{i}';
end
datas = {};
percent=0.1;
n_view_a = size(raw_data{2}, 1);

perc_num=fix(n_view_a*percent);

% randIndex=randperm(n_view_a);
% B=raw_data{2}(randIndex(1:perc_num),:);
% raw_data{2}(randIndex(1:perc_num),:)=B(randperm(perc_num),:);
% Yi(2,:)  = Y(randIndex(1:perc_num),:);

rand('seed',500);
randIndex=randperm(n_view_a);
loc=randIndex(1:perc_num);%location
B=raw_data{2}(loc,:);%value
% By=Y(loc);
randIndex2=randperm(perc_num);%daluan
Z=B(randIndex2,:);
% Zy=By(randIndex2);
raw_data{2}(loc,:)=Z;
% Y(loc)=Zy;

% Yi = zeros(n_view_a,1);
%     for i = 1:n_view_a
%         Yi(i,:)  = Y(mapping{best_view}(i),:);
%     end
    
Yi=Y;    
% datas{a} = zeros(size(raw_data{a}));
for a = 1:n_views
    datas{a} = raw_data{a};
end





for i =1 :n_views
    datas{i}= datas{i}';
end
mapping=1;
end



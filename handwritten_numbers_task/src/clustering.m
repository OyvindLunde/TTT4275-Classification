clc
close all


%%%Create clusters%%%
train_lists = cell(10,1);

for i=0:9
    class_i = trainv(trainlab==i,:);
    train_lists{i+1} = class_i;
end

clusters = cell(10,1);
m = 64;

for i=1:10
    [~,Ci] = kmeans(train_lists{i},m);
    clusters{i} = Ci;
end

trainv_clusters = cell2mat(clusters);
trainlab_clusters = NaN(10*m, 1);

for i=0:9
    trainlab_clusters(i*m+1:(i+1)*m) = i*ones(m,1);
end





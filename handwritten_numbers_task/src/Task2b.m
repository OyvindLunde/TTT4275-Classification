clc
close all


test_size = 10000;
pred = zeros(test_size,1);
test_labels = testlab(1:test_size);
miss_classified = [];

for k=1:test_size
     d = dist(trainv_clusters, testv(k,:).');
     [minimum, i] = min(d);
     pred(k) = trainlab_clusters(i);
end

pred = categorical(pred);
test_labels = categorical(test_labels);

plotconfusion(test_labels, pred)
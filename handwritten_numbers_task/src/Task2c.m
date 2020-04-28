clc
close all


test_size = 10000;
k = 7;
pred = zeros(test_size,1);
test_labels = testlab(1:test_size);
miss_classified = [];

for j=1:test_size
     d = dist(trainv_clusters, testv(j,:).');
     [minimum, indices] = mink(d, k);
     label = PredictLabel(indices, trainlab_clusters);
     pred(j) = label;
end

pred = categorical(pred);
test_labels = categorical(test_labels);

plotconfusion(test_labels, pred)


function label = PredictLabel(minIndices, trainlab_clusters)
    label_list = zeros(10,1);
    for i=1:length(minIndices)
        value = label_list(trainlab_clusters(minIndices(i))+1);
        label_list(trainlab_clusters(minIndices(i))+1) = value+1;
    end
    [~, argmax] = max(label_list);
    label = argmax-1; %%Minus 1 to get range from 0-9, not from 1-10
end
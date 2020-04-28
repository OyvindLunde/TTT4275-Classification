clc
close all

test_size = 10000;
pred = zeros(test_size,1);
test_labels = testlab(1:test_size);
miss_classified = [];

tic 
for k = 1:test_size
    d = dist(trainv, testv(k,:).');
    [m, i] = min(d);
    pred(k) = trainlab(i);
    if pred(k) ~= test_labels(k)
        miss_classified(end+1) = k;
    end
end
toc

pred = categorical(pred);
test_labels = categorical(test_labels);

plotconfusion(test_labels, pred)

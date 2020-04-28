close all
clc

x = zeros(28,28);
y = zeros(28,28);
z = zeros(28,28);

x(:) = testv(miss_classified(1), :);
y(:) = testv(miss_classified(2), :);
z(:) = testv(miss_classified(3), :);

testlab(miss_classified(1))
pred(miss_classified(1))
testlab(miss_classified(2))
pred(miss_classified(2))
testlab(miss_classified(3))
pred(miss_classified(3))

figure
subplot(1,3,1)
image(x)
title({'Actual: 4','Predicted: 9'})
subplot(1,3,2)
image(y)
title({'Actual: 3','Predicted: 5'})
subplot(1,3,3)
image(z)
title({'Actual: 9','Predicted: 8'})



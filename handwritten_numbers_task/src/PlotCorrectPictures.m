close all
clc

x = zeros(28,28);
y = zeros(28,28);
z = zeros(28,28);

x(:) = testv(1, :);
y(:) = testv(2, :);
z(:) = testv(3, :);

testlab(1)
pred(1)
testlab(2)
pred(2)
testlab(3)
pred(3)

figure
subplot(1,3,1)
image(x)
title({'Actual: 7','Predicted: 7'})
subplot(1,3,2)
image(y)
title({'Actual: 2','Predicted: 2'})
subplot(1,3,3)
image(z)
title({'Actual: 1','Predicted: 1'})
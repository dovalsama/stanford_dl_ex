function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
numLayers = numHidden + 1;
hAct = cell(numLayers + 1, 1);
deltaStack = cell(numLayers, 1);
gradStack = cell(numLayers, 1);
hAct{1} = data;
%% forward prop
%%% YOUR CODE HERE %%%

for l = 2:numLayers + 1
    z = bsxfun(@plus, stack{l - 1}.W * hAct{l - 1}, stack{l - 1}.b);
    if l == numLayers + 1
        hAct{l} = exp(z);
    else
        hAct{l} = sigmoid(z);
    end
end
pred_prob = bsxfun(@rdivide, hAct{end}, sum(hAct{end}, 1));

K = size(hAct{end}, 1);
n = size(data, 1);
m = size(data, 2);

%% return here if only predictions desired.
if po
  cost = -1; % ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ind = sub2ind(size(pred_prob), labels', 1:m);
val = pred_prob(ind);
ceCost = -sum(log(val));
target = zeros(K, m);
target(ind) = 1;
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

deltaStack{numLayers} = pred_prob - target;
for l = numHidden:-1:1
    deltaStack{l} = stack{l + 1}.W' * deltaStack{l + 1} .* (hAct{l + 1} .* (1 - hAct{l + 1}));
end

for l = 1:numLayers
    gradStack{l}.W = deltaStack{l} * hAct{l}';
    gradStack{l}.b = sum(deltaStack{l}, 2);
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for l = 1:numLayers
    wCost = wCost + 0.5 * ei.lambda * sum(stack{l}.W(:) .^ 2);
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end
cost = ceCost + wCost;
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end




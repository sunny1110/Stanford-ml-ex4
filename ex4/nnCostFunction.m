function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Calculating h, by considering the network layers

a1 = [ones(size(X), 1), X]; %Adding ones to X
z2 = a1 * transpose(Theta1);
a2 = sigmoid(z2);
a2= [ones(size(a2),1), a2]; %Adding ones to a2
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);
h = a3;

% The resulting value of h will be of the following dimensions:
% 5000, 10
% The above me seem like an extremely complicated dimension set for a hypothesis but it is a classification problem.
% Dimensions of X (before bias) : m X 400.
% Dimensions of X (after bias) : m X 401.
% Dimensions of Theta1: 401 X 25, where 401 is input units size (bias included), and 25 is hidden units size.
% Dimensions of a2 (before bias) : m X 25.
% Dimensions of a2 (after bias) : m X 26.
% Dimensions of Theta2: 26 X 10, where 26 is the hidden units size (bias included), and 10 is the Hypothesis units size

% Calulations of Costs will follow:
y_matrix = eye(num_labels)(y,:);



pdiff = log(h) .* y_matrix;
ndiff = log(1-h) .* (1-y_matrix);
diff = pdiff + ndiff;
rowsum = sum(diff, 2); % Size: 5000 X 1
colsum = sum(rowsum); % Size: 1 X 1
J = -colsum /m;
NT1 =Theta1(1:size(Theta1, 1), 2:size(Theta1, 2));
NT2 =Theta2(1:size(Theta2, 1), 2:size(Theta2, 2));
R = (sum(NT1(:).^2) + sum(NT2(:).^2))/(2*m) * lambda;
J = J + R;


% The implementation for backpropagation will now begin.
% This is a higly vectorised solution, hence the solutions might appear complicated.


% This is the difference matrix, from which we obtain delta3 from.
d3 = a3 - y_matrix;

% This gives us delta2, from which we obtain delta2 from.
% TEMPORARILY HIDING THIS CODE d2 =(d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);
% a1(:, 2:end), this is a1 without the Bias.
d2 =(d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);
Delta1 = transpose(d2) * a1;
Delta2 = transpose(d3) * a2;

% ([zeros(size(Theta1,2),1)'; Theta1])
% The above statement appends a row of zeros to make regularization easier.


Theta1_grad = Delta1 ./m .+ ([zeros(size(Theta1(:, 2:end),1),1), Theta1(:, 2:end)] .* (lambda ./m));

Theta2_grad = Delta2 ./m .+ ([zeros(size(Theta2(:, 2:end),1),1), Theta2(:, 2:end)] .* (lambda ./m));







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

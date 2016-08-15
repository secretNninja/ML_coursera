function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
a=0;
for k=1:size(X,2)
for i=1:m
z=X(i,:)*theta(:,1);
g=sigmoid(z);
a=a+(g-y(i))*X(i,k);
end
grad(k,1)=a/m;
a=0;
end

for i=1:m
z=X(i,:)*theta(:,1);
g=sigmoid(z);
J=J+y(i)*log(g)+(1-y(i))*log(1-g);
end
J=-J/m;





% =============================================================

end

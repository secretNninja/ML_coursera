function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

a=0;
for k=1:size(X,2)
for i=1:m
z=X(i,:)*theta(:,1);
g=sigmoid(z);
a=a+(g-y(i))*X(i,k);
end
if k==1
grad(k)=a/m;
end
if k>1  
grad(k)=a/m+lambda*theta(k)/m;
end
a=0;
end

for i=1:m
z=X(i,:)*theta(:,1);
g=sigmoid(z);
J=J+y(i)*log(g)+(1-y(i))*log(1-g);
end
thet=0;
for k=2:size(theta)
thet=theta(k)*theta(k);
end

J=-J/m+lambda*thet/m;





% =============================================================

end

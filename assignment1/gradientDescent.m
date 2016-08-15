function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thetaNew=zeros(2,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
sum=0;
sum2=0;
for i=1:m
sum=sum+(theta(1,1)+theta(2,1)*X(i,2)-y(i,1))*X(i,1);
sum2=sum2+(theta(1,1)+theta(2,1)*X(i,2)-y(i,1))*X(i,2);
endfor
thetaNew(1,1)=theta(1,1) - alpha *sum/m;
thetaNew(2,1)=theta(2,1)-alpha*sum2/m;
theta=thetaNew;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
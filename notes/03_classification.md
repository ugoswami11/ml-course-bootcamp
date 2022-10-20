Problem statement: Churn Prediction
Imagine we are working for a telecom company and we have to identify clients that wnats to leave our company or churn and assign everyone score from 0 to 1 for the chances of leaving the company.
And by predicting this value we would be able to retain our client by providing discounted prices to those client to avoid them from leaving our company service.

Binary Classification

g(Xi) ~ yi

yi = {0,1}

the output of our model is a score between 0 and 1 based on the likelihood of customers to churn.

How to predict is that we can look at the historical details of customers who left and which factors leads to churn.

Feature Importance
1. Difference
 Global CR - Group CR >0 - less likely to churn
 Global CR - Group CR <0 - more likely to churn
2. RIsk ratio
 risk   = group CR/ global CR > 1 - more likely to churn
 risk   = group CR/ global CR < 1 - less likely to churn

Mutual Information
way to measure the importance of categorical variables

correlation
way to measure the importance of numerical variables
when corr value is negative that means one value will go hihger other will be go lower

when corr positive than when one increase the other also increas

y = {0,1}

x tenure - 0.72
y churn 
the when tenure is high the churn rate will be high

One hot encoding


Logistic Regression:
g(Xi) = yi
where g is the model 
y is the target

classification - binary and multiclass

y = {0,1}
0 - no churn
1 - churn

g(Xi) = 0 or 1
probability of Xi belonging to the positive class

linearreg - g(Xi) = Wo + W_T*Xi
logistic reg - same as linear reg but the range is between 0 and 1
Sigmoid function


The W0+W_T*Xi is the score and with sigmoid we can convert it into probability


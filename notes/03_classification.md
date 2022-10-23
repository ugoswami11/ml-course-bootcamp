## Logistic Regression

### Problem statement: Churn Prediction
Imagine we are working for a telecom company and we have to identify clients that wants to leave our company or churn and assign everyone score from 0 to 1 for the chances of leaving the company and by predicting this value we would be able to retain our client by providing discounted prices to those client to avoid them from leaving our company service.

### Binary Classification
The above problem statement is an example of binary classification as the traget variable consists of only 2 values i.e 0 and 1.

So here also we are trying to predict the y value which is the target column by feeding the X values which are the features into the model g. we can rewrite this as below:
```
g(Xi) ~ yi
```
where the y column consists of only two values ```yi = {0,1}```

Now we will use logistic regression to generate prediction but first we will have look at some important terms that will help us in feature engineering and doing exploratory data analysis.

### Feature Importance
From our telco dataset we will take few columns that can be used as feature and we study further if they have any impact on the target column.

First we will calculate the average churn of churn column to have a look at the average churn of customers and we will call this data as global.

Then we can take a particular group from the dataset like gender column or contract column and we will take the average churn for that particular group and let's call this data as group

Now we can do two operation to find out which group of people are more likely to churn.

-  Difference:
    - If ( Global - Group ) > 0 then that group is less likely to churn
    - If ( Global - Group ) <0 then that group is more likely to churn
- Risk ratio
    - If ( Group / Global ) > 1 then that group is more likely to churn
    - If ( Group / Global ) < 1 then that group is less likely to churn

So from above two measures i.e. the difference and risk ration we can get an idea which group of customers are more likely to churn and we can target those customers by providing some offers and discounts.

### Mutual Information
One more way to measure which features provide a more imapct on the target column is by using *Mutual Information* 

Mutual Information is a way to measure the importance of categorical variables

### Correlation
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


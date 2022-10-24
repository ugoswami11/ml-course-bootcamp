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

*Mutual information* of 2 random variables is a measure of the mutual dependence between them.

The mutual info score is between 0 and 1. The more closer to 1 the more important the feature is for the target variable.

### Correlation
The correlation coefficient measures the linear correlation between two sets of data. It's the ratio between the covariance of 2 variables and the product of their standard deviations.

The value of correlation is always in the interval [-1, 1]. when the correlation value is negative that means one value will go higher and other will be go lower when the value is positive that means if one values go higher then the other will also go higher

We can calculate correlation using the corr and corrwith methods.

### One-hot encoding
One hot encoding is method applied for the categorical columns where a new column is created for each category and the values are coded as 0 and 1.

Example: Let's take the gender column in the dataset which has two categoies ```male``` and ```female```. When we will apply one hot encoding the gender column will be converted into two more columns - male and female and the values will be represented as below

| category=male| category=female|
---------------|---------------
| 1 | 0 |
| 0 | 1 |
| 0 | 1 |
| 1 | 0 |
| 1 | 0 |

### Implementing Logistic Regression:
The equation of logistic regression is same as the equation of linear regression as both are linear functions. The only differene is of the sigmoid function we use in logistic regression. 

So we know that ```g(Xi) = yi``` where g is the model and y is the target

But in this case the target consists of only two values that is 0 and 1. where,
- 0 - no churn
- 1 - churn

So that means we need to predict ```g(Xi) = 0``` or ```g(Xi) = 1``` 

Let's look at how our implementation of linear regression is similar and different from logistic regression.

The below code is our implementation of linear egression where  ```w0``` is the bias term, ```xi``` is the feature matrix and ```w``` is the corresponding weights of the features. Then we add the bias term with the dot product of weight and feature matrix which gives us our prediction.
```
def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result
```    
In case of logistic regression we just use our sigmoid function in the result we got from our linear regression method and that will be our prediction. 
```
def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        score = score + xi[j] * w[j]
        
    result = sigmoid(score)
    return result
```
### Sigmoid function
The sigmoid function is used to convert the result into probabilities. It maps any real values into another value between 0 and 1. 

We can calculate the sigmoid value as below
```
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```




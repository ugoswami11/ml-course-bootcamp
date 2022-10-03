## Linear Regression
*Linear regression* is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables.

It is Supervised Machine Learning algorithm which help us to generate predictions based on the dependent variables or features.

### General formula for ML
The general formula for can be listed as below
```
g(X) ~ y
```
- ```g``` is the model
- ```X``` is the feature matrix
- ```y``` is the target matrix

When we feed the model with the features it will generate our target.

### Problem
Imagine we have a car classified website where can list your car if you want to sell. Now as a user if someone wants to sell the car, they will put features of the car like mileage, engine capacity, pictures etc and list their price. Now the seller want's to find a price which is relevant to the car and not go too high or too low with the prices. Now as an owner of the webiste we will be using *Linear Regression* model to help the user to list the relevant price.

### Creating a simplified model
We will first see for a single observation by taking only one car

We will use the general formula of ML:
```
g(xᵢ) ~ yᵢ
```
where, 
- ```xᵢ``` is a observation of one car (one row in a dataset),
- ```yᵢ``` is the price of that car,

The observation ```xᵢ``` will consists of many features like ```xᵢ₁, xᵢ₂, ..., xᵢₙ```

Hence, we can also write the general formula of ML as below:
```
g(xᵢ₁, xᵢ₂, ..., xᵢₙ) ~ yᵢ
```
For our examples we will take these features of our car:

```Engine Horse Power: 453;```
```Miles per Galon: 11;``` 
```Popularity: 86```

So our ```xᵢ``` value would be this:
```
xi = [453,11,86]
```

Now we will look at the formula of linear regression
```
g(Xᵢ) = w₀ + w₁·xᵢ₁ + w₂·xᵢ₂ + w₃·xᵢ₃
```
- ```w₀``` is the weight bias term
- ```w₁,w₂,w₃``` are the weights of each feature

We can also rewrite the formula as:
```
g(Xᵢ) = w₀ + ∑( wⱼ·xᵢⱼ, j=[1,3])
```
Depending on the values of the weights, our predicted price ```yᵢ``` will be different.

We have already defined xi and now we need to define the weight bias term and the corresponding weights
```
w0 = 7.17
w = [0.01,0.04,0.002]
```
We will create our simple linear regression model
```
def linear_regression(xi):
    n = len(xi)

    pred = w0

    for j in range(n):
        pred = pred + w[j]*xi[j]

    return pred
```
### Linear Regression in vector form

Here we will look at multiple observation of cars.

So we have already derived the general formula for linear regression i.e. ```g(Xᵢ) = w₀ + ∑( wⱼ·xᵢⱼ, j=[1,n]) ```

And we will use the same features from our dataset of cars: 
```Engine Horse Power, Miles per Galon and Popularity```

So now our ```xᵢⱼ``` would be a m x n matrix like this:
```
453  11  86
500  12  88
480  10  90
...  ... ...
```

The term ```wⱼ·xᵢⱼ``` is actually a dot product of ```wⱼ``` with matrix ```xᵢⱼ``` So , we can rewrite the formula as 
```
g(Xᵢ) = w₀ + Xᵢᵀ · W
```
Here ```Xᵢᵀ``` is the transposed feature vector and W is the weight vector.

To implement this formula we first need to calculate the dot product so first we will implement the calculation of dot product
```
def dot(xi, w):
    n = len(xi)
    res = 0.0

    for j in range(n):
        res = res+ xi[j] * w[j]
    
    return res
```
We have created a ```dot``` function to calculate the dot product between the feature matrix and the weight, alternatively we can also use the dot method from numpy to calculate the dot product between two matrices.

Now our linear regression implementation would be like this:
```
def linear_regression(xi):
    return w0 + dot(xi, w)
```
We can also convert this formula purely into vector form by incorporating the bias term w0 to our dot product, simulating a new feature xi0 which is always equal to one. We can rewrite our vectors as below:

```W = [w₀, w₁, w₂, ... , wₙ]```

```Xᵢ = [1, xᵢ₁, xᵢ₂, ... , xᵢₙ]```

So we can write linear regression as ```Wᵀ · Xᵢ = Xᵢᵀ · W```

Now let's take few rows of the car price dataset features and try to generate preditictions using our linear regression function.

```
w0 = 7.17
w = [0.01, 0.04, 0.002]
w_new = [w0] + w
```

```
x1 = [1, 148, 24, 1385]
x2 = [1, 132,25, 2031]
x3 = [1, 453, 11, 86]

X = [x1, x2, x3]
X = numpy.array(X)
```

```
def linear_regression(X):
    return X.dot(w.new)
```
The dot method used above is from numpy.
```
linear_regression(X)
```
This will generate some prediction array like ``` [12.38, 13.552, 12.312]```

### Normal Equation
In our linear regression equation we were using ```W`` as weights and we will see now from where this W is coming.

From the our linear regression we know that ```g(X) = X·W``` which we can also write as ``` X·W ~ y``` where ```y``` is the prediction.

In an ideal situation we want our equation to generate exact prediction then our equation would be like ```X·W = y```

Now we will solve this equation to calculate our ```W```
```
X·W = y
```
Multiplying ```Xᵀ``` to both sides so that we can calculate the inverse of the X matrix.
Sometimes X matrix may not have an inverse and to make sure the inverse exists we are multiplying the transpose of X.
```
Xᵀ·X·W = y·Xᵀ
```
The ```Xᵀ·X``` will result in a square matrix and is also known as *Gram matrix*

Now, multiplying both sides with inverse of X ```Xᵀ·X⁻¹```
```
(Xᵀ·X)⁻¹·Xᵀ·X·W = (Xᵀ·X)⁻¹·y·Xᵀ
```
The term ```(Xᵀ·X)⁻¹·Xᵀ·X``` is equal to identity matrix ```I``` and multipying any matrix with an identity matrix results in the matrix itself.

So that results in our ```W``` value which is also know as the *Normal Equation*
```
W = (Xᵀ·X)⁻¹·Xᵀ·y
```

Now let's implement this normal equation to calculate our ```W``` value.

Let's again take the same features of car that we have used previously and add more observations to it
```
X= [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38, 54, 185],
    [142, 25, 431],
    [453, 31, 86],
]

X = nupmy.array(X)
```
and we also the ``y`` matrix which stores the prediction from the training set,
```
y = [100, 200, 150, 250, 100, 200, 150, 250, 120]
```

Now we will calculate the gram matrix
```
XTX = X.T.dot(X)
```
Then we will calculate it's inverse
```
XTX_inv = numpy.linalg.inv(XTX)
```
Now if we multipy ```XTX``` and ```XTX_inv``` we should be able to get an identity matrix
```
XTX.dot(XTX_inv).round(1)
```
We used the round function as we would not get the exact inverse matrix but we will get very small value close to 0 and 1.

And now we will calculate our ```W``` value
```
W = XTX_inv.dot(X.T).dot(y)
```
Here we have not incorporated the weight bias term but we can add the weight bias in the X matrix as we did previously.

### Root Mean Squared Error (RMSE)
RMSE calculates the square root of squared differnece between the predicted and actual values. It helps us to determin the accuracy of our model. The lower the RMSE value the more accurate prediction.

The equation of RMSE is like this:
```
RMSE = √( 1/m * ∑( (g(Xᵢ) - yᵢ)² , i=[1,m] ))
```
where ```g(Xᵢ)``` is the prediction for Xᵢ and ```yᵢ ``` is the actual value.

We can implement RMSE as below:
```
def rmse(y_actual, y_pred):
    sq_error = (y_actual-y_pred) ** 2
    mse = sq_error.mean()
    
    return numpy.sqrt(mse)
```

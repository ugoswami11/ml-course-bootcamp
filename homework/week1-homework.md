## Question 1
### What's the version of NumPy that you installed?
```
1.22.3
```
## Question 2
### How many records are in the dataset? (Car Price dataset)
```
11914
```
## Question 3
### Who are the most frequent car manufacturers (top-3) according to the dataset?
```
Chevrolet, Ford, Toyota
```
## Question 4
### What's the number of unique Audi car models in the dataset?
```
34
```
## Question 5
### How many columns in the dataset have missing values?
```
5
```
## Question 6
### 1. Find the median value of "Engine Cylinders" column in the dataset.

### 2. Next, calculate the most frequent value of the same "Engine Cylinders".

### 3. Use the fillna method to fill the missing values in "Engine Cylinders" with the most frequent value from the previous step.

### 4. Now, calculate the median value of "Engine Cylinders" once again. Has it changed?
```
No
```
## Question 7
### 1. Select all the "Lotus" cars from the dataset.
### 2. Select only columns "Engine HP", "Engine Cylinders".
### 3. Now drop all duplicated rows using drop_duplicates method (you should get a dataframe with 9 rows).
### 4. Get the underlying NumPy array. Let's call it X.
### 5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
### 6. Invert XTX.
### 7. Create an array y with values [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800].
### 8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w. What's the value of the first element of w?
```
4.5949
```
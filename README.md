# python-neural-network [![Build Status](https://travis-ci.org/zpbappi/python-neural-network.svg?branch=master)](https://travis-ci.org/zpbappi/python-neural-network)
A neural network implementation using python. 
It supports variable size and number of hidden layers, 
uses numpy and scipy to implement feed-forward and back-propagation effeciently.

## Features

- [x] Any purpose neural network training.
- [x] Binary classification (_0_ or _1_).
- [x] Multiclass classification (_class 0_ to _class k-1_).
- [x] Raw evaluation values for custom classification logic implementation.
- [ ] Ability to register a callback method to be called 
after processing every single input data row when training.
- [ ] Separate utility to draw learning curves using the existing neural network.
- [ ] Separate utility to automatically select optimal value of the regularization 
parameter (lambda).
- [ ] Ability to register a callback method to facilitate gradient checking.

## Basic workflow

- When you _train_ a _NeuralNetwork_, you get a _Model_.
- You use the _Model_ to _predict_ classification or _evaluate_ the hypothesis on input.
- Knowledge is nothing but some floating point numbers. 
So, you can create a _Model_ using your own _knowledge values_ without training
any _NeuralNetwork_.

## Usage

### Initializing neural network
```python
nn = NeuralNetwork.init(
    lambda_val = 0.03, # you know what lambda is, right? is is the regularization parameter.
    input_layer_size = 10, # number of features in each input row
    output_layer_size = 4, # number of output classes (use 1 for binary)
    hidden_layer_sizes = [30, 20] # array like structure, mentioning size of hidden layers
)
```

This will initialize the neural network with some _random_ initial parameters (theta values).
However, if you are not really satisfied with default initialization of parameters with radom
values, feel free to use another variant of the neural network initializer.

```python
thetas = []
thetas.append([[0.1, 0.2, 0.03],
                [0.01, 0.02, 0.5],
                [1, 2, 3]])
thetas.append([[1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                [1, 2, 3, 4]])
thetas.append([[1, 2, 3, 4, 5]])

nn = NeuralNetwork.init_with_theta(
    lambda_val = 0.01, # the regularization parameter
    thetas = thetas # initialization with your magical theta values
)
```

_Note: You don't need to tell any size for this variant as I went to primary and learnt 
enough arithmatic to figure out the sizes myself._


### How to train your neural network

Once you have initialized the neural network, all you need to do is train it.
Here is how you do it:
```python
nn = NeuralNetwork.init(0.03, 10, 4, [30, 20])
model = nn.train(X_train, Y_train)
```

Here, `X_train` is a matrix (or, multi-dimensional array if you prefer) of size (`m` x `n`), where

- `m` is the number of training data you have
- `n` is the number of features in each data (i.e. number of columns)
- `X` is expected to be normalized, if needed.
- You do ___NOT___ need to pad `1` in the first column. Neural network will that as a part
of training. But, it will keep your original data unchanged. 

And, `Y_train` is a matrix (or, multi-dimensional array) of size (`m` x `k`), where

- `m` is the number of training data you have
- `k` is the number of the output class you have (also, is the number of columns in output data). 
For any particular output data of `k` class (where `k` > 2), only one of the columns 
(column `0` to `k-1`) should have the value `1`, while other columns of the same row 
should have the value `0`. 
- for binary classification, `k=1`. This single output will denote two classes with the values `0` and `1`.

### Predicting

Once you have the model, you can make prediction using that.

For __binary classification__, you are expected to have an output layer size one. That is, 
a single column in your output data when you were training the model. When you use the trained
model to predict, you will have similar single column output with values either `0` or `1`
for each input data. This is how you use the model for binary classification:
```python
# initialize neural network
nn = NeuralNetwork.init(0.03, 10, 4, [30, 20])

# train the network to get the model
model = nn.train(X_train, Y_train)

# get prediction
prediction = model.predict_binary_classification(X_in)
```
`X_in` is the input data matrix of size (`m` x `n`) that you want to predict the output of. Where,

- `m` is the number of input data rows.
- `n` is the number of features.
- `X_in` should be normalized using the exact same normalization applied to `X_train`
before training the network (if applicable).
- There is, however, an optional second parameter `positive_threshold_inclusive` for `predict_binary_classification` method
with default value `0.5`. This value is used to decide the class based on the probability.
If probability of an input data row of being in class `1` is >= `positive_threshold_inclusive`,
then it belongs to class `1`. Otherwise, the input row is classified as class `0`. You can overwrite
this optional parameter's value in some special cases.

The returned value in `prediction` is a matrix of size (`m` x `1`) containing only `0` and `1` as values.
`0` and `1` indicates two different classes that same way it did in your training dataset `Y_train`.

For __multiclass classfication__, it is somewhat similar. You need to use the following function:
```python
nn = NeuralNetwork.init(0.03, 10, 4, [30, 20])
model = nn.train(X_train, Y_train)
prediction = model.predict_multiclass_classification(X_in)
```

I this case, the returned value in `prediction` is a matrix of size (`m` x `1`). Where,

- `m` is the number of input data rows.
- Each output will have an ineger value betwen `0` (inclusive) and `k-1` (inclusive), where 
`k` is the number of classes the output has (or, the size of output layer) and is defintely > `2`.
- The value in the output (between `0` and `k-1`) will indicate the predicted class of 
the corresponding input data row.


### Raw hypothesis evaluation

In case you do not like the sugar-coated methods for binary classification and multiclass 
classification, you can always use the hypothesis evaluation result directly and have your 
own complex classification logic. Here is how you use it:

```python
nn = NeuralNetwork.init(0.03, 10, 4, [30, 20])
model = nn.train(X_train, Y_train)
hypothesis = model.evaluate(X_in)
```

The returned value in `hypothesis` is a matrix of size (`m` x `k`), where `m` and `k` holds
the same meaning as before. In this case, the values in this matrix are real numbers between
`0` and `1`. One way to interpret these values woule be:
> The value in the `i`th row's `j`th cell of the returned matrix
is the probability of `i`th input data row's being in output class `j`.
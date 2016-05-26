##### Faculdade de Engenharia da Universidade do Porto

# Final Report
## Neural Network for News Popularity Prediction
### Artificial Intelligence

##### Ricardo Magalhães & Neven Miculinic
##### Integrated Master of Informatics and Computer Engineering

<div style="page-break-after: always;"></div>

# Index
1. [Goal](#goal)
2. [Specification](#spec)
3. [Development](#dev)
4. [Experiments](#exp)
5. [Conclusions](#con)
6. [Improvements](#imp)
7. [Resources](#res)
8. [Appendix](#app)

<div style="page-break-after: always;"></div>

# 1. Goal <a name="goal"></a>

The goal of this project is to implement an artificial neural network for predicting news popularity, based on number of social network shares. The dataset is from two years of Mashable articles, summarizing an heterogenous set of features. We should succesfully train a multi-layer neural network, in order to get a model capable of predicting new articles. In order to achieve that, the "Back-Propagation" algorithm should be implemented and used.

<div style="page-break-after: always;"></div>

# 2. Specification <a name="spec"></a>

In this project data set, we have 58 predictive attributes, where some of it are binary vectorization of categorical variables (e.g. day of the week, data channel, etc). It contains 39797 data points. 

The articles were published by Mashable (www.mashable.com) and their content as the rights to reproduce it belongs to them. Hence, this dataset does not share the original content but some statistics associated with it. The original content be publicly accessed and retrieved using the provided urls. It was acquired on January 8, 2015. 

We are going to scale all input features to [0,1] range. Also, we are going to divide our data set into three parts: training set (70%), cross validation (15%) and test set (15%).  
For base line prediction, we are going to use k-nearest-neighbour regression from sklearn Python library. Our aim is to implement a neural network, which has a better score than base line prediction. The cost function we are going to use to evaluate our news prediction score is mean squared error on data set.

<div style="page-break-after: always;"></div>

# 3. Development <a name="dev"></a>

In order to develop our neural network, we are using Python with Jupyter Notebook, previously known as IPython, to ease our work. There are a number of libraries used to help our development, such as TensorFlow, Sklearn, TFLearn, NumPy, Pandas and Seaborn.


<div style="page-break-after: always;"></div>

# 4. Experiments <a name="exp"></a>

We first started to experiment with a linear regression learning algorithm, without using TensorFlow library, to see how well it would perform. Then, we analyzed the correlation between the given dataset attributes. We concluded there are some correlations between variables. After that, we use t-SNE to embed data in two dimensions to visualize it.

<img style="float: center;" src="figure1.jpg">
Figure 1 - t-SNE Data Visualization

Using K-Means clustering, we further tried to check if there were some exploits in input data, but we didn't find any.

<img style="float: center;" src="figure2.jpg">

Our first approach was Mini-Batch Stochastic Gradient Descent, used to minimize an objective function written as a sum of differentiable functions, with exponential decay learning rate and with varied network architetures im order to optimize and get better results. However, to get faster optimizations, we applied Adam Optimizer (check chapter 7, "Resources") and Batch normalization techniques. For further generalization, we are trying L2 regularization on weight.

<div style="page-break-after: always;"></div>

# 5. Conclusions <a name="con"></a>

In conclusion, we got better results than our baseline prediction. Our model uses relative errors of logarithm of 10

<div style="page-break-after: always;"></div>

# 6. Improvements <a name="imp"></a>

<div style="page-break-after: always;"></div>

# 7. Resources <a name="res"></a>

<div style="page-break-after: always;"></div>

# 8. Appendix <a name="app"></a>

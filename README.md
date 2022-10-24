# Neural_Network_Charity_Analysis
Neural Networks and Deep Learning Models

## Background

### Overview of Analysis

This project consists of four technical analysis deliverables.

* Deliverable 1: Preprocessing Data for a Neural Network Model

* Deliverable 2: Compile, Train, and Evaluate the Model

* Deliverable 3: Optimize the Model

* Deliverable 4: A Written Report on the Neural Network Model (README.md)


### Purpose

To help a foundation predict where to make investments. Using machine learning and neural networks, we’ll use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by the fundation.

## Resources

### Data source:

* (1) Alphabet Soup Charity starter code, 

* (2) Alphabet Soup Charity dataset (charity_data.csv)

    * EIN and NAME—Identification columns

    * APPLICATION_TYPE—Alphabet Soup application type

    * AFFILIATION—Affiliated sector of industry

    * CLASSIFICATION—Government organization classification

    * USE_CASE—Use case for funding

    * ORGANIZATION—Organization type

    * STATUS—Active status

    * INCOME_AMT—Income classification

    * SPECIAL_CONSIDERATIONS—Special consideration for application

    * ASK_AMT—Funding amount requested

    * IS_SUCCESSFUL—Was the money used effectively*

### Software:

- Python 3.9.10, Jupyter Lab 4.6, Visual Studio Code 1.71.2
 
<br/>

## Methodology

### D1: Preprocessing Data for a Neural Network Model

Using Pandas and the Scikit-Learn’s StandardScaler(), we’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

<br/>


### D2: Compile, Train, and Evaluate the Model

Using TensorFlow, we’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

<br/>

### Deliverable 3: Optimize the Model

Using TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.


<br/>


## Results:

### D1: Preprocessing the Data for PCA 

The following five preprocessing steps have been performed on the crypto_df DataFrame:

* All cryptocurrencies that are not being traded are removed

* The IsTrading column is dropped

* All the rows that have at least one null value are removed

* All the rows that do not have coins being mined are removed

* The CoinName column is dropped

* A new DataFrame is created that stores all cryptocurrency names from the CoinName column and retains the index from the crypto_df DataFrame

* The get_dummies() method is used to create variables for the text features, which are then stored in a new DataFrame, X 

* The features from the X DataFrame have been standardized using the StandardScaler fit_transform() function

The final DataFrame is shown below, Figure 1.1

![X_scaled](./Images/x_scaled.png)
 
<sub> Figure (1.1) X_scaled DataFrame: X DataFrame have been standardized using the StandardScaler fit_transform() function.

<br/>


### D2: Reducing Data Dimensions Using PCA

* The pca algorithm reduces the dimensions of the X DataFrame down to three principal components
* The X_pca_df DataFrame is created and has the following three columns, PC 1, PC 2, and PC 3, and has the index from the crypto_df DataFrame

The final DataFrame is shown below, Figure 1.2
 
 ![X_pca_df](./Images/X_pca_df.png)
 
<sub> Figure (1.2) X_pca_df DataFrame

<br/>

### D3: Clustering Cryptocurrencies Using K-means

The K-means algorithm is used to cluster the cryptocurrencies using the PCA data, where the following steps have been completed:

* An elbow curve is created using hvPlot to find the best value for K

![Elbow_curve](./Images/elbow%20curve.png)
 
<sub> Figure (1.3) Elbow curve

<br/>

* Predictions are made on the K clusters of the cryptocurrencies’ data

![K_Means_algorithm](./Images/K-means%20algorithm.png)
 
<sub> Figure (1.3) K-Means Algorithm: used to cluster the cryptocurrencies.

<br/>

* A new DataFrame is created with the same index as the crypto_df DataFrame and has the following columns: Algorithm, ProofType, TotalCoinsMined, TotalCoinSupply, PC 1, PC 2, PC 3, CoinName, and Class.

![clustered_df](./Images/clustered_df.png)
 
<sub> Figure (1.3) Clustered_df DataFrame.

<br/>


## Summary

On this project, we worked primarily with the K-means algorithm, the main unsupervised algorithm that groups similar data into clusters. And  build on this by speeding up the process using principal component analysis (PCA), which employs many different features to reduce the dimensions of the DataFrame. 
 
Then using the K-means algorithm, we created an elbow curve using hvPlot to find the best value for K. Then, runned the K-means algorithm to predict the K clusters for the cryptocurrencies’ data.
 
Finally we created  scatter plots with Plotly Express and hvplot, to visualize the distinct groups that correspond to the three principal components. Then created a table with all the currently tradable cryptocurrencies using the hvplot.table() function. 
 
The ultimate goal for this visualizations is to present the data in a story that would be interactive, easy to understanding and that provide the correct information to help the stakeholders in the decision making process. 


## References

[Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

[scikit-learn](https://scikit-learn.org/stable/)
 
[K-Means Elbow](https://predictivehacks.com/k-means-elbow-method-code-for-python/)

[matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)


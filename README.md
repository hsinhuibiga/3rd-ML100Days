# 3rd-ML100Days
Introduction to Machine Learning

    Data introduction and evaluation data (application + code) What are the challenges? Think twice before you start analyzing
    Introduction to machine learning (application topic) What is the difference between machine learning, deep learning and artificial intelligence? What are the topic applications of machine learning?
    Machine Learning-Processes and Steps (Application Topics) Data Pre-processing> Training / Test Set Segmentation> Selected Targets and Evaluation Benchmarks> Building a Model> Adjusting Parameters. Familiar with the entire ML process
    EDA / Read Data and Analysis Process How to read data and extract what you want to know

Data cleaning data pre-processing

    How to create a new dataframe? How to read other data? (Non-csv data) 1 Create a dataframe from scratch 2. How to read different forms of data (such as file, text file, json, etc.)
    EDA: Introduction and processing of field data types Understand the types of data that can be represented in pandas
    Feature type Feature engineering can be roughly divided into three types of features: value / category / time according to the type of feature.
    EDA data distribution
    EDA: Outlier and processing Value points for detecting and processing exceptions: 1. Finding exceptions through common detection methods 2. Judging whether exceptions are normal (presuming possible causes)
    Numerical features-remove outliers If there are a small number of outliers in numerical features, they need to be removed to keep the rest of the data unaffected
    Commonly used value substitution: normalization of continuous median and quantile values ​​Detection and processing of exceptional values ​​1. Missing values ​​or exception substitution 2. Data standardization
    Numerical features-missing missing values ​​and normalization. Numerical features must first fill in missing values ​​and normalize. Here we review and show the differences in prediction results.
    DataFrame operationData frame merge / Common DataFrame operations 1 Common data operation methods 2. Data table concatenation
    Implementation of the program EDA: introduction of correlation / correlation coefficient 1 Understanding the correlation coefficient 2. Use the correlation coefficient to intuitively understand the relationship between the field and the prediction target
    EDA from Correlation in-depth understanding of the data, starting from the results of correlation
    EDA: How to view / draw and style the features between different numerical ranges Kernel Density Estimation (KDE) 1 How to adjust the visual way to view the numerical range 2. Beautiful image repair-Convert drawing style
    EDA: Discretize continuous variables and simplify continuous variables
    Program implementation Discretize continuous variables. Learn more about data and start with simplified discrete variables.
    Subplots Exploratory Data Analysis-Data Visualization-Multi-Picture View 1. Group the data and present it once. 2. Show the data related to the same group of data at once.
    Heatmap & Grid-plot Exploratory Data Analysis-Data Visualization-Thermal Map / Grid Plot 1. Heat Map: View Correlation Between Variables Intuitively 2. Grid Plot: Draw Scatter Plot and Distribution Between Variables
    Model experience Logistic Regression Before we start using any complex model, it is a good practice to have the simplest model as the baseline

Data science feature engineering technology

    Introduction to Feature Engineering Introduce the location of feature engineering and the process architecture in the complete steps of machine learning
    Numerical features-remove skewness If numerical features are significantly skewed to one side, you need to remove skewness to eliminate prediction bias
    Category Features-Basic Processing Introduce the most basic methods of category features: tag coding and one-hot coding
    Categorical Features-Mean Coding The most important encoding of categorical features: Mean encoding, which replaces the label with the target mean
    Categorical features-other advanced processing Other common encodings of categorical features: Count encoding corresponds to features that appear in frequency, and hash encoding corresponds to features that cannot be sorted for many categories
    Time-type features Time-type features can extract multiple sub-features, or cycle, or take the number of times in a continuous period
    Feature Combinations-Numerical and Numerical Combinations Basis of Feature Combinations: Combining the four arithmetic operations to form more predictive features
    Feature Combination-Category and Value Combination Category-type log-valued features can be clustered, similar to the target mean code, but with different uses
    Feature selection introduces several common feature selection methods
    Feature evaluation Introduce and compare two important feature evaluation methods to help detect the importance of features
    Classification Type Feature Optimization-Leaf Coding Leaf Coding: Improved Tree Prediction Model for Classification Problems

Machine learning basic model building

    How does machine learn? Learn the definition of machine learning, what is overfit, and how to solve it
    The concept of training / test set segmentation Why do we need training / test set segmentation? Is there any way to split it?
    regression vs. classification What is the difference between a regression problem and a classification problem? How to define project goals
    How to choose evaluation metrics? What are the commonly used indicators?
    Introduction to regression model-linear regression / logistics regressionTheoretical basis and precautions of linear regression / logistics regression models
    regression model code writing How to use Scikit-learn to write code for linear regression / Logistics regression model
    Introduction to regression model-LASSO regression / Ridge regression The theoretical basis of LASSO regression / Ridge regression
    regression model code writing Scikit-learn code for LASSO regression / Ridge regression model
    tree based model-Introduction to the Decision Tree model The theoretical basis of the Decision Tree model and precautions when using it
    tree based model-Decision tree code writing Scikit-learn code for Decision Tree model
    tree based model-Introduction to Random Forest The theoretical basis of Random Forest model and precautions when using it
    tree based model-Writing Random Forest Code Using Scikit-learn to write Random Forest model code
    tree based model-Introduction to Gradient Boosting Machine The theoretical basis of Gradient Boosting Machine model and precautions when using it
    tree based model-Gradient Booster Code Writing Scikit-learn code for Gradient Boosting Machine model

Machine learning tuning parameters

    Hyperparameter adjustment and optimization What is a hyper-paramter? How to adjust hyperparameters correctly? What are the commonly used parameters?
    Kaggle Contest Platform Introduction Introducing the world's largest data science competition website. How to participate in the competition?
    Integration method: Blending What is integration? What are the integration methods? What is Blending's writing method and effect?
    Integration method: Stacking

51-53 Kaggle First Midterm Exam
Unsupervised machine learning

    clustering 1 Introduction to unsupervised machine learning Introduction to unsupervised learning, application scenarios
    clustering 2 K-means
    K-mean observation: use contour analysis.Unsupervised models are measured by special evaluation methods (not evaluation functions). Today we introduce you to one of them: contour analysis.
    clustering 3 hierarchical clustering algorithm
    Hierarchical clustering method Observation: Using 2D sample data set Non-supervised evaluation method: What is the 2D sample data set? How is it generated and used?
    dimension reduction 1-PCA
    PCA observation: use handwriting recognition data set. For a more complex example: sklearn version of handwriting recognition data set, demonstrate the PCA's dimensionality reduction and data interpretation capabilities.
    dimension reduction 2-T-SNE TSNE
    t-sne observation: clustering and manifold reduction What is manifold reduction? In addition to t-sne, are there other common manifold reduction methods?

Theory and Practice of Deep Learning

    Introduction to neural networks
    Deep learning experience: Model adjustment and learning curve Introduce the experience platform TensorFlow PlayGround and get a preliminary understanding of model adjustment
    Deep learning experience: start function and normalization Experience advanced version of deep learning parameter adjustment on TF PlayGround

Explore deep learning using Keras

    Keras installation and introduction
    Keras Dataset
    Keras Sequential API
    Keras Module API
    Multi-layer Perception
    Loss function
    Start function
    Gradient Descent
    Gradient Descent Mathematical Principles
    Introduction to BackPropagation
    Optimizers
    Details and techniques for training neural networks-Validation and overfit
    Precautions before training neural network Is the data properly processed? What are the computing resources? Are the hyperparameters set correctly?
    Details and techniques for training neural networks-learning rate effect compares the differences between different learning rates on the training process and results
    [Exercise Day] Combination and comparison of optimizer and learning rate
    Details and techniques for training neural networks-regularization
    Details and techniques for training neural networks-Dropout
    Details and techniques for training neural networks-Overview of methods for Batch normalization in response to overfit-Batch Normalization
    [Exercise Day] Combination and comparison of regularization / machine removal / batch standardization Exercise time: Hyper-parameters hodgepodge
    Details and techniques of training neural networks-Overview of methods for earlystop using callbacks function to respond to overfit-Early reluctance to brake (EarlyStopping)
    Details and techniques for training neural networks-use callbacks function to store model Use Keras' built-in callback function to store trained model
    Details and techniques for training neural networks-use the callbacks function to reduce the learning rate and use Keras' built-in callback function to reduce the learning rate
    Details and techniques for training neural networks-write your own callbacks function
    Details and techniques for training neural networks-writing your own Loss function
    Image recognition using traditional computer vision and machine learning Learn how to use traditional machine learning algorithms to process image recognition before the development of neural networks
    [Exercise Day] Using traditional computer vision and machine learning for image recognition Applying traditional computer vision methods + machine learning for CIFAR-10 classification

Convolutional neural networks for deep learning applications

    Introduction to Convolution Neural Network (CNN) Understand the importance of CNN and the structure of CNN
    Convolutional neural network architecture details why it is more suitable for image problems than DNN, and how to implement CNN on Keras
    Convolutional neural network-Convolution layer and parameter adjustment
    Convolutional neural network-pooling layers and parameter adjustmentsIntroducing CNN layers commonly used in Keras
    -CNN layers in Keras
-Complete the CIFAR-10 dataset with CNN Train CIFAR-10 with CNN and compare its differences with DNN
-Details and techniques for training convolutional neural networks-Handling large amounts of data How to use Python's generator?
-Details and techniques for training convolutional neural networks-Handling small amounts of data How to use data augmentation to improve accuracy?
-Details and techniques for training convolutional neural networks-Transfer learning What is transfer learning? How to use it?

Kaggle final exam

-Through Kaggle image recognition test, comprehensive application of deep learning course content, and experience the power of transfer learning

Optimizer

Optimizer

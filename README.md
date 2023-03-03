## Fake News Detection Project

This project aims to detect fake news using machine learning techniques in Python. It uses a dataset named **'train.csv'**, which can be downloaded from Kaggle. The project uses various Python libraries such as Pandas, Scikit-Learn, NLTK, and Numpy.

### Overview

Fake news is a growing problem in today's society, and this project aims to tackle this issue by using machine learning techniques to detect fake news. The project uses a dataset of news articles and their labels (fake or real) to train a machine learning model. The model is then used to predict the label of new news articles, helping to identify fake news more effectively.

### Dataset

The dataset used in this project is a collection of news articles with their labels (fake or real). The dataset is stored in the **'train.csv'** file, which contains the following columns:

- id: Unique identifier for the article
- title: Title of the article
- author: Author of the article
- text: Text of the article
- label: Label of the article (fake or real)
The dataset was collected from various sources and has been preprocessed to remove any personal identifying information.

### Installation

To run this project, you need to have Python 3 and pip installed on your machine. You can follow the below steps to install the dependencies:

1. Clone the repository:

git clone https://github.com/tiyi1/FakeNewsPrediction.git

2. Navigate to the project directory:

cd FakeNewsPrediction

3. Install the dependencies:

pip install -r requirements.txt


### Customizing the Preprocessing Steps or Model Parameters

To customize the preprocessing steps or model parameters, you can modify the relevant code in the **'preprocess.py'** and **'model.py'** files. For example, you can add additional preprocessing steps such as spell checking or part-of-speech tagging, or you can modify the hyperparameters of the SVM classifier such as the kernel function or regularization parameter.

### Libraries Used
- Pandas: Pandas is a Python library for data manipulation and analysis. It provides easy-to-use data structures and data analysis tools for working with structured data such as CSV files.
- Scikit-Learn: Scikit-Learn is a Python library for machine learning. It provides a wide range of algorithms for classification, regression, clustering, and other machine learning tasks, as well as tools for data preprocessing, model selection, and evaluation.
- NLTK: NLTK (Natural Language Toolkit) is a Python library for natural language processing. It provides tools for text preprocessing, tokenization, stemming, and other tasks related to working with text data.
- Numpy: Numpy is a Python library for numerical computing. It provides high-performance arrays and matrices for working with numerical data, as well as tools for linear algebra, Fourier analysis, and other mathematical operations.

### Usage
After installing the dependencies, you can run the project by executing the main.py file. The script takes the **'train.csv'** file as input and outputs the accuracy score of the trained model.


python main.py train.csv

The script will preprocess the data, split it into training and testing sets, train the model, and print the accuracy score. You can modify the preprocessing steps or model parameters by editing the preprocess.py and model.py files.

### Model

The machine learning model used in this project is a Support Vector Machine (SVM) classifier. The model is trained on the preprocessed news articles and their labels, and is then used to predict the label of new news articles.

The SVM classifier is a popular choice for text classification tasks due to its ability to handle high-dimensional feature spaces and its ability to handle non-linear decision boundaries.

### Performance

The performance of the model is evaluated using various metrics such as accuracy, precision, recall, and F1 score. The model achieves an accuracy of 0.93, precision of 0.92, recall of 0.95, and F1 score of 0.93 on the test data.

### Project Structure

The project is organized into the following files and directories:

- main.py: The main script that runs the project and prints the accuracy score.
- preprocess.py: The script that preprocesses the news articles by removing stop words, stemming, and lemmatizing the text.
- model.py: The script that trains the SVM classifier on the preprocessed data and makes predictions on new articles.
- data: The directory that contains the train.csv file.
- utils: The directory that contains utility functions used in the project.

Machine Learning & NLP Projects
This repository contains a collection of data science projects demonstrating skills in Machine Learning and Natural Language Processing (NLP). Each project is contained within its own Jupyter Notebook.

Table of Contents
Getting Started

Prerequisites

Installation

Projects

Project 1: Handwritten Digit Recognizer

Project 2: Twitter Sentiment Analysis

Getting Started
Follow these instructions to get a local copy of the projects up and running on your machine.

Prerequisites
You will need Python and pip installed on your system. It is highly recommended to use a virtual environment to manage your dependencies.

Python (3.8+)

pip

Installation
Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>

Create and activate a virtual environment (recommended):

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required libraries:
A requirements.txt file is the best way to manage this, but you can also install the key libraries manually:

pip install tensorflow numpy matplotlib pandas nltk jupyterlab

Projects
✍️ Project 1: Handwritten Digit Recognizer
File: Task_1_HandWrittenDigitRecogniser.ipynb

This project implements a deep learning model to recognize and classify handwritten digits from 0 to 9. It utilizes the famous MNIST dataset and a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

Key Libraries Used:

TensorFlow / Keras: For building and training the neural network.

NumPy: For numerical operations and data manipulation.

Matplotlib: For visualizing the dataset images and training results.

Running the Notebook:

Launch Jupyter Lab or Jupyter Notebook:

jupyter lab

Open the Task_1_HandWrittenDigitRecogniser.ipynb file.

Run the cells sequentially from top to bottom to train the model and see the predictions.

🐦 Project 2: Twitter Sentiment Analysis
File: Task_2_Twitter_Sentiment_Analysis.ipynb

This project analyzes text data (such as tweets) to determine its sentiment—classifying it as positive, negative, or neutral. It uses Natural Language Processing (NLP) techniques and the NLTK (Natural Language Toolkit) library, specifically the VADER sentiment analysis tool.

Key Libraries Used:

NLTK (Natural Language Toolkit): The primary library for NLP tasks.

Pandas: For data handling and manipulation.

NumPy: For numerical operations.

Running the Notebook:

Launch Jupyter Lab or Jupyter Notebook:

jupyter lab

Open the Task_2_Twitter_Sentiment_Analysis.ipynb file.

Important: The first time you run this, you will likely need to download the NLTK VADER lexicon. The notebook should contain a cell to do this. If not, you can run this Python code in a cell:

import nltk
nltk.download('vader_icon')

Run the cells sequentially to see the sentiment analysis results.

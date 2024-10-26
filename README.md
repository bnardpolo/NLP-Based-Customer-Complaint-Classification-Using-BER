# NLP-Based-Customer-Complaint-Classification-Using-BER
Multi-Class Text Classification Using BERT Model
Project Overview
This project demonstrates multi-class text classification using the pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. BERT, developed by Google, is an open-source framework that has achieved state-of-the-art performance in various NLP tasks. This project focuses on classifying customer complaints about consumer financial products into different categories based on the provided text.

Business Use Case
The dataset used in this project contains over two million customer complaints regarding financial products. The goal is to accurately classify these complaints based on the product type, which can help in addressing customer issues more efficiently.

Tech Stack
Programming Language: Python
Libraries:
Pandas
Torch (PyTorch)
NLTK (Natural Language Toolkit)
NumPy
Sklearn
Transformers (HuggingFace)
TQDM
Re, Pickle
Data Description
Input Dataset: The dataset consists of more than two million records of customer complaints, including two columns:
Complaint Text: The actual text of the complaint.
Product: The product associated with the complaint.
Project Structure
graphql
Copy code
├── Input
│   ├── complaints.csv            # The input dataset
├── Output
│   ├── bert_pre_trained.pth      # Pre-trained BERT model
│   ├── label_encoder.pkl         # Encoded labels
│   ├── labels.pkl                # Label mapping
│   ├── tokens.pkl                # Tokenized dataset
├── Source
│   ├── model.py                  # BERT model definition
│   ├── data.py                   # Data loading and processing
│   ├── utils.py                  # Utility functions for preprocessing
├── config.py                     # Configuration settings for the project
├── processing.py                 # Script for processing text data
├── predict.py                    # Script for making predictions
├── bert.ipynb                    # Jupyter notebook with detailed explanations and steps
├── Engine.py                     # Main engine file to run the entire pipeline
├── requirements.txt              # Required libraries and versions
└── README.md                     # Project documentation (this file)
Prerequisites
Before running the project, ensure that you have the following libraries installed. You can install them using the provided requirements.txt file:

bash
Copy code
pip install -r requirements.txt
The following knowledge is also beneficial:

Basic understanding of text classification algorithms like Naïve Bayes.
Familiarity with word embeddings (Skip Gram Model).
Experience building classification models using RNN and LSTM.
Understanding of Attention Mechanism in NLP.
Approach
1. Data Preprocessing
Reading the CSV file and dropping null values.
Cleaning the text: lowercasing, removing punctuation, digits, and unnecessary spaces.
Tokenizing the text.
Encoding labels for the target column.
2. Model Building
Using a pre-trained BERT model for text representation.
Defining PyTorch datasets for training, validation, and testing.
Splitting the data into training, validation, and test sets.
Creating PyTorch data loaders.
Defining the loss function, optimizer, and training the model.
3. Training the Model
Loading preprocessed data and training the BERT model.
Saving the trained model and tokenizer for future use.
4. Making Predictions
The trained BERT model can be used to predict the class for new text data.
How to Run the Project
Clone the repository.
Install the necessary dependencies:
bash
Copy code
pip install -r requirements.txt
Run the engine file to train the model:
bash
Copy code
python Engine.py
To make predictions using the trained model, run the predict.py script:
bash
Copy code
python predict.py
Key Takeaways
Understanding and utilizing pre-trained models like BERT for NLP tasks.
Performing text cleaning and preprocessing.
Tokenizing text data using BERT's tokenizer.
Building and training a BERT-based model using PyTorch.
Making predictions using the trained model on new data.

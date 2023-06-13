<h1 align="center">Buzz Wise Machine Learning Models</h1>
This is a documentation for the provided code.
<br>
<h3>Importing Required Libraries</h3>
import pandas as pd
import numpy as np
from scipy import sparse
import tensorflow.compat.v1 as tf
import os
<h3>Reading the Data</h3>
jobdata = r"C:\Users\jason\Downloads\data job posts.csv"
df = pd.read_csv(jobdata, error_bad_lines=False)
dataset = df['jobpost']
<br>
The code reads a CSV file named "data job posts.csv" located at the specified path (C:\Users\jason\Downloads). It uses the pandas library to read the data into a DataFrame (df). The 'jobpost' column from the DataFrame is stored in the dataset variable.
<h3>Data Preprocessing</h3>
replace_str = ['\r', '\n', 'TITLE:', ',', '"', '.', ':', '/', '(', ')', ';', '', '-', '1', '2', '3', '4',
               '5', '6', '7', '8', '9', '0', '@', 'Armenia', '#', '$', '+', '%', '&'
               , '!', '*', '?', '<', '>', '_', '[', ']', 'or', 'AA', 'AAA', 'AAAA', 'AAAAS', 'aaas', 'AAB', 'AAFF', 'AAFPC', 'aafpc', 'and', 'Yerevan']
replacement = ''

for string in replace_str:
    dataset = dataset.str.replace(string, replacement)
<br>
This code segment replaces specific strings defined in the replace_str list with an empty string. It loops through each string in the list and replaces occurrences of that string in the dataset variable (which contains the 'jobpost' column) using the str.replace() function.
<h3>Configuring Pandas Options</h3>
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
<br>
These lines of code configure pandas to display all rows and columns when printing DataFrames.
<h3>Data Preprocessing</h3>
dataset_list = dataset.tolist()
combined_keywords = ' '.join(dataset_list)
myKeywords = [word.lower() for word in combined_keywords.split() if len(word) > 2]
new_list = [' '.join(sublist) for sublist in [myKeywords[i:i+3] for i in range(0, len(myKeywords), 3)]]
<br>
This code converts the dataset column to a Python list (dataset_list). Then, it combines all the text from the list into a single string (combined_keywords). The string is then split into words, converted to lowercase, and stored in the myKeywords list as long as the word length is greater than 2. Finally, the myKeywords list is divided into sublists of three consecutive words, and these sublists are joined into a new list called new_list.
<h3>Word2Vec Model Preparation</h3>
corpus = new_list[0:6000]
words = myKeywords[0:9000]
word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())

WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

df = pd.DataFrame(data, columns=['input', 'label'])
<br>
This section prepares the data for training a Word2

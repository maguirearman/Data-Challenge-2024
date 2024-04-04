import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
from sklearn.metrics import (
    confusion_matrix,
)





#This function takes in text
#It returns the feature vector, and updates the wordlist to contain all word stems
def feature_vector(entries): 
    print("\ncreating feature vector...")
    matrix = []
    wordList = []
    stemmer = PorterStemmer()
    

    for text in entries:
        words = word_tokenize(text)
        # Remove stop words and punctuation
        words = [word for word in words if word.lower() not in stopwords.words('english')]
        words = [word for word in words if word.lower() not in string.punctuation]
        
        #TODO: Remove words not in medical dictionary (need to find a library for this)
        newRow = [0] * (len(wordList))
        
        #If the word is in the word list, increment the correct column in feature vector, 
        #otherwise append the word list and set feature vector to 1
        for word in words:
            stem = stemmer.stem(word.lower()) # this gets the 'stem' word in lowercase
            if(stem not in wordList):
                wordList.append(stem)
                newRow.append(1)
            else:
                newRow[wordList.index(stem)]+=1
        matrix.append(newRow)                   

    #normalize the row lengths
    count = len(wordList)
    for m in matrix:
       while len(m) < count :
          m.append(0)
    print("Finished!\n")
    return matrix



def main():
    #create df
    df = pd.read_csv("train/train_radiology.csv")
    
    #***************REMOVE THIS THIS IS ONLY FOR RUN TIME / TESTING PURPOSES****************************
    #df = df.head(250)
    #*****************************************

    #remove chart time, as it wont be a predictor in the model
    df.drop(columns = ['charttime'], inplace=True)

    #Group by patient_id and aggregate note_type and note_seq
    df_grouped = df.groupby('patient_id').agg({'note_type': lambda x: ''.join(sorted(set(x))),  # Combine unique note types
                                               'note_seq': 'max',  # Get the max note_seq
                                               'text': ' '.join})  # Combine all text

    # Reset index to have patient_id as a column
    df_grouped = df_grouped.reset_index()

    #change text to be a feature vector of word counts
    wordMatrix = feature_vector(df_grouped['text'])
    
    #replace each text entry with its corresponding row in the wordMatrix
    wordsDF = pd.DataFrame(wordMatrix)
    df_combined = pd.concat([df_grouped, wordsDF], axis=1)
    df_combined.drop(columns=['text'], inplace=True)
    
    #Change note type to be of levels 1, 2, or 3 depending on type
    df_combined['note_type'] = df_combined['note_type'].replace({'RR': 1, 'AR': 2, 'RRAR': 3})
    
    print(df_combined.head(20))


    sys.exit(0)
  
main()
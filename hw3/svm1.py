import pandas as pd
from scipy.stats import itemfreq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import nltk.stem


filename = 'heart.csv'
dataset = pd.read_csv(filename).values
clf = LinearSVC(C=1,max_iter=10000)

# kaggle_test=pd.read_csv("./kaggle/test.tsv", delimiter='\t')

# # preserve the id column of the test examples
# kaggle_ids=kaggle_test['PhraseId'].values
#
# # read in the text content of the examples
# kaggle_X_test=kaggle_test['Phrase'].values
#
# # vectorize the test examples using the vocabulary fitted from the 60% training data
# kaggle_X_test_vec=vec.transform(kaggle_X_test)

# predict using the NB classifier that we built
import time
starttime = time.time()
clf.fit(np.array(dataset[:,:-1]), np.array(dataset[:,-1]))
kaggle_pred=clf.predict(np.array(dataset[:,:-1]))

endtime = time.time()
dtime = endtime - starttime

print("used time：%.8s s" % dtime)

print(accuracy_score(np.array(dataset[:,-1]),kaggle_pred))





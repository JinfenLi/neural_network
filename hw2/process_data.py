import pandas as pd
import numpy as np
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import csv
from emoji.unicode_codes import UNICODE_EMOJI
from nltk.tokenize import TweetTokenizer
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from numpy import asarray, zeros
import emoji
import pickle
import random

def load_data():


    data=pd.read_csv('train.csv')
    texts1= data['question_text'].values
    labels1=data['target'].values
    i0=0
    i1=0
    texts=[]
    labels=[]
    for t,l in zip(texts1,labels1):
        if l==1 and i1<5000:
            texts.append(t)
            labels.append(l)
            i1+=1
        elif l==0 and i0<5000:
            texts.append(t)
            labels.append(l)
            i0+=1

    training_data = list(zip(texts, labels))
    random.seed(60)
    random.shuffle(training_data)
    texts = list(map(lambda x: x[0], training_data))
    labels = list(map(lambda x: x[1], training_data))

    return texts, labels
print("Loading data...")
texts,labels = load_data()

print(Counter(labels))


stopwords = []
slangs = {}
negated = {}



def load_lexicons():


    with open('lexicons/stopwords.txt', 'r') as f:
        for line in f:
            stopwords.append(line.strip())

    with open('lexicons/slangs.txt', 'r') as f:
        for line in f:
            splitted = line.strip().split(',', 1)
            slangs[splitted[0]] = splitted[1]

    with open('lexicons/negated_words.txt', 'r') as f:
        for line in f:
            splitted = line.strip().split(',', 1)
            negated[splitted[0]] = splitted[1]



load_lexicons()




def clean_tweets(texts):
    cleaned_tweets = []


    for text in texts:

        text=str(text)
        text = re.sub('(!){2,}', ' <!repeat> ', text)
        text = re.sub('(\?){2,}', ' <?repeat> ', text)

        # Tokenize using tweet tokenizer
        # tokens = nltk.word_tokenize(text.lower())
        tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
        tokens = tokenizer.tokenize(text.lower())



        # Replace slangs and negated words
        temp = []
        for word in tokens:
            if word in slangs:
                temp += slangs[word].split()
            elif word in negated:
                temp += negated[word].split()
            else:
                temp.append(word)
        tokens = temp

        # Replace user names
        tokens = ['<user>' if '@' in word else word for word in tokens]

        # Replace numbers
        tokens = ['<number>' if word.isdigit() else word for word in tokens]

        # Remove urls
        tokens = ['' if 'http' in word else word for word in tokens]


        # Remove stop words
        tokens = [word for word in tokens if word not in stopwords]

        # Remove tokens having length 1
        tokens = [word for word in tokens if word != '' and len(word) > 1]

        cleaned_tweets.append(tokens)

    return cleaned_tweets


def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_length(lines):
    # return np.average([len(s) for s in lines])
    return max([len(s) for s in lines])


def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

def clean_data():
    print("Cleaning Data...")
    cleaned_tweets = clean_tweets(texts)
    print("Cleaning Completed!")


    print("Encoding Data...")
    # For Tweet Matrix
    tokenizer_tweets = create_tokenizer(cleaned_tweets)
    max_tweet_length = max_length(cleaned_tweets)
    vocab_size = len(tokenizer_tweets.word_index) + 1
    print('Vocabulary size: %d' % vocab_size)
    print("max tweet length ",max_tweet_length)
    X = encode_text(tokenizer_tweets, cleaned_tweets, max_tweet_length)


    # Labels
    # lb = LabelBinarizer()
    # lb.fit(labels)
    Y = []
    for l in labels:
        if l==1:
            Y.append([0,1])
        else:
            Y.append([1,0])
    print(len(X))
    print(len(Y))

    print("Encoding Completed!")

    # Load embedding
    embedding_dir = 'lexicons/glove.6B.100d.txt'
    dimension = 100
    print("Loading word embeddings...")
    embeddings_index = dict()
    f = open(embedding_dir)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # Generate embedding matrices
    print("Generating embedding matrices...")
    tweet_matrix = zeros((vocab_size, dimension))
    for word, i in tokenizer_tweets.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            tweet_matrix[i] = np.array(list(embedding_vector))
        else:
            tweet_matrix[i] = np.array(list(np.random.uniform(low=-1, high=1, size=(100,))))

    print("Embedding matrices genearation completed!")
    with open('lexicons/model.p','wb') as file:
        pickle.dump((X, Y, tweet_matrix, vocab_size, dimension, max_tweet_length),file)

    return X, Y, tweet_matrix, vocab_size, dimension, max_tweet_length



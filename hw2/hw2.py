from __future__ import print_function, division
import operator
import numpy as np
from numpy import array
from sklearn.model_selection import KFold
from keras.layers import Bidirectional
import pickle
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,GRU,SimpleRNN
from keras.layers.merge import concatenate
from keras.models import Model
import process_data
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
import time

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.validation_data[:1], self.validation_data[1], verbose=0)
        val_predict = np.argmax(self.model.predict(self.validation_data[:1]),axis=1)
        val_targ = np.argmax(self.validation_data[1],axis=1)
        _val_f1 = f1_score(val_targ, val_predict,average='weighted')
        _val_recall = recall_score(val_targ, val_predict,average='weighted')
        _val_precision = precision_score(val_targ, val_predict,average='weighted')
        self.accs.append(acc)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(classification_report(val_targ, val_predict))



        print("-acc: % f — val_f1: % f — val_precision: % f — val_recall % f" % (acc,_val_f1, _val_precision, _val_recall))

        return



num_categories = 2
batch_size = 128
epochs = 10
seed = 60

print("processing data")
# X, Y, tweet_matrix, vocab_size, dimension, max_tweet_length= process_data.clean_data()
with open('lexicons/model.p','rb') as file:
    X, Y, tweet_matrix, vocab_size, dimension, max_tweet_length=pickle.load(file)



def Lstm(max_tweet_length,  vocab_size, tweet_matrix,
          dimension, num_categories, train_embedding=False):
    input1 = Input(shape=(max_tweet_length, ))

    embedding1 = Embedding(vocab_size, dimension, weights=[tweet_matrix],
                  trainable=False)(input1)
    # x1 = LSTM(300, return_sequences=False, dropout=0.25,
    #                        recurrent_dropout=0.25)(embedding1)
    # x1 = SimpleRNN(300, return_sequences=False, dropout=0.25,
    #           recurrent_dropout=0.25)(embedding1)
    # print(x1.shape)

    x1 = GRU(300, return_sequences=False, dropout=0.25,
              recurrent_dropout=0.25)(embedding1)

    x1 = Dense(256, activation="relu")(x1)
    x1 = Dropout(0.25)(x1)



    outputs = Dense(num_categories, activation='softmax')(x1)

    model = Model(inputs=input1, outputs=outputs)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_val():
    # X: n*max_len
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    metrics = Metrics()
    accuracies = []
    f1=[]
    precision = []
    recall = []
    counter = 1

    starttime = time.time()
    for train, test in kf.split(X):
        print('Fold#', counter)
        counter += 1
        model_GloVe = Lstm(max_tweet_length, vocab_size, tweet_matrix,
              dimension, num_categories, train_embedding=False)

        model_GloVe.fit(x=X[train],
                        y=array(Y)[train],validation_data=(X[test], np.array(Y)[test]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[metrics],
                        verbose=1)

        index, value = max(enumerate(metrics.accs), key=operator.itemgetter(1))
        max_f1_index = int(np.argmax(metrics.val_f1s))
        accuracies.append(value)
        f1.append(metrics.val_f1s[max_f1_index])
        precision.append(metrics.val_precisions[max_f1_index])
        recall.append(metrics.val_recalls[max_f1_index])
        print("each epoch's accuracy ",metrics.accs)
        print("each epoch's f1 score ",metrics.val_f1s)
        print("each epoch's precisions ",metrics.val_precisions)
        print("each epoch's recalls ",metrics.val_recalls)
        print("\n")

    print("five folds' average accuracy ", np.mean(accuracies))
    print("five folds' average f1 ", np.mean(f1))
    print("five folds' average precision ", np.mean(precision))
    print("five folds' average recall ", np.mean(recall))
    endtime = time.time()
    dtime = endtime - starttime

    print("used time：%.8s s" % dtime)

train_val()




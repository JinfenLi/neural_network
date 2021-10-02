import numpy as np
import nltk, itertools, csv
import pandas as pd
import random
TXTCODING = 'utf-8'
unknown_token = 'UNKNOWN_TOKEN'
start_token = 'START_TOKEN'
end_token = 'END_TOKEN'

# 解析评论文件为数值向量
class tokenFile2vector:
    def __init__(self, file_path, dict_size):
        self.file_path = file_path
        self.dict_size = dict_size

    # 将文本拆成句子，并加上句子开始和结束标志
    def _get_sentences(self,file_path):

        data=pd.read_csv(file_path)
        sents=data['question_text'][:500].values
        label=data['target'][:500].values
        training_data=list(zip(sents,label))
        random.seed(60)
        random.shuffle(training_data)
        sents=list(map(lambda x:x[0],training_data))
        label = list(map(lambda x: x[1], training_data))


        print ('Get {} sentences.'.format(len(sents)))

        return sents,label

    # 得到每句话的单词，并得到字典及字典中每个词的下标
    def _get_dict_wordsIndex(self, sents):
        sent_words = [nltk.word_tokenize(sent) for sent in sents]
        word_freq = nltk.FreqDist(itertools.chain(*sent_words))
        print('Get {} words.'.format(len(word_freq)))

        common_words = word_freq.most_common(self.dict_size-1)
        # 生成词典
        dict_words = [word[0] for word in common_words]
        dict_words.append(unknown_token)
        # 得到每个词的下标，用于生成词向量
        index_of_words = dict((word, ix) for ix, word in enumerate(dict_words))

        return sent_words, dict_words, index_of_words

    # 得到训练数据
    def get_vector(self):

        # dict_size = 100
        sents,label = self._get_sentences(self.file_path)
        sent_words, dict_words, index_of_words = self._get_dict_wordsIndex(sents)
        max_length=max([len(sent) for sent in sent_words])
        print("max length is ",max_length)

        # 将每个句子中没包含进词典dict_words中的词替换为unknown_token
        for i, words in enumerate(sent_words):
            sent_words[i] = [w if w in dict_words else unknown_token for w in words]
        X=[]

        for x in sent_words:
            seq=[[0]*self.dict_size]*max_length
            for i in range(len(x)):
                j=int(index_of_words[x[i]])
                seq[i][j]=1
            X.append(seq)
        X=np.array(X)
        final_label=[]
        for l in label:
            if l==1:
                final_label.append([0,1])
            else:
                final_label.append([1,0])
        final_label=np.array(final_label)
        return X, final_label, dict_words, index_of_words,max_length


# myTokenFile = tokenFile2vector('train.csv', 50)
# X_train, y_train, dict_words, index_of_words = myTokenFile.get_vector()

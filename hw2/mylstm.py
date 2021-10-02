import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#number of input units or embedding size
input_units = 100

#number of hidden neurons
hidden_units = 256

#number of output units i.e vocab size
output_units = 27

#learning rate
learning_rate = 0.005

#beta1 for V parameters used in Adam Optimizer
beta1 = 0.90

#beta2 for S parameters used in Adam Optimizer
beta2 = 0.99

def load_name():
    #data
    path = r'NationalNames.csv'
    data = pd.read_csv(path)

    #get names from the dataset
    data['Name'] = data['Name']

    #get first 10000 names
    data = np.array(data['Name'][:10000]).reshape(-1,1)

    #covert the names to lowee case
    data = [x.lower() for x in data[:,0]]

    data = np.array(data).reshape(-1,1)
    return data

def pad_name(data):
    # to store the transform data, [['mary........'],['anna........']...]
    transform_data = np.copy(data)

    # find the max length name
    max_length = 0
    for index in range(len(data)):
        max_length = max(max_length, len(data[index, 0]))
    print("the longest name has %d c"%max_length)

    # make every name of max length by adding '.'
    for index in range(len(data)):
        length = (max_length - len(data[index, 0]))
        string = '.' * length
        transform_data[index, 0] = ''.join([transform_data[index, 0], string])

    return transform_data


def store_voc(transform_data):
    # to store the vocabulary
    vocab = list()
    for name in transform_data[:, 0]:
        vocab.extend(list(name))

    vocab = set(vocab)
    vocab_size = len(vocab)

    print("Vocab size = {}".format(vocab_size))

    return vocab


def map_char_id(vocab):
    char_id = dict()
    id_char = dict()

    for i, char in enumerate(vocab):
        char_id[char] = i
        id_char[i] = char

    # print('a-{}, 22-{}'.format(char_id['a'], id_char[22]))
    return char_id,id_char


def load_train_data():
    data = load_name()
    transform_data=pad_name(data)
    vocab=store_voc(transform_data)
    char_id, id_char=map_char_id(vocab)
    # list of batches of size = 20
    train_dataset = []

    # batch_size = 20
    # batch_dataset=[]
    # if len(transform_data)<batch_size:
    #     batch_dataset = transform_data
    # else:
    #     for i in range(len(transform_data)// batch_size+1):
    #         start = i * batch_size
    #         end = start + batch_size
    #         batch_data=transform_data[start:end]
    #
    #         batch_dataset.append(batch_data)
    # # print(np.array(batch_dataset))
    #
    # for batch in batch_dataset:
    #     batchlist=[]
    #     for name in batch:
    #         seqlist = []
    #
    #         for k in range(len(name[0])):
    #             onehotlist=np.zeros(len(vocab))
    #             char_index=char_id[name[0][k]]
    #             onehotlist[char_index]=1.0
    #             seqlist.append(onehotlist)
    #         seqlist = np.array(seqlist)
    #         batchlist.append(seqlist)
    #     train_dataset.append(batchlist)
    # batch*batch_size*seq*voc
    batch_size = 20

    # split the trasnform data into batches of 20
    for i in range(len(transform_data) - batch_size + 1):
        start = i * batch_size
        end = start + batch_size

        # batch data
        batch_data = transform_data[start:end]

        if (len(batch_data) != batch_size):
            break

        # convert each char of each name of batch data into one hot encoding
        char_list = []
        for k in range(len(batch_data[0][0])):
            batch_dataset = np.zeros([batch_size, len(vocab)])
            for j in range(batch_size):
                name = batch_data[j][0]
                char_index = char_id[name[k]]
                batch_dataset[j, char_index] = 1.0

            # store the ith char's one hot representation of each name in batch_data
            char_list.append(batch_dataset)

        # store each char's of every name in batch dataset into train_dataset
        train_dataset.append(char_list)
    return np.array(train_dataset)
print(np.array(load_train_data()[0]).shape)

#sigmoid
def sigmoid(X):
    return 1/(1+np.exp(-X))

#tanh activation
def tanh_activation(X):
    return np.tanh(X)

#softmax activation
# X:[[y0,y1,y2],[y0,y1,y2]]
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
    exp_X = exp_X/exp_X_sum
    return exp_X

#derivative of tanh
def tanh_derivative(X):
    return 1-(X**2)


# initialize parameters
def initialize_parameters():
    # initialize the parameters with 0 mean and 0.01 standard deviation
    mean = 0
    std = 0.01

    # lstm cell weights
    # fa = sigmoid(Wf * [xt,at-1])
    # ia = sigmoid(Wi * [xt, at - 1])
    # ga = tanh(Wg * [xt, at - 1]) (Ct(~))
    # oa = sigmoid(Wo * [xt, at - 1])
    # ct = (fa * ct-1) + (ia * ga)
    # at = oa * tanh(ct)
    forget_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    input_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    output_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))
    gate_gate_weights = np.random.normal(mean, std, (input_units + hidden_units, hidden_units))

    # hidden to output weights (output cell)
    hidden_output_weights = np.random.normal(mean, std, (hidden_units, output_units))

    parameters = dict()
    parameters['fgw'] = forget_gate_weights
    parameters['igw'] = input_gate_weights
    parameters['ogw'] = output_gate_weights
    parameters['ggw'] = gate_gate_weights
    parameters['how'] = hidden_output_weights

    return parameters


# single lstm cell
def lstm_cell(batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters):
    # get parameters
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    ggw = parameters['ggw']

    # concat batch data and prev_activation matrix
    # batch_datset: batch_size*emb_size
    concat_dataset = np.concatenate((batch_dataset, prev_activation_matrix), axis=1)

    # forget gate activations fa:seq*hidden_size
    fa = np.matmul(concat_dataset, fgw)
    fa = sigmoid(fa)

    # input gate activations
    ia = np.matmul(concat_dataset, igw)
    ia = sigmoid(ia)

    # output gate activations
    oa = np.matmul(concat_dataset, ogw)
    oa = sigmoid(oa)

    # gate gate activations
    ga = np.matmul(concat_dataset, ggw)
    ga = tanh_activation(ga)

    # new cell memory matrix
    cell_memory_matrix = np.multiply(fa, prev_cell_matrix) + np.multiply(ia, ga)

    # current activation matrix
    activation_matrix = np.multiply(oa, tanh_activation(cell_memory_matrix))

    # lets store the activations to be used in back prop
    lstm_activations = dict()
    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['oa'] = oa
    lstm_activations['ga'] = ga

    return lstm_activations, cell_memory_matrix, activation_matrix


def output_cell(activation_matrix, parameters):
    # get hidden to output parameters  seq*hidden_size x hidden_size*output_units
    # ot = W * at
    # ot = softmax(ot)
    how = parameters['how']

    # get outputs
    output_matrix = np.matmul(activation_matrix, how)
    output_matrix = softmax(output_matrix)

    return output_matrix

def get_embeddings(batch_dataset,embeddings):
    # batch_dataset:batch_size*voc_size, embeddings: voc_size*emb_size
    embedding_dataset = np.matmul(batch_dataset,embeddings)
    return embedding_dataset


# forward propagation
#batch: batch_size*voc_size
def forward_propagation(batches, parameters, embeddings):
    # get batch size
    batch_size = batches[0].shape[0]

    # to store the activations of all the unrollings.
    lstm_cache = dict()  # lstm cache
    activation_cache = dict()  # activation cache
    cell_cache = dict()  # cell cache
    output_cache = dict()  # output cache
    embedding_cache = dict()  # embedding cache

    # initial activation_matrix(a0) and cell_matrix(c0)
    a0 = np.zeros([batch_size, hidden_units], dtype=np.float32)
    c0 = np.zeros([batch_size, hidden_units], dtype=np.float32)

    # store the initial activations in cache
    activation_cache['a0'] = a0
    cell_cache['c0'] = c0

    # unroll the names, -1: the last one do not need to forward
    for i in range(len(batches) - 1):
        # get first first character batch
        batch_dataset = batches[i]

        # get embeddings batch_dataset:batch_size*emb_size
        batch_dataset = get_embeddings(batch_dataset, embeddings)
        embedding_cache['emb' + str(i)] = batch_dataset

        # lstm cell
        lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)

        # output cell
        ot = output_cell(at, parameters)

        # store the time 't' activations in caches
        lstm_cache['lstm' + str(i + 1)] = lstm_activations
        activation_cache['a' + str(i + 1)] = at
        cell_cache['c' + str(i + 1)] = ct
        output_cache['o' + str(i + 1)] = ot

        # update a0 and c0 to new 'at' and 'ct' for next lstm cell
        a0 = at
        c0 = ct

    return embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache
import tokenFile
import numpy as np

# 输出单元激活函数
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

class myLSTM:
    def __init__(self, output_size, voc_size, hidden_dim=100):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.voc_size=voc_size


        # 初始化权重向量
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()
        self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   ( self.hidden_dim, self.output_size)), \
                           np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   self.output_size)

    # 初始化 wh, wx, b
    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),
                                   (self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0/self.voc_size), np.sqrt(1.0/self.voc_size),
                                   (self.voc_size,self.hidden_dim))
        b = np.random.uniform(-np.sqrt(1.0/self.voc_size), np.sqrt(1.0/self.voc_size),
                              (1,self.hidden_dim))

        return wh, wx, b

    # 初始化各个状态向量
    def _init_s(self, T):
        iss = np.array([np.zeros(self.hidden_dim)] * (T+1 ))  # input gate +1 because initiate the first element
        fss = np.array([np.zeros(self.hidden_dim)] * (T+1 ))  # forget gate
        oss = np.array([np.zeros(self.hidden_dim)] * (T+1 ))  # output gateƒ
        ass = np.array([np.zeros(self.hidden_dim)] * (T+1 ))  # current inputstate
        hss = np.array([np.zeros(self.hidden_dim)] * (T+1 ))  # hidden state
        css = np.array([np.zeros(self.hidden_dim)] * (T+1 ))  # cell state
        ys = np.array([np.zeros(self.output_size)]*(T+1)  ) # output value the first is useless

        return {'iss': iss, 'fss': fss, 'oss': oss,
                'ass': ass, 'hss': hss, 'css': css,
                'ys': ys}

    # 前向传播，单个x
    def forward(self, x):
        # 向量时间长度
        T = len(x)

        # 初始化各个状态向量
        stats = self._init_s(T)
        # print(stats['iss'].shape)
        for t in range(1,T+1):
            # 前一时刻隐藏状态
            ht_pre = np.array(stats['hss'][t-1])

            # input gate
            stats['iss'][t] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t-1], sigmoid)
            # forget gate
            stats['fss'][t] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t-1], sigmoid)
            # output gate
            stats['oss'][t] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t-1], sigmoid)
            # current inputstate c_hat
            stats['ass'][t] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t-1], tanh)

            # cell state, ct = ft * ct_pre + it * at
            stats['css'][t] = stats['fss'][t] * stats['css'][t-1] + stats['iss'][t] * stats['ass'][t]
            # hidden state, ht = ot * tanh(ct)
            stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])

            # output value, yt = softmax(self.wy.dot(ht) + self.by)
            # print(stats['hss'][t])
            # print(stats['ys'])
            stats['ys'][t] = softmax(stats['hss'][t].dot(self.wy) + self.by)
            # print(stats['ys'][t].shape)

        return stats

    # 计算各个门的输出
    def _cal_gate(self, wh, wx, b, ht_pre, x, activation):
        return activation(ht_pre.dot(wh) + x.dot(wx) + b)

    # 预测输出，单个x
    def predict(self, x):
        stats = self.forward(x)
        pre_y = np.argmax(stats['ys'])
        return pre_y

    # 计算损失， softmax交叉熵损失函数， (x,y)为多个样本
    def loss(self, x, y):
        cost = 0
        for i in range(len(y)):
            stats = self.forward(x[i])
            # 取出 y[i] 中最后时刻对应的预测值
            pre_yi = stats['ys'][-1]
            # print(pre_yi)
            # print(y[i])
            cost += np.sum(np.multiply(y[i],np.log(pre_yi)) + np.multiply(1-y[i],np.log(1-pre_yi)))

        # 统计所有y中词的个数, 计算平均损失
        N = len(y)
        ave_loss = cost / N

        return ave_loss

     # 初始化偏导数 dwh, dwx, db
    def _init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)

        return dwh, dwx, db

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dwhi, dwxi, dbi = self._init_wh_wx_grad()
        dwhf, dwxf, dbf = self._init_wh_wx_grad()
        dwho, dwxo, dbo = self._init_wh_wx_grad()
        dwha, dwxa, dba = self._init_wh_wx_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((1,self.hidden_dim))

        # 前向计算
        stats = self.forward(x)
        # print(stats['ys'][0].shape)
        # 目标函数对输出 y 的偏导数 delta_o=pi-yi  stats['yes]:(t+1)*output_size
        # print(stats['ys'][-1])
        loss=-np.sum(np.multiply(y[-1],np.log(stats['ys'][-1])) + np.multiply(1-y[-1],np.log(1-stats['ys'][-1])))
        # print("loss is %f"%-np.sum(np.multiply(y[-1],np.log(stats['ys'][-1])) + np.multiply(1-y[-1],np.log(1-stats['ys'][-1]))))
        delta_o = stats['ys']
        delta_o[np.arange(len(y)), y] -= 1
        # print(delta_o.shape)
        # print(stats['hss'][-1].shape)

        for t in np.arange(len(x))[::-1]:
            # 输出层wy, by的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
            # print(stats['hss'][t].reshape(-1,1).shape)
            # print(delta_o[t].shape)
            dwy += stats['hss'][t].reshape(-1,1).dot(delta_o[t].reshape(1,-1))
            dby += delta_o[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.dot(delta_o[t]).reshape(1,-1)
            # print(delta_ht.shape)

            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * tanh(stats['css'][t])

            delta_ct += delta_ht * stats['oss'][t] * (1-tanh(stats['css'][t])**2)
            # print(delta_ct.shape)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t-1]
            delta_at = delta_ct * stats['iss'][t]

            delta_at_net = delta_at * (1-stats['ass'][t]**2)
            delta_it_net = delta_it * stats['iss'][t] * (1-stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1-stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1-stats['oss'][t])

            # 更新各权重矩阵的偏导数，由于所有时刻共享权值，故所有时刻累加
            dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x[t])
            dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x[t])
            dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x[t])
            dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x[t])

        return [dwhf, dwxf, dbf,
                dwhi, dwxi, dbi,
                dwha, dwxa, dba,
                dwho, dwxo, dbo,
                dwy, dby,loss]

    # 更新各权重矩阵的偏导数
    def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        # print(delta_net.shape)
        # print(x.shape)
        dwx += delta_net * x
        db += delta_net

        return dwh, dwx, db

    # 计算梯度, (x,y)为一个样本
    def sgd_step(self, x, y, learning_rate):
        dwhf, dwxf, dbf, \
        dwhi, dwxi, dbi, \
        dwha, dwxa, dba, \
        dwho, dwxo, dbo, \
        dwy, dby,losses = self.bptt(x, y)

        # 更新权重矩阵
        self.whf, self.wxf, self.bf = self._update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.whi, self.wxi, self.bi = self._update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
        self.wha, self.wxa, self.ba = self._update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
        self.who, self.wxo, self.bo = self._update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)

        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby
        return losses

    # 更新权重矩阵
    def _update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db

        return wh, wx, b

    # 训练 LSTM
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=5):
        losses = []
        num_examples = 0

        for epoch in range(n_epoch):
            loss=0
            for i in range(len(y_train)):
                loss+=self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1
            print('epoch {0}: loss = {1}'.format(epoch + 1, loss/len(y_train)))
            losses.append(loss)

            # loss = self.loss(X_train, y_train)
            # losses.append(loss)
            # print ('epoch {0}: loss = {1}'.format(epoch+1, loss))
            # if len(losses) > 1 and losses[-1] > losses[-2]:
            #     learning_rate *= 0.5
            #     print ('decrease learning_rate to', learning_rate)
# file_path = r'/home/display/pypys/practices/rnn/results-20170508-103637.csv'
dict_size = 100
myTokenFile = tokenFile.tokenFile2vector('train.csv', dict_size)
X_train, y_train, dict_words, index_of_words ,seq_len= myTokenFile.get_vector()
# print(X_train.shape)
# print(y_train.shape)

# 训练LSTM
lstm = myLSTM(2,dict_size,hidden_dim=100)
lstm.train(X_train[:1000], y_train[:1000],
          learning_rate=0.001,
          n_epoch=10)

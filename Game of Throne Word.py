import time
from collections import namedtuple
import gensim 
import os
import numpy as np
import tensorflow as tf
from itertools import chain

def get_batches(arr, n_seqs, n_steps, model_size):
    '''
    对已有的数组进行mini-batch分割
    
    arr: 待分割的数组
    n_seqs: 一个batch中序列个数
    n_steps: 单个序列包含的字符数
    '''
    
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # 这里我们仅保留完整的batch，对于不能整出的部分进行舍弃
    arr = arr[:batch_size * n_batches]
    
    # 重塑
    arr = arr.reshape((n_seqs, -1, model_size))
    
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n+n_steps, :]
        # targets
        y = np.zeros_like(x)
        y[:, :-1, :], y[:, -1, :] = x[:, 1:, :], x[:, 0, :]
        yield x, y



def build_inputs(num_seqs, num_steps, model_size):
    '''
    构建输入层
    
    num_seqs: 每个batch中的序列个数
    num_steps: 每个序列包含的字符数
    '''
    
    inputs = tf.placeholder(tf.float32, shape=(num_seqs, num_steps, model_size), name='inputs')
    targets = tf.placeholder(tf.float32, shape=(num_seqs, num_steps, model_size), name='targets')
    
    # 加入keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' 
    构建lstm层
        
    keep_prob
    lstm_size: lstm隐层中结点数目
    num_layers: lstm的隐层数目
    batch_size: batch_size

    '''
    # 构建一个基本lstm单元
 
    def drop_cell():
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # 堆叠
    cell = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    ''' 
    构造输出层
        
    lstm_output: lstm层的输出结果
    in_size: lstm输出层重塑后的size
    out_size: softmax层的size
    
    '''

    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
    # tf.concat的结果是[1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output, axis=1) # tf.concat(concat_dim, values)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])
    
    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    
    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b
    
    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')
    
    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    '''
    根据logits和targets计算损失
    
    logits: 全连接层的输出结果（不经过softmax）
    targets: targets
    ls|tm_size
    num_classes: vocab_size
        
    '''
    y_reshaped = tf.reshape(targets, logits.get_shape())
    
    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    
    return loss


def build_optimizer(loss, learning_rate, grad_clip):
    ''' 
    构造Optimizer
   
    loss: 损失
    learning_rate: 学习率
    
    '''
    
    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer

class CharRNN:
    
    def __init__(self, num_classes, batch_size=64, num_steps=50, model_size=200, 
                       lstm_size=128, num_layers=2, learning_rate=0.001, 
                       grad_clip=5, sampling=False):
    
        # 如果sampling是True，则采用SGD
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # 输入层
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps, model_size)

        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        # 对输入进行one-hot编码
        
        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
        self.final_state = state
        
        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)
        
        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        

def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符
    
    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="Game of throne"):
    """
    生成新文本
    
    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = prime.split()
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(model_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in samples:
            x = np.zeros((1, 1, model_size))
            # 输入单个字符
            x[0,0,:] = model_Word2Vec[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, model_size)
        # 添加字符到samples中
        samples.append(int_to_vocab[c])
        
        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0,0,:] = model_Word2Vec[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)
#########################################################################################



os.chdir("C:/Users/cxin/Documents/zhihu/anna_lstm")

text = []
for i in range(1, 6):
    txt_name = os.path.join('Game of Throne', '00'+str(i)+'ssb.txt') 
    for line in open(txt_name,'r').readlines():
        text.append(line.split())

model_size = 200        
model_Word2Vec = gensim.models.Word2Vec(text, size=model_size, min_count=0)
# model.similarity('Jon', 'Jaime')
# model.most_similar(positive=['Jon', 'Theon'], negative=['Jaime'], topn=1)

text = list(chain.from_iterable(text))
    
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([model_Word2Vec[c] for c in text], dtype=np.float)


batches = get_batches(encoded, 10, 50, 200)
x, y = next(batches)

    

batch_size = 100         # Sequences per batch
num_steps = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5         # Dropout keep probability



epochs = 1
# 每n轮进行一次变量保存
save_every_n = 200

model = CharRNN(model_size, batch_size=batch_size, num_steps=num_steps, model_size = model_size,
                lstm_size=lstm_size, num_layers=num_layers, 
                learning_rate=learning_rate)


saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps, model_size):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            
            end = time.time()
            # control the print lines
            if counter % 100 == 0:
                print('轮数: {}/{}... '.format(e+1, epochs),
                      '训练步数: {}... '.format(counter),
                      '训练误差: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    

checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 1000, lstm_size, model_size, prime="The")
print(samp)
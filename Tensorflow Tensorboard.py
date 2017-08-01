import input_data
import tensorflow as tf

def weight_variable(shape, names):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = names)

def bias_variable(shape, names):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = names)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
  
  
###########################################################

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

##########################################################

graphCNN = tf.Graph()
with graphCNN.as_default():

    # define x and y_
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    
    x_image = tf.reshape(x, [-1,28,28,1])
      
    
    with tf.name_scope('CNN1'):
        W_conv1 = weight_variable([5, 5, 1, 32], 'weights1')
        b_conv1 = bias_variable([32], 'bias1')
        variable_summaries(W_conv1)
        variable_summaries(b_conv1)
        
    with tf.name_scope('CNN1-Act'):
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)   
        tf.summary.histogram('post_activations1', h_pool1)
    
    
    with tf.name_scope('CNN2'):
        W_conv2 = weight_variable([5, 5, 32, 64], 'weights2')
        b_conv2 = bias_variable([64], 'bias2')
        variable_summaries(W_conv2)
        variable_summaries(b_conv2)
    
    with tf.name_scope('CNN2-Act'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram('post_activations2', h_pool2)
    
    with tf.name_scope('FC1'):
        # 7 = 28/2/2
        W_fc1 = weight_variable([7 * 7 * 64, 1024], 'weightFC1')
        b_fc1 = bias_variable([1024], 'biasFC1')
        variable_summaries(W_fc1)
        variable_summaries(b_fc1)
    
    with tf.name_scope('FC1-Act'):
        # reshape the h_pool2 for the fully connected layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    tf.summary.histogram('post_activations2', h_fc1)

  
    with tf.name_scope('Dropout'):       
        keep_prob = tf.placeholder("float")
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    
    with tf.name_scope('FC2'):
        W_fc2 = weight_variable([1024, 10], 'weightFC2')
        b_fc2 = bias_variable([10], 'biasFC2')
        variable_summaries(W_fc2)
        variable_summaries(b_fc2)
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)


#######################################################################


# with tf.Session(graph=graphCNN) as sess:
        
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    summary_writer = tf.summary.FileWriter("tmp/tensorflowlogs", sess.graph)  # not /tmp/...
    
    sess.run(init)
    
    for i in range(1000):
        
        batch = mnist.train.next_batch(50)
        
        if i%500 == 0:   
            # notice accuracy.eval; predicting using keep prob 1.0
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})   
            print("step %d, training accuracy %g"%(i, train_accuracy))
        
        # notice train_step.run; trainning using keep prob 0.5
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                
        if i%100 == 0:   
           summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
           summary_writer.add_summary(summary_str, i)
           summary_writer.flush() 
    


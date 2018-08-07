import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

def deconv2d_wo_bias(layer, stride, class_num, batch_size, name="deconv2d_wo_bias"):
    """
    A simple 2-dimensional convolutional transpose layer.
    Layer Architecture: conv2d.tranpose
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - stride: (int) size of the stride to do (usually 16 or 32)
    - class_num: (int) number of filters to be made
    - batch_size: (int) the explicit batch size
    - name: (string) unique name for this convolution layer
    """
    _,m,n,c = layer.get_shape().as_list()
    weight_shape = [stride*2, stride*2, class_num, c]
    with tf.device("/cpu:0"):
        with tf.variable_scope(name+"_param"):
            W = tf.get_variable("W", weight_shape,
                                initializer=tf.random_normal_initializer(stddev=0.01))
    layer = tf.nn.conv2d_transpose(layer, W, [batch_size, m*stride, n*stride, class_num],
                                   strides=[1,stride,stride,1], padding='SAME')
    return layer

def deconv2d_w_bias(layer, stride, class_num, batch_size, name="deconv2d_w_bias"):
    """
    A simple 2-dimensional convolutional transpose layer.
    Layer Architecture: conv2d.tranpose
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - stride: (int) size of the stride to do (usually 16 or 32)
    - class_num: (int) number of filters to be made
    - batch_size: (int) the explicit batch size
    - name: (string) unique name for this convolution layer
    """
    layer = deconv2d_wo_bias(layer, stride, class_num, batch_size, name=name)
    with tf.device("/cpu:0"):
        with tf.variable_scope(name+"_param"):
            B = tf.get_variable("B", class_num, initializer=tf.constant_initializer(0.0))
    layer += B
    return layer
    

def conv2d_wo_bias(layer, filt_size, filt_num, stride=1, name="conv2d_wo_bias"):
    """
    A simple 2-dimensional convolution layer.
    Layer Architecture: 2d-convolution
    All weights are created with a (hopefully) unique scope.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - filt_size: (int) size of the square filter to be made
    - filt_num: (int) number of filters to be made
    - stride: (int) stride of our convolution
    - name: (string) unique name for this convolution layer
    """
    # Creating and Doing the Convolution.
    input_size = layer.get_shape().as_list()
    if not isinstance(filt_size, list):
        filt_size = [filt_size, filt_size]
    weight_shape = [filt_size[0], filt_size[1], input_size[3], filt_num]
    std = np.sqrt(2.0 / (filt_size[0] * filt_size[0] * input_size[3]))
    with tf.device("/cpu:0"):
      with tf.variable_scope(name+"_param"):
        W = tf.get_variable("W", weight_shape,
                            initializer=tf.random_normal_initializer(stddev=std))
    tf.add_to_collection("reg_variables", W)
    layer = tf.nn.conv2d(layer, W, strides=[1, stride, stride, 1], padding='SAME')
    return layer

def conv2d_w_bias(layer, filt_size, filt_num, stride=1, name="conv2d_w_bias"):
    """
    A simple 2-dimensional convolution layer.
    Layer Architecture: 2d-convolution
    All weights are created with a (hopefully) unique scope.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - filt_size: (int) size of the square filter to be made
    - filt_num: (int) number of filters to be made
    - stride: (int) stride of our convolution
    - name: (string) unique name for this convolution layer
    """
    # Doing the conv2d
    layer = conv2d_wo_bias(layer, filt_size, filt_num, stride=1, name=name)
    # Creating weights for bias.
    with tf.device("/cpu:0"):
        with tf.variable_scope(name+"_param"):
            B = tf.get_variable('B', shape=[filt_num], initializer=tf.zeros_initializer())
    layer = tf.nn.bias_add(layer, B)
    return layer

def batch_norm(layer, is_training, beta=0.0, gamma=1.0, decay=0.9, stddev=0.002, name="bn"):
    """
    Does batch normalization.
    Heavily based off of the code from tflearn.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - is_training: (bool) are we in training size
    - beta: (float) initialize the beta shift (trainable)
    - gamma: (float) initialize the gamma scale (trainable)
    - decay: (float) between 0 and 1.  Should be leaky learning rate.
    - stddev: (float) standard dev of random normal initializer for scaling.  should be small.
    - name: (string) name of the layer
    """
    epsilon=1e-5
    # Determining the sizes.
    input_size = layer.get_shape().as_list()
    input_dim = len(input_size)
    axis = list(range(input_dim - 1))
    # Creating Variables
    gamma_init = tf.random_normal_initializer(mean=gamma, stddev=stddev)
    with tf.device("/cpu:0"):
      with tf.variable_scope(name + "_param"):
        beta = tf.get_variable('beta', shape=[input_size[-1]],
                               initializer=tf.constant_initializer(beta))
        gamma = tf.get_variable('gamma', shape=[input_size[-1]],
                                initializer=gamma_init)
        mov_mean = tf.get_variable('mov_mean', input_size[-1:],
                                   initializer=tf.zeros_initializer(), trainable=False)
        mov_var = tf.get_variable('mov_var', input_size[-1:],
                                  initializer=tf.ones_initializer(), trainable=False)
    # Creating Functions
    def update_mean_var():
        mean, var = tf.nn.moments(layer, axis)
        update_mov_mean = moving_averages.assign_moving_average(mov_mean, mean, decay)
        update_mov_var = moving_averages.assign_moving_average(mov_var, var, decay)
        with tf.control_dependencies([update_mov_mean, update_mov_var]):
            return tf.identity(mean), tf.identity(var)
    # Doing the batch norm
    mean,var = tf.cond(is_training > 0, update_mean_var, lambda:(mov_mean, mov_var))
    inference = tf.nn.batch_normalization(layer, mean, var, beta, gamma, epsilon)
    return inference

def conv2d_bn_relu(layer, is_training, filt_size, filt_num, stride=1, alpha=0.0, name="conv2d_bn_relu"):
    """
    A simple 2-dimensional convolution layer.
    Layer Architecture: 2d-convolution - batch_norm - reLU
    All weights are created with a (hopefully) unique scope.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - is_training: (bool) are we in training size
    - filt_size: (int) size of the square filter to be made
    - filt_num: (int) number of filters to be made
    - stride: (int) stride of our convolution
    - alpha: (float) for the leaky ReLU.  Do 0.0 for ReLU.
    - name: (string) unique name for this convolution layer
    """
    # Doing the conv2d
    layer = conv2d_wo_bias(layer, filt_size, filt_num, stride=stride, name=name)
    # Normalization
    layer = batch_norm(layer, is_training, name=name)
    # ReLU
    if alpha != 1:
        layer = tf.maximum(layer, layer*alpha)
    return layer

def max_pool(layer, k=2, stride=None):
    """
    A simple 2-dimensional max pooling layer.
    Strides and size of max pool kernel is constrained to be the same.
    INPUTS:
    - layer: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - k: (int) size of the max_filter to be made.
    - stride: (int) size of stride
    """
    if stride==None:
        stride=k
    # Doing the Max Pool
    max_layer = tf.nn.max_pool(layer, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding='SAME')
    return max_layer

def dense_wo_bias(layer, hidden_size, name="dense_wo_bias"):
    """
    Dense (Fully Connected) layer.
    Architecture: reshape - Affine
    INPUTS:
    - layer: (tensor.2d or more) basically, of size [batch_size, etc...]
    - hidden_size: (int) Number of hidden neurons.
    - name: (string) unique name for layer.
    """
    # Flatten Input Layer
    input_size = layer.get_shape().as_list()
    reshape_size = 1
    for iter_size in range(1, len(input_size)):
        reshape_size *= input_size[iter_size]
    reshape_layer = tf.reshape(layer, [-1, reshape_size])
    # Creating and Doing Affine Transformation
    weight_shape = [reshape_layer.get_shape().as_list()[1], hidden_size]
    std = np.sqrt(2.0 / reshape_layer.get_shape().as_list()[1])
    with tf.device("/cpu:0"):
      with tf.variable_scope(name+"_param"):
        W = tf.get_variable("W", weight_shape, initializer=tf.random_normal_initializer(stddev=std))
    tf.add_to_collection("reg_variables", W)
    tf.add_to_collection("l1_variables", W)
    layer = tf.matmul(reshape_layer, W)
    return layer

def dense_w_bias(layer, hidden_size, name="dense_w_bias"):
    """
    Dense (Fully Connected) layer.
    Architecture: reshape - Affine - bias
    Tips: Use as OUTPUT layer.
    INPUTS:
    - layer: (tensor.2d or more) basically, of size [batch_size, etc...]
    - hidden_size: (int) Number of hidden neurons.
    - name: (string) unique name for layer.
    """
    # Do the Math
    layer = dense_wo_bias(layer, hidden_size, name=name)
    with tf.device("/cpu:0"):
        with tf.variable_scope(name+"_param"):
            b = tf.get_variable("b", hidden_size, initializer=tf.constant_initializer(0.0))
    layer += b
    return layer

def dense_bn_do_relu(layer, is_training, hidden_size, keep_prob, alpha=0.0, name="dense_bn_do_relu"):
    """
    Dense (Fully Connected) layer.
    Architecture: reshape - Affine - batch_norm - dropout - relu
    WARNING: should not be the output layer.  Use "output" for that.
    INPUTS:
    - layer: (tensor.2d or more) basically, of size [batch_size, etc...]
    - is_training: (bool) are we in training size
    - hidden_size: (int) Number of hidden neurons.
    - keep_prob: (float) Probability to keep neuron during dropout layer.
    - alpha: (float) Slope for leaky ReLU.  Set 0.0 for ReLU.
    - name: (string) unique name for layer.
    """
    layer = dense_wo_bias(layer, hidden_size, name=name)
    # Batch Normalization
    layer = batch_norm(layer, is_training, name=name)
    # Dropout
    layer = tf.nn.dropout(layer, keep_prob)
    # ReLU
    if alpha != 1:
        layer = tf.maximum(layer, layer*alpha)
    return layer


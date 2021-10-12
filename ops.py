import tensorflow as tf
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers import variance_scaling_initializer as he_init
# from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

"""

pytorch xavier (gain)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html

USE < tf.contrib.layers.variance_scaling_initializer() >
if uniform :
    factor = gain * gain
    mode = 'FAN_AVG'
else :
    factor = (gain * gain) / 1.3
    mode = 'FAN_AVG'

pytorch : trunc_stddev = gain * sqrt(2 / (fan_in + fan_out))
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / (fan_in + fan_out))

"""

"""
pytorch kaiming (a=0)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html

if uniform :
    a = 0 -> gain = sqrt(2)
    factor = gain * gain
    mode='FAN_IN'
else :
    a = 0 -> gain = sqrt(2)
    factor = (gain * gain) / 1.3
    mode = 'FAN_OUT', # FAN_OUT is correct, but more use 'FAN_IN

pytorch : trunc_stddev = gain * sqrt(2 / fan_in)
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / fan_in)

"""

# Xavier : tf.contrib.layers.xavier_initializer()
# He : tf.contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(0.02)

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf.contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) # orthogonal_regularizer_fully(0.0001)

# factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
# weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)

# weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
weight_regularizer_fully = tf.contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layers
##################################################################################

# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        # if scope.__contains__("discriminator") :
        #     weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        # else :
        #     weight_init = tf_contrib.layers.xavier_initializer()
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                            x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                       strides=[1, stride, stride, 1], padding=padding)

            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init,
                                           kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x


def conv_pixel_shuffle_up(x, scale_factor=2, use_bias=True, sn=False, scope='pixel_shuffle'):
    channel = x.get_shape()[-1] * (scale_factor ** 2)
    x = conv(x, channel, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope=scope)
    x = tf.depth_to_space(x, block_size=scale_factor)

    return x


def conv_pixel_shuffle_down(x, scale_factor=2, use_bias=True, sn=False, scope='pixel_shuffle'):
    channel = x.get_shape()[-1] // (scale_factor ** 2)
    x = conv(x, channel, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope=scope)
    x = tf.space_to_depth(x, block_size=scale_factor)

    return x


def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x


##################################################################################
# Blocks
##################################################################################

def resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = batch_norm(x, is_training)

        if channels != x_init.shape[-1]:
            with tf.variable_scope('skip'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)
                return relu(x + x_init)

        return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, sn=False, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            x = adaptive_instance_norm(x, mu, sigma)

        return x + x_init




def self_attention(x, use_bias=True, sn=False, scope='self_attention'):
    with tf.variable_scope(scope):
        channels = x.shape[-1]
        f = conv(x, channels // 8, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='f_conv')  # [bs, h, w, c']
        g = conv(x, channels // 8, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='g_conv')  # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn, scope='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x


def scconv(x, use_bias=False, sn=False, scope='scconv'):
    with tf.variable_scope(scope):
        x1, x2 = tf.split(x, 2, axis=-1)
        with tf.variable_scope('k2'):
            x12 = tf.layers.average_pooling2d(x1, pool_size=4, strides=4, padding='SAME')
            x12 = conv(x12, x1.shape[-1], kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
            # x12 = deconv(x12, int(x1.shape[-1]), kernel=4, stride=4, use_bias=use_bias, sn=sn)
            # x12 = nearest_up_sample(x12, scale_factor=4)
            x12 = up_pooling(x12, pool_size=4)
            x12 = x12 + x1
            x12 = sigmoid(x12)
        with tf.variable_scope('k3'):
            x13 = conv(x1, x1.shape[-1], kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
        with tf.variable_scope('k4'):
            y1 = conv(x12*x13, x1.shape[-1], kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
        with tf.variable_scope('k1'):
            y2 = conv(x2, x2.shape[-1], kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
        return tf.concat([y1, y2], axis=-1)


##################################################################################
# Normalization
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm'):
    return tf.contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP
    # See https://github.com/taki0112/MUNIT-Tensorflow

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def adaptive_layer_instance_norm(x, gamma, beta, smoothing=True, scope='ada_layer_instance_norm') :
    # proposed by UGATIT
    # https://github.com/taki0112/UGATIT
    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

        if smoothing :
            rho = tf.clip_by_value(rho - tf.constant(0.1), 0.0, 1.0)

        x_hat = rho * x_ins + (1 - rho) * x_ln


        x_hat = x_hat * gamma + beta

        return x_hat


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Activation Function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x):
    return tf.sigmoid(x)


def swish(x):
    return x * tf.sigmoid(x)


def elu(x):
    return tf.nn.elu(x)


##################################################################################
# Pooling & Resize
##################################################################################

def nearest_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    # return tf.image.resize_images(x, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def bilinear_up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_bilinear(x, size=new_size)

def nearest_down_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor, w // scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def bilinear_down_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor, w // scale_factor]
    return tf.image.resize_bilinear(x, size=new_size)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap


def global_max_pooling(x):
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    return gmp


def max_pooling(x, pool_size=2):
    x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=pool_size, padding='SAME')
    return x


def avg_pooling(x, pool_size=2):
    x = tf.layers.average_pooling2d(x, pool_size=pool_size, strides=pool_size, padding='SAME')
    return x


def up_pooling(x, pool_size=2):
    _, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1,1,1,h*w*c])
    x = tf.tile(x, [1,pool_size,pool_size,1])
    x = tf.split(x, h*w, axis=-1)
    x = tf.concat(x, axis=2)
    x = tf.split(x, h, axis=2)
    x = tf.concat(x, axis=1)
    return x


def flatten(x):
    return tf.layers.flatten(x)


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


##################################################################################
# Loss Function
##################################################################################

def classification_loss(logit, label):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

def classification_loss2(logit, label):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy

# def classification_loss2(logit, label) :
#     logit = sigmoid(logit)
#     loss = tf.reduce_mean(-label*tf.log(tf.clip_by_value(logit,1e-10,1.0))-(1-label)*tf.log(tf.clip_by_value((1-logit),1e-10,1.0)))
#     prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
#     accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

#     return loss, accuracy


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss


def huber_loss(x, y):
    return tf.losses.huber_loss(x, y)


def regularization_loss(scope_name):
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization:
        if scope_name in item.name:
            loss.append(item)

    return tf.reduce_sum(loss)



def normalization(x):
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))
    return x




##################################################################################
# GAN Loss Function
##################################################################################

def discriminator_loss(gan_type, real, fake, Ra = 0):
    # Ra = Relativistic
    real_loss = 0
    fake_loss = 0

    if Ra and (gan_type.__contains__('wgan') or gan_type == 'sphere'):
        print("No exist [Ra + WGAN or Ra + Sphere], so use the {} loss function".format(gan_type))
        Ra = False

    if Ra:
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))

        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(real_logit - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake_logit + 1.0))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real_logit))
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake_logit))

        if gan_type == 'hinge':
            real_loss = tf.reduce_mean(relu(1.0 - real_logit))
            fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))

    else:
        if gan_type.__contains__('wgan'):
            real_loss = -tf.reduce_mean(real)
            fake_loss = tf.reduce_mean(fake)

        if gan_type == 'lsgan':
            real_loss = tf.reduce_mean(tf.square(real - 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

        if gan_type == 'hinge':
            real_loss = tf.reduce_mean(relu(1.0 - real))
            fake_loss = tf.reduce_mean(relu(1.0 + fake))

        if gan_type == 'sphere':
            bs, c = real.get_shape().as_list()
            moment = 3
            north_pole = tf.one_hot(tf.tile([c], multiples=[bs]), depth=c + 1)  # [bs, c+1] -> [0, 0, 0, ... , 1]

            real_projection = inverse_stereographic_projection(real)
            fake_projection = inverse_stereographic_projection(fake)

            for i in range(1, moment + 1):
                real_loss += -tf.reduce_mean(tf.pow(sphere_loss(real_projection, north_pole), i))
                fake_loss += tf.reduce_mean(tf.pow(sphere_loss(fake_projection, north_pole), i))


    loss = real_loss + fake_loss

    return loss


def generator_loss(gan_type, real, fake, Ra = 0):
    # Ra = Relativistic
    fake_loss = 0
    real_loss = 0

    if Ra and (gan_type.__contains__('wgan') or gan_type == 'sphere'):
        print("No exist [Ra + WGAN or Ra + Sphere], so use the {} loss function".format(gan_type))
        Ra = False

    if Ra:
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))

        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))
            real_loss = tf.reduce_mean(tf.square(real_logit + 1.0))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake_logit))
            real_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real_logit))

        if gan_type == 'hinge':
            fake_loss = tf.reduce_mean(relu(1.0 - fake_logit))
            real_loss = tf.reduce_mean(relu(1.0 + real_logit))

    else:
        if gan_type.__contains__('wgan'):
            fake_loss = -tf.reduce_mean(fake)

        if gan_type == 'lsgan':
            fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

        if gan_type == 'gan' or gan_type == 'gan-gp' or gan_type == 'dragan':
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

        if gan_type == 'hinge':
            fake_loss = -tf.reduce_mean(fake)

        if gan_type == 'sphere':
            bs, c = real.get_shape().as_list()
            moment = 3
            north_pole = tf.one_hot(tf.tile([c], multiples=[bs]), depth=c + 1)  # [bs, c+1] -> [0, 0, 0, ... , 1]

            fake_projection = inverse_stereographic_projection(fake)

            for i in range(1, moment + 1):
                fake_loss += -tf.reduce_mean(tf.pow(sphere_loss(fake_projection, north_pole), i))

    loss = fake_loss + real_loss

    return loss



def simple_gp(real_logit, fake_logit, real_images, fake_images, r1_gamma=10, r2_gamma=0):
    # Used in StyleGAN

    r1_penalty = 0
    r2_penalty = 0

    if r1_gamma != 0:
        real_loss = tf.reduce_sum(real_logit)  # In some cases, you may use reduce_mean
        real_grads = tf.gradients(real_loss, real_images)[0]

        r1_penalty = 0.5 * r1_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))

    if r2_gamma != 0:
        fake_loss = tf.reduce_sum(fake_logit)  # In some cases, you may use reduce_mean
        fake_grads = tf.gradients(fake_loss, fake_images)[0]

        r2_penalty = 0.5 * r2_gamma * tf.reduce_mean(tf.reduce_sum(tf.square(fake_grads), axis=[1, 2, 3]))

    return r1_penalty + r2_penalty

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, Lambda
from tensorflow.keras.activations import sigmoid

# 定义 h_sigmoid 类，继承自 tf.keras.layers.Layer。
class h_sigmoid(Layer):
    def __init__(self):
        super(h_sigmoid, self).__init__()

    def call(self, inputs):
        return tf.nn.relu6(inputs + 3) / 6

# 定义 h_swish 类，继承自 tf.keras.layers.Layer。
class h_swish(Layer):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def call(self, inputs):
        return inputs * self.sigmoid(inputs)

# 定义 CoordAtt 类，继承自 tf.keras.layers.Layer。
class CoordAtt(Layer):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)
        self.pool_h = Lambda(lambda x: tf.reduce_mean(x, axis=2, keepdims=True))
        self.pool_w = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))
        self.conv1 = Conv2D(mip, kernel_size=1, strides=1, padding="valid")
        self.bn1 = BatchNormalization()
        self.act = h_swish()
        self.conv_h = Conv2D(oup, kernel_size=1, strides=1, padding="valid")
        self.conv_w = Conv2D(oup, kernel_size=1, strides=1, padding="valid")

    def call(self, inputs):
        identity = inputs
        #print("input shape", identity.shape)
        n, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        #print(f"h shape:", {h.numpy()})
        #print(f"w shape:", {w.numpy()})
        x_h = self.pool_h(inputs)
        #print("x_h shape:", x_h.shape)
        x_w = self.pool_w(inputs)
        #print("x_w shape:", x_w.shape)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])
        #print("x_w shape:", x_w.shape)

        y = tf.concat([x_h, x_w], axis=1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        #print("Shape of y:", y.shape)

        x_h, x_w = tf.split(y, [h, w], axis=1)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])

        a_h = sigmoid(self.conv_h(x_h))
        a_w = sigmoid(self.conv_w(x_w))
        # print("identity shape:", identity.shape)
        # print("a_w shape:", a_w.shape)
        # print("a_h shape:", a_h.shape)
        out = identity * a_w * a_h
        return out



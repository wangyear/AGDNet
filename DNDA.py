import tensorflow as tf
from tensorflow.keras import layers, models

class Activate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Activate, self).__init__(**kwargs)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.01)

    def call(self, inputs):
        return self.activation(inputs)

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_layers = models.Sequential([
            layers.Conv2D(64, 3, padding='same'),
            Activate(),
            layers.Conv2D(32, 3, padding='same'),
            Activate(),
            layers.Conv2D(16, 3, padding='same'),
            Activate(),
            layers.Conv2D(1, 3, padding='same')
        ])

    def call(self, inputs):
        return self.conv_layers(inputs)

class SELayer(tf.keras.layers.Layer):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channel // reduction, activation='relu')
        self.fc2 = layers.Dense(channel, activation='sigmoid')

    def call(self, inputs):
        b, h, w, c = inputs.shape
        y = self.avg_pool(inputs)
        y = tf.reshape(y, (b, c))
        y = self.fc2(self.fc1(y))
        y = tf.reshape(y, (b, 1, 1, c))
        return inputs * y

class Encoder(tf.keras.Model):
    def __init__(self, alpha):
        super(Encoder, self).__init__()
        self.alpha = alpha  # Store alpha
        self.Conv1 = layers.Conv2D(32, 3, padding='same')
        self.Activate = Activate()

        self.Conv_d = layers.Conv2D(16, 3, padding='same')
        self.conv_layers = {
            'DenseConv1': layers.Conv2D(16, 3, padding='same'),
            'DenseConv2': layers.Conv2D(16, 3, padding='same'),
            'DenseConv3': layers.Conv2D(16, 3, padding='same')
        }

        self.Conv2 = layers.Conv2D(64, 3, strides=2, padding='same')
        self.Conv3 = layers.Conv2D(128, 3, strides=2, padding='same')
        self.Conv4 = layers.Conv2D(64, 3, strides=2, padding='same')
        self.Upsample = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')

    def con(self, x, isTest=False):
        x = self.Activate(self.Conv1(x))
        x_d = self.Activate(self.Conv_d(x))
        x_s = x

        for key in self.conv_layers:
            out = self.Activate(self.conv_layers[key](x_d))
            x_d = tf.concat([x_d, out], axis=-1)

        x_s = self.Activate(self.Conv2(x_s))
        x_s = self.Activate(self.Conv3(x_s))
        x_s = self.Activate(self.Conv4(x_s))
        x_s = self.Upsample(x_s)

        if isTest:
            x_s = tf.image.resize(x_s, size=(x.shape[1], x.shape[2]), method='bilinear')

        # Apply weights during concatenation
        out = tf.concat([
            x_d * self.alpha,  # Apply weight1 to semantic feature
            x_s * (1 - self.alpha)   # Apply weight2 to detail feature
        ], axis=-1)
        return out

    def call(self, x, isTest=False):
        x = self.con(x, isTest=isTest)
        return x

class DNDA(tf.keras.Model):
    def __init__(self, alpha):
        super(DNDA, self).__init__()
        self.encoder = Encoder(alpha)  # Pass alpha to encoder
        self.decoder = Decoder()

    def call(self, x, isTest=False):
        x = self.encoder(x, isTest=isTest)
        out = self.decoder(x)
        return out
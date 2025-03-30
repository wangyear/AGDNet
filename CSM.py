import tensorflow as tf
from tensorflow.keras import layers, Model

class ChannelAttention(layers.Layer):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化和最大池化
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()

        # MLP层
        self.fc1 = layers.Dense(in_channels // reduction_ratio, use_bias=False, activation='relu')
        self.fc2 = layers.Dense(in_channels, use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        # 平均池化和最大池化的输出
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))

        # 将两个输出相加
        out = avg_out + max_out
        attention = self.sigmoid(out)
        attention = tf.reshape(attention, (-1, 1, 1, attention.shape[-1]))  # 变为 [batch_size, 1, 1, num_channels]
        return attention

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # 卷积层
        self.conv = layers.Conv2D(1, kernel_size, padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        # 平均池化和最大池化
        avg_out = tf.reduce_mean(x, axis=3, keepdims=True)
        max_out = tf.reduce_max(x, axis=3, keepdims=True)
        
        # 拼接后通过卷积层
        x = tf.concat([avg_out, max_out], axis=3)
        x = self.conv(x)
        return self.sigmoid(x)

class CSM(layers.Layer):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CSM, self).__init__()
        
        # 通道注意力和空间注意力
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, x):
        # 通道注意力
        out = self.channel_attention(x) * x
        # 空间注意力
        out = self.spatial_attention(out) * out
        return out

# 示例用法
# if __name__ == "__main__":
#     input_tensor = tf.random.normal((1, 32, 32, 64))  # 创建一个随机输入张量
#     cbam = CBAM(in_channels=64)
#     output_tensor = cbam(input_tensor)
#     print(output_tensor.shape)

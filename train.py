import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LayerNormalization,MaxPooling2D, MaxPool2D, Add, Dropout, Concatenate, Conv2DTranspose, Dense, Reshape, Flatten, Softmax, Lambda, UpSampling2D, AveragePooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import datetime
from DNDA import DNDA
from CSM import CSM
from CA import CoordAtt
'''texts = [
    "tumor epithelial tissue",
    "necrotic tissue",
    "lymphocytic tissue",
    "tumor associated stromal tissue",
    "coagulative necrosis",
    "liquefactive necrosis",
    "desmoplasia",
    "granular and non granular leukocytes",
    "perinuclear halo",
    "interstitial space",
    "neutrophils",
    "macrophages",
    "collagen",
    "fibronectin",
    "hyperplasia",
    "dysplasia"
]'''
# You can replace the prior information here.
# Read text labels
text = pd.read_csv(r'CSV file to your folder', header=None)
text = tf.convert_to_tensor(np.asarray(text), dtype=tf.float32)
text = Dense(32, activation='relu')(text)
text = Dense(32, activation='relu')(tf.transpose(text, perm=[1,0]))
text = tf.expand_dims(text, axis=0)
text = tf.expand_dims(text, axis=-1)






class DistributionModel(tf.keras.Model):
    def __init__(self, text):
        super(DistributionModel, self).__init__()
        self.mean_layer = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=1, padding='same', activation='softplus')
        self.stddev_layer = tf.keras.layers.SeparableConv2D(filters=1, kernel_size=1, padding='same', activation='softplus')
        self.distribution_layer = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=t[..., 1:])
        )
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_t1 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.conv_t2 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.conv_t3 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.conv_t4 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding="same", activation='relu')
        self.text = text
        self.conv_tt1 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')
        self.conv_tt2 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')
        self.conv_tt3 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')
        self.conv_tt4 = tf.keras.layers.Conv2DTranspose(1, 2, strides=2, padding="same", activation='relu')

        #计算x和text的均值和标准差
    def distribution(self, x, text):

        mean = self.mean_layer(x) + tf.math.reduce_mean(text, axis=[-1, -2, -3])
        stddev = tf.math.sqrt(tf.math.square(self.stddev_layer(x)) + tf.math.reduce_variance(text, axis=[-1, -2, -3]))

        parameters = self.concat([mean, stddev])
        distribution = self.distribution_layer(parameters)
        return distribution

    def distribution_attn(self, x):
        x1 = self.conv_t1(x)
        x2 = self.conv_t2(x1)
        x3 = self.conv_t3(x2)
        x4 = self.conv_t4(x3)
        text = self.conv_tt1(self.text)
        dis1 = self.distribution(x1, text)
        text = self.conv_tt2(text)
        dis2 = self.distribution(x2, text)
        text = self.conv_tt3(text)
        dis3 = self.distribution(x3, text)
        text = self.conv_tt4(text)
        dis4 = self.distribution(x4, text)
        return dis1, dis2, dis3, dis4

    def call(self, inputs):
        return self.distribution_attn(inputs)

def conv_block(x, num_filters, kernel_size, padding="same", act=True,dropout_rate=0.5):
    x = Conv2D(num_filters, kernel_size, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    return x

def multires_block(x, num_filters, alpha=1.67, dropout_rate=0.55):   #you can adjust parameters by yourself
    W = num_filters * alpha
    x0 = x
    x1 = conv_block(x0, int(W * 0.167), 3)
    x2 = conv_block(x1, int(W * 0.333), 3)
    x3 = conv_block(x2, int(W * 0.5), 3)
    xc = Concatenate()([x1, x2, x3])
    xc = BatchNormalization()(xc)
    x = Dropout(dropout_rate)(x)
    nf = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    sc = conv_block(x0, nf, 1, act=False)
    x = Activation("relu")(xc + sc)
    x = BatchNormalization()(x)
    return x

def res_path(x, num_filters, length):
    check = L.GlobalMaxPooling2D()(x)
    check = L.Dense(1, activation='sigmoid')(x)
    check = tf.math.reduce_mean(check, axis=0)
    x01 = x
    x11 = conv_block(x01, num_filters, 3, act=False)
    sc1 = conv_block(x01, num_filters, 1, act=False)
    x = Activation("relu")(x11 + sc1)

    CSM = CSM(x.shape[3])
    x = CSM(x)

    x = BatchNormalization()(x)
    x02 = Concatenate()([x, x01])
    x12 = conv_block(x02, num_filters, 3, act=False)
    sc2 = conv_block(x02, num_filters, 1, act=False)
    x = Activation("relu")(x12 + sc2)
    x = BatchNormalization()(x)
    x03 = Concatenate()([x, x01, x02])
    x13 = conv_block(x03, num_filters, 3, act=False)
    sc3 = conv_block(x03, num_filters, 1, act=False)
    x = Activation("relu")(x13 + sc3)
    x = BatchNormalization()(x)
    x04 = Concatenate()([x, x01, x02, x03])
    x14 = conv_block(x04, num_filters, 3, act=False)
    sc4 = conv_block(x04, num_filters, 1, act=False)
    x = Activation("relu")(x14 + sc4)
    CSM = CSM(x.shape[3])
    x = CSM(x)

    x = BatchNormalization()(x)
    return x * check

def encoder_block(x, num_filters, length, dropout_rate=0.55):
    x = multires_block(x, num_filters)
    s = res_path(x, num_filters, length)
    p = MaxPooling2D((2, 2))(x)
    p = Dropout(dropout_rate)(p)
    return s, p

def decoder_block(x, skip, num_filters, dropout_rate=0.55):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = multires_block(x, num_filters)
    x = Dropout(dropout_rate)(x)
    return x

def build_multiresunet(shape, text):
    inputs = Input(shape)
    Dnda = DNDA(alpha=0.5)
    dense_fuse_outputs = Dnda(inputs)
    s1, p1 = encoder_block(dense_fuse_outputs, 32, 4)
    s2, p2 = encoder_block(p1, 64, 4)
    s3, p3 = encoder_block(p2, 128, 4)
    s4, p4 = encoder_block(p3, 256, 4)
    b1 = multires_block(p4, 512)
    dis1, dis2, dis3, dis4 = DistributionModel(text)(b1)

    d1 = decoder_block(b1, s4, 256)   #D5编码器
    d1 = d1 * dis1


    ca1 = CoordAtt(d1.shape[3], d1.shape[3])
    d1 = ca1(d1)
    d1_fuesd = Concatenate()([p3, d1])


    d2 = decoder_block(d1_fuesd, s3, 128)   #D4编码器
    d2 = d2 * dis2

    ca2 = CoordAtt(d2.shape[3], d2.shape[3])
    d2 = ca2(d2)
    d2_fused = Concatenate()([p2, d2])

    d3 = decoder_block(d2_fused, s2, 64)    #D3编码器
    d3 = d3 * dis3

    ca3 = CoordAtt(d3.shape[3], d3.shape[3])
    d3 = ca3(d3)
    d3_fused = Concatenate()([p1, d3])

    d4 = decoder_block(d3_fused, s1, 32)    #D2编码器
    d4 = d4 * dis4
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="MultiResUNET")
    return model

def dice_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def iou(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / (union + K.epsilon())

def combined_loss(y_true, y_pred):
    bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1 - dice_score(y_true, y_pred)
    return bce_loss + dice_loss




def load_images_and_masks(image_dir, mask_dir, image_size=(512, 512)):
    images = []
    masks = []
    for img_filename in os.listdir(image_dir):
        img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, img_filename), target_size=image_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)
        #mask_filename = img_filename.replace('.png', '.png')
        mask_filename = img_filename.replace('.png', '_bin_mask.png')  # or other corresponding mask file format
        mask = tf.keras.preprocessing.image.load_img(os.path.join(mask_dir, mask_filename), color_mode="grayscale", target_size=image_size)
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0
    return images, masks


# Load images and masks
image_dir = 'path/to/your/trainingdatasets/folder'
mask_dir =  'path/to/your/trainingdatasets/folder'

images, masks = load_images_and_masks(image_dir, mask_dir)

# Split data into training and validation sets
image_train, image_test, label_train, label_test = train_test_split(images, masks, test_size=0.1, random_state=40)
#image_train, label_train = images, masks







# Print shapes to verify
# print("Training images shape:", image_train.shape)
# print("Training masks shape:", image_test.shape)
# print("Testing images shape:", label_train.shape)
# print("Testing masks shape:", label_test.shape)


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, metrics_file_path):
        super(MetricsLogger, self).__init__()
        self.metrics_file_path = metrics_file_path

    def on_train_begin(self, logs=None):
        # 在训练开始时设置一个空的文件
        with open(self.metrics_file_path, 'w') as f:
            f.write("Epoch\tLoss\tAccuracy\tDice Score\tRecall\tPrecision\tIoU\tVal Loss\tVal Accuracy\tVal Dice Score\tVal Recall\tVal Precision\tVal IoU\n")

    def on_epoch_end(self, epoch, logs=None):
        # 每个 epoch 结束时记录指标
        with open(self.metrics_file_path, 'a') as f:
            f.write(f"{epoch + 1}\t{logs['loss']:.4f}\t"
                     f"{logs['accuracy']:.4f}\t"
                     f"{logs['dice_score']:.4f}\t"
                     f"{logs['recall']:.4f}\t"
                     f"{logs['precision']:.4f}\t"
                     f"{logs['iou']:.4f}\t"
                     f"{logs['val_loss']:.4f}\t"
                     f"{logs['val_accuracy']:.4f}\t"
                     f"{logs['val_dice_score']:.4f}\t"
                     f"{logs['val_recall']:.4f}\t"
                     f"{logs['val_precision']:.4f}\t"
                     f"{logs['val_iou']:.4f}\n")
        # 输出到控制台（可选）
        print(f"Epoch {epoch + 1} finished. "
              f"Loss: {logs['loss']:.4f}, "
              f"Accuracy: {logs['accuracy']:.4f}, "
              f"Dice Score: {logs['dice_score']:.4f}, "
              f"Recall: {logs['recall']:.4f}, "
              f"Precision: {logs['precision']:.4f}, "
              f"IoU: {logs['iou']:.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, "
              f"Val Accuracy: {logs['val_accuracy']:.4f}, "
              f"Val Dice Score: {logs['val_dice_score']:.4f}, "
              f"Val Recall: {logs['val_recall']:.4f}, "
              f"Val Precision: {logs['val_precision']:.4f}, "
              f"Val IoU: {logs['val_iou']:.4f}")

if __name__ == "__main__":
    # Build and compile the model
    model = build_multiresunet((512, 512, 3), text)
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss=combined_loss, metrics=["accuracy", dice_score, recall, precision, iou], optimizer=optimizer)
    model.summary()

    # 添加模型保存回调
    checkpoint_path = "model_checkpoints/best_model.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='dice_score',
        mode='max',
        save_best_only=True)

    # 添加TensorBoard回调
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 指标保存文件路径
    metrics_file_path = "training_metrics.txt"
    metrics_logger_callback = MetricsLogger(metrics_file_path)

    # Train model
    history = model.fit(image_train, label_train, epochs=60, batch_size=1,
                        validation_data=(image_test, label_test),
                        callbacks=[model_checkpoint_callback, tensorboard_callback, metrics_logger_callback])

    # Evaluate the model
    evaluation = model.evaluate(image_test, label_test, batch_size=2)
    print(f"Test Loss: {evaluation[0]}")
    print(f"Test Accuracy: {evaluation[1]}")
    print(f"Test Dice Score: {evaluation[2]}")
    print(f"Test Recall: {evaluation[3]}")
    print(f"Test Precision: {evaluation[4]}")
    print(f"Test IOU: {evaluation[5]}")

    # 进行预测
    predictions = model.predict(image_test, batch_size=2)

    # 可视化一些预测结果
    num_images_to_display = 3
    fig, axes = plt.subplots(num_images_to_display, 3, figsize=(15, num_images_to_display * 5))
    for i in range(num_images_to_display):
        axes[i, 0].imshow(image_test[i])
        axes[i, 0].set_title("Original Image")
        axes[i, 1].imshow(label_test[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("True Mask")
        axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Predicted Mask")
    plt.show()
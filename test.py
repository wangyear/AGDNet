import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from train import build_multiresunet,dice_score, recall, precision, iou
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, MaxPool2D, Add, Dropout, Concatenate, Conv2DTranspose, Dense, Reshape, Flatten, Softmax, Lambda, UpSampling2D, AveragePooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, SeparableConv2D
import pandas as pd


def load_images_and_masks(image_dir, mask_dir, image_size=(512, 512)):
    images = []
    masks = []
    for img_filename in os.listdir(image_dir):
        img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, img_filename), target_size=image_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img)

        mask_filename = img_filename.replace('.png', '.png')  # or other corresponding mask file format
        mask = tf.keras.preprocessing.image.load_img(os.path.join(mask_dir, mask_filename), color_mode="grayscale", target_size=image_size)
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0
    return images, masks


# Load images and masks

image_dir = 'path/to/your/test/datasets/folder'
mask_dir =  'path/to/your/test/datasets/folder'
images, masks = load_images_and_masks(image_dir, mask_dir)



# Read text labels
text = pd.read_csv(r'path/to/your/csv file/folder', header=None)
text = tf.convert_to_tensor(np.asarray(text), dtype=tf.float32)
text = Dense(32, activation='relu')(text)
text = Dense(32, activation='relu')(tf.transpose(text, perm=[1,0]))
text = tf.expand_dims(text, axis=0)
text = tf.expand_dims(text, axis=-1)
# 重新构建模型
model = build_multiresunet((512, 512, 3), text)

weights_path = 'model_checkpoints/best_model.h5'
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"无法找到权重文件：{weights_path}")


# 加载训练好的权重
model.load_weights('model_checkpoints/best_model.h5')

# 这里需要重新编译模型 ，用与训练一致的配置
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00051)  # 或者根据训练时的学习率调整
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',  # 根据训练时指定的损失函数，如果不同需特别注意！
    metrics=['accuracy', dice_score, recall, precision, iou]  # 再次指定训练时一致的评估指标
)


# 评估模型
evaluation = model.evaluate(images, masks, batch_size=1)
print(f"Test Loss: {evaluation[0]}")
print(f"Test Accuracy: {evaluation[1]}")

# 进行预测
predictions = model.predict(images, batch_size=2)

# 可视化一些预测结果
num_images_to_display = 3
fig, axes = plt.subplots(num_images_to_display, 3, figsize=(15, num_images_to_display * 5))
for i in range(num_images_to_display):
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title("Original Image")
    axes[i, 1].imshow(masks[i].squeeze(), cmap='gray')
    axes[i, 1].set_title("True Mask")
    axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
    axes[i, 2].set_title("Predicted Mask")
plt.show()


# 预测并保存结果
save_dir = "test_results"  # 保存结果的目录
os.makedirs(save_dir, exist_ok=True)

for i in range(len(images)):
    prediction = model.predict(np.expand_dims(images[i], axis=0))[0] # 预测单张图片

    # 创建一个新的图形，并将三个图像（原始图像、真实掩码和预测掩码）绘制在一起
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(images[i])
    axes[0].set_title("Original Image")

    axes[1].imshow(masks[i].squeeze(), cmap='gray')
    axes[1].set_title("True Mask")

    axes[2].imshow(prediction.squeeze(), cmap='gray')
    axes[2].set_title("Predicted Mask")



    plt.tight_layout()  # 调整子图参数，使之填充整个图像区域

    # 保存图像
    filename = os.path.splitext(os.listdir(image_dir)[i])[0] + "_prediction.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath)
    plt.close(fig) # 关闭图形，释放内存

    print(f"Saved prediction for image {i+1} to {filepath}")

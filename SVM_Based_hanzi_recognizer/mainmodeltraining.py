import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog
import pickle

# 定义数据集路径
output_base_dir = 'C:\\Users\\lixee\\PycharmProjects\\SVM_Based_hanzi_recognizer\\hanzi\\'


# 映射文件夹名称与汉字的对应关系
char_map = {"char1": "你", "char2": "好", "char3": "东", "char4": "华"}


# 提取HOG特征
def extract_hog_features(image):
    hog_features = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)
    return hog_features


# 数据增强（例如旋转和翻转）
def augment_image(image):
    # 旋转15度
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))

    # 翻转图像
    flipped_img = cv2.flip(image, 1)  # 水平翻转

    return rotated_img, flipped_img


# 加载图像并进行预处理和增强
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            print(f"正在加载图像: {img_path}")  # 打印文件路径
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            if img is not None:
                img = cv2.resize(img, (28, 28))  # 调整为28x28
                img_hog = extract_hog_features(img)  # 提取HOG特征
                images.append(img_hog)
                labels.append(label)

                # 数据增强后再添加图像
                augmented_images = augment_image(img)
                for augmented_img in augmented_images:
                    augmented_img_hog = extract_hog_features(augmented_img)
                    images.append(augmented_img_hog)
                    labels.append(label)  # 同样标签
            else:
                print(f"无法读取图像: {img_path}")
    return images, labels


# 初始化数据集和标签
X = []
y = []

# 遍历每个映射后的文件夹并加载数据
for folder, char in char_map.items():
    folder_path = os.path.join(output_base_dir, folder)
    images, labels = load_images_from_folder(folder_path, char)
    X.extend(images)  # 添加图像数据
    y.extend(labels)  # 添加对应的标签

# 转换为 NumPy 数组
X = np.array(X)
y = np.array(y)

# 如果没有有效图像，结束程序
if len(X) == 0:
    print("没有有效的图像数据，程序结束。")
    exit()

# 将标签转换为数值编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 将汉字转换为数值标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 使用SVM进行训练（使用RBF核和自动类别权重）
svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)
svm.fit(X_train, y_train)

# 预测和评估模型
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")

# 保存模型和标签编码器
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm, model_file)

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

# 输出训练完成的提示
print("训练已完成！")

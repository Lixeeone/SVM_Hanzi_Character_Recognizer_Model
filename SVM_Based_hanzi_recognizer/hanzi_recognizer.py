import os
import cv2
import numpy as np
import pickle
from tkinter import Tk, Label, Text, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage.feature import hog

# 全局变量
svm = None
le = None

# HOG特征提取函数
def extract_hog_features(image):
    hog_features = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=False)
    return hog_features

# 图像预处理函数
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None
    img = cv2.resize(img, (28, 28))  # 缩放到28x28大小
    img_hog = extract_hog_features(img)  # 提取HOG特征
    return img_hog

# 预测函数
def predict_image(image_path):
    img_hog = preprocess_image(image_path)
    if img_hog is not None:
        img_hog = img_hog.reshape(1, -1)  # 调整形状以适应SVM模型输入
        predicted_label = svm.predict(img_hog)  # 预测标签
        probabilities = svm.predict_proba(img_hog)[0]  # 获取概率
        predicted_char = le.inverse_transform(predicted_label)[0]  # 将数值标签转换为对应汉字
        max_prob = probabilities[np.argmax(probabilities)]  # 获取最高概率
        return predicted_char, max_prob
    return None, None

# 拖拽上传事件处理
def on_drop(event):
    file_path = event.data
    if file_path.lower().endswith('.png'):
        predicted_char, prob = predict_image(file_path)
        if predicted_char:
            result_label.config(text=f"预测的汉字: {predicted_char}")
            prob_label.config(text=f"对应的概率: {prob:.2f}")
        else:
            messagebox.showerror("错误", "图像处理失败！")
    else:
        messagebox.showwarning("警告", "请拖拽PNG图像文件。")

# 创建拖拽上传界面
def create_drag_and_drop_interface():
    root = TkinterDnD.Tk()
    root.title("汉字识别")
    root.geometry("500x400")
    root.config(bg="#f0f0f0")

    # 设置标题标签
    title_label = Label(root, text="SVM汉字识别", font=("Arial", 20), bg="#f0f0f0")
    title_label.pack(pady=10)

    # 拖拽区域
    drop_area = Text(root, wrap='word', height=10, width=50, bg="#ffffff", fg="#333333", font=("Arial", 12))
    drop_area.pack(pady=20, padx=20)
    drop_area.insert("end", "                                   请拖拽PNG图像到这里")
    drop_area.config(state="disabled")  # 禁用编辑

    # 结果标签
    global result_label, prob_label
    result_label = Label(root, text="预测的汉字: ", font=("Arial", 16), bg="#f0f0f0")
    result_label.pack(pady=10)

    prob_label = Label(root, text="对应的概率: ", font=("Arial", 16), bg="#f0f0f0")
    prob_label.pack(pady=10)

    # 绑定拖拽事件
    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind('<<Drop>>', on_drop)

    root.mainloop()

# 主程序入口
if __name__ == "__main__":
    # 加载训练好的模型和标签编码器
    try:
        with open('svm_model.pkl', 'rb') as model_file:
            svm = pickle.load(model_file)  # 加载已经训练好的SVM模型

        with open('label_encoder.pkl', 'rb') as le_file:
            le = pickle.load(le_file)  # 加载标签编码器

        # 创建拖拽上传界面
        create_drag_and_drop_interface()

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        messagebox.showerror("错误", "模型文件或标签编码器文件不存在，请检查路径。")

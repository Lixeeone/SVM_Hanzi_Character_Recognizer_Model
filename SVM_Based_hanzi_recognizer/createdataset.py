import os
import random
from PIL import Image, ImageDraw, ImageFont

# 定义汉字和样本数量
characters = ["你", "好", "东", "华"]
char_map = {"你": "char1", "好": "char2", "东": "char3", "华": "char4"}  # 映射关系
samples_per_character = 200
output_base_dir = 'C:\\Users\\lixee\\PycharmProjects\\SVM_Based_hanzi_recognizer\\hanzi\\' # 输出目录基础路径

# 定义字体路径列表（增加华文行楷的权重最大）
font_paths = [
    "C:\\Windows\\Fonts\\msyh.ttc",  # 微软雅黑
    "C:\\Windows\\Fonts\\Deng.ttf",  # 等线
    "C:\\Windows\\Fonts\\STCAIYUN.TTF",  # 华文彩云
    "C:\\Windows\\Fonts\\STHUPO.TTF",  # 华文琥珀
    "C:\\Windows\\Fonts\\STLITI.TTF",  # 华文隶书
    "C:\\Windows\\Fonts\\STXINWEI.TTF",  # 华文新魏
    "C:\\Windows\\Fonts\\simsun.ttc",  # 宋体
    "C:\\Windows\\Fonts\\simyou.ttf"  # 幼圆
]

# 增加华文行楷和华文琥珀的权重，华文行楷权重最大
weighted_fonts = font_paths + ["C:\\Windows\\Fonts\\STXINGKA.TTF"] * 10 + ["C:\\Windows\\Fonts\\STHUPO.TTF"] * 3

# 为每个汉字创建映射后命名的输出文件夹
for char in characters:
    char_dir = os.path.join(output_base_dir, char_map[char])  # 使用映射后的名称创建文件夹
    os.makedirs(char_dir, exist_ok=True)  # 创建对应文件夹

    # 生成图像
    for i in range(1, samples_per_character + 1):
        # 创建一个白色背景的图像
        img = Image.new('RGB', (100, 100), color='white')
        d = ImageDraw.Draw(img)

        # 随机选择一个字体，增加权重后的选择
        font_path = random.choice(weighted_fonts)
        font_size = 72  # 固定字体大小

        try:
            # 尝试加载字体
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            # 如果加载失败，输出出错的字体路径并跳过该图像生成
            print(f"无法加载字体：{font_path}")
            continue

        # 计算文本位置，居中显示
        bbox = d.textbbox((0, 0), char, font=font)  # 获取文本的边界框
        text_width = bbox[2] - bbox[0]  # 计算宽度
        text_height = bbox[3] - bbox[1]  # 计算高度
        position = ((100 - text_width) // 2, (100 - text_height) // 2)

        # 在图像上绘制汉字
        d.text(position, char, fill=(0, 0, 0), font=font)

        # 添加适度的随机噪声
        for _ in range(random.randint(0, 10)):  # 随机噪声数量，确保不会影响可读性
            x = random.randint(0, 100)
            y = random.randint(0, 100)
            d.point((x, y), fill=(0, 0, 0))

        # 随机轻微旋转图像，保持在 -5 到 5 度之间
        if random.random() < 0.5:  # 50% 的概率进行旋转
            angle = random.randint(-5, 5)
            img = img.rotate(angle, expand=1)

        # 保存图像到映射后的文件夹
        img.save(os.path.join(char_dir, f"{char_map[char]}_{i}.png"))

# 输出生成的文件夹路径
print(output_base_dir)
print("创建数据集完毕！")

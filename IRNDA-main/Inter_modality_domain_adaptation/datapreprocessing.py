import pandas as pd
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision.transforms import transforms
from matplotlib import pyplot as plt




class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = "", p: float = 0.5):
        self.hairs = hairs
        self.hairs_folder = hairs_folder
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)

        if not n_hairs:
            return img

        if random.random() < self.p:
            # 将 PIL Image 转换为 numpy.ndarray
            img = np.array(img)

            height, width, _ = img.shape  # target image width and height
            hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]

            for _ in range(n_hairs):
                hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
                hair = cv2.flip(hair, random.choice([-1, 0, 1]))
                hair = cv2.rotate(hair, random.choice([0, 1, 2]))

                h_height, h_width, _ = hair.shape  # hair image width and height
                roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
                roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
                roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

                # Creating a mask and inverse mask
                img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # Now black-out the area of hair in ROI
                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                # Take only region of hair from hair image.
                hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

                # Put hair in ROI and modify the target image
                dst = cv2.add(img_bg, hair_fg)

                img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

            # 将 numpy.ndarray 转换回 PIL Image
            img = Image.fromarray(img)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}",p={self.p})'





class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        初始化数据集
        :param annotations_file: 包含图像路径和标签的CSV文件
        :param img_dir: 图像存储的目录
        :param transform: 对图像进行的转换操作（如数据增强）
        """
        # 读取CSV文件
        self.img_labels = pd.read_csv(annotations_file, low_memory=False)

        # 确保target列存在
        if 'class' not in self.img_labels.columns:
            raise ValueError("CSV文件中必须包含'class'列")

        # 存储图像目录和转换操作
        self.img_dir = img_dir
        self.transform = transform

        # 获取标签的类别数
        self.num_classes = self.img_labels['class'].nunique()

        self.targets = self.img_labels['class'].tolist()

    def __len__(self):
        """返回数据集的大小"""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        获取给定索引的数据

        :param idx: 数据集的索引
        :return: 图像和标签
        """
        # 获取图像路径
        img_name = self.img_labels.iloc[idx, 0] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)

        # 检查文件是否存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到：{img_path}")

        # 打开图像并转换为RGB模式
        image = Image.open(img_path).convert("RGB")


        # 获取标签（注意使用iloc来按位置索引）
        label = self.img_labels.iloc[idx]["class"]

        # 如果提供了transform，应用转换操作
        if self.transform:
            image = self.transform(image)

        return image, label















class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            # 将 PIL Image 转换为 numpy.ndarray
            img = np.array(img)
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8),  # image placeholder
                                (img.shape[0] // 2, img.shape[1] // 2),  # center point of circle
                                random.randint(img.shape[0] // 2 - 3, img.shape[0] // 2 + 15),  # radius
                                (0, 0, 0),  # color
                                -1)

            mask = circle - 255
            img = np.multiply(img, mask)
            # 将 numpy.ndarray 转换回 PIL Image
            img = Image.fromarray(img)

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'








class PAD_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        初始化数据集
        :param annotations_file: 包含图像文件名和诊断类别的CSV文件
        :param img_dir: 图像存储的目录
        :param transform: 对图像进行的转换操作（如数据增强）
        """
        self.img_labels = pd.read_csv(annotations_file, low_memory=False)
        self.img_dir = img_dir
        self.transform = transform
        # 定义类别到标签的映射
        self.class_to_label = {
            'NEV': 0,  # 这里根据数据里的实际值是NEV，若原需求是NV需调整
            'MEL': 1,
            'BCC': 2,
        }
        self.targets = self.img_labels['diagnostic'].map(self.class_to_label).tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx]['img_id']
        img_path = os.path.join(self.img_dir, img_name)


        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到：{img_path}")


        image = Image.open(img_path).convert('RGB')

        diagnostic_class = self.img_labels.iloc[idx]['diagnostic']
        label = self.class_to_label[diagnostic_class]

        if self.transform:
            image = self.transform(image)

        return image, label




class HAM10000_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        初始化数据集
        :param annotations_file: 包含图像文件名和诊断类别的CSV文件
        :param img_dir: 图像存储的目录
        :param transform: 对图像进行的转换操作（如数据增强）
        """
        self.img_labels = pd.read_csv(annotations_file, low_memory=False)
        self.img_dir = img_dir
        self.transform = transform
        # 定义类别到标签的映射
        self.class_to_label = {
            'nv': 0,
            'mel': 1,
            'bcc': 2,
        }
        self.targets = self.img_labels['dx'].map(self.class_to_label).tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)


        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到：{img_path}")


        image = Image.open(img_path).convert('RGB')

        diagnostic_class = self.img_labels.iloc[idx]['dx']
        label = self.class_to_label[diagnostic_class]

        if self.transform:
            image = self.transform(image)

        return image, label

class D_c_4(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        初始化数据集
        :param annotations_file: 包含图像文件名和诊断类别的CSV文件
        :param img_dir: 图像存储的目录
        :param transform: 对图像进行的转换操作（如数据增强）
        """
        self.img_labels = pd.read_csv(annotations_file, low_memory=False)
        self.img_dir = img_dir
        self.transform = transform
        # 定义类别到标签的映射
        self.class_to_label = {
            'nevus': 0,
            'mel': 1,
            'sek': 2,
            'bcc': 3
        }
        self.targets = self.img_labels['diagnosis'].map(self.class_to_label).tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)


        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到：{img_path}")


        image = Image.open(img_path).convert('RGB')

        diagnostic_class = self.img_labels.iloc[idx]['diagnosis']
        label = self.class_to_label[diagnostic_class]

        if self.transform:
            image = self.transform(image)

        return image, label










class H_4(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        初始化数据集
        :param annotations_file: 包含图像文件名和诊断类别的CSV文件
        :param img_dir: 图像存储的目录
        :param transform: 对图像进行的转换操作（如数据增强）
        """
        self.img_labels = pd.read_csv(annotations_file, low_memory=False)
        self.img_dir = img_dir
        self.transform = transform
        # 定义类别到标签的映射
        self.class_to_label = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,

        }
        self.targets = self.img_labels['dx'].map(self.class_to_label).tolist()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)


        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件未找到：{img_path}")


        image = Image.open(img_path).convert('RGB')

        diagnostic_class = self.img_labels.iloc[idx]['dx']
        label = self.class_to_label[diagnostic_class]

        if self.transform:
            image = self.transform(image)

        return image, label
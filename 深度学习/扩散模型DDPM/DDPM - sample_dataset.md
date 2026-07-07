---
title: 数据集类、sample_dataset.py
tags:
  - DDPM
  - Dataset
related:
---
# 概念
==功能一句话解释：**把硬盘上的 SAR 图片变成 PyTorch 模型能吃的训练数据。**==

训练 DDPM 时，你需要不断喂给它：
- 一张 SAR 图像（128×128 灰度图）
- 它的类别标签（"这是一辆坦克"还是"一辆装甲车"）
- 它的方位角（"这张图是从 60° 角度拍的"）
这个类就是负责**自动化地把硬盘上的图片读出来，打包好这些信息，交给模型**。

`SAMPLE_Dataset` 类本身是通用的，不管是 DDPM、SNGAN 还是其他生成模型，数据加载部分完全一样。

## 核心问题：数据在哪里？长什么样？
硬盘上的目录结构是这样的：
```
dataset/SAMPLE/png_images/qpm/real/
├── 2s1/          ← 全是 2S1 自行火炮的 SAR 图像
│   ├── 2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png
│   ├── 2s1_real_A_elevDeg_016_azCenter_090_50_serial_b02.png
│   └── ...
├── bmp2/         ← 全是 BMP2 步兵战车的 SAR 图像
├── btr70/
├── m1/
├── m2/
├── m35/
├── m60/
├── m548/
├── t72/
└── zsu23/
```

两个关键信息隐藏在**文件名**和**目录名**里：
```
2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png
                                    ^^^^^ ^^
                                    60.22° ← 这张图是从 60.22° 方位角拍摄的
↑
'2s1' 目录 → 类别标签 = 0
```
这个文件做的事情就是**自动提取这些信息，而不是让用户手动标注**。

## 具体实现：
### 1. `SAMPLE_Dataset` 类 — 把数据变成标准接口

它继承 PyTorch 的 `Dataset`，实现了三个必须的方法：
```
DataLoader 调用流程：
                      
  DataLoader 问: "你有几个样本？"
         │
         ▼
  __len__()  →  return 2747    ← 返回总数
         │
         │  DataLoader 循环: for i in range(2747)
         ▼
  __getitem__(i)  →  返回第 i 个样本的字典:
                      {
                        'image': Tensor(1,128,128),  ← 模型输入
                        'label': 3,                   ← 类别标签
                        'az': 60.22,                  ← 方位角度数
                        'name': '2s1_real_...'        ← 文件名
                      }
```

**`__init__` 做了什么？** 扫描数据源，生成一个列表 `[[路径1, 标签1], [路径2, 标签2], ...]`。这是唯一一次访问硬盘，之后 `__getitem__` 按索引读就行。

**`__getitem__` 做了什么？** 从列表取出一条 → 打开图片 → 灰度化 → 归一化到 [0,1] → 转成 tensor → 从文件名提取方位角 → 打包返回。

**为什么要有两种加载方式（txt_file vs data_root）？** 因为`data_root` 扫描目录很方便但慢（每次启动要遍历所有文件），`txt_file` 需要提前准备但快（直接读文件列表）。实际使用流程是：先用 `data_root` 生成 txt，之后训练都用 txt。

### 2. `create_sample_txt_file` 函数 — 生成 txt 文件
```python
data_root = "dataset/SAMPLE/png_images/qpm/real"

create_sample_txt_file(data_root, "data/sample_train.txt")
```
运行一次，生成 `sample_train.txt`：
```
real/2s1/2s1_real_A_...azCenter_060_22_...png 0
real/bmp2/bmp2_real_A_...azCenter_035_49_...png 1
real/btr70/btr70_real_A_...azCenter_120_10_...png 2
...
```
之后训练脚本直接用 `txt_file="data/sample_train.txt"` 加载，不再扫描目录。





# 代码1
（代码 1 来源：EM_deeplearning_beifen\DDPM\ddpm_model.py）
## 初始化函数
`__init__` 做的事情就是：**统一数据加载入口 → 构建一个 `[路径, 标签]` 列表**。
后续 `__getitem__` 只需按索引从这个列表取数据，不用关心数据来源是 txt 还是目录扫描。

```python
class SAMPLE_Dataset(Dataset):
    """
    SAMPLE dataset for DDPM training
    """

    def __init__(self, data_root=None, txt_file=None, transform=None):
        """
        参数：
			data_root：数据集根目录路径（如果使用 txt_file，则可选）
			txt_file：txt 文件路径($path$ $label$)（如果使用 data_root，则可选）
			transform：要应用于样本的可选转换
        """
```

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| `data_root` | `str` | `None` | 数据集根目录路径，内含按类别分好的子文件夹 |
| `txt_file` | `str` | `None` | 预先生成的 txt 文件路径，每行记录图片路径和标签 |
| `transform` | `callable` | `None` | 图像预处理函数（如 `transforms.ToTensor()`） |

```python
		self.transform = transform
```
把数据增强/预处理函数存为实例属性，供后面 `__getitem__` 使用。如果传了 `None`，`__getitem__` 里会走默认逻辑（手动转 tensor）。

```python
		self.classes = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
```
这是 MSTAR SAR 数据集的标准 10 类军事目标。列表的**索引天然对应标签编号**：
'2s1'→0, 'bmp2'→1, ..., 'zsu23'→9

```python
		self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
```
字典推导式，把上面的列表反过来映射。等价于：{
    '2s1': 0,
    'bmp2': 1,
    'btr70': 2,
    'm1': 3,
    'm2': 4,
    'm35': 5,
    'm60': 6,
    'm548': 7,
    't72': 8,
    'zsu23': 9
}
**为什么需要这个？** `scan_dataset_directory` 遍历子目录时，目录名是字符串（如 `"bmp2"`），需要快速查到对应的数字标签 `1`。字典查找是 O(1)。


```python
		if txt_file:                                          # 分支 A
		    self.sample_path_label = self.read_dataset_txt(txt_file)
		elif data_root:                                       # 分支 B
		    self.sample_path_label = self.scan_dataset_directory(data_root)
		else:                                                 # 兜底
		    raise ValueError("Either data_root or txt_file must be provided")
```
两个分支最终都产生同一种数据结构 `self.sample_path_label`，它是一个列表：
```
[
    ["real/2s1/xxx.png", 0],    # [图片路径(str), 标签(int)]
    ["real/bmp2/yyy.png", 1],
    ...
]
```

|          | 分支 A：`txt_file`      | 分支 B：`data_root`           |
| -------- | -------------------- | -------------------------- |
| **触发条件** | 传了 txt 文件路径          | 传了根目录路径                    |
| **调用方法** | `read_dataset_txt()` | `scan_dataset_directory()` |
| **标签来源** | txt 文件里直接写好的数字       | 从子目录名 + `class_to_idx` 推断  |
| **优点**   | 速度快，数据划分明确           | 无需预处理，即开即用                 |
如果有 txt 又有 data_root，优先走 txt（`if-elif` 结构决定的）。
两个都不传就抛异常，因为没有任何数据来源。



## 从文本文件读取
```python
    def read_dataset_txt(self, txt_file):
        """Read dataset from txt file"""
        list_path_label = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    file_path = parts[0]
                    label = int(parts[1])
                    list_path_label.append([file_path, label])
        return list_path_label
```
**文本文件格式**：
```
real/2s1/2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png 0
real/bmp2/bmp2_real_A_elevDeg_016_azCenter_035_49_serial_9563.png 1
...
```
每行：`图片路径 类别编号`

## 从目录扫描
```python
def scan_dataset_directory(self, data_root):
        """Scan dataset directory and extract all images with labels and azimuth angles"""
        list_path_label_az = []
        for class_name in self.classes:
            class_dir = os.path.join(data_root, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(class_dir, filename)
                    # Extract label
                    label = self.class_to_idx[class_name]
                    list_path_label_az.append([file_path, label])
        return list_path_label_az
```
当用户传入 `data_root` 而非 `txt_file` 时调用。**从目录结构推断标签**，无需预先准备 txt 文件。
预期的目录结构：
```
data_root/
├── 2s1/        → label = 0
│   ├── 2s1_real_A_..._azCenter_060_22_...png
│   └── ...
├── bmp2/       → label = 1
├── btr70/      → label = 2
├── m1/         → label = 3
├── m2/         → label = 4
├── m35/        → label = 5
├── m60/        → label = 6
├── m548/       → label = 7
├── t72/        → label = 8
└── zsu23/      → label = 9
```
扫描过程：遍历 `self.classes` 中的每个类别名 → 进入对应子目录 → 收集所有图片文件 → 标签由目录名经 `class_to_idx` 映射得到。

方式对比**：

|方式|优点|缺点|
|---|---|---|
|`txt_file`|加载快，可灵活划分训练/测试集|需预先生成 txt 文件|
|`data_root`|即开即用，无需预处理|不能控制数据划分|


## 方位角提取
```python
    def extract_azimuth_from_filename(self, filename):
        """
        从文件名中提取方位角
		格式：...azCenter_XXX_XX... 其中 XXX_XX 代表度数（例如，010_22 = 10.22 度）
        """

		# 匹配 azCenter_XXX_XX 或 azCenter_XXXX_XX 等的模式。
        pattern = r'azCenter_(\d+)_(\d+)'
        match = re.search(pattern, filename)
        if match:
            degrees_part = match.group(1)  # e.g., '010'
            decimal_part = match.group(2)  # e.g., '22'
            # Convert to float: degrees_part + decimal_part/100
            azimuth = float(degrees_part) + float(decimal_part) / 100.0
            return azimuth
        else:
            # If no azimuth found, return random azimuth between 0-360
            print(f"Warning: No azimuth found in filename {filename}, using random azimuth")
            return float(np.random.uniform(0, 360))
```
**文件名解析示例**：
```
2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png
                                ^^^^^ ^^
                            degrees=60  decimal=22
                            → azimuth = 60.22°
```

- 用正则从文件名提取方位角（度+百分位）
- 如果提取失败，返回随机角度作为降级方案


## 数据集大小
```python
    def __len__(self): 
        return len(self.sample_path_label)
```
返回数据集样本总数。`DataLoader` 通过它知道要迭代多少次。


## 获取单个样本
```python
	def __getitem__(self, idx):
	        """
	        参数:
	            idx: 数据索引
	        Returns:
	            sample: 包含 'image'、'name'、'label' 和 'az' 的字典
	        """
	
	        file_path = self.sample_path_label[idx][0]
	        label = self.sample_path_label[idx][1]
	        
	        # 提取不带扩展名的文件名
	        filename = os.path.basename(file_path)
	        name = os.path.splitext(filename)[0]
	        
	        # 从文件名中提取方位角
	        az = self.extract_azimuth_from_filename(filename)
	        
	        # 加载和处理图像
	        try:
	            # Load PNG image
	            image = Image.open(file_path).convert('L')  # 转换为灰度图
	            # 转换为 numpy 数组并归一化到 [0, 1] 区间
	            img_array = np.array(image, dtype=np.float32) / 255.0
	            # 如果指定，则应用transform
	            if self.transform:
	                img_array = self.transform(img_array)
	            else:
	                # 默认值：转换为张量
	                img_array = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
	                
	        except Exception as e:
	            print(f"Error loading image {file_path}: {e}")
	            # 如果出错，则返回虚拟数据。
	            img_array = torch.zeros((1, 128, 128))  # 根据需要调整尺寸
	            az = 0.0
	        
	        sample = {
	            'image': img_array,
	            'name': name,
	            'label': label,
	            'az': az
	        }
	        
	        return sample
```
**步骤**：
1. 从 `sample_path_label` 获取文件路径和标签
2. 用 `PIL.Image.open(file_path).convert('L')` 加载为灰度图
3. 转换为 numpy 数组并归一化到 [0, 1]
4. 应用 transform（如 `ToTensor()`）
5. 从文件名提取方位角
6. 返回字典 `{'image', 'name', 'label', 'az'}`
```python
sample = {
    'image': img_array,    # Tensor, (1, 128, 128), 值域 [0,1]
    'name': name,          # 文件名（不含扩展名）
    'label': label,        # int, 0~9
    'az': az               # float, 方位角度数
}
```




## 生成数据集 txt 文件
```python
# Utility function to create dataset txt file 创建数据集txt文件的实用函数
def create_sample_txt_file(data_root, output_txt):
    """
    Create a txt file for SAMPLE dataset in the same format as MSTAR
    Args:
        data_root: root directory of SAMPLE dataset
        output_txt: output txt file path
    """

    dataset = SAMPLE_Dataset(data_root=data_root)
    with open(output_txt, 'w') as f:
        for item in dataset.sample_path_label:
            file_path = item[0]
            label = item[1]
            f.write(f"{file_path} {label}\n")
    print(f"Created txt file: {output_txt} with {len(dataset)} samples")

```
**作用**：扫描 `data_root` 目录，将所有图片路径和标签写入一个 txt 文件，供后续用 `txt_file` 方式快速加载。

**使用场景**：首次获得数据集时，目录结构是现成的但还没有 txt 文件。运行这个函数一次性生成 txt，之后训练和推理都用 txt 加载，避免每次都扫描目录。

生成的 txt 格式与 `read_dataset_txt` 读取的格式一致：
```
real/2s1/2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01.png 0
real/bmp2/bmp2_real_A_elevDeg_016_azCenter_035_49_serial_9563.png 1
...
```


## 模块入口 — 独立运行示例
```python
if __name__ == '__main__':
    # Example usage
    data_root = r"D:\taojiawei\tjw\Deep_Learning\EM_deeplearning\dataset\SAMPLE\png_images\qpm\real"
    
    # Create train and test txt files
    create_sample_txt_file(data_root, '../data/sample_train.txt')
    # Test the dataset
    dataset = SAMPLE_Dataset(data_root=data_root)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Azimuth: {sample['az']}")
        print(f"Name: {sample['name']}")
```
当直接运行 `python sample_dataset.py` 时执行，功能是：
1. 调用 `create_sample_txt_file` 生成训练集 txt 文件
2. 加载数据集并打印第一个样本的各项属性作为**完整性检查**

**示例输出**：
```
Created txt file: ../data/sample_train.txt with 2747 samples
Dataset size: 2747
Sample keys: dict_keys(['image', 'name', 'label', 'az'])
Image shape: torch.Size([1, 128, 128])
Label: 0
Azimuth: 60.22
Name: 2s1_real_A_elevDeg_015_azCenter_060_22_serial_b01
```
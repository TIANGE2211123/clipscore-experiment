# 视频CLIP相似度评估工具

这是一个用于评估视频与文本描述之间CLIP相似度的Python工具集，包含两个主要组件：改进的视频CLIP评估器和快速测试脚本。

## 项目概述

本项目提供了完整的视频-文本相似度评估解决方案，支持：
- 视频帧提取和预处理
- CLIP模型相似度计算
- 时间一致性分析
- 批量评估和单场景测试
- 结果可视化和导出

## 文件结构

```
├── video_clip_evaluator.py    # 核心评估器类
├── quick_clip_test.py         # 快速测试脚本
├── requirements.txt # 依赖包列表
└── README.md                  # 本文档
```

## 核心组件

### 1. ImprovedVideoCLIPEvaluator (`video_clip_evaluator.py`)

主要的CLIP评估器类，提供以下功能：

#### 主要方法

- **`__init__(device, model_name)`**: 初始化评估器
- **`extract_frames_from_video(video_path, max_frames, frame_interval)`**: 从视频中提取帧
- **`calculate_clip_score(video_path, text_prompt, max_frames, frame_interval)`**: 计算CLIP相似度分数
- **`calculate_temporal_consistency(video_path, max_frames, frame_interval)`**: 计算时间一致性
- **`batch_calculate_clip_scores(video_text_pairs, max_frames, frame_interval)`**: 批量计算

#### 特性

- 支持多种视频格式 (MP4, AVI, MOV, MKV, WMV)
- 自动设备检测 (CUDA/CPU)
- 标准化的CLIP特征提取和相似度计算
- 时间一致性分析
- 错误处理和日志记录

### 2. QuickTester (`quick_clip_test.py`)

快速测试工具，提供交互式和命令行两种使用方式：

#### 主要功能

- **单场景测试**: 测试指定场景的所有视频类型
- **单视频测试**: 测试单个视频文件
- **场景列表**: 列出可用的测试场景
- **交互模式**: 用户友好的交互界面

#### 测试模式

1. **场景测试**: 评估一个场景的safe、near-crash、crash三种视频类型
2. **单视频测试**: 评估单个视频与文本描述的匹配度
3. **场景列表**: 查看可用的测试场景

## 安装和配置

### 环境要求

- Python 3.7+
- PyTorch
- OpenCV
- Transformers
- PIL/Pillow
- NumPy
- Matplotlib
- tqdm

### 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包列表

```
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.20.0
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.21.0
matplotlib>=3.3.0
tqdm>=4.60.0
```

## 使用方法

### 1. 命令行使用

#### 快速测试脚本

   ```bash
# 测试单个视频
python quick_clip_test.py video --video-path "test_data/001376/crash.mp4" --text-prompt "A car on a wet highway is unable to stop in time and T-bones a vehicle that merges directly into its path, causing a high-speed collision."
python quick_clip_test.py video --video-path "test_data/001376/safe.mp4" --text-prompt "A car drives cautiously on a wet highway, maintaining a safe following distance as a white vehicle merges smoothly into the lane ahead."
python quick_clip_test.py video --video-path "test_data/001376/near-crash.mp4" --text-prompt "A car is forced to brake suddenly on a wet highway as another vehicle swerves into its lane with little warning, narrowly avoiding a side-swipe."
   ```

#### 批量评估当前数据集

从项目根目录运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 code/video_clip_evaluator.py \
  --test_data_dir test_data \
  --description_file test_data/description.txt \
  --max_frames 73 \
  --frame_interval 1 \
  --output_dir output/metrics \
  --device cpu \
  --local_files_only
```

说明：

- 脚本会自动发现 `test_data/` 下的数字场景目录，不再要求手动维护 `1376` 或 `001376` 两种编号。
- 当本地已经缓存 Hugging Face CLIP 模型时，`--local_files_only` 会直接从本地快照加载，避免重复联网请求。
- 结果会同时输出到 `output/metrics/clip_evaluation_results.json` 和 `output/metrics/clip_evaluation_results.csv`。


### 2. 编程接口使用

#### 基本使用示例

```python
from video_clip_evaluator import ImprovedVideoCLIPEvaluator

# 初始化评估器
evaluator = ImprovedVideoCLIPEvaluator(
    device="cuda",  # 或 "cpu"
    model_name="openai/clip-vit-base-patch32"
)

# 计算CLIP分数
result = evaluator.calculate_clip_score(
    video_path="path/to/video.mp4",
    text_prompt="车辆在道路上正常行驶",
    max_frames=30,
    frame_interval=1
)

print(f"CLIP分数: {result['mean_score']:.4f}")
print(f"分数范围: [{result['min_score']:.4f}, {result['max_score']:.4f}]")
```

#### 批量评估示例

```python
# 准备视频-文本对
video_text_pairs = [
    ("video1.mp4", "安全驾驶场景"),
    ("video2.mp4", "接近碰撞场景"),
    ("video3.mp4", "碰撞场景")
]

# 批量计算
results = evaluator.batch_calculate_clip_scores(
    video_text_pairs,
    max_frames=30,
    frame_interval=1
)

# 处理结果
for result in results:
    print(f"视频: {result['video_path']}")
    print(f"CLIP分数: {result['mean_score']:.4f}")
```

### 3. 交互模式使用

运行交互模式：

```bash
python quick_clip_test.py
```

按照提示选择：
1. 测试单个场景
2. 测试单个视频文件
3. 列出可用场景
4. 退出

## 数据格式

### 场景描述文件格式

场景描述文件 (`description.txt`) 应包含以下格式：

```
1376
Safe: 车辆在道路上正常行驶，没有异常情况
Near-Crash: 车辆接近碰撞，但最终避免了事故
Crash: 车辆发生碰撞事故

1377
Safe: 车辆安全通过交叉路口
Near-Crash: 车辆在交叉路口险些发生碰撞
Crash: 车辆在交叉路口发生碰撞
```

### 视频文件组织

测试数据应按以下结构组织：

```
test_data/
├── description.txt
├── 1376/
│   ├── safe.mp4
│   ├── near-crash.mp4
│   └── crash.mp4
├── 1377/
│   ├── safe.mp4
│   ├── near-crash.mp4
│   └── crash.mp4
└── ...
```

## 输出结果

### CLIP分数结果

```python
{
    'mean_score': 0.3245,      # 平均CLIP分数
    'std_score': 0.0234,       # 分数标准差
    'max_score': 0.3891,       # 最高分数
    'min_score': 0.2678,       # 最低分数
    'frame_scores': [...],     # 逐帧分数列表
    'num_frames': 30,          # 处理帧数
    'video_path': '...',       # 视频路径
    'text_prompt': '...'       # 文本提示
}
```

### 时间一致性结果

```python
{
    'mean_temporal_score': 0.9123,  # 平均时间一致性
    'std_temporal_score': 0.0156,   # 时间一致性标准差
    'temporal_scores': [...],       # 相邻帧相似度列表
    'num_pairs': 29                 # 相邻帧对数
}
```

## 参数说明

### 主要参数

- **`max_frames`**: 每个视频最大处理帧数 (默认: 30)
- **`frame_interval`**: 帧间隔，每隔几帧取一帧 (默认: 1)
- **`device`**: 计算设备 ("cuda" 或 "cpu")
- **`model_name`**: CLIP模型名称 (默认: "openai/clip-vit-base-patch32")

### 支持的CLIP模型

- `openai/clip-vit-base-patch32` (默认)
- `openai/clip-vit-base-patch16`
- `openai/clip-vit-large-patch14`
- `openai/clip-vit-large-patch14-336`

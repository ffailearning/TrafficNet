# TrafficNet

TrafficNet 是一个专为交通场景设计的目标检测网络

为了便于部署与与当前主流方法的公平对比，本项目将自定义模块**额外集成进 Ultralytics 框架中**，在保留其原生训练与推理流程的基础上实现功能拓展。

# 📦 项目结构

本项目基于 Ultralytics YOLOv8 实现，**核心结构**定义在：

- `ultralytics/cfg/models/v8/yolov8-traffic.yaml`  
  → 网络结构定义文件，包含自定义模块的使用方式，可参考其配置分析模型结构和数据流转。

**核心模块**源码位于：

- `ultralytics/nn/MLFENet.py`
- `ultralytics/nn/TPFusion.py`
- `ultralytics/nn/AFusion.py`

# 🚀 快速开始

## 1. 安装依赖

```bash
git clone https://github.com/ffailearning/TrafficNet.git
cd TrafficNet
pip install -r requirements.txt
```

## 2. 准备数据集

将你的**数据集标注为 YOLO 格式**，并编辑 `train.py` 中的数据配置路径：

```python
# train.py
model.train(
    data='data.yaml',	# 修改为你的 data.yaml 路径
    ...
```

确保 `your_data.yaml` 配置如下所示：

```yaml
train: /path/to/train/images
val: /path/to/val/images
nc: 3  # 类别数
names: ['car', 'pedestrian', 'cyclist']  # 类别名称
```

## 3. 训练模型

```
python train.py
```

可根据资源情况自定义超参数（如 `--img`, `--batch`, `--epochs`, `--device` 等）。

# 📊 评估与推理

训练完成后，模型将自动保存在 `runs/train/exp*/weights/best.pt`，可用于评估与推理：

- 评估：

```bash
python val.py 
```

- 推理：

```python
python predict.py 
```

# 📁 文件结构概览

```
TrafficNet-YOLOv8/
├── ultralytics/
│   ├── nn/
│   │   ├── MLFENet.py       
│   │   ├── TPFusion.py      
│   │   └── AFusion.py       
│   └── cfg/models/v8/
│       └── yolov8-traffic.yaml  # 网络结构定义
├── train.py
├── val.py
├── predict.py
└── README.md
```

# 📌 注意事项

* 模型结构已经通过 yaml 配置无缝嵌入原 YOLOv8 框架，确保兼容原始训练和部署流程。
* 如果你在使用时遇到任何问题，欢迎提 issue 或 fork 后贡献代码。

---

# ✨ 致谢

本项目基于 Ultralytics YOLOv8 构建，感谢其优秀的开源贡献：[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

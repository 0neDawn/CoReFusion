# CoReFusion

论文题目：**CoReFusion: One-Step Infrared and Visible Image Fusion via Decoupled Reconstruction**。

本发布包包含最终选定的 D1 网络、推理权重、MSRS 训练配置，以及训练和测试所需的少量配对示例。模型通过一次潜在状态更新预测粗尺度内容，再结合两级源图细节选择和有界残差补偿完成重建。其他实验变体及服务器专用脚本不包含在本包中。

## 环境安装

整理代码时的源环境为 Python 3.9.13、PyTorch 2.6.0 + CUDA 12.4、NumPy 1.26.4、OpenCV 4.9.0。训练使用 `torch.func.jvp`。建议创建独立环境，并在本项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

PyTorch 的 CUDA 版本应与本机环境匹配，安装方法可查阅 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)。训练和全分辨率推理建议使用 CUDA GPU。

## 使用已提供的权重测试

所选 D1 权重保存为 `checkpoints/latest.pth`，其来源和用途见 [checkpoints/README.md](checkpoints/README.md)。依次测试三个数据集的示例：

```bash
python test.py
python test.py --ir data/M3FD/test/ir --vis data/M3FD/test/vis --output results/M3FD
python test.py --ir data/FMB/test/ir --vis data/FMB/test/vis --output results/FMB
```

默认命令测试 MSRS。输出目录中的 `fused/` 保存彩色融合图，`fused_y/` 保存融合亮度图，`manifest.json` 记录逐图设置。彩色图由融合亮度和可见光色度组合得到。

输入图像须配准、同尺寸，且红外与可见光文件的主文件名一一对应。输出保持原始空间尺寸，不对输入图像做整体缩放；上下文缩略图单独缩放，网络所需的临时边界填充在输出时裁去。噪声由 `--seed` 和图像主文件名共同决定，改变文件名会改变噪声。已有输出不会默认覆盖，需要指定新目录或添加 `--overwrite`。

可使用 `--device cpu` 运行 CPU 推理，或减小 `--patch-batch` 降低显存需求。单步指一次潜在状态更新，仍包含编码、重叠块处理和重建。输出记录中的耗时包含读写图像，不能作为论文中的纯推理延迟。

## 在 MSRS 上训练

训练示例包含五对图像，固定划分为四对训练、一对验证。它们用于验证流程，不能复现完整数据集上的论文指标。先运行两步：

```bash
python train.py --stop-after 2 --batch-size 1 --output runs/smoke
```

再从相同运行目录续训两步：

```bash
python train.py --resume --stop-after 2 --batch-size 1 --output runs/smoke
```

`--stop-after` 保留原定 50,000 步的学习率调度。完整训练需要自行准备 MSRS 全量训练数据：

```bash
python train.py --train-ir /path/to/MSRS/train/ir --train-vis /path/to/MSRS/train/vis --output runs/CoReFusion_MSRS --save-every 10000
```

默认配置 `configs/corefusion_msrs.json` 使用 50,000 步、批量大小 4、256 × 256 配对随机裁剪、同步水平翻转和 128 × 128 上下文缩略图。优化器为 AdamW，权重衰减为 0.0001，基础学习率为 0.0001，预热 500 步后按余弦衰减至 0.00001。强度、梯度和结构损失权重分别为 2、4、1，其余重建与 shortcut 训练项见配置文件。训练不需要真实融合图像作为监督。

训练目录保存 `latest.pth`、`best.pth`、定期检查点、配置和日志。`best.pth` 按留出样本上的源图融合目标选择，不使用测试集指标。续训时保留原命令的配置、数据、代码和输出目录，增加 `--resume` 即可。训练产生的检查点含优化器及随机数状态；发布包自带的推理权重不含这些状态，不能直接用于断点续训。`--steps` 会改变训练总步数和学习率调度，临时暂停应使用 `--stop-after`。

## 可选指标计算

评测入口保留 MI、SCD、VIF、Qabf、SSIM 的原有计算实现：

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py --ir data/MSRS/test/ir --vis data/MSRS/test/vis --fused results/MSRS/fused --output results/MSRS/metrics.json
```

其他数据集替换对应目录即可。五对示例的均值不能与完整评测集的论文结果直接比较；不同方法之间应保持相同的图像形式及评测代码。

代码目录说明见 [英文 README](README.md)，示例数据说明见 [data/README.md](data/README.md)，代码与数据的归属说明见 [NOTICE.md](NOTICE.md)。

# CoReFusion

单步红外与可见光图像融合。

[English](README.md)

## 发布进度

- [x] D1 模型与推理代码
- [x] MI、SCD、VIF、Qabf、SSIM 指标评估代码
- [x] MSRS、M3FD、FMB 配对示例图像
- [ ] 训练脚本 `train.py`：即将发布
- [ ] 预训练权重：即将发布

## 安装

使用 Python 3.9，在仓库根目录执行。依赖固定为 PyTorch 2.6.0、torchvision 0.21.0 等版本。

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-eval.txt
```

## 推理与评估

仓库暂不包含权重。推理前请将已有权重放到 `checkpoints/latest.pth`。

```bash
python test.py --device cpu
python evaluate.py \
  --ir data/MSRS/test/ir \
  --vis data/MSRS/test/vis \
  --fused results/MSRS/fused \
  --output results/MSRS/metrics.json
```

使用 CUDA GPU 时，将 `--device cpu` 改为 `--device cuda:0`。推理输出的彩色融合图保存在 `fused/`，亮度图保存在 `fused_y/`；上述评估使用彩色融合图。输入图像须已配准、同尺寸，且主文件名一一对应。

若已有融合图，可直接运行 `evaluate.py`，无需权重。替换输入和输出路径即可评估 M3FD、FMB 或自己的数据。

已用本地权重在 CPU 上跑通全部 15 对测试示例及五项指标。该验证仅覆盖示例流程，不代表完整数据集上的论文结果复现。

详见[示例数据](data/README.md)、[指标约定](corefusion/metrics/README.md)和[归属说明](NOTICE.md)。

# ArSSR 运行环境与脚本 (ENV.md)

ArSSR：3D 任意倍率（浮点）隐式神经网络超分。默认工具链见上级 `../CLAUDE.md` §5。

## 关键事实（摸代码所得）
- **预训练权重已在仓库**：`pre_trained_models/ArSSR_{RDN,ResCNN,SRResnet}.pkl`（脑 HCP），无需下载。
- `test.py`：`-input_path` 是**目录**（遍历里面的 LR nii），`-scale` 浮点，输出各向同性 ×scale。
- `train.py`：自监督——HR patch 为目标、data.py 内部下采样当 LR；需 `data/hr_train|hr_val/`。
- **ArSSR 做各向同性 SR（三轴同倍率）**；我们的是各向异性（只 z 需超分）。要 z-only 需给 test.py 加逐轴 scale（见下 TODO）。
- **训练需要"高分辨"腹部体**抠 40³ patch；我们的 our-data z 太薄（≤32）抠不出，印证需外部高分辨数据。

## 部署（H100 或任意 CUDA Linux 机，免 sudo）
```bash
cd ArSSR
bash scripts/deploy.sh
# 默认建 ~/.venvs/arssr (py3.10) + torch cu132 + ArSSR 依赖。
# H100 驱动若 <CUDA13：CUDA_IDX=cu128 bash scripts/deploy.sh
```
实测（本地 4090）：torch 2.13.0+cu132、SimpleITK/scikit-image/tensorboard 齐全，`import model,data,...` OK。

## 训练 / 微调
```bash
# 1) 建 HR patch（HR_DIR 指向高分辨腹部 NIfTI 目录）
python scripts/build_data.py --hr_dir /path/to/hr_abdomen --out_dir data --patches_per_vol 6

# 2) 微调（默认从 RDN 脑预训练权重 fine-tune）
HR_DIR=/path/to/hr_abdomen EPOCH=1000 bash scripts/train.sh
#   FROM_SCRATCH=1  -> 从头训；ENCODER=ResCNN|SRResNet 可换编码器
# checkpoint 存 ./model/，tensorboard 在 ./log/（tensorboard --logdir log）
```
- `train.py` 已改：加 `-pre_trained_model`（微调）、修 batch_size view 的不满批 bug、val 空时不再除零、checkpoint 保存与 val 解耦、建 model/log 目录。

## 推理（任意浮点倍率）
```bash
LR_DIR=/path/to/lr SCALE=2.67 bash scripts/infer.sh
# MODEL=./model/model_param_1000.pkl 可换成你训的权重（默认用脑预训练）
# 输出 test/output/ArSSR_<enc>_recon_<scale>x_<name>.nii.gz
```

## 本地冒烟验证（4090，全通过）
- deploy.sh：环境 OK，cuda True。
- infer：32³→64³（2×）正常输出。
- train：4 个合成 128³ 体 → 16 patch → 从 RDN 微调 1 epoch → val loss 46，存 `model_param_1.pkl`。

## TODO / 已知局限
- **各向异性 SR**：ArSSR test.py 现为各向同性（三轴 ×scale）。我们要"T1map 只在 z 超分 2.67×"，需给 test.py 加逐轴 scale（`-scale_x -scale_y -scale_z`）——下一步可加。
- 训练数据：需高分辨腹部体；our-data 不够，需外部数据集（或 HCP 复现）。

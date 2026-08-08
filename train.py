import os
from pathlib import Path

from ultralytics import YOLO

# ====================== 核心配置 ======================
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据集不包含在仓库中。可通过环境变量覆盖默认相对路径。
DATA_YAML_PATH = Path(
    os.getenv("SA_YOLO_DATA", PROJECT_ROOT / "datasets" / "kaist_barcode" / "data.yaml")
).expanduser()
# SA-YOLO 自定义模型配置。
MODEL_YAML_PATH = Path(
    os.getenv(
        "SA_YOLO_MODEL_CONFIG",
        PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-c2fema-msscem.yaml",
    )
).expanduser()
# YOLO11原生预训练权重
# PRETRAINED_WEIGHTS = "yolo11s.pt"
# 输出项目与实验名称
PROJECT_NAME = str(PROJECT_ROOT / "runs" / "detect")
RUN_NAME = "sa_yolo"

"""
改yaml路径
改损失函数
改项目名称
改早停
"""

# ==============================================================================

def main():
    if not DATA_YAML_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML_PATH}\n"
            "Set SA_YOLO_DATA to the path of your dataset YAML file."
        )
    if not MODEL_YAML_PATH.is_file():
        raise FileNotFoundError(f"Model configuration not found: {MODEL_YAML_PATH}")

    model = YOLO(str(MODEL_YAML_PATH))  # 先加载自定义结构
    # model.load(PRETRAINED_WEIGHTS)  # 再加载预训练权重，会自动迁移匹配的层

    # 启动训练，超参数针对条形码任务优化
    print("开始训练...")
    results = model.train(
        # -------------------------- 核心基础配置 --------------------------
        # pretrained=PRETRAINED_WEIGHTS,
        data=DATA_YAML_PATH,
        epochs=200,
        batch=16,  # RTX4060 8G推荐16，显存充足可改为32
        imgsz=640,
        workers=0,  # Windows系统稳定优先设为0，Linux可设为4
        device=0,  # 调用RTX4060显卡


        # -------------------------- 优化器与学习率（解决loss震荡） --------------------------
        optimizer="SGD",  # 比SGD更稳定，适配小数据集
        lr0=0.001,  # 降低初始学习率，避免loss剧烈震荡
        lrf=0.01,
        warmup_epochs=5,  # 延长warmup，提升训练稳定性
        weight_decay=0.001,
        momentum=0.937,

        # -------------------------- 早停机制（避免无效训练与过拟合） --------------------------
        patience=20,  # 早停
        save=True,
        val=True,
        plots=True,
        verbose=True,
        exist_ok=False,
        seed=42,  # 固定随机种子，保证实验可复现
        deterministic=True,

        # -------------------------- 损失函数优化（提升框回归精度） --------------------------
        box=10.0,  # 提升框回归损失权重，拉高mAP50-95
        cls=0.2,  # 降低分类损失权重，单类别任务无需高权重
        dfl=1.5,  # 原生分布焦点损失权重

        # -------------------------- 条形码专属数据增强 --------------------------
        degrees=180.0,  # 360°全角度旋转，适配条码全方向特性
        shear=15.0,  # 剪切变换，适配倾斜畸变条码
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,  # 降低马赛克强度，避免条码被截断
        close_mosaic=0,  # 提前关闭马赛克，稳定训练
        mixup=0.5,
        copy_paste=0.0,

        # -------------------------- 输出配置 --------------------------
        project=PROJECT_NAME,
        name=RUN_NAME,
    )

    # 训练完成后，在官方测试集上做最终无偏评估
    print("\n" + "=" * 60)
    print("训练完成！正在官方测试集上进行最终评估...")
    test_metrics = model.val(
        data=DATA_YAML_PATH,
        split="test",
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        plots=True
    )

    # 打印最终性能指标
    print("\n【官方测试集最终性能指标】")
    print(f"Precision: {test_metrics.box.mp:.4f}")
    print(f"Recall:    {test_metrics.box.mr:.4f}")
    print(f"mAP50:     {test_metrics.box.map50:.4f}")
    print(f"mAP50-95:  {test_metrics.box.map:.4f}")
    print(f"模型参数量: {model.info()[0] / 1e6:.2f} M")
    print(f"计算量GFLOPs: {model.info()[1]:.2f}")
    print("=" * 60)
    print(f"训练日志与权重保存路径：{os.path.join(PROJECT_NAME, RUN_NAME)}")


if __name__ == "__main__":
    main()

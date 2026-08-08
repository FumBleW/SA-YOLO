import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent

# ====================== 核心配置（可通过环境变量覆盖） ======================
# 1. 训练好的模型权重路径（通常是 runs/detect/.../weights/best.pt）
MODEL_WEIGHTS_PATH = Path(
    os.getenv(
        "SA_YOLO_WEIGHTS",
        PROJECT_ROOT / "runs" / "detect" / "sa_yolo" / "weights" / "best.pt",
    )
).expanduser()
# 2. 数据集配置文件路径
DATA_YAML_PATH = Path(
    os.getenv("SA_YOLO_DATA", PROJECT_ROOT / "datasets" / "kaist_barcode" / "data.yaml")
).expanduser()
# 3. 评估结果保存路径
SAVE_DIR = Path(
    os.getenv("SA_YOLO_TEST_OUTPUT", PROJECT_ROOT / "outputs" / "test_set_evaluation_results")
).expanduser()
# 4. 可视化参数
VISUALIZE_SAMPLES = 50  # 可视化保存前50张检测结果图片
CONF_THRESHOLD = 0.25  # 可视化时的置信度阈值


# ==============================================================================

def main():
    print("=" * 80)
    print("开始测试集评估与可视化")
    print("=" * 80)

    if not MODEL_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_WEIGHTS_PATH}\n"
            "Set SA_YOLO_WEIGHTS to the path of your trained .pt file."
        )
    if not DATA_YAML_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML_PATH}\n"
            "Set SA_YOLO_DATA to the path of your dataset YAML file."
        )

    # 1. 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    vis_dir = os.path.join(SAVE_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 2. 加载训练好的模型
    print(f"\n[1/5] 正在加载模型：{MODEL_WEIGHTS_PATH}")
    model = YOLO(str(MODEL_WEIGHTS_PATH))
    print("模型加载成功！")

    # 3. 在官方测试集上进行量化评估
    print("\n[2/5] 正在测试集上进行量化评估...")
    metrics = model.val(
        data=DATA_YAML_PATH,
        split="test",  # 明确指定在test集评估
        imgsz=640,
        batch=32,
        conf=0.001,  # 高召回率设置
        iou=0.6,
        device=0,
        plots=True,  # 自动生成PR曲线、混淆矩阵等
        save_json=True,  # 保存JSON格式的详细结果
        project=SAVE_DIR,
        name="quantitative_eval"
    )

    # 4. 提取并保存核心性能指标（用于论文写作）
    print("\n[3/5] 正在提取并保存核心性能指标...")
    result_text = "=" * 80 + "\n"
    result_text += "【论文官方测试集最终性能指标】\n"
    result_text += "=" * 80 + "\n\n"
    result_text += f"模型权重: {MODEL_WEIGHTS_PATH}\n"
    result_text += f"数据集: {DATA_YAML_PATH}\n\n"

    # 核心检测指标
    result_text += "--- 核心检测指标 ---\n"
    result_text += f"Precision (P): {metrics.box.mp:.4f}\n"
    result_text += f"Recall (R):    {metrics.box.mr:.4f}\n"
    result_text += f"mAP50:         {metrics.box.map50:.4f}\n"
    result_text += f"mAP50-95:      {metrics.box.map:.4f}\n\n"

    # 模型复杂度指标
    model_info = model.info()
    result_text += "--- 模型复杂度指标 ---\n"
    result_text += f"参数量 (Params): {model_info[0] / 1e6:.2f} M\n"
    result_text += f"计算量 (GFLOPs): {model_info[1]:.2f}\n\n"

    # 速度指标
    result_text += "--- 推理速度指标 ---\n"
    result_text += f"单张图片推理速度: 约 {1000 / metrics.speed['inference']:.1f} FPS\n"
    result_text += f"  - 预处理: {metrics.speed['preprocess']:.2f} ms\n"
    result_text += f"  - 推理:   {metrics.speed['inference']:.2f} ms\n"
    result_text += f"  - 后处理: {metrics.speed['postprocess']:.2f} ms\n"
    result_text += "=" * 80 + "\n"

    # 打印并保存指标
    print(result_text)
    with open(os.path.join(SAVE_DIR, "test_set_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"核心指标已保存至：{os.path.join(SAVE_DIR, 'test_set_metrics.txt')}")

    # 5. 可视化部分检测结果（用于论文配图）
    print(f"\n[4/5] 正在可视化保存前{VISUALIZE_SAMPLES}张检测结果...")

    # 获取测试集图片路径
    import yaml
    with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    test_img_dir = os.path.join(data_cfg['path'], data_cfg['test'])

    # 遍历并可视化
    img_list = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][
               :VISUALIZE_SAMPLES]
    for img_name in tqdm(img_list, desc="可视化进度"):
        img_path = os.path.join(test_img_dir, img_name)

        # 推理
        results = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=0.6,
            imgsz=640,
            device=0,
            verbose=False
        )

        # 保存检测结果图片
        for r in results:
            annotated_img = r.plot()  # 自动绘制检测框
            save_path = os.path.join(vis_dir, f"det_{img_name}")
            cv2.imwrite(save_path, annotated_img)

    print(f"\n[5/5] 所有评估任务完成！")
    print(f"完整结果保存路径：{SAVE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

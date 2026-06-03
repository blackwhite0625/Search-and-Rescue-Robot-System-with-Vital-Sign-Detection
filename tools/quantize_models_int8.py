"""
YOLO ONNX → INT8 動態量化工具
=============================
把 model/yolov8n.onnx 與 model/yolov8n-pose.onnx 量化為 int8 版本。
INT8 在 RPi 5 CPU 推論可加速 20–40%，精度下降通常 <1% mAP。

使用方式（在 RPi 上執行一次即可）：
    pip install onnxruntime onnx
    python3 tools/quantize_models_int8.py

產出：
    model/yolov8n_int8.onnx
    model/yolov8n-pose_int8.onnx

detector.py 啟動時會優先載入 *_int8.onnx；找不到則 fallback 到原 onnx。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(PROJECT, "model")

MODELS = [
    ("yolov8n.onnx", "yolov8n_int8.onnx"),
    ("yolov8n-pose.onnx", "yolov8n-pose_int8.onnx"),
]


def main() -> int:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("❌ 缺少 onnxruntime。請先 `pip install onnxruntime`。", file=sys.stderr)
        return 1

    for src_name, dst_name in MODELS:
        src = os.path.join(MODEL_DIR, src_name)
        dst = os.path.join(MODEL_DIR, dst_name)
        if not os.path.exists(src):
            print(f"⚠️  找不到 {src}，略過")
            continue
        if os.path.exists(dst):
            print(f"🟡 {dst_name} 已存在；若要重新量化請先刪除")
            continue
        print(f"⏳ 量化 {src_name} → {dst_name} (QUInt8, 動態)...")
        try:
            quantize_dynamic(
                model_input=src,
                model_output=dst,
                weight_type=QuantType.QUInt8,
            )
            src_mb = os.path.getsize(src) / 1e6
            dst_mb = os.path.getsize(dst) / 1e6
            print(f"✅ 完成 {dst_name}：{src_mb:.1f} MB → {dst_mb:.1f} MB")
        except Exception as e:
            print(f"❌ {src_name} 量化失敗：{e}", file=sys.stderr)
            return 1

    print("\n下一步：重啟 app.py；detector 會自動偵測並優先載入 *_int8.onnx。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

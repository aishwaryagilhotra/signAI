import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

ONNX_PATH = "asl_model.onnx"
TFLITE_PATH = "asl_model.tflite"

# load ONNX model
onnx_model = onnx.load(ONNX_PATH)

# convert ONNX → TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model")

# load the TF model
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")

# optimize model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convert to TFLite
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Exported TFLite model → {TFLITE_PATH}")

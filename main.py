import tensorflow as tf
print(tf.lite.Interpreter(model_path="model.tflite").get_input_details())
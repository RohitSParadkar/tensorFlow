import tensorflow as tf
from ultralytics import YOLO

def convert_pb_to_pbtxt(pb_file_path, pbtxt_file_path):
    # Load the frozen TensorFlow graph
    with tf.io.gfile.GFile(pb_file_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Convert the graph to a text representation
    with tf.io.gfile.GFile(pbtxt_file_path, 'w') as f:
        f.write(str(graph_def))

    print(f'Converted {pb_file_path} to {pbtxt_file_path}')


def yolov8_to_tflite(yolov8_file_path,tflite_file_path):
    model = YOLO(yolov8_file_path)
    model.export(format='tflite') # creates 'yolov8n_float32.tflite'
    # Load the exported TFLite model
    tflite_model = YOLO(tflite_file_path)
    print("Model converted successfully")

if __name__ == "__main__":
    # # Path to the frozen TensorFlow graph (.pb file)
    # pb_file_path = '../models/frozen_inference_graph.pb'

    # # Path for the output .pbtxt file
    # pbtxt_file_path = '../models/text.pbtxt'

    # # Convert the frozen graph to pbtxt
    # convert_pb_to_pbtxt(pb_file_path, pbtxt_file_path)

    yolov8_file_path='../models/best_v8_latest.pt'
    tflite_file_path='../models/yolov8_float32.tflite'
    yolov8_to_tflite(yolov8_file_path,tflite_file_path)
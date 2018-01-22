import sys
import numpy as np
import os
import tensorflow as tf
import time
import cv2

# TensorFlow API path
TF_API_DIR = './tf_api/research/'
sys.path.append(TF_API_DIR)

from queue import Queue
from threading import Thread
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Constants
# Model names:
# ssd_mobilenet_v1_coco_2017_11_17
# ssd_inception_v2_coco_2017_11_17
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

NUM_CLASSES = 90
SCORE_TRESHOLD = 0.5

THREAD_NUM = 1

PATH_TO_CKPT = 'models/{}/frozen_inference_graph.pb'.format(MODEL_NAME)
PATH_TO_LABELS = os.path.join(TF_API_DIR, 'object_detection/data/{}'.format('mscoco_label_map.pbtxt'))

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class CameraStream:

    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        self.cap.release()

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def detect(input_q, output_q):

    # Load detection graph and category index
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            # Start inferrence session
            sess = tf.Session(graph=detection_graph)

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Detection loop
    while True:

        # Read frame from stream
        image_np = input_q.get()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # Actual detection
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Put results in output queue
        objects = []
        for index, value in enumerate(classes[0]):
            object_dict = {}
            object_dict['class'] = (category_index.get(value)).get('name')
            object_dict['score'] = np.float64(scores[0, index])
            if object_dict['score'] > SCORE_TRESHOLD:
                object_dict['xmin'] = np.float64(boxes[0, index, 0])
                object_dict['ymin'] = np.float64(boxes[0, index, 1])
                object_dict['xmax'] = np.float64(boxes[0, index, 2])
                object_dict['ymax'] = np.float64(boxes[0, index, 3])
                objects.append(object_dict)
        
        output_q.put((image_np, objects, boxes, classes, scores))

    sess.close()


def main():

    # Processing queue
    input_q = Queue(1)
    output_q = Queue()

    # Run inferrence in separate thread
    for i in range(THREAD_NUM):
        thread = Thread(target=detect, args=(input_q, output_q))
        thread.daemon = True
        thread.start()

    # Start camera stream
    cam_stream = CameraStream(src=0, width=480, height=360)

    while True:

        # Read frame from stream
        frame = cam_stream.read()
        input_q.put(frame)

        if output_q.empty():
            pass
        else:

            # Get results
            (image_np, objects, boxes, classes, scores) = output_q.get()
    
            # Visualization
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            cv2.imshow('object detection', image_np)

            # Print results
            print(objects)

            # Breaking
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    # Stop camera stream
    cam_stream.stop()
            

if __name__ == '__main__':
    main()



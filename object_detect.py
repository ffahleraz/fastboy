import sys
import numpy as np
import os
import tensorflow as tf
import time
import cv2

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
PATH_TO_LABELS = 'object_detection/data/{}'.format('mscoco_label_map.pbtxt')


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

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.cap.release()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def read(self):
        return self.frame


class ObjectDetector:

    def __init__(self, input_queue, output_queue, detection_graph, category_index):
        
        # Input and output queue
        self.input_queue = input_queue
        self.output_queue = output_queue

        # Load model and label
        self.detection_graph = detection_graph
        self.sess = tf.Session(graph=self.detection_graph)
        self.category_index = category_index

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Run detection loop
        self.stopped = False
        self.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.sess.close()

    def update(self):
        while not self.stopped:
            if not self.input_queue.empty():
                frame = self.input_queue.get()
                detection = self.detect_objects(image_np=frame)
                self.output_queue.put(detection)

    def detect_objects(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        # Actual detection
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # Put results in output queue
        objects = []
        for index, value in enumerate(classes[0]):
            object_dict = {}
            object_dict['class'] = (self.category_index.get(value)).get('name')
            object_dict['score'] = np.float64(scores[0, index])
            if object_dict['score'] > SCORE_TRESHOLD:
                object_dict['xmin'] = np.float64(boxes[0, index, 0])
                object_dict['ymin'] = np.float64(boxes[0, index, 1])
                object_dict['xmax'] = np.float64(boxes[0, index, 2])
                object_dict['ymax'] = np.float64(boxes[0, index, 3])
                objects.append(object_dict)

        return (image_np, objects, boxes, classes, scores)


def main():

    # Load model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Load labels
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, 
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Queue frames to ensure no skipped frames
    input_queue = Queue(1)
    output_queue = Queue()

    # Start things up
    with CameraStream(src=0, width=480, height=360) as cam_stream:
        with ObjectDetector(input_queue=input_queue, output_queue=output_queue, 
            detection_graph=detection_graph, category_index=category_index) as object_detector:

            # Start timer
            start_time = time.time()

            # Detection loop
            while True:

                # Read frame from stream
                frame = cam_stream.read()
                input_queue.put(frame)

                if not output_queue.empty():

                    # Get detection
                    (image_np, objects, boxes, classes, scores) = output_queue.get()

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

                    # # Print results
                    # print(objects)

                    # Calculate framerate
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    start_time = current_time
                    print('FPS: {}'.format(1 / elapsed_time))

                # Breaking
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            

if __name__ == '__main__':
    main()



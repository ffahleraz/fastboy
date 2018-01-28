# FastBoy

Realtime object detection on CPU only machines.

## TL;DR

This is an implementation of deep learning-based object detection capable of running real-time (with acceptable framerate) on machines that doesn't have a dedicated GPU, thus only running inference on the CPU.

Key takeaways:
- Uses the SSD Mobilenet model trained on the COCO dataset, this model is provided by the TensorFlow Object Detection API.
- Uses OpenCV to load frames from the input source (i.e. camera).
- Separates frame loading, inference, and visualization into different threads.

## Setup and Running

### Requirements

To run this program you need:
- TensorFlow 1.4 or above
- OpenCV 3.0 or above
- Python 3.5

### Running

Just clone or download this repo and run the `object_detect.py` file:

```
$ python3 object_detect.py
```

To see all available options, just run with a `--help` tag:

```
$ python3 object_detect.py --help
```

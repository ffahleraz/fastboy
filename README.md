### FastBoy

Realtime object detection on CPU only machines.

## TL;DR

This is an implementation of realtime object detection capable of running on machines that doesn't have a discrete GPU (in the case of TensorFlow, CUDA enabled GPU) thus only running the inference on the CPU.

Key takeaways:
- Uses a modified version of TensorFlow Object Detection API to speed up inference.
- Uses OpenCV to load frames from a camera (in this case, a webcam).
- Separate inference and frame loading into separate threads.


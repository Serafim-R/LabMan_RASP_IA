# LabMan Raspberry 5 IA
This project implements an edge-based computer vision system for industrial safety validation using a Raspberry Pi 5 and a Camera Module 3 mounted on a hydraulic press. A Convolutional Neural Network (CNN) is used to detect the presence of both tooling and workpiece inside the press in real time.

All processing is performed locally on the device (edge computing), ensuring low latency and independence from network connectivity. Based on the detection results, the system outputs a signal indicating whether the press is safe to operate or not.

The solution is designed as a redundant safety layer, complementing existing safety mechanisms rather than replacing them. It demonstrates the application of embedded AI and machine vision in industrial environments for enhanced operational safety.

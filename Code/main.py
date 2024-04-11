from Detector import *
import os

def main():
    videoPath = "./testcar.mp4"
    configPath = "../models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    modelPath = "../models/frozen_inference_graph.pb"
    classPath = "../models/cocos.names"
    detector = Detector(videoPath,configPath,modelPath,classPath)
    detector.onVideo()

if __name__ == '__main__':
    main()
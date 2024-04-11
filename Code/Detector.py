import cv2
import numpy as np 
import time

np.random.seed(20)
class Detector:
    def __init__(self,videoPath, configPath,modelPath,classPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classPath = classPath
        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)
        self.readClasses()

    def readClasses(self):
        with open(self.classPath,'r') as f:
            self.classList = f.read().splitlines()
            self.classList.insert(0,'__Background__')  #0 class is for background
            self.colorList = np.random.uniform(low=0,high=255,size=(len(self.classList),3))
            print(self.classList)

    def onVideo(self):
         cap =  cv2.VideoCapture(self.videoPath)
         if(cap.isOpened()==False):
             print("Error opening file")
             return
         (success,image) = cap.read()
         while success:
             classLableIDS , confidence ,bboxs = self.net.detect(image,confThreshold=0.4)
             detectList = self.net.detect(image,confThreshold=0.4)
             bboxs = list(bboxs)
             print("bboxs",bboxs)
             confidence = list(np.array(confidence).reshape(1,-1)[0])
             confidence = list(map(float,confidence))
            #  print("conf\n",confidence)
            #  print("bboxs",bboxs)
             bboxIdx = cv2.dnn.NMSBoxes(bboxs,confidence,score_threshold=0.5,nms_threshold=0.2)
             print("bboxIdx:\n",bboxIdx)
             if(len(bboxIdx)!=0):
                 for i in range(0,len(bboxIdx)):
                     bbox = bboxs[np.squeeze(bboxIdx[i])]
                     classConfidence  = confidence[np.squeeze(bboxIdx[i])]
                     classLableID = np.squeeze(classLableIDS[np.squeeze(bboxIdx[i])])
                     classLabel = self.classList[classLableID]
                     classColor = [int(c) for c in self.colorList[classLableID]]
                     displayText = "{}:{:.2f}".format(classLabel,classConfidence)
                     x,y,w,h = bbox
                     cv2.rectangle(image,(x,y),(x+w,y+h),color=classColor,thickness=1)
                     cv2.putText(image,displayText,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,classColor,2)
                     lineWidth = min(int(w*0.3),int(h*0.3))
                   
                     ##################Top line###########################
                     cv2.line(image,(x,y),(x+lineWidth,y),classColor,thickness=4)
                     cv2.line(image,(x,y),(x,y+lineWidth),classColor,thickness=4)

                     cv2.line(image,(x+w,y),(x+w-lineWidth,y),classColor,thickness=4)
                     cv2.line(image,(x+w,y),(x+w,y+lineWidth),classColor,thickness=4)
                     #################### Bottom line ################################## 
                     cv2.line(image,(x,y+h),(x+lineWidth,y+h),classColor,thickness=4)
                     cv2.line(image,(x,y+h),(x,y+h-lineWidth),classColor,thickness=4)

                     cv2.line(image,(x+w,y+h),(x+w-lineWidth,y+h),classColor,thickness=4)
                     cv2.line(image,(x+w,y+h),(x+w,y+h-lineWidth),classColor,thickness=4)


             cv2.namedWindow("Result", cv2.WINDOW_NORMAL) 
             cv2.resizeWindow("Result", 1000, 1000) 
             cv2.imshow("Result",image)
             key =cv2.waitKey(1)& 0xFF
             if(key==ord("q")):
                 break
             
             (success,image) = cap.read()
         cv2.destroyAllWindows(0)
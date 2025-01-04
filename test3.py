import cv2

# Threshold to detect object
thres = 0.55

#Capturing the video from the system camera
cap = cv2.VideoCapture(0)

#Sets given to class and objects
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
classNames= []
#Importing the Object name files from different location.
classFile = 'obj.names'
# Open function to read the data from the file.
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#Getting weightage and paths from different locations.
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# As the camera will not turn off because of True condition.
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=4)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]-10,box[1]-15),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
            cv2.putText(img,str(round(confidence*100,2)) + "%",(box[0]+150,box[1]-15),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            #cv2.putText(img, "objects: " + classN  ames[classId-1].upper(), (0,30) ,cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # we display captured output.
    cv2.imshow("Output",img)
    cv2.waitKey(1)
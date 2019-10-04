import cv2
import numpy as np

#Loading YOLO algo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layerNames = net.getLayerNames()
outputLayer = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Loading Image
n = 4
imageName = []
for i in range(1,n+1):
    imageName.append("desk"+str(i)+".jpg")
print (imageName)
for i in range(n):
    image = cv2.resize(cv2.imread(imageName[i]), None, fx=1, fy=1)
    width, height, channels = image.shape

    #Detecting
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0,), True, crop=False)     #RGB extraction
    #Viewing blob images
    #for i in blob:
    #    for j, img in enumerate(i):
    #        cv2.imshow(str(j), img)
    net.setInput(blob)
    outputs = net.forward(outputLayer)

    #Display info
    boxes = []
    confidences = []
    classIDs = []
    for out in outputs:
        for detect in out:
            score = detect[5:]
            classID = np.argmax(score)
            confidence = score[classID]
            if confidence > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    n = len(boxes)
    for i in range(n):
        x,y,w,h = boxes[i]
        label = str(classes[classIDs[i]])
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y+30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow('Image',image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()


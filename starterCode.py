import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # load model

red = (0, 0, 255)
green = (0, 255, 0)

def drawboxWithNameAndConf(result, image):
    h, w, _ = image.shape # dimensions of image
    
    x1 = int(result.xmin*w) #dimensions of object detected
    y1 = int(result.ymin*h)
    x2 = int(result.xmax*w)
    y2 = int(result.ymax*h)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2) # draw bounding box
    cv2.putText(image, f'{result.values[6]} {result.confidence:.2f}',
                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2) # write confidence

def watchVideo():
    video = cv2.VideoCapture('intersection.mp4') # oepn and read video
    ret, frame = video.read()

    while ret:
        
        frameForModel = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB) # preprocessing
        
        results = model(frameForModel) # inference 

        for _, r in results.pandas().xyxyn[0].iterrows(): # parse results
            drawboxWithNameAndConf(r, frame)
        
        cv2.imshow("Video", frame) # show video
        if cv2.waitKey(35) & 0xFF == ord('q'):
            break

        ret, frame = video.read() # read next frame
        
watchVideo()
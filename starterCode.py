import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

red = (0, 0, 255)
green = (0, 255, 0)

def drawboxWithNameAndConf(result, image):
    h, w, _ = image.shape
    x1 = int(result.xmin*w)
    y1 = int(result.ymin*h)
    x2 = int(result.xmax*w)
    y2 = int(result.ymax*h)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, f'{result.values[6]} {result.confidence:.2f}',
                (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def watchVideo():
    video = cv2.VideoCapture('intersection.mp4')
    ret, frame = video.read()

    while ret:
        
        frameForModel = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB) 
        results = model(frameForModel)

        for _, r in results.pandas().xyxyn[0].iterrows():
            if r.values[6] == 'person':
                drawboxWithNameAndConf(r, frame)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(35) & 0xFF == ord('q'):
            break

        ret, frame = video.read()
        
watchVideo()
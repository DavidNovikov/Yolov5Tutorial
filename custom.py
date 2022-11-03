import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
    
    w  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    slope1 = (100 - 200) / 1280
    slope2 = (450 - 600) / 1280
    intercept1 = 200
    intercept2 = 600

    while ret:
        dangerous = False
        
        frameForModel = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB) 
        results = model(frameForModel)

        for _, r in results.pandas().xyxyn[0].iterrows():
            if r.values[6] == 'person':
                drawboxWithNameAndConf(r, frame)
                
                x1 = r.xmin*w
                y1 = r.ymin*h
                x2 = r.xmax*w
                y2 = r.ymax*h
                
                x = (x1+x2)/2
                y = (y1+y2)/2
                
                if x * slope1 + intercept1 < y and x * slope2 + intercept2 > y:
                    dangerous = True
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 13)

        color = None
        text = ''
        if dangerous:
            color = (0, 0, 255)
            text = 'DANGER'
        else:
            color = (0, 255, 0)
            text = 'SAFE'
            
        cv2.putText(frame, text, (600, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.line(frame, (0, 200), (1280, 100), color, 2)
        cv2.line(frame, (0, 600), (1280, 450), color, 2)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = video.read()
        
watchVideo()
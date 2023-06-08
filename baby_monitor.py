import cv2
import threading
import time
import serial
import numpy as np

from flask import (Flask, Response, render_template)

app = Flask(__name__)


class Datapipe:
    def __init__(self):
        self.msg = []
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()

    def get_msg(self):
        msg = []
        if not self.consumer_lock.locked():
            self.consumer_lock.acquire()
            msg = self.msg
            self.producer_lock.release()
        return msg

    def set_msg(self, msg):
        self.producer_lock.acquire()
        self.msg = msg
        self.consumer_lock.release()


def poll_sensors(dp, rate):
    with serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=2.) as device:
        while True:
            device.readline()
            time.sleep(rate)

            data = device.readline().decode().rstrip().split(",")
            if len(data) > 1:
                fahrenheit = float(data[0]) * (9/5) + 32
                brightness = 100. if float(data[1]) > 100. else float(data[1])
                dp.set_msg([fahrenheit, brightness])


def get_bbox(frame, outs, idx=0, objectness_threshold=0.5, conf_threshold=0.5, nms_threshold=0.5):
    frame_height, frame_width = frame.shape[:2]

    class_ids = []
    boxes = []
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == idx and detection[4] > objectness_threshold and confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    return [] if not len(indices) else boxes[indices[0]]


def acquire_frames():
    cap = cv2.VideoCapture(0)

    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    tracker = cv2.TrackerCSRT_create()

    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    
    dp = Datapipe()
    poll_sensors_t = threading.Thread(target=poll_sensors, args=(dp, 1,), daemon=True)
    poll_sensors_t.start()

    tracking = False
    last = [0., 0.]
    last_midpoint = [-1., -1.]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        msg = dp.get_msg() 
        if not len(msg):
            msg = last

        bbox = []
        if not tracking:
            blob = cv2.dnn.blobFromImage(frame, 1./255., (320, 320), [0,0,0], 1, crop=False)

            net.setInput(blob)

            outs = net.forward(output_layers)

            bbox = get_bbox(frame, outs)
            error_txt = 'Baby not detected'
            if len(bbox) == 0:
                cv2.putText(frame, error_txt, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
            else:
                tracker.init(frame, bbox)
                tracking = True

        else:
            ok, bbox = tracker.update(frame)
            if not ok:
                cv2.putText(frame, error_txt, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
                tracking = False

        if len(bbox) == 4:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0))

            midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
            if -1 not in last_midpoint:
                d = np.linalg.norm((np.array(midpoint) - np.array(last_midpoint)))
                if d > 10:
                    cv2.putText(frame, "Baby is moving", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)

            last_midpoint = midpoint

        txt = 'Temp: {0:0.1f}F Brightness: {1:0.1f}%'.format(msg[0], msg[1])
        last = msg

        h, w, c = frame.shape
        cv2.putText(frame, txt, (20, h-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

        ret, img = cv2.imencode('.jpg', frame)
        if not ret:
            break

        frame = img.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stream')
def stream():
    return Response(acquire_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

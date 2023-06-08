import cv2
import Adafruit_DHT
import threading
import time
import RPi.GPIO as GPIO

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
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    while True:
        humidity, celsius = Adafruit_DHT.read_retry(11, 4)
        fahrenheit = celsius * (9/5) + 32

        photo_resistor = GPIO.input(17)

        dp.set_msg([humidity, fahrenheit, photo_resistor])
        time.sleep(rate)


def acquire_frames():
    cap = cv2.VideoCapture(0)
    
    dp = Datapipe()
    poll_sensors_t = threading.Thread(target=poll_sensors, args=(dp,10,), daemon=True)
    poll_sensors_t.start()

    last = [0., 0., 0]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        msg = dp.get_msg() 
        if not len(msg):
            msg = last

        brightness = 'room is bright' if not msg[2] else 'room is dark'
        txt = 'Temp: {0:0.1f}F Hummidity: {1:0.1f}%'.format(msg[1], msg[0])
        txt += ' and ' + brightness
        last = msg

        h, w, c = frame.shape
        cv2.putText(frame, txt, (20, h-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

        ret, img = cv2.imencode('.jpg', frame)
        if not ret:
            break

        frame = img.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stream')
def stream():
    return Response(acquire_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)

from flask import Flask, render_template, Response
import threading
import queue
import cv2
import time
import datetime
import face_recognition
import numpy as np
import rtsp

RTSP_URL = "rtsp://...1"
app=Flask(__name__)

work_stat = 1
current_name = "" 

class ThreadingVideoCapture:
    def __init__(self, src, max_queue_size=256):
        self.video = cv2.VideoCapture(src)
        # fps = self.video.get(cv2.CAP_PROP_FPS)
        # self.wait_sec = 1 / fps
        self.wait_sec = 1 / 500
        self.q = queue.Queue(maxsize=max_queue_size)
        self.stopped = False

        self.krish_image = face_recognition.load_image_file("/home/user/py_projects/bad_video/Dima/Dima.jpg")
        self.krish_face_encoding = face_recognition.face_encodings(self.krish_image)[0]

        # Load a second sample picture and learn how to recognize it.
        self.bradley_image = face_recognition.load_image_file("/home/user/py_projects/bad_video/Gabe/Gabe.jpg")
        self.bradley_face_encoding = face_recognition.face_encodings(self.bradley_image)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            self.krish_face_encoding,
            self.bradley_face_encoding
        ]
        self.known_face_names = [
            "Slava",
            "Zhirobas"
        ]

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while True:
            # print('q_size:', self.q.qsize())

            if self.stopped:
                return

            if not self.q.full():
                try:  # have to remove !!!!!!!!!!!!!!!!!!!!!!!!
                    ok, frame = self.video.read()
                except Exception as e:
                    if 'Unknown C++' in str(e):
                        print(e)
                        time.sleep(self.wait_sec)
                        continue
                    else:
                        raise e
                self.q.put((ok, frame))
                # print('pos:', self.video.get(cv2.CAP_PROP_POS_FRAMES))

                if not ok:
                    # self.stop()
                    # return
                    time.sleep(1)

            else:
                time.sleep(self.wait_sec)

    def read(self):
        # print(f'q_size: {self.q.qsize()}')
        return self.q.get()

    def stop(self):
        self.stopped = True

    def release(self):
        self.stopped = True
        self.video.release()

    def isOpened(self):
        return self.video.isOpened()

    def get(self, i):
        return self.video.get(i)

    def set(self, i, v):
        return self.video.set(i, v)

RTSP_URL = "rtsp://tapocam:neural@95.84.148.126:554/stream1"

def gen_frames():
    cap = ThreadingVideoCapture(RTSP_URL)
    cap.start()
    false_count = 0
    while True:
            ret, frame = cap.read()
            if ret:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                if cap.process_this_frame:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []
                    if work_stat == 1:
                        #face_names = []
                        for face_encoding in face_encodings:


                            matches = face_recognition.compare_faces(cap.known_face_encodings, face_encoding)
                            name = "Unknown!"
                            face_distances = face_recognition.face_distance(cap.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = cap.known_face_names[best_match_index]
                                global current_name
                                current_name = name
                            face_names.append(name)
                            print(f'Detected: {name}')
                        else:
                            pass
                cap.process_this_frame = not cap.process_this_frame
                if work_stat == 1:
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                else:
                    pass
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            else:
                print(f'retval: {ret}')
                print(f'image: {frame}')
                print(f'q_size: {cap.q.qsize()}')
                print(datetime.datetime.now())
                false_count += 1
                time.sleep(1 / 30)

            if false_count > 10:
                print(f'false_count: {false_count}')
                break

            if cv2.waitKey(1) == 27:
                break


@app.route('/set_work_stat_to_one', methods=['POST'])
def set_work_stat_to_one():
    global work_stat
    work_stat = 1
    return 'OK'
@app.route('/set_work_stat_to_zero', methods=['POST'])
def set_work_stat_to_zero():
    global work_stat
    work_stat = 0
    return 'OK'

@app.route('/')
def index_1():
    return render_template('index.html')

@app.route('/video')
def index():
    return render_template('main_index.html')

@app.route('/video_feed')
def video_feed():
     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hidden')
def index_hid():
    return render_template('hidden.html', current_name=current_name)
@app.route('/get_current_name')
def get_current_name():
    global current_name
    return current_name
if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0", port=8006)
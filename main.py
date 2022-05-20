from apscheduler.schedulers.background import BackgroundScheduler
import cv2
import time
import queue
import re
from threading import Thread
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np

# 이전 차량 번호 (ex. 서울00가0000) :
old_car_number_format = re.compile(r'[가-힣]{2}\d{2}[가-힣]\d{4}')
# 신규 차량 번호 (ex. 23호 6144) :
new_car_number_format = re.compile(r'\d{2,3}[가-힣]\s\d{4}')
new_car_number_format2 = re.compile(r'\d{2,3}[가-힣]\d{4}')
car_numbers = []
view_frame = None
input_size = 416

def main():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    cam1 = 'rtsp://admin:123456@192.168.1.103/streaming/channel/101'
    cam1_queue = queue.Queue()
    cam1_receive_thread = Thread(target=receive_image, args=(cam1, cam1_queue))
    cam1_receive_thread.start()
    time.sleep(5)
    vision_scheduler = BackgroundScheduler(timezone="Asia/Seoul")
    vision_scheduler.add_job(ocr_process1, 'interval', args=(cam1_queue, infer), seconds=0.4)
    vision_scheduler.start()

    while True:
        if view_frame is not None:
            # print("view_frame is not None")
            cv2.imshow("result", view_frame)
            cv2.waitKey(10)


def ocr_process1(video_que, infer):
    global car_numbers, view_frame
    # print("ocr_process1 실행")
    frame = get_image(video_que)
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    # prev_time = time.time()

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(frame, pred_bbox)

    # curr_time = time.time()
    # exec_time = curr_time - prev_time
    # info = "time: %.2f ms" % (1000 * exec_time)
    # print(info)

    view_frame = np.asarray(image)


def receive_image(src, q):
    cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    q.put(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(src)
            continue
        if frame is None:
            cap.release()
            cap = cv2.VideoCapture(src)
            continue
        q.put(frame)


def get_image(q):
    if q.empty() is not True:
        frame = q.get()
        q.queue.clear()
        return frame


main()
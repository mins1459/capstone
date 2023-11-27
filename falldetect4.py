import cv2
import time
import torch
import argparse
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from falltrain import Get_coord
from acceleration import AccelerationCalculator
from twilio.rest import Client
import threading
import pygame
import os
import sys
import subprocess
import pymysql
import configparser
from flask import Flask, jsonify
import requests

AWS_SERVER_IP = "43.202.187.77"
AWS_SERVER_PORT = "5000"


config = configparser.ConfigParser()
config.read('config.ini')

sms_send_lock = threading.Lock()

RDS_HOST = config.get('DATABASE', 'HOST')
RDS_PORT = config.getint('DATABASE', 'PORT')
RDS_USER = config.get('DATABASE', 'USER')
RDS_PASSWORD = config.get('DATABASE', 'PASSWORD')
RDS_DB = config.get('DATABASE', 'DB_NAME')
RDS_CHARSET = config.get('DATABASE', 'CHARSET')

def execute_query(query, args=None):
    connection = pymysql.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        user=RDS_USER,
        passwd=RDS_PASSWORD,
        db=RDS_DB,
        charset=RDS_CHARSET
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, args)
            result = cursor.fetchall()
        return result
    finally:
        connection.close()

# Twilio setup
ACCOUNT_SID = config.get('TWILIO', 'ACCOUNT_SID')
AUTH_TOKEN = config.get('TWILIO', 'AUTH_TOKEN')
TWILIO_PHONE = config.get('TWILIO', 'PHONE_NUMBER')
client = Client(ACCOUNT_SID, AUTH_TOKEN)

last_sms_time = 0
last_detection_time = 0

userfall = False
messagecount = 0
last_fall_time = 0

volume_increment = 0.5
max_volume = 1.0

def send_sms(phone_number, message):
    formatted_phone_number = "+82" + phone_number[1:]
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE,
        to=formatted_phone_number
    )


def send_fall_detection_alert():
    with sms_send_lock:
        global userfall, messagecount, last_sms_time
        print("Starting SMS alert")

        emergency_numbers = execute_query("SELECT emergencycall FROM emergency")
        emergency_numbers = [row[0] for row in emergency_numbers]

        member_info = execute_query(
            "SELECT memberAD, memberphone, alertphonenumber1, alertphonenumber2, alertphonenumber3 FROM member WHERE cameraid = '1'")

        emergency_sent = False

        while userfall:
            if time.time() - last_sms_time >= 15:
                if not emergency_sent:
                    for number in emergency_numbers:
                        send_sms(number, "낙상발생")
                    emergency_sent = True

                for memberAD, memberphone, alertphonenumber1, alertphonenumber2, alertphonenumber3 in member_info:
                    message = f"낙상발생\n{memberAD} 에서 낙상이 발생했습니다.\n보호자번호 : {memberphone}으로 연락주세요."
                    if memberphone:
                        send_sms(memberphone, message)
                    if alertphonenumber1:
                        send_sms(alertphonenumber1, message)
                    if alertphonenumber2:
                        send_sms(alertphonenumber2, message)
                    if alertphonenumber3:
                        send_sms(alertphonenumber3, message)

                print("SMS sent.")
                last_sms_time = time.time()
                messagecount += 1
            time.sleep(1)

        for memberAD, memberphone, alertphonenumber1, alertphonenumber2, alertphonenumber3 in member_info:
            message = f"낙상상태 해제\n{memberAD} 에서 사용자가 낙상상태에서 벗어나 정상적으로 활동중입니다."
            if memberphone:
                send_sms(memberphone, message)
            if alertphonenumber1:
                send_sms(alertphonenumber1, message)
            if alertphonenumber2:
                send_sms(alertphonenumber2, message)
            if alertphonenumber3:
                send_sms(alertphonenumber3, message)

        for number in emergency_numbers:
            if number:
                send_sms(number, "낙상상태 해제")

        print("SMS alert ended.")

def play_alert_sound():
    global userfall

    pygame.mixer.init()
    pygame.mixer.music.load('sound/alert.wav')
    current_volume = pygame.mixer.music.get_volume()
    print("Starting alert sound...")

    while userfall:
        pygame.mixer.music.play()
        time.sleep(10)

        current_volume += volume_increment
        if current_volume > max_volume:
            current_volume = max_volume
        pygame.mixer.music.set_volume(current_volume)

    pygame.mixer.music.stop()
    print("Alert sound ended.")

def update_fall_status_on_server(fall_status):
    url = f"http://{AWS_SERVER_IP}:{AWS_SERVER_PORT}/fall_status"
    data = {"fall_status": fall_status}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Fall status updated successfully on server")
        else:
            print("Failed to update fall status on server")
    except Exception as e:
        print(f"Error updating fall status on server: {str(e)}")


@torch.no_grad()
def run(poseweights='yolov7-w6-pose', source='0', device='cpu'):
    global userfall, last_fall_time
    print("Starting fall detection...")
    video_path = source
    device = select_device(device)
    half = device.type != 'cpu'
    model = attempt_load(poseweights, map_location=device)
    model.eval()
    last_print_time = 0

    if video_path == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam", 960, 540)

    frame_count, total_fps = 0, 0
    acceleration_calculator = AccelerationCalculator()

    while True:
        ret, frame = cap.read()
        if ret:
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            image = image.to(device)
            image = image.float()
            start_time = time.time()
            with torch.no_grad():
                output, _ = model(image)
            output = non_max_suppression_kpt(
                output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            img = image[0].permute(1, 2, 0) * 255
            img = img.cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            thre = (frame_height // 2) + 100
            for idx in range(output.shape[0]):
                kpts = output[idx, 7:].T

                if kpts.ndim == 2 and kpts.shape[0] > 1:
                    point = (kpts[1][0], kpts[1][1])
                    acceleration = acceleration_calculator.compute_acceleration(kpts)
                else:
                    acceleration = 0

                plot_skeleton_kpts(img, kpts, 3)
                xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
                xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
                difference = int(ymax) - int(ymin) - (int(xmax) - int(xmin))
                ph = Get_coord(kpts, 2)

                if ((difference < 0) and (int(ph) > thre)) or (difference < 0) and acceleration > 1500:
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
                    if not userfall:
                        if last_fall_time == 0:
                            last_fall_time = time.time()
                        elif time.time() - last_fall_time >= 15:
                            userfall = True
                            update_fall_status_on_server(True)
                            last_fall_time = time.time()
                            threading.Thread(target=send_fall_detection_alert).start()
                            threading.Thread(target=play_alert_sound).start()
                else:

                    if userfall:
                        userfall = False
                        update_fall_status_on_server(False)
                        messagecount = 0

            img_ = img.copy()
            img_ = cv2.resize(img_, (960, 540), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Webcam", img_)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            if time.time() - last_print_time >= 3:
                fall_status = "Fall detected" if userfall else "No fall"
                print(f"{fall_status}, FPS: {fps:.2f}")
                last_print_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print("Program terminated.")



if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default='0', help='video/0 for webcam')
        parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
        args = parser.parse_args()

        strip_optimizer(args.device, args.poseweights)
        run(**vars(args))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
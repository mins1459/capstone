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

config = configparser.ConfigParser()
config.read('config.ini')

sms_send_lock = threading.Lock()

RDS_HOST = config.get('DATABASE', 'HOST')
RDS_PORT = config.getint('DATABASE', 'PORT')  # 정수로 변환
RDS_USER = config.get('DATABASE', 'USER')
RDS_PASSWORD = config.get('DATABASE', 'PASSWORD')
RDS_DB = config.get('DATABASE', 'DB_NAME')
RDS_CHARSET = config.get('DATABASE', 'CHARSET')

# 데이터베이스 연결
db = pymysql.connect(
    host=RDS_HOST,
    port=RDS_PORT,
    user=RDS_USER,
    passwd=RDS_PASSWORD,
    db=RDS_DB,
    charset=RDS_CHARSET
)
cursor = db.cursor()

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

volume_increment = 0.5  # 소리 크기를 10%씩 증가시킵니다.
max_volume = 1.0

def send_sms(phone_number, message):
    formatted_phone_number = "+82" + phone_number[1:]  # '0'을 제거하고 '+82'를 추가
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE,
        to=formatted_phone_number
    )


def send_fall_detection_alert():
    with sms_send_lock:
        global userfall, messagecount, last_sms_time
        print("Starting SMS alert")

        # emergency 테이블에서 emergencycall 번호 가져오기
        cursor.execute("SELECT emergencycall FROM emergency")
        emergency_numbers = [row[0] for row in cursor.fetchall()]

        # member 테이블에서 필요한 정보 가져오기
        cursor.execute("SELECT memberAD, memberphone, alertphonenumber1, alertphonenumber2, alertphonenumber3 FROM member WHERE camerid = '1'")
        member_info = cursor.fetchall()

        emergency_sent = False  # emergency 번호로 문자를 보냈는지 확인하는 플래그

        while userfall:
            # Check if it's been more than 15 seconds since the last message
            if time.time() - last_sms_time >= 15:
                # emergency 번호로 문자 보내기 (한 번만)
                if not emergency_sent:
                    for number in emergency_numbers:
                        send_sms(number, "낙상발생")
                    emergency_sent = True

                # member 번호로 문자 보내기
                for memberAD, memberphone, alertphonenumber1, alertphonenumber2, alertphonenumber3 in member_info:
                    message = f"낙상발생\n{memberAD} 에서 낙상이 발생했습니다.\n보호자번호 : {memberphone}으로 연락주세요."
                    if memberphone:  # memberphone이 빈 문자열이 아니면
                        send_sms(memberphone, message)
                    if alertphonenumber1:  # alertphonenumber1이 빈 문자열이 아니면
                        send_sms(alertphonenumber1, message)
                    if alertphonenumber2:  # alertphonenumber2이 빈 문자열이 아니면
                        send_sms(alertphonenumber2, message)
                    if alertphonenumber3:  # alertphonenumber3이 빈 문자열이 아니면
                        send_sms(alertphonenumber3, message)

                print("SMS sent.")  # 문자 전송 문구
                # Update the last_sms_time and messagecount
                last_sms_time = time.time()
                messagecount += 1
            time.sleep(1)

        # 낙상상태 해제 문자 보내기
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

            # emergency 번호로 낙상상태 해제 문자 보내기
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
        time.sleep(10)  # 10초 대기

        # 볼륨 증가
        current_volume += volume_increment
        if current_volume > max_volume:
            current_volume = max_volume
        pygame.mixer.music.set_volume(current_volume)

    pygame.mixer.music.stop()
    print("Alert sound ended.")

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
                p1 = (int(xmin), int(ymin))
                p2 = (int(xmax), int(ymax))
                difference = int(ymax) - int(ymin) - (int(xmax) - int(xmin))
                ph = Get_coord(kpts, 2)

                if ((difference < 0) and (int(ph) > thre)) or (difference < 0) and acceleration > 500:
                    cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
                    if not userfall:
                        if last_fall_time == 0:  # 처음으로 바운딩 박스가 감지된 경우
                            last_fall_time = time.time()
                        elif time.time() - last_fall_time >= 15:  # 바운딩 박스가 15초 동안 지속된 경우
                            userfall = True
                            last_fall_time = time.time()
                            threading.Thread(target=send_fall_detection_alert).start()
                            threading.Thread(target=play_alert_sound).start()  # 경고음 재생 시작
                else:
                    last_fall_time = 0  # 바운딩 박스가 없는 경우 시간을 리셋
                    if userfall:
                        userfall = False
                        messagecount = 0

            img_ = img.copy()
            img_ = cv2.resize(img_, (960, 540), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Webcam", img_)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            if time.time() - last_print_time >= 3:  # 3초마다 출력
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



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='video/0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    try:
        opt = parse_opt()
        strip_optimizer(opt.device, opt.poseweights)
        run(**vars(opt))
    finally:
        db.close()

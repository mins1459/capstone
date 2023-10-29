import cv2
import time
import torch
import argparse
import numpy as np
import pygame
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from falltrain import Get_coord, draw_border
from PIL import ImageFont, ImageDraw, Image
from acceleration import AccelerationCalculator
from twilio.rest import Client
import threading

sms_send_lock = threading.Lock()

# Twilio setup
ACCOUNT_SID = 'AC75ce5232bea371f04832254638cedd84'
AUTH_TOKEN = '099b4cab4264c45dc386733201086640'
TWILIO_PHONE = '+16166752568'
RECIPIENT_PHONE = '+821052444339'
client = Client(ACCOUNT_SID, AUTH_TOKEN)

last_sms_time = 0
last_detection_time = 0

fall_detected = False
fall_detection_start_time = 0
alert_volume = 0.1  # 초기 볼륨 설정

def play_alert_sound():
    global fall_detected
    global alert_volume

    pygame.mixer.init()
    pygame.mixer.music.load('alert_sound.mp3')  # 경고음 파일 경로
    pygame.mixer.music.set_volume(alert_volume)
    pygame.mixer.music.play(-1)  # 무한 반복

    while fall_detected:
        time.sleep(10)  # 10초마다 볼륨 증가
        alert_volume += 0.1  # 볼륨 10% 증가
        pygame.mixer.music.set_volume(alert_volume)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

def send_fall_detection_alert():
    global fall_detected
    global fall_detection_start_time

    while True:
        with sms_send_lock:
            global last_sms_time

            # Check if it's been more than 3 minutes since the last message
            if time.time() - last_sms_time < 180:  # 180 seconds = 3 minutes
                continue

            # Check if fall_detected has been True for 15 seconds
            if fall_detected and (time.time() - fall_detection_start_time >= 15):
                time.sleep(15)  # Wait for 15 seconds
                message = client.messages.create(
                    body="user falldown.",
                    from_=TWILIO_PHONE,
                    to=RECIPIENT_PHONE
                )
                print(message.sid)

                # Update the last_sms_time
                last_sms_time = time.time()
            elif not fall_detected:
                fall_detection_start_time = 0

@torch.no_grad()
def run(poseweights='yolov7-w6-pose', source='0', device='cpu'):
    video_path = source
    device = select_device(device)
    half = device.type != 'cpu'
    model = attempt_load(poseweights, map_location=device)
    model.eval()

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
                    global fall_detected
                    global fall_detection_start_time

                    if not fall_detected:
                        fall_detected = True
                        threading.Thread(target=play_alert_sound).start()
                        if fall_detection_start_time == 0:
                            fall_detection_start_time = time.time()

                    draw_border(img, p1, p2, (84, 61, 247), 10, 25, 25)
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    draw.rounded_rectangle((int((xmin + xmax) / 2) - 10, int((ymin + ymax) / 2) - 10,
                                            int((xmin + xmax) / 2) + 60, int((ymin + ymax) / 2) + 60), fill=(84, 61, 247),
                                           radius=15)
                    img = np.array(im)
                else:
                    fall_detected = False

            img_ = img.copy()
            img_ = cv2.resize(img_, (960, 540), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Webcam", img_)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            # 낙상감지 강제 종료
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 32:  # 스페이스바 키 코드는 32입니다.
                fall_detected = False
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='video/0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    threading.Thread(target=send_fall_detection_alert).start()  # Start the SMS sending thread
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    run(**vars(opt))

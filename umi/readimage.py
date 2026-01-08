import cv2
import numpy as np
import pathlib

DEV = 0                 # /dev/video0
W, H, FPS = 1920, 1080, 100


cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError(f"无法打开 /dev/video{DEV}")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))  # YU12 == I420
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, FPS)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

fcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
fcc = "".join([chr((fcc_int >> (8*i)) & 0xFF) for i in range(4)])
print("FOURCC from driver:", fcc)

def grab_bgr():
    ok, raw = cap.read()
    if not ok:
        return None

    yuv = np.ascontiguousarray(raw).reshape(H * 3 // 2, W)  # I420 平面堆叠
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)         # 关键：I420 转 BGR
    return bgr

id = 0
while True:
    frame = grab_bgr()
    if frame is None:
        continue
    cv2.imshow("YU12(I420) @1080p", frame)
    if id < 10:
        pathlib.Path(".").mkdir(exist_ok=True)
        cv2.imwrite(f"./frame_{id:03d}.png", frame)
        id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import json
import threading
import time
import socket
import quaternion
import numpy as np


class quest_teleop:
    def __init__(self):
        self.last_input = None
        thread1 = threading.Thread(target=self.udp_receiver)
        thread2 = threading.Thread(target=self.udp_sender)
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        time.sleep(50000)

    def hand_input(self, input):
        if self.last_input is None:
            self.last_input = input
        if input['rightHand'] < 0.5:
            tcp_pos = (np.array([input['rightPos']["x"], input['rightPos']["y"], input['rightPos']["z"]]) -
            np.array([self.last_input['rightPos']["x"],self.last_input['rightPos']["y"], self.last_input['rightPos']["z"]]))
            last_quat = quaternion.quaternion(self.last_input['rightQuat']["w"], self.last_input['rightQuat']["x"],
            self.last_input['rightQuat']["y"], self.last_input['rightQuat']["z"])
            quat = quaternion.quaternion(input['rightQuat']["w"], input['rightQuat']["x"],
            input['rightQuat']["y"], input['rightQuat']["z"])
            last_quat = quaternion.quaternion.inverse(last_quat)
            tcp_quat = last_quat * quat

        self.last_input = input

    def udp_receiver(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 10001))
        while True:
            try:
                # 接收数据
                data, addr = s.recvfrom(102400)
                data = data.decode('utf-8')
                data = json.loads(data)
                self.hand_input(data)
            except KeyboardInterrupt:
                print("Receiver stopped.")

    def udp_sender(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', 59858))

addres

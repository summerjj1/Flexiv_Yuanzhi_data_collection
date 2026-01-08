import json
import threading
import time
import socket
import argparse
import logging
import quaternion
import numpy as np
import spdlog
from typing import Tuple, Optional, Dict


class QuestTeleop:
    def __init__(self, receiver_port: int = 10001, sender_ip: str = "192.168.2.250", sender_port: int = 10004, sleep_interval: float = 0.02):
        """Initialize QuestTeleop with configurable parameters."""
        self.last_input: Optional[Dict] = None
        self.joint_states: np.ndarray = np.array([0.0] * 8)
        self.offset_tcp_pos: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.offset_tcp_quat: quaternion.quaternion = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        self.receiver_port: int = receiver_port
        self.sender_ip: str = sender_ip
        self.sender_port: int = sender_port
        self.sleep_interval: float = sleep_interval

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = spdlog.ConsoleLogger("Quest Teleop")

        # Start threads
        self._start_threads()

    def _start_threads(self) -> None:
        """Start UDP receiver and sender threads."""
        try:
            thread1 = threading.Thread(target=self.udp_receiver)
            thread2 = threading.Thread(target=self.udp_sender)
            thread1.daemon = True
            thread2.daemon = True
            thread1.start()
            thread2.start()
            self.logger.info("UDP receiver and sender threads started successfully.")
        except Exception as e:
            self.logger.error(f"Failed to start threads: {e}")
            raise

    def get_input_frame(self) -> Tuple[Optional[Dict], np.ndarray, quaternion.quaternion]:
        """Return the last input, TCP position, and quaternion."""
        return self.last_input, self.offset_tcp_pos.copy(), self.offset_tcp_quat.copy()

    def hand_input(self, input_data: Dict) -> None:
        """Process input data to update TCP position and quaternion."""
        try:
            if self.last_input is None:
                self.last_input = input_data

            if input_data.get('rightHand', 0.0) > 0.5:
                # Calculate position offset
                current_pos = np.array([
                    input_data['rightPos']['z'],
                    -input_data['rightPos']['x'],
                    input_data['rightPos']['y']
                ])
                last_pos = np.array([
                    self.last_input['rightPos']['z'],
                    -self.last_input['rightPos']['x'],
                    self.last_input['rightPos']['y']
                ])
                self.offset_tcp_pos = current_pos - last_pos

                # Calculate quaternion offset (commented out as in original)
                # start_quat = quaternion.quaternion(
                #     self.last_input['rightQuat']['w'],
                #     self.last_input['rightQuat']['x'],
                #     self.last_input['rightQuat']['y'],
                #     self.last_input['rightQuat']['z']
                # )
                # current_quat = quaternion.quaternion(
                #     input_data['rightQuat']['w'],
                #     input_data['rightQuat']['x'],
                #     input_data['rightQuat']['y'],
                #     input_data['rightQuat']['z']
                # )
                # self.offset_tcp_quat = quaternion.quaternion.inverse(start_quat) * current_quat
            else:
                self.offset_tcp_pos = np.array([0.0, 0.0, 0.0])
                self.offset_tcp_quat = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

            self.last_input = input_data
        except KeyError as e:
            self.logger.error(f"Invalid input data format: missing key {e}")
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")

    def udp_receiver(self) -> None:
        """Receive UDP packets and process them."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.receiver_port))
            self.logger.info(f"UDP receiver bound to port {self.receiver_port}")
            while True:
                data, addr = s.recvfrom(102400)
                try:
                    data = data.decode('utf-8')
                    input_data = json.loads(data)
                    self.hand_input(input_data)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self.logger.error(f"Failed to decode or parse UDP data: {e}")
        except Exception as e:
            self.logger.error(f"UDP receiver error: {e}")
            raise

    def udp_sender(self) -> None:
        """Send joint states over UDP."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            address = (self.sender_ip, self.sender_port)
            self.logger.info(f"UDP sender configured for {self.sender_ip}:{self.sender_port}")
            while True:
                joint_state = {"jointPositions": (self.joint_states * 180 / np.pi).tolist()}
                try:
                    joint_state_bytes = json.dumps(joint_state).encode('utf-8')
                    s.sendto(joint_state_bytes, address)
                except Exception as e:
                    self.logger.error(f"Failed to send UDP data: {e}")
                time.sleep(self.sleep_interval)
        except Exception as e:
            self.logger.error(f"UDP sender error: {e}")
            raise

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quest Teleop UDP Server")
    parser.add_argument('--receiver-port', type=int, default=10001, help="Port for UDP receiver")
    parser.add_argument('--sender-ip', type=str, default="192.168.2.250", help="IP address for UDP sender")
    parser.add_argument('--sender-port', type=int, default=10004, help="Port for UDP sender")
    parser.add_argument('--sleep-interval', type=float, default=0.02, help="Sleep interval for sender (seconds)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        quest_teleop = QuestTeleop(
            receiver_port=args.receiver_port,
            sender_ip=args.sender_ip,
            sender_port=args.sender_port,
            sleep_interval=args.sleep_interval
        )
        # Keep the main thread alive
        while True:
            print(quest_teleop.get_input_frame())
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down QuestTeleop")
    except Exception as e:
        logging.error(f"Main process error: {e}")

# python quest_receive.py --receiver-port 10001 --sender-ip 192.168.2.250 --sender-port 10004 --sleep-interval 0.02
import pickle
from typing import Any, Dict

from xdeploy.common.logger_utils import logger


def debugpy_listen(ip: str = "0.0.0.0", port: int = 10092):
    import debugpy

    debugpy.listen((ip, port))
    print(f"Waiting for client to attach {port}...")
    debugpy.wait_for_client()


def dump_data(data: Dict[str, Any], filepath: str) -> None:
    """
    Serialize and save data to a file using pickle.

    Args:
        data: Dictionary containing controllers and sensors data.
            Format: {"controllers": controller_data, "sensors": sensor_data}
        filepath: Path to the file where data will be saved.

    Example:
        data = robot.get()
        dump_data(data, "robot_state.pkl")
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Data saved to {filepath}")


def load_data(filepath: str) -> Dict[str, Any]:
    """
    Load and deserialize data from a file using pickle.

    Args:
        filepath: Path to the file containing serialized data.

    Returns:
        Dictionary containing controllers and sensors data.
            Format: {"controllers": controller_data, "sensors": sensor_data}

    Example:
        data = load_data("robot_state.pkl")
        controller_data = data["controllers"]
        sensor_data = data["sensors"]
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Data loaded from {filepath}")
    return data

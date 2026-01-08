import json

from xdeploy.common.prof_utils import get_and_print_system_info

if __name__ == "__main__":
    # 获取运行环境信息
    info = get_and_print_system_info()
    # 将信息保存到文件
    with open("runtime_env.json", "w") as f:
        json.dump(info, f)

import json
import matplotlib.pyplot as plt
import numpy as np

# 打开 JSON 文件
with open('72.json', 'r') as file:
    data = json.load(file)
    # door_info=data["ev_info"]
    # door_conf=data["ev_conf"]
    door_info = data["door_info"]
    door_conf = data["door_conf"]


def smooth_door_state(door_info, door_conf, window_size=3, threshold=0.8):
    smoothed_door_info = []
    for i in range(len(door_info)):
        if i < window_size // 2 or i >= len(door_info) - window_size // 2:
            smoothed_door_info.append(door_info[i])
        else:
            window = door_info[i - window_size // 2:i + window_size // 2 + 1]
            conf_window = door_conf[i - window_size // 2:i + window_size // 2 + 1]
            if sum(conf_window) / window_size >= threshold:
                smoothed_door_info.append(round(sum(window) / window_size))
            else:
                smoothed_door_info.append(door_info[i])
    return smoothed_door_info

# door_info=smooth_door_state(door_info,door_conf,threshold=0.8)

# 绘制原始曲线
# plt.plot(door_info, label='Original')


# 绘制平滑后的曲线
smoothed_door_info = smooth_door_state(door_info, door_conf, window_size=3, threshold=0.5)
# plt.plot(smoothed_door_info, label='Smoothed')
# plt.legend()
# plt.show()

# # 设置窗口大小和判定阈值
# windows = [10, 20, 30]  # 窗口大小
# thresholds = [0.1, 0.2, 0.3]  # 判定阈值
#
# # 生成门状态曲线数据
# door_status = np.random.randint(0, 2, size=100)
#
# # 判断门是否开启
# for window in windows:
#     for threshold in thresholds:
#         # 对门状态曲线进行滑动窗口处理
#         for i in range(len(door_status) - window):
#             window_data = door_status[i:i+window]
#             if sum(window_data) / window < threshold:
#                 print("门已开启！")
#                 break

print(smoothed_door_info)


# def detect_door_status(door_info, frame_rate=25, window_size=30, open_threshold=0.8, close_threshold=0.2):
#     door_status = []  # 门状态数组
#     open_time = []  # 门打开时间点
#     close_time = []  # 门关闭时间点
#
#     # 滑动窗口计算门状态
#     for i in range(len(door_info) - window_size):
#         window_data = door_info[i:i + window_size]
#         status = sum(window_data) / window_size
#         door_status.append(status)
#
#         # 判断门是否打开
#         if status < open_threshold:
#             open_time.append(round((i + window_size) / frame_rate, 2))
#
#         # 判断门是否关闭
#         if status > close_threshold:
#             close_time.append(round((i + window_size) / frame_rate, 2))
#
#     return [open_time, close_time]

def detect_door_status(door_info, frame_rate=25, window_size=30, open_threshold=0.8, close_threshold=0.8):
    open_time = []  # 门打开时间点
    close_time = []  # 门关闭时间点
    is_open = door_info[0] < open_threshold  # 当前门状态是否为开启

    # 滑动窗口计算门状态
    for i in range(len(door_info) - window_size):
        window_data = door_info[i:i + window_size]
        close_ratio = sum(window_data) / window_size
        open_ratio = 1 - close_ratio

        # 判断门是否打开
        if not is_open and open_ratio >= open_threshold:
            open_index = window_data.index(0)
            open_time.append(round((i + open_index) / frame_rate, 2))
            is_open = True

        # 判断门是否关闭
        if is_open and close_ratio >= close_threshold:
            close_index = window_data.index(1)
            close_time.append(round((i + close_index) / frame_rate, 2))
            is_open = False

    return [open_time, close_time]


state=detect_door_status(smoothed_door_info)
print(state)
openState=state[0]
closeState=state[1]
print("开门的时间为:{}".format(openState))
print("关门的时间为:{}".format(closeState))


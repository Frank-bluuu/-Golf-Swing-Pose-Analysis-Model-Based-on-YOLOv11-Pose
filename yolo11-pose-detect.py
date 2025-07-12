# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
import json
import math
import os
from collections import deque

# --- 配置 ---
VIDEO_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\video\测试快动作.mp4'
# VIDEO_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\video\测试慢动作 (online-video-cutter.com).mp4'  # 输入视频文件路径
OUTPUT_VIDEO_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\output\output_golf_pose_video.mp4'  # 输出带标注的视频文件路径
OUTPUT_JSON = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\output\pose_data.json'  # 输出关键点和角度数据的文件
MODEL_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\新数据-loss修改-m模型-C2DA-200epoch-prune-finetune.pt'  # YOLOv11 姿态估计模型路径
CONFIDENCE_THRESHOLD = 0.5  # 用于计算角度和绘制的关键点置信度阈值

# --- 关键点显示配置 ---
# 设置为 True 显示该关键点，False 则不显示
SHOW_KEYPOINTS = {
    "nose": False, "left_eye": False, "right_eye": False, "left_ear": False, "right_ear": False,
    "left_shoulder": True, "right_shoulder": True, "left_elbow": True, "right_elbow": True,
    "left_wrist": True, "right_wrist": True, "middle_club": True, "head_club": True,
    "left_hip": True, "right_hip": True, "left_knee": True, "right_knee": True, "left_ankle": True, "right_ankle": True
}

# --- 骨架连线显示配置 ---
# 设置为 True 显示该连线，False 则不显示
SHOW_SKELETON = {
    "left_ankle_to_left_knee": True, "left_knee_to_left_hip": True, "right_ankle_to_right_knee": True,
    "right_knee_to_right_hip": True, "left_hip_to_right_hip": True, "left_shoulder_to_left_hip": True,
    "right_shoulder_to_right_hip": True, "left_shoulder_to_right_shoulder": True, "left_shoulder_to_left_elbow": True, "right_shoulder_to_right_elbow": True,
    "left_elbow_to_left_wrist": True, "right_elbow_to_right_wrist": True, "middle_club_to_head_club": True, "left_wrist_to_middle_club": True,
    "left_shoulder_to_left_ankle": True, "right_shoulder_to_left_ankle": True,
}

# --- 角度显示配置 ---
# 设置为 True 显示该角度，False 则不显示
SHOW_ANGLES = {
    "right_hip_angle": False, "right_knee_angle": False, "right_elbow_angle": False, "right_shoulder_angle": False,
    "left_hip_angle": False, "left_knee_angle": False, "left_elbow_angle": True, "left_shoulder_angle": False,
    "club_angle": True, "left_wrist_angle": True  # 新增：显示左手腕角度
}

# --- 角度颜色配置 ---
ANGLE_COLORS = {
    "right_hip_angle": (0, 255, 255),  # Yellow
    "right_knee_angle": (255, 255, 0),  # Cyan
    "right_elbow_angle": (255, 0, 255),  # Magenta
    "right_shoulder_angle": (0, 165, 255),  # Orange
    "left_hip_angle": (0, 255, 0),  # Green
    "left_knee_angle": (255, 0, 0),  # Blue
    "left_elbow_angle": (128, 0, 128),  # Purple
    "left_shoulder_angle": (0, 128, 128),  # Teal
    "club_angle": (255, 192, 203),  # Pink
    "left_wrist_angle": (255, 100, 0)  # 新增：左手腕角度颜色（橙色偏红）
}

# 角度名称缩写
ANGLE_SHORT_NAMES = {
    "right_hip_angle": "RH",
    "right_knee_angle": "RK",
    "right_elbow_angle": "RE",
    "right_shoulder_angle": "RS",
    "left_hip_angle": "LH",
    "left_knee_angle": "LK",
    "left_elbow_angle": "LE",
    "left_shoulder_angle": "LS",
    "club_angle": "CLUB",
    "left_wrist_angle": "LW"  # 新增：左手腕角度缩写
}

# --- 关键点索引 (COCO keypoints) ---
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "middle_club", "head_club",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# 将索引映射到名称，方便查找
kp_name_to_index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

RIGHT_SHOULDER_IDX = kp_name_to_index["right_shoulder"]
RIGHT_ELBOW_IDX = kp_name_to_index["right_elbow"]
RIGHT_WRIST_IDX = kp_name_to_index["right_wrist"]
RIGHT_HIP_IDX = kp_name_to_index["right_hip"]
RIGHT_KNEE_IDX = kp_name_to_index["right_knee"]
RIGHT_ANKLE_IDX = kp_name_to_index["right_ankle"]
LEFT_SHOULDER_IDX = kp_name_to_index["left_shoulder"]
LEFT_ELBOW_IDX = kp_name_to_index["left_elbow"]
LEFT_WRIST_IDX = kp_name_to_index["left_wrist"]
LEFT_HIP_IDX = kp_name_to_index["left_hip"]
LEFT_KNEE_IDX = kp_name_to_index["left_knee"]
LEFT_ANKLE_IDX = kp_name_to_index["left_ankle"]
MIDDLE_CLUB_IDX = kp_name_to_index["middle_club"]
HEAD_CLUB_IDX = kp_name_to_index["head_club"]

# 定义关键点连接关系 (用于绘制骨架)
SKELETON = [
    (kp_name_to_index["left_ankle"], kp_name_to_index["left_knee"], "left_ankle_to_left_knee"),
    (kp_name_to_index["left_knee"], kp_name_to_index["left_hip"], "left_knee_to_left_hip"),
    (kp_name_to_index["right_ankle"], kp_name_to_index["right_knee"], "right_ankle_to_right_knee"),
    (kp_name_to_index["right_knee"], kp_name_to_index["right_hip"], "right_knee_to_right_hip"),
    (kp_name_to_index["left_hip"], kp_name_to_index["right_hip"], "left_hip_to_right_hip"),
    (kp_name_to_index["left_shoulder"], kp_name_to_index["left_hip"], "left_shoulder_to_left_hip"),
    (kp_name_to_index["right_shoulder"], kp_name_to_index["right_hip"], "right_shoulder_to_right_hip"),
    (kp_name_to_index["left_shoulder"], kp_name_to_index["right_shoulder"], "left_shoulder_to_right_shoulder"),
    (kp_name_to_index["left_shoulder"], kp_name_to_index["left_elbow"], "left_shoulder_to_left_elbow"),
    (kp_name_to_index["right_shoulder"], kp_name_to_index["right_elbow"], "right_shoulder_to_right_elbow"),
    (kp_name_to_index["left_elbow"], kp_name_to_index["left_wrist"], "left_elbow_to_left_wrist"),
    (kp_name_to_index["right_elbow"], kp_name_to_index["right_wrist"], "right_elbow_to_right_wrist"),
    (kp_name_to_index["middle_club"], kp_name_to_index["head_club"], "middle_club_to_head_club"),
    (kp_name_to_index["left_wrist"], kp_name_to_index["middle_club"], "left_wrist_to_middle_club"),
    (kp_name_to_index["left_shoulder"], kp_name_to_index["left_ankle"], "left_shoulder_to_left_ankle"), # 用于姿态纠正
    (kp_name_to_index["right_shoulder"], kp_name_to_index["left_ankle"], "right_shoulder_to_left_ankle"), # 用于姿态纠正
]

# 定义所有要计算的角度（关节）
ANGLE_DEFINITIONS = {
    "right_hip_angle": (RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX, RIGHT_KNEE_IDX),
    "right_knee_angle": (RIGHT_HIP_IDX, RIGHT_KNEE_IDX, RIGHT_ANKLE_IDX),
    "right_elbow_angle": (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX),
    "right_shoulder_angle": (RIGHT_ELBOW_IDX, RIGHT_SHOULDER_IDX, RIGHT_HIP_IDX),
    "left_hip_angle": (LEFT_SHOULDER_IDX, LEFT_HIP_IDX, LEFT_KNEE_IDX),
    "left_knee_angle": (LEFT_HIP_IDX, LEFT_KNEE_IDX, LEFT_ANKLE_IDX),
    "left_elbow_angle": (LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, LEFT_WRIST_IDX),
    "left_shoulder_angle": (LEFT_ELBOW_IDX, LEFT_SHOULDER_IDX, LEFT_HIP_IDX),
    "club_angle": (LEFT_WRIST_IDX, MIDDLE_CLUB_IDX, HEAD_CLUB_IDX),
    "left_wrist_angle": (LEFT_ELBOW_IDX, LEFT_WRIST_IDX, MIDDLE_CLUB_IDX)  # 新增：左手腕角度定义（肘-腕-杆）
}

# --- 定义每个角度的正方向 ---
# 每个角度的正方向由参考向量和旋转方向决定
# 格式: (参考向量, 旋转方向)
# 参考向量: 'horizontal' 或 'vertical'
# 旋转方向: 'cw' (顺时针) 或 'ccw' (逆时针)
ANGLE_DIRECTIONS = {
    "right_hip_angle": ('horizontal', 'ccw'),  # 右髋关节角度，逆时针为正
    "right_knee_angle": ('vertical', 'ccw'),  # 右膝关节角度，逆时针为正
    "right_elbow_angle": ('horizontal', 'ccw'),  # 右肘关节角度，逆时针为正
    "right_shoulder_angle": ('horizontal', 'ccw'),  # 右肩关节角度，逆时针为正
    "left_hip_angle": ('horizontal', 'cw'),  # 左髋关节角度，顺时针为正
    "left_knee_angle": ('vertical', 'cw'),  # 左膝关节角度，顺时针为正
    "left_elbow_angle": ('horizontal', 'cw'),  # 左肘关节角度，顺时针为正
    "left_shoulder_angle": ('horizontal', 'cw'),  # 左肩关节角度，顺时针为正
    "club_angle": ('horizontal', 'ccw'),  # 球杆角度，逆时针为正
    "left_wrist_angle": ('horizontal', 'ccw')  # 新增：左手腕角度方向（逆时针为正）
}

# --- 初始化模型 ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print(f"Ensure the model file '{MODEL_PATH}' exists and required libraries are installed.")
    exit()

# --- 打开视频文件 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_PATH}")
    exit()

# --- 获取视频信息 ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Info: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} Frames")

# --- 初始化视频写入器 (VideoWriter) ---
# 使用 MP4V 编解码器，可以根据需要更改 (例如 'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    print(f"Error initializing VideoWriter for path: {OUTPUT_VIDEO_PATH}")
    cap.release()
    exit()

print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

# --- 存储结果的列表 ---
all_frame_data = []

# --- 绘图颜色 ---
POINT_COLOR = (0, 0, 255)  # BGR: Red for points
LINE_COLOR = (0, 255, 0)  # BGR: Green for lines
TEXT_COLOR = (255, 0, 255)  # BGR: Magenta for text

# 5mm偏移量（假设1mm≈4像素）
OFFSET_MM = 5
OFFSET_PX = OFFSET_MM * 4  # 20像素偏移

# --- 动作关键帧检测状态机 ---
# 状态: 0=预备动作, 1=上杆顶点, 2=击球, 3=收杆, 4=完成
current_state = 0
detected_states = []  # 存储已检测到的状态和帧索引

# 新增：其他关键帧检测标志
other_keyframe_recorded = False  # 是否已记录其他关键帧

# 位移阈值 (用于检测速度接近零)
DISPLACEMENT_THRESHOLD = 20  # 像素
WINDOW_SIZE = 20  # 用于速度计算的帧窗口大小

# 存储历史位置 (用于速度计算)
head_club_history = deque(maxlen=WINDOW_SIZE)
middle_club_history = deque(maxlen=WINDOW_SIZE)

# 提示词样式
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 2
TEXT_THICKNESS = 5
TEXT_COLOR = (0, 0, 0)  # 黑色
TEXT_BG_COLOR = (255, 255, 255)  # 白色背景


# --- 函数：计算三个点构成的角度（带方向）---
def calculate_angle_between_points(p1, p2, p3, angle_name=None):
    """
    计算由 p1-p2 和 p3-p2 两条线段在点 p2 形成的夹角，带方向。
    :param p1: 第一个点坐标 [x, y]
    :param p2: 中间点/顶点坐标 [x, y]
    :param p3: 第三个点坐标 [x, y]
    :param angle_name: 角度名称，用于确定正方向
    :return: 带方向的角度 (0-360 度)，如果无法计算则返回 None
    """
    # 创建向量 v1 = p1 - p2, v2 = p3 - p2
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # 计算向量的模长 (magnitude)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # 检查模长是否为零，避免除零错误
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None

    # 计算向量的点积和叉积
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # 计算无方向角度 (0-180度)
    cos_angle = np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    # 如果没有指定角度名称，返回无方向角度
    if angle_name is None:
        return angle_deg

    # 获取该角度的正方向配置
    reference_vector, rotation_direction = ANGLE_DIRECTIONS.get(angle_name, ('horizontal', 'ccw'))

    # 确定参考向量
    if reference_vector == 'horizontal':
        ref_vec = np.array([1, 0])  # 水平向右
    else:  # vertical
        ref_vec = np.array([0, -1])  # 垂直向上（图像坐标系中y轴向下为正）

    # 计算v1相对于参考向量的角度
    dot_ref = np.dot(v1, ref_vec)
    cross_ref = v1[0] * ref_vec[1] - v1[1] * ref_vec[0]
    angle_from_ref = np.degrees(np.arctan2(cross_ref, dot_ref))
    angle_from_ref = angle_from_ref % 360  # 确保在0-360度范围内

    # 根据旋转方向确定最终角度
    if rotation_direction == 'ccw':  # 逆时针为正
        # 如果v2在v1的逆时针方向，则角度增加
        if cross_product < 0:  # 叉积为负表示v2在v1的逆时针方向
            final_angle = (angle_from_ref + angle_deg) % 360
        else:
            final_angle = (angle_from_ref - angle_deg) % 360
    else:  # cw，顺时针为正
        if cross_product > 0:  # 叉积为正表示v2在v1的顺时针方向
            final_angle = (angle_from_ref + angle_deg) % 360
        else:
            final_angle = (angle_from_ref - angle_deg) % 360

    # 四舍五入到小数点后一位
    if final_angle is not None:
        final_angle = round(final_angle, 1)

    return final_angle


# --- 函数：计算两个向量的角度 ---
def calculate_vector_angle(v1, v2):
    """
    计算两个向量之间的角度（0-360度）
    :param v1: 向量1 [x, y]
    :param v2: 向量2 [x, y]
    :return: 角度值（度）
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle_rad = np.arctan2(det, dot)
    angle_deg = np.degrees(angle_rad)
    return angle_deg if angle_deg >= 0 else angle_deg + 360


# --- 函数：绘制角度圆弧 ---
def draw_angle_arc(image, p1, p2, p3, angle_value, color, radius=20):
    """
    在顶点p2处绘制角度圆弧
    :param image: 要绘制的图像
    :param p1: 第一个点
    :param p2: 顶点
    :param p3: 第三个点
    :param angle_value: 角度值
    :param color: 圆弧颜色
    :param radius: 圆弧半径
    """
    # 确保angle_value是数值类型
    if angle_value is None:
        return

    # 计算向量
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # 计算向量角度
    angle1 = calculate_vector_angle([1, 0], v1)  # 相对于x轴的角度
    angle2 = calculate_vector_angle([1, 0], v2)

    # 确定起始和结束角度
    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)

    # 如果角度差大于180度，绘制较小的圆弧
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360

    # 绘制圆弧
    center = (int(p2[0]), int(p2[1]))
    cv2.ellipse(image, center, (radius, radius), 0, start_angle, end_angle, color, 2)

    # 在圆弧中点显示角度值（只显示整数部分）
    mid_angle = (start_angle + end_angle) / 2
    text_x = int(p2[0] + radius * np.cos(np.radians(mid_angle)))
    text_y = int(p2[1] + radius * np.sin(np.radians(mid_angle)))

    # 使用整数格式显示角度，确保不出现小数部分
    cv2.putText(image, f"{int(angle_value)}°", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# --- 函数：在图像上绘制姿态和角度 ---
def draw_pose_on_image(image, keypoints, confidences, angles):
    """在传入的图像上绘制关键点、骨架和角度"""
    # 绘制骨架连线
    for p1_idx, p2_idx, connection_name in SKELETON:
        # 检查是否应该显示该连线
        if not SHOW_SKELETON.get(connection_name, True):
            continue

        if confidences[p1_idx] > CONFIDENCE_THRESHOLD and confidences[p2_idx] > CONFIDENCE_THRESHOLD:
            p1 = tuple(map(int, keypoints[p1_idx]))  # 转为整数坐标元组
            p2 = tuple(map(int, keypoints[p2_idx]))
            # 确保坐标在图像范围内 (虽然 YOLO 通常不会超出太多)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(image, p1, p2, LINE_COLOR, 2)

    # 绘制关键点
    for i, kp in enumerate(keypoints):
        kp_name = KEYPOINT_NAMES[i]
        # 检查是否应该显示该关键点
        if not SHOW_KEYPOINTS.get(kp_name, True):
            continue

        if confidences[i] > CONFIDENCE_THRESHOLD:
            center = tuple(map(int, kp))
            if center[0] > 0 and center[1] > 0:
                cv2.circle(image, center, 5, POINT_COLOR, -1)  # -1 表示实心圆

    # 绘制所有角度和圆弧
    for angle_name, (idx1, idx2, idx3) in ANGLE_DEFINITIONS.items():
        if not SHOW_ANGLES.get(angle_name, True):
            continue

        angle_value = angles.get(angle_name)
        if angle_value is not None:
            # 获取三个点
            p1 = keypoints[idx1]
            p2 = keypoints[idx2]  # 顶点
            p3 = keypoints[idx3]

            # 检查置信度
            if (confidences[idx1] > CONFIDENCE_THRESHOLD and
                    confidences[idx2] > CONFIDENCE_THRESHOLD and
                    confidences[idx3] > CONFIDENCE_THRESHOLD):
                # 获取该角度的颜色
                color = ANGLE_COLORS.get(angle_name, (0, 255, 255))  # 默认为黄色

                # 绘制角度圆弧
                draw_angle_arc(image, p1, p2, p3, angle_value, color)

                # 在顶点附近显示角度名称（偏移5mm）
                # 使用缩写名称以减少重叠
                short_name = ANGLE_SHORT_NAMES.get(angle_name, angle_name.split('_')[0])

                # 计算偏移方向（沿y轴向下偏移）
                offset_x = 0
                offset_y = OFFSET_PX

                # 绘制角度名称（偏移5mm）
                cv2.putText(image, short_name,
                            (int(p2[0]) + offset_x, int(p2[1]) + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# --- 函数：计算两点间欧氏距离 ---
def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# --- 函数：计算球杆与水平面夹角 ---
def calculate_club_horizontal_angle(mid_club, head_club):
    """
    计算球杆与水平面的夹角 (0-180度)
    :param mid_club: 球杆中点坐标 (x, y)
    :param head_club: 球杆头坐标 (x, y)
    :return: 角度值 (度)
    """
    dx = head_club[0] - mid_club[0]
    dy = head_club[1] - mid_club[1]

    # 计算与水平线的夹角 (0-180度)
    angle_rad = math.atan2(abs(dy), abs(dx))  # 使用绝对值得到0-90度
    angle_deg = math.degrees(angle_rad)

    # 如果dy为负（球杆向上），则角度在0-90度
    # 如果dy为正（球杆向下），则角度在90-180度
    if dy < 0:
        return angle_deg
    else:
        return 180 - angle_deg


# --- 逐帧处理视频 ---
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频结束或读取错误

    frame_count += 1
    # 减少打印频率，例如每 30 帧打印一次
    if frame_count % 30 == 0:
        print(f"Processing frame {frame_count}/{total_frames}")

    # --- 使用 YOLOv11 进行姿态估计 ---
    # 注意：YOLO 处理的是原始帧 `frame`
    results = model(frame, verbose=False)  # verbose=False 减少控制台输出

    # 创建一个副本用于绘制，保留原始帧用于可能的其他处理
    annotated_frame = frame.copy()

    frame_data = {
        "frame_index": frame_count,
        "persons": []
    }

    # 当前帧的关键点数据 (用于动作检测)
    current_kps = None
    current_confs = None
    current_angles = {}
    head_club_pos = None
    middle_club_pos = None

    # --- 处理检测结果 ---
    if results and results[0].keypoints and results[0].keypoints.data.shape[0] > 0 and results[
        0].keypoints.xy is not None and results[0].keypoints.conf is not None:
        keypoints_data = results[0].keypoints.xy.cpu().numpy()
        confidences = results[0].keypoints.conf.cpu().numpy()

        for person_idx in range(keypoints_data.shape[0]):
            person_keypoints = keypoints_data[person_idx]
            person_confs = confidences[person_idx]

            # --- 计算所有角度 ---
            person_angles = {}
            for angle_name, (idx1, idx2, idx3) in ANGLE_DEFINITIONS.items():
                # 检查置信度
                if (person_confs[idx1] > CONFIDENCE_THRESHOLD and
                        person_confs[idx2] > CONFIDENCE_THRESHOLD and
                        person_confs[idx3] > CONFIDENCE_THRESHOLD):
                    p1 = person_keypoints[idx1]
                    p2 = person_keypoints[idx2]  # 顶点
                    p3 = person_keypoints[idx3]
                    angle = calculate_angle_between_points(p1, p2, p3, angle_name)
                    person_angles[angle_name] = angle

            # --- 存储数据 ---
            # 确保角度值保留一位小数
            rounded_angles = {}
            for angle_name, angle_value in person_angles.items():
                if angle_value is not None:
                    rounded_angles[angle_name] = int(angle_value)
                else:
                    rounded_angles[angle_name] = None

            person_data = {
                "person_id": person_idx,
                "keypoints": person_keypoints.tolist(),
                "confidences": [round(conf, 4) for conf in person_confs.tolist()],  # 置信度保留4位小数
                "angles": rounded_angles  # 使用保留一位小数的角度值
            }
            frame_data["persons"].append(person_data)

            # 只处理第一个检测到的人 (假设视频中只有一个高尔夫球员)
            if person_idx == 0:
                # 绘制当前人物的姿态和角度
                draw_pose_on_image(annotated_frame, person_keypoints, person_confs, person_angles)

                # 存储当前帧数据 (用于动作检测)
                current_kps = person_keypoints
                current_confs = person_confs
                current_angles = person_angles

                # 存储球杆位置 (如果置信度足够)
                if person_confs[HEAD_CLUB_IDX] > CONFIDENCE_THRESHOLD:
                    head_club_pos = tuple(person_keypoints[HEAD_CLUB_IDX])
                if person_confs[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD:
                    middle_club_pos = tuple(person_keypoints[MIDDLE_CLUB_IDX])

    # 将当前帧的数据添加到总列表
    all_frame_data.append(frame_data)

    # --- 更新位置历史 ---
    head_club_history.append(head_club_pos)
    middle_club_history.append(middle_club_pos)

    # --- 动作关键帧检测 ---
    # 只处理第一个检测到的人
    if current_kps is not None and current_confs is not None:
        # 获取髋关节位置 (左右髋关节的平均值)
        if (current_confs[LEFT_HIP_IDX] > CONFIDENCE_THRESHOLD and
                current_confs[RIGHT_HIP_IDX] > CONFIDENCE_THRESHOLD):
            left_hip = current_kps[LEFT_HIP_IDX]
            right_hip = current_kps[RIGHT_HIP_IDX]
            hip_avg = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        else:
            hip_avg = None

        # 状态0: 检测预备动作
        if current_state == 0:
            if (current_confs[LEFT_WRIST_IDX] > CONFIDENCE_THRESHOLD and
                    hip_avg is not None and
                    "left_wrist_angle" in current_angles):

                left_wrist_y = current_kps[LEFT_WRIST_IDX][1]
                left_wrist_angle = current_angles["left_wrist_angle"]

                # 条件1: 左手腕在髋关节上方
                # 条件2: 左手腕和球杆几乎成直线 (170-190度)
                if left_wrist_y > hip_avg[1] and 170 < left_wrist_angle < 360:
                    current_state = 1
                    detected_states.append(("Preparation", frame_count))
                    print(f"Frame {frame_count}: Preparation detected")
                    other_keyframe_recorded = False  # 重置其他关键帧标志
            else:
                # 新增：未检测到预备动作关键帧时判定为其他关键帧
                if not other_keyframe_recorded:
                    detected_states.append(("Other", frame_count))
                    print(f"Frame {frame_count}: Other detected (no preparation)")
                    other_keyframe_recorded = True

        # 状态1: 检测上杆顶点
        elif current_state == 1:
            # 条件1: 左手腕角度小于50
            cond1 = "left_wrist_angle" in current_angles and current_angles["left_wrist_angle"] < 50

            # # 条件2: 20帧内球杆头速度接近零
            # cond2 = False
            # if len(head_club_history) >= WINDOW_SIZE:
            #     # 找到窗口中第一个有效位置
            #     first_valid_pos = None
            #     for pos in head_club_history:
            #         if pos is not None:
            #             first_valid_pos = pos
            #             break
            #
            #     # 计算位移
            #     if first_valid_pos is not None and head_club_pos is not None:
            #         displacement = calculate_distance(first_valid_pos, head_club_pos)
            #         cond2 = displacement < DISPLACEMENT_THRESHOLD

            # 条件3: 球杆与水平面夹角接近水平 (0-5度 或 175-180度)
            cond3 = False
            if head_club_pos is not None and middle_club_pos is not None:
                club_angle = calculate_club_horizontal_angle(middle_club_pos, head_club_pos)
                cond3 = club_angle < 10 or club_angle > 170

            # if cond1 and cond2 and cond3:
            if cond1 and cond3:
                current_state = 2
                detected_states.append(("Top of Backswing", frame_count))
                print(f"Frame {frame_count}: Top of Backswing detected")

        # 状态2: 检测击球
        elif current_state == 2:
            if (current_confs[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD and
                    hip_avg is not None):

                middle_club_y = current_kps[MIDDLE_CLUB_IDX][1]

                # 条件: 球杆中点低于髋关节
                if middle_club_y > hip_avg[1]:
                    current_state = 3
                    detected_states.append(("Impact", frame_count))
                    print(f"Frame {frame_count}: Impact detected")

        # 状态3: 检测收杆
        elif current_state == 3:

            # # 先初始化位移条件为False
            # displacement_condition = False
            # # 检查位移条件
            # if len(middle_club_history) >= WINDOW_SIZE:
            #     # 找到窗口中第一个有效位置
            #     first_valid_pos = None
            #     for pos in middle_club_history:
            #         if pos is not None:
            #             first_valid_pos = pos
            #             break
            #
            #     # 计算位移
            #     if first_valid_pos is not None and middle_club_pos is not None:
            #         displacement = calculate_distance(first_valid_pos, middle_club_pos)
            #         displacement_condition = displacement < DISPLACEMENT_THRESHOLD

            # 条件：检查左手肘角度条件
            elbow_angle_condition = False
            if "left_elbow_angle" in current_angles and current_angles["left_elbow_angle"] is not None:
                elbow_angle_condition = 260 <= current_angles["left_elbow_angle"] <= 320
            # print(frame_count,current_angles)

            # 条件：球杆与水平面夹角在指定范围内
            club_angle_condition = False
            if head_club_pos is not None and middle_club_pos is not None:
                club_angle = calculate_club_horizontal_angle(middle_club_pos, head_club_pos)
                club_angle_condition = 150 < club_angle
            # print(frame_count,club_angle)

            # 条件：肩膀的连线与水平面夹角几乎平行
            shoulder_horizontal_condition = False
            if (current_confs[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                    current_confs[RIGHT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD):
                left_shoulder = current_kps[LEFT_SHOULDER_IDX]
                right_shoulder = current_kps[RIGHT_SHOULDER_IDX]

                # 计算肩膀连线向量
                shoulder_vec = [right_shoulder[0] - left_shoulder[0],
                                right_shoulder[1] - left_shoulder[1]]

                # 计算与水平线的夹角 (0-90度)
                angle_rad = math.atan2(abs(shoulder_vec[1]), abs(shoulder_vec[0]))
                shoulder_angle_deg = math.degrees(angle_rad)

                # 如果夹角小于10度或大于170度（接近水平）
                shoulder_horizontal_condition = (shoulder_angle_deg <= 10 or
                                                 shoulder_angle_deg >= 170)

            # 同时满足以上两个条件
            if elbow_angle_condition and club_angle_condition and shoulder_horizontal_condition:
                current_state = 4
                detected_states.append(("Finish", frame_count))
                print(f"Frame {frame_count}: Finish detected")

        # 状态4: 完成
        elif current_state == 4:
            # 新增：超出收杆关键帧的条件时判定为其他关键帧
            if not other_keyframe_recorded:
                detected_states.append(("Other", frame_count))
                print(f"Frame {frame_count}: Other detected (beyond finish)")
                other_keyframe_recorded = True

    # --- 在视频顶部显示当前动作提示 ---
    # 获取当前动作的提示词
    current_action_text = ""
    for action, frame_idx in detected_states:
        # 如果是最近检测到的动作 (在当前帧或之前)
        if frame_idx <= frame_count:
            current_action_text = action

    # 绘制提示词背景
    if current_action_text:
        text_size = cv2.getTextSize(current_action_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        text_x = int((frame_width - text_size[0]) / 2)
        text_y = int(text_size[1] * 1.5)

        # 绘制背景矩形
        cv2.rectangle(annotated_frame,
                      (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      TEXT_BG_COLOR, -1)

        # 绘制文本
        cv2.putText(annotated_frame, current_action_text,
                    (text_x, text_y),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

    # --- 将带标注的帧写入输出视频 ---
    out.write(annotated_frame)

# --- 释放资源 ---
cap.release()
out.release()  # 确保释放 VideoWriter

# try:
#     with open(OUTPUT_JSON, 'w') as f:
#         # 确保JSON中的角度值保留一位小数
#         formatted_data = {
#             "video_info": {
#                 "width": frame_width,
#                 "height": frame_height,
#                 "fps": fps,
#                 "total_frames": total_frames
#             },yolo11-pose-detect-neu.py
#             "frames": all_frame_data
#         }
#         json.dump(formatted_data, f, indent=4)
#     print(f"Pose data saved to {OUTPUT_JSON}")
# except Exception as e:
#     print(f"Error saving data to JSON: {e}")

print(f"Processing finished. Annotated video saved to {OUTPUT_VIDEO_PATH}")
print("Detected key frames:")
for action, frame_idx in detected_states:
    print(f"  {action} at frame {frame_idx}")
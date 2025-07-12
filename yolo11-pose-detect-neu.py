# -*- coding: utf-8 -*-
# 指定文件编码为UTF-8，支持中文等特殊字符

# 导入必要的库
import cv2  # OpenCV库，用于图像和视频处理
import numpy as np  # 数值计算库
from ultralytics import YOLO  # YOLO姿态估计模型
import json  # JSON数据格式处理
import math  # 数学运算
import os  # 操作系统接口
from collections import deque  # 双端队列数据结构

# --- 配置部分 ---
# 视频文件路径
# VIDEO_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\video\测试慢动作 (online-video-cutter.com).mp4'
VIDEO_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\video\测试快动作.mp4'
# 输出视频文件路径
OUTPUT_VIDEO_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\output\output_golf_pose_video.mp4'
# 输出JSON数据文件路径
OUTPUT_JSON = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\output\pose_data.json'
# YOLO模型路径
MODEL_PATH = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\新数据-loss修改-m模型-C2DA-200epoch-prune-finetune.pt'
# 关键帧输出目录
OUTPUT_KEYFRAME_DIR = r'D:\文档\Project\YOLO\prune\ultralytics-yolo11-8.3.9\output\key_frames'
os.makedirs(OUTPUT_KEYFRAME_DIR, exist_ok=True)  # 创建目录（如果不存在）
# 关键点置信度阈值
CONFIDENCE_THRESHOLD = 0.5

# --- 关键点显示配置 ---
# 布尔字典，控制是否显示特定关键点
SHOW_KEYPOINTS = {
    "nose": False, "left_eye": False, "right_eye": False, "left_ear": False, "right_ear": False,
    "left_shoulder": True, "right_shoulder": True, "left_elbow": True, "right_elbow": True,
    "left_wrist": True, "right_wrist": True, "middle_club": True, "head_club": True,
    "left_hip": True, "right_hip": True, "left_knee": True, "right_knee": True, "left_ankle": True, "right_ankle": True
}

# --- 骨架连线显示配置 ---
# 布尔字典，控制是否显示特定骨架连线
SHOW_SKELETON = {
    "left_ankle_to_left_knee": True, "left_knee_to_left_hip": True, "right_ankle_to_right_knee": True,
    "right_knee_to_right_hip": True, "left_hip_to_right_hip": True, "left_shoulder_to_left_hip": True,
    "right_shoulder_to_right_hip": True, "left_shoulder_to_right_shoulder": True,
    "left_shoulder_to_left_elbow": True, "right_shoulder_to_right_elbow": True,
    "left_elbow_to_left_wrist": True, "right_elbow_to_right_wrist": True,
    "middle_club_to_head_club": True, "left_wrist_to_middle_club": True,
    "left_shoulder_to_left_ankle": True, "right_shoulder_to_left_ankle": True,
}

# --- 角度显示配置 ---
# 布尔字典，控制是否显示特定角度
SHOW_ANGLES = {
    "right_hip_angle": False, "right_knee_angle": False, "right_elbow_angle": False, "right_shoulder_angle": False,
    "left_hip_angle": False, "left_knee_angle": False, "left_elbow_angle": True, "left_shoulder_angle": False,
    "club_angle": True, "left_wrist_angle": True
}

# 角度显示颜色配置（BGR格式）
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
    "left_wrist_angle": (255, 100, 0)  # 橙色偏红
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
    "left_wrist_angle": "LW"
}

# --- 关键点索引 ---
# 关键点名称列表
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "middle_club", "head_club",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# 创建名称到索引的映射字典
kp_name_to_index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# 为常用关键点定义常量索引
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

# 定义关键点连接关系（骨架连线）
# 格式：(起点索引, 终点索引, 连线名称)
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
    (kp_name_to_index["left_shoulder"], kp_name_to_index["left_ankle"], "left_shoulder_to_left_ankle"),
    (kp_name_to_index["right_shoulder"], kp_name_to_index["left_ankle"], "right_shoulder_to_left_ankle"),
]

# 定义角度计算的三点组合
# 格式：角度名称: (点A索引, 点B索引, 点C索引)
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
    "left_wrist_angle": (LEFT_ELBOW_IDX, LEFT_WRIST_IDX, MIDDLE_CLUB_IDX)
}

# 角度方向配置
# 格式：角度名称: (参考向量, 旋转方向)
ANGLE_DIRECTIONS = {
    "right_hip_angle": ('horizontal', 'ccw'),  # 水平参考，逆时针为正
    "right_knee_angle": ('vertical', 'ccw'),  # 垂直参考，逆时针为正
    "right_elbow_angle": ('horizontal', 'ccw'),
    "right_shoulder_angle": ('horizontal', 'ccw'),
    "left_hip_angle": ('horizontal', 'cw'),  # 水平参考，顺时针为正
    "left_knee_angle": ('vertical', 'cw'),  # 垂直参考，顺时针为正
    "left_elbow_angle": ('horizontal', 'cw'),
    "left_shoulder_angle": ('horizontal', 'cw'),
    "club_angle": ('horizontal', 'ccw'),
    "left_wrist_angle": ('horizontal', 'ccw')
}

# 初始化YOLO模型
try:
    model = YOLO(MODEL_PATH)  # 加载预训练模型
except Exception as e:
    print(f"Error loading YOLO model: {e}")  # 模型加载失败处理
    exit()  # 退出程序

# 打开视频文件
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():  # 检查视频是否成功打开
    print(f"Error opening video file: {VIDEO_PATH}")
    exit()  # 退出程序

# 获取视频信息
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数

# 打印视频信息
print(f"Video Info: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} Frames")

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编解码器
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():  # 检查写入器是否成功初始化
    print(f"Error initializing VideoWriter for path: {OUTPUT_VIDEO_PATH}")
    cap.release()  # 释放视频捕获对象
    exit()  # 退出程序

# 打印输出路径信息
print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

# 存储每帧结果的列表
all_frame_data = []

# 绘图颜色配置
POINT_COLOR = (0, 0, 255)  # 关键点颜色（红色）
LINE_COLOR = (0, 255, 0)  # 骨架连线颜色（绿色）
TEXT_COLOR = (255, 0, 255)  # 文本颜色（品红色）

# 5mm偏移量（用于角度文本显示位置）
OFFSET_MM = 5
OFFSET_PX = OFFSET_MM * 4  # 转换为像素（假设1mm≈4像素）

# 动作关键帧检测状态机
current_state = 0  # 当前状态：0=预备动作, 1=上杆顶点, 2=击球, 3=收杆, 4=完成
detected_states = []  # 存储已检测到的状态和帧索引
# 关键帧字典，存储四种关键动作的帧号
key_frames = {
    "Preparation": None,  # 预备动作
    "Top of Backswing": None,  # 上杆顶点
    "Impact": None,  # 击球
    "Finish": None  # 收杆
}

# 其他关键帧检测标志
other_keyframe_recorded = False  # 是否已记录其他关键帧

# 位移阈值（用于检测球杆移动）
DISPLACEMENT_THRESHOLD = 20  # 像素
WINDOW_SIZE = 20  # 用于速度计算的帧窗口大小

# 存储历史位置（用于球杆移动检测）
head_club_history = deque(maxlen=WINDOW_SIZE)  # 球杆头位置历史
middle_club_history = deque(maxlen=WINDOW_SIZE)  # 球杆中点位置历史

# 提示词样式配置
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX  # 字体
TEXT_SCALE = 2  # 字体大小
TEXT_THICKNESS = 5  # 字体粗细
TEXT_COLOR = (0, 0, 0)  # 字体颜色（黑色）
TEXT_BG_COLOR = (255, 255, 255)  # 背景颜色（白色）

# 存储差异总结
differences_summary = []  # 用于存储姿势判定差异结果


# --- 函数定义部分 ---

# 计算三个点构成的角度（带方向）
def calculate_angle_between_points(p1, p2, p3, angle_name=None):
    """
    计算由三个点构成的夹角（带方向）
    p1, p2, p3: 三个点的坐标
    angle_name: 角度名称，用于确定正方向
    返回：角度值（0-360度）
    """
    # 计算向量
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # 计算向量模长
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # 检查模长是否为零
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return None  # 无法计算角度

    # 计算点积和叉积
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # 计算无方向角度（0-180度）
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


# 计算两个向量的角度
def calculate_vector_angle(v1, v2):
    """
    计算两个向量之间的角度（0-360度）
    v1, v2: 两个向量
    返回：角度值（度）
    """
    # 计算点积和行列式
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    # 计算角度（弧度）
    angle_rad = np.arctan2(det, dot)
    # 转换为角度
    angle_deg = np.degrees(angle_rad)
    # 确保角度在0-360度范围内
    return angle_deg if angle_deg >= 0 else angle_deg + 360


# 绘制角度圆弧
def draw_angle_arc(image, p1, p2, p3, angle_value, color, radius=20):
    """
    在图像上绘制角度圆弧
    image: 要绘制的图像
    p1, p2, p3: 三个点坐标
    angle_value: 角度值
    color: 圆弧颜色
    radius: 圆弧半径
    """
    if angle_value is None:
        return  # 无角度值则不绘制

    # 计算向量
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # 计算向量相对于x轴的角度
    angle1 = calculate_vector_angle([1, 0], v1)
    angle2 = calculate_vector_angle([1, 0], v2)

    # 确定起始和结束角度
    start_angle = min(angle1, angle2)
    end_angle = max(angle1, angle2)

    # 如果角度差大于180度，绘制较小的圆弧
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360

    # 绘制圆弧
    center = (int(p2[0]), int(p2[1]))  # 圆心位置（顶点）
    cv2.ellipse(image, center, (radius, radius), 0, start_angle, end_angle, color, 2)

    # 在圆弧中点显示角度值（只显示整数部分）
    mid_angle = (start_angle + end_angle) / 2  # 计算中点角度
    # 计算文本位置
    text_x = int(p2[0] + radius * np.cos(np.radians(mid_angle)))
    text_y = int(p2[1] + radius * np.sin(np.radians(mid_angle)))
    # 绘制角度值文本
    cv2.putText(image, f"{int(angle_value)}°", (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# 在图像上绘制姿态和角度
def draw_pose_on_image(image, keypoints, confidences, angles):
    """在图像上绘制关键点、骨架和角度"""
    # 绘制骨架连线
    for p1_idx, p2_idx, connection_name in SKELETON:
        # 检查是否应该显示该连线
        if not SHOW_SKELETON.get(connection_name, True):
            continue  # 跳过不显示的连线

        # 检查关键点置信度
        if confidences[p1_idx] > CONFIDENCE_THRESHOLD and confidences[p2_idx] > CONFIDENCE_THRESHOLD:
            p1 = tuple(map(int, keypoints[p1_idx]))  # 转换为整数坐标
            p2 = tuple(map(int, keypoints[p2_idx]))
            # 确保坐标在图像范围内
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(image, p1, p2, LINE_COLOR, 2)  # 绘制连线

    # 绘制关键点
    for i, kp in enumerate(keypoints):
        kp_name = KEYPOINT_NAMES[i]
        # 检查是否应该显示该关键点
        if not SHOW_KEYPOINTS.get(kp_name, True):
            continue  # 跳过不显示的关键点

        # 检查置信度
        if confidences[i] > CONFIDENCE_THRESHOLD:
            center = tuple(map(int, kp))  # 转换为整数坐标
            # 确保坐标在图像范围内
            if center[0] > 0 and center[1] > 0:
                cv2.circle(image, center, 5, POINT_COLOR, -1)  # 绘制实心圆

    # 绘制所有角度和圆弧
    for angle_name, (idx1, idx2, idx3) in ANGLE_DEFINITIONS.items():
        # 检查是否应该显示该角度
        if not SHOW_ANGLES.get(angle_name, True):
            continue  # 跳过不显示的角度

        angle_value = angles.get(angle_name)  # 获取角度值
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
                # 使用缩写名称以减少重叠
                short_name = ANGLE_SHORT_NAMES.get(angle_name, angle_name.split('_')[0])
                # 计算偏移位置（沿y轴向下偏移）
                offset_x = 0
                offset_y = OFFSET_PX
                # 绘制角度名称
                cv2.putText(image, short_name,
                            (int(p2[0]) + offset_x, int(p2[1]) + offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# 计算两点间欧氏距离
def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# 计算球杆与水平面夹角
def calculate_club_horizontal_angle(mid_club, head_club):
    """
    计算球杆与水平面的夹角 (0-180度)
    mid_club: 球杆中点坐标
    head_club: 球杆头坐标
    返回：角度值 (度)
    """
    dx = head_club[0] - mid_club[0]  # x方向差值
    dy = head_club[1] - mid_club[1]  # y方向差值

    # 计算与水平线的夹角 (0-90度)
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)

    # 根据方向确定最终角度
    if dy < 0:  # 球杆向上
        return angle_deg
    else:  # 球杆向下
        return 180 - angle_deg


# 计算两点连线与水平面的夹角
def calculate_line_horizontal_angle(p1, p2):
    """
    计算两点连线与水平面的夹角
    p1, p2: 两个点坐标
    返回：角度值 (度)
    """
    dx = p2[0] - p1[0]  # x方向差值
    dy = p2[1] - p1[1]  # y方向差值
    return math.degrees(math.atan2(abs(dy), abs(dx)))  # 计算角度


# 保存关键帧并进行姿势判定
def save_and_evaluate_keyframe(frame_idx, frame, keypoints, confidences, angles, action):
    """
    保存关键帧图像并进行姿势判定
    frame_idx: 帧索引
    frame: 原始帧图像
    keypoints: 关键点坐标
    confidences: 关键点置信度
    angles: 角度值字典
    action: 动作名称
    """
    # 创建带标注的帧副本
    annotated_frame = frame.copy()
    # 在图像上绘制姿态和角度
    draw_pose_on_image(annotated_frame, keypoints, confidences, angles)

    # 在顶部显示动作名称
    text = action  # 动作名称
    # 计算文本尺寸
    text_size = cv2.getTextSize(text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
    text_x = int((frame_width - text_size[0]) / 2)  # 水平居中
    text_y = int(text_size[1] * 1.5)  # 垂直位置

    # 绘制背景矩形
    cv2.rectangle(annotated_frame,
                  (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10),
                  TEXT_BG_COLOR, -1)  # -1表示填充矩形

    # 绘制文本
    cv2.putText(annotated_frame, text, (text_x, text_y),
                TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

    # 在下方显示判定信息
    y_offset = text_y + 50  # 起始y位置
    line_height = 40  # 行高

    # 存储当前动作的判定结果
    action_results = []

    # 根据不同的动作进行不同的姿势判定
    if action == "Preparation":  # 预备动作
        # 条件1: 左手肘、左手腕、球杆中点是否在一条直线上
        if (confidences[LEFT_ELBOW_IDX] > CONFIDENCE_THRESHOLD and
                confidences[LEFT_WRIST_IDX] > CONFIDENCE_THRESHOLD and
                confidences[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD):
            # 获取关键点坐标
            elbow = keypoints[LEFT_ELBOW_IDX]
            wrist = keypoints[LEFT_WRIST_IDX]
            club_mid = keypoints[MIDDLE_CLUB_IDX]

            # 计算向量
            vec1 = [elbow[0] - wrist[0], elbow[1] - wrist[1]]  # 手腕到肘部向量
            vec2 = [club_mid[0] - wrist[0], club_mid[1] - wrist[1]]  # 手腕到球杆中点向量

            # 计算夹角
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]  # 点积
            magnitude1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)  # 向量1模长
            magnitude2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)  # 向量2模长

            if magnitude1 > 0 and magnitude2 > 0:
                # 计算余弦值
                cos_theta = dot_product / (magnitude1 * magnitude2)
                # 计算角度（确保余弦值在有效范围内）
                angle = math.degrees(math.acos(max(min(cos_theta, 1), -1)))

                # 判定是否接近180度（允许5度偏差）
                deviation = abs(angle - 180)  # 计算偏差
                is_correct = deviation <= 5  # 是否满足条件
                color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

                # 生成状态文本
                status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
                text_line = f"Elbow-Wrist-Club Angle: {angle:.1f}° ({status})"
                # 在图像上绘制文本
                cv2.putText(annotated_frame, text_line, (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y_offset += line_height  # 移动到下一行位置

                # 存储结果
                action_results.append(("Elbow-Wrist-Club Alignment", is_correct, deviation))
            else:
                # 向量模长为零的情况
                text_line = "Elbow-Wrist-Club: Not enough data"
                cv2.putText(annotated_frame, text_line, (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += line_height
                action_results.append(("Elbow-Wrist-Club Alignment", False, None))
        else:
            # 关键点未检测到的情况
            text_line = "Elbow-Wrist-Club: Keypoints not detected"
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += line_height
            action_results.append(("Elbow-Wrist-Club Alignment", False, None))

        # 条件2: 左肩和左脚踝连线是否垂直
        if (confidences[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
            # 获取关键点坐标
            shoulder = keypoints[LEFT_SHOULDER_IDX]
            ankle = keypoints[LEFT_ANKLE_IDX]

            # 计算连线与水平面的夹角
            angle = calculate_line_horizontal_angle(shoulder, ankle)
            deviation = abs(angle - 90)  # 计算与90度的偏差
            is_correct = deviation <= 5  # 是否满足条件
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

            # 生成状态文本
            status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
            text_line = f"Shoulder-Ankle Vertical: {angle:.1f}° ({status})"
            # 在图像上绘制文本
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += line_height

            # 存储结果
            action_results.append(("Shoulder-Ankle Vertical", is_correct, deviation))
        else:
            # 关键点未检测到的情况
            text_line = "Shoulder-Ankle: Keypoints not detected"
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += line_height
            action_results.append(("Shoulder-Ankle Vertical", False, None))

    # 上杆顶点动作的判定
    elif action == "Top of Backswing":
        # 条件1: 球杆是否平行于水平面
        if (confidences[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD and
                confidences[HEAD_CLUB_IDX] > CONFIDENCE_THRESHOLD):
            # 获取关键点坐标
            mid_club = keypoints[MIDDLE_CLUB_IDX]
            head_club = keypoints[HEAD_CLUB_IDX]

            # 计算球杆与水平面的夹角
            club_angle = calculate_club_horizontal_angle(mid_club, head_club)
            # 计算与水平线（0度或180度）的最小偏差
            deviation = min(abs(club_angle - 0), abs(club_angle - 180))
            is_correct = deviation <= 10  # 允许10度偏差
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

            # 生成状态文本
            status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
            text_line = f"Club Horizontal Angle: {club_angle:.1f}° ({status})"
            # 在图像上绘制文本
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += line_height

            # 存储结果
            action_results.append(("Club Horizontal", is_correct, deviation))
        else:
            # 关键点未检测到的情况
            text_line = "Club Horizontal: Keypoints not detected"
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += line_height
            action_results.append(("Club Horizontal", False, None))

        # 条件2: 左肩x坐标是否在脚踝中间
        if (confidences[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD and
                confidences[RIGHT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
            # 获取关键点x坐标
            shoulder_x = keypoints[LEFT_SHOULDER_IDX][0]
            left_ankle_x = keypoints[LEFT_ANKLE_IDX][0]
            right_ankle_x = keypoints[RIGHT_ANKLE_IDX][0]

            # 计算脚踝中心点x坐标
            ankle_center = (left_ankle_x + right_ankle_x) / 2
            # 计算阈值（脚踝间距的10%）
            threshold = abs(right_ankle_x - left_ankle_x) * 0.1
            # 计算与中心的偏差
            deviation = abs(shoulder_x - ankle_center)
            is_correct = deviation <= threshold  # 是否满足条件
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

            # 生成状态文本
            status = "Correct" if is_correct else f"Deviation: {deviation:.1f}px"
            text_line = f"Shoulder X Position: {shoulder_x:.1f}, Center: {ankle_center:.1f} ({status})"
            # 在图像上绘制文本
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += line_height

            # 存储结果
            action_results.append(("Shoulder X Alignment", is_correct, deviation))
        else:
            # 关键点未检测到的情况
            text_line = "Shoulder X: Keypoints not detected"
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += line_height
            action_results.append(("Shoulder X Alignment", False, None))

    # 击球动作的判定
    elif action == "Impact":
        # 条件: 左肩和左脚踝连线是否垂直
        if (confidences[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
            # 获取关键点坐标
            shoulder = keypoints[LEFT_SHOULDER_IDX]
            ankle = keypoints[LEFT_ANKLE_IDX]

            # 计算连线与水平面的夹角
            angle = calculate_line_horizontal_angle(shoulder, ankle)
            deviation = abs(angle - 90)  # 计算与90度的偏差
            is_correct = deviation <= 5  # 是否满足条件
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

            # 生成状态文本
            status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
            text_line = f"Left_shoulder-Ankle Vertical: {angle:.1f}° ({status})"
            # 在图像上绘制文本
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += line_height

            # 存储结果
            action_results.append(("Left_shoulder-Ankle Vertical", is_correct, deviation))
        else:
            # 关键点未检测到的情况
            text_line = "Left_shoulder-Ankle: Keypoints not detected"
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += line_height
            action_results.append(("Left_shoulder-Ankle Vertical", False, None))

    # 收杆动作的判定
    elif action == "Finish":
        # 条件: 右肩和左脚踝连线是否垂直
        if (confidences[RIGHT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                confidences[LEFT_ANKLE_IDX] > CONFIDENCE_THRESHOLD):
            # 获取关键点坐标
            shoulder = keypoints[RIGHT_SHOULDER_IDX]
            ankle = keypoints[LEFT_ANKLE_IDX]

            # 计算连线与水平面的夹角
            angle = calculate_line_horizontal_angle(shoulder, ankle)
            deviation = abs(angle - 90)  # 计算与90度的偏差
            is_correct = deviation <= 5  # 是否满足条件
            color = (0, 255, 0) if is_correct else (0, 0, 255)  # 绿色正确，红色错误

            # 生成状态文本
            status = "Correct" if is_correct else f"Deviation: {deviation:.1f}°"
            text_line = f"Right_shoulder-Ankle Vertical: {angle:.1f}° ({status})"
            # 在图像上绘制文本
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_offset += line_height

            # 存储结果
            action_results.append(("Right_shoulder-Ankle Vertical", is_correct, deviation))
        else:
            # 关键点未检测到的情况
            text_line = "Right_shoulder-Ankle: Keypoints not detected"
            cv2.putText(annotated_frame, text_line, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += line_height
            action_results.append(("Right_shoulder-Ankle Vertical", False, None))

    # 保存关键帧图像
    output_path = os.path.join(OUTPUT_KEYFRAME_DIR, f"{action.replace(' ', '_')}_{frame_idx}.jpg")
    cv2.imwrite(output_path, annotated_frame)  # 写入图像文件
    print(f"Saved {action} key frame to {output_path}")  # 打印保存信息

    # 存储差异信息
    differences_summary.append({
        "action": action,  # 动作名称
        "frame": frame_idx,  # 帧索引
        "results": action_results  # 判定结果
    })


# --- 主程序：逐帧处理视频 ---
frame_count = 0  # 当前帧计数器
while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:  # 检查是否成功读取帧
        break  # 视频结束或读取错误，退出循环

    frame_count += 1  # 帧计数器递增
    # 每30帧打印一次处理进度
    if frame_count % 30 == 0:
        print(f"Processing frame {frame_count}/{total_frames}")

    # 使用YOLO模型进行姿态估计
    results = model(frame, verbose=False)  # verbose=False减少控制台输出

    # 创建带标注的帧副本
    annotated_frame = frame.copy()
    # 当前帧数据字典
    frame_data = {
        "frame_index": frame_count,  # 帧索引
        "persons": []  # 检测到的人物列表
    }

    # 初始化当前帧的关键点数据
    current_kps = None  # 关键点坐标
    current_confs = None  # 关键点置信度
    current_angles = {}  # 角度值
    head_club_pos = None  # 球杆头位置
    middle_club_pos = None  # 球杆中点位置

    # 检查是否有有效的检测结果
    if results and results[0].keypoints and results[0].keypoints.data.shape[0] > 0 and results[
        0].keypoints.xy is not None and results[0].keypoints.conf is not None:
        # 获取关键点数据（转换为numpy数组）
        keypoints_data = results[0].keypoints.xy.cpu().numpy()
        confidences = results[0].keypoints.conf.cpu().numpy()

        # 遍历检测到的每个人
        for person_idx in range(keypoints_data.shape[0]):
            person_keypoints = keypoints_data[person_idx]  # 当前人物的关键点
            person_confs = confidences[person_idx]  # 当前人物的关键点置信度

            # 计算所有角度
            person_angles = {}
            for angle_name, (idx1, idx2, idx3) in ANGLE_DEFINITIONS.items():
                # 检查关键点置信度是否满足阈值
                if (person_confs[idx1] > CONFIDENCE_THRESHOLD and
                        person_confs[idx2] > CONFIDENCE_THRESHOLD and
                        person_confs[idx3] > CONFIDENCE_THRESHOLD):
                    # 获取三个点的坐标
                    p1 = person_keypoints[idx1]
                    p2 = person_keypoints[idx2]
                    p3 = person_keypoints[idx3]
                    # 计算角度
                    angle = calculate_angle_between_points(p1, p2, p3, angle_name)
                    # 存储角度值（转换为整数）
                    person_angles[angle_name] = int(angle) if angle is not None else None

            # 构建当前人物的数据字典
            person_data = {
                "person_id": person_idx,  # 人物ID
                "keypoints": person_keypoints.tolist(),  # 关键点坐标（转换为列表）
                "confidences": [round(conf, 4) for conf in person_confs.tolist()],  # 置信度（保留4位小数）
                "angles": person_angles  # 角度值
            }
            # 添加到当前帧的人物列表
            frame_data["persons"].append(person_data)

            # 只处理第一个检测到的人（假设视频中只有一个高尔夫球员）
            if person_idx == 0:
                # 在当前帧上绘制姿态和角度
                draw_pose_on_image(annotated_frame, person_keypoints, person_confs, person_angles)
                # 存储当前帧数据（用于动作检测）
                current_kps = person_keypoints
                current_confs = person_confs
                current_angles = person_angles

                # 存储球杆位置（如果置信度足够）
                if person_confs[HEAD_CLUB_IDX] > CONFIDENCE_THRESHOLD:
                    head_club_pos = tuple(person_keypoints[HEAD_CLUB_IDX])
                if person_confs[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD:
                    middle_club_pos = tuple(person_keypoints[MIDDLE_CLUB_IDX])

    # 将当前帧的数据添加到总列表
    all_frame_data.append(frame_data)

    # 更新球杆位置历史
    head_club_history.append(head_club_pos)
    middle_club_history.append(middle_club_pos)

    # 动作关键帧检测（只处理第一个检测到的人）
    if current_kps is not None and current_confs is not None:
        # 获取髋关节位置（左右髋关节的平均值）
        if (current_confs[LEFT_HIP_IDX] > CONFIDENCE_THRESHOLD and
                current_confs[RIGHT_HIP_IDX] > CONFIDENCE_THRESHOLD):
            left_hip = current_kps[LEFT_HIP_IDX]
            right_hip = current_kps[RIGHT_HIP_IDX]
            hip_avg = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        else:
            hip_avg = None  # 无法计算髋关节平均值

        # 状态0: 检测预备动作
        if current_state == 0:
            # 检查必要条件：左手腕、髋关节平均值、左手腕角度都存在
            if (current_confs[LEFT_WRIST_IDX] > CONFIDENCE_THRESHOLD and
                    hip_avg is not None and
                    "left_wrist_angle" in current_angles):

                left_wrist_y = current_kps[LEFT_WRIST_IDX][1]  # 左手腕y坐标
                left_wrist_angle = current_angles["left_wrist_angle"]  # 左手腕角度

                # 条件1: 左手腕在髋关节上方
                # 条件2: 左手腕和球杆几乎成直线 (170-360度)
                if left_wrist_y > hip_avg[1] and 170 < left_wrist_angle < 360:
                    # 满足条件，进入下一状态
                    current_state = 1
                    # 记录检测到的状态
                    detected_states.append(("Preparation", frame_count))
                    # 存储关键帧
                    key_frames["Preparation"] = frame_count
                    print(f"Frame {frame_count}: Preparation detected")  # 打印检测信息
                    other_keyframe_recorded = False  # 重置其他关键帧标志
            else:
                # 未检测到预备动作关键帧
                if not other_keyframe_recorded:
                    # 记录为其他关键帧
                    detected_states.append(("Other", frame_count))
                    print(f"Frame {frame_count}: Other detected (no preparation)")
                    other_keyframe_recorded = True  # 设置标志

        # 状态1: 检测上杆顶点
        elif current_state == 1:
            # 条件1: 左手腕角度小于50
            cond1 = "left_wrist_angle" in current_angles and current_angles["left_wrist_angle"] < 50

            # 条件2: 球杆与水平面夹角接近水平 (0-10度 或 170-180度)
            cond2 = False
            if head_club_pos is not None and middle_club_pos is not None:
                club_angle = calculate_club_horizontal_angle(middle_club_pos, head_club_pos)
                cond2 = club_angle < 10 or club_angle > 170

            # 同时满足两个条件
            if cond1 and cond2:
                # 满足条件，进入下一状态
                current_state = 2
                # 记录检测到的状态
                detected_states.append(("Top of Backswing", frame_count))
                # 存储关键帧
                key_frames["Top of Backswing"] = frame_count
                print(f"Frame {frame_count}: Top of Backswing detected")

        # 状态2: 检测击球
        elif current_state == 2:
            # 检查必要条件：球杆中点和髋关节平均值存在
            if (current_confs[MIDDLE_CLUB_IDX] > CONFIDENCE_THRESHOLD and
                    hip_avg is not None):

                middle_club_y = current_kps[MIDDLE_CLUB_IDX][1]  # 球杆中点y坐标

                # 条件: 球杆中点低于髋关节
                if middle_club_y > hip_avg[1]:
                    # 满足条件，进入下一状态
                    current_state = 3
                    # 记录检测到的状态
                    detected_states.append(("Impact", frame_count))
                    # 存储关键帧
                    key_frames["Impact"] = frame_count
                    print(f"Frame {frame_count}: Impact detected")

        # 状态3: 检测收杆
        elif current_state == 3:
            # 条件1: 左手肘角度在260-320度之间
            elbow_angle_condition = False
            if "left_elbow_angle" in current_angles and current_angles["left_elbow_angle"] is not None:
                elbow_angle_condition = 260 <= current_angles["left_elbow_angle"] <= 320

            # 条件2: 球杆角度大于150度
            club_angle_condition = False
            if head_club_pos is not None and middle_club_pos is not None:
                club_angle = calculate_club_horizontal_angle(middle_club_pos, head_club_pos)
                club_angle_condition = 150 < club_angle

            # 条件3: 肩膀连线接近水平
            shoulder_horizontal_condition = False
            if (current_confs[LEFT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD and
                    current_confs[RIGHT_SHOULDER_IDX] > CONFIDENCE_THRESHOLD):
                left_shoulder = current_kps[LEFT_SHOULDER_IDX]
                right_shoulder = current_kps[RIGHT_SHOULDER_IDX]

                # 计算肩膀连线向量
                shoulder_vec = [right_shoulder[0] - left_shoulder[0],
                                right_shoulder[1] - left_shoulder[1]]

                # 计算与水平线的夹角
                angle_rad = math.atan2(abs(shoulder_vec[1]), abs(shoulder_vec[0]))
                shoulder_angle_deg = math.degrees(angle_rad)

                # 检查是否接近水平（小于10度或大于170度）
                shoulder_horizontal_condition = (shoulder_angle_deg <= 10 or
                                                 shoulder_angle_deg >= 170)

            # 同时满足三个条件
            if elbow_angle_condition and club_angle_condition and shoulder_horizontal_condition:
                # 满足条件，进入完成状态
                current_state = 4
                # 记录检测到的状态
                detected_states.append(("Finish", frame_count))
                # 存储关键帧
                key_frames["Finish"] = frame_count
                print(f"Frame {frame_count}: Finish detected")

        # 状态4: 完成
        elif current_state == 4:
            # 超出收杆关键帧的条件
            if not other_keyframe_recorded:
                # 记录为其他关键帧
                detected_states.append(("Other", frame_count))
                print(f"Frame {frame_count}: Other detected (beyond finish)")
                other_keyframe_recorded = True  # 设置标志

    # 在视频顶部显示当前动作提示
    current_action_text = ""  # 初始化当前动作文本
    # 遍历所有检测到的状态
    for action, frame_idx in detected_states:
        # 只显示当前帧或之前的状态
        if frame_idx <= frame_count:
            current_action_text = action  # 更新当前动作文本

    # 如果有当前动作文本，则绘制在视频顶部
    if current_action_text:
        # 计算文本尺寸
        text_size = cv2.getTextSize(current_action_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        text_x = int((frame_width - text_size[0]) / 2)  # 水平居中
        text_y = int(text_size[1] * 1.5)  # 垂直位置

        # 绘制背景矩形
        cv2.rectangle(annotated_frame,
                      (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      TEXT_BG_COLOR, -1)  # -1表示填充矩形

        # 绘制文本
        cv2.putText(annotated_frame, current_action_text,
                    (text_x, text_y),
                    TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

    # 将带标注的帧写入输出视频
    out.write(annotated_frame)

# --- 释放资源 ---
cap.release()  # 释放视频捕获对象
out.release()  # 释放视频写入器

# 保存关键帧并进行姿势判定
if any(key_frames.values()):  # 检查是否有任何关键帧被检测到
    # 重新打开视频文件
    cap = cv2.VideoCapture(VIDEO_PATH)
    # 遍历所有关键动作
    for action, frame_idx in key_frames.items():
        if frame_idx is None:
            continue  # 跳过未检测到的动作

        # 定位到关键帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = cap.read()  # 读取关键帧
        if not ret:
            continue  # 读取失败则跳过

        # 获取对应帧的数据索引
        data_index = frame_idx - 1
        if data_index >= len(all_frame_data):
            continue  # 索引超出范围则跳过

        # 获取帧数据
        frame_data = all_frame_data[data_index]
        if not frame_data['persons']:
            continue  # 没有检测到人物则跳过

        # 获取第一个人的数据
        person0 = frame_data['persons'][0]
        keypoints = np.array(person0['keypoints'])  # 关键点坐标
        confidences = np.array(person0['confidences'])  # 关键点置信度
        angles_dict = person0['angles']  # 角度值

        # 保存并评估关键帧
        save_and_evaluate_keyframe(frame_idx, frame, keypoints, confidences, angles_dict, action)

    cap.release()  # 释放视频捕获对象

# 打印差异总结
print("\nSummary of differences from ideal pose:")
# 遍历所有差异项
for item in differences_summary:
    print(f"\nAction: {item['action']} (Frame {item['frame']})")
    # 遍历该动作的所有判定结果
    for (cond_name, cond_ok, deviation) in item['results']:
        if cond_ok:
            print(f"  {cond_name}: OK")  # 条件满足
        else:
            if deviation is not None:
                print(f"  {cond_name}: Not OK, Deviation: {deviation:.1f}")  # 条件不满足且有偏差值
            else:
                print(f"  {cond_name}: Keypoints not available")  # 关键点不可用

# 打印处理完成信息
print(f"Processing finished. Annotated video saved to {OUTPUT_VIDEO_PATH}")
print("Detected key frames:")
# 打印所有检测到的关键帧
for action, frame_idx in detected_states:
    print(f"  {action} at frame {frame_idx}")
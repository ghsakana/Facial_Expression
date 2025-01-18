import cv2
import mediapipe as mp
import numpy as np
import time

# 初始化mediapipe FaceMesh模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   min_detection_confidence=0.75,
                                   min_tracking_confidence=0.5)

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils

# 定义一个函数来获取特定点的坐标
def get_landmark_coordinates(landmarks, landmark_id):
    if landmark_id < len(landmarks.landmark):
        return landmarks.landmark[landmark_id]
    else:
        return None

# 主函数
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    camera_on = False
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 默认空白帧
    expression = "No face detected"
    start_time = time.time()
    frame_count = 0
    fps = 0  # 初始化帧率为0

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # 按 'c' 键切换摄像头状态
            camera_on = not camera_on
            if camera_on:
                print("Camera turned on")
                start_time = time.time()  # 重新开始计时
                frame_count = 0  # 重置帧数计数
            else:
                print("Camera turned off")

        if camera_on:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # 将帧转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 进行人脸网格检测
            results = face_mesh.process(rgb_frame)

            # 绘制人脸网格
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 绘制脸部轮廓
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

                    # 获取并绘制关键点
                    keypoints = [
                        468, 473, 463, 468,  # 左右眼内侧和外侧
                        52, 270,             # 嘴巴左右角
                        61, 291,             # 嘴巴角落上扬点
                        105, 336,           # 左眉毛上下点
                        334, 296            # 右眉毛上下点
                    ]
                    
                    for kp in keypoints:
                        lm = get_landmark_coordinates(face_landmarks, kp)
                        if lm:
                            h, w, _ = frame.shape
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                    # 分析表情
                    left_lip = get_landmark_coordinates(face_landmarks, 61)
                    right_lip = get_landmark_coordinates(face_landmarks, 291)
                    upper_lip = get_landmark_coordinates(face_landmarks, 13)
                    lower_lip = get_landmark_coordinates(face_landmarks, 14)

                    if all([left_lip, right_lip, upper_lip, lower_lip]):
                        lip_distance = abs(upper_lip.y - lower_lip.y)
                        lip_width = abs(left_lip.x - right_lip.x)
                        if lip_distance / lip_width > 0.05:
                            expression = "Smiling"
                        else:
                            expression = "Neutral"
                    else:
                        expression = "Unknown"
            else:
                expression = "No face detected"
        else:
            frame[:] = (0, 0, 0)  # 清空帧内容
            expression = "Camera is off"
            fps = 0  # 当摄像头关闭时，将帧率设置为0

        # 显示视频流、表情和帧率
        cv2.putText(frame, f'Expression: {expression}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 根据摄像头状态显示不同的提示信息
        if camera_on:
            cv2.putText(frame, 'Press c to turn off camera', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Press c to turn on camera', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, 'Press q to quit', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Facial Expression Analysis', frame)

        if key == ord('q'):  # 按 'q' 键退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




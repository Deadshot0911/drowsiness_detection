import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import time
import pygame
import sys
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308

# Modern Glassmorphism Palette (BGR)
COLOR_BG = (20, 20, 20)            
COLOR_TEXT_MAIN = (255, 255, 255)  
COLOR_TEXT_SUB = (180, 180, 180)   
COLOR_ACCENT = (255, 180, 50)      
COLOR_GOOD = (120, 255, 150)       
COLOR_WARN = (0, 200, 255)         
COLOR_ALERT = (80, 80, 255)        
COLOR_GLASS_BG = (40, 40, 40)      

# Neon wireframe colors
COLOR_MESH_EYE = (255, 255, 0)
COLOR_MESH_MOUTH = (0, 255, 255)
COLOR_MESH_FACE = (150, 150, 150)

CALIBRATION_FRAMES = 60
CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.60
YAWN_CONSEC_FRAMES = 15

# ==========================================
# AUDIO & LOGGING SETUP
# ==========================================
pygame.mixer.init()
def generate_beep():
    sr, dur, freq = 44100, 0.4, 880
    t = np.linspace(0, dur, int(sr * dur), False)
    w = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack([w, w]))
alarm_sound = generate_beep()

alarm_on = False
def play_alarm():
    global alarm_on
    if not alarm_on:
        alarm_sound.play(-1)
        alarm_on = True
def stop_alarm():
    global alarm_on
    if alarm_on:
        alarm_sound.stop()
        alarm_on = False

# Log Buffer
system_logs = deque(maxlen=5)
def add_log(message, color):
    timestamp = time.strftime("%H:%M:%S")
    system_logs.appendleft((f"[{timestamp}] {message}", color))

# ==========================================
# MATH UTILITIES
# ==========================================
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    A = distance.euclidean(p2, p6)
    B = distance.euclidean(p3, p5)
    C = distance.euclidean(p1, p4)
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, w, h):
    top, bottom = landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM]
    left, right = landmarks[MOUTH_LEFT], landmarks[MOUTH_RIGHT]
    v = distance.euclidean((top.x * w, top.y * h), (bottom.x * w, bottom.y * h))
    hz = distance.euclidean((left.x * w, left.y * h), (right.x * w, right.y * h))
    return v / hz if hz > 0 else 0.0

# ==========================================
# TECHNICAL MESH DRAWING
# ==========================================
def draw_mesh(frame, landmarks, w, h):
    # Draw Eye Meshes
    for i in range(len(LEFT_EYE)):
        p1 = (int(landmarks[LEFT_EYE[i]].x * w), int(landmarks[LEFT_EYE[i]].y * h))
        p2 = (int(landmarks[LEFT_EYE[(i+1)%len(LEFT_EYE)]].x * w), int(landmarks[LEFT_EYE[(i+1)%len(LEFT_EYE)]].y * h))
        cv2.line(frame, p1, p2, COLOR_MESH_EYE, 1, cv2.LINE_AA)
        
    for i in range(len(RIGHT_EYE)):
        p1 = (int(landmarks[RIGHT_EYE[i]].x * w), int(landmarks[RIGHT_EYE[i]].y * h))
        p2 = (int(landmarks[RIGHT_EYE[(i+1)%len(RIGHT_EYE)]].x * w), int(landmarks[RIGHT_EYE[(i+1)%len(RIGHT_EYE)]].y * h))
        cv2.line(frame, p1, p2, COLOR_MESH_EYE, 1, cv2.LINE_AA)

    # Draw Mouth Mesh
    p_top = (int(landmarks[MOUTH_TOP].x * w), int(landmarks[MOUTH_TOP].y * h))
    p_bot = (int(landmarks[MOUTH_BOTTOM].x * w), int(landmarks[MOUTH_BOTTOM].y * h))
    p_lft = (int(landmarks[MOUTH_LEFT].x * w), int(landmarks[MOUTH_LEFT].y * h))
    p_rgt = (int(landmarks[MOUTH_RIGHT].x * w), int(landmarks[MOUTH_RIGHT].y * h))
    cv2.line(frame, p_lft, p_top, COLOR_MESH_MOUTH, 1, cv2.LINE_AA)
    cv2.line(frame, p_top, p_rgt, COLOR_MESH_MOUTH, 1, cv2.LINE_AA)
    cv2.line(frame, p_lft, p_bot, COLOR_MESH_MOUTH, 1, cv2.LINE_AA)
    cv2.line(frame, p_bot, p_rgt, COLOR_MESH_MOUTH, 1, cv2.LINE_AA)

    # Draw Jawline/Neck tracking dots
    jaw_points = [152, 234, 454, 132, 361, 58, 288]
    for pt in jaw_points:
        x = int(landmarks[pt].x * w)
        y = int(landmarks[pt].y * h)
        cv2.circle(frame, (x, y), 1, COLOR_MESH_FACE, -1)
        # Connect jawline
        if pt in [152, 234, 454]:
            cv2.line(frame, (x,y), (int(landmarks[152].x * w), int(landmarks[152].y * h)), COLOR_MESH_FACE, 1, cv2.LINE_AA)

# ==========================================
# MODERN GLASS UI DRAWING
# ==========================================
def create_glass_card(w, h):
    card = np.zeros((h, w, 4), dtype=np.uint8)
    card[:, :, 0] = COLOR_GLASS_BG[0]
    card[:, :, 1] = COLOR_GLASS_BG[1]
    card[:, :, 2] = COLOR_GLASS_BG[2]
    card[:, :, 3] = 180 
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (w-20, h-20), 255, -1) 
    cv2.circle(mask, (20, 20), 20, 255, -1)
    cv2.circle(mask, (w-20, 20), 20, 255, -1)
    cv2.circle(mask, (20, h-20), 20, 255, -1)
    cv2.circle(mask, (w-20, h-20), 20, 255, -1)
    
    card[:, :, 3] = np.where(mask == 0, 0, card[:, :, 3])
    cv2.rectangle(card, (2, 2), (w-3, h-3), (255, 255, 255, 50), 1)
    return card

def overlay_transparent(bg_img, img_to_overlay_t, x, y):
    bg_img = bg_img.copy()
    h, w = img_to_overlay_t.shape[:2]
    y_end = min(y + h, bg_img.shape[0])
    x_end = min(x + w, bg_img.shape[1])
    overlay_h = y_end - y
    overlay_w = x_end - x
    
    if overlay_h <= 0 or overlay_w <= 0: return bg_img

    alpha = img_to_overlay_t[:overlay_h, :overlay_w, 3] / 255.0
    for c in range(0, 3):
        bg_img[y:y_end, x:x_end, c] = (alpha * img_to_overlay_t[:overlay_h, :overlay_w, c] +
                                    (1 - alpha) * bg_img[y:y_end, x:x_end, c])
    return bg_img

def draw_ui(frame, ear, mar, fps, elapsed, status, status_color, threshold, calibrating, calib_prog, drowsy_c, yawn_c):
    h, w = frame.shape[:2]
    canvas_w = w + 400
    canvas = np.full((h, canvas_w, 3), COLOR_BG, dtype=np.uint8)
    
    for i in range(h):
        canvas[i, :] = np.clip(np.array(COLOR_BG) + int(10 * (i/h)), 0, 255)
    
    # Webcam Feed
    frame_resized = cv2.resize(frame, (w, h))
    cv2.rectangle(frame_resized, (0, 0), (w-1, h-1), (60, 60, 60), 2)
    canvas[0:h, 400:canvas_w] = frame_resized
    
    # --- CARD 1: Header & Status ---
    card1 = create_glass_card(360, 120)
    canvas = overlay_transparent(canvas, card1, 20, 20)
    cv2.putText(canvas, "DROWSINESS DETECTOR", (40, 55), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(canvas, "AI Monitoring System v2.1", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT_SUB, 1, cv2.LINE_AA)
    
    status_text = status
    (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(canvas, (40, 100), (40 + tw + 40, 130), status_color, -1)
    cv2.putText(canvas, status_text, (60, 122), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # --- CARD 2: Metrics ---
    card2 = create_glass_card(360, 200)
    canvas = overlay_transparent(canvas, card2, 20, 160)
    cv2.putText(canvas, "BIOMETRIC METRICS", (40, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    
    cv2.putText(canvas, "Eye Aspect Ratio (EAR)", (40, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{ear:.3f}", (310, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (40, 235), (340, 245), (50, 50, 50), -1)
    ear_color = COLOR_GOOD if ear > threshold else COLOR_ALERT
    cv2.rectangle(canvas, (40, 235), (40 + int(300 * min(1.0, ear / 0.4)), 245), ear_color, -1)
    
    cv2.putText(canvas, "Mouth Aspect Ratio (MAR)", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{mar:.3f}", (310, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (40, 290), (340, 300), (50, 50, 50), -1)
    mar_color = COLOR_WARN if mar > MAR_THRESHOLD else COLOR_ACCENT
    cv2.rectangle(canvas, (40, 290), (40 + int(300 * min(1.0, mar / 0.8)), 300), mar_color, -1)

    if calibrating:
        cv2.putText(canvas, "CALIBRATING...", (40, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARN, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (40, 345), (340, 355), (50, 50, 50), -1)
        cv2.rectangle(canvas, (40, 345), (40 + int((calib_prog / CALIBRATION_FRAMES) * 300), 355), COLOR_GOOD, -1)

    # --- CARD 3: Telemetry ---
    card3 = create_glass_card(360, 100)
    canvas = overlay_transparent(canvas, card3, 20, 380)
    cv2.putText(canvas, "SESSION TELEMETRY", (40, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Drowsy Alerts", (40, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{drowsy_c}", (310, 445), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_ALERT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Yawns Detected", (40, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_SUB, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{yawn_c}", (310, 470), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_WARN, 1, cv2.LINE_AA)

    # --- CARD 4: System Logs ---
    card4 = create_glass_card(360, 130)
    canvas = overlay_transparent(canvas, card4, 20, 500)
    cv2.putText(canvas, "SYSTEM LOGS", (40, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1, cv2.LINE_AA)
    
    y_offset = 555
    for log_text, log_color in system_logs:
        cv2.putText(canvas, log_text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, log_color, 1, cv2.LINE_AA)
        y_offset += 20

    cv2.putText(canvas, f"FPS: {fps:.1f}", (40, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"UPTIME: {int(elapsed//60):02d}:{int(elapsed%60):02d}", (160, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    return canvas

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    start_time = time.time()
    
    calibrating = True
    calib_frames = 0
    calib_data = []
    baseline_ear = 0.30
    EAR_THRESHOLD = 0.25
    
    ear_counter = 0
    yawn_counter = 0
    drowsy_count = 0
    yawn_count = 0
    
    # Flags to prevent log spamming
    eyes_closing_logged = False
    drowsy_logged = False
    yawning_logged = False

    add_log("SYSTEM: Boot sequence initiated", COLOR_ACCENT)
    add_log("SYSTEM: Calibrating biometrics...", COLOR_WARN)

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        elapsed = curr_time - start_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear, mar = 0.0, 0.0
        status = "NO FACE"
        status_color = (100, 100, 100)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(landmarks, w, h)

            # Draw the technical mesh
            draw_mesh(frame, landmarks, w, h)

            if calibrating:
                calib_data.append(ear)
                calib_frames += 1
                status = "CALIBRATING"
                status_color = COLOR_WARN
                
                if calib_frames >= CALIBRATION_FRAMES:
                    calibrating = False
                    baseline_ear = sum(calib_data) / len(calib_data)
                    EAR_THRESHOLD = max(0.18, baseline_ear * 0.75)
                    add_log(f"SYS: Calibrated. Thr: {EAR_THRESHOLD:.2f}", COLOR_GOOD)
            
            else:
                # Drowsy Logic
                if ear < EAR_THRESHOLD:
                    ear_counter += 1
                    if ear_counter >= CONSEC_FRAMES:
                        if not drowsy_logged:
                            drowsy_count += 1
                            add_log("ALERT: Drowsiness detected!", COLOR_ALERT)
                            drowsy_logged = True
                        status = "DROWSY ALERT"
                        status_color = COLOR_ALERT
                        play_alarm()
                    else:
                        if not eyes_closing_logged:
                            add_log("WARN: Eyes closing...", COLOR_WARN)
                            eyes_closing_logged = True
                        status = "EYES CLOSING"
                        status_color = COLOR_WARN
                else:
                    if drowsy_logged or eyes_closing_logged:
                        add_log("SYS: Eyes open. Alert reset.", COLOR_GOOD)
                        drowsy_logged = False
                        eyes_closing_logged = False
                    ear_counter = 0
                    stop_alarm()

                # Yawning Logic
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= YAWN_CONSEC_FRAMES:
                        if not yawning_logged:
                            yawn_count += 1
                            add_log("WARN: Yawning detected.", COLOR_WARN)
                            yawning_logged = True
                        status = "YAWNING"
                        status_color = COLOR_WARN
                else:
                    if yawning_logged:
                        add_log("SYS: Mouth closed.", COLOR_GOOD)
                        yawning_logged = False
                    yawn_counter = 0

                if ear_counter == 0 and yawn_counter < YAWN_CONSEC_FRAMES:
                    status = "ALERT"
                    status_color = COLOR_GOOD
        else:
            stop_alarm()
            ear_counter = 0
            status = "NO FACE"
            status_color = (100, 100, 100)

        ui_frame = draw_ui(frame, ear, mar, fps, elapsed, status, status_color, EAR_THRESHOLD, calibrating, calib_frames, drowsy_count, yawn_count)
        cv2.imshow("Drowsiness Detector", ui_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        # Check if the user clicked the 'X' button to close the window
        if cv2.getWindowProperty("Drowsiness Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pygame.mixer.quit()
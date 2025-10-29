import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import torch
import sqlite3

# ----------------------------
# Fixed defaults (no UI controls)
# ----------------------------
CONF_THRESHOLD = 0.5
STOP_LINE_Y = 400
HELMET_MODEL_PATH = 'best2.pt'
TRAFFIC_MODEL_PATH = 'yolov8n.pt'
DB_PATH = 'violations.db'

# ----------------------------
# Database helpers (persistent connection stored in session_state)
# ----------------------------

def init_db():
    # Keep a persistent connection in session_state to avoid locking issues
    if 'db_conn' not in st.session_state:
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            conn.execute('PRAGMA busy_timeout = 30000;')
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    violation_type TEXT,
                    tracker_id INTEGER
                )
            ''')
            conn.commit()
            st.session_state['db_conn'] = conn
        except Exception as e:
            st.error(f"Failed to initialize DB: {e}")

def get_db_conn():
    init_db()
    return st.session_state.get('db_conn')

def log_violation_db(tracker_id, violation_type):
    try:
        conn = get_db_conn()
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO violations (timestamp, violation_type, tracker_id) VALUES (?, ?, ?)",
                  (timestamp, violation_type, tracker_id))
        conn.commit()
    except Exception as e:
        # avoid spamming the UI with warnings during heavy processing
        st.session_state.setdefault('db_errors', []).append(str(e))

def read_violation_log(limit=10000):
    try:
        conn = get_db_conn()
        df = pd.read_sql_query(f"SELECT * FROM violations ORDER BY id DESC LIMIT {limit}", conn)
        return df
    except Exception as e:
        st.error(f"Failed to read DB: {e}")
        return pd.DataFrame()

def clear_violations_db():
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("DELETE FROM violations")
        conn.commit()
    except Exception as e:
        st.error(f"Failed to clear violations: {e}")

# ----------------------------
# Model loading (cached)
# ----------------------------
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model '{path}': {e}")
        return None

# ----------------------------
# Lightweight annotation (faster than model.plot())
# ----------------------------

def draw_boxes(annotated, boxes, names):
    # boxes: ultralytics boxes object
    try:
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None
    except Exception:
        return annotated

    h, w = annotated.shape[:2]
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        label = ''
        cls = int(clss[i]) if len(clss) > 0 else -1
        conf = float(confs[i]) if len(confs) > 0 else 0.0
        name = names.get(cls, str(cls)) if names is not None else str(cls)
        if ids is not None:
            tracker_id = int(ids[i])
            label = f'{name}:{tracker_id} {conf:.2f}'
        else:
            label = f'{name} {conf:.2f}'
        # choose color based on class name keywords to keep it deterministic
        color = (0, 255, 0)
        if 'traffic' in name.lower() or 'car' in name.lower() or 'motor' in name.lower():
            color = (255, 165, 0)
        if 'helmet' in name.lower() or 'without' in name.lower() or 'no' in name.lower():
            color = (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return annotated

# ----------------------------
# Helpers for traffic-light detection
# ----------------------------

def crop_and_sanitize(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
    y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def infer_traffic_light_color(crop):
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return 'unknown'
    v = hsv[:, :, 2]
    if v.mean() < 50:
        return 'unknown'
    lower_red1 = np.array([0, 90, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 90, 120])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([15, 90, 120])
    upper_yellow = np.array([35, 255, 255])
    lower_green = np.array([40, 50, 70])
    upper_green = np.array([90, 255, 255])
    mask_r = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    r = cv2.countNonZero(mask_r)
    y = cv2.countNonZero(mask_y)
    g = cv2.countNonZero(mask_g)
    counts = {'red': r, 'yellow': y, 'green': g}
    best = max(counts, key=counts.get)
    if counts[best] < 15:
        return 'unknown'
    return best

# ----------------------------
# Core processing (image/video)
# ----------------------------

def process_image(image_file, helmet_model, traffic_model):
    data = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error('Could not decode image')
        return None, 0, False
    try:
        with torch.no_grad():
            helmet_res = helmet_model(img, conf=CONF_THRESHOLD)
            traffic_res = traffic_model(img, conf=CONF_THRESHOLD)
    except Exception as e:
        st.error(f"Model inference error: {e}")
        return None, 0, False
    helmet_names = getattr(helmet_model, 'names', {})
    helmet_violations = 0
    if helmet_res and helmet_res[0].boxes is not None:
        for cls in helmet_res[0].boxes.cls:
            if helmet_names.get(int(cls), str(cls)).lower() in ['without helmet', 'no helmet', 'no_helmet']:
                helmet_violations += 1
    is_light_red = False
    traffic_names = getattr(traffic_model, 'names', {})
    if traffic_res and traffic_res[0].boxes is not None and len(traffic_res[0].boxes.cls) > 0:
        candidate_lights = []
        for i, cls in enumerate(traffic_res[0].boxes.cls):
            cls_i = int(cls)
            name = traffic_names.get(cls_i, str(cls_i)).lower()
            conf = float(traffic_res[0].boxes.conf[i])
            box = traffic_res[0].boxes.xyxy[i].cpu().numpy().tolist()
            if 'traffic light' in name and conf >= CONF_THRESHOLD:
                candidate_lights.append((conf, box))
        if candidate_lights:
            best = max(candidate_lights, key=lambda x: x[0])
            crop = crop_and_sanitize(img, best[1])
            if crop is not None:
                color = infer_traffic_light_color(crop)
                if color == 'red':
                    is_light_red = True
    # annotate manually (faster)
    annotated = img.copy()
    if helmet_res and helmet_res[0].boxes is not None:
        annotated = draw_boxes(annotated, helmet_res[0].boxes, helmet_names)
    if traffic_res and traffic_res[0].boxes is not None:
        annotated = draw_boxes(annotated, traffic_res[0].boxes, traffic_names)
    return annotated, helmet_violations, is_light_red


def process_video(video_file, helmet_model, traffic_model, progress_callback=None):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.flush()
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error('Cannot open video')
        return 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    helmet_violator_ids = set()
    signal_violator_ids = set()
    vehicle_positions = {}
    total_helmet_violations = 0
    total_signal_violations = 0
    helmet_names = getattr(helmet_model, 'names', {})
    traffic_names = getattr(traffic_model, 'names', {})
    st_frame = st.empty()
    # Use tracking mode for performance
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # draw stop line
        cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (0, 0, 255), 2)
        try:
            with torch.no_grad():
                helmet_res = helmet_model.track(frame, persist=True)
                traffic_res = traffic_model.track(frame, persist=True)
        except Exception as e:
            st.warning(f"Inference error on frame {frame_idx}: {e}")
            continue
        annotated = frame.copy()
        # helmet processing
        if helmet_res and len(helmet_res) > 0 and helmet_res[0].boxes is not None:
            boxes = helmet_res[0].boxes
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            clss = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            for tracker_id, cls, conf in zip(ids, clss, confs):
                name = helmet_names.get(int(cls), str(cls)).lower()
                if conf >= CONF_THRESHOLD and name in ['without helmet', 'no helmet', 'no_helmet'] and tracker_id not in helmet_violator_ids:
                    helmet_violator_ids.add(tracker_id)
                    total_helmet_violations += 1
                    log_violation_db(tracker_id, 'Without Helmet')
            annotated = draw_boxes(annotated, boxes, helmet_names)
        # traffic processing
        is_light_red = False
        traffic_light_box = None
        if traffic_res and len(traffic_res) > 0 and traffic_res[0].boxes is not None:
            boxes = traffic_res[0].boxes
            candidate_lights = []
            for i, cls in enumerate(boxes.cls):
                name = traffic_names.get(int(cls), str(cls)).lower()
                conf = float(boxes.conf[i])
                box = boxes.xyxy[i].cpu().numpy().tolist()
                if 'traffic light' in name and conf >= CONF_THRESHOLD:
                    candidate_lights.append((conf, box))
            if candidate_lights:
                best = max(candidate_lights, key=lambda x: x[0])
                traffic_light_box = best[1]
                crop = crop_and_sanitize(frame, traffic_light_box)
                if crop is not None:
                    color = infer_traffic_light_color(crop)
                    if color == 'red':
                        is_light_red = True
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            clss = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
            confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
            xys = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
            for tracker_id, cls, conf, box in zip(ids, clss, confs, xys):
                name = traffic_names.get(int(cls), str(cls)).lower()
                if conf >= CONF_THRESHOLD and name in ['car', 'motorcycle', 'bus', 'truck']:
                    y_pos = int(box[3])
                    prev = vehicle_positions.get(tracker_id, 0)
                    if is_light_red and prev < STOP_LINE_Y and y_pos >= STOP_LINE_Y and tracker_id not in signal_violator_ids:
                        signal_violator_ids.add(tracker_id)
                        total_signal_violations += 1
                        log_violation_db(tracker_id, f'Signal Jump ({name})')
                    vehicle_positions[tracker_id] = y_pos
            annotated = draw_boxes(annotated, boxes, traffic_names)
        if is_light_red and traffic_light_box is not None:
            x1, y1, x2, y2 = map(int, traffic_light_box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, 'RED LIGHT', (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)
    cap.release()
    try:
        os.remove(tfile.name)
    except Exception:
        pass
    return total_helmet_violations, total_signal_violations

# ----------------------------
# Streamlit UI (minimal, no advanced settings visible)
# ----------------------------

def main():
    st.set_page_config(page_title='Smart Traffic Violation Detector', layout='wide', initial_sidebar_state='collapsed')
    st.title('Smart Traffic Violation Detector')

    # initialize DB and models (fixed paths)
    init_db()
    with st.spinner('Loading models...'):
        helmet_model = load_model(HELMET_MODEL_PATH)
        traffic_model = load_model(TRAFFIC_MODEL_PATH)

    if helmet_model is None or traffic_model is None:
        st.error('Models not loaded. Ensure the files best2.pt and yolov8n.pt are present in the app root.')
        return

    col1, col2 = st.columns([2,1])
    with col1:
        tabs = st.tabs(['Process Image', 'Process Video'])
        with tabs[0]:
            st.header('Process a single image')
            img_file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
            if img_file:
                if st.button('Analyze image', key='analyze_img'):
                    annotated, helmet_count, is_red = process_image(img_file, helmet_model, traffic_model)
                    if annotated is not None:
                        st.metric('Without Helmet (count)', helmet_count)
                        st.markdown(f"**Red light detected:** {'Yes' if is_red else 'No'}")
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with tabs[1]:
            st.header('Process a video file')
            video_file = st.file_uploader('Upload video', type=['mp4', 'mov', 'avi', 'mkv'], key='video_upload')
            if video_file:
                if st.button('Start video analysis', key='start_video'):
                    progress = st.progress(0)
                    def _cb(p):
                        try:
                            progress.progress(min(100, int(p*100)))
                        except Exception:
                            pass
                    helmet_v, signal_v = process_video(video_file, helmet_model, traffic_model, progress_callback=_cb)
                    st.success(f'Finished. Helmet violations: {helmet_v} | Signal violations: {signal_v}')
    with col2:
        st.header('Violation Log (database)')
        df = read_violation_log()
        if df.empty:
            st.info('No violations logged yet.')
        else:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', data=csv, file_name=f'violations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        if st.button('Clear Log'):
            clear_violations_db()
            st.experimental_rerun()
    st.caption('')

if __name__ == '__main__':
    main()

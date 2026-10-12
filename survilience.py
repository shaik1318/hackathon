import csv
import logging
import os
import socket
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import winsound

    def beep():
        winsound.Beep(1000, 400)
except ImportError:
    def beep():
        print("\a", end="", flush=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
ALERT_DIR = os.path.join(BASE_DIR, "alerts")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
LOG_PATH = os.path.join(ALERT_DIR, "log.csv")
LOG_FIELDS = ["time", "track_id", "class", "direction", "snapshot"]
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

CLASSES = {
    "person": [0],
    "car": [2],
    "motorcycle": [3],
    "bus": [5],
    "truck": [7],
    "animal": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
PORT = 5000               # this PC (http), used on the same computer
PHONE_PORT = 5001         # phones on the same Wi-Fi (https)

# ---- colour of the border line (OpenCV uses BGR order) ----
LINE_COLOR = (255, 229, 0)       # cyan, same as #00E5FF on the web page
LINE_WIDTH = 3

# ---- speed and display ----
INFER_SIZE = 320          # detection size: 320 fastest, 416 balanced, 640 most accurate
STREAM_WIDTH = 640        # width of the picture sent to the browser (IP cameras and video files)
STREAM_QUALITY = 65       # JPEG quality sent to the browser (lower = faster)
STREAM_FPS = 20           # pictures per second for IP cameras and video files
WAIT_AFTER = 3.0          # seconds without camera pictures before "Waiting for camera" shows
BOX_MAX_AGE = 1.5         # seconds a detected box stays on screen without a newer detection
SHOW_TIMING = True

# ---- border tracking: every frame is matched directly to the drawn picture ----
TRACK_WIDTH = 480         # size of the small picture used for matching
TRACK_FEATURES = 800      # features per picture (more = steadier, slower)
MATCH_RATIO = 0.75        # how distinct a match must be (lower = stricter)
MIN_INLIERS = 18          # matching points needed to accept a position
RANSAC_PX = 4.0           # how exact a match must be, in small-picture pixels
HOLD_FRAMES = 6           # frames the last border is kept if one frame fails to match
SMOOTH = 0.7              # weight of each new position (1 = no smoothing; lower = steadier)
OFF_SCREEN = 2.0          # how far (in picture widths) the border may be off screen and still be kept
MIN_ZONE_POINTS = 3
MAX_ZONE_POINTS = 30

os.makedirs(ALERT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(LOG_FIELDS)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024


@app.after_request
def allow_web_page(resp):
    """Lets the GitHub Pages web page (a different address) talk to this server."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Frame-Id"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return resp


LOCK = threading.Lock()
LOG_LOCK = threading.RLock()
S = {"running": False, "stop": False, "jpeg": None, "jpeg_id": 0, "preview": None,
     "crossings": 0, "status": "Idle", "error": "", "events": [], "dets": [],
     "overlay": None, "source": None}
_model = None


class Slot:
    """Keeps only the newest item. Writers overwrite it; readers always get the latest."""

    def __init__(self):
        self.lock = threading.Lock()
        self.item = None
        self.seq = 0

    def put(self, item):
        with self.lock:
            self.item = item
            self.seq += 1

    def get(self):
        with self.lock:
            return self.item, self.seq

    def clear(self):
        with self.lock:
            self.item = None
            self.seq = 0


RAW = Slot()        # newest picture: (image, frame_id sent by the phone, or None)
DETECT_IN = Slot()  # newest picture with its border position, handed to detection


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


def parse_source(value):
    text = str(value).strip().strip('"').strip("'")
    if text == "":
        return ""
    return int(text) if text.isdigit() else text


def is_live(source):
    return isinstance(source, int) or str(source).lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://"))


def read_first_frame(source):
    cap = cv2.VideoCapture(source)
    frame = None
    for _ in range(15):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    return frame


def decode_jpeg(data):
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def valid_line(line):
    try:
        if not line or len(line) != 2:
            return None
        pts = [(float(x), float(y)) for x, y in line]
    except (TypeError, ValueError):
        return None
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in pts) or pts[0] == pts[1]:
        return None
    return pts


def valid_zone(zone):
    if not zone:
        return []
    try:
        pts = [(float(x), float(y)) for x, y in zone]
    except (TypeError, ValueError):
        return None
    if not (MIN_ZONE_POINTS <= len(pts) <= MAX_ZONE_POINTS):
        return None
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in pts):
        return None
    return pts


def side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def near_segment(p, a, b, margin=0.15):
    ab = np.array(b, float) - np.array(a, float)
    t = np.dot(np.array(p, float) - np.array(a, float), ab) / np.dot(ab, ab)
    return -margin <= t <= 1.0 + margin


# ------------------------------------------------------------ photo storage and log
def clean_log_rows():
    with LOG_LOCK:
        try:
            with open(LOG_PATH, newline="") as f:
                return [{k: (v or "") for k, v in r.items() if k} for r in csv.DictReader(f)]
        except OSError:
            return []


def write_log_rows(rows):
    with LOG_LOCK:
        tmp = LOG_PATH + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, LOG_PATH)


def log_alert(frame, tid, label, direction):
    now = datetime.now()
    name = f"{now:%Y%m%d_%H%M%S_%f}_id{tid}.jpg"
    path = os.path.join(ALERT_DIR, name)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    stamp = now.isoformat(timespec="seconds")
    with LOG_LOCK:
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([stamp, tid, label, direction, name])
    print(f"ALERT {now:%H:%M:%S}: {label} ID {tid} - {direction}")
    threading.Thread(target=beep, daemon=True).start()
    return {"time": stamp, "id": str(tid), "class": label, "direction": direction,
            "image": "/alerts/" + name}


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip.startswith(("192.168.", "10.", "172.")):
                return ip
    except OSError:
        pass
    return "YOUR-PC-IP"


def https_ready():
    try:
        import OpenSSL  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------ the scene: border and zone stay fixed
def th_of(h, w):
    return max(1, int(round(h * TRACK_WIDTH / float(w))))


def small_gray(img):
    h, w = img.shape[:2]
    th = th_of(h, w)
    small = cv2.resize(img, (TRACK_WIDTH, th), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)), th


def make_features():
    """SIFT copes with changes of viewing angle. ORB is used only if SIFT is not in this OpenCV build."""
    try:
        return cv2.SIFT_create(nfeatures=TRACK_FEATURES), cv2.NORM_L2
    except AttributeError:
        return cv2.ORB_create(nfeatures=TRACK_FEATURES), cv2.NORM_HAMMING


def sane(H, ref_th):
    """Reject matches that would squash, stretch or throw the picture away."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    corners = np.float32([[0, 0], [TRACK_WIDTH, 0], [TRACK_WIDTH, ref_th], [0, ref_th]]).reshape(-1, 1, 2)
    m = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.all(np.isfinite(m)):
        return False
    ratio = cv2.contourArea(m.astype(np.float32)) / float(TRACK_WIDTH * ref_th)
    return 0.1 <= ratio <= 10.0


class BorderTracker:
    """Keeps the border line and zone on the same place in the scene.

    Every frame is matched directly to the picture the border was drawn on. The border is then
    mapped from that picture onto the current frame. Nothing is chained from frame to frame,
    so errors cannot build up and the line cannot slide with the camera.
    If the drawn scene is not in view, the border is hidden instead of being drawn in a wrong place.
    """

    def __init__(self, ref_bgr, points):
        self.det, norm = make_features()
        self.bf = cv2.BFMatcher(norm)
        self.points = [(float(x), float(y)) for x, y in points]
        g, self.ref_th = small_gray(ref_bgr)
        kp, des = self.det.detectAndCompute(g, None)
        self.ref_des = des
        self.ref_kp = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        self.enabled = des is not None and len(kp) >= MIN_INLIERS + 5
        self.ref_px = [(x * TRACK_WIDTH, y * self.ref_th) for x, y in self.points]
        self.last = None      # last shown border (fractions of the picture)
        self.smooth = None    # smoothed border used for display
        self.miss = 0
        self.state = "anchor" if self.enabled else "fixed"
        if not self.enabled:
            print("Not enough detail in the drawn picture to lock the border. It stays where it was drawn.")

    def _homography(self, kp, des):
        """Homography from the drawn picture to the current picture, or None."""
        if self.ref_des is None or des is None or len(des) < 4 or len(self.ref_des) < 4:
            return None
        pairs = self.bf.knnMatch(self.ref_des, des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < MATCH_RATIO * p[1].distance]
        if len(good) < MIN_INLIERS:
            return None
        src = self.ref_kp[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if H is None or mask is None or int(mask.sum()) < MIN_INLIERS:
            return None
        if not sane(H, self.ref_th):
            return None
        return H

    def _locate(self, frame):
        """Returns the border points as fractions of this frame, or None if the scene is not found."""
        g, th = small_gray(frame)
        kp, des = self.det.detectAndCompute(g, None)
        if des is None or not kp:
            return None
        H = self._homography(kp, des)
        if H is None:
            return None
        m = cv2.perspectiveTransform(np.float32(self.ref_px).reshape(-1, 1, 2), H).reshape(-1, 2)
        out = [(float(u) / TRACK_WIDTH, float(v) / th) for u, v in m]
        if any(not (-OFF_SCREEN <= x <= 1 + OFF_SCREEN and -OFF_SCREEN <= y <= 1 + OFF_SCREEN)
               for x, y in out):
            return None
        return out

    def _set(self, state):
        if state != self.state:
            self.state = state
            print(f"Border: {state}")

    def update(self, frame):
        """Returns (border points as fractions of this frame, or None if hidden, state)."""
        if not self.enabled:
            return list(self.points), "fixed"

        out = self._locate(frame)
        if out is None:
            self.miss += 1
            if self.miss > HOLD_FRAMES:
                self.last = None
                self.smooth = None
                self._set("searching")
                return None, "searching"
            if self.last is not None:
                return list(self.last), "holding"
            return None, "searching"

        self.miss = 0
        if self.smooth is None:
            self.smooth = out
        else:
            self.smooth = [(SMOOTH * a + (1 - SMOOTH) * c, SMOOTH * b + (1 - SMOOTH) * d)
                           for (a, b), (c, d) in zip(out, self.smooth)]
        self.last = self.smooth
        self._set("anchor")
        return list(self.smooth), "anchor"


# ------------------------------------------------------------ drawing on the picture
def draw_line(frame, A, B):
    h, w = frame.shape[:2]
    inside, p1, p2 = cv2.clipLine((0, 0, w, h),
                                  (int(round(A[0])), int(round(A[1]))),
                                  (int(round(B[0])), int(round(B[1]))))
    if inside:
        cv2.line(frame, p1, p2, LINE_COLOR, LINE_WIDTH)


def draw_zone(frame, zone):
    h, w = frame.shape[:2]
    poly = np.array([(int(round(x * w)), int(round(y * h))) for x, y in zone], np.int32)
    x, y, bw, bh = cv2.boundingRect(poly)
    x, y = max(x, 0), max(y, 0)
    x2, y2 = min(x + bw, w), min(y + bh, h)
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    layer = roi.copy()
    cv2.fillPoly(layer, [poly - [x, y]], (60, 60, 230))
    frame[y:y2, x:x2] = cv2.addWeighted(layer, 0.25, roi, 0.75, 0)
    cv2.polylines(frame, [poly], True, (60, 60, 230), 2)


STATE_TEXT = {
    "anchor": "Border: locked to the drawn place",
    "holding": "Border: holding position",
    "searching": "Border: out of view, turn back to the drawn view",
    "fixed": "Border: fixed",
}


def compose(frame, pts, dets, state):
    """Draw the border, zone and boxes on one picture. People below the line stay in front of it."""
    h, w = frame.shape[:2]
    img = frame.copy()
    A = B = None
    if pts:
        A = (pts[0][0] * w, pts[0][1] * h)
        B = (pts[1][0] * w, pts[1][1] * h)
        draw_line(img, A, B)
        if len(pts) > 2:
            draw_zone(img, pts[2:])

    for d in dets:
        bx1, by1, bx2, by2 = d["box"]
        x1, y1 = max(int(bx1 * w), 0), max(int(by1 * h), 0)
        x2, y2 = min(int(bx2 * w), w), min(int(by2 * h), h)
        if x2 <= x1 or y2 <= y1 or A is None or abs(B[0] - A[0]) < 1e-6:
            continue
        line_y = A[1] + ((x1 + x2) / 2.0 - A[0]) * (B[1] - A[1]) / (B[0] - A[0])
        if y2 >= line_y:
            img[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    for d in dets:
        if not d["alert"]:
            continue
        bx1, by1, bx2, by2 = d["box"]
        x1, y1 = max(int(bx1 * w), 0), max(int(by1 * h), 0)
        x2, y2 = min(int(bx2 * w), w), min(int(by2 * h), h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"ALERT: {d['label']} {d['tid']}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if state:
        cv2.putText(img, STATE_TEXT.get(state, state), (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_COLOR, 2)
    return img


def publish_frame(img, payload=None):
    """Store the picture for IP cameras and video files, and the overlay for the device camera."""
    h0, w0 = img.shape[:2]
    if w0 > STREAM_WIDTH:
        img = cv2.resize(img, (STREAM_WIDTH, int(h0 * STREAM_WIDTH / w0)), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
    with LOCK:
        if ok:
            S["jpeg"] = buf.tobytes()
            S["jpeg_id"] += 1
        if payload is not None:
            S["overlay"] = payload


def set_status(text):
    with LOCK:
        if S["running"] and S["status"] != "Stopping...":
            S["status"] = text


# ------------------------------------------------------------ reading cameras and video files
class CaptureReader(threading.Thread):
    """Reads an IP camera or a video file and keeps the newest picture in RAW.
    Video files are read at their own speed."""

    def __init__(self, source, is_file):
        super().__init__(daemon=True)
        self.is_file = is_file
        self.cap = cv2.VideoCapture(source)
        self.opened = self.cap.isOpened()
        if not is_file:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        fps = self.cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        self.period = 1.0 / fps if 1 < fps < 120 else 1.0 / 25
        self.halt = threading.Event()
        self.ended = threading.Event()

    def run(self):
        try:
            next_t = time.time()
            while not self.halt.is_set():
                ok, f = self.cap.read()
                if not ok:
                    if self.is_file:
                        self.ended.set()
                        break
                    time.sleep(0.05)
                    continue
                RAW.put((f, None))
                if self.is_file:
                    next_t += self.period
                    spare = next_t - time.time()
                    if spare > 0:
                        time.sleep(spare)
                    else:
                        next_t = time.time()
        finally:
            self.cap.release()

    def stop(self):
        self.halt.set()
        if self.is_alive():
            self.join(timeout=3)
        else:
            self.cap.release()


# ------------------------------------------------------------ the three working parts
def pipeline(source, tracker, halt, reader):
    """Works through every new picture in order. The border is worked out for each picture
    and sent with that picture's ID, so the browser draws it on exactly that picture."""
    device = source == "device"
    live_input = device or is_live(source)
    gap = 1.0 / STREAM_FPS
    last, last_new, waiting = 0, time.time(), False
    while not halt.is_set():
        t0 = time.time()
        try:
            item, seq = RAW.get()
            if item is None or seq == last:
                if reader is not None and reader.is_file and reader.ended.is_set():
                    break
                if live_input and not waiting and time.time() - last_new > WAIT_AFTER:
                    waiting = True
                    set_status("Waiting for camera...")
                time.sleep(0.004)
                continue
            img, cid = item
            fid = cid if cid is not None else seq
            last, last_new = seq, time.time()
            if waiting:
                waiting = False
                set_status("Running")

            pts, state = tracker.update(img)
            DETECT_IN.put((img, pts))

            now = time.time()
            with LOCK:
                dets = [d for d in S["dets"] if now - d["t"] < BOX_MAX_AGE]
            payload = {
                "fid": fid,
                "line": [list(p) for p in pts[:2]] if pts else None,
                "zone": [list(p) for p in pts[2:]] if pts and len(pts) > 2 else [],
                "state": state,
                "boxes": [{"box": list(d["box"]), "label": d["label"],
                           "tid": d["tid"], "alert": d["alert"]} for d in dets],
                "t": now,
            }
            if device:
                with LOCK:
                    S["overlay"] = payload
            else:
                publish_frame(compose(img, pts, dets, state), payload)
        except Exception as e:
            print("Display step skipped:", e)
            time.sleep(0.05)
        spare = gap - (time.time() - t0)
        if spare > 0:
            time.sleep(spare)


def detect_worker(class_ids, halt):
    """Finds people, cars and animals in the newest picture. Crossings and zone entries are
    checked against the border position for that same picture."""
    model = get_model()
    model.predictor = None
    crossed, zone_alerted, last_side = set(), set(), {}
    last_seq, n, t_det, t0, last_err = 0, 0, 0.0, time.time(), 0.0

    while not halt.is_set():
        item, seq = DETECT_IN.get()
        if item is None or seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq
        img, pts = item
        try:
            h, w = img.shape[:2]
            line = pts[:2] if pts else None
            zone = pts[2:] if pts else []
            A = B = None
            if line:
                A = (line[0][0] * w, line[0][1] * h)
                B = (line[1][0] * w, line[1][1] * h)
            zone_px = np.float32([(x * w, y * h) for x, y in zone]) if len(zone) >= 3 else None

            ta = time.time()
            result = model.track(img, persist=True, classes=class_ids, imgsz=INFER_SIZE,
                                 tracker="bytetrack.yaml", verbose=False)[0]
            t_det += time.time() - ta
            n += 1
            now = time.time()

            found, events = [], []
            if result.boxes is not None and result.boxes.id is not None:
                for box, tid, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    tid = int(tid)
                    label = result.names[int(cls)]
                    foot = ((x1 + x2) / 2.0, y2)

                    if A is not None:
                        side_now = 1 if side(foot, A, B) > 0 else -1
                        prev = last_side.get(tid)
                        last_side[tid] = side_now
                        if (prev is not None and prev != side_now and tid not in crossed
                                and near_segment(foot, A, B)):
                            crossed.add(tid)
                            events.append((tid, label, "side 1 to 2" if prev < 0 else "side 2 to 1"))
                            with LOCK:
                                S["crossings"] += 1

                    if zone_px is not None and tid not in zone_alerted:
                        if cv2.pointPolygonTest(zone_px, foot, False) >= 0:
                            zone_alerted.add(tid)
                            events.append((tid, label, "entered restricted zone"))
                            with LOCK:
                                S["crossings"] += 1

                    found.append({"box": (x1 / w, y1 / h, x2 / w, y2 / h), "label": label,
                                  "tid": tid, "alert": tid in crossed or tid in zone_alerted,
                                  "t": now})

            with LOCK:
                new_ids = {f["tid"] for f in found}
                keep = [d for d in S["dets"] if now - d["t"] < BOX_MAX_AGE and d["tid"] not in new_ids]
                S["dets"] = found + keep

            for tid, label, direction in events:
                snap = compose(img, pts, found + keep, None)
                rec = log_alert(snap, tid, label, direction)
                with LOCK:
                    S["events"].insert(0, rec)
                    del S["events"][50:]

            if SHOW_TIMING and n % 30 == 0:
                el = time.time() - t0
                print(f"Speed {n / el:.1f} detections/s | detect {t_det / n * 1000:.0f} ms")
        except Exception as e:
            if time.time() - last_err > 5:
                print("Detection skipped a picture:", e)
                last_err = time.time()


def worker(source, points, class_ids, ref_jpg):
    halt = threading.Event()
    reader = pipe = det = None
    try:
        ref = decode_jpeg(ref_jpg)
        if ref is None:
            raise RuntimeError("Load the preview first, then draw the border on it.")
        tracker = BorderTracker(ref, points)
        get_model()
        RAW.clear()
        DETECT_IN.clear()
        with LOCK:
            S["dets"] = []
            S["overlay"] = None

        if source != "device":
            reader = CaptureReader(source, is_file=not is_live(source))
            if not reader.opened:
                raise RuntimeError("Could not open this source. Check the URL, file or camera.")
            reader.start()

        det = threading.Thread(target=detect_worker, args=(class_ids, halt), daemon=True)
        pipe = threading.Thread(target=pipeline, args=(source, tracker, halt, reader), daemon=True)
        det.start()
        pipe.start()
        with LOCK:
            if S["status"] == "Starting...":
                S["status"] = "Running"
        while pipe.is_alive() and not S["stop"]:
            time.sleep(0.1)
    except Exception as e:
        with LOCK:
            S["error"] = str(e)
            S["status"] = "Error"
    finally:
        halt.set()
        if pipe is not None:
            pipe.join(timeout=3)
        if det is not None:
            det.join(timeout=3)
        if reader is not None:
            reader.stop()
        with LOCK:
            if S["status"] != "Error":
                S["status"] = "Stopped" if S["stop"] else "Video ended"
            S["running"] = False


# ------------------------------------------------------------ pages (the same pages also work locally)
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "front.html")


@app.get("/gallery")
def gallery():
    return send_from_directory(FRONTEND_DIR, "gallery.html")


# ------------------------------------------------------------ routes
@app.get("/api/ping")
def ping():
    return jsonify(ok=True, message="Connection works. The web page can use this server.")


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(ok=False, error="No file received."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_VIDEO:
        return jsonify(ok=False, error="Choose a video file (.mp4, .avi, .mov, .mkv or .webm)."), 400
    safe = secure_filename(f.filename) or "video" + ext
    path = os.path.join(UPLOAD_DIR, f"{datetime.now():%Y%m%d_%H%M%S}_{safe}")
    f.save(path)
    return jsonify(ok=True, source=os.path.abspath(path))


@app.post("/api/preview")
def preview():
    data = request.get_json(silent=True) or {}
    source = parse_source(data.get("source", ""))
    if source == "":
        return jsonify(ok=False, error="Enter a source first."), 400
    if source == "device":
        return jsonify(ok=False, error="Use the device camera preview."), 400
    frame = read_first_frame(source)
    if frame is None:
        return jsonify(ok=False,
                       error="Could not open this source. Check the URL, file or camera."), 400
    ok, buf = cv2.imencode(".jpg", frame)
    if ok:
        with LOCK:
            S["preview"] = buf.tobytes()
    return jsonify(ok=True)


@app.post("/api/preview_frame")
def preview_frame():
    img = decode_jpeg(request.get_data())
    if img is None:
        return jsonify(ok=False, error="The camera picture could not be read."), 400
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        with LOCK:
            S["preview"] = buf.tobytes()
    return jsonify(ok=True)


@app.post("/api/frame")
def frame_in():
    """One picture from the browser's camera. Only the newest one is kept.
    The browser sends its frame number in the X-Frame-Id header, so the border can be
    matched to exactly that picture."""
    with LOCK:
        running = S["running"]
    if not running:
        return jsonify(ok=True)
    img = decode_jpeg(request.get_data())
    if img is None:
        return jsonify(ok=False), 400
    try:
        cid = int(request.headers.get("X-Frame-Id", ""))
    except ValueError:
        cid = None
    RAW.put((img, cid))
    return jsonify(ok=True)


@app.get("/api/preview.jpg")
def preview_image():
    with LOCK:
        jpg = S["preview"]
    if jpg is None:
        return Response(status=404)
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/start")
def start():
    data = request.get_json(silent=True) or {}
    source = parse_source(data.get("source", ""))
    if source == "":
        return jsonify(ok=False, error="Enter a source first."), 400
    line = valid_line(data.get("line"))
    if line is None:
        return jsonify(ok=False, error="Draw the border first: click two points on the picture."), 400
    zone = valid_zone(data.get("zone"))
    if zone is None:
        return jsonify(ok=False,
                       error=f"The restricted zone needs {MIN_ZONE_POINTS} to {MAX_ZONE_POINTS} points."), 400
    class_ids = sorted({i for c in data.get("classes", []) if c in CLASSES for i in CLASSES[c]})
    if not class_ids:
        return jsonify(ok=False, error="Pick at least one thing to detect."), 400

    with LOCK:
        if S["running"]:
            return jsonify(ok=False, error="Detection is already running."), 409
        ref_jpg = S["preview"]
        S.update(running=True, stop=False, jpeg=None, crossings=0,
                 status="Starting...", error="", events=[], dets=[], overlay=None, source=source)
    points = line + zone
    threading.Thread(target=worker, args=(source, points, class_ids, ref_jpg),
                     daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    with LOCK:
        if S["running"]:
            S["stop"] = True
            S["status"] = "Stopping..."
    return jsonify(ok=True)


@app.get("/api/status")
def status():
    with LOCK:
        return jsonify(running=S["running"], crossings=S["crossings"],
                       status=S["status"], error=S["error"],
                       device=(S.get("source") == "device"))


@app.get("/api/overlay")
def overlay_state():
    """Border, zone and boxes for the newest analysed picture, with that picture's ID (fid)."""
    with LOCK:
        return jsonify(S["overlay"] or {})


@app.get("/api/alerts")
def alerts():
    with LOCK:
        return jsonify(S["events"][:12])


@app.get("/api/history")
def history():
    out = []
    for r in reversed(clean_log_rows()[-500:]):
        name = os.path.basename(r.get("snapshot", "") or "")
        has_picture = bool(name) and os.path.exists(os.path.join(ALERT_DIR, name))
        out.append({
            "time": r.get("time", ""),
            "id": r.get("track_id", ""),
            "class": r.get("class", "") or "unknown",
            "direction": r.get("direction", ""),
            "image": "/alerts/" + name if has_picture else "",
        })
    return jsonify(out)


@app.get("/api/photos")
def photos():
    out = []
    for r in reversed(clean_log_rows()):
        name = os.path.basename(r.get("snapshot", "") or "")
        if not name or not os.path.exists(os.path.join(ALERT_DIR, name)):
            continue
        out.append({
            "name": name,
            "time": r.get("time", ""),
            "id": r.get("track_id", ""),
            "class": r.get("class", "") or "unknown",
            "direction": r.get("direction", ""),
            "image": "/alerts/" + name,
            "download": "/api/photo/" + name + "/download",
        })
    return jsonify(out)


@app.delete("/api/photos/<path:name>")
def photo_delete(name):
    name = os.path.basename(name)
    with LOG_LOCK:
        rows = clean_log_rows()
        keep = [r for r in rows if os.path.basename(r.get("snapshot", "") or "") != name]
        write_log_rows(keep)
        try:
            os.remove(os.path.join(ALERT_DIR, name))
        except OSError:
            pass
    return jsonify(ok=True)


@app.get("/api/photo/<path:name>/download")
def photo_download(name):
    return send_from_directory(ALERT_DIR, os.path.basename(name), as_attachment=True)


@app.get("/api/log.csv")
def log_csv():
    return send_from_directory(ALERT_DIR, "log.csv", as_attachment=True)


@app.get("/alerts/<path:name>")
def alert_image(name):
    return send_from_directory(ALERT_DIR, os.path.basename(name))


@app.get("/video_feed")
def video_feed():
    """Live picture for IP cameras and video files. Every frame already has the overlay drawn in."""
    def gen():
        last, last_send = -1, 0.0
        while True:
            with LOCK:
                jpg, ver = S["jpeg"], S["jpeg_id"]
            now = time.time()
            if jpg and (ver != last or now - last_send > 1.0):
                last, last_send = ver, now
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            else:
                time.sleep(0.02)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.errorhandler(413)
def too_large(_):
    return jsonify(ok=False, error="That file is too large."), 413


# ------------------------------------------------------------ servers
def run_https():
    try:
        app.run(host="0.0.0.0", port=PHONE_PORT, threaded=True, debug=False,
                use_reloader=False, ssl_context="adhoc")
    except Exception as e:
        print(f"\nPhone server did not start: {e}\n")


def run_http():
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False, use_reloader=False)
    except OSError as e:
        print(f"\nCould not start on port {PORT}: {e}")
        print("Another program is using it. Run:  netstat -ano | findstr :5000")
        print("then stop that PID with:  taskkill /PID <number> /F\n")


if __name__ == "__main__":
    from flask import cli as flask_cli

    flask_cli.show_server_banner = lambda *args, **kwargs: None
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    ip = lan_ip()
    print("\n" + "=" * 60)
    print(" Keep this window open while you use the web page.")
    print("")
    print(" Server address to type on the web page:")
    print(f"   This PC:   http://127.0.0.1:{PORT}")
    if https_ready():
        print(f"   Phone:     https://{ip}:{PHONE_PORT}")
        print("              (open that address once on the phone and tap Advanced, Proceed)")
    print("=" * 60 + "\n")

    if https_ready():
        threading.Thread(target=run_https, daemon=True).start()
    run_http()
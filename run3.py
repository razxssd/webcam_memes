import argparse
import collections
import threading
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


BASE_DIR = Path(__file__).resolve().parent
FACE_MODEL_PATH = BASE_DIR / "face_landmarker.task"
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MEME_PATH = BASE_DIR / "memes" / "image.png"
# YUTA_MEME_PATH disabilitato: la relativa posa non viene caricata o rilevata.
MEMEOMG_PATH = BASE_DIR / "memes" / "memeomg.png"
STOP_MEME_PATH = BASE_DIR / "memes" / "stop.png"
SEGMENTER_MODEL_PATH = BASE_DIR / "selfie_segmenter.tflite"
KAWARIMI_IMAGE_PATH = BASE_DIR / "memes" / "tronco.jpeg"
KAWARIMI_SMOKE_PATH = BASE_DIR / "memes" / "nuvole.jpeg"

WINDOW_NAME = "Meme Pose Tracker"
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
VIRTUAL_CAMERA_FPS = 30

# Tecnica della sostituzione: fotogrammi consecutivi del sigillo per attivarla,
# durata della nuvola di fumo e pausa minima tra due attivazioni (secondi).
KAWARIMI_REQUIRED_FRAMES = 4
KAWARIMI_SMOKE_SECONDS = 0.7
KAWARIMI_COOLDOWN_SECONDS = 1.5
# Distanza verticale minima (in proporzione alla mano) tra le punte di indice e
# medio e quelle di anulare e mignolo perche questi contino come piegati.
KAWARIMI_CURLED_TIP_GAP = 0.25
# Fuoco dalla mano dopo uno schiocco di dita. Le soglie sono misurate su
# registrazioni reali: pollice e medio a contatto, poi il medio sbatte sul palmo.
SNAP_PINCH_MAX = 0.18          # distanza pollice-medio (in mani) per il contatto
SNAP_RELEASE_MIN = 0.40        # distanza pollice-medio dopo lo scatto
SNAP_MIDDLE_CURLED_MAX = 0.75  # punta del medio vicina al polso dopo lo scatto
SNAP_WINDOW_SECONDS = 0.25     # tempo massimo tra contatto e scatto
AUDIO_CLICK_FACTOR = 8.0       # clic: energia acuta oltre N volte il rumore
FIRE_MAX_SECONDS = 10.0
FIRE_FIST_FRAMES = 6           # fotogrammi di pugno chiuso per spegnere

# Sfondo virtuale: None (sfondo reale), "blur" (sfocato) oppure il percorso di
# un'immagine. Si puo cambiare anche all'avvio con --sfondo. Usandolo,
# disattiva lo sfondo virtuale di Meet/Zoom: il tronco resta sopra al nostro.
VIRTUAL_BACKGROUND = "memes/background.png"

# Modalita debug (--debug oppure MEME_DEBUG=1): mostra nell'anteprima locale
# riquadri delle mani, stato e confidenza, e stampa le condizioni del sigillo.
# Il video inviato alla webcam virtuale resta comunque pulito.
DEBUG = os.environ.get("MEME_DEBUG") == "1"
_last_debug_print = 0.0

# Indici MediaPipe delle parti principali della mano.
WRIST = 0
PALM_INDICES = (0, 5, 9, 13, 17)
FINGERTIP_INDICES = (8, 12, 16, 20)


def load_meme(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {path}")
    return image


def load_rgba(path, min_brightness=175):
    """Carica un'immagine con trasparenza; se manca, scontorna lo sfondo chiaro.

    min_brightness: luminosita minima (0-255) dei pixel considerati sfondo.
    175 rimuove anche una scacchiera grigia; 250 solo il bianco quasi puro.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image

    # Sfondo bianco o scacchiera "finta trasparenza": pixel chiari e grigi
    # collegati al bordo dell'immagine.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    light = ((hsv[..., 2] >= min_brightness) & (hsv[..., 1] <= 30)).astype(np.uint8)
    count, labels = cv2.connectedComponents(light, connectivity=4)
    border_labels = np.unique(np.concatenate([
        labels[0], labels[-1], labels[:, 0], labels[:, -1],
    ]))
    border_labels = border_labels[border_labels != 0]
    background = np.isin(labels, border_labels).astype(np.uint8)
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = (1 - background) * 255
    alpha = cv2.morphologyEx(alpha.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1)

    ys, xs = np.nonzero(alpha > 128)
    rgba = np.dstack([image, alpha])
    if len(xs):
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba


def open_camera(index=0):
    """Apre la webcam provando i backend del sistema e attende il primo frame."""
    if sys.platform == "darwin":
        backends = (
            ("AVFoundation", cv2.CAP_AVFOUNDATION),
            ("automatico", cv2.CAP_ANY),
        )
    else:
        backends = (
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
            ("automatico", cv2.CAP_ANY),
        )
    attempts = []

    for backend_name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            attempts.append(f"{backend_name}: non disponibile")
            cap.release()
            continue

        # 1280x720 e una modalita molto piu comune di 960x540. Il frame viene
        # ridimensionato solo dopo l'acquisizione, per la visualizzazione.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print(f"[WEBCAM] Backend attivo: {backend_name}")
                return cap, frame
            time.sleep(0.05)

        attempts.append(f"{backend_name}: aperta, ma nessun frame ricevuto")
        cap.release()

    details = "; ".join(attempts)
    raise RuntimeError(
        "Non riesco a ricevere immagini dalla webcam. "
        "Chiudi eventuali programmi che la stanno usando e controlla i permessi "
        "della fotocamera (su macOS: Impostazioni di Sistema > Privacy e "
        f"sicurezza > Fotocamera, per il Terminale). Tentativi: {details}"
    )


def face_bounding_box(face_landmarks):
    xs = [landmark.x for landmark in face_landmarks]
    ys = [landmark.y for landmark in face_landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def hand_features(hand_landmarks):
    palm_x = float(np.mean([hand_landmarks[i].x for i in PALM_INDICES]))
    palm_y = float(np.mean([hand_landmarks[i].y for i in PALM_INDICES]))
    fingertip_y = float(np.mean([hand_landmarks[i].y for i in FINGERTIP_INDICES]))

    xs = [landmark.x for landmark in hand_landmarks]
    ys = [landmark.y for landmark in hand_landmarks]

    return {
        "landmarks": hand_landmarks,
        "center": (palm_x, palm_y),
        "wrist_y": hand_landmarks[WRIST].y,
        "fingertip_y": fingertip_y,
        "bbox": (min(xs), min(ys), max(xs), max(ys)),
    }


def hands_on_face_pose(hands, face_box):
    """Riconosce due mani verticali e ravvicinate davanti al volto."""
    if len(hands) < 2 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    features = sorted(
        (hand_features(hand) for hand in hands[:2]),
        key=lambda item: item["center"][0],
    )

    valid_hands = []
    overlap_scores = []

    for hand in features:
        center_x, center_y = hand["center"]
        landmarks = hand["landmarks"]

        center_near_face = (
            fx1 - 0.20 * face_width <= center_x <= fx2 + 0.20 * face_width
            and fy1 + 0.18 * face_height <= center_y <= fy2 + 0.35 * face_height
        )

        # Nel gesto di riferimento le dita sono sopra al polso.
        vertical_hand = hand["wrist_y"] - hand["fingertip_y"] >= 0.08 * face_height

        points_near_face = sum(
            fx1 - 0.18 * face_width <= landmark.x <= fx2 + 0.18 * face_width
            and fy1 - 0.08 * face_height <= landmark.y <= fy2 + 0.35 * face_height
            for landmark in landmarks
        )
        overlap = points_near_face / len(landmarks)

        open_hand = extended_finger_count(landmarks) >= 3
        valid_hands.append(
            center_near_face
            and vertical_hand
            and overlap >= 0.45
            and open_hand
        )
        overlap_scores.append(overlap)

    left_hand, right_hand = features
    left_x = left_hand["center"][0]
    right_x = right_hand["center"][0]
    horizontal_gap = (right_x - left_x) / face_width

    # Le mani devono stare ai due lati della zona centrale del volto.
    balanced = (
        left_x <= face_center_x + 0.16 * face_width
        and right_x >= face_center_x - 0.16 * face_width
    )
    close_together = 0.04 <= horizontal_gap <= 0.90

    score = float(np.mean(overlap_scores))
    matched = all(valid_hands) and balanced and close_together
    return matched, score


def extended_finger_states(landmarks):
    """Restituisce lo stato di indice, medio, anulare e mignolo."""
    wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
    states = []
    for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip = np.array([landmarks[tip_index].x, landmarks[tip_index].y])
        pip = np.array([landmarks[pip_index].x, landmarks[pip_index].y])
        states.append(
            np.linalg.norm(tip - wrist) >= 1.18 * np.linalg.norm(pip - wrist)
        )
    return states


def extended_finger_count(landmarks):
    """Conta indice, medio, anulare e mignolo visibilmente distesi."""
    return sum(extended_finger_states(landmarks))


def yuta_hand_pose(hands, face_box):
    """Riconosce una mano aperta, grande e distesa davanti al volto."""
    if len(hands) != 1 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    hand = hand_features(hands[0])
    landmarks = hand["landmarks"]
    center_x, center_y = hand["center"]
    hx1, hy1, hx2, hy2 = hand["bbox"]

    center_on_face = (
        fx1 - 0.16 * face_width <= center_x <= fx2 + 0.22 * face_width
        and fy1 + 0.12 * face_height <= center_y <= fy2 + 0.30 * face_height
    )

    points_near_face = sum(
        fx1 - 0.22 * face_width <= landmark.x <= fx2 + 0.25 * face_width
        and fy1 - 0.18 * face_height <= landmark.y <= fy2 + 0.45 * face_height
        for landmark in landmarks
    )
    overlap = points_near_face / len(landmarks)

    hand_width = hx2 - hx1
    hand_height = hy2 - hy1
    large_hand = hand_width >= 0.38 * face_width and hand_height >= 0.55 * face_height
    vertical_hand = hand["wrist_y"] - hand["fingertip_y"] >= 0.10 * face_height

    # Una falange e considerata distesa se la punta e sensibilmente piu
    # lontana dal polso rispetto alla sua articolazione intermedia.
    extended_fingers = extended_finger_count(landmarks)

    fingertip_xs = [landmarks[i].x for i in (4, 8, 12, 16, 20)]
    fingers_spread = max(fingertip_xs) - min(fingertip_xs) >= 0.38 * face_width
    open_hand = extended_fingers >= 3 and fingers_spread

    size_score = min(1.0, hand_height / (0.85 * face_height))
    finger_score = extended_fingers / 4
    score = float(np.mean([overlap, size_score, finger_score]))
    matched = center_on_face and overlap >= 0.50 and large_hand and vertical_hand and open_hand
    return matched, score


def hands_behind_head_pose(hands, face_box):
    """Riconosce due mani separate ai lati della testa, dietro le orecchie."""
    if len(hands) < 2 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2

    features = sorted(
        (hand_features(hand) for hand in hands[:2]),
        key=lambda item: item["center"][0],
    )
    left_hand, right_hand = features
    left_x, left_y = left_hand["center"]
    right_x, right_y = right_hand["center"]

    left_at_head_side = (
        fx1 - 0.95 * face_width <= left_x <= fx1 + 0.18 * face_width
        and left_x <= face_center_x - 0.24 * face_width
    )
    right_at_head_side = (
        fx2 - 0.18 * face_width <= right_x <= fx2 + 0.95 * face_width
        and right_x >= face_center_x + 0.24 * face_width
    )
    ear_height = all(
        fy1 - 0.18 * face_height <= y <= fy2 + 0.16 * face_height
        for y in (left_y, right_y)
    )
    horizontally_separated = right_x - left_x >= 0.78 * face_width
    vertically_aligned = abs(right_y - left_y) <= 0.48 * face_height

    large_enough = True
    for hand in features:
        hx1, hy1, hx2, hy2 = hand["bbox"]
        hand_width = hx2 - hx1
        hand_height = hy2 - hy1
        if hand_width < 0.22 * face_width and hand_height < 0.32 * face_height:
            large_enough = False

    separation_score = min(1.0, (right_x - left_x) / (1.20 * face_width))
    alignment_score = max(0.0, 1.0 - abs(right_y - left_y) / face_height)
    side_score = float(left_at_head_side and right_at_head_side)
    score = float(np.mean([separation_score, alignment_score, side_score]))
    matched = (
        left_at_head_side
        and right_at_head_side
        and ear_height
        and horizontally_separated
        and vertically_aligned
        and large_enough
    )
    return matched, score


def timeout_hands_pose(hands, face_box):
    """Riconosce due mani aperte e perpendicolari che formano una T."""
    if len(hands) < 2 or face_box is None:
        return False, 0.0

    fx1, fy1, fx2, fy2 = face_box
    face_width = max(0.001, fx2 - fx1)
    face_height = max(0.001, fy2 - fy1)
    face_center_x = (fx1 + fx2) / 2
    features = [hand_features(hand) for hand in hands[:2]]

    axes = []
    for hand in features:
        landmarks = hand["landmarks"]
        wrist = np.array([landmarks[WRIST].x, landmarks[WRIST].y])
        fingertip_center = np.mean(
            [[landmarks[i].x, landmarks[i].y] for i in FINGERTIP_INDICES],
            axis=0,
        )
        axes.append(fingertip_center - wrist)

    first_axis, second_axis = axes
    first_horizontal = abs(first_axis[0]) >= 1.25 * abs(first_axis[1])
    second_horizontal = abs(second_axis[0]) >= 1.25 * abs(second_axis[1])
    first_vertical = abs(first_axis[1]) >= 1.25 * abs(first_axis[0])
    second_vertical = abs(second_axis[1]) >= 1.25 * abs(second_axis[0])
    one_horizontal_one_vertical = (
        (first_horizontal and second_vertical)
        or (second_horizontal and first_vertical)
    )

    first_norm = max(0.001, float(np.linalg.norm(first_axis)))
    second_norm = max(0.001, float(np.linalg.norm(second_axis)))
    cosine = abs(float(np.dot(first_axis, second_axis))) / (first_norm * second_norm)
    perpendicular = cosine <= 0.48

    both_open = all(
        extended_finger_count(hand["landmarks"]) >= 3
        for hand in features
    )

    ax1, ay1, ax2, ay2 = features[0]["bbox"]
    bx1, by1, bx2, by2 = features[1]["bbox"]
    first_size = max(ax2 - ax1, ay2 - ay1)
    second_size = max(bx2 - bx1, by2 - by1)
    hand_scale = max(0.001, (first_size + second_size) / 2)
    horizontal_distance = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    vertical_distance = max(0.0, max(ay1, by1) - min(ay2, by2))
    hands_touching = (
        horizontal_distance <= 0.20 * hand_scale
        and vertical_distance <= 0.20 * hand_scale
    )

    mean_x = float(np.mean([hand["center"][0] for hand in features]))
    mean_y = float(np.mean([hand["center"][1] for hand in features]))
    in_front_of_chest = (
        abs(mean_x - face_center_x) <= 1.10 * face_width
        and fy2 - 0.22 * face_height <= mean_y <= fy2 + 1.80 * face_height
    )

    angle_score = max(0.0, 1.0 - cosine)
    orientation_score = float(one_horizontal_one_vertical)
    open_score = min(
        1.0,
        sum(extended_finger_count(hand["landmarks"]) for hand in features) / 8,
    )
    score = float(np.mean([angle_score, orientation_score, open_score]))
    matched = (
        one_horizontal_one_vertical
        and perpendicular
        and both_open
        and hands_touching
        and in_front_of_chest
    )
    return matched, score


def square_from_hand(hand_landmarks, frame_width, frame_height):
    xs = np.array([landmark.x * frame_width for landmark in hand_landmarks])
    ys = np.array([landmark.y * frame_height for landmark in hand_landmarks])

    center_x = float((xs.min() + xs.max()) / 2)
    center_y = float((ys.min() + ys.max()) / 2)
    side = float(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.28)
    side = max(side, 72.0)

    x1 = max(0.0, center_x - side / 2)
    y1 = max(0.0, center_y - side / 2)
    x2 = min(float(frame_width - 1), center_x + side / 2)
    y2 = min(float(frame_height - 1), center_y + side / 2)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def update_squares(hand_landmarks, old_squares, frame_width, frame_height):
    ordered_hands = sorted(
        hand_landmarks[:2],
        key=lambda hand: float(np.mean([landmark.x for landmark in hand])),
    )

    new_squares = {}
    for index, hand in enumerate(ordered_hands):
        target = square_from_hand(hand, frame_width, frame_height)
        if index in old_squares:
            target = 0.68 * old_squares[index] + 0.32 * target
        new_squares[index] = target
    return new_squares


def draw_analysis_square(frame, square, label, phase):
    x1, y1, x2, y2 = square.astype(int)
    color = (0, 255, 210)
    dark_color = (0, 110, 90)

    cv2.rectangle(frame, (x1, y1), (x2, y2), dark_color, 1, cv2.LINE_AA)

    corner = max(12, min(28, (x2 - x1) // 5))
    thickness = 3
    segments = (
        ((x1, y1 + corner), (x1, y1), (x1 + corner, y1)),
        ((x2 - corner, y1), (x2, y1), (x2, y1 + corner)),
        ((x1, y2 - corner), (x1, y2), (x1 + corner, y2)),
        ((x2 - corner, y2), (x2, y2), (x2, y2 - corner)),
    )
    for first, middle, last in segments:
        cv2.line(frame, first, middle, color, thickness, cv2.LINE_AA)
        cv2.line(frame, middle, last, color, thickness, cv2.LINE_AA)

    scan_y = y1 + int((y2 - y1) * phase)
    cv2.line(frame, (x1 + 4, scan_y), (x2 - 4, scan_y), color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def resize_contain(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def overlay_meme_on_face(frame, meme, face_box):
    """Sovrappone il meme al volto e restituisce il riquadro occupato."""
    if face_box is None:
        return None

    frame_height, frame_width = frame.shape[:2]
    fx1, fy1, fx2, fy2 = face_box
    face_width = max(1, round((fx2 - fx1) * frame_width))
    face_height = max(1, round((fy2 - fy1) * frame_height))
    center_x = round(((fx1 + fx2) / 2) * frame_width)
    center_y = round(((fy1 + fy2) / 2) * frame_height)

    # L'area e abbastanza grande da coprire il viso senza occupare tutto il video.
    max_width = min(frame_width, round(face_width * 1.70))
    max_height = min(frame_height, round(face_height * 1.50))
    shown = resize_contain(meme, max_width, max_height)
    image_height, image_width = shown.shape[:2]

    destination_x1 = center_x - image_width // 2
    destination_y1 = center_y - image_height // 2
    destination_x2 = destination_x1 + image_width
    destination_y2 = destination_y1 + image_height

    # Ritaglia solamente la parte che eventualmente supera i bordi del frame.
    source_x1 = max(0, -destination_x1)
    source_y1 = max(0, -destination_y1)
    source_x2 = image_width - max(0, destination_x2 - frame_width)
    source_y2 = image_height - max(0, destination_y2 - frame_height)

    destination_x1 = max(0, destination_x1)
    destination_y1 = max(0, destination_y1)
    destination_x2 = min(frame_width, destination_x2)
    destination_y2 = min(frame_height, destination_y2)

    if destination_x1 >= destination_x2 or destination_y1 >= destination_y2:
        return None

    frame[destination_y1:destination_y2, destination_x1:destination_x2] = shown[
        source_y1:source_y2,
        source_x1:source_x2,
    ]
    return destination_x1, destination_y1, destination_x2, destination_y2


def kawarimi_seal_pose(hands, face_box):
    """Riconosce il sigillo della Tigre: mani unite, indice e medio in su."""
    if len(hands) < 2:
        return False, 0.0

    features = [hand_features(hand) for hand in hands[:2]]
    first, second = features
    ax1, ay1, ax2, ay2 = first["bbox"]
    bx1, by1, bx2, by2 = second["bbox"]
    hand_scale = max(
        0.001,
        (max(ax2 - ax1, ay2 - ay1) + max(bx2 - bx1, by2 - by1)) / 2,
    )

    center_distance = float(np.hypot(
        first["center"][0] - second["center"][0],
        first["center"][1] - second["center"][1],
    ))
    hands_close = center_distance <= 1.10 * hand_scale

    # Nel sigillo le punte degli indici quasi si toccano.
    first_tip = first["landmarks"][8]
    second_tip = second["landmarks"][8]
    tips_distance = float(np.hypot(first_tip.x - second_tip.x, first_tip.y - second_tip.y))
    tips_close = tips_distance <= 0.55 * hand_scale

    fingers_up = 0
    fingers_curled = 0
    tip_gaps = []
    for hand in features:
        landmarks = hand["landmarks"]
        index_up, middle_up, ring_up, pinky_up = extended_finger_states(landmarks)
        # Indice e medio devono puntare verso l'alto rispetto alle nocche.
        pointing_up = landmarks[8].y < landmarks[5].y and landmarks[12].y < landmarks[9].y
        fingers_up += int(index_up and pointing_up) + int(middle_up and pointing_up)

        # Anulare e mignolo intrecciati sono coperti e MediaPipe spesso li
        # stima distesi: conta anche la loro punta molto piu in basso
        # rispetto a quelle di indice e medio.
        raised_y = min(landmarks[8].y, landmarks[12].y)
        for tip_index, extended in ((16, ring_up), (20, pinky_up)):
            gap = (landmarks[tip_index].y - raised_y) / hand_scale
            tip_gaps.append(gap)
            fingers_curled += int(not extended or gap >= KAWARIMI_CURLED_TIP_GAP)

    seal_shape = fingers_up >= 3 and fingers_curled >= 3

    # Il sigillo si fa davanti al mento o al petto, non sopra gli occhi.
    if face_box is not None:
        _, fy1, _, fy2 = face_box
        lowest_center = max(first["center"][1], second["center"][1])
        below_eyes = lowest_center >= fy1 + 0.45 * (fy2 - fy1)
    else:
        below_eyes = True

    closeness_score = max(0.0, 1.0 - center_distance / (1.10 * hand_scale))
    shape_score = (fingers_up + fingers_curled) / 8
    score = float(np.mean([closeness_score, shape_score]))
    matched = hands_close and tips_close and seal_shape and below_eyes
    if DEBUG:
        global _last_debug_print
        if time.monotonic() - _last_debug_print > 0.25:
            _last_debug_print = time.monotonic()
            print(
                f"[SIGILLO] ok={matched} mani_vicine={hands_close}"
                f"({center_distance / hand_scale:.2f}<=1.10) "
                f"punte_vicine={tips_close}({tips_distance / hand_scale:.2f}<=0.55) "
                f"dita_su={fingers_up}/4(>=3) dita_piegate={fingers_curled}/4(>=3) "
                f"dislivello_punte={[round(gap, 2) for gap in tip_gaps]}(>={KAWARIMI_CURLED_TIP_GAP}) "
                f"altezza_ok={below_eyes}",
                flush=True,
            )
    return matched, score


def paste_rgba(frame, image, left, top, opacity=1.0):
    """Incolla un'immagine BGRA rispettando la trasparenza e i bordi del frame."""
    frame_height, frame_width = frame.shape[:2]
    image_height, image_width = image.shape[:2]
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(frame_width, left + image_width), min(frame_height, top + image_height)
    if x1 >= x2 or y1 >= y2:
        return
    crop = image[y1 - top:y2 - top, x1 - left:x2 - left]
    alpha = crop[..., 3:4].astype(np.float32) * (opacity / 255)
    region = frame[y1:y2, x1:x2]
    region[:] = (crop[..., :3] * alpha + region * (1 - alpha)).astype(np.uint8)


def resize_cover(image, width, height):
    """Ridimensiona e ritaglia l'immagine per riempire esattamente width x height."""
    image_height, image_width = image.shape[:2]
    scale = max(width / image_width, height / image_height)
    resized = cv2.resize(
        image,
        (max(width, round(image_width * scale)), max(height, round(image_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top:top + height, left:left + width]


def blur_background(image):
    """Sfocatura forte ed economica: calcolata a bassa risoluzione."""
    height, width = image.shape[:2]
    small = cv2.resize(image, (width // 8, height // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def soft_person_mask(person_mask):
    """Maschera della persona con bordi sfumati, pronta per la fusione."""
    soft = cv2.GaussianBlur(person_mask.astype(np.float32), (0, 0), 6)
    return np.clip(soft * 1.6, 0, 1)[..., None]


class Kawarimi:
    """Stato della tecnica: idle -> vanishing -> substituted -> appearing -> idle."""

    def __init__(self, log_image, smoke_image, background_mode=None):
        self.log_image = log_image
        self.smoke_image = smoke_image
        # None, "blur" oppure un'immagine BGR gia alla risoluzione del frame.
        self.background_mode = background_mode
        self.state = "idle"
        self.state_time = -1e9
        self.smoke_center = None
        self.smoke_size = 1.0
        self.smoke_seed = 0
        self.background = None
        self.background_known = None
        self.filled_background = None
        self.filled_time = -1e9
        self.last_person_box = None

    def trigger(self, now):
        if now - self.state_time < KAWARIMI_COOLDOWN_SECONDS:
            return
        if self.state == "idle":
            self.state = "vanishing"
        elif self.state == "substituted":
            self.state = "appearing"
        else:
            return
        self.state_time = now
        self.smoke_seed = int(now * 1000)
        self.smoke_center = None
        # Lo sfondo ricostruito viene ricalcolato con quanto imparato finora.
        self.filled_background = None

    def update_background(self, frame, person_mask):
        """Impara lo sfondo nelle zone in cui la persona non e presente."""
        if self.background is None:
            self.background = frame.copy()
            self.background_known = np.zeros(frame.shape[:2], bool)
        free = (person_mask < 0.25).astype(np.uint8)
        # Margine di sicurezza attorno alla sagoma per evitare aloni.
        free = cv2.erode(free, np.ones((15, 15), np.uint8)) > 0
        self.background[free] = frame[free]
        self.background_known |= free

    def filled(self, now):
        """Sfondo completo: le zone mai viste vengono ricostruite con inpainting."""
        if self.filled_background is None or now - self.filled_time > 1.0:
            height, width = self.background.shape[:2]
            unknown = ~self.background_known
            result = self.background.copy()
            if np.count_nonzero(unknown):
                small = cv2.resize(self.background, (width // 4, height // 4))
                small_unknown = cv2.resize(
                    unknown.astype(np.uint8) * 255,
                    (width // 4, height // 4),
                    interpolation=cv2.INTER_NEAREST,
                )
                small = cv2.inpaint(small, small_unknown, 5, cv2.INPAINT_TELEA)
                patched = cv2.GaussianBlur(cv2.resize(small, (width, height)), (0, 0), 3)
                result[unknown] = patched[unknown]
            self.filled_background = result
            self.filled_time = now
        return self.filled_background

    @staticmethod
    def person_box(person_mask):
        ys, xs = np.nonzero(person_mask > 0.5)
        if len(xs) < 500:
            return None
        return (
            int(np.percentile(xs, 2)),
            int(np.percentile(ys, 1)),
            int(np.percentile(xs, 98)),
            int(person_mask.shape[0] - 1),
        )

    def scene_background(self, now):
        """Cio che deve apparire al posto della persona quando sparisce."""
        if isinstance(self.background_mode, np.ndarray):
            return self.background_mode
        if self.background_mode == "blur":
            return blur_background(self.filled(now))
        return self.filled(now)

    def apply_virtual_background(self, frame, person_mask):
        """Sostituisce lo sfondo reale con quello virtuale, se configurato."""
        if self.background_mode is None:
            return
        if isinstance(self.background_mode, np.ndarray):
            background = self.background_mode
        else:
            background = blur_background(frame)
        soft = soft_person_mask(person_mask)
        frame[:] = (frame * soft + background * (1 - soft)).astype(np.uint8)

    def draw_substitute(self, frame, person_mask, now):
        """Cancella la persona e mette il tronco al suo posto."""
        background = self.scene_background(now)
        if self.background_mode is not None:
            # Con lo sfondo virtuale tutto il frame e gia sfondo: nessun alone.
            frame[:] = background
        else:
            soft = soft_person_mask(person_mask)
            frame[:] = (frame * (1 - soft) + background * soft).astype(np.uint8)

        box = self.person_box(person_mask) or self.last_person_box
        if box is None:
            return
        if self.last_person_box is None:
            self.last_person_box = box
        # Il tronco segue la persona con un movimento smorzato.
        self.last_person_box = tuple(
            round(0.85 * old + 0.15 * new)
            for old, new in zip(self.last_person_box, box)
        )
        x1, y1, x2, y2 = self.last_person_box
        frame_height = frame.shape[0]
        log_height = max(40, min(frame_height, round((y2 - y1) * 0.95)))
        log_width = round(log_height * self.log_image.shape[1] / self.log_image.shape[0])
        log = cv2.resize(self.log_image, (log_width, log_height), interpolation=cv2.INTER_AREA)
        paste_rgba(frame, log, (x1 + x2) // 2 - log_width // 2, frame_height - log_height)

    def draw_smoke(self, frame, progress):
        """Nuvola di fumo che si espande e poi svanisce."""
        frame_height, frame_width = frame.shape[:2]
        if self.smoke_center is None:
            box = self.last_person_box or (
                frame_width // 3, frame_height // 4, 2 * frame_width // 3, frame_height - 1
            )
            self.smoke_center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            self.smoke_size = max(box[2] - box[0], box[3] - box[1]) * 0.55
        rng = np.random.default_rng(self.smoke_seed)
        cx, cy = self.smoke_center
        layer = frame.copy()
        grow = 0.35 + 0.65 * min(1.0, progress / 0.45)
        for _ in range(34):
            angle = rng.uniform(0, 2 * np.pi)
            distance = rng.uniform(0.0, 1.0) ** 0.6 * self.smoke_size * grow
            radius = int(rng.uniform(0.22, 0.42) * self.smoke_size * grow)
            x = int(cx + np.cos(angle) * distance)
            y = int(cy + np.sin(angle) * distance * 0.9 - progress * 30)
            tone = int(rng.uniform(200, 250))
            cv2.circle(layer, (x, y), radius, (tone, tone, tone + 5), -1, cv2.LINE_AA)
        for _ in range(14):
            angle = rng.uniform(0, 2 * np.pi)
            distance = rng.uniform(0.2, 0.9) * self.smoke_size * grow
            x = int(cx + np.cos(angle) * distance)
            y = int(cy + np.sin(angle) * distance * 0.9 - progress * 30)
            radius = int(0.12 * self.smoke_size * grow)
            cv2.circle(layer, (x, y), radius, (175, 175, 182), -1, cv2.LINE_AA)
        alpha = 0.95 if progress < 0.55 else 0.95 * (1 - (progress - 0.55) / 0.45)
        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, dst=frame)

        # Sbuffi disegnati che si allargano dal centro e svaniscono per ultimi.
        wisp_width = max(20, round(self.smoke_size * (1.6 + 1.6 * progress)))
        wisp_height = round(wisp_width * self.smoke_image.shape[0] / self.smoke_image.shape[1])
        wisps = cv2.resize(self.smoke_image, (wisp_width, wisp_height), interpolation=cv2.INTER_AREA)
        wisp_opacity = min(1.0, progress / 0.2) * (1.0 if progress < 0.6 else (1 - progress) / 0.4)
        for flipped, offset in ((False, -0.08), (True, 0.08)):
            image = cv2.flip(wisps, 1) if flipped else wisps
            paste_rgba(
                frame,
                image,
                cx - wisp_width // 2 + round(offset * wisp_width),
                cy - wisp_height // 2 - round(progress * 40),
                wisp_opacity,
            )

    def render(self, frame, person_mask, now):
        """Applica l'effetto al frame. Restituisce True se la persona e nascosta."""
        if self.state == "idle":
            box = self.person_box(person_mask)
            if box is not None:
                self.last_person_box = box
            return False

        progress = (now - self.state_time) / KAWARIMI_SMOKE_SECONDS
        hidden = (
            self.state == "substituted"
            or (self.state == "vanishing" and progress >= 0.5)
            or (self.state == "appearing" and progress < 0.5)
        )
        if hidden:
            self.draw_substitute(frame, person_mask, now)
        if self.state in ("vanishing", "appearing"):
            if progress >= 1.0:
                self.state = "substituted" if self.state == "vanishing" else "idle"
                self.state_time = now
            else:
                self.draw_smoke(frame, progress)
        return hidden


def acquire_single_instance_lock():
    """Impedisce due istanze contemporanee, che si contenderebbero le webcam."""
    lock_path = Path(tempfile.gettempdir()) / "meme_pose_tracker.lock"
    lock_file = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            "Meme Pose Tracker e gia in esecuzione: chiudi l'altra istanza (Q/ESC)."
        )
    # Il riferimento va tenuto vivo per tutta la durata del programma.
    return lock_file


def parse_args():
    parser = argparse.ArgumentParser(description="Meme Pose Tracker")
    parser.add_argument(
        "--sfondo",
        default=VIRTUAL_BACKGROUND,
        help='sfondo virtuale: "blur" per sfocare, "none" per lo sfondo reale, '
        'oppure il percorso di un\'immagine',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="mostra le informazioni di analisi nell'anteprima locale",
    )
    return parser.parse_args()


def load_virtual_background(value):
    if value is None or value.lower() == "none":
        return None
    if value == "blur":
        return "blur"
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path if (BASE_DIR / path).exists() else Path.cwd() / path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Impossibile leggere lo sfondo: {path}")
    print(f"[SFONDO] Virtuale: {path.name}")
    return resize_cover(image, CAMERA_WIDTH, CAMERA_HEIGHT)


class SnapAudioDetector:
    """Ascolta il microfono e annota gli istanti dei clic secchi (schiocchi)."""

    def __init__(self):
        import sounddevice

        self.clicks = collections.deque(maxlen=16)
        self.lock = threading.Lock()
        self.noise = None
        self.last_click = -1e9
        self.stream = sounddevice.InputStream(
            samplerate=48000,
            channels=1,
            blocksize=480,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time_info, status):
        # La derivata del segnale mette in risalto le alte frequenze del clic.
        energy = float(np.sqrt(np.mean(np.diff(indata[:, 0]) ** 2)))
        now = time.monotonic()
        if self.noise is None:
            self.noise = max(energy, 1e-5)
            return
        threshold = max(AUDIO_CLICK_FACTOR * self.noise, 4e-4)
        if energy > threshold and now - self.last_click > 0.25:
            self.last_click = now
            with self.lock:
                self.clicks.append(now)
            if DEBUG:
                print(f"[SCHIOCCO] clic audio {energy / self.noise:.0f}x", flush=True)
        elif energy < 3 * self.noise:
            self.noise = 0.995 * self.noise + 0.005 * max(energy, 1e-6)

    def recent_clicks(self, since):
        with self.lock:
            return [t for t in self.clicks if t >= since]

    def close(self):
        self.stream.stop()
        self.stream.close()


def snap_hand_metrics(hand_landmarks):
    """Misure della mano usate per lo schiocco e per il fuoco."""
    points = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks])
    size = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-3)
    wrist = points[WRIST]
    middle_pip_distance = max(float(np.linalg.norm(points[10] - wrist)), 1e-3)
    return {
        "points": points,
        "size": size,
        "wrist": wrist,
        "palm": points[list(PALM_INDICES)].mean(axis=0),
        "pinch": float(np.linalg.norm(points[4] - points[12])) / size,
        "middle_extension": float(np.linalg.norm(points[12] - wrist)) / middle_pip_distance,
        "extended": extended_finger_count(hand_landmarks),
    }


class FireParticles:
    """Fiamme a particelle con fusione additiva."""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.data = np.zeros((0, 7), np.float32)  # x, y, vx, vy, eta, vita, raggio

    def emit(self, x, y, scale, count):
        rng = self.rng
        new = np.zeros((count, 7), np.float32)
        new[:, 0] = x + rng.normal(0, 0.18 * scale, count)
        new[:, 1] = y + rng.normal(0, 0.06 * scale, count)
        new[:, 2] = rng.normal(0, 0.20 * scale, count)
        new[:, 3] = -rng.uniform(1.0, 2.0, count) * scale
        new[:, 5] = rng.uniform(0.25, 0.55, count)
        new[:, 6] = rng.uniform(0.09, 0.17, count) * scale
        self.data = np.concatenate([self.data, new])

    def update(self, dt, center_x, scale):
        if not len(self.data):
            return
        rng = self.rng
        d = self.data
        d[:, 3] -= 2.0 * scale * dt                          # spinta verso l'alto
        d[:, 2] += rng.normal(0, 3.0 * scale * dt, len(d))   # turbolenza
        if center_x is not None:
            d[:, 2] += (center_x - d[:, 0]) * 6.0 * dt       # la fiamma si stringe in punta
        d[:, 0] += d[:, 2] * dt
        d[:, 1] += d[:, 3] * dt
        d[:, 4] += dt
        self.data = d[d[:, 4] < d[:, 5]]

    def draw(self, frame):
        if not len(self.data):
            return
        height, width = frame.shape[:2]
        layer = np.zeros((height // 2, width // 2, 3), np.float32)
        for x, y, _, _, age, life, radius in self.data:
            k = age / life
            # Dal nucleo bianco-giallo all'arancio, poi al rosso che svanisce.
            if k < 0.2:
                color = (90, 210, 255)
            elif k < 0.5:
                color = (20, 120, 255)
            else:
                color = (10, 35, 190)
            intensity = (1 - k) ** 1.1
            cv2.circle(
                layer,
                (int(x / 2), int(y / 2)),
                max(1, int(radius * (1 - 0.55 * k) / 2)),
                tuple(float(c * intensity) for c in color),
                -1,
                cv2.LINE_AA,
            )
        layer = cv2.GaussianBlur(layer, (0, 0), 2)
        layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_LINEAR)
        np.add(frame, layer, out=layer)
        np.clip(layer, 0, 255, out=layer)
        frame[:] = layer.astype(np.uint8)


class FireController:
    """Riconosce lo schiocco e gestisce la fiamma che segue la mano."""

    def __init__(self, audio_detector):
        self.audio = audio_detector
        self.particles = FireParticles()
        self.tracks = []          # mani seguite tra un fotogramma e l'altro
        self.burning = False
        self.burn_start = 0.0
        self.fire_position = None  # coordinate normalizzate della mano in fiamme
        self.fire_hand = None
        self.fire_lost_since = None
        self.seen_open = False
        self.fist_frames = 0
        self.last_update = None

    def _match_tracks(self, metrics, now):
        """Associa ogni mano alla traccia piu vicina del fotogramma precedente."""
        tracks = []
        unused = list(self.tracks)
        for hand in metrics:
            best = None
            for track in unused:
                distance = float(np.linalg.norm(track["wrist"] - hand["wrist"]))
                if distance < 0.18 and (best is None or distance < best[0]):
                    best = (distance, track)
            if best is not None:
                track = best[1]
                # Confronto per identita: le tracce contengono array numpy.
                unused = [other for other in unused if other is not track]
            else:
                track = {"last_pinch": -1e9}
            track["wrist"] = hand["wrist"]
            track["hand"] = hand
            if hand["pinch"] <= SNAP_PINCH_MAX:
                track["last_pinch"] = now
            tracks.append(track)
        self.tracks = tracks

    def _snapping_track(self, now):
        clicks = self.audio.recent_clicks(now - 0.3) if self.audio else []
        for track in self.tracks:
            hand = track["hand"]
            released = (
                hand["pinch"] >= SNAP_RELEASE_MIN
                and hand["middle_extension"] <= SNAP_MIDDLE_CURLED_MAX
            )
            # Firma video: contatto pollice-medio e subito dopo il medio sul palmo.
            if released and now - track["last_pinch"] <= SNAP_WINDOW_SECONDS:
                return track, "video"
            # Clic dal microfono poco dopo un contatto pollice-medio.
            for click in clicks:
                if -0.15 <= click - track["last_pinch"] <= 0.7:
                    return track, "audio"
        return None, None

    def update(self, hand_landmarks, now, allowed=True):
        metrics = [snap_hand_metrics(hand) for hand in hand_landmarks]
        self._match_tracks(metrics, now)

        if not self.burning:
            track, source = self._snapping_track(now)
            if track is not None and allowed:
                self.burning = True
                self.burn_start = now
                self.fire_position = track["hand"]["palm"]
                self.seen_open = False
                self.fist_frames = 0
                self.fire_lost_since = None
                if self.audio:
                    # Evita che lo stesso clic riaccenda il fuoco appena spento.
                    with self.audio.lock:
                        self.audio.clicks.clear()
                if DEBUG:
                    print(f"[FUOCO] acceso ({source})", flush=True)
            return

        # La fiamma segue la mano piu vicina alla posizione precedente.
        self.fire_hand = None
        if metrics:
            nearest = min(metrics, key=lambda m: float(np.linalg.norm(m["palm"] - self.fire_position)))
            if float(np.linalg.norm(nearest["palm"] - self.fire_position)) < 0.25:
                self.fire_hand = nearest
        if self.fire_hand is None:
            self.fire_lost_since = self.fire_lost_since or now
        else:
            self.fire_lost_since = None
            self.fire_position = self.fire_hand["palm"]
            if self.fire_hand["extended"] >= 3:
                self.seen_open = True
                self.fist_frames = 0
            elif self.seen_open and self.fire_hand["extended"] == 0:
                self.fist_frames += 1

        if (
            not allowed
            or self.fist_frames >= FIRE_FIST_FRAMES
            or now - self.burn_start > FIRE_MAX_SECONDS
            or (self.fire_lost_since is not None and now - self.fire_lost_since > 0.7)
        ):
            self.burning = False
            self.fire_hand = None
            if DEBUG:
                print("[FUOCO] spento", flush=True)

    def render(self, frame, now):
        dt = 0.033 if self.last_update is None else min(0.1, now - self.last_update)
        self.last_update = now
        height, width = frame.shape[:2]
        center_x = None
        scale = 120.0
        if self.burning and self.fire_hand is not None:
            hand = self.fire_hand
            scale = hand["size"] * height
            # La fiamma nasce sopra il palmo, verso le dita.
            top_y = float(hand["points"][:, 1].min())
            x = float(hand["palm"][0]) * width
            y = (0.6 * float(hand["palm"][1]) + 0.4 * top_y) * height
            center_x = x
            self.particles.emit(x, y, scale, 22)
        self.particles.update(dt, center_x, scale)
        self.particles.draw(frame)


def main():
    global DEBUG
    args = parse_args()
    DEBUG = DEBUG or args.debug
    instance_lock = acquire_single_instance_lock()
    memes = {
        "two_hands": load_meme(MEME_PATH),
        "memeomg": load_meme(MEMEOMG_PATH),
        "stop": load_meme(STOP_MEME_PATH),
    }
    try:
        audio_detector = SnapAudioDetector()
        print("[MICROFONO] Attivo per riconoscere gli schiocchi.")
    except Exception as exc:
        audio_detector = None
        print(f"[MICROFONO] Non disponibile, schiocchi solo dal video. Dettagli: {exc}")
    fire = FireController(audio_detector)
    kawarimi = Kawarimi(
        load_rgba(KAWARIMI_IMAGE_PATH),
        load_rgba(KAWARIMI_SMOKE_PATH, min_brightness=250),
        load_virtual_background(args.sfondo),
    )

    face_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    segmenter_options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
    segmenter = mp_vision.ImageSegmenter.create_from_options(segmenter_options)

    try:
        cap, pending_frame = open_camera(0)
    except Exception:
        face_landmarker.close()
        hand_landmarker.close()
        segmenter.close()
        raise

    try:
        virtual_camera = pyvirtualcam.Camera(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=VIRTUAL_CAMERA_FPS,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend="obs",
        )
        print(f"[WEBCAM VIRTUALE] Attiva: {virtual_camera.device}")
    except Exception as exc:
        # Senza webcam virtuale il programma resta utilizzabile come anteprima.
        virtual_camera = None
        print(
            "[WEBCAM VIRTUALE] Non disponibile, mostro solo l'anteprima locale. "
            "Chiudi OBS Studio o disattiva la sua Virtual Camera, quindi riprova. "
            f"Dettagli: {exc}"
        )

    print("Premi Q o ESC nella finestra di anteprima per uscire.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    last_face_box = None
    last_face_time = 0.0
    squares = {}
    candidate_pose = None
    candidate_frames = 0
    miss_frames = 0
    active_pose = None
    confidence = 0.0
    start_time = time.monotonic()
    camera_read_failures = 0
    seal_frames = 0

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                ok = True
            else:
                ok, frame = cap.read()

            if not ok or frame is None or not frame.size:
                camera_read_failures += 1
                if camera_read_failures < 30:
                    time.sleep(0.03)
                    continue
                raise RuntimeError(
                    "La webcam ha smesso di inviare fotogrammi per circa un secondo."
                )
            camera_read_failures = 0

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            segment_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            person_mask = segment_result.confidence_masks[0].numpy_view()
            if person_mask.ndim == 3:
                person_mask = person_mask[..., 0]

            now = time.monotonic()
            if face_result.face_landmarks:
                last_face_box = face_bounding_box(face_result.face_landmarks[0])
                last_face_time = now
            elif now - last_face_time > 1.2:
                last_face_box = None

            detected_hands = hand_result.hand_landmarks
            squares = update_squares(
                detected_hands,
                squares,
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            )

            two_hands_match, two_hands_confidence = hands_on_face_pose(
                detected_hands,
                last_face_box,
            )
            memeomg_match, memeomg_confidence = hands_behind_head_pose(
                detected_hands,
                last_face_box,
            )
            stop_match, stop_confidence = timeout_hands_pose(
                detected_hands,
                last_face_box,
            )

            seal_match, seal_confidence = kawarimi_seal_pose(
                detected_hands,
                last_face_box,
            )
            seal_frames = seal_frames + 1 if seal_match else 0
            if seal_frames >= KAWARIMI_REQUIRED_FRAMES:
                kawarimi.trigger(now)
                seal_frames = 0

            # Il sigillo della sostituzione ha la precedenza su tutti i meme;
            # le mani dietro la testa sono molto separate e hanno precedenza
            # sul gesto con le mani unite davanti al volto.
            if seal_match or kawarimi.state != "idle":
                detected_pose = None
                current_confidence = seal_confidence
            elif memeomg_match:
                detected_pose = "memeomg"
                current_confidence = memeomg_confidence
            elif stop_match:
                detected_pose = "stop"
                current_confidence = stop_confidence
            elif two_hands_match:
                detected_pose = "two_hands"
                current_confidence = two_hands_confidence
            else:
                detected_pose = None
                current_confidence = max(
                    two_hands_confidence,
                    memeomg_confidence,
                    stop_confidence,
                )

            confidence = 0.72 * confidence + 0.28 * current_confidence

            if detected_pose is not None:
                miss_frames = 0
                if detected_pose == candidate_pose:
                    candidate_frames += 1
                else:
                    candidate_pose = detected_pose
                    candidate_frames = 1
            else:
                miss_frames += 1
                candidate_pose = None
                candidate_frames = 0

            required_frames = 5
            if candidate_pose is not None and candidate_frames >= required_frames:
                active_pose = candidate_pose
            elif miss_frames >= 8:
                active_pose = None

            if kawarimi.state == "idle":
                kawarimi.update_background(frame, person_mask)
            else:
                active_pose = None
            kawarimi.apply_virtual_background(frame, person_mask)

            meme_box = None
            if active_pose is not None:
                meme_box = overlay_meme_on_face(
                    frame,
                    memes[active_pose],
                    last_face_box,
                )

            hidden = kawarimi.render(frame, person_mask, now)
            fire.update(detected_hands, now, allowed=kawarimi.state == "idle")
            if not hidden:
                fire.render(frame, now)

            # Il video trasmesso e sempre pulito: viene inviato prima di
            # disegnare le informazioni di analisi.
            if virtual_camera is not None:
                virtual_camera.send(frame)

            if DEBUG:
                if meme_box is not None:
                    cv2.rectangle(
                        frame,
                        meme_box[:2],
                        (meme_box[2] - 1, meme_box[3] - 1),
                        (0, 255, 210),
                        2,
                        cv2.LINE_AA,
                    )
                phase = (time.monotonic() * 0.65) % 1.0
                if not hidden:
                    for index, square in squares.items():
                        draw_analysis_square(frame, square, f"MANO {index + 1}", phase)

                status_color = (0, 255, 210) if active_pose else (0, 210, 255)
                if kawarimi.state != "idle":
                    status = "KAWARIMI NO JUTSU!"
                    status_color = (40, 140, 255)
                elif seal_match:
                    status = "SIGILLO DELLA TIGRE..."
                elif fire.burning:
                    status = "FUOCO!"
                    status_color = (30, 140, 255)
                elif active_pose == "memeomg":
                    status = "MANI DIETRO LA TESTA RILEVATE"
                elif active_pose == "stop":
                    status = "GESTO TIMEOUT RILEVATO"
                elif active_pose == "two_hands":
                    status = "POSE DUE MANI RILEVATA"
                else:
                    status = "ANALISI MANI..."
                cv2.putText(
                    frame,
                    status,
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Q/ESC: esci",
                    (18, CAMERA_HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (225, 225, 225),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"confidenza: {confidence:.0%}",
                    (CAMERA_WIDTH - 185, CAMERA_HEIGHT - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (225, 225, 225),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()

    finally:
        if audio_detector is not None:
            audio_detector.close()
        if virtual_camera is not None:
            virtual_camera.close()
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        hand_landmarker.close()
        segmenter.close()


if __name__ == "__main__":
    main()

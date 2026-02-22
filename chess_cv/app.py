"""
Main application logic for Real-Time Hand Gesture Chess (Human vs Computer)
"""
import cv2
import mediapipe as mp
import numpy as np
import chess
import sys
import math

# Utility functions
def get_fingertips(hand_landmarks):
    # hand_landmarks is a list of 21 landmarks
    index_tip = hand_landmarks[8]
    thumb_tip = hand_landmarks[4]
    return (index_tip.x, index_tip.y), (thumb_tip.x, thumb_tip.y)

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

PINCH_THRESHOLD = 0.05

def get_square_from_coords(x, y):
    file = int(x * 8)
    rank = 7 - int(y * 8)
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None

def get_piece_at_square(square, board):
    if square is not None:
        return board.piece_at(square)
    return None

def get_piece_image(piece, chessboard_ui):
    if piece is None:
        return None
    key = (piece.piece_type, piece.color)
    return chessboard_ui.piece_images.get(key)

def highlight_square(img, square, chessboard_ui):
    if square is not None:
        x, y = chessboard_ui.get_square_center(square)
        sz = chessboard_ui.square_size
        cv2.rectangle(img, (x - sz//2, y - sz//2), (x + sz//2, y + sz//2), (0, 255, 255), 3)

def draw_piece_at(img, piece_img, x, y, chessboard_ui):
    if piece_img is not None:
        sz = chessboard_ui.square_size
        # If x and y are in [0,1], treat as normalized; else, treat as pixel
        h, w = img.shape[:2]
        if 0 <= x <= 1 and 0 <= y <= 1:
            px = int(x * w)
            py = int(y * h)
        else:
            px = int(x)
            py = int(y)
        px0 = px - sz // 2
        py0 = py - sz // 2
        px1 = px0 + sz
        py1 = py0 + sz
        piece_img_resized = cv2.resize(piece_img, (sz, sz))
        if piece_img_resized.shape[2] == 4:
            alpha = piece_img_resized[:, :, 3] / 255.0
            for c in range(3):
                img[py0:py1, px0:px1, c] = (
                    alpha * piece_img_resized[:, :, c] + (1 - alpha) * img[py0:py1, px0:px1, c]
                )
        else:
            img[py0:py1, px0:px1] = piece_img_resized[:, :, :3]

def computer_move(board):
    import random
    legal_moves = list(board.legal_moves)
    if legal_moves:
        move = random.choice(legal_moves)
        board.push(move)
import math

from .hand_tracker import HandTracker
from .chessboard import ChessboardUI
from .gesture import GestureController
from .engine import ChessEngine

def main():
    # Initialize modules
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    hand_tracker = HandTracker()
    # Track if a move was made this frame
    move_made = False
    chessboard_ui = ChessboardUI()
    gesture_ctrl = GestureController()
    engine = ChessEngine()

    board = chess.Board()
    selected_square = None
    dragging_piece = None
    drag_pixel_pos = None
    is_pinching = False
    human_turn = True
    prev_drag_pixel_pos = None  # For smoothing
    DRAG_SMOOTH_ALPHA = 0.35
    running = True
    fps_time = cv2.getTickCount()
    frame_count = 0

    cv2.namedWindow("Hand Gesture Chess")
    while running:
        hover_square = None
        ret, camera_frame = cap.read()
        if not ret:
            print("Error: Camera frame not received.")
            break
        camera_frame = cv2.flip(camera_frame, 1)
        camera_frame = cv2.resize(camera_frame, (960, 720))
        h, w, _ = camera_frame.shape

        hand_landmarks, handedness = hand_tracker.process(camera_frame)
        pinch_now = False
        ix = iy = None
        if hand_landmarks:
            (ix, iy), (tx, ty) = get_fingertips(hand_landmarks)
            pinch_dist = euclidean_distance((ix, iy), (tx, ty))
            pinch_now = pinch_dist < PINCH_THRESHOLD

            # Map normalized coords to board pixel area
            board_left = chessboard_ui.margin / w
            board_top = chessboard_ui.margin / h
            board_right = (chessboard_ui.margin + chessboard_ui.square_size * 8) / w
            board_bottom = (chessboard_ui.margin + chessboard_ui.square_size * 8) / h
            # Only allow selection if finger is inside board area
            if board_left <= ix <= board_right and board_top <= iy <= board_bottom:
                board_ix = (ix - board_left) / (board_right - board_left)
                board_iy = (iy - board_top) / (board_bottom - board_top)
                hover_square = get_square_from_coords(board_ix, board_iy)
            else:
                board_ix = board_iy = None
                hover_square = None

        # TURN SAFETY LOCK: ignore all gesture input if not human's turn
        gesture_enabled = human_turn

        # Pinch START
        if gesture_enabled and pinch_now and not is_pinching:
            if hover_square is not None:
                piece = get_piece_at_square(hover_square, board)
                if piece and piece.color == chess.WHITE:
                    selected_square = hover_square
                    dragging_piece = piece
                    drag_pixel_pos = (ix, iy)
                    prev_drag_pixel_pos = (ix, iy)
                    print(f"Selected square: {selected_square}, piece: {dragging_piece}")

        # WHILE PINCHING (with smoothing)
        if gesture_enabled and pinch_now:
            if dragging_piece is not None:
                if prev_drag_pixel_pos is not None:
                    # Linear interpolation for smoothing
                    new_x = DRAG_SMOOTH_ALPHA * ix + (1 - DRAG_SMOOTH_ALPHA) * prev_drag_pixel_pos[0]
                    new_y = DRAG_SMOOTH_ALPHA * iy + (1 - DRAG_SMOOTH_ALPHA) * prev_drag_pixel_pos[1]
                    drag_pixel_pos = (new_x, new_y)
                    prev_drag_pixel_pos = drag_pixel_pos
                else:
                    drag_pixel_pos = (ix, iy)
                    prev_drag_pixel_pos = (ix, iy)

        # PINCH RELEASE
        if gesture_enabled and not pinch_now and is_pinching:
            if dragging_piece is not None and drag_pixel_pos is not None:
                # Map drag_pixel_pos to board area
                px, py = drag_pixel_pos
                if board_left <= px <= board_right and board_top <= py <= board_bottom:
                    board_px = (px - board_left) / (board_right - board_left)
                    board_py = (py - board_top) / (board_bottom - board_top)
                    target_square = get_square_from_coords(board_px, board_py)
                else:
                    target_square = None
                if target_square is not None:
                    move = chess.Move(selected_square, target_square)
                    if move in board.legal_moves:
                        board.push(move)
                        human_turn = False
                        # Computer move (ONE only)
                        computer_move(board)
                        human_turn = True
                # If illegal, just snap back (do nothing)
            # Reset state after release
            selected_square = None
            dragging_piece = None
            drag_pixel_pos = None
            prev_drag_pixel_pos = None

        # --- RENDERING ---
        # 1. Start with camera frame
        frame = camera_frame.copy()

        # 2. Draw board squares
        chessboard_ui.draw(frame, board)

        # 3. Draw all pieces except selected_square when dragging
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if dragging_piece is not None and selected_square == square and is_pinching:
                    continue
                img_piece = get_piece_image(piece, chessboard_ui)
                if img_piece is not None:
                    x, y = chessboard_ui.get_square_center(square)
                    sz = chessboard_ui.square_size
                    px0 = x - sz // 2
                    py0 = y - sz // 2
                    px1 = px0 + sz
                    py1 = py0 + sz
                    piece_img = cv2.resize(img_piece, (sz, sz))
                    if piece_img.shape[2] == 4:
                        alpha = piece_img[:, :, 3] / 255.0
                        for c in range(3):
                            frame[py0:py1, px0:px1, c] = (
                                alpha * piece_img[:, :, c] + (1 - alpha) * frame[py0:py1, px0:px1, c]
                            )
                    else:
                        frame[py0:py1, px0:px1] = piece_img[:, :, :3]

        # Highlight hovered square (yellow) if not dragging
        if hover_square is not None and not is_pinching:
            highlight_square(frame, hover_square, chessboard_ui)

        # 4. If dragging, draw dragged piece at drag_pixel_pos LAST
        if dragging_piece is not None and drag_pixel_pos is not None and is_pinching:
            # Convert drag_pixel_pos (normalized) to pixel coordinates for drawing
            px = int(drag_pixel_pos[0] * w)
            py = int(drag_pixel_pos[1] * h)
            draw_piece_at(frame, get_piece_image(dragging_piece, chessboard_ui), px, py, chessboard_ui)
            # 5. Draw debug red circle at drag_pixel_pos
            cv2.circle(frame, (px, py), 18, (0,0,255), 3)

            # TARGET SQUARE PREVIEW
            preview_square = get_square_from_coords(drag_pixel_pos[0], drag_pixel_pos[1])
            if preview_square is not None:
                move = chess.Move(selected_square, preview_square)
                x, y = chessboard_ui.get_square_center(preview_square)
                sz = chessboard_ui.square_size
                color = (0,255,0) if move in board.legal_moves else (0,0,255)
                cv2.rectangle(frame, (x - sz//2, y - sz//2), (x + sz//2, y + sz//2), color, 3)

        # Draw fingertip marker LAST (green circle)
        if hand_landmarks:
            (ix, iy), _ = get_fingertips(hand_landmarks)
            cv2.circle(frame, (int(ix * w), int(iy * h)), 10, (0,255,0), -1)

        # Highlight selected square if any
        if selected_square is not None:
            highlight_square(frame, selected_square, chessboard_ui)

        # Update pinch state
        is_pinching = pinch_now

        # Show the frame
        cv2.imshow("Hand Gesture Chess", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            running = False

    cap.release()
    cv2.destroyAllWindows()

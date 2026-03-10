"""
Main application logic for Real-Time Hand Gesture Chess (Human vs Computer)

Keyboard shortcuts (while game window is focused):
  T – cycle board theme (wood → marble → dark → classic)
  F – flip board
  U – undo last move pair
  N – new game
  ESC – quit

Gesture controls:
  Pinch (thumb+index)  – select & drag a piece
  Release              – drop piece on target square
  Thumb-up             – undo last move pair
  Two-finger tap       – start new game
  Open palm            – pause / show help overlay
  Fist                 – resign game
"""
import cv2
import mediapipe as mp
import numpy as np
import chess
import sys
import math
import time

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

import math

from .hand_tracker import HandTracker
from .chessboard import ChessboardUI
from .gesture import GestureController
from .engine import ChessEngine

def _draw_overlay_text(frame, lines, alpha=0.7):
    """Draw a semi-transparent dark overlay with centered text lines."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 60
    for line in lines:
        scale = 0.7 if len(line) < 40 else 0.5
        (tw, th), _ = cv2.getTextSize(line, font, scale, 1)
        cv2.putText(frame, line, ((w - tw) // 2, y), font, scale,
                    (220, 220, 220), 1, cv2.LINE_AA)
        y += th + 18


def _rebuild_captures(board):
    """Rebuild captured piece lists by replaying the move stack."""
    captured_white = []
    captured_black = []
    tmp = chess.Board()
    for move in board.move_stack:
        cap = tmp.piece_at(move.to_square)
        if cap:
            if cap.color == chess.WHITE:
                captured_white.append(cap.piece_type)
            else:
                captured_black.append(cap.piece_type)
        tmp.push(move)
    return captured_white, captured_black


def main():
    # Initialize modules
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    hand_tracker = HandTracker()
    chessboard_ui = ChessboardUI()
    gesture_ctrl = GestureController()
    engine = ChessEngine(difficulty="medium")  # "easy", "medium", or "hard"

    board = chess.Board()
    selected_square = None
    dragging_piece = None
    drag_pixel_pos = None
    is_pinching = False
    human_turn = True
    prev_drag_pixel_pos = None  # For smoothing
    DRAG_SMOOTH_ALPHA = 0.35
    running = True
    last_move = None              # track last move for highlight
    captured_white = []           # piece types captured FROM white
    captured_black = []           # piece types captured FROM black
    eval_score = 0                # centipawns from white's perspective
    game_over = False             # True after checkmate / resign
    game_over_msg = ""
    paused = False                # open-palm pause overlay

    PAUSE_HELP = [
        "PAUSED  -  Hand Gesture Chess",
        "",
        "GESTURES:",
        "Pinch         - Select & drag piece",
        "Release       - Drop piece",
        "Thumb-up      - Undo last move",
        "Two-finger    - New game",
        "Open palm     - Pause (this screen)",
        "Fist          - Resign",
        "",
        "KEYS:  T=Theme  F=Flip  U=Undo  N=New  ESC=Quit",
    ]

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

        # ── Gesture detection ──────────────────────────────────────────────
        gesture = gesture_ctrl.detect(hand_landmarks)

        pinch_now = False
        ix = iy = None
        board_left = board_top = board_right = board_bottom = 0
        if hand_landmarks:
            (ix, iy), (tx, ty) = get_fingertips(hand_landmarks)
            pinch_dist = euclidean_distance((ix, iy), (tx, ty))
            pinch_now = pinch_dist < PINCH_THRESHOLD

            # Map normalized coords to board pixel area
            board_left = chessboard_ui.margin / w
            board_top = chessboard_ui.margin / h
            board_right = (chessboard_ui.margin + chessboard_ui.square_size * 8) / w
            board_bottom = (chessboard_ui.margin + chessboard_ui.square_size * 8) / h
            if board_left <= ix <= board_right and board_top <= iy <= board_bottom:
                board_ix = (ix - board_left) / (board_right - board_left)
                board_iy = (iy - board_top) / (board_bottom - board_top)
                hover_square = get_square_from_coords(board_ix, board_iy)
            else:
                board_ix = board_iy = None
                hover_square = None

        # ── Handle non-pinch gestures ──────────────────────────────────────
        if gesture == 'open_palm':
            paused = True
        elif paused and gesture is not None and gesture != 'open_palm':
            paused = False  # any other gesture dismisses pause

        if not paused and not game_over:
            # Thumb-up → Undo
            if gesture == 'thumb_up':
                if len(board.move_stack) >= 2:
                    board.pop()  # undo AI
                    board.pop()  # undo human
                    last_move = board.peek() if board.move_stack else None
                    captured_white, captured_black = _rebuild_captures(board)
                    eval_score = engine.evaluate(board)
                    print("[Gesture] Thumb-up → Undo")

            # Two-finger tap → New game
            elif gesture == 'two_finger_tap':
                board.reset()
                selected_square = None
                dragging_piece = None
                drag_pixel_pos = None
                is_pinching = False
                human_turn = True
                last_move = None
                captured_white = []
                captured_black = []
                eval_score = 0
                game_over = False
                game_over_msg = ""
                print("[Gesture] Two-finger tap → New game")

            # Fist → Resign
            elif gesture == 'fist':
                game_over = True
                game_over_msg = "You resigned. Black wins!"
                print("[Gesture] Fist → Resign")

        # ── Pinch-based piece dragging (only when not paused) ──────────────
        gesture_enabled = human_turn and not game_over and not paused

        # Pinch START
        if gesture_enabled and pinch_now and not is_pinching:
            if hover_square is not None:
                piece = get_piece_at_square(hover_square, board)
                if piece and piece.color == chess.WHITE:
                    selected_square = hover_square
                    dragging_piece = piece
                    drag_pixel_pos = (ix, iy)
                    prev_drag_pixel_pos = (ix, iy)

        # WHILE PINCHING (with smoothing)
        if gesture_enabled and pinch_now:
            if dragging_piece is not None:
                if prev_drag_pixel_pos is not None:
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
                        captured = board.piece_at(move.to_square)
                        if captured:
                            if captured.color == chess.WHITE:
                                captured_white.append(captured.piece_type)
                            else:
                                captured_black.append(captured.piece_type)
                        board.push(move)
                        last_move = move

                        # Check game end after human move
                        if board.is_checkmate():
                            game_over = True
                            game_over_msg = "Checkmate! You win!"
                        elif board.is_stalemate():
                            game_over = True
                            game_over_msg = "Stalemate – Draw!"
                        elif board.is_insufficient_material():
                            game_over = True
                            game_over_msg = "Draw – Insufficient material"
                        elif not game_over:
                            human_turn = False
                            ai_move = engine.choose_move(board)
                            if ai_move:
                                ai_captured = board.piece_at(ai_move.to_square)
                                if ai_captured:
                                    if ai_captured.color == chess.WHITE:
                                        captured_white.append(ai_captured.piece_type)
                                    else:
                                        captured_black.append(ai_captured.piece_type)
                                board.push(ai_move)
                                last_move = ai_move
                                chessboard_ui.start_animation(ai_move)
                                # Check game end after AI move
                                if board.is_checkmate():
                                    game_over = True
                                    game_over_msg = "Checkmate! Black wins."
                                elif board.is_stalemate():
                                    game_over = True
                                    game_over_msg = "Stalemate – Draw!"
                            human_turn = True
                        eval_score = engine.evaluate(board)
            selected_square = None
            dragging_piece = None
            drag_pixel_pos = None
            prev_drag_pixel_pos = None

        # ── RENDERING ──────────────────────────────────────────────────────
        frame = camera_frame.copy()

        drag_skip = selected_square if (dragging_piece is not None and is_pinching) else None

        chessboard_ui.draw(frame, board,
                           hover_square=hover_square if not is_pinching else None,
                           selected_square=selected_square,
                           last_move=last_move,
                           captured_white=captured_white,
                           captured_black=captured_black,
                           skip_square=drag_skip)

        chessboard_ui.draw_eval_bar(frame, eval_score)

        # Dragging piece overlay
        if dragging_piece is not None and drag_pixel_pos is not None and is_pinching:
            px = int(drag_pixel_pos[0] * w)
            py = int(drag_pixel_pos[1] * h)
            draw_piece_at(frame, get_piece_image(dragging_piece, chessboard_ui), px, py, chessboard_ui)
            cv2.circle(frame, (px, py), 18, (0, 0, 255), 3)
            preview_square = get_square_from_coords(drag_pixel_pos[0], drag_pixel_pos[1])
            if preview_square is not None:
                move = chess.Move(selected_square, preview_square)
                x, y = chessboard_ui.get_square_center(preview_square)
                sz = chessboard_ui.square_size
                color = (0, 255, 0) if move in board.legal_moves else (0, 0, 255)
                cv2.rectangle(frame, (x - sz // 2, y - sz // 2),
                              (x + sz // 2, y + sz // 2), color, 3)

        # Fingertip marker
        if hand_landmarks:
            (ix, iy), _ = get_fingertips(hand_landmarks)
            cv2.circle(frame, (int(ix * w), int(iy * h)), 10, (0, 255, 0), -1)

        if selected_square is not None:
            highlight_square(frame, selected_square, chessboard_ui)

        # Pause overlay
        if paused:
            _draw_overlay_text(frame, PAUSE_HELP)

        # Game over overlay
        if game_over:
            _draw_overlay_text(frame, [
                game_over_msg,
                "",
                "Two-finger tap or N to start a new game",
            ], alpha=0.6)

        # Update pinch state
        is_pinching = pinch_now

        # Show the frame
        cv2.imshow("Hand Gesture Chess", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC – quit
            running = False
        elif key == ord('t') or key == ord('T'):
            new_theme = chessboard_ui.cycle_theme()
            print(f"[Theme] Switched to: {new_theme}")
        elif key == ord('f') or key == ord('F'):
            chessboard_ui.flipped = not chessboard_ui.flipped
            print(f"[Board] Flipped: {chessboard_ui.flipped}")
        elif key == ord('u') or key == ord('U'):
            if len(board.move_stack) >= 2:
                board.pop()
                board.pop()
                last_move = board.peek() if board.move_stack else None
                captured_white, captured_black = _rebuild_captures(board)
                eval_score = engine.evaluate(board)
                print("[Key] U → Undo")
        elif key == ord('n') or key == ord('N'):
            board.reset()
            selected_square = None
            dragging_piece = None
            drag_pixel_pos = None
            is_pinching = False
            human_turn = True
            last_move = None
            captured_white = []
            captured_black = []
            eval_score = 0
            game_over = False
            game_over_msg = ""
            print("[Key] N → New game")

    engine.quit()
    cap.release()
    cv2.destroyAllWindows()

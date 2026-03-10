"""
Chessboard UI rendering and coordinate mapping.

Features:
  1. Multiple board themes (wood, marble, dark, classic)
  2. Custom piece sets (wood PNG folder, with Unicode fallback)
  3. Board flip (view as white or black)
  4. Move animation (smooth sliding for computer moves)
  5. Captured pieces tray
  6. Rank / file labels (a-h, 1-8)
  7. Last move highlight
"""

import numpy as np
import cv2
import chess
import os
import time

# ── Board colour themes ────────────────────────────────────────────────────
THEMES = {
    "wood":    {"light": (222, 202, 163), "dark": (139, 108, 66)},
    "marble":  {"light": (230, 230, 230), "dark": (160, 160, 170)},
    "dark":    {"light": (120, 120, 130), "dark": (50,  50,  60)},
    "classic": {"light": (238, 238, 210), "dark": (118, 150, 86)},
}
THEME_NAMES = list(THEMES.keys())


class ChessboardUI:
    def __init__(self, board_size=360, margin=20, theme="wood", flipped=False):
        self.board_size = board_size
        self.margin = margin
        self.square_size = (board_size - 2 * margin) // 8
        self.flipped = flipped

        # Theme
        self.set_theme(theme)

        # Highlight colours (BGR)
        self.highlight_color = (0, 255, 255)   # hover – cyan
        self.selected_color  = (0, 128, 255)   # selected – orange
        self.legal_color     = (0, 255, 0)     # legal target – green
        self.last_move_color = (50, 180, 255)  # last move – gold/amber

        # Piece images
        self.piece_images = self.load_piece_images()

        # Animation state (set externally, consumed by draw())
        self._anim_move  = None   # chess.Move
        self._anim_start = None   # time.time() when animation began
        self._anim_dur   = 0.25   # seconds

    # ── Theme management ───────────────────────────────────────────────────

    def set_theme(self, name):
        """Switch board colour theme by name."""
        t = THEMES.get(name, THEMES["wood"])
        self.theme_name = name
        self.colors = [t["light"], t["dark"]]

    def cycle_theme(self):
        """Advance to the next theme and return its name."""
        idx = (THEME_NAMES.index(self.theme_name) + 1) % len(THEME_NAMES)
        self.set_theme(THEME_NAMES[idx])
        return self.theme_name

    # ── Piece image loading ────────────────────────────────────────────────

    def load_piece_images(self):
        base = os.path.join(os.path.dirname(__file__), "assets", "wood")
        mapping = {
            (chess.PAWN, True):   "wP.png",  (chess.PAWN, False):   "bP.png",
            (chess.KNIGHT, True): "wN.png",  (chess.KNIGHT, False): "bN.png",
            (chess.BISHOP, True): "wB.png",  (chess.BISHOP, False): "bB.png",
            (chess.ROOK, True):   "wR.png",  (chess.ROOK, False):   "bR.png",
            (chess.QUEEN, True):  "wQ.png",  (chess.QUEEN, False):  "bQ.png",
            (chess.KING, True):   "wK.png",  (chess.KING, False):   "bK.png",
        }
        images = {}
        for key, fname in mapping.items():
            # Try exact case first, then lower‑case (the files on disk are lower)
            for candidate in (fname, fname.lower()):
                path = os.path.join(base, candidate)
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        images[key] = img
                    break
        return images

    # ── Coordinate helpers (respect flip) ──────────────────────────────────

    def _visual_file_rank(self, square):
        """Return (visual_file, visual_rank_row) for drawing, accounting for flip."""
        f = chess.square_file(square)
        r = chess.square_rank(square)
        if self.flipped:
            return 7 - f, r          # file reversed, rank 0 at top
        return f, 7 - r              # normal: rank 7 at top

    def get_square_center(self, square):
        vf, vr = self._visual_file_rank(square)
        x = self.margin + vf * self.square_size + self.square_size // 2
        y = self.margin + vr * self.square_size + self.square_size // 2
        return (x, y)

    def get_square_from_pos(self, pos):
        if pos is None:
            return None
        x, y = pos
        bx = x - self.margin
        by = y - self.margin
        if bx < 0 or by < 0 or bx >= self.square_size * 8 or by >= self.square_size * 8:
            return None
        vf = int(bx // self.square_size)
        vr = int(by // self.square_size)
        if self.flipped:
            file = 7 - vf
            rank = vr
        else:
            file = vf
            rank = 7 - vr
        if 0 <= file < 8 and 0 <= rank < 8:
            return chess.square(file, rank)
        return None

    # ── Animation helpers ──────────────────────────────────────────────────

    def start_animation(self, move):
        """Begin a slide animation for *move* (a chess.Move)."""
        self._anim_move = move
        self._anim_start = time.time()

    def _is_animating(self):
        if self._anim_move is None:
            return False
        return (time.time() - self._anim_start) < self._anim_dur

    def _anim_progress(self):
        """Return 0‑1 progress (ease‑out)."""
        t = min((time.time() - self._anim_start) / self._anim_dur, 1.0)
        return 1 - (1 - t) ** 2       # ease‑out quad

    # ── Main draw method ───────────────────────────────────────────────────

    def draw(self, img, board, hover_square=None, selected_square=None,
             legal_moves=None, last_move=None, captured_white=None,
             captured_black=None, skip_square=None):
        """
        Render the board onto *img* (in‑place).

        Parameters added for the 7 features:
          last_move       – chess.Move (or None) to highlight
          captured_white  – list of chess.PieceType captured *from* white
          captured_black  – list of chess.PieceType captured *from* black
          skip_square     – square whose piece should NOT be drawn (dragging)
        """
        animating = self._is_animating()
        anim_square = self._anim_move.to_square if animating else None

        # ---- 1. Draw squares + 7. last move highlight ---------------------
        for rank_row in range(8):
            for file_col in range(8):
                # Map visual grid back to chess square
                if self.flipped:
                    sq = chess.square(7 - file_col, rank_row)
                else:
                    sq = chess.square(file_col, 7 - rank_row)

                color = self.colors[(file_col + rank_row) % 2]
                x0 = self.margin + file_col * self.square_size
                y0 = self.margin + rank_row * self.square_size
                x1 = x0 + self.square_size
                y1 = y0 + self.square_size
                cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

                # Last move highlight (semi‑transparent overlay)
                if last_move and sq in (last_move.from_square, last_move.to_square):
                    overlay = img[y0:y1, x0:x1].copy()
                    cv2.rectangle(overlay, (0, 0), (x1 - x0, y1 - y0), self.last_move_color, -1)
                    cv2.addWeighted(overlay, 0.35, img[y0:y1, x0:x1], 0.65, 0, img[y0:y1, x0:x1])

                # Legal move dots
                if legal_moves and sq in legal_moves:
                    cx = x0 + self.square_size // 2
                    cy = y0 + self.square_size // 2
                    r = self.square_size // 6
                    cv2.circle(img, (cx, cy), r, self.legal_color, -1)

                # Hover highlight
                if hover_square == sq:
                    cv2.rectangle(img, (x0, y0), (x1, y1), self.highlight_color, 3)

                # Selected highlight
                if selected_square == sq:
                    cv2.rectangle(img, (x0, y0), (x1, y1), self.selected_color, 4)

        # ---- 6. Rank / file labels ----------------------------------------
        self._draw_labels(img)

        # ---- Draw pieces (skip dragging square & animation target) --------
        for sq in chess.SQUARES:
            if sq == skip_square:
                continue
            if animating and sq == anim_square:
                continue  # drawn separately during animation
            piece = board.piece_at(sq)
            if piece:
                self._draw_piece_on_square(img, piece, sq)

        # ---- 4. Move animation (sliding piece) ---------------------------
        if animating:
            prog = self._anim_progress()
            src = self.get_square_center(self._anim_move.from_square)
            dst = self.get_square_center(self._anim_move.to_square)
            cx = int(src[0] + (dst[0] - src[0]) * prog)
            cy = int(src[1] + (dst[1] - src[1]) * prog)
            piece = board.piece_at(self._anim_move.to_square)
            if piece:
                self._draw_piece_at_pixel(img, piece, cx, cy)
        elif self._anim_move is not None:
            # Animation finished – clear
            self._anim_move = None

        # ---- 5. Captured pieces tray --------------------------------------
        if captured_white or captured_black:
            self._draw_captured_tray(img, captured_white or [], captured_black or [])

        return img

    # ── Internal drawing helpers ───────────────────────────────────────────

    def _draw_piece_on_square(self, img, piece, square):
        x, y = self.get_square_center(square)
        self._draw_piece_at_pixel(img, piece, x, y)

    def _draw_piece_at_pixel(self, img, piece, cx, cy):
        key = (piece.piece_type, piece.color)
        piece_img = self.piece_images.get(key)
        sz = self.square_size
        px0 = cx - sz // 2
        py0 = cy - sz // 2
        px1 = px0 + sz
        py1 = py0 + sz

        h, w = img.shape[:2]
        # Bounds check
        if px0 < 0 or py0 < 0 or px1 > w or py1 > h:
            return

        if piece_img is not None:
            resized = cv2.resize(piece_img, (sz, sz))
            if resized.shape[2] == 4:
                alpha = resized[:, :, 3] / 255.0
                for c in range(3):
                    img[py0:py1, px0:px1, c] = (
                        alpha * resized[:, :, c] + (1 - alpha) * img[py0:py1, px0:px1, c]
                    )
            else:
                img[py0:py1, px0:px1] = resized[:, :, :3]
        else:
            # Unicode fallback
            unicode_map = {
                (chess.PAWN, True):   "P", (chess.PAWN, False):   "p",
                (chess.KNIGHT, True): "N", (chess.KNIGHT, False): "n",
                (chess.BISHOP, True): "B", (chess.BISHOP, False): "b",
                (chess.ROOK, True):   "R", (chess.ROOK, False):   "r",
                (chess.QUEEN, True):  "Q", (chess.QUEEN, False):  "q",
                (chess.KING, True):   "K", (chess.KING, False):   "k",
            }
            symbol = unicode_map.get(key, "?")
            clr = (255, 255, 255) if piece.color == chess.WHITE else (30, 30, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(symbol, font, 1.4, 2)
            cv2.putText(img, symbol, (cx - tw // 2, cy + th // 2),
                        font, 1.4, clr, 2, cv2.LINE_AA)

    def _draw_labels(self, img):
        """Draw a‑h file labels and 1‑8 rank labels around the board."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thickness = 1
        clr = (200, 200, 200)
        for i in range(8):
            # File labels (bottom)
            file_char = chr(ord('a') + (i if not self.flipped else 7 - i))
            fx = self.margin + i * self.square_size + self.square_size // 2 - 4
            fy = self.margin + 8 * self.square_size + 15
            cv2.putText(img, file_char, (fx, fy), font, scale, clr, thickness, cv2.LINE_AA)
            # Rank labels (left)
            rank_char = str(8 - i if not self.flipped else i + 1)
            rx = self.margin - 15
            ry = self.margin + i * self.square_size + self.square_size // 2 + 4
            cv2.putText(img, rank_char, (rx, ry), font, scale, clr, thickness, cv2.LINE_AA)

    def _draw_captured_tray(self, img, captured_white, captured_black):
        """Draw captured pieces to the right of the board."""
        piece_order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        tray_x = self.margin + 8 * self.square_size + 15
        tray_sz = self.square_size // 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Captured from black (white player took these) – top
        cv2.putText(img, "Captured:", (tray_x, self.margin + 12),
                    font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y = self.margin + 25
        for pt in piece_order:
            count = captured_black.count(pt)
            for _ in range(count):
                piece_img = self.piece_images.get((pt, False))
                if piece_img is not None and tray_x + tray_sz <= img.shape[1] and y + tray_sz <= img.shape[0]:
                    resized = cv2.resize(piece_img, (tray_sz, tray_sz))
                    if resized.shape[2] == 4:
                        alpha = resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[y:y + tray_sz, tray_x:tray_x + tray_sz, c] = (
                                alpha * resized[:, :, c]
                                + (1 - alpha) * img[y:y + tray_sz, tray_x:tray_x + tray_sz, c]
                            )
                y += tray_sz + 2

        # Captured from white (black player took these) – lower
        y = max(y + 15, self.margin + 8 * self.square_size // 2)
        for pt in piece_order:
            count = captured_white.count(pt)
            for _ in range(count):
                piece_img = self.piece_images.get((pt, True))
                if piece_img is not None and tray_x + tray_sz <= img.shape[1] and y + tray_sz <= img.shape[0]:
                    resized = cv2.resize(piece_img, (tray_sz, tray_sz))
                    if resized.shape[2] == 4:
                        alpha = resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[y:y + tray_sz, tray_x:tray_x + tray_sz, c] = (
                                alpha * resized[:, :, c]
                                + (1 - alpha) * img[y:y + tray_sz, tray_x:tray_x + tray_sz, c]
                            )
                y += tray_sz + 2

    # ── 8. Evaluation bar ──────────────────────────────────────────────────

    def draw_eval_bar(self, img, score_cp):
        """Draw a vertical evaluation bar to the left of the board.

        *score_cp* is in centipawns from white's POV (positive = white better).
        """
        if score_cp is None:
            return

        bar_w = 14
        bar_h = self.square_size * 8
        bar_x = max(self.margin - bar_w - 4, 1)
        bar_y = self.margin

        # Clamp score to [-1000, 1000] for display; map to 0..1 for white portion
        clamped = max(-1000, min(1000, score_cp))
        white_frac = (clamped + 1000) / 2000.0   # 0 = black winning, 1 = white winning

        white_h = int(bar_h * white_frac)
        black_h = bar_h - white_h

        # Black portion (top)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + black_h),
                      (40, 40, 40), -1)
        # White portion (bottom)
        cv2.rectangle(img, (bar_x, bar_y + black_h), (bar_x + bar_w, bar_y + bar_h),
                      (230, 230, 230), -1)
        # Border
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), 1)

        # Score text
        if abs(score_cp) >= 10000:
            txt = "M" if score_cp > 0 else "-M"
        else:
            txt = f"{score_cp / 100:+.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(txt, font, 0.35, 1)
        tx = bar_x + (bar_w - tw) // 2
        ty = bar_y + bar_h + th + 4
        cv2.putText(img, txt, (tx, ty), font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    # Legacy helpers kept for backward compat --------------------------------

    def draw_piece(self, img, piece, center):
        self._draw_piece_at_pixel(img, piece, center[0], center[1])

    def overlay_png(self, bg, fg, x, y):
        h, w = fg.shape[:2]
        if fg.shape[2] == 4:
            alpha = fg[:, :, 3] / 255.0
            for c in range(3):
                bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * fg[:, :, c]
        else:
            bg[y:y+h, x:x+w] = fg

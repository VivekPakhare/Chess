"""
Chess engine with Stockfish integration and fallback greedy evaluation.

Difficulty levels:
  - "easy"   : greedy capture-only engine (original)
  - "medium" : Stockfish depth 5, skill level 5
  - "hard"   : Stockfish depth 15, skill level 20
"""
import chess
import chess.engine
import random
import shutil
import os


def _find_stockfish():
    """Locate the Stockfish binary on the system."""
    # 1. Explicit environment variable
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2. Common names on PATH
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return found
    # 3. Known install locations
    common_paths = [
        r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\stockfish\stockfish.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p
    return None


class ChessEngine:
    DIFFICULTIES = {
        "easy":   {"use_stockfish": False},
        "medium": {"use_stockfish": True, "depth": 5,  "skill": 5},
        "hard":   {"use_stockfish": True, "depth": 15, "skill": 20},
    }

    def __init__(self, difficulty="medium"):
        self.difficulty = difficulty
        self.settings = self.DIFFICULTIES.get(difficulty, self.DIFFICULTIES["medium"])
        self._stockfish = None
        self._stockfish_available = False

        if self.settings["use_stockfish"]:
            sf_path = _find_stockfish()
            if sf_path:
                try:
                    self._stockfish = chess.engine.SimpleEngine.popen_uci(sf_path)
                    self._stockfish.configure({"Skill Level": self.settings["skill"]})
                    self._stockfish_available = True
                    print(f"[Engine] Stockfish loaded ({difficulty}) — depth {self.settings['depth']}, skill {self.settings['skill']}")
                except Exception as e:
                    print(f"[Engine] Failed to start Stockfish: {e}. Falling back to greedy engine.")
            else:
                print("[Engine] Stockfish not found on PATH. Falling back to greedy engine.")
                print("         Install Stockfish and ensure it's on your PATH, or set STOCKFISH_PATH.")

    # ---- public API -------------------------------------------------------

    def choose_move(self, board):
        if not list(board.legal_moves):
            return None
        if self._stockfish_available:
            return self._stockfish_move(board)
        return self._greedy_move(board)

    def quit(self):
        """Cleanly shut down Stockfish process."""
        if self._stockfish is not None:
            try:
                self._stockfish.quit()
            except Exception:
                pass
            self._stockfish = None
            self._stockfish_available = False

    def evaluate(self, board):
        """Return position evaluation in centipawns from white's perspective.
        Positive = white is better, negative = black is better.
        Returns None if evaluation is unavailable."""
        if not self._stockfish_available:
            return self._greedy_evaluate(board)
        try:
            info = self._stockfish.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].white()
            if score.is_mate():
                mate_in = score.mate()
                return 10000 if mate_in > 0 else -10000
            return score.score()
        except Exception:
            return self._greedy_evaluate(board)

    @staticmethod
    def _greedy_evaluate(board):
        """Simple material count evaluation (fallback)."""
        values = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
                  chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
        score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                v = values.get(piece.piece_type, 0)
                score += v if piece.color == chess.WHITE else -v
        return score

    # ---- Stockfish --------------------------------------------------------

    def _stockfish_move(self, board):
        try:
            result = self._stockfish.play(
                board,
                chess.engine.Limit(depth=self.settings["depth"]),
            )
            return result.move
        except Exception as e:
            print(f"[Engine] Stockfish error: {e}. Using greedy fallback.")
            return self._greedy_move(board)

    # ---- Greedy fallback --------------------------------------------------

    def _greedy_move(self, board):
        best_value = -9999
        best_moves = []
        for move in board.legal_moves:
            value = self._evaluate_move(board, move)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        return random.choice(best_moves) if best_moves else None

    @staticmethod
    def _evaluate_move(board, move):
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                return values.get(captured.piece_type, 0)
        return 0

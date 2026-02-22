"""
Simple chess engine for computer move selection (greedy evaluation).
"""
import chess
import random

class ChessEngine:
    def __init__(self):
        pass

    def choose_move(self, board):
        # Greedy: pick move with highest capture value, else random legal move
        best_value = -9999
        best_moves = []
        for move in board.legal_moves:
            value = self.evaluate_move(board, move)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        if best_moves:
            return random.choice(best_moves)
        return None

    def evaluate_move(self, board, move):
        # Simple: material gain only
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                return self.piece_value(captured.piece_type)
        return 0

    def piece_value(self, piece_type):
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        return values.get(piece_type, 0)

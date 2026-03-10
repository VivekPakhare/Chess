# Hand Gesture Chess ♟️

A real-time chess game controlled by hand gestures using computer vision. Move chess pieces on a virtual board by pinching and dragging with your fingers — detected via webcam and MediaPipe. Play against **Stockfish** AI with adjustable difficulty.

## Features

### Core Gameplay
- Play chess against the computer using hand gestures (no mouse needed)
- Drag and drop pieces with pinch gesture (thumb + index finger)
- Full move validation via `python-chess` (legal moves, castling, en passant, promotion)
- Game-end detection: checkmate, stalemate, insufficient material, resignation

### AI Opponent (Stockfish Integration)
- Three difficulty levels: **Easy** (greedy), **Medium** (Stockfish depth 5), **Hard** (Stockfish depth 15)
- Automatic fallback to basic engine if Stockfish is not installed
- Real-time **evaluation bar** showing who's winning

### Board & Visual Customization
- **4 board themes**: Wood, Marble, Dark, Classic (press `T` to cycle)
- **Board flip**: View from Black's perspective (press `F`)
- **Move animation**: Computer's pieces slide smoothly across the board
- **Captured pieces tray**: Shows captured pieces beside the board
- **Rank/file labels**: a-h and 1-8 displayed around the board
- **Last move highlight**: From/to squares highlighted in amber

### Gesture Controls
| Gesture | Action |
|---------|--------|
| **Pinch** (thumb + index) | Select & drag a piece |
| **Release** | Drop piece on target square |
| **Thumb-up** | Undo last move pair |
| **Two-finger tap** | Start new game |
| **Open palm** (5 fingers) | Pause / show help overlay |
| **Fist** | Resign game |

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `T` | Cycle board theme |
| `F` | Flip board |
| `U` | Undo last move |
| `N` | New game |
| `ESC` | Quit |

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Chess.git
cd Chess
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the hand tracking model
```bash
python hand_landmarker_download.py
```

### 4. (Optional) Install Stockfish for a stronger AI
- Download from https://stockfishchess.org/download/
- Extract and add to your system PATH, **or** set the environment variable:
  ```bash
  set STOCKFISH_PATH=C:\path\to\stockfish.exe
  ```
- Without Stockfish, the game still works using a basic greedy engine.

### 5. Run the game
```bash
python main.py
```

## Project Structure
```
Chess/
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
├── hand_landmarker.task       # MediaPipe hand model (downloaded)
├── hand_landmarker_download.py
├── chess_cv/
│   ├── app.py                 # Main game loop & gesture handling
│   ├── chessboard.py          # Board rendering, themes, eval bar
│   ├── engine.py              # Stockfish integration & fallback AI
│   ├── gesture.py             # Gesture recognition (6 gestures)
│   ├── hand_tracker.py        # MediaPipe hand landmark detection
│   └── assets/
│       └── wood/              # Piece sprite images (PNG)
```

## Requirements
- Python 3.8+
- Webcam
- OpenCV, MediaPipe, python-chess, NumPy
- (Optional) Stockfish chess engine

## License
See [LICENSE](LICENSE) for details.

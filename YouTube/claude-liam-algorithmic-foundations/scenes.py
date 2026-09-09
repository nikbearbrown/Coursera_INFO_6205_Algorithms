"""scenes.py — proxy; Manim clips pre-rendered in manim/<BID>.mp4."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "manim"))
from scenes import *  # noqa: F401, F403

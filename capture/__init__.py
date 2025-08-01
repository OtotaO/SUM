"""
SUM Zero-Friction Capture System

Revolutionary capture system that transforms content capture from
"another great tool" into "the future of human-computer cognitive collaboration."

Features:
- Global hotkey system for instant desktop capture
- Browser extension for seamless webpage capture
- Background processing with optimized SumEngine
- Mobile-first voice and OCR capture
- API webhooks for team collaboration
- Email integration with smart filtering

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

from .capture_engine import CaptureEngine, CaptureSource, CaptureStatus, capture_engine
from .global_hotkey import GlobalHotkeyManager, hotkey_manager

__version__ = "1.0.0"
__all__ = [
    'CaptureEngine',
    'CaptureSource', 
    'CaptureStatus',
    'capture_engine',
    'GlobalHotkeyManager',
    'hotkey_manager'
]
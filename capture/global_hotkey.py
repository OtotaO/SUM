"""
Global Hotkey System - Cross-Platform Instant Capture

Revolutionary global hotkey system that works anywhere on desktop.
Provides instant text capture with beautiful popup UI that appears
in under 100ms.

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import sys
import logging
from typing import Optional, Callable, Dict, Any
import json
import os

# Cross-platform hotkey support
try:
    import keyboard  # pip install keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("Warning: 'keyboard' library not available. Install with: pip install keyboard")

# Import capture engine
from .capture_engine import capture_engine, CaptureSource

logger = logging.getLogger(__name__)


class CapturePopup:
    """
    Beautiful, minimal capture popup that appears instantly.
    Designed for zero-friction user experience.
    """
    
    def __init__(self, on_capture: Callable[[str], None]):
        self.on_capture = on_capture
        self.root = None
        self.text_widget = None
        self.status_label = None
        self.progress_bar = None
        self.is_visible = False
        
        # Styling configuration
        self.style_config = {
            'bg_color': '#1e1e1e',
            'text_color': '#ffffff',
            'accent_color': '#0078d4',
            'success_color': '#4caf50',
            'error_color': '#f44336',
            'font_family': 'Segoe UI' if sys.platform.startswith('win') else 'SF Pro Display' if sys.platform == 'darwin' else 'Ubuntu',
            'font_size': 12
        }
    
    def show(self):
        """Show the capture popup with sub-100ms response time."""
        if self.is_visible:
            return
        
        start_time = time.time()
        
        # Create root window
        self.root = tk.Tk()
        self.root.title("SUM - Quick Capture")
        
        # Window configuration for instant appearance
        self.root.geometry("600x400")
        self.root.configure(bg=self.style_config['bg_color'])
        self.root.resizable(True, True)
        
        # Make window stay on top and center it
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.95)  # Slight transparency for modern look
        
        # Center window on screen
        self._center_window()
        
        # Remove window decorations for minimal look (optional)
        if sys.platform.startswith('win'):
            self.root.attributes('-toolwindow', True)
        
        # Create UI elements
        self._create_ui()
        
        # Bind events
        self._bind_events()
        
        # Focus on text area immediately
        self.text_widget.focus_set()
        
        self.is_visible = True
        
        # Log performance
        creation_time = (time.time() - start_time) * 1000
        logger.info(f"Capture popup created in {creation_time:.1f}ms")
        
        # Start the GUI event loop
        self.root.mainloop()
    
    def _center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position
        x = (screen_width - 600) // 2
        y = (screen_height - 400) // 2
        
        self.root.geometry(f"600x400+{x}+{y}")
    
    def _create_ui(self):
        """Create the beautiful, minimal UI."""
        # Configure custom style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Custom.TFrame', background=self.style_config['bg_color'])
        style.configure('Custom.TLabel', 
                       background=self.style_config['bg_color'], 
                       foreground=self.style_config['text_color'],
                       font=(self.style_config['font_family'], self.style_config['font_size']))
        
        # Main container
        main_frame = ttk.Frame(self.root, style='Custom.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, 
                               text="✨ Quick Capture", 
                               style='Custom.TLabel',
                               font=(self.style_config['font_family'], 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # Instructions
        instructions = ttk.Label(header_frame,
                               text="Paste or type your text • Ctrl+Enter to capture • Esc to cancel",
                               style='Custom.TLabel',
                               font=(self.style_config['font_family'], 10))
        instructions.pack(side=tk.RIGHT)
        
        # Text input area
        text_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create text widget with custom styling
        self.text_widget = tk.Text(text_frame,
                                  wrap=tk.WORD,
                                  font=(self.style_config['font_family'], self.style_config['font_size']),
                                  bg='#2d2d2d',
                                  fg=self.style_config['text_color'],
                                  insertbackground=self.style_config['accent_color'],
                                  selectbackground=self.style_config['accent_color'],
                                  selectforeground='white',
                                  relief=tk.FLAT,
                                  padx=15,
                                  pady=15)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status and progress area
        status_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame,
                                     text="Ready to capture",
                                     style='Custom.TLabel',
                                     font=(self.style_config['font_family'], 10))
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar (initially hidden)
        self.progress_bar = ttk.Progressbar(status_frame, 
                                           mode='indeterminate',
                                           length=150)
        
        # Buttons
        button_frame = ttk.Frame(status_frame, style='Custom.TFrame')
        button_frame.pack(side=tk.RIGHT)
        
        # Capture button
        capture_btn = tk.Button(button_frame,
                               text="Capture & Summarize",
                               command=self._handle_capture,
                               bg=self.style_config['accent_color'],
                               fg='white',
                               font=(self.style_config['font_family'], 10, 'bold'),
                               relief=tk.FLAT,
                               padx=20,
                               pady=8,
                               cursor='hand2')
        capture_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Cancel button
        cancel_btn = tk.Button(button_frame,
                              text="Cancel",
                              command=self._handle_cancel,
                              bg='#404040',
                              fg=self.style_config['text_color'],
                              font=(self.style_config['font_family'], 10),
                              relief=tk.FLAT,
                              padx=20,
                              pady=8,
                              cursor='hand2')
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Auto-paste clipboard content if available
        self._auto_paste_clipboard()
    
    def _bind_events(self):
        """Bind keyboard events for quick interactions."""
        # Ctrl+Enter to capture
        self.root.bind('<Control-Return>', lambda e: self._handle_capture())
        
        # Escape to cancel
        self.root.bind('<Escape>', lambda e: self._handle_cancel())
        
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._handle_cancel)
        
        # Auto-resize based on content
        self.text_widget.bind('<KeyRelease>', self._on_text_change)
    
    def _auto_paste_clipboard(self):
        """Auto-paste clipboard content if it looks like text."""
        try:
            clipboard_content = self.root.clipboard_get()
            if clipboard_content and len(clipboard_content.strip()) > 10:
                self.text_widget.insert(tk.END, clipboard_content)
                self.text_widget.mark_set(tk.INSERT, tk.END)
                self.text_widget.see(tk.INSERT)
        except tk.TclError:
            # Clipboard is empty or contains non-text data
            pass
    
    def _on_text_change(self, event=None):
        """Handle text changes for dynamic UI updates."""
        text_content = self.text_widget.get(1.0, tk.END).strip()
        word_count = len(text_content.split()) if text_content else 0
        
        if word_count > 0:
            self.status_label.config(text=f"Ready to capture • {word_count} words")
        else:
            self.status_label.config(text="Ready to capture")
    
    def _handle_capture(self):
        """Handle the capture action with beautiful progress indication."""
        text_content = self.text_widget.get(1.0, tk.END).strip()
        
        if not text_content:
            self._update_status("Please enter some text to capture", "error")
            return
        
        # Show progress
        self._show_progress("Processing your text...")
        
        # Capture the text
        try:
            self.on_capture(text_content)
            self._update_status("Captured successfully!", "success")
            
            # Close after brief delay
            self.root.after(1500, self._handle_cancel)
            
        except Exception as e:
            self._update_status(f"Capture failed: {str(e)}", "error")
            logger.error(f"Capture failed: {e}", exc_info=True)
    
    def _handle_cancel(self):
        """Handle cancel/close action."""
        self.is_visible = False
        if self.root:
            self.root.destroy()
    
    def _show_progress(self, message: str):
        """Show progress indicator."""
        self.status_label.config(text=message)
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        self.progress_bar.start(10)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status with color coding."""
        colors = {
            "info": self.style_config['text_color'],
            "success": self.style_config['success_color'],
            "error": self.style_config['error_color']
        }
        
        self.status_label.config(text=message, foreground=colors.get(status_type, colors["info"]))
        
        if hasattr(self, 'progress_bar'):
            self.progress_bar.stop()
            self.progress_bar.pack_forget()


class GlobalHotkeyManager:
    """
    Cross-platform global hotkey manager for instant capture.
    Monitors for Ctrl+Shift+T and shows beautiful capture popup.
    """
    
    def __init__(self):
        self.is_active = False
        self.hotkey_thread = None
        self.current_popup = None
        
        # Configuration
        self.config = {
            'hotkey': 'ctrl+shift+t',
            'enabled': True,
            'auto_paste_clipboard': True,
            'popup_timeout': 30  # seconds
        }
        
        # Load saved configuration
        self._load_config()
        
        logger.info("GlobalHotkeyManager initialized")
    
    def start(self):
        """Start monitoring for global hotkeys."""
        if not KEYBOARD_AVAILABLE:
            logger.error("Cannot start hotkey monitoring: keyboard library not available")
            return False
        
        if self.is_active:
            logger.warning("Hotkey monitoring already active")
            return True
        
        try:
            # Register hotkey
            keyboard.add_hotkey(self.config['hotkey'], self._on_hotkey_triggered)
            self.is_active = True
            
            logger.info(f"Global hotkey registered: {self.config['hotkey']}")
            print(f"🚀 SUM Global Capture is active! Press {self.config['hotkey'].upper()} anywhere to capture text.")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register global hotkey: {e}")
            return False
    
    def stop(self):
        """Stop monitoring for global hotkeys."""
        if not self.is_active:
            return
        
        try:
            if KEYBOARD_AVAILABLE:
                keyboard.unhook_all_hotkeys()
            
            self.is_active = False
            logger.info("Global hotkey monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping hotkey monitoring: {e}")
    
    def _on_hotkey_triggered(self):
        """Handle hotkey trigger - show capture popup."""
        if self.current_popup and self.current_popup.is_visible:
            # Popup already visible, ignore
            return
        
        logger.info("Global hotkey triggered - showing capture popup")
        
        # Create and show popup in separate thread to avoid blocking
        popup_thread = threading.Thread(target=self._show_capture_popup, daemon=True)
        popup_thread.start()
    
    def _show_capture_popup(self):
        """Show the capture popup with instant response."""
        try:
            self.current_popup = CapturePopup(self._handle_text_capture)
            self.current_popup.show()
        except Exception as e:
            logger.error(f"Error showing capture popup: {e}", exc_info=True)
    
    def _handle_text_capture(self, text: str):
        """Handle captured text by sending to capture engine."""
        try:
            # Submit to capture engine for processing
            request_id = capture_engine.capture_text(
                text=text,
                source=CaptureSource.GLOBAL_HOTKEY,
                context={'hotkey_used': self.config['hotkey']},
                callback=self._on_capture_complete
            )
            
            logger.info(f"Text captured via global hotkey, request_id: {request_id}")
            
        except Exception as e:
            logger.error(f"Error handling text capture: {e}", exc_info=True)
    
    def _on_capture_complete(self, result):
        """Handle capture completion callback."""
        logger.info(f"Capture completed: {result.request_id} in {result.processing_time:.3f}s")
        
        # Could show notification or save to history here
        print(f"✅ Captured and summarized in {result.processing_time:.3f}s")
        print(f"Summary: {result.summary[:100]}...")
    
    def _load_config(self):
        """Load configuration from file."""
        config_path = os.path.join(os.path.dirname(__file__), 'hotkey_config.json')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    logger.info("Hotkey configuration loaded")
        except Exception as e:
            logger.warning(f"Could not load hotkey config: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        config_path = os.path.join(os.path.dirname(__file__), 'hotkey_config.json')
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                logger.info("Hotkey configuration saved")
        except Exception as e:
            logger.error(f"Could not save hotkey config: {e}")
    
    def update_config(self, **kwargs):
        """Update configuration settings."""
        self.config.update(kwargs)
        self.save_config()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hotkey system statistics."""
        return {
            'is_active': self.is_active,
            'hotkey': self.config['hotkey'],
            'keyboard_available': KEYBOARD_AVAILABLE,
            'current_popup_visible': self.current_popup and self.current_popup.is_visible
        }


# Global instance
hotkey_manager = GlobalHotkeyManager()


def main():
    """Main function for testing the global hotkey system."""
    import signal
    
    def signal_handler(signum, frame):
        print("\nShutting down global hotkey system...")
        hotkey_manager.stop()
        capture_engine.shutdown()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the hotkey system
    if hotkey_manager.start():
        try:
            # Keep the main thread alive
            keyboard.wait()
        except KeyboardInterrupt:
            pass
        finally:
            hotkey_manager.stop()
            capture_engine.shutdown()
    else:
        print("Failed to start global hotkey system")
        sys.exit(1)


if __name__ == "__main__":
    main()
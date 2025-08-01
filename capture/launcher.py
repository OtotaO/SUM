"""
SUM Capture System Launcher

Unified launcher for the zero-friction capture system.
Starts all capture services including global hotkeys and API server.

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import sys
import signal
import threading
import time
import logging
import argparse
from typing import List, Optional

# Import capture components
from .global_hotkey import hotkey_manager
from .capture_engine import capture_engine
from .api_server import app, start_cleanup_thread, server_stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaptureSystemLauncher:
    """
    Unified launcher for the SUM capture system.
    Manages all capture services in a coordinated manner.
    """
    
    def __init__(self):
        self.services_running = False
        self.hotkey_active = False
        self.api_server_active = False
        self.shutdown_event = threading.Event()
        
        # Service threads
        self.api_server_thread: Optional[threading.Thread] = None
        
        logger.info("SUM Capture System Launcher initialized")
    
    def start_all_services(self, 
                          enable_hotkey: bool = True,
                          enable_api_server: bool = True,
                          api_host: str = '127.0.0.1',
                          api_port: int = 8000):
        """Start all capture services."""
        logger.info("🚀 Starting SUM Zero-Friction Capture System...")
        
        services_started = []
        
        try:
            # Start global hotkey system
            if enable_hotkey:
                if hotkey_manager.start():
                    self.hotkey_active = True
                    services_started.append("Global Hotkey System")
                    logger.info("✅ Global hotkey system started (Ctrl+Shift+T)")
                else:
                    logger.warning("⚠️  Global hotkey system failed to start")
            
            # Start API server
            if enable_api_server:
                self.api_server_thread = threading.Thread(
                    target=self._run_api_server,
                    args=(api_host, api_port),
                    daemon=True
                )
                self.api_server_thread.start()
                
                # Wait a moment for server to start
                time.sleep(1)
                self.api_server_active = True
                services_started.append(f"API Server (http://{api_host}:{api_port})")
                logger.info(f"✅ API server started on http://{api_host}:{api_port}")
            
            # Start cleanup thread
            start_cleanup_thread()
            
            self.services_running = True
            
            # Print startup summary
            self._print_startup_summary(services_started, api_host, api_port)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start capture system: {e}", exc_info=True)
            self.shutdown_all_services()
            return False
    
    def shutdown_all_services(self):
        """Gracefully shutdown all services."""
        if not self.services_running:
            return
        
        logger.info("🛑 Shutting down SUM Capture System...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop global hotkey system
        if self.hotkey_active:
            hotkey_manager.stop()
            self.hotkey_active = False
            logger.info("✅ Global hotkey system stopped")
        
        # Shutdown capture engine
        capture_engine.shutdown()
        logger.info("✅ Capture engine shutdown")
        
        # API server will be stopped when main thread exits
        if self.api_server_active:
            self.api_server_active = False
            logger.info("✅ API server shutdown")
        
        self.services_running = False
        logger.info("🎯 All services stopped successfully")
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def get_status(self) -> dict:
        """Get status of all services."""
        return {
            'services_running': self.services_running,
            'hotkey_active': self.hotkey_active,
            'api_server_active': self.api_server_active,
            'capture_engine_stats': capture_engine.get_stats(),
            'server_stats': server_stats.copy(),
            'uptime': time.time() - server_stats['server_start_time'] if self.services_running else 0
        }
    
    def _run_api_server(self, host: str, port: int):
        """Run the API server in a separate thread."""
        try:
            app.run(
                host=host,
                port=port,
                debug=False,
                threaded=True,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"API server error: {e}", exc_info=True)
    
    def _print_startup_summary(self, services: List[str], api_host: str, api_port: int):
        """Print a beautiful startup summary."""
        print("\n" + "="*60)
        print("🎯 SUM ZERO-FRICTION CAPTURE SYSTEM")
        print("="*60)
        print("✨ Revolutionary capture system is now active!")
        print("")
        
        if services:
            print("🚀 ACTIVE SERVICES:")
            for service in services:
                print(f"   • {service}")
            print("")
        
        if self.hotkey_active:
            print("⌨️  GLOBAL HOTKEY:")
            print("   • Press Ctrl+Shift+T anywhere to capture text")
            print("   • Instant popup with sub-100ms response time")
            print("")
        
        if self.api_server_active:
            print("🌐 API ENDPOINTS:")
            print(f"   • Capture: POST http://{api_host}:{api_port}/api/capture")
            print(f"   • Status:  GET  http://{api_host}:{api_port}/health")
            print(f"   • Stats:   GET  http://{api_host}:{api_port}/api/stats")
            print("")
        
        print("🔥 FEATURES READY:")
        print("   • Context-aware summarization")
        print("   • Sub-second processing for most content") 
        print("   • Beautiful progress indication")
        print("   • Cross-platform compatibility")
        print("")
        
        print("💡 TIP: Install the browser extension for webpage capture!")
        print("="*60)
        print("Press Ctrl+C to stop all services")
        print("")


# Global launcher instance
launcher = CaptureSystemLauncher()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n")
    logger.info("Received shutdown signal")
    launcher.shutdown_all_services()
    sys.exit(0)


def main():
    """Main entry point for the capture system."""
    parser = argparse.ArgumentParser(
        description="SUM Zero-Friction Capture System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m capture.launcher                    # Start all services
  python -m capture.launcher --no-hotkey       # Start without hotkey
  python -m capture.launcher --port 9000       # Custom API port
  python -m capture.launcher --api-only        # API server only
        """
    )
    
    parser.add_argument(
        '--no-hotkey', 
        action='store_true',
        help='Disable global hotkey system'
    )
    
    parser.add_argument(
        '--no-api',
        action='store_true', 
        help='Disable API server'
    )
    
    parser.add_argument(
        '--api-only',
        action='store_true',
        help='Start only the API server'
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='API server host (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Configure service flags
    enable_hotkey = not args.no_hotkey and not args.api_only
    enable_api_server = not args.no_api
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start services
        if launcher.start_all_services(
            enable_hotkey=enable_hotkey,
            enable_api_server=enable_api_server,
            api_host=args.host,
            api_port=args.port
        ):
            # Wait for shutdown
            launcher.wait_for_shutdown()
        else:
            logger.error("Failed to start capture system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        launcher.shutdown_all_services()


if __name__ == "__main__":
    main()
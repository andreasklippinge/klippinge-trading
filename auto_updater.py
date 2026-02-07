"""
Auto-Updater med Skip Version Logik
====================================
Ersätt din nuvarande auto_updater.py med denna version.

Förändringar:
- Sparar vilken version användaren valt att hoppa över
- Visar inte popup igen förrän en NY version släpps
- Konfigurationsfil: skipped_version.json i user data directory
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QMessageBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QProgressBar, QApplication
)
from PyQt5.QtCore import QTimer, QThread, QObject, pyqtSignal as Signal, Qt
from PyQt5.QtGui import QFont

# Försök importera requests för GitHub API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available for auto-updater")

# Import app config för version och paths
try:
    from app_config import APP_VERSION, get_user_data_dir
except ImportError:
    APP_VERSION = "0.0.0"
    def get_user_data_dir():
        return os.path.expanduser("~/.klippinge_terminal")


# ============================================================================
# CONFIGURATION
# ============================================================================

GITHUB_REPO = "your-username/your-repo"  # Ändra till din repo
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CHECK_INTERVAL_MS = 2 * 60 * 60 * 1000  # 2 timmar i millisekunder
SKIPPED_VERSION_FILE = "skipped_version.json"


# ============================================================================
# SKIPPED VERSION MANAGEMENT
# ============================================================================

def get_skipped_version_path() -> str:
    """Get path to skipped version config file."""
    user_dir = get_user_data_dir()
    return os.path.join(user_dir, SKIPPED_VERSION_FILE)


def load_skipped_version() -> Optional[str]:
    """Load the version that user chose to skip."""
    try:
        path = get_skipped_version_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('skipped_version')
    except Exception as e:
        print(f"[AutoUpdater] Error loading skipped version: {e}")
    return None


def save_skipped_version(version: str):
    """Save the version that user chose to skip."""
    try:
        path = get_skipped_version_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'skipped_version': version,
                'skipped_at': datetime.now().isoformat()
            }, f, indent=2)
        print(f"[AutoUpdater] Saved skipped version: {version}")
    except Exception as e:
        print(f"[AutoUpdater] Error saving skipped version: {e}")


def clear_skipped_version():
    """Clear the skipped version (called when user updates)."""
    try:
        path = get_skipped_version_path()
        if os.path.exists(path):
            os.remove(path)
            print("[AutoUpdater] Cleared skipped version")
    except Exception as e:
        print(f"[AutoUpdater] Error clearing skipped version: {e}")


# ============================================================================
# VERSION COMPARISON
# ============================================================================

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple for comparison.
    
    Handles formats like: "1.0.0", "v1.0.0", "1.0.0-beta"
    """
    # Remove 'v' prefix if present
    version_str = version_str.lstrip('v').strip()
    
    # Remove any suffix like -beta, -rc1, etc.
    if '-' in version_str:
        version_str = version_str.split('-')[0]
    
    # Split and convert to integers
    try:
        parts = version_str.split('.')
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def is_newer_version(remote: str, local: str) -> bool:
    """Check if remote version is newer than local."""
    remote_tuple = parse_version(remote)
    local_tuple = parse_version(local)
    return remote_tuple > local_tuple


# ============================================================================
# UPDATE CHECKER WORKER
# ============================================================================

class UpdateCheckerWorker(QObject):
    """Background worker for checking updates."""
    
    finished = Signal()
    update_available = Signal(str, str, str)  # (version, download_url, release_notes)
    no_update = Signal()
    error = Signal(str)
    
    def __init__(self, current_version: str, skipped_version: Optional[str] = None):
        super().__init__()
        self.current_version = current_version
        self.skipped_version = skipped_version
    
    def run(self):
        """Check for updates from GitHub releases."""
        if not REQUESTS_AVAILABLE:
            self.error.emit("requests library not available")
            self.finished.emit()
            return
        
        try:
            print(f"[AutoUpdater] Checking for updates... (current: {self.current_version})")
            
            response = requests.get(
                GITHUB_API_URL,
                headers={'Accept': 'application/vnd.github.v3+json'},
                timeout=10
            )
            
            if response.status_code == 404:
                print("[AutoUpdater] No releases found")
                self.no_update.emit()
                self.finished.emit()
                return
            
            response.raise_for_status()
            release = response.json()
            
            remote_version = release.get('tag_name', '').lstrip('v')
            download_url = ""
            release_notes = release.get('body', 'No release notes available.')
            
            # Hitta Windows installer/exe i assets
            for asset in release.get('assets', []):
                name = asset.get('name', '').lower()
                if name.endswith('.exe') or 'setup' in name or 'installer' in name:
                    download_url = asset.get('browser_download_url', '')
                    break
            
            # Fallback till release page om ingen asset hittades
            if not download_url:
                download_url = release.get('html_url', '')
            
            print(f"[AutoUpdater] Remote version: {remote_version}")
            
            # Kontrollera om detta är en ny version
            if is_newer_version(remote_version, self.current_version):
                # Kontrollera om användaren redan hoppat över denna version
                if self.skipped_version and remote_version == self.skipped_version:
                    print(f"[AutoUpdater] Version {remote_version} was skipped by user")
                    self.no_update.emit()
                else:
                    print(f"[AutoUpdater] New version available: {remote_version}")
                    self.update_available.emit(remote_version, download_url, release_notes)
            else:
                print("[AutoUpdater] No new version available")
                self.no_update.emit()
            
        except requests.exceptions.Timeout:
            self.error.emit("Update check timed out")
        except requests.exceptions.RequestException as e:
            self.error.emit(f"Network error: {e}")
        except Exception as e:
            self.error.emit(f"Update check failed: {e}")
        finally:
            self.finished.emit()


# ============================================================================
# UPDATE DIALOG
# ============================================================================

class UpdateDialog(QDialog):
    """Dialog for showing update information with Skip option."""
    
    def __init__(self, version: str, download_url: str, release_notes: str, parent=None):
        super().__init__(parent)
        self.version = version
        self.download_url = download_url
        self.release_notes = release_notes
        self.result_action = None  # 'update', 'skip', 'later'
        
        self.setWindowTitle("Update Available")
        self.setMinimumWidth(450)
        self.setModal(True)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel(f"🚀 Version {self.version} is available!")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet("color: #d4a574;")
        layout.addWidget(header)
        
        # Current version info
        from app_config import APP_VERSION
        current_label = QLabel(f"Your current version: {APP_VERSION}")
        current_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(current_label)
        
        # Release notes (truncated)
        notes_preview = self.release_notes[:500] + "..." if len(self.release_notes) > 500 else self.release_notes
        notes_label = QLabel(f"What's new:\n{notes_preview}")
        notes_label.setWordWrap(True)
        notes_label.setStyleSheet("""
            background-color: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 4px;
            padding: 10px;
            color: #cccccc;
            font-size: 11px;
        """)
        notes_label.setMaximumHeight(150)
        layout.addWidget(notes_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        # Skip this version button
        self.skip_btn = QPushButton("Skip This Version")
        self.skip_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #555555;
                color: #888888;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                border-color: #777777;
                color: #aaaaaa;
            }
        """)
        self.skip_btn.clicked.connect(self.on_skip)
        btn_layout.addWidget(self.skip_btn)
        
        btn_layout.addStretch()
        
        # Remind me later button
        self.later_btn = QPushButton("Later")
        self.later_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                color: #cccccc;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
        """)
        self.later_btn.clicked.connect(self.on_later)
        btn_layout.addWidget(self.later_btn)
        
        # Download button
        self.download_btn = QPushButton("Download Update")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #d4a574;
                border: none;
                color: #0a0a0a;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0b585;
            }
        """)
        self.download_btn.clicked.connect(self.on_download)
        btn_layout.addWidget(self.download_btn)
        
        layout.addLayout(btn_layout)
    
    def on_skip(self):
        """User chose to skip this version."""
        self.result_action = 'skip'
        save_skipped_version(self.version)
        self.accept()
    
    def on_later(self):
        """User chose to be reminded later."""
        self.result_action = 'later'
        self.reject()
    
    def on_download(self):
        """User chose to download the update."""
        self.result_action = 'update'
        clear_skipped_version()  # Clear any skipped version when updating
        
        # Öppna nedladdningslänken i webbläsaren
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        QDesktopServices.openUrl(QUrl(self.download_url))
        
        self.accept()


# ============================================================================
# AUTO UPDATER CLASS
# ============================================================================

class AutoUpdater(QObject):
    """Auto-updater that checks for new versions periodically."""
    
    update_available = Signal(str, str, str)  # version, url, notes
    
    def __init__(self, parent_window, check_interval_ms: int = CHECK_INTERVAL_MS):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.check_interval_ms = check_interval_ms
        
        self._check_thread = None
        self._check_worker = None
        self._checking = False
        
        # Timer för periodiska kontroller
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_updates)
        
        # Anslut signal
        self.update_available.connect(self._show_update_dialog)
    
    def start(self):
        """Start the auto-updater (check now + start timer)."""
        print(f"[AutoUpdater] Starting with {self.check_interval_ms/1000/60:.0f} minute interval")
        
        # Kör en första kontroll efter kort delay (låt GUI ladda först)
        QTimer.singleShot(5000, self.check_for_updates)
        
        # Starta periodisk timer
        self.timer.start(self.check_interval_ms)
    
    def stop(self):
        """Stop the auto-updater."""
        self.timer.stop()
        print("[AutoUpdater] Stopped")
    
    def check_for_updates(self):
        """Check for updates in background thread."""
        if self._checking:
            print("[AutoUpdater] Already checking, skipping")
            return
        
        self._checking = True
        
        # Ladda skipped version
        skipped = load_skipped_version()
        
        # Skapa worker och tråd
        self._check_thread = QThread()
        self._check_worker = UpdateCheckerWorker(APP_VERSION, skipped)
        self._check_worker.moveToThread(self._check_thread)
        
        # Koppla signaler
        self._check_thread.started.connect(self._check_worker.run)
        self._check_worker.finished.connect(self._on_check_finished)
        self._check_worker.update_available.connect(self._on_update_found)
        self._check_worker.no_update.connect(self._on_no_update)
        self._check_worker.error.connect(self._on_check_error)
        
        self._check_thread.start()
    
    def _on_check_finished(self):
        """Clean up after check completes."""
        self._checking = False
        
        if self._check_worker:
            self._check_worker.deleteLater()
        if self._check_thread:
            self._check_thread.quit()
            self._check_thread.wait()
            self._check_thread.deleteLater()
        
        self._check_worker = None
        self._check_thread = None
    
    def _on_update_found(self, version: str, url: str, notes: str):
        """Handle update found."""
        print(f"[AutoUpdater] Update found: {version}")
        self.update_available.emit(version, url, notes)
    
    def _on_no_update(self):
        """Handle no update available."""
        print("[AutoUpdater] No update available")
    
    def _on_check_error(self, error: str):
        """Handle check error."""
        print(f"[AutoUpdater] Error: {error}")
    
    def _show_update_dialog(self, version: str, url: str, notes: str):
        """Show the update dialog."""
        dialog = UpdateDialog(version, url, notes, self.parent_window)
        dialog.exec_()


# ============================================================================
# SETUP FUNCTION
# ============================================================================

def setup_auto_updater(parent_window, check_interval_ms: int = CHECK_INTERVAL_MS) -> AutoUpdater:
    """Setup and start the auto-updater.
    
    Args:
        parent_window: The main application window
        check_interval_ms: Interval between checks in milliseconds (default 2 hours)
    
    Returns:
        AutoUpdater instance
    """
    updater = AutoUpdater(parent_window, check_interval_ms)
    updater.start()
    return updater


# ============================================================================
# MANUAL CHECK FUNCTION
# ============================================================================

def check_for_updates_now(parent_window):
    """Manually trigger an update check (e.g., from menu).
    
    This bypasses the skipped version check.
    """
    if not REQUESTS_AVAILABLE:
        QMessageBox.warning(
            parent_window,
            "Update Check Failed",
            "Cannot check for updates: requests library not installed."
        )
        return
    
    try:
        response = requests.get(
            GITHUB_API_URL,
            headers={'Accept': 'application/vnd.github.v3+json'},
            timeout=10
        )
        
        if response.status_code == 404:
            QMessageBox.information(
                parent_window,
                "No Updates",
                f"You are running the latest version ({APP_VERSION})."
            )
            return
        
        response.raise_for_status()
        release = response.json()
        
        remote_version = release.get('tag_name', '').lstrip('v')
        
        if is_newer_version(remote_version, APP_VERSION):
            download_url = ""
            for asset in release.get('assets', []):
                name = asset.get('name', '').lower()
                if name.endswith('.exe') or 'setup' in name:
                    download_url = asset.get('browser_download_url', '')
                    break
            if not download_url:
                download_url = release.get('html_url', '')
            
            release_notes = release.get('body', 'No release notes available.')
            
            dialog = UpdateDialog(remote_version, download_url, release_notes, parent_window)
            dialog.exec_()
        else:
            QMessageBox.information(
                parent_window,
                "No Updates",
                f"You are running the latest version ({APP_VERSION})."
            )
    
    except Exception as e:
        QMessageBox.warning(
            parent_window,
            "Update Check Failed",
            f"Could not check for updates:\n{e}"
        )
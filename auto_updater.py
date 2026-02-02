"""
Klippinge Trading Terminal - Auto-Updater
==========================================

Checks GitHub Releases for new versions and handles the update flow.

Flow:
    1. On startup, check latest GitHub release tag
    2. Compare with current APP_VERSION
    3. If newer version found → show dialog with changelog
    4. User clicks "Update" → download installer → run → exit app
    
Usage:
    from auto_updater import AutoUpdater
    updater = AutoUpdater(parent_widget)
    updater.check_for_updates()           # non-blocking (background)
    updater.check_for_updates_blocking()  # blocking (shows dialog)
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QUrl
from PyQt5.QtGui import QFont, QDesktopServices

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class UpdateInfo:
    """Information about an available update."""
    version: str
    download_url: str
    changelog: str
    published_at: str
    file_size: int = 0
    file_name: str = ""


class UpdateChecker(QThread):
    """Background thread that checks GitHub for updates."""
    
    update_available = Signal(object)   # UpdateInfo
    no_update = Signal()
    error = Signal(str)
    
    def __init__(self, github_repo: str, current_version: str):
        super().__init__()
        self.github_repo = github_repo
        self.current_version = current_version
    
    def run(self):
        try:
            info = self._check_github()
            if info:
                self.update_available.emit(info)
            else:
                self.no_update.emit()
        except Exception as e:
            self.error.emit(str(e))
    
    def _check_github(self) -> Optional[UpdateInfo]:
        """Check GitHub API for latest release."""
        if not REQUESTS_AVAILABLE:
            return None
        
        url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
        
        try:
            resp = requests.get(url, timeout=10, headers={
                "Accept": "application/vnd.github.v3+json"
            })
        except requests.RequestException:
            return None
        
        if resp.status_code == 404:
            return None  # No releases yet
        
        resp.raise_for_status()
        data = resp.json()
        
        latest_version = data.get("tag_name", "").lstrip("v")
        
        if not self._is_newer(latest_version, self.current_version):
            return None
        
        # Find the Windows installer asset
        download_url = ""
        file_size = 0
        file_name = ""
        
        for asset in data.get("assets", []):
            name = asset["name"].lower()
            if name.endswith(".exe") or name.endswith(".msi"):
                download_url = asset["browser_download_url"]
                file_size = asset.get("size", 0)
                file_name = asset["name"]
                break
        
        if not download_url:
            # Fallback: link to the release page
            download_url = data.get("html_url", "")
        
        return UpdateInfo(
            version=latest_version,
            download_url=download_url,
            changelog=data.get("body", "No changelog available."),
            published_at=data.get("published_at", ""),
            file_size=file_size,
            file_name=file_name,
        )
    
    @staticmethod
    def _is_newer(latest: str, current: str) -> bool:
        """Compare semantic version strings (e.g. '1.2.3' > '1.2.0')."""
        try:
            def parse(v):
                return tuple(int(x) for x in v.split("."))
            return parse(latest) > parse(current)
        except (ValueError, AttributeError):
            return False


class DownloadThread(QThread):
    """Background thread to download the update installer."""
    
    progress = Signal(int, int)  # (bytes_downloaded, total_bytes)
    finished = Signal(str)       # path to downloaded file
    error = Signal(str)
    
    def __init__(self, url: str, dest_path: str):
        super().__init__()
        self.url = url
        self.dest_path = dest_path
    
    def run(self):
        try:
            resp = requests.get(self.url, stream=True, timeout=300)
            resp.raise_for_status()
            
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            
            with open(self.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.progress.emit(downloaded, total)
            
            self.finished.emit(self.dest_path)
        
        except Exception as e:
            self.error.emit(str(e))


# ── Update Dialog ────────────────────────────────────────────────────────────

class UpdateDialog(QDialog):
    """Professional update dialog with changelog and download progress."""
    
    # Bloomberg-style dark theme for the dialog
    DIALOG_STYLE = """
        QDialog {
            background-color: #1a1a2e;
            color: #e0e0e0;
        }
        QLabel {
            color: #e0e0e0;
        }
        QLabel#title {
            font-size: 16px;
            font-weight: bold;
            color: #ffb300;
        }
        QLabel#version {
            font-size: 13px;
            color: #aaa;
        }
        QTextEdit {
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #333;
            border-radius: 4px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            padding: 8px;
        }
        QPushButton {
            background-color: #ffb300;
            color: #1a1a2e;
            border: none;
            border-radius: 4px;
            padding: 8px 24px;
            font-weight: bold;
            font-size: 13px;
        }
        QPushButton:hover {
            background-color: #ffc107;
        }
        QPushButton#skip {
            background-color: transparent;
            color: #888;
            border: 1px solid #444;
        }
        QPushButton#skip:hover {
            border-color: #666;
            color: #aaa;
        }
        QProgressBar {
            border: 1px solid #333;
            border-radius: 4px;
            background-color: #0d1117;
            text-align: center;
            color: #e0e0e0;
            height: 24px;
        }
        QProgressBar::chunk {
            background-color: #ffb300;
            border-radius: 3px;
        }
    """
    
    def __init__(self, update_info: UpdateInfo, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.update_info = update_info
        self.download_thread = None
        self._setup_ui()
    
    def _setup_ui(self):
        self.setWindowTitle("Update Available")
        self.setMinimumSize(520, 420)
        self.setStyleSheet(self.DIALOG_STYLE)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title = QLabel("🚀  A New Version is Available!")
        title.setObjectName("title")
        layout.addWidget(title)
        
        # Version info
        from app_config import APP_VERSION
        ver_label = QLabel(
            f"Current: v{APP_VERSION}  →  New: v{self.update_info.version}"
        )
        ver_label.setObjectName("version")
        layout.addWidget(ver_label)
        
        # Changelog
        changelog_label = QLabel("What's New:")
        changelog_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        layout.addWidget(changelog_label)
        
        self.changelog_text = QTextEdit()
        self.changelog_text.setPlainText(self.update_info.changelog)
        self.changelog_text.setReadOnly(True)
        self.changelog_text.setMaximumHeight(200)
        layout.addWidget(self.changelog_text)
        
        # Progress bar (hidden until download starts)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.skip_btn = QPushButton("Skip This Version")
        self.skip_btn.setObjectName("skip")
        self.skip_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.skip_btn)
        
        self.update_btn = QPushButton("Download & Install")
        self.update_btn.clicked.connect(self._start_download)
        btn_layout.addWidget(self.update_btn)
        
        layout.addLayout(btn_layout)
    
    def _start_download(self):
        """Start downloading the update."""
        url = self.update_info.download_url
        
        # If no direct download (just a release page URL), open in browser
        if not (url.endswith(".exe") or url.endswith(".msi") or "/download/" in url):
            QDesktopServices.openUrl(QUrl(url))
            self.accept()
            return
        
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Downloading...")
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        
        # Download to temp directory
        temp_dir = tempfile.gettempdir()
        filename = self.update_info.file_name or "KlippingeTrading_Setup.exe"
        dest = os.path.join(temp_dir, filename)
        
        self.download_thread = DownloadThread(url, dest)
        self.download_thread.progress.connect(self._on_progress)
        self.download_thread.finished.connect(self._on_download_complete)
        self.download_thread.error.connect(self._on_download_error)
        self.download_thread.start()
    
    def _on_progress(self, downloaded: int, total: int):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(downloaded)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self.status_label.setText(f"Downloading: {mb_down:.1f} / {mb_total:.1f} MB")
        else:
            self.progress_bar.setMaximum(0)  # Indeterminate
            mb_down = downloaded / (1024 * 1024)
            self.status_label.setText(f"Downloading: {mb_down:.1f} MB")
    
    def _on_download_complete(self, filepath: str):
        self.status_label.setText("Download complete! Launching installer...")
        
        # Launch the installer and close the app
        try:
            subprocess.Popen([filepath], shell=True)
            # Exit the application so the installer can replace files
            from PyQt5.QtWidgets import QApplication
            QApplication.instance().quit()
        except Exception as e:
            QMessageBox.warning(self, "Error",
                f"Could not launch installer:\n{e}\n\n"
                f"The file was saved to:\n{filepath}")
            self.accept()
    
    def _on_download_error(self, error_msg: str):
        self.update_btn.setEnabled(True)
        self.update_btn.setText("Retry Download")
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff5252;")


# ── Main Updater Class ──────────────────────────────────────────────────────

class AutoUpdater:
    """
    Main updater interface. Use from your main window:
    
        self.updater = AutoUpdater(self)
        self.updater.check_for_updates()  # background check on startup
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        self.parent = parent
        self._checker = None
        
        from app_config import APP_VERSION, GITHUB_REPO
        self.current_version = APP_VERSION
        self.github_repo = GITHUB_REPO
    
    def check_for_updates(self, silent: bool = True):
        """
        Check for updates in background.
        
        Args:
            silent: If True, only show dialog when update is available.
                    If False, also show "you're up to date" message.
        """
        if not REQUESTS_AVAILABLE:
            return
        
        self._silent = silent
        self._checker = UpdateChecker(self.github_repo, self.current_version)
        self._checker.update_available.connect(self._on_update_available)
        
        if not silent:
            self._checker.no_update.connect(self._on_no_update)
            self._checker.error.connect(self._on_error)
        
        self._checker.start()
    
    def _on_update_available(self, info: UpdateInfo):
        dialog = UpdateDialog(info, self.parent)
        dialog.exec_()
    
    def _on_no_update(self):
        if self.parent:
            QMessageBox.information(
                self.parent, "Up to Date",
                f"You're running the latest version (v{self.current_version})."
            )
    
    def _on_error(self, msg: str):
        if self.parent and not self._silent:
            QMessageBox.warning(
                self.parent, "Update Check Failed",
                f"Could not check for updates:\n{msg}"
            )

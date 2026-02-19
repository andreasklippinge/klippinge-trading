"""
Pairs Trading Terminal - PyQt5 Version

Requirements:
    pip install pyqtgraph numpy pandas yfinance PyQtWebEngine beautifulsoup4 requests
"""

import sys
import os
import re
import math
import socket
import time
import copy
import logging
import traceback
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QLineEdit, QTextEdit, QSplitter, QTabWidget,
    QHeaderView, QAbstractItemView, QSpinBox, QDoubleSpinBox,
    QProgressBar, QStatusBar, QMenuBar, QMenu,
    QFrame, QScrollArea, QGridLayout, QSizePolicy, QMessageBox,
    QFileDialog, QCheckBox, QAction, QStackedWidget, QCompleter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QThread, QObject, pyqtSlot as Slot, QSize, QUrl, QPointF, QRectF
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter, QBrush, QPen, QPolygonF, QDesktopServices

import numpy as np
import pandas as pd

# Web scraping imports
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    print("Warning: requests/beautifulsoup4 not installed. Mini futures scraping unavailable.")

# Mini futures price scraping
try:
    from scrape_prices_MS import get_buy_price, get_quotes_batch, ProductQuote
    MF_PRICE_SCRAPING_AVAILABLE = True
except ImportError:
    MF_PRICE_SCRAPING_AVAILABLE = False
    print("Warning: scrape_prices_MS not found. Live mini futures prices unavailable.")

# Portfolio history and benchmark tracking
try:
    from portfolio_history import PortfolioHistoryManager, format_performance_table, BENCHMARK_TICKER
    PORTFOLIO_HISTORY_AVAILABLE = True
except ImportError:
    PORTFOLIO_HISTORY_AVAILABLE = False
    print("Warning: portfolio_history.py not found.")

# Markov chain analysis
try:
    from markov_chain import MarkovChainAnalyzer, MarkovResult, MARKOV_STATES, N_MARKOV_STATES
    MARKOV_AVAILABLE = True
except ImportError:
    MARKOV_AVAILABLE = False
    print("Warning: markov_chain.py not found.")

# Import trading engine
from pairs_engine import PairsTradingEngine, OUProcess, load_tickers_from_csv


# ── Application configuration (portable paths for distribution) ──
from app_config import (
    Paths, APP_VERSION, APP_NAME,
    get_discord_webhook_url, save_discord_webhook_url,
    initialize_user_data, resource_path, get_user_data_dir,
    find_ticker_csv, find_matched_tickers_csv, setup_logging,
    print_config, _is_frozen
)

# ── Screen scaling for different monitor sizes ──
try:
    from screen_scaling import (
        get_scaled_typography,
        get_scale_factor,
        apply_screen_scaling,
        get_layout_recommendations,
        print_screen_info,
        BASE_TYPOGRAPHY
    )
    SCREEN_SCALING_AVAILABLE = True
except ImportError:
    SCREEN_SCALING_AVAILABLE = False
    print("Note: screen_scaling.py not found. Using default typography.")

# ============================================================================
# WEBENGINE MAP CONFIGURATION
# ============================================================================
# Set to False to disable WebEngine/Plotly map (use pyqtgraph fallback instead)
# This can help if PyQtWebEngine causes kernel crashes in Spyder
ENABLE_WEBENGINE_MAP = True

# Import QWebEngineView (only if enabled - can crash kernel in some environments)
WEBENGINE_AVAILABLE = False
QWebEngineView = None
if ENABLE_WEBENGINE_MAP:
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineSettings
        WEBENGINE_AVAILABLE = True
        print("QWebEngineView loaded successfully")
    except ImportError as e:
        print(f"Note: PyQtWebEngine not available: {e}")
        print("Install with: pip install PyQtWebEngine")
    except Exception as e:
        print(f"Warning: QWebEngineView import failed: {e}")
        print("Map will use fallback display.")
else:
    print("WebEngine map disabled by configuration (ENABLE_WEBENGINE_MAP = False)")

# Lazy import for pyqtgraph (moderately heavy - load on demand)
PYQTGRAPH_AVAILABLE = None  # None = not checked yet
pg = None

def get_pyqtgraph():
    """Lazy load pyqtgraph - only import when actually needed."""
    global PYQTGRAPH_AVAILABLE, pg
    if PYQTGRAPH_AVAILABLE is None:
        try:
            import pyqtgraph as _pg
            _pg.setConfigOptions(antialias=True, background='#0a0a0a', foreground='#888888')
            pg = _pg
            PYQTGRAPH_AVAILABLE = True
        except ImportError:
            PYQTGRAPH_AVAILABLE = False
            print("Warning: pyqtgraph not installed. Charts will be limited.")
    return pg

def ensure_pyqtgraph():
    """Ensure pyqtgraph is loaded, return availability status."""
    get_pyqtgraph()
    return PYQTGRAPH_AVAILABLE


# ============================================================================
# CUSTOM PYQTGRAPH COMPONENTS - Date Axis and Crosshair (Lazy loaded)
# ============================================================================

# These classes are created on-demand when pyqtgraph is first needed
_DateAxisItem = None
_CrosshairManager = None

def get_date_axis_item_class():
    """Get DateAxisItem class, creating it if needed (lazy load)."""
    global _DateAxisItem
    if _DateAxisItem is not None:
        return _DateAxisItem
    
    _pg = get_pyqtgraph()
    if _pg is None:
        return None
    
    class DateAxisItem(_pg.AxisItem):
        """Custom axis item that displays dates instead of numeric indices."""
        
        def __init__(self, dates=None, orientation='bottom', *args, **kwargs):
            super().__init__(orientation, *args, **kwargs)
            self.dates = dates
        
        def set_dates(self, dates):
            """Update the dates array."""
            self.dates = dates
        
        def tickStrings(self, values, scale, spacing):
            """Convert numeric values to date strings."""
            if self.dates is None or len(self.dates) == 0:
                return [str(int(v)) for v in values]
            
            strings = []
            for v in values:
                try:
                    idx = int(v)
                    if 0 <= idx < len(self.dates):
                        date = self.dates[idx]
                        if spacing < 5:
                            strings.append(date.strftime('%Y-%m-%d'))
                        elif spacing < 30:
                            strings.append(date.strftime('%b %d'))
                        else:
                            strings.append(date.strftime('%b %y'))
                    else:
                        strings.append('')
                except (ValueError, IndexError):
                    strings.append('')
            return strings
    
    _DateAxisItem = DateAxisItem
    return _DateAxisItem


_EpochAxisItem = None

def get_epoch_axis_item_class():
    """Get EpochAxisItem class that formats epoch seconds as timestamps."""
    global _EpochAxisItem
    if _EpochAxisItem is not None:
        return _EpochAxisItem

    _pg = get_pyqtgraph()
    if _pg is None:
        return None

    class EpochAxisItem(_pg.AxisItem):
        """Axis item that converts epoch seconds to human-readable timestamps."""
        def tickStrings(self, values, scale, spacing):
            strings = []
            for v in values:
                try:
                    dt = datetime.fromtimestamp(float(v))
                    strings.append(dt.strftime('%m-%d %H:%M'))
                except (ValueError, OSError):
                    strings.append('')
            return strings

    _EpochAxisItem = EpochAxisItem
    return _EpochAxisItem


def get_crosshair_manager_class():
    """Get CrosshairManager class, creating it if needed (lazy load)."""
    global _CrosshairManager
    if _CrosshairManager is not None:
        return _CrosshairManager
    
    _pg = get_pyqtgraph()
    if _pg is None:
        return None
    
    class CrosshairManager:
        """Manages crosshair and tooltips for interactive plots with synchronization support."""
        
        def __init__(self, plot_widget, dates=None, data_series=None, label_format="{:.2f}", synced_managers=None):
            self.plot = plot_widget
            self.dates = dates
            self.data_series = data_series
            self.label_format = label_format
            self.synced_managers = synced_managers or []  # List of other CrosshairManagers to sync with
            self._is_syncing = False  # Prevent recursive sync
            
            self.vLine = _pg.InfiniteLine(angle=90, movable=False, pen=_pg.mkPen('#d4a574', width=1, style=Qt.DashLine))
            self.hLine = _pg.InfiniteLine(angle=0, movable=False, pen=_pg.mkPen('#d4a574', width=1, style=Qt.DashLine))
            plot_widget.addItem(self.vLine, ignoreBounds=True)
            plot_widget.addItem(self.hLine, ignoreBounds=True)
            self.vLine.setZValue(1000)
            self.hLine.setZValue(1000)
            
            self.label = _pg.TextItem(
                anchor=(0, 1), 
                fill=_pg.mkBrush(26, 26, 26, 230),
                border=_pg.mkPen('#d4a574', width=1)
            )
            self.label.setFont(QFont('JetBrains Mono', 9))
            self.label.setColor('#e8e8e8')
            plot_widget.addItem(self.label, ignoreBounds=True)
            self.label.setZValue(1001)
            self.label.hide()
            
            self.proxy = _pg.SignalProxy(plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

            # Hide crosshair when mouse leaves the plot widget entirely
            self._event_filter = self._LeaveEventFilter(self)
            plot_widget.installEventFilter(self._event_filter)

        class _LeaveEventFilter(QObject):
            """Hides crosshair when mouse leaves the plot widget."""
            def __init__(self, manager):
                super().__init__()
                self._manager = manager

            def eventFilter(self, obj, event):
                if event.type() == event.Type.Leave:
                    self._manager.hide_crosshair()
                return False

        def add_synced_manager(self, manager):
            """Add another CrosshairManager to sync with."""
            if manager not in self.synced_managers:
                self.synced_managers.append(manager)
        
        def set_data(self, dates=None, data_series=None):
            """Update data for hover display."""
            if dates is not None:
                self.dates = dates
            if data_series is not None:
                self.data_series = data_series
        
        def update_crosshair_position(self, x_pos, mouse_y=None, from_sync=False):
            """Update crosshair position, optionally syncing to other managers."""
            self.vLine.setPos(x_pos)
            self.vLine.show()
            
            # Update label
            x = int(round(x_pos))
            text_lines = []
            data_y_value = None  # Will hold the Y value from data for label positioning
            
            if self.dates is not None and 0 <= x < len(self.dates):
                try:
                    date_str = self.dates[x].strftime('%Y-%m-%d')
                    text_lines.append(date_str)
                except (IndexError, AttributeError, ValueError):
                    pass
        
            if self.data_series:
                y_values = []
                for name, values in self.data_series.items():
                    if 0 <= x < len(values):
                        try:
                            val = values[x]
                            if not np.isnan(val):
                                text_lines.append(f"{name}: {self.label_format.format(val)}")
                                y_values.append(val)
                        except (IndexError, ValueError, TypeError):
                            pass
                # Use average of data values for Y position
                if y_values:
                    data_y_value = sum(y_values) / len(y_values)
            
            # Update horizontal line position
            if from_sync and data_y_value is not None:
                # When synced, position hLine at the data value
                self.hLine.setPos(data_y_value)
                self.hLine.show()
            elif mouse_y is not None:
                self.hLine.setPos(mouse_y)
                self.hLine.show()
        
            if text_lines:
                self.label.setText("\n".join(text_lines))
                
                # Get view range for relative positioning
                view_range = self.plot.getPlotItem().vb.viewRange()
                x_min, x_max = view_range[0]
                y_min, y_max = view_range[1]
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Determine Y position for label:
                # - If from_sync or no mouse_y: use data value
                # - Otherwise use mouse position
                if from_sync or mouse_y is None:
                    if data_y_value is not None:
                        label_y_base = data_y_value
                    else:
                        label_y_base = (y_max + y_min) / 2
                else:
                    label_y_base = mouse_y
                
                # Small offset relative to view range (2% offset)
                offset_x = x_range * 0.02
                offset_y = y_range * 0.04
                
                # Calculate relative position in view (0-1)
                rel_x = (x_pos - x_min) / x_range if x_range > 0 else 0.5
                rel_y = (label_y_base - y_min) / y_range if y_range > 0 else 0.5
                
                # Position label on opposite side if near edge
                if rel_x > 0.75:
                    # Near right edge - put label to the left
                    label_x = x_pos - offset_x
                    anchor_x = 1  # Right-aligned
                else:
                    # Normal - put label to the right
                    label_x = x_pos + offset_x
                    anchor_x = 0  # Left-aligned
                
                if rel_y > 0.80:
                    # Near top - put label below
                    label_y = label_y_base - offset_y
                    anchor_y = 0  # Bottom-aligned
                else:
                    # Normal - put label above
                    label_y = label_y_base + offset_y
                    anchor_y = 1  # Top-aligned
                
                self.label.setAnchor((anchor_x, anchor_y))
                self.label.setPos(label_x, label_y)
                self.label.show()
            else:
                self.label.hide()
        
            # Sync to other managers (don't pass mouse_y - let each calculate its own)
            if not from_sync and not self._is_syncing:
                self._is_syncing = True
                for manager in self.synced_managers:
                    manager.update_crosshair_position(x_pos, mouse_y=None, from_sync=True)
                self._is_syncing = False
        
        def hide_crosshair(self, from_sync=False):
            """Hide crosshair and label."""
            self.label.hide()
            self.vLine.hide()
            self.hLine.hide()
            
            # Sync hide to other managers
            if not from_sync and not self._is_syncing:
                self._is_syncing = True
                for manager in self.synced_managers:
                    manager.hide_crosshair(from_sync=True)
                self._is_syncing = False
        
        def mouse_moved(self, evt):
            pos = evt[0]
            if self.plot.sceneBoundingRect().contains(pos):
                mouse_point = self.plot.getPlotItem().vb.mapSceneToView(pos)
                
                # Update horizontal line to follow mouse Y
                self.hLine.setPos(mouse_point.y())
                self.hLine.show()
                
                # Update crosshair position och ge musens Y
                self.update_crosshair_position(mouse_point.x(), mouse_y=mouse_point.y())
            else:
                self.hide_crosshair()
        
        def cleanup(self):
            """Remove items from plot."""
            try:
                self.plot.removeEventFilter(self._event_filter)
                self.plot.removeItem(self.vLine)
                self.plot.removeItem(self.hLine)
                self.plot.removeItem(self.label)
            except (RuntimeError, AttributeError):
                pass
    
    _CrosshairManager = CrosshairManager
    return _CrosshairManager


# ============================================================================
# SCHEDULED SCAN CONFIGURATION
# ============================================================================

# CSV file to use for automatic daily scans
SCHEDULED_CSV_PATH = Paths.scheduled_csv_path()

# Discord webhook URL - loaded from user config file (NOT hardcoded for security!)
DISCORD_WEBHOOK_URL = get_discord_webhook_url()

# Schedule time (24-hour format)
SCHEDULED_HOUR = 22
SCHEDULED_MINUTE = 15

# Minimum |Z-score| to show in Pair Signals tab dropdown (set to 2.0 for production, 0.5 for testing)
SIGNAL_TAB_THRESHOLD = 2.0  # Change this! 2.0 = standard entry threshold

# Portfolio persistence file path
PORTFOLIO_FILE = Paths.portfolio_file()

# Engine cache file path (syncs price data, viable pairs, etc. between computers)
ENGINE_CACHE_FILE = Paths.engine_cache_file()
# Portfolio history file
PORTFOLIO_HISTORY_FILE = Paths.portfolio_history_file()
# Volatility percentile cache (sorterade historiska serier för live-percentilberäkning)
VOLATILITY_CACHE_FILE = Paths.volatility_cache_file()

# ============================================================================
# GOOGLE DRIVE SYNC - Override paths for multi-computer sync
# ============================================================================
# If Google Drive paths exist, use them instead of local AppData
_GDRIVE_TRADING_DIR = r"G:\Min enhet\Python\Aktieanalys\Python\Trading"

if os.path.isdir(_GDRIVE_TRADING_DIR):
    _gdrive_portfolio = os.path.join(_GDRIVE_TRADING_DIR, "portfolio_positions.json")
    _gdrive_history = os.path.join(_GDRIVE_TRADING_DIR, "portfolio_history.json")
    _gdrive_engine = os.path.join(_GDRIVE_TRADING_DIR, "engine_cache.pkl")
    
    if os.path.exists(_gdrive_portfolio) or not os.path.exists(PORTFOLIO_FILE):
        PORTFOLIO_FILE = _gdrive_portfolio
        print(f"[GDrive Sync] Portfolio: {PORTFOLIO_FILE}")
    
    if os.path.exists(_gdrive_history) or not os.path.exists(PORTFOLIO_HISTORY_FILE):
        PORTFOLIO_HISTORY_FILE = _gdrive_history
        print(f"[GDrive Sync] History: {PORTFOLIO_HISTORY_FILE}")
    
    if os.path.exists(_gdrive_engine) or not os.path.exists(ENGINE_CACHE_FILE):
        ENGINE_CACHE_FILE = _gdrive_engine
        print(f"[GDrive Sync] Engine cache: {ENGINE_CACHE_FILE}")

# ============================================================================
# COLOR PALETTE - Institutional Amber/Gold Theme
# ============================================================================

COLORS = {
    # Primary accent - Amber/Gold (more sophisticated than pure orange)
    'accent': '#d4a574',           # Primary amber
    'accent_bright': '#e8b86d',    # Bright amber for highlights
    'accent_dark': '#b8956a',      # Darker amber
    'accent_glow': 'rgba(212, 165, 116, 0.3)',  # Glow effect
    
    # Background hierarchy
    'bg_darkest': '#050505',       # Deepest background
    'bg_dark': '#0a0a0a',          # Main background
    'bg_card': '#0d0d0d',          # Card background
    'bg_elevated': '#111111',      # Elevated elements
    'bg_hover': '#1a1a1a',         # Hover state
    
    # Text hierarchy
    'text_primary': '#e8e8e8',     # Primary text
    'text_secondary': '#a0a0a0',   # Secondary text
    'text_muted': '#666666',       # Muted text
    'text_disabled': '#444444',    # Disabled text
    
    # Status colors - improved contrast
    'positive': '#22c55e',         # Green for positive values
    'positive_bg': 'rgba(34, 197, 94, 0.15)',
    'negative': '#ef4444',         # Red for negative values
    'negative_bg': 'rgba(239, 68, 68, 0.15)',
    'warning': '#f59e0b',          # Warning/caution
    'info': '#3b82f6',             # Info blue
    'neutral': '#c5ba22',
    
    # Regime colors
    'regime_risk_on': '#22c55e',
    'regime_risk_off': '#f59e0b', 
    'regime_deflation': '#3b82f6',
    'regime_stagflation': '#ef4444',
    
    # Borders
    'border_subtle': '#1a1a1a',
    'border_default': '#2a2a2a',
    'border_strong': '#3a3a3a',
}

# ── Cachade QColor-objekt (undvik att skapa nya per cell-uppdatering) ──
_QCOLOR_TEXT = QColor("#e8e8e8")
_QCOLOR_MUTED = QColor("#666666")
_QCOLOR_POSITIVE = QColor("#22c55e")
_QCOLOR_NEGATIVE = QColor("#ff1744")
_QCOLOR_AMBER = QColor("#ffaa00")

# ── Cachade stylesheet-strängar för spinboxes i portfolio-tabellen ──
_DSPINBOX_STYLESHEET = """
    QDoubleSpinBox {
        background: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #333;
        border-radius: 3px;
        padding: 4px 6px;
        font-size: 13px;
    }
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        width: 18px;
        background: #252525;
        border: none;
    }
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
        background: #333;
    }
"""

_SPINBOX_STYLESHEET = """
    QSpinBox {
        background: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #333;
        border-radius: 3px;
        padding: 4px 6px;
        font-size: 13px;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 18px;
        background: #252525;
        border: none;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background: #333;
    }
"""

# ============================================================================
# TYPOGRAPHY - Dynamically Scaled Based on Screen Size
# ============================================================================

# Base typography (fallback if screen_scaling not available)
_BASE_TYPOGRAPHY = {
    # Headers
    'header_large': 20,        # Main title (KLIPPINGE INVESTMENT)
    'header_section': 13,      # Section headers (ORNSTEIN-UHLENBECK DETAILS, etc.)
    'header_sub': 12,          # Sub-headers within sections
    
    # Body text
    'body_large': 13,          # Primary body text
    'body_medium': 12,         # Standard text
    'body_small': 11,          # Secondary/muted text
    
    # Data display - INCREASED for better readability
    'metric_value': 22,        # Large metric values (Z-score, prices)
    'metric_label': 12,        # Metric labels (MEAN REVERSION, HALF-LIFE, etc.)
    'table_header': 12,        # Table column headers (was 10)
    'table_cell': 13,          # Table cell content (was 11)
    
    # UI elements
    'button': 12,              # Button text
    'input': 12,               # Input field text
    'tab': 13,                 # Tab labels
    'status': 11,              # Status indicators, timestamps
    'clock_time': 18,          # World clock time
    'clock_city': 12,          # World clock city names
}

# Use scaled typography if available, otherwise use base
if SCREEN_SCALING_AVAILABLE:
    try:
        TYPOGRAPHY = get_scaled_typography()
        print(f"[Typography] Using scaled typography (factor: {get_scale_factor():.2f})")
    except Exception as e:
        print(f"[Typography] Scaling failed, using defaults: {e}")
        TYPOGRAPHY = _BASE_TYPOGRAPHY.copy()
else:
    TYPOGRAPHY = _BASE_TYPOGRAPHY.copy()

# ============================================================================
# STYLESHEET - Professional Institutional Design
# ============================================================================

STYLESHEET = f"""
/* ===== BASE STYLES ===== */
QMainWindow, QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: {TYPOGRAPHY['body_large']}px;
}}

/* ===== TYPOGRAPHY ===== */
/* Monospace for numbers - critical for alignment */
QLabel[class="mono"], QTableWidget {{
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace;
}}

/* Headers hierarchy */
QLabel#headerLarge {{
    font-size: {TYPOGRAPHY['header_large']}px;
    font-weight: 700;
    letter-spacing: 2px;
    color: {COLORS['accent']};
}}

QLabel#sectionHeader {{
    color: {COLORS['accent']};
    font-size: {TYPOGRAPHY['header_section']}px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 8px 0;
}}

QLabel#subHeader {{
    color: {COLORS['text_secondary']};
    font-size: {TYPOGRAPHY['header_sub']}px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ===== TAB WIDGET ===== */
QTabWidget::pane {{
    border: none;
    background-color: {COLORS['bg_dark']};
}}

QTabBar::tab {{
    background-color: transparent;
    color: {COLORS['text_muted']};
    padding: 12px 20px;
    min-width: 300px;
    border: none;
    font-size: {TYPOGRAPHY['tab']}px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QTabBar::tab:selected {{
    color: {COLORS['accent']};
    border-bottom: 1px solid {COLORS['accent']};
}}

QTabBar::tab:hover:!selected {{
    color: {COLORS['accent_bright']};
}}

/* ===== CARDS WITH GRADIENT ===== */
QFrame#metricCard {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLORS['bg_elevated']}, 
        stop:1 {COLORS['bg_card']});
    border: 1px solid {COLORS['border_subtle']};
    border-radius: 6px;
    padding: 12px;
}}

QFrame#metricCard:hover {{
    border-color: {COLORS['accent_dark']};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #151515, 
        stop:1 {COLORS['bg_elevated']});
}}

/* ===== TABLES ===== */
QTableWidget {{
    background-color: {COLORS['bg_dark']};
    gridline-color: {COLORS['border_subtle']};
    border: none;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: {TYPOGRAPHY['table_cell']}px;
}}

QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border_subtle']};
}}

QTableWidget::item:selected {{
    background-color: rgba(212, 165, 116, 0.15);
    color: {COLORS['accent_bright']};
}}

QTableWidget::item:hover {{
    background-color: {COLORS['bg_hover']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_muted']};
    padding: 10px;
    border: none;
    border-bottom: 1px solid {COLORS['border_default']};
    font-weight: 600;
    text-transform: uppercase;
    font-size: {TYPOGRAPHY['table_header']}px;
    letter-spacing: 0.5px;
}}

/* ===== BUTTONS ===== */
QPushButton {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_default']};
    color: {COLORS['text_primary']};
    padding: 10px 16px;
    font-weight: 600;
    border-radius: 4px;
}}

QPushButton:hover {{
    background-color: {COLORS['bg_hover']};
    border-color: {COLORS['accent']};
}}

QPushButton:pressed {{
    background-color: {COLORS['bg_card']};
}}

QPushButton#primaryButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {COLORS['accent_bright']}, 
        stop:1 {COLORS['accent']});
    border: none;
    color: #000000;
    font-weight: 700;
}}

QPushButton#primaryButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f0c080, 
        stop:1 {COLORS['accent_bright']});
}}

QPushButton#dangerButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f87171, 
        stop:1 {COLORS['negative']});
    border: none;
    color: #ffffff;
    font-weight: 700;
}}

/* ===== INPUT FIELDS ===== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_default']};
    color: {COLORS['text_primary']};
    padding: 8px 12px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border-color: {COLORS['accent']};
    background-color: {COLORS['bg_hover']};
}}

QComboBox::drop-down {{
    border: none;
    width: 30px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLORS['text_muted']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_default']};
    selection-background-color: rgba(212, 165, 116, 0.2);
    selection-color: {COLORS['accent_bright']};
}}

/* ===== SCROLLBAR ===== */
QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_default']};
    min-height: 30px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['accent_dark']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_dark']};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['border_default']};
    min-width: 30px;
    border-radius: 5px;
}}

/* ===== PROGRESS BAR ===== */
QProgressBar {{
    background-color: {COLORS['bg_hover']};
    border: none;
    height: 6px;
    border-radius: 3px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, 
        stop:1 {COLORS['accent_bright']});
    border-radius: 3px;
}}

/* ===== MENU BAR ===== */
QMenuBar {{
    background-color: {COLORS['bg_darkest']};
    border-bottom: 1px solid {COLORS['border_subtle']};
    padding: 2px;
}}

QMenuBar::item {{
    padding: 6px 12px;
    color: {COLORS['text_secondary']};
    border-radius: 4px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['bg_hover']};
    color: {COLORS['accent']};
}}

QMenu {{
    background-color: {COLORS['bg_elevated']};
    border: 1px solid {COLORS['border_default']};
    border-radius: 4px;
    padding: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: rgba(212, 165, 116, 0.15);
    color: {COLORS['accent']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {COLORS['border_subtle']};
    margin: 4px 8px;
}}

/* ===== TOOLTIPS ===== */
QToolTip {{
    background-color: #1a1a2e;
    color: #e8e8e8;
    border: 1px solid {COLORS['accent']};
    padding: 10px 12px;
    border-radius: 5px;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}}

/* ===== STATUS BAR ===== */
QStatusBar {{
    background-color: {COLORS['bg_darkest']};
    border-top: 1px solid {COLORS['border_subtle']};
    color: {COLORS['text_muted']};
    font-size: 10px;
}}

QStatusBar::item {{
    border: none;
}}

/* ===== CHECKBOX ===== */
QCheckBox {{
    spacing: 8px;
    color: {COLORS['text_primary']};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border_default']};
    border-radius: 3px;
    background-color: {COLORS['bg_elevated']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['accent']};
}}
"""


# ============================================================================
# PORTFOLIO PERSISTENCE (med fillåsning för Google Drive-synk)
# ============================================================================

def _get_lock_filepath(filepath: str) -> str:
    """Returnera sökväg till låsfilen."""
    return filepath + ".lock"

def _acquire_lock(filepath: str, timeout: float = 10.0) -> bool:
    """
    Försök skaffa fillås. Returnerar True om lyckat.
    Låsfilen innehåller datornamn och tid för debugging.
    """
    lock_path = _get_lock_filepath(filepath)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Kolla om låsfilen finns och är gammal (>60 sek = stale lock)
            if os.path.exists(lock_path):
                lock_age = time.time() - os.path.getmtime(lock_path)
                if lock_age > 60:
                    # Gammal låsfil, ta bort den
                    print(f"[Portfolio] Removing stale lock ({lock_age:.0f}s old)")
                    os.remove(lock_path)
                else:
                    # Någon annan har låset, vänta
                    time.sleep(0.2)
                    continue
            
            # Skapa låsfil
            with open(lock_path, 'w') as f:
                f.write(f"{socket.gethostname()}\n{datetime.now().isoformat()}")
            return True
            
        except Exception as e:
            print(f"[Portfolio] Lock error: {e}")
            time.sleep(0.2)
    
    print(f"[Portfolio] Could not acquire lock within {timeout}s")
    return False

def _release_lock(filepath: str):
    """Släpp fillåset."""
    lock_path = _get_lock_filepath(filepath)
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        print(f"[Portfolio] Error releasing lock: {e}")

def save_portfolio(portfolio: list, filepath: str = PORTFOLIO_FILE,
                   trade_history: list = None) -> bool:
    """
    Save portfolio positions and trade history to JSON file with file locking.
    Safe for Google Drive sync between multiple computers.
    """
    if not _acquire_lock(filepath):
        print("[Portfolio] Could not save - file is locked")
        return False
    
    try:
        # Preserve existing trade_history if not provided
        existing_history = []
        if trade_history is None:
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    existing_history = existing_data.get("trade_history", [])
            except Exception:
                pass
        
        data = {
            "positions": portfolio,
            "trade_history": trade_history if trade_history is not None else existing_history,
            "last_updated": datetime.now().isoformat(),
            "last_saved_by": socket.gethostname(),
            "version": "1.2"
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Fix #9: Explicit JSON serializer for better type handling
        def json_serializer(obj):
            """Custom JSON serializer for objects not serializable by default."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'isoformat'):  # pd.Timestamp
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        
        # Skriv till temporär fil först, sedan rename (atomisk operation)
        temp_path = filepath + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        
        # Ersätt originalfilen
        if os.path.exists(filepath):
            os.replace(temp_path, filepath)
        else:
            os.rename(temp_path, filepath)
        
        # print(f"[Portfolio] Saved {len(portfolio)} position(s) to {filepath}")
        return True
        
    except Exception as e:
        print(f"[Portfolio] Error saving: {e}")
        return False
    finally:
        _release_lock(filepath)


def load_portfolio(filepath: str = PORTFOLIO_FILE) -> list:
    """
    Load portfolio positions from JSON file.
    Called on application startup.
    
    Args:
        filepath: Path to save file
        
    Returns:
        List of position dictionaries (empty list if file doesn't exist)
    """
    try:
        if not os.path.exists(filepath):
            print(f"[Portfolio] No saved portfolio found at {filepath}")
            return []
        
        # Vänta kort om filen nyligen ändrades (Google Drive sync)
        file_age = time.time() - os.path.getmtime(filepath)
        if file_age < 2:
            print(f"[Portfolio] File recently modified, waiting for sync...")
            time.sleep(2)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        positions = data.get("positions", [])
        last_updated = data.get("last_updated", "unknown")
        saved_by = data.get("last_saved_by", "unknown")
        
        print(f"[Portfolio] Loaded {len(positions)} position(s) from {filepath}")
        print(f"[Portfolio] Last updated: {last_updated} by {saved_by}")
        
        return positions
        
    except json.JSONDecodeError as e:
        print(f"[Portfolio] JSON error (file may be syncing): {e}")
        # Försök igen efter kort väntan
        time.sleep(1)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("positions", [])
        except (json.JSONDecodeError, OSError, KeyError):
            return []
        
    except Exception as e:
        print(f"[Portfolio] Error loading: {e}")
        return []


def load_trade_history(filepath: str = PORTFOLIO_FILE) -> list:
    """Load trade history from portfolio JSON file."""
    try:
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("trade_history", [])
    except Exception:
        return []


# ============================================================================
# ENGINE CACHE PERSISTENCE (syncs price data, viable pairs between computers)
# ============================================================================

import pickle

def save_engine_cache(engine, filepath: str = ENGINE_CACHE_FILE) -> bool:
    """
    Save engine state (price data, viable pairs, etc.) to pickle file.
    Called after each scan completes for sync between computers.
    """
    if engine is None:
        print("[Engine Cache] No engine to save")
        return False
    
    if not _acquire_lock(filepath):
        print("[Engine Cache] Could not save - file is locked")
        return False
    
    try:
        # Hämta tickers från price_data kolumner
        tickers = list(engine.price_data.columns) if engine.price_data is not None else []
        
        # Extrahera relevant data från engine
        cache_data = {
            'price_data': engine.price_data,
            'raw_price_data': getattr(engine, 'raw_price_data', None),
            'viable_pairs': engine.viable_pairs,
            'pairs_stats': engine.pairs_stats,
            'tickers': tickers,
            # NEW - store params as dicts, not OUProcess objects
            'ou_models': {
                k: {'theta': v.theta, 'mu': v.mu, 'sigma': v.sigma}
                for k, v in getattr(engine, 'ou_models', {}).items()
                if hasattr(v, 'theta')
            },            
            'config': getattr(engine, 'config', {}),
            'scan_timestamp': datetime.now().isoformat(),
            'scanned_by': socket.gethostname(),
            'version': '2.0'
        }
        
        # Spara spread data för varje viable pair
        spread_cache = {}
        if engine.viable_pairs is not None and len(engine.viable_pairs) > 0:
            # Fix #5: Use itertuples instead of iterrows
            for row in engine.viable_pairs.itertuples():
                pair = row.pair
                try:
                    spread_cache[pair] = {
                        'hedge_ratio': getattr(row, 'hedge_ratio', 1.0),
                        'half_life': getattr(row, 'half_life', 20),
                        'current_z': getattr(row, 'current_z', 0),
                    }
                except (AttributeError, TypeError):
                    pass
        cache_data['spread_cache'] = spread_cache

        # Save window_details for per-window dropdown analysis
        cache_data['window_details'] = getattr(engine, '_window_details', {})

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Atomisk skrivning med pickle
        temp_path = filepath + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if os.path.exists(filepath):
            os.replace(temp_path, filepath)
        else:
            os.rename(temp_path, filepath)
        
        n_pairs = len(engine.viable_pairs) if engine.viable_pairs is not None and len(engine.viable_pairs) > 0 else 0
        n_tickers = len(tickers)
        file_size = os.path.getsize(filepath) / 1024 / 1024  # MB
        
        print(f"[Engine Cache] Saved {n_tickers} tickers, {n_pairs} viable pairs ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"[Engine Cache] Error saving: {e}")
        traceback.print_exc()
        return False
    finally:
        _release_lock(filepath)


def load_engine_cache(filepath: str = ENGINE_CACHE_FILE) -> dict:
    """
    Load engine cache from pickle file.
    Returns dict with price_data, viable_pairs, etc.
    """
    try:
        if not os.path.exists(filepath):
            print(f"[Engine Cache] No cache found at {filepath}")
            return None
        
        # Vänta kort om filen nyligen ändrades (Google Drive sync)
        file_age = time.time() - os.path.getmtime(filepath)
        if file_age < 2:
            print(f"[Engine Cache] File recently modified, waiting for sync...")
            time.sleep(2)
        
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Reconstruct OUProcess objects from saved params
        from pairs_engine import OUProcess
        if 'ou_models' in cache_data and cache_data['ou_models']:
            reconstructed = {}
            for k, v in cache_data['ou_models'].items():
                if isinstance(v, dict) and 'theta' in v:
                    # New format: dict with params
                    reconstructed[k] = OUProcess(v['theta'], v['mu'], v['sigma'])
                elif hasattr(v, 'theta'):
                    # Old format: already an OUProcess object
                    reconstructed[k] = v
            cache_data['ou_models'] = reconstructed
        
        scan_time = cache_data.get('scan_timestamp', 'unknown')
        scanned_by = cache_data.get('scanned_by', 'unknown')
        n_pairs = len(cache_data.get('viable_pairs', [])) if cache_data.get('viable_pairs') is not None else 0
        n_tickers = len(cache_data.get('price_data', {}).columns) if cache_data.get('price_data') is not None else 0
        
        print(f"[Engine Cache] Loaded {n_tickers} tickers, {n_pairs} viable pairs")
        print(f"[Engine Cache] Scanned at: {scan_time} by {scanned_by}")
        
        return cache_data
        
    except Exception as e:
        print(f"[Engine Cache] Error loading: {e}")
        traceback.print_exc()
        return None

def save_volatility_cache(hist_cache: dict, median_cache: dict, mode_cache: dict,
                          sparkline_cache: dict = None,
                          filepath: str = VOLATILITY_CACHE_FILE) -> bool:
    """Spara volatilitets-percentildata till disk (körs efter yf.download).

    Sparar sorterade historiska serier + median/mode så att live-percentil
    kan beräknas direkt vid uppstart utan yf.download.
    """
    try:
        cache_data = {
            'hist': {k: v.tolist() for k, v in hist_cache.items()},  # numpy → list
            'median': median_cache,
            'mode': mode_cache,
            'sparkline': sparkline_cache or {},
            'saved_at': datetime.now().isoformat(),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[VolCache] Saved percentile cache for {list(hist_cache.keys())}")
        return True
    except Exception as e:
        print(f"[VolCache] Save error: {e}")
        return False


def load_volatility_cache(filepath: str = VOLATILITY_CACHE_FILE) -> dict:
    """Ladda volatilitets-percentildata från disk.

    Returns dict med 'hist', 'median', 'mode', 'sparkline', 'saved_at' eller None.
    """
    try:
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        saved = cache_data.get('saved_at', '')
        print(f"[VolCache] Loaded percentile cache (saved: {saved})")
        return cache_data
    except Exception as e:
        print(f"[VolCache] Load error: {e}")
        return None


# ============================================================================
# TTL CACHE UTILITY (Fix #6 - Proper cache with TTL)
# ============================================================================

class TTLCache:
    """Simple TTL cache to replace global variables with proper expiration."""
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache = {}
        self._timestamps = {}
        self._ttl = ttl_seconds
    
    def get(self, key: str):
        """Get value if not expired, otherwise return None."""
        if key not in self._cache:
            return None
        if time.time() - self._timestamps.get(key, 0) > self._ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value):
        """Set value with current timestamp."""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()


# ============================================================================
# MINI FUTURES SCRAPING (Morgan Stanley)
# ============================================================================

# Cache for mini futures data (Fix #6 - using TTLCache)
_minifutures_cache = TTLCache(ttl_seconds=300)  # 5 min TTL
_ticker_mapping_cache = None

def load_ticker_mapping(csv_path: str = None, force_reload: bool = False) -> tuple:
    """Load the mapping between yfinance tickers and MS underlying asset names.
    
    Returns:
        tuple: (ticker_to_ms, ms_to_ticker, ticker_to_ms_asset)
        - ticker_to_ms: yfinance ticker -> MS display name (e.g. 'ATT.ST' -> 'ATTENDO')
        - ms_to_ticker: MS display name -> yfinance ticker
        - ticker_to_ms_asset: yfinance ticker -> MS asset code (e.g. 'ATT.ST' -> 'ATT_SS_Equity')
    """
    global _ticker_mapping_cache
    
    if _ticker_mapping_cache is not None and not force_reload:
        # Check if cache has MS_Asset data
        if len(_ticker_mapping_cache) >= 3 and _ticker_mapping_cache[2]:
            return _ticker_mapping_cache
        else:
            print("[load_ticker_mapping] Cache exists but no MS_Asset data, reloading...")
    
    try:
        # Try various paths - PRIORITIZE files with MS_Asset column!
        # Google Drive paths FIRST for multi-computer sync
        gdrive_matched = r"G:\Min enhet\Python\Aktieanalys\Python\Trading\underliggande_matchade_tickers.csv"
        gdrive_index = r"G:\Min enhet\Python\Aktieanalys\Python\Trading\index_tickers.csv"
        
        paths_to_try = [
            csv_path,
            # Google Drive FIRST (for syncing between computers)
            gdrive_matched if os.path.exists(gdrive_matched) else None,
            gdrive_index if os.path.exists(gdrive_index) else None,
            # Check underliggande_matchade_tickers.csv (has MS_Asset)
            find_matched_tickers_csv(),
            resource_path('underliggande_matchade_tickers.csv'),
            # Fallback to index_tickers.csv (no MS_Asset) LAST
            find_ticker_csv(),
            resource_path('index_tickers.csv'),
        ]
        
        best_result = None
        
        for path in paths_to_try:
            if path and os.path.exists(path):
                print(f"[load_ticker_mapping] Checking: {path}")
                df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
                print(f"[load_ticker_mapping] Columns found: {list(df.columns)}")
                
                # Handle different column name formats
                ticker_to_ms = {}
                ms_to_ticker = {}
                if 'Ticker' in df.columns and 'Name' in df.columns:
                    ticker_to_ms = dict(zip(df['Ticker'], df['Name']))
                    ms_to_ticker = dict(zip(df['Name'], df['Ticker']))
                elif 'Ticker' in df.columns and 'Underliggande tillgång' in df.columns:
                    ticker_to_ms = dict(zip(df['Ticker'], df['Underliggande tillgång']))
                    ms_to_ticker = dict(zip(df['Underliggande tillgång'], df['Ticker']))
                else:
                    print(f"[load_ticker_mapping] Required columns not found, skipping {path}")
                    continue
                
                # Load MS_Asset mapping if available
                ticker_to_ms_asset = {}
                if 'MS_Asset' in df.columns:
                    ticker_to_ms_asset = {
                        k: v for k, v in zip(df['Ticker'], df['MS_Asset']) 
                        if pd.notna(v) and v != ''
                    }
                    if ticker_to_ms_asset:
                        print(f"[load_ticker_mapping] ✓ Loaded {len(ticker_to_ms_asset)} MS_Asset mappings from {path}")
                        examples = list(ticker_to_ms_asset.items())[:3]
                        print(f"[load_ticker_mapping] Examples: {examples}")
                        # Found file with MS_Asset - use it!
                        _ticker_mapping_cache = (ticker_to_ms, ms_to_ticker, ticker_to_ms_asset)
                        return _ticker_mapping_cache
                else:
                    print(f"[load_ticker_mapping] No MS_Asset column in {path}")
                
                # Save as fallback if no MS_Asset file found yet
                if best_result is None:
                    best_result = (ticker_to_ms, ms_to_ticker, ticker_to_ms_asset)
        
        # Use best result if no file with MS_Asset was found
        if best_result:
            print(f"[load_ticker_mapping] WARNING: No file with MS_Asset found, using fallback without certificate support")
            _ticker_mapping_cache = best_result
            return _ticker_mapping_cache
                
    except Exception as e:
        print(f"Could not load ticker mapping: {e}")
        traceback.print_exc()
    
    return {}, {}, {}


def fetch_ms_page(session, page: int) -> list:
    """Fetch a single page of Morgan Stanley mini futures."""
    if not SCRAPING_AVAILABLE:
        return []
    
    BASE_URL = "https://etp.morganstanley.com/se/sv/produkter"
    params = {
        "f_pc": "LeverageProducts",
        "f_pt": "MiniFuture",
        "p_s": 100,
        "s_c": "c_u",
        "s_o": "asc",
        "p_n": page
    }
    
    try:
        r = session.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return []
        
        trs = table.find_all("tr")
        headers = [th.get_text(strip=True) for th in trs[0].find_all("th")]
        
        rows = []
        for tr in trs[1:]:
            tds = []
            for td in tr.find_all("td"):
                tds.append(td.get_text(strip=True))

            if len(tds) == len(headers):
                rows.append(dict(zip(headers, tds)))

        return rows
    except Exception as e:
        return []


def _scrape_ms_table_page(url: str, params: dict, session) -> tuple:
    """Scrape en sida från Morgan Stanleys produkttabell.

    Returns:
        (headers, rows, total_count) — headers är list[str], rows list[dict],
        total_count totalt antal träffar (eller None om det inte går att läsa).
    """
    r = session.get(url, params=params, timeout=10)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return [], [], None

    trs = table.find_all("tr")
    if len(trs) < 2:
        return [], [], None

    headers = [th.get_text(strip=True) for th in trs[0].find_all("th")]

    rows = []
    for tr in trs[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) == len(headers):
            rows.append(dict(zip(headers, tds)))

    # Försök läsa totalantal från paginering (t.ex. "Visar 1-100 av 237")
    total_count = None
    pag = soup.find(string=re.compile(r'av\s+\d+'))
    if pag:
        m = re.search(r'av\s+(\d+)', pag)
        if m:
            total_count = int(m.group(1))

    return headers, rows, total_count


def _fetch_ms_all_pages(product_type: str, ms_asset: str, session=None,
                         page_size: int = 100) -> pd.DataFrame:
    """Hämta ALLA sidor av en produkttyp från Morgan Stanley.

    Args:
        product_type: 'MiniFuture' eller 'ConstantLeverage'
        ms_asset: MS asset code
        session: requests session
        page_size: antal per sida (max 100)
    """
    if not SCRAPING_AVAILABLE or not ms_asset:
        return pd.DataFrame()

    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

    BASE_URL = "https://etp.morganstanley.com/se/sv/produkter"
    all_rows = []
    page = 1

    while True:
        params = {
            "f_pc": "LeverageProducts",
            "f_pt": product_type,
            "f_asset": ms_asset,
            "p_s": page_size,
            "p_n": page
        }
        try:
            headers, rows, total_count = _scrape_ms_table_page(BASE_URL, params, session)
        except Exception as e:
            print(f"Error fetching {product_type} page {page} for {ms_asset}: {e}")
            break

        if not rows:
            break

        all_rows.extend(rows)

        # Kolla om det finns fler sidor
        if total_count is not None and len(all_rows) >= total_count:
            break
        if len(rows) < page_size:
            break  # Sista sidan

        page += 1
        if page > 10:  # Säkerhetsgräns
            break

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def fetch_minifutures_for_asset(ms_asset: str, session=None) -> pd.DataFrame:
    """
    Fetch ALL mini futures for a specific underlying asset (alla sidor).
    """
    df = _fetch_ms_all_pages('MiniFuture', ms_asset, session)
    if df.empty:
        return df

    def parse_number(x):
        if not isinstance(x, str):
            return None
        clean = re.sub(r"[^\d,\.]", "", x)
        clean = clean.replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return None

    if 'Finansieringsnivå' in df.columns:
        df['FinansieringsnivåNum'] = df['Finansieringsnivå'].apply(parse_number)

    return df


def fetch_certificates_for_asset(ms_asset: str, session=None) -> pd.DataFrame:
    """Fetch ALL Bull/Bear certificates for a specific underlying (alla sidor)."""
    df = _fetch_ms_all_pages('ConstantLeverage', ms_asset, session)
    if df.empty:
        return df

    def parse_leverage(x):
        if not isinstance(x, str):
            return None
        clean = x.replace('x', '').replace('X', '').strip()
        try:
            return float(clean)
        except ValueError:
            return None

    leverage_cols = ['Daglig hävstång', 'Hävstång', 'Leverage']
    for col in leverage_cols:
        if col in df.columns:
            df['DailyLeverage'] = df[col].apply(parse_leverage)
            break

    def parse_number(x):
        if not isinstance(x, str):
            return None
        try:
            clean = re.sub(r'[^\d,.\-]', '', x)
            clean = clean.replace(',', '.')
            return float(clean) if clean else None
        except ValueError:
            return None

    fin_cols = ['Finansieringsnivå', 'FinansieringsnivåNum', 'Financing Level']
    for col in fin_cols:
        if col in df.columns:
            df['FinansieringsnivåNum'] = df[col].apply(parse_number) if col != 'FinansieringsnivåNum' else df[col]
            break

    mult_cols = ['Multiplikator', 'Multiplier']
    for col in mult_cols:
        if col in df.columns:
            df['MultiplikatorNum'] = df[col].apply(parse_number)
            break

    ratio_cols = ['Ratio', 'Kvot']
    for col in ratio_cols:
        if col in df.columns:
            df['RatioNum'] = df[col].apply(parse_number)
            if 'MultiplikatorNum' not in df.columns:
                df['MultiplikatorNum'] = 1.0 / df['RatioNum'].replace(0, float('nan'))
            break

    return df


def find_best_certificate(ticker: str, direction: str, ticker_to_ms: dict,
                          ticker_to_ms_asset: dict = None, 
                          target_leverage: float = 2.0) -> dict:
    """
    Find the best Bull/Bear certificate for a given ticker and direction.
    
    Args:
        ticker: yfinance ticker symbol
        direction: 'Long' or 'Short'
        ticker_to_ms: mapping from yfinance ticker to MS underlying name
        ticker_to_ms_asset: mapping from yfinance ticker to MS asset code
        target_leverage: Preferred leverage (default 2.0, will find closest)
        
    Returns:
        dict with certificate info or None if not found
    """
    # Get the MS underlying name
    ms_name = ticker_to_ms.get(ticker)
    if not ms_name:
        base_ticker = ticker.split('.')[0]
        ms_name = ticker_to_ms.get(base_ticker)
    
    if not ms_name:
        return None
    
    # Get MS asset code for direct fetch
    ms_asset = None
    if ticker_to_ms_asset:
        ms_asset = ticker_to_ms_asset.get(ticker)
        if not ms_asset:
            base_ticker = ticker.split('.')[0]
            ms_asset = ticker_to_ms_asset.get(base_ticker)
    
    if not ms_asset:
        return None
    
    # Fetch certificates for this asset
    df_certs = fetch_certificates_for_asset(ms_asset)
    
    if df_certs.empty:
        return None
    
    # VIKTIGT: Verifiera att de hämtade certifikaten faktiskt är för rätt underliggande!
    # Morgan Stanley kan returnera "populära produkter" om tillgången inte finns.
    if 'Underliggande tillgång' in df_certs.columns:
        # Filtrera på underliggande som matchar ms_name (case-insensitive)
        mask = df_certs['Underliggande tillgång'].str.lower().str.contains(ms_name.lower(), na=False, regex=False)
        df_certs = df_certs[mask]
        
        if df_certs.empty:
            print(f"[Certificate] No certificates found matching underlying '{ms_name}' for {ticker}")
            return None
    
    # Filter by direction
    # For Bull/Bear: "Long" = Bull (positive leverage), "Short" = Bear (negative leverage)
    if 'Riktning' in df_certs.columns:
        mask = df_certs['Riktning'].str.contains(direction, case=False, na=False)
        df_filtered = df_certs[mask].copy()
    elif 'DailyLeverage' in df_certs.columns:
        # Fallback: Use leverage sign to determine direction
        if direction.lower() == 'long':
            df_filtered = df_certs[df_certs['DailyLeverage'] > 0].copy()
        else:
            df_filtered = df_certs[df_certs['DailyLeverage'] < 0].copy()
    else:
        return None
    
    if df_filtered.empty:
        return None
    
    # Must have leverage info
    if 'DailyLeverage' not in df_filtered.columns:
        return None
    
    # Filter out invalid leverage
    df_filtered = df_filtered[df_filtered['DailyLeverage'].notna()]
    
    if df_filtered.empty:
        return None
    
    # Find certificate closest to target leverage
    # For shorts, leverage is negative, so we compare absolute values
    df_filtered['LeverageAbs'] = df_filtered['DailyLeverage'].abs()
    df_filtered['LeverageDiff'] = (df_filtered['LeverageAbs'] - abs(target_leverage)).abs()
    
    # Sort by difference to target, then by absolute leverage (prefer lower)
    best = df_filtered.sort_values(['LeverageDiff', 'LeverageAbs']).iloc[0]
    
    # Extract ISIN and create Avanza link
    isin_url = best.get('ISIN', '')
    isin = extract_isin_from_url(isin_url)
    avanza_link = create_avanza_link(isin)
    
    # Get spot price for reference
    spot_price = get_spot_price(ticker)
    
    # Get financing level and multiplier if available
    financing_level = best.get('FinansieringsnivåNum', None)
    multiplier = best.get('MultiplikatorNum', None)
    ratio = best.get('RatioNum', None)
    
    # Calculate multiplier from ratio if not directly available
    if multiplier is None and ratio is not None and ratio > 0:
        multiplier = 1.0 / ratio
    
    # Calculate theoretical instrument price if we have the data
    instrument_price = None
    if financing_level is not None and spot_price is not None and multiplier is not None:
        if direction.lower() == 'long':
            # BULL: pris = multiplier × (spot - financing_level)
            instrument_price = multiplier * (spot_price - financing_level)
        else:
            # BEAR: pris = multiplier × (financing_level - spot)
            instrument_price = multiplier * (financing_level - spot_price)
        instrument_price = max(0.01, abs(instrument_price))  # Ensure positive
    
    return {
        'name': best.get('Namn', 'N/A'),
        'underlying': best.get('Underliggande tillgång', ms_name),
        'direction': direction,
        'financing_level': financing_level,
        'leverage': best['LeverageAbs'],  # Use absolute value for display
        'daily_leverage': best['DailyLeverage'],  # Keep original with sign
        'spot_price': spot_price,
        'multiplier': multiplier,
        'ratio': ratio,
        'instrument_price': instrument_price,
        'isin': isin or 'N/A',
        'avanza_link': avanza_link,
        'ticker': ticker,
        'product_type': 'Certificate'  # Mark as certificate, not mini future
    }


def fetch_all_instruments_for_ticker(ticker: str, direction: str, ticker_to_ms: dict,
                                      ticker_to_ms_asset: dict = None) -> list:
    """
    Hämta ALLA tillgängliga instrument (mini futures + certifikat) för en ticker och riktning.

    Returnerar en lista av dicts sorterade efter hävstång (lägst först).
    Varje dict har samma format som find_best_minifuture() returnerar.
    """
    instruments = []

    # Slå upp MS-namn och asset-kod
    ms_name = ticker_to_ms.get(ticker)
    if not ms_name:
        base_ticker = ticker.split('.')[0]
        ms_name = ticker_to_ms.get(base_ticker)

    ms_asset = None
    if ticker_to_ms_asset:
        ms_asset = ticker_to_ms_asset.get(ticker)
        if not ms_asset:
            base_ticker = ticker.split('.')[0]
            ms_asset = ticker_to_ms_asset.get(base_ticker)

    if not ms_asset:
        return instruments

    spot_price = get_spot_price(ticker)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # ── Mini Futures ──
    df_mf = fetch_minifutures_for_asset(ms_asset, session)
    if not df_mf.empty and ms_name:
        # Verifiera underliggande
        underlying_col = None
        for col in df_mf.columns:
            if 'underliggande' in col.lower():
                underlying_col = col
                break
        if underlying_col:
            mask = df_mf[underlying_col].str.lower().str.contains(ms_name.lower(), na=False, regex=False)
            df_mf = df_mf[mask]

        # Filtrera riktning
        riktning_col = None
        for col in df_mf.columns:
            if 'riktning' in col.lower():
                riktning_col = col
                break
        if riktning_col and not df_mf.empty:
            df_mf = df_mf[df_mf[riktning_col].str.contains(direction, case=False, na=False)]

        # Beräkna hävstång för varje mini future
        if not df_mf.empty and spot_price is not None:
            fin_col = None
            for col in df_mf.columns:
                if 'finansieringsnivånum' in col.lower() or col == 'FinansieringsnivåNum':
                    fin_col = col
                    break

            if fin_col:
                for _, row in df_mf.iterrows():
                    fin_level = row.get(fin_col)
                    if fin_level is None or pd.isna(fin_level):
                        continue

                    # Filtrera ogiltiga finansieringsnivåer
                    if direction.lower() == 'long' and fin_level >= spot_price:
                        continue
                    if direction.lower() == 'short' and fin_level <= spot_price:
                        continue

                    # Beräkna teoretisk hävstång
                    if direction.lower() == 'long':
                        leverage = spot_price / (spot_price - fin_level)
                    else:
                        leverage = spot_price / (fin_level - spot_price)

                    if leverage <= 1:
                        continue

                    # Hitta ISIN
                    isin_col = None
                    for col in df_mf.columns:
                        if 'isin' in col.lower():
                            isin_col = col
                            break
                    name_col = None
                    for col in df_mf.columns:
                        if col.lower() == 'namn' or 'name' in col.lower():
                            name_col = col
                            break
                    ul_col = underlying_col

                    isin_url = row.get(isin_col, '') if isin_col else ''
                    isin = extract_isin_from_url(isin_url)
                    avanza_link = create_avanza_link(isin)

                    instruments.append({
                        'name': row.get(name_col, 'N/A') if name_col else 'N/A',
                        'underlying': row.get(ul_col, ms_name) if ul_col else ms_name,
                        'direction': direction,
                        'financing_level': fin_level,
                        'leverage': leverage,
                        'spot_price': spot_price,
                        'isin': isin or 'N/A',
                        'avanza_link': avanza_link,
                        'ticker': ticker,
                        'product_type': 'Mini Future'
                    })

    # ── Certifikat ──
    df_cert = fetch_certificates_for_asset(ms_asset, session)
    if not df_cert.empty and ms_name:
        # Verifiera underliggande
        if 'Underliggande tillgång' in df_cert.columns:
            mask = df_cert['Underliggande tillgång'].str.lower().str.contains(ms_name.lower(), na=False, regex=False)
            df_cert = df_cert[mask]

        # Filtrera riktning
        if not df_cert.empty:
            if 'Riktning' in df_cert.columns:
                df_cert = df_cert[df_cert['Riktning'].str.contains(direction, case=False, na=False)]
            elif 'DailyLeverage' in df_cert.columns:
                if direction.lower() == 'long':
                    df_cert = df_cert[df_cert['DailyLeverage'] > 0]
                else:
                    df_cert = df_cert[df_cert['DailyLeverage'] < 0]

        if not df_cert.empty and 'DailyLeverage' in df_cert.columns:
            df_cert = df_cert[df_cert['DailyLeverage'].notna()]

            for _, row in df_cert.iterrows():
                leverage_abs = abs(row['DailyLeverage'])

                # Beräkna instrumentpris om möjligt
                instrument_price = None
                financing_level = row.get('FinansieringsnivåNum', None)
                multiplier = row.get('MultiplikatorNum', None)
                ratio = row.get('RatioNum', None)
                if multiplier is None and ratio is not None and ratio > 0:
                    multiplier = 1.0 / ratio

                if financing_level is not None and spot_price is not None and multiplier is not None:
                    if direction.lower() == 'long':
                        instrument_price = multiplier * (spot_price - financing_level)
                    else:
                        instrument_price = multiplier * (financing_level - spot_price)
                    instrument_price = max(0.01, abs(instrument_price))

                isin_url = row.get('ISIN', '')
                isin = extract_isin_from_url(isin_url)
                avanza_link = create_avanza_link(isin)

                name_col = None
                for col in df_cert.columns:
                    if col.lower() == 'namn' or 'name' in col.lower():
                        name_col = col
                        break

                instruments.append({
                    'name': row.get(name_col, 'N/A') if name_col else row.get('Namn', 'N/A'),
                    'underlying': row.get('Underliggande tillgång', ms_name),
                    'direction': direction,
                    'financing_level': financing_level,
                    'leverage': leverage_abs,
                    'daily_leverage': row['DailyLeverage'],
                    'spot_price': spot_price,
                    'multiplier': multiplier,
                    'ratio': ratio,
                    'instrument_price': instrument_price,
                    'isin': isin or 'N/A',
                    'avanza_link': avanza_link,
                    'ticker': ticker,
                    'product_type': 'Certificate'
                })

    # Sortera efter hävstång (lägst först)
    instruments.sort(key=lambda x: x['leverage'])

    return instruments


def scrape_ms_minifutures() -> pd.DataFrame:
    """Scrape all Morgan Stanley mini futures with parallel requests."""
    global _minifutures_cache
    
    # Return cached data if still valid (Fix #6 - using TTLCache)
    cached = _minifutures_cache.get('minifutures')
    if cached is not None:
        return cached
    
    if not SCRAPING_AVAILABLE:
        return pd.DataFrame()
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    rows_all = []
    pages = range(1, 35)  # Usually ~30-34 pages
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_ms_page, session, p) for p in pages]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                rows_all.extend(result)
    
    if not rows_all:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows_all)
    
    # Parse financing level
    def parse_number(x):
        if not isinstance(x, str):
            return None
        clean = re.sub(r"[^\d,\.]", "", x)
        clean = clean.replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return None
    
    if 'Finansieringsnivå' in df.columns:
        df['FinansieringsnivåNum'] = df['Finansieringsnivå'].apply(parse_number)
    
    # Cache the result (Fix #6)
    _minifutures_cache.set('minifutures', df)
    
    return df


def extract_isin_from_url(url: str) -> str:
    """Extract ISIN from Morgan Stanley product URL."""
    if not isinstance(url, str):
        return None
    isin = url.upper().split("/")[-1]
    if re.fullmatch(r"[A-Z]{2}[A-Z0-9]{10}", isin):
        return isin
    return None


def create_avanza_link(isin: str) -> str:
    """Create Avanza search URL from ISIN."""
    if not isin:
        return None
    return f"https://www.avanza.se/sok.html?query=&q={isin}&p=0"


def display_ticker(ticker: str) -> str:
    """Strip ^, =X, =F from ticker for display (e.g. '^OMX' → 'OMX')."""
    return ticker.lstrip('^').replace('=X', '').replace('=F', '')


def get_spot_price(ticker: str) -> float:
    """Get current spot price for a ticker."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def find_best_minifuture(ticker: str, direction: str, ticker_to_ms: dict, 
                         minifutures_df: pd.DataFrame = None, 
                         ticker_to_ms_asset: dict = None,
                         fallback_to_certificates: bool = True) -> dict:
    """
    Find the best mini future for a given ticker and direction.
    Falls back to Bull/Bear certificates if no mini futures are found.
    
    OPTIMIZED: Uses direct URL fetching via MS_Asset code when available,
    which is ~10x faster than filtering the full minifutures DataFrame.
    
    Args:
        ticker: yfinance ticker symbol
        direction: 'Long' or 'Short'
        ticker_to_ms: mapping from yfinance ticker to MS underlying name
        minifutures_df: DataFrame of all MS mini futures (fallback, can be None)
        ticker_to_ms_asset: mapping from yfinance ticker to MS asset code (e.g. 'ATT_SS_Equity')
        fallback_to_certificates: If True, try Bull/Bear certificates when no mini futures found
        
    Returns:
        dict with mini future/certificate info or None if not found
    """
    
    # Get the MS underlying name
    ms_name = ticker_to_ms.get(ticker)
    if not ms_name:
        # Try without exchange suffix
        base_ticker = ticker.split('.')[0]
        ms_name = ticker_to_ms.get(base_ticker)
    
    if not ms_name:
        # Still try certificates if we have MS asset code
        if fallback_to_certificates and ticker_to_ms_asset:
            ms_asset = ticker_to_ms_asset.get(ticker)
            if not ms_asset:
                base_ticker = ticker.split('.')[0]
                ms_asset = ticker_to_ms_asset.get(base_ticker)
            if ms_asset:
                certificate = find_best_certificate(
                    ticker, direction, ticker_to_ms, ticker_to_ms_asset, target_leverage=2.0
                )
                if certificate:
                    return certificate
        return None
    
    
    # Get MS asset code
    ms_asset = None
    if ticker_to_ms_asset:
        ms_asset = ticker_to_ms_asset.get(ticker)
        if not ms_asset:
            base_ticker = ticker.split('.')[0]
            ms_asset = ticker_to_ms_asset.get(base_ticker)
        if not ms_asset:
            # Debug: Show similar keys
            similar = [k for k in list(ticker_to_ms_asset.keys())[:10]]
    else:
        if not ms_asset:
            base_ticker = ticker.split('.')[0]
            ms_asset = ticker_to_ms_asset.get(base_ticker)
    
    # Get spot price
    spot_price = get_spot_price(ticker)
    if spot_price is None:
        # Still try certificates
        if fallback_to_certificates and ticker_to_ms_asset and ms_asset:
            certificate = find_best_certificate(
                ticker, direction, ticker_to_ms, ticker_to_ms_asset, target_leverage=2.0
            )
            if certificate:
                return certificate
        return None
    
    
    # OPTIMIZATION: Try direct fetch via MS_Asset first (10x faster!)
    df_filtered = None
    if ms_asset:
        # Fetch directly for this specific underlying - only 1 request!
        df_asset = fetch_minifutures_for_asset(ms_asset)
        
        if not df_asset.empty:
            # VIKTIGT: Verifiera att hämtade minifutures är för rätt underliggande!
            # Morgan Stanley kan returnera "populära produkter" om tillgången inte finns.
            underlying_col = None
            for col in df_asset.columns:
                if 'underliggande' in col.lower():
                    underlying_col = col
                    break
            
            if underlying_col:
                # Filtrera på underliggande som matchar ms_name
                underlying_mask = df_asset[underlying_col].str.lower().str.contains(ms_name.lower(), na=False, regex=False)
                df_asset = df_asset[underlying_mask]
                
                if df_asset.empty:
                    print(f"[MiniFuture] No products found matching underlying '{ms_name}' for {ticker}")
            
            if not df_asset.empty:
                # Find Riktning column with flexible matching
                riktning_col = None
                for col in df_asset.columns:
                    if 'riktning' in col.lower():
                        riktning_col = col
                        break
                
                if riktning_col:
                    mask = df_asset[riktning_col].str.contains(direction, case=False, na=False)
                    df_filtered = df_asset[mask].copy()
    
    # Fallback: Use the full DataFrame if direct fetch didn't work
    if df_filtered is None or df_filtered.empty:
        if minifutures_df is not None and not minifutures_df.empty:
            # Find columns with flexible matching
            underlying_col = None
            riktning_col = None
            for col in minifutures_df.columns:
                if 'underliggande' in col.lower():
                    underlying_col = col
                if 'riktning' in col.lower():
                    riktning_col = col
            
            if underlying_col and riktning_col:
                mask = (
                    minifutures_df[underlying_col].str.contains(ms_name, case=False, na=False, regex=False) &
                    minifutures_df[riktning_col].str.contains(direction, case=False, na=False)
                )
                df_filtered = minifutures_df[mask].copy()
    
    # Process mini futures if we found any
    if df_filtered is not None and not df_filtered.empty:
        
        # Find FinansieringsnivåNum column
        fin_col = None
        for col in df_filtered.columns:
            if 'finansieringsnivånum' in col.lower() or col == 'FinansieringsnivåNum':
                fin_col = col
                break
        
        if fin_col:
            if direction.lower() == 'long':
                df_filtered = df_filtered[df_filtered[fin_col] < spot_price]
            else:
                df_filtered = df_filtered[df_filtered[fin_col] > spot_price]
            
            
            if not df_filtered.empty:
                # Calculate theoretical leverage
                if direction.lower() == 'long':
                    df_filtered['TheoreticalLeverage'] = spot_price / (spot_price - df_filtered[fin_col])
                else:
                    df_filtered['TheoreticalLeverage'] = spot_price / (df_filtered[fin_col] - spot_price)
                
                # Filter valid leverage (> 1)
                df_filtered = df_filtered[df_filtered['TheoreticalLeverage'] > 1]
                
                if not df_filtered.empty:
                    # Select lowest leverage (most conservative)
                    best = df_filtered.sort_values('TheoreticalLeverage').iloc[0]
                    
                    # Find ISIN column
                    isin_col = None
                    for col in df_filtered.columns:
                        if 'isin' in col.lower():
                            isin_col = col
                            break
                    
                    # Find name column
                    name_col = None
                    for col in df_filtered.columns:
                        if col.lower() == 'namn' or 'name' in col.lower():
                            name_col = col
                            break
                    
                    # Find underlying column
                    underlying_col = None
                    for col in df_filtered.columns:
                        if 'underliggande' in col.lower():
                            underlying_col = col
                            break
                    
                    isin_url = best.get(isin_col, '') if isin_col else ''
                    isin = extract_isin_from_url(isin_url)
                    avanza_link = create_avanza_link(isin)
                    
                    result = {
                        'name': best.get(name_col, 'N/A') if name_col else 'N/A',
                        'underlying': best.get(underlying_col, ms_name) if underlying_col else ms_name,
                        'direction': direction,
                        'financing_level': best[fin_col],
                        'leverage': best['TheoreticalLeverage'],
                        'spot_price': spot_price,
                        'isin': isin or 'N/A',
                        'avanza_link': avanza_link,
                        'ticker': ticker,
                        'product_type': 'Mini Future'
                    }
                    return result
    
    # FALLBACK: Try Bull/Bear certificates if no mini futures found
    
    if fallback_to_certificates and ticker_to_ms_asset is not None:
        certificate = find_best_certificate(
            ticker, direction, ticker_to_ms, ticker_to_ms_asset, target_leverage=2.0
        )
        if certificate:
            return certificate
    
    return None


def calculate_minifuture_position(notional: float, hedge_ratio: float, mf_y: dict, mf_x: dict, direction: str) -> dict:
    """
    Calculate mini future position sizes for a pairs trade, respecting minimum
    allocation constraints while maintaining the hedge ratio.
    
    The hedge ratio β from OLS regression (Y = α + βX + ε) represents a SHARES 
    relationship: N_x = N_y * β. Converting to notional exposure:
    
        exposure_x = N_x * price_x = N_y * β * price_x
        exposure_y = N_y * price_y
        
        Therefore: exposure_x = β * (price_x / price_y) * exposure_y
        
        We call this the "notional_ratio" = β * price_x / price_y
    
    Constraints:
    - capital_y >= 1000 SEK
    - capital_x >= 1000 SEK
    - exposure_x = notional_ratio * exposure_y (proper hedge must be maintained)
    
    We find the minimum exposure_y that satisfies both capital constraints,
    then scale up if notional allows.
    
    Args:
        notional: Total notional value in SEK
        hedge_ratio: The beta/hedge ratio from the pair (OLS coefficient)
        mf_y: Mini future info for leg Y (dict with 'leverage' and 'spot_price')
        mf_x: Mini future info for leg X (dict with 'leverage' and 'spot_price')
        direction: 'LONG' or 'SHORT' (spread direction)
        
    Returns:
        dict with position sizing info
    """
    MIN_ALLOCATION = 1000  # Minimum 1000 SEK per leg
    
    leverage_y = mf_y['leverage'] if mf_y else 1.0
    leverage_x = mf_x['leverage'] if mf_x else 1.0
    beta = abs(hedge_ratio)
    
    # Get spot prices to convert share-based hedge ratio to notional ratio
    price_y = mf_y['spot_price'] if mf_y and 'spot_price' in mf_y else 1.0
    price_x = mf_x['spot_price'] if mf_x and 'spot_price' in mf_x else 1.0
    
    # Calculate the notional ratio: this is the correct multiplier for exposures
    # exposure_x = notional_ratio * exposure_y
    # This ensures: shares_x = shares_y * beta (proper hedge)
    if price_y > 0:
        notional_ratio = beta * price_x / price_y
    else:
        notional_ratio = beta  # Fallback if prices unavailable
    
    # Calculate minimum exposure_y that satisfies both minimum capital constraints:
    # Constraint 1: capital_y >= 1000 → exposure_y >= 1000 * leverage_y
    # Constraint 2: capital_x >= 1000 → (notional_ratio * exposure_y) / leverage_x >= 1000 
    #               → exposure_y >= 1000 * leverage_x / notional_ratio
    
    min_exposure_y_from_y = MIN_ALLOCATION * leverage_y
    if notional_ratio > 0:
        min_exposure_y_from_x = MIN_ALLOCATION * leverage_x / notional_ratio
    else:
        min_exposure_y_from_x = MIN_ALLOCATION * leverage_x
    
    # The binding constraint is whichever requires higher exposure_y
    min_exposure_y = max(min_exposure_y_from_y, min_exposure_y_from_x)
    min_exposure_x = notional_ratio * min_exposure_y
    
    # Calculate minimum capital requirements
    min_capital_y = min_exposure_y / leverage_y
    min_capital_x = min_exposure_x / leverage_x
    min_total_capital = min_capital_y + min_capital_x
    
    # Determine which leg was the binding constraint
    binding_leg = 'Y' if min_exposure_y_from_y >= min_exposure_y_from_x else 'X'
    
    # If notional is higher than minimum, scale up proportionally
    if notional > min_total_capital:
        scale = notional / min_total_capital
        exposure_y = min_exposure_y * scale
        exposure_x = min_exposure_x * scale
        capital_y = min_capital_y * scale
        capital_x = min_capital_x * scale
    else:
        # Use minimum (notional is too low to scale up)
        exposure_y = min_exposure_y
        exposure_x = min_exposure_x
        capital_y = min_capital_y
        capital_x = min_capital_x
    
    return {
        'capital_y': capital_y,
        'capital_x': capital_x,
        'total_capital': capital_y + capital_x,
        'exposure_y': exposure_y,
        'exposure_x': exposure_x,
        'total_exposure': exposure_y + exposure_x,
        'leverage_y': leverage_y,
        'leverage_x': leverage_x,
        'effective_leverage': (exposure_y + exposure_x) / (capital_y + capital_x) if (capital_y + capital_x) > 0 else 1.0,
        'min_total_capital': min_total_capital,
        'binding_leg': binding_leg,
        'is_at_minimum': notional <= min_total_capital,
        'notional_ratio': notional_ratio,  # Added for transparency
        'beta': beta,
        'price_ratio': price_x / price_y if price_y > 0 else 1.0
    }


def calculate_minifuture_minimum_units(hedge_ratio: float, mf_y: dict, mf_x: dict,
                                       dir_y: str, dir_x: str) -> Optional[dict]:
    """
    Beräkna antal instrumentenheter per ben med korrekt EXPONERING baserat på
    hedge ratio, hävstång och instrumentpris.

    Hedge ratio β gäller underliggande aktier: shares_x = β × shares_y.
    Eftersom varje instrumentenhet ger exponering = pris × hävstång,
    måste vi matcha exponeringar, inte enhetsantal direkt:

        units_x × price_x × lev_x = β × units_y × price_y × lev_y

    Returns dict med units, priser, kapital, exponering per ben, eller None.
    """
    MIN_CAPITAL_PER_LEG = 1000  # SEK

    def _instrument_price(mf_data: dict, direction: str) -> float:
        """Calculate instrument price for a mini future / certificate."""
        product_type = mf_data.get('product_type', 'Mini Future')
        spot_price = mf_data.get('spot_price', 0)
        leverage = mf_data.get('leverage', 1)
        fin_level = mf_data.get('financing_level')
        price = mf_data.get('instrument_price')

        if price is None:
            if product_type == 'Mini Future' and fin_level is not None:
                if direction == 'Long':
                    price = spot_price - fin_level
                else:
                    price = fin_level - spot_price
            elif product_type == 'Certificate':
                multiplier = mf_data.get('multiplier')
                if fin_level is not None and multiplier is not None:
                    if direction == 'Long':
                        price = multiplier * (spot_price - fin_level)
                    else:
                        price = multiplier * (fin_level - spot_price)
                else:
                    price = spot_price / leverage if leverage > 0 else 100
            else:
                price = spot_price / leverage if leverage > 0 else 100

        return max(0.01, abs(price))

    if not mf_y or not mf_x:
        return None

    price_y = _instrument_price(mf_y, dir_y)
    price_x = _instrument_price(mf_x, dir_x)
    leverage_y = mf_y.get('leverage', 1)
    leverage_x = mf_x.get('leverage', 1)
    beta = abs(hedge_ratio)

    # Exponering per instrumentenhet
    exp_per_unit_y = price_y * leverage_y
    exp_per_unit_x = price_x * leverage_x

    # Minimum enheter per ben från kapitalvillkor (1000 SEK)
    min_units_y = max(1, math.ceil(MIN_CAPITAL_PER_LEG / price_y))
    min_units_x = max(1, math.ceil(MIN_CAPITAL_PER_LEG / price_x))

    # Alternativ A: Utgå från min X-enheter, beräkna Y från hedge ratio
    units_x_a = min_units_x
    # units_x * exp_x = β * units_y * exp_y  →  units_y = units_x * exp_x / (β * exp_y)
    units_y_a = max(min_units_y, math.ceil(
        units_x_a * exp_per_unit_x / (beta * exp_per_unit_y)
    )) if beta > 0 else min_units_y
    total_a = units_y_a * price_y + units_x_a * price_x

    # Alternativ B: Utgå från min Y-enheter, beräkna X, sedan justera Y tillbaka
    units_y_b = min_units_y
    units_x_b = max(min_units_x, math.ceil(
        beta * units_y_b * exp_per_unit_y / exp_per_unit_x
    ))
    # Efter avrundning av X uppåt — beräkna om Y för att bibehålla hedge ratio
    units_y_b = max(min_units_y, math.ceil(
        units_x_b * exp_per_unit_x / (beta * exp_per_unit_y)
    )) if beta > 0 else units_y_b
    total_b = units_y_b * price_y + units_x_b * price_x

    # Välj kombinationen med lägst totalt kapital
    if total_a <= total_b:
        units_y, units_x = units_y_a, units_x_a
    else:
        units_y, units_x = units_y_b, units_x_b

    capital_y = units_y * price_y
    capital_x = units_x * price_x
    exposure_y = units_y * exp_per_unit_y
    exposure_x = units_x * exp_per_unit_x

    return {
        'units_y': units_y,
        'units_x': units_x,
        'price_y': price_y,
        'price_x': price_x,
        'leverage_y': leverage_y,
        'leverage_x': leverage_x,
        'capital_y': capital_y,
        'capital_x': capital_x,
        'total_capital': capital_y + capital_x,
        'exposure_y': exposure_y,
        'exposure_x': exposure_x,
        'total_exposure': exposure_y + exposure_x,
    }


def calculate_optimal_stop_loss(ou, entry_z_abs: float, current_z: float) -> dict:
    """
    Calculate optimal stop-loss level.
    TP is always 0 (mean reversion strategy).
    
    Returns dict with optimal stop_z and metrics.
    """
    stop_z_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    exit_z = 0.0  # TP is always mean (0)
    all_points = []
    
    for stop_z in stop_z_values:
        if current_z > 0:
            # Short position
            S0 = ou.spread_from_z(entry_z_abs)
            tp = ou.spread_from_z(exit_z)
            sl = ou.spread_from_z(stop_z)
        else:
            # Long position
            S0 = ou.spread_from_z(-entry_z_abs)
            tp = ou.spread_from_z(-exit_z)
            sl = ou.spread_from_z(-stop_z)
        
        result = ou.expected_pnl(S0, tp, sl)
        
        exp_pnl_zscore = (result['expected_pnl'] / ou.eq_std) * 100 if ou.eq_std > 0 else 0
        
        all_points.append({
            'stop_z': stop_z,
            'rr': result['risk_reward'],
            'exp_pnl_pct': exp_pnl_zscore,
            'win_prob': result['win_prob'],
            'kelly': result['kelly_fraction']
        })
    
    # Find optimal SL (best balanced score)
    rr_values = [p['rr'] for p in all_points]
    exp_values = [p['exp_pnl_pct'] for p in all_points]
    
    rr_min, rr_max = min(rr_values), max(rr_values)
    exp_min, exp_max = min(exp_values), max(exp_values)
    
    if rr_max > rr_min and exp_max > exp_min:
        for p in all_points:
            rr_norm = (p['rr'] - rr_min) / (rr_max - rr_min)
            exp_norm = (p['exp_pnl_pct'] - exp_min) / (exp_max - exp_min)
            p['score'] = rr_norm * exp_norm
        best = max(all_points, key=lambda x: x['score'])
    else:
        best = max(all_points, key=lambda x: x['win_prob'])
    
    return {
        'stop_z': best['stop_z'],
        'exit_z': 0.0,
        'rr': best['rr'],
        'exp_pnl_pct': best['exp_pnl_pct'],
        'win_prob': best['win_prob'],
        'kelly': best['kelly']
    }


# ============================================================================
# WORKER THREADS
# ============================================================================

class SyncWorker(QObject):
    """Worker thread for sync operations (Fix #2 - async file I/O)."""
    finished = Signal()
    portfolio_changed = Signal(list)
    engine_changed = Signal(dict)
    status_message = Signal(str)
    
    def __init__(self, portfolio_file: str, engine_cache_file: str, 
                 portfolio_mtime: float, engine_mtime: float):
        super().__init__()
        self.portfolio_file = portfolio_file
        self.engine_cache_file = engine_cache_file
        self.portfolio_mtime = portfolio_mtime
        self.engine_mtime = engine_mtime
    
    @Slot()
    def run(self):
        try:
            if os.path.exists(self.portfolio_file):
                current_mtime = os.path.getmtime(self.portfolio_file)
                if current_mtime > self.portfolio_mtime + 1:
                    new_positions = load_portfolio(self.portfolio_file)
                    if new_positions is not None:
                        self.portfolio_changed.emit(new_positions)
                        self.status_message.emit(f"Portfolio synced: {len(new_positions)} position(s)")
            
            if os.path.exists(self.engine_cache_file):
                current_mtime = os.path.getmtime(self.engine_cache_file)
                if current_mtime > self.engine_mtime + 1:
                    cache_data = load_engine_cache(self.engine_cache_file)
                    if cache_data and cache_data.get('price_data') is not None:
                        self.engine_changed.emit(cache_data)
        except Exception as e:
            pass
        finally:
            self.finished.emit()


class StartupWorker(QObject):
    """Worker thread for startup data loading (prevents GUI freeze on launch)."""
    finished = Signal()
    portfolio_loaded = Signal(list)
    engine_loaded = Signal(dict)
    status_message = Signal(str)

    def __init__(self, portfolio_file: str, engine_cache_file: str):
        super().__init__()
        self.portfolio_file = portfolio_file
        self.engine_cache_file = engine_cache_file

    @Slot()
    def run(self):
        try:
            # 1. Load portfolio (fast - small JSON)
            self.status_message.emit("Loading portfolio...")
            if os.path.exists(self.portfolio_file):
                positions = load_portfolio(self.portfolio_file)
                if positions:
                    self.portfolio_loaded.emit(positions)

            # 2. Load engine cache (slowest - large pickle with price data)
            self.status_message.emit("Loading market data cache...")
            if os.path.exists(self.engine_cache_file):
                cache_data = load_engine_cache(self.engine_cache_file)
                if cache_data and cache_data.get('price_data') is not None:
                    self.engine_loaded.emit(cache_data)
            
            self.status_message.emit("Startup complete")
            
        except Exception as e:
            self.status_message.emit(f"Startup error: {str(e)[:50]}")
        finally:
            self.finished.emit()


class AnalysisWorker(QObject):
    """Worker thread for running pair analysis."""
    finished = Signal()
    progress = Signal(int, str)
    error = Signal(str)
    result = Signal(object)
    
    def __init__(self, tickers: List[str], lookback: str, config: Dict):
        super().__init__()
        self.tickers = tickers
        self.lookback = lookback
        self.config = config
        self._is_cancelled = False
    
    @Slot()
    def run(self):
        try:
            self.progress.emit(5, "Initializing engine...")
            engine = PairsTradingEngine(self.config)
            
            self.progress.emit(15, f"Fetching data for {len(self.tickers)} tickers...")
            engine.fetch_data(self.tickers, 'max')
            
            if engine.price_data is None or len(engine.price_data.columns) == 0:
                self.error.emit("No price data loaded")
                return
            
            loaded = len(engine.price_data.columns)
            self.progress.emit(40, f"Loaded {loaded} tickers, screening pairs...")
            
            engine.screen_pairs(correlation_prefilter=True)
            
            self.progress.emit(100, "Complete!")
            self.result.emit(engine)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()



class DiscordWorker(QObject):
    """Worker thread for Discord notifications (async HTTP to prevent GUI freeze)."""
    finished = Signal()
    success = Signal(str)
    error = Signal(str)
    
    def __init__(self, webhook_url: str, payload: dict):
        super().__init__()
        self.webhook_url = webhook_url
        self.payload = payload
    
    @Slot()
    def run(self):
        try:
            response = requests.post(
                self.webhook_url,
                json=self.payload,
                headers={"Content-Type": "application/json"},
                timeout=10  # Timeout to prevent infinite hang
            )
            if response.status_code == 204:
                self.success.emit("Discord notification sent successfully")
            else:
                self.error.emit(f"Discord notification failed: {response.status_code}")
        except requests.exceptions.Timeout:
            self.error.emit("Discord notification timed out")
        except Exception as e:
            self.error.emit(f"Discord notification error: {e}")
        finally:
            self.finished.emit()


class PriceFetchWorker(QObject):
    """Worker thread for fetching mini futures prices (async HTTP to prevent GUI freeze)."""
    finished = Signal()
    result = Signal(dict)  # isin -> ProductQuote
    error = Signal(str)
    status_message = Signal(str)
    
    def __init__(self, isins: list):
        super().__init__()
        self.isins = isins
    
    @Slot()
    def run(self):
        try:
            self.status_message.emit(f"Fetching {len(self.isins)} mini futures prices...")
            quotes = get_quotes_batch(self.isins)
            self.result.emit(quotes)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class MarketWatchWorker(QObject):
    """Worker thread for fetching market data via yf.download().

    Downloads 5 days of daily data for all instruments, computes daily
    change percentage from the last two trading days' close prices,
    and emits a list of dicts for the treemap heatmap.
    """
    finished = Signal()
    result = Signal(list)  # List[dict] with {market, symbol, name, price, change, change_pct}
    error = Signal(str)
    status_message = Signal(str)

    def __init__(self, instruments: dict):
        super().__init__()
        self.instruments = dict(instruments)
        self.tickers = list(instruments.keys())

    @Slot()
    def run(self):
        """Fetch market data using yf.download and compute change %."""
        try:
            import yfinance as yf

            self.status_message.emit(f"Fetching market data for {len(self.tickers)} instruments...")
            print(f"[MarketWatch] Starting yf.download for {len(self.tickers)} instruments...")

            try:
                data = yf.download(self.tickers, period='5d', interval="15m", progress=False, threads=True, ignore_tz=True)
            except (TypeError, RuntimeError) as e:
                print(f"[MarketWatch] First attempt failed ({e}), retrying in batches...")
                all_data = []
                batch_size = 100
                for i in range(0, len(self.tickers), batch_size):
                    batch = self.tickers[i:i+batch_size]
                    try:
                        batch_data = yf.download(batch, period='5d', interval="15m", progress=False, threads=True, ignore_tz=True)
                        if not batch_data.empty:
                            all_data.append(batch_data)
                    except Exception as batch_e:
                        print(f"[MarketWatch] Batch {i//batch_size + 1} failed: {batch_e}")
                        continue
                if all_data:
                    data = pd.concat(all_data, axis=1)
                else:
                    data = pd.DataFrame()

            if data.empty:
                self.error.emit("No market data returned")
                return

            # Extract OHLC prices
            is_multi = isinstance(data.columns, pd.MultiIndex)
            if is_multi:
                close = data['Close']
                open_p = data['Open'] if 'Open' in data.columns.get_level_values(0) else None
                high_p = data['High'] if 'High' in data.columns.get_level_values(0) else None
                low_p = data['Low'] if 'Low' in data.columns.get_level_values(0) else None
            elif 'Close' in data.columns:
                close = data[['Close']]
                close.columns = [self.tickers[0]] if len(self.tickers) == 1 else close.columns
                open_p = high_p = low_p = None
            else:
                close = data
                open_p = high_p = low_p = None

            # Build items list using pct_change() for daily change
            all_items = []
            for ticker, (name, region) in self.instruments.items():
                try:
                    if ticker not in close.columns:
                        continue
                    col = close[ticker].dropna()
                    if len(col) < 1:
                        continue

                    last_price = float(col.iloc[-1])
                    if len(col) >= 2:
                        pct = col.pct_change().iloc[-1]
                        pct_val = float(pct * 100) if pd.notna(pct) else 0.0
                        # Guard against inf values from zero-division
                        change_pct = round(pct_val, 2) if math.isfinite(pct_val) else 0.0
                        change = round(last_price - float(col.iloc[-2]), 4)
                    else:
                        change = 0.0
                        change_pct = 0.0

                    # Build close-only history (keep datetime for sparklines)
                    history = []
                    for dt_idx, val in col.items():
                        history.append((dt_idx.strftime('%m/%d %H:%M'), float(val)))

                    # Build OHLC history for candlestick charts (15m intervals)
                    ohlc_history = []
                    if open_p is not None and ticker in open_p.columns:
                        try:
                            ohlc_df = pd.DataFrame({
                                'O': open_p[ticker], 'H': high_p[ticker],
                                'L': low_p[ticker], 'C': close[ticker]
                            }).dropna()
                            # Resample to 1h candles to reduce data size
                            ohlc_1h = ohlc_df.resample('1h').agg(
                                {'O': 'first', 'H': 'max', 'L': 'min', 'C': 'last'}
                            ).dropna()
                            for dt_idx, row in ohlc_1h.iterrows():
                                ohlc_history.append([
                                    dt_idx.strftime('%m/%d %H:%M'),
                                    round(float(row['O']), 2),
                                    round(float(row['H']), 2),
                                    round(float(row['L']), 2),
                                    round(float(row['C']), 2),
                                ])
                        except Exception:
                            pass

                    all_items.append({
                        'market': region,
                        'symbol': ticker,
                        'name': name,
                        'price': last_price,
                        'change': change,
                        'change_pct': change_pct,
                        'history': history,
                        'ohlc_history': ohlc_history,
                    })
                except Exception as e:
                    print(f"[MarketWatch] Error processing {ticker}: {e}")
                    continue

            if not all_items:
                self.error.emit("No market data could be processed")
                return

            print(f"[MarketWatch] Processed {len(all_items)} instruments successfully")
            self.result.emit(all_items)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class IntradayOHLCWorker(QObject):
    """Background worker: fetch today's intraday 5-min OHLC for all instruments.

    Seeds the overlay candlestick chart with data from market open (or
    yesterday's session for closed markets).  Runs once at startup.
    """
    finished = Signal()
    result = Signal(dict)   # {symbol: {'ohlc': [[ts,O,H,L,C], ...], 'info': {day_high,day_low,...}}}
    error = Signal(str)

    def __init__(self, tickers: list, instruments: dict):
        super().__init__()
        self.tickers = list(tickers)
        self.instruments = dict(instruments)

    @Slot()
    def run(self):
        try:
            import yfinance as yf
            print(f"[Intraday] Fetching 5d/15m OHLC for {len(self.tickers)} tickers...")

            # period='5d' ger data även för stängda marknader (senaste handelsdagar)
            data = yf.download(
                self.tickers, period='5d', interval='15m',
                progress=False, threads=True, ignore_tz=True,
            )
            if data.empty:
                self.error.emit("No intraday data returned")
                return

            is_multi = isinstance(data.columns, pd.MultiIndex)

            # Extract metric DataFrames
            if is_multi:
                close_df = data['Close']
                open_df = data['Open'] if 'Open' in data.columns.get_level_values(0) else None
                high_df = data['High'] if 'High' in data.columns.get_level_values(0) else None
                low_df = data['Low'] if 'Low' in data.columns.get_level_values(0) else None
                vol_df = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None
            else:
                close_df = open_df = high_df = low_df = vol_df = None

            out = {}
            for ticker in self.tickers:
                try:
                    if is_multi:
                        if ticker not in close_df.columns:
                            continue
                        col_c = close_df[ticker].dropna()
                        if col_c.empty:
                            continue
                        odf = pd.DataFrame({'C': col_c})
                        if open_df is not None and ticker in open_df.columns:
                            odf['O'] = open_df[ticker]
                        if high_df is not None and ticker in high_df.columns:
                            odf['H'] = high_df[ticker]
                        if low_df is not None and ticker in low_df.columns:
                            odf['L'] = low_df[ticker]
                        if vol_df is not None and ticker in vol_df.columns:
                            odf['V'] = vol_df[ticker]
                        odf = odf.dropna(subset=['C'])
                    else:
                        odf = pd.DataFrame({
                            'C': data['Close'], 'O': data['Open'],
                            'H': data['High'], 'L': data['Low'],
                        })
                        if 'Volume' in data.columns:
                            odf['V'] = data['Volume']
                        odf = odf.dropna(subset=['C'])

                    if odf.empty:
                        continue

                    # Gruppera per dag för att separera senaste handelsdag vs föregående
                    odf['_date'] = odf.index.date
                    unique_dates = sorted(odf['_date'].unique())

                    # Senaste handelsdagens bars (för candle chart)
                    last_date = unique_dates[-1]
                    last_day_df = odf[odf['_date'] == last_date]

                    bars = []
                    for dt_idx, row in last_day_df.iterrows():
                        bars.append([
                            float(dt_idx.timestamp()),
                            round(float(row.get('O', row['C'])), 4),
                            round(float(row.get('H', row['C'])), 4),
                            round(float(row.get('L', row['C'])), 4),
                            round(float(row['C']), 4),
                        ])

                    # Beräkna previous_close från föregående handelsdag
                    prev_close = 0.0
                    if len(unique_dates) >= 2:
                        prev_date = unique_dates[-2]
                        prev_day_df = odf[odf['_date'] == prev_date]
                        if not prev_day_df.empty:
                            prev_close = float(prev_day_df['C'].iloc[-1])

                    last_close = float(last_day_df['C'].iloc[-1])
                    # Daglig change% (senaste close vs föregående close)
                    change_pct = 0.0
                    if prev_close > 0:
                        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

                    info = {
                        'open_price': round(float(last_day_df['O'].iloc[0]), 4) if 'O' in last_day_df.columns else 0,
                        'day_high': round(float(last_day_df['H'].max()), 4) if 'H' in last_day_df.columns else 0,
                        'day_low': round(float(last_day_df['L'].min()), 4) if 'L' in last_day_df.columns else 0,
                        'day_volume': int(last_day_df['V'].sum()) if 'V' in last_day_df.columns else 0,
                        'close_price': last_close,
                        'previous_close': prev_close,
                        'change_pct': change_pct,
                    }
                    out[ticker] = {'ohlc': bars, 'info': info}

                except Exception:
                    continue

            print(f"[Intraday] Got OHLC for {len(out)} / {len(self.tickers)} tickers")
            self.result.emit(out)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class MarketWatchWebSocket(QObject):
    """Persistent AsyncWebSocket thread for live market price updates via yfinance.

    Använder yfinance AsyncWebSocket med inbyggd auto-reconnect (3s backoff)
    och heartbeat (re-subscribe var 15:e sekund).
    """
    price_update = Signal(dict)  # {symbol, price, change, change_pct}
    status_message = Signal(str)
    error = Signal(str)
    connected = Signal()

    def __init__(self, tickers: list):
        super().__init__()
        self.tickers = tickers
        self._ws = None
        self._stopped = False
        self._loop = None

    @Slot()
    def run(self):
        try:
            from yfinance import AsyncWebSocket
        except ImportError:
            self.error.emit("yfinance AsyncWebSocket not available (upgrade yfinance)")
            return

        # Tysta yfinance/websockets loggers — reconnect-tracebacks är brus, inte fel
        for logger_name in ('yfinance', 'websockets'):
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)

        async def on_message(msg):
            symbol = msg.get('id', '')
            price = float(msg.get('price', 0))
            prev_close = float(msg.get('previous_close', 0))
            # Använd previous_close som fallback när price=0 (stängda marknader)
            display_price = price if price else prev_close
            if symbol and (price or prev_close):
                self.price_update.emit({
                    'symbol': symbol,
                    'price': display_price,
                    'change': float(msg.get('change', 0)),
                    'change_pct': round(float(msg.get('change_percent', 0)), 2),
                    'day_high': float(msg.get('day_high', 0)),
                    'day_low': float(msg.get('day_low', 0)),
                    'day_volume': int(msg.get('day_volume', 0)),
                    'previous_close': prev_close,
                    'open_price': float(msg.get('open_price', 0)),
                    'short_name': msg.get('short_name', ''),
                    'market_hours': msg.get('market_hours', 0),
                    'timestamp': time.time(),
                })

        async def _run_ws():
            # Egen reconnect-loop för att hantera anslutningsfel (t.ex. "Too many open files")
            # AsyncWebSocket har inbyggd reconnect när den väl är ansluten,
            # men om initial connect misslyckas behöver vi retry.
            while not self._stopped:
                try:
                    async with AsyncWebSocket(verbose=False) as ws:
                        self._ws = ws
                        await ws.subscribe(self.tickers)
                        self.connected.emit()
                        self.status_message.emit(f"AsyncWebSocket connected: {len(self.tickers)} tickers")
                        print(f"[WS] AsyncWebSocket connected, subscribed to {len(self.tickers)} tickers")
                        await ws.listen(on_message)
                except Exception as e:
                    if self._stopped:
                        break
                    print(f"[WS] Connection error: {e}, reconnecting in 5s...")
                finally:
                    self._ws = None
                if self._stopped:
                    break
                await asyncio.sleep(5)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(_run_ws())
        except Exception as e:
            if not self._stopped:
                print(f"[WS] AsyncWebSocket error: {e}")
                self.error.emit(str(e))
        finally:
            # Rensa pågående asyncio tasks (t.ex. websockets keepalive) före stängning
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()
            self._loop = None
            print("[WS] Stopped")

    def stop(self):
        """Stäng AsyncWebSocket-anslutningen."""
        self._stopped = True
        if self._ws is not None and self._loop is not None and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            except Exception:
                pass

    def add_tickers(self, new_tickers: list):
        """Dynamically subscribe additional tickers to the active WebSocket."""
        added = [t for t in new_tickers if t not in self.tickers]
        if not added:
            return
        self.tickers.extend(added)
        if self._ws is not None and self._loop is not None and self._loop.is_running():
            async def _subscribe():
                try:
                    if self._ws is not None:
                        await self._ws.subscribe(added)
                        print(f"[WS] Subscribed {len(added)} portfolio tickers: {added}")
                except Exception as e:
                    print(f"[WS] add_tickers error: {e}")
            asyncio.run_coroutine_threadsafe(_subscribe(), self._loop)
        else:
            print(f"[WS] add_tickers: not connected, {len(added)} tickers queued for next connect")


class VolatilityDataWorker(QObject):
    """Worker thread for fetching volatility/market data (async yfinance to prevent GUI freeze).

    OPTIMERING: Flyttar tung yfinance.download från main thread till bakgrund.
    """
    finished = Signal()
    result = Signal(object)  # DataFrame with close prices
    error = Signal(str)
    status_message = Signal(str)

    def __init__(self, tickers: list, period: str = 'max'):
        super().__init__()
        self.tickers = list(tickers)  # Kopiera för säkerhet
        self.period = period

    @Slot()
    def run(self):
        try:
            import yfinance as yf
            self.status_message.emit(f"Fetching volatility data for {len(self.tickers)} tickers (period={self.period})...")

            # Ladda ner alla tickers i ett anrop
            data = yf.download(self.tickers, period=self.period, interval="1d",
                               progress=False, threads=False, ignore_tz=True)

            if data.empty:
                self.error.emit("No volatility data returned")
                return

            # Extrahera Close-priser (hanterar olika yfinance-versioner)
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker: kolumner = ('Close', ticker) eller ('Price', ticker)
                top_levels = data.columns.get_level_values(0).unique()
                if 'Close' in top_levels:
                    close = data['Close']
                elif 'Price' in top_levels:
                    close = data['Price']
                else:
                    close = data[top_levels[0]]
            else:
                close = data if 'Close' not in data.columns else data[['Close']].rename(columns={'Close': self.tickers[0]})

            # Gör till DataFrame om det är en Series
            if isinstance(close, pd.Series):
                close = close.to_frame(name=self.tickers[0])

            print(f"[Volatility] Combined download returned columns: {list(close.columns)}")

            # Fallback: ladda ner saknade tickers individuellt
            missing = [t for t in self.tickers if t not in close.columns]
            if missing:
                print(f"[Volatility] Missing from combined download: {missing}, trying individually...")
                for ticker in missing:
                    try:
                        self.status_message.emit(f"Fetching {ticker} individually...")
                        t = yf.Ticker(ticker)
                        hist = t.history(period=self.period, interval="1d")
                        if hist is not None and not hist.empty and 'Close' in hist.columns:
                            close[ticker] = hist['Close']
                            print(f"[Volatility] {ticker}: got {len(hist)} rows via individual download")
                        else:
                            print(f"[Volatility] {ticker}: no data from individual download")
                    except Exception as e:
                        print(f"[Volatility] {ticker}: individual download failed: {e}")

            print(f"[Volatility] Final columns: {list(close.columns)}")
            self.result.emit(close)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class MarkovChainWorker(QObject):
    """Worker thread for Markov chain analysis (async yfinance to prevent GUI freeze)."""
    finished = Signal()
    result = Signal(object)   # MarkovResult
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, ticker: str):
        super().__init__()
        self.ticker = ticker

    @Slot()
    def run(self):
        try:
            analyzer = MarkovChainAnalyzer()
            res = analyzer.analyze(self.ticker, progress_callback=self.progress.emit)
            self.result.emit(res)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class PortfolioRefreshWorker(QObject):
    """Worker thread for refreshing portfolio Z-scores and MF prices asynchronously.

    OPTIMERING: Flyttar synkrona HTTP-anrop från main thread till bakgrund.
    """
    finished = Signal()
    result = Signal(list, int, int)  # (updated_portfolio, z_count, mf_count)
    error = Signal(str)
    status_message = Signal(str)
    
    def __init__(self, portfolio: list, engine, mf_scraping_available: bool):
        super().__init__()
        # Deep copy portfolio to avoid threading issues
        self.portfolio = copy.deepcopy(portfolio)
        self.engine = engine
        self.mf_scraping_available = mf_scraping_available
    
    @Slot()
    def run(self):
        try:
            updated_z = 0
            updated_mf = 0
            
            # Update Z-scores if engine is available
            if self.engine is not None:
                self.status_message.emit("Updating Z-scores...")
                for pos in self.portfolio:
                    try:
                        pair = pos['pair']
                        ou, spread, current_z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                        pos['previous_z'] = pos.get('current_z', pos['entry_z'])
                        pos['current_z'] = current_z
                        
                        updated_z += 1
                    except Exception as e:
                        pass
            
            # Update MF prices if scraping is available
            if self.mf_scraping_available:
                self.status_message.emit("Fetching mini futures prices...")
                try:
                    isins_to_fetch = []
                    isin_to_positions = {}
                    
                    for idx, pos in enumerate(self.portfolio):
                        for leg, key in [('y', 'mini_y_isin'), ('x', 'mini_x_isin')]:
                            isin = pos.get(key)
                            if isin:
                                if isin not in isin_to_positions:
                                    isin_to_positions[isin] = []
                                    isins_to_fetch.append(isin)
                                isin_to_positions[isin].append((idx, leg))
                    
                    if isins_to_fetch:
                        quotes = get_quotes_batch(isins_to_fetch)
                        for isin, quote in quotes.items():
                            if isin in isin_to_positions and quote.buy_price is not None:
                                for pos_idx, leg in isin_to_positions[isin]:
                                    self.portfolio[pos_idx][f'mf_current_price_{leg}'] = quote.buy_price
                                    updated_mf += 1
                except Exception as e:
                    pass
            
            self.result.emit(self.portfolio, updated_z, updated_mf)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


# ============================================================================
# MARKET CLOCK (DST-AWARE)
# ============================================================================

class MarketClock(QFrame):
    """Individual market clock widget (DST-aware) with dynamic scaling."""

    def __init__(
        self,
        city: str,
        timezone: str,
        market_open: tuple,
        market_close: tuple,
        lunch_break: tuple | None = None,
        parent=None
    ):
        super().__init__(parent)

        self.city = city
        self.tz = ZoneInfo(timezone)
        self.market_open = market_open
        self.market_close = market_close
        self.lunch_break = lunch_break
        
        # Default size (will be updated by dynamic layout)
        self._base_width = 95
        self._base_height = 52
        self.setMinimumSize(80, 45)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)

        # City label
        self.city_label = QLabel(city)
        self.city_label.setAlignment(Qt.AlignCenter)
        self.city_label.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:{TYPOGRAPHY['clock_city']}px; font-weight:600; letter-spacing:1px;"
        )
        layout.addWidget(self.city_label)

        # Time label
        self.time_label = QLabel("--:--")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet(
            f"""
            color:{COLORS['text_primary']};
            font-size:{TYPOGRAPHY['clock_time']}px;
            font-weight:700;
            font-family:'JetBrains Mono','Consolas',monospace;
            """
        )
        layout.addWidget(self.time_label)

        # Status label
        self.status_label = QLabel("●")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:{TYPOGRAPHY['status']}px;"
        )
        layout.addWidget(self.status_label)

        self.update_time()
    
    def scale_to(self, scale: float):
        """Scale the clock widget based on window size."""
        width = int(self._base_width * scale)
        height = int(self._base_height * scale)
        self.setFixedSize(max(80, width), max(45, height))
        
        # Scale fonts
        city_font = max(9, int(12 * scale))
        time_font = max(14, int(18 * scale))
        status_font = max(8, int(11 * scale))
        
        self.city_label.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:{city_font}px; font-weight:600; letter-spacing:1px;"
        )
        self.time_label.setStyleSheet(
            f"""
            color:{COLORS['text_primary']};
            font-size:{time_font}px;
            font-weight:700;
            font-family:'JetBrains Mono','Consolas',monospace;
            """
        )
        self.status_label.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:{status_font}px;"
        )

    def update_time(self):
        now = datetime.now(self.tz)
        self.time_label.setText(now.strftime("%H:%M"))

        weekday = now.weekday()
        current_minutes = now.hour * 60 + now.minute

        open_minutes = self.market_open[0] * 60 + self.market_open[1]
        close_minutes = self.market_close[0] * 60 + self.market_close[1]

        # Weekend
        if weekday >= 5:
            self._set_closed("WEEKEND")
            return

        # Lunch break
        if self.lunch_break:
            lb_start = self.lunch_break[0][0] * 60 + self.lunch_break[0][1]
            lb_end = self.lunch_break[1][0] * 60 + self.lunch_break[1][1]
            if lb_start <= current_minutes < lb_end:
                self._set_closed("LUNCH")
                return

        # Market open / closed
        if open_minutes <= current_minutes < close_minutes:
            self._set_open()
        else:
            self._set_closed("CLOSED")

    def _set_open(self):
        self.status_label.setText("● OPEN")
        self.status_label.setStyleSheet(
            f"color:{COLORS['positive']}; font-size:8px; font-weight:600;"
        )

    def _set_closed(self, text):
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(
            f"color:{COLORS['negative']}; font-size:8px;"
        )
    
    def is_open(self) -> bool:
        """Check if market is currently open (for filtering market data updates)."""
        now = datetime.now(self.tz)
        weekday = now.weekday()
        
        # Weekend - closed
        if weekday >= 5:
            return False
        
        current_minutes = now.hour * 60 + now.minute
        open_minutes = self.market_open[0] * 60 + self.market_open[1]
        close_minutes = self.market_close[0] * 60 + self.market_close[1]
        
        # Lunch break - closed
        if self.lunch_break:
            lb_start = self.lunch_break[0][0] * 60 + self.lunch_break[0][1]
            lb_end = self.lunch_break[1][0] * 60 + self.lunch_break[1][1]
            if lb_start <= current_minutes < lb_end:
                return False
        
        return open_minutes <= current_minutes < close_minutes


# ============================================================================
# HEADER BAR
# ============================================================================

class HeaderBar(QFrame):
    """Professional header bar with logo, clocks, and market status - dynamically scalable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Fix #4: Cache clock references instead of using findChildren
        self._market_clocks = []
        
        # Base dimensions for scaling
        self._base_height = 62
        self.setMinimumHeight(50)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 6, 20, 6)
        layout.setSpacing(20)

        # === LEFT: LOGO + TITLE ===
        left_section = QHBoxLayout()
        left_section.setSpacing(12)

        self.logo = QLabel("◈")
        self.logo.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:32px; font-weight:bold;"
        )
        left_section.addWidget(self.logo)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)

        self.title = QLabel("KLIPPINGE INVESTMENT")
        self.title.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:20px; font-weight:700; letter-spacing:1px;"
        )
        title_layout.addWidget(self.title)

        self.subtitle = QLabel("TRADING TERMINAL")
        self.subtitle.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:15px; font-weight:500; letter-spacing:2px;"
        )
        title_layout.addWidget(self.subtitle)

        left_section.addLayout(title_layout)
        layout.addLayout(left_section)

        layout.addStretch()

        # === CENTER: MARKET CLOCKS ===
        clocks = QHBoxLayout()
        clocks.setSpacing(5)

        # Fix #4: Create clocks and store references
        clock_configs = [
            ("NEW YORK", "America/New_York", (9, 30), (16, 0), None),
            ("SÃO PAULO", "America/Sao_Paulo", (10, 0), (16, 55), None),
            ("LONDON", "Europe/London", (8, 0), (16, 30), None),
            ("STOCKHOLM", "Europe/Stockholm", (9, 0), (17, 30), None),
            ("TEL AVIV", "Asia/Jerusalem", (10, 0), (17, 30), None),
            ("DUBAI", "Asia/Dubai", (10, 0), (15, 0), None),
            ("HONG KONG", "Asia/Hong_Kong", (9, 30), (16, 8), ((12, 0), (13, 0))),
            ("TOKYO", "Asia/Tokyo", (9, 0), (15, 30), ((11, 30), (12, 30))),
            ("SYDNEY", "Australia/Sydney", (10, 0), (16, 0), None),
        ]
        
        for city, tz, open_time, close_time, lunch in clock_configs:
            clock = MarketClock(city, tz, open_time, close_time, lunch)
            self._market_clocks.append(clock)
            clocks.addWidget(clock)

        layout.addLayout(clocks)

        layout.addStretch()

        # === RIGHT: STATUS ===
        right = QVBoxLayout()
        right.setSpacing(2)

        self.status_indicator = QLabel("● CONNECTED")
        self.status_indicator.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        self.status_indicator.setStyleSheet(
            f"color:{COLORS['positive']}; font-size:12px; font-weight:600; letter-spacing:1px;"
        )
        right.addWidget(self.status_indicator)

        self.timestamp = QLabel("--")
        self.timestamp.setAlignment(Qt.AlignRight | Qt.AlignCenter)
        self.timestamp.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:10px; font-family:'JetBrains Mono','Consolas',monospace;"
        )
        right.addWidget(self.timestamp)

        layout.addLayout(right)

        # Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clocks)
        self.timer.start(1000)

        self.update_clocks()
    
    def scale_to(self, scale: float):
        """Scale the header bar and all its children based on window size."""
        # Scale header height
        height = int(self._base_height * scale)
        self.setFixedHeight(max(50, height))
        
        # Scale logo
        logo_size = max(24, int(32 * scale))
        self.logo.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:{logo_size}px; font-weight:bold;"
        )
        
        # Scale title
        title_size = max(16, int(20 * scale))
        self.title.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:{title_size}px; font-weight:700; letter-spacing:1px;"
        )
        
        # Scale subtitle
        subtitle_size = max(12, int(15 * scale))
        self.subtitle.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:{subtitle_size}px; font-weight:500; letter-spacing:2px;"
        )
        
        # Scale status indicator
        status_size = max(10, int(12 * scale))
        # Keep current color state
        current_style = self.status_indicator.styleSheet()
        if COLORS['positive'] in current_style:
            color = COLORS['positive']
        else:
            color = COLORS['negative']
        self.status_indicator.setStyleSheet(
            f"color:{color}; font-size:{status_size}px; font-weight:600; letter-spacing:1px;"
        )
        
        # Scale timestamp
        ts_size = max(8, int(10 * scale))
        self.timestamp.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:{ts_size}px; font-family:'JetBrains Mono','Consolas',monospace;"
        )
        
        # Scale all market clocks
        for clock in self._market_clocks:
            clock.scale_to(scale)

    def update_clocks(self):
        # Fix #4: Use cached clock references instead of findChildren
        for clock in self._market_clocks:
            clock.update_time()

        self.timestamp.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def set_status(self, connected: bool, message: str | None = None):
        if connected:
            self.status_indicator.setText("● CONNECTED")
            self.status_indicator.setStyleSheet(
                f"color:{COLORS['positive']}; font-size:12px; font-weight:600;"
            )
        else:
            self.status_indicator.setText(f"● {message or 'DISCONNECTED'}")
            self.status_indicator.setStyleSheet(
                f"color:{COLORS['negative']}; font-size:9px; font-weight:600;"
            )
    
    def get_open_markets(self) -> set:
        """Return set of open market regions based on clock status.
        
        Maps clock cities to index regions for smart market data filtering.
        Returns regions that have at least one open market.
        """
        # Mapping: clock city -> index regions it represents
        CLOCK_TO_REGIONS = {
            'NEW YORK': ['AMERICA'],
            'SÃO PAULO': ['AMERICA'],
            'LONDON': ['EUROPE'],
            'STOCKHOLM': ['EUROPE'],
            'TEL AVIV': ['MIDDLE EAST'],
            'DUBAI': ['MIDDLE EAST'],
            'HONG KONG': ['ASIA'],
            'TOKYO': ['ASIA'],
            'SYDNEY': ['OCEANIA'],
        }
        
        open_regions = set()
        print(f"[MarketWatch] Checking {len(self._market_clocks)} clocks:")
        for clock in self._market_clocks:
            is_open = clock.is_open()
            regions = CLOCK_TO_REGIONS.get(clock.city, [])
            print(f"  - {clock.city}: {'OPEN' if is_open else 'CLOSED'} -> {regions}")
            if is_open:
                open_regions.update(regions)
        
        print(f"[MarketWatch] Open regions: {open_regions}")
        return open_regions


class LoadingOverlay(QWidget):
    """Subtle loading overlay with animation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: rgba(10, 10, 10, 0.7);")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Loading indicator
        self.spinner = QLabel("⟳")
        self.spinner.setStyleSheet(f"""
            color: {COLORS['accent']}; 
            font-size: 32px;
            background: transparent;
        """)
        self.spinner.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.spinner)
        
        # Loading text
        self.text_label = QLabel("Loading...")
        self.text_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']}; 
            font-size: 12px;
            background: transparent;
        """)
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)
        
        # Animation
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)
        
        self.hide()
    
    def rotate(self):
        """Rotate the spinner."""
        self.angle = (self.angle + 30) % 360
        # Simple rotation effect using different characters
        chars = ["◐", "◓", "◑", "◒"]
        self.spinner.setText(chars[(self.angle // 90) % 4])
    
    def show_loading(self, message: str = "Loading..."):
        """Show the loading overlay."""
        self.text_label.setText(message)
        self.timer.start(100)
        self.show()
        self.raise_()
    
    def hide_loading(self):
        """Hide the loading overlay."""
        self.timer.stop()
        self.hide()


# ============================================================================
# CUSTOM WIDGETS
# ============================================================================

class MetricCard(QFrame):
    """Metric display card with gradient background - dynamically scalable."""
    
    def __init__(self, label: str, value: str = "-", tooltip: str = "", parent=None):
        super().__init__(parent)
        self._base_label_font = 12
        self._base_value_font = 22
        self._current_color = COLORS['text_primary']
        
        if tooltip:
            self.setToolTip(tooltip)
        
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['bg_elevated']}, 
                    stop:1 {COLORS['bg_card']});
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
            }}
            QFrame:hover {{
                border-color: {COLORS['accent_dark']};
            }}
            QToolTip {{
                background-color: #1a1a2e;
                color: #e8e8e8;
                border: 1px solid {COLORS['accent']};
                padding: 10px 12px;
                border-radius: 5px;
                font-size: 12px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        
        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet(f"""
            color: {COLORS['text_muted']}; 
            font-size: {TYPOGRAPHY['metric_label']}px; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
            background: transparent; 
            border: none;
        """)
        self.label_widget.setWordWrap(True)
        
        self.value_widget = QLabel(value)
        self.value_widget.setStyleSheet(f"""
            color: {COLORS['text_primary']}; 
            font-size: {TYPOGRAPHY['metric_value']}px; 
            font-weight: 600; 
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: transparent; 
            border: none;
        """)
        self.value_widget.setWordWrap(True)
        
        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
        layout.addStretch()
    
    def scale_to(self, scale: float):
        """Scale the metric card based on window size."""
        label_font = max(9, int(self._base_label_font * scale))
        value_font = max(16, int(self._base_value_font * scale))
        
        self.label_widget.setStyleSheet(f"""
            color: {COLORS['text_muted']}; 
            font-size: {label_font}px; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
            background: transparent; 
            border: none;
        """)
        
        self.value_widget.setStyleSheet(f"""
            color: {self._current_color}; 
            font-size: {value_font}px; 
            font-weight: 600; 
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: transparent; 
            border: none;
        """)
    
    def set_value(self, value: str, color: str = None):
        if color is None:
            color = COLORS['text_primary']
        self._current_color = color
        self.value_widget.setText(value)
        self.value_widget.setStyleSheet(f"""
            color: {color}; 
            font-size: {TYPOGRAPHY['metric_value']}px; 
            font-weight: 600; 
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: transparent; 
            border: none;
        """)
    
    def set_title(self, title: str):
        """Update the card title/label."""
        self.label_widget.setText(title)


class WindowResultRow(QFrame):
    """Compact per-window result row for OU analytics panel."""

    def __init__(self, window_data: dict, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(48)
        passed = window_data.get('passed', False)
        border_color = COLORS['positive'] if passed else COLORS['negative']
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border-left: 3px solid {border_color};
                border-radius: 3px;
                margin: 1px 0;
            }}
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        ws = str(window_data.get('window_size', '?'))
        ws_label = QLabel(ws)
        ws_label.setFixedWidth(42)
        ws_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-weight: 700; font-size: 12px; "
            f"background: transparent; border: none;")
        layout.addWidget(ws_label)

        status_text = "PASS" if passed else "FAIL"
        status_color = COLORS['positive'] if passed else COLORS['negative']
        status_label = QLabel(status_text)
        status_label.setFixedWidth(36)
        status_label.setStyleSheet(
            f"color: {status_color}; font-weight: 600; font-size: 11px; "
            f"background: transparent; border: none;")
        layout.addWidget(status_label)

        # Compact metrics
        parts = []
        hl = window_data.get('half_life_days')
        if hl is not None:
            parts.append(f"HL:{hl:.0f}d")
        eg = window_data.get('eg_pvalue')
        if eg is not None:
            parts.append(f"EG:{eg:.3f}")
        h = window_data.get('hurst_exponent')
        if h is not None:
            parts.append(f"H:{h:.2f}")
        metrics_text = "  ".join(parts) if parts else "-"
        metrics_label = QLabel(metrics_text)
        metrics_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; "
            f"font-family: 'JetBrains Mono', 'Consolas', monospace; "
            f"background: transparent; border: none;")
        layout.addWidget(metrics_label, 1)

        failed_at = window_data.get('failed_at', '')
        if failed_at:
            fa_label = QLabel(f"[{failed_at}]")
            fa_label.setStyleSheet(
                f"color: {COLORS['warning']}; font-size: 10px; "
                f"background: transparent; border: none;")
            layout.addWidget(fa_label)


class CompactMetricCard(QFrame):
    """Compact metric display card for dense layouts."""
    
    def __init__(self, label: str, value: str = "-", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['bg_elevated']}, stop:1 {COLORS['bg_card']});
                border: none;
                border-radius: 4px;
            }}
        """)
        self.setMaximumHeight(70)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        
        self.label_widget = QLabel(label)
        self.label_widget.setStyleSheet(f"""
            color: {COLORS['text_muted']}; 
            font-size: {TYPOGRAPHY['body_small']}px; 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
            background: transparent;
        """)
        
        self.value_widget = QLabel(value)
        self.value_widget.setStyleSheet(f"""
            color: {COLORS['text_primary']}; 
            font-size: 16px; 
            font-weight: 600; 
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: transparent;
        """)
        
        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
    
    def set_value(self, value: str, color: str = None):
        """Update the displayed value."""
        if color is None:
            color = COLORS['text_primary']
        self.value_widget.setText(value)
        self.value_widget.setStyleSheet(f"""
            color: {color}; 
            font-size: 16px; 
            font-weight: 600; 
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            background: transparent;
        """)


class VolatilitySparkline(QWidget):
    """Sparkline chart for volatility cards with median line - dynamically scalable."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.color = QColor(COLORS['accent'])
        self.median_value = None
        self._base_height = 35
        self.setMinimumHeight(25)
        self.setMinimumWidth(80)
        self.setStyleSheet("background: transparent; border: none;")
    
    def scale_to(self, scale: float):
        """Scale the sparkline height based on window size."""
        height = max(25, int(self._base_height * scale))
        self.setFixedHeight(height)
    
    def set_data(self, values: list, color: str = None, median: float = None):
        """Set data for sparkline."""
        if color is None:
            color = COLORS['accent']
        self.values = values if values else []
        self.color = QColor(color)
        self.median_value = median
        self.update()
    
    def paintEvent(self, event):
        """Draw the sparkline with optional median line."""
        if len(self.values) < 2:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate bounds
        min_v = min(self.values)
        max_v = max(self.values)
        value_range = max_v - min_v if max_v != min_v else 1
        
        w = self.width()
        h = self.height()
        padding_x = 4
        padding_y = 3
        
        # Helper function to convert value to y coordinate
        def value_to_y(val):
            return h - padding_y - ((val - min_v) / value_range) * (h - 2 * padding_y)
        
        # Create points
        points = []
        for i, val in enumerate(self.values):
            x = padding_x + (i / (len(self.values) - 1)) * (w - 2 * padding_x)
            y = value_to_y(val)
            points.append((x, y))
        
        # Draw filled area (gradient)
        fill_color = QColor(self.color)
        fill_color.setAlpha(30)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)
        
        polygon = QPolygonF()
        polygon.append(QPointF(points[0][0], h))
        for x, y in points:
            polygon.append(QPointF(x, y))
        polygon.append(QPointF(points[-1][0], h))
        painter.drawPolygon(polygon)
        
        # Draw median line if available
        if self.median_value is not None and min_v <= self.median_value <= max_v:
            median_y = value_to_y(self.median_value)
            pen = QPen(QColor(COLORS['text_muted']), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(padding_x, median_y), QPointF(w - padding_x, median_y))
        
        # Draw main line
        pen = QPen(self.color, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        
        for i in range(len(points) - 1):
            painter.drawLine(
                QPointF(points[i][0], points[i][1]),
                QPointF(points[i+1][0], points[i+1][1])
            )
        
        # Draw end point (current value)
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(QPointF(points[-1][0], points[-1][1]), 3, 3)


class VolatilityCard(QFrame):
    """Volatility indicator card with gradient background and sparkline chart."""
    
    def __init__(self, ticker: str, name: str, description: str = "", parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.name = name
        self._history_data = []  # Store historical data for sparkline
        
        self.setMinimumHeight(165)  # Increased to accommodate sparkline
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(212, 165, 116, 0.06), 
                    stop:1 {COLORS['bg_card']});
                border: 1px solid {COLORS['border_subtle']};
                border-left: 2px solid {COLORS['accent']};
                border-radius: 4px;
            }}
            QFrame:hover {{
                border-color: {COLORS['accent_dark']};
                border-left-color: {COLORS['accent_bright']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        # Row 1: Name and change %
        row1 = QHBoxLayout()
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; font-weight: 600; background: transparent; border: none;")
        row1.addWidget(self.name_label)
        row1.addStretch()
        self.change_label = QLabel("")
        self.change_label.setStyleSheet("font-size: 13px; background: transparent; border: none;")
        row1.addWidget(self.change_label)
        layout.addLayout(row1)
        
        # Row 2: Value and percentile dot
        row2 = QHBoxLayout()
        self.value_label = QLabel("-")
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 18px; font-family: 'JetBrains Mono', 'Consolas', monospace; font-weight: 600; background: transparent; border: none;")
        row2.addWidget(self.value_label)
        row2.addStretch()
        self.pct_label = QLabel("")
        self.pct_label.setStyleSheet(f"font-size: 13px; color: {COLORS['text_muted']}; background: transparent; border: none;")
        row2.addWidget(self.pct_label)
        layout.addLayout(row2)
        
        # Row 3: Median and Mode
        self.median_label = QLabel("")
        self.median_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        layout.addWidget(self.median_label)
        
        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        layout.addWidget(self.mode_label)
        
        # Row 4: Sparkline chart (NEW!)
        self.sparkline = VolatilitySparkline()
        layout.addWidget(self.sparkline)
        
        # Row 5: Description
        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; font-style: italic; background: transparent; border: none;")
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)
        
        layout.addStretch()
    
    def update_data(self, value: float, change_pct: float, percentile: float, 
                    median: float, mode: float, description: str,
                    history: list = None):
        """Update card with new data.
        
        Args:
            history: Optional list of historical values for sparkline (last ~60 days)
        """
        # Value with color based on level
        self.value_label.setText(f"{value:.2f}")
        
        # Change with color
        change_color = COLORS['positive'] if change_pct >= 0 else COLORS['negative']
        self.change_label.setText(f"{change_pct:+.2f}%")
        self.change_label.setStyleSheet(f"color: {change_color}; font-size: 13px; font-family: 'JetBrains Mono', monospace; background: transparent; border: none;")
        
        # Percentile with colored dot
        if percentile < 25:
            pct_color = COLORS['positive']  
        elif percentile < 50:
            pct_color = COLORS['accent']  
        elif percentile < 90:
            pct_color = COLORS['warning'] 
        else:
            pct_color = COLORS['negative']  
        
        self.pct_label.setText(f"● {percentile:.0f}th pct")
        self.pct_label.setStyleSheet(f"color: {pct_color}; font-size: 13px; background: transparent; border: none;")
        
        # Stats
        self.median_label.setText(f"Median: {median:.1f}")
        self.mode_label.setText(f"Mode: {mode:.1f}")
        
        # Description
        self.desc_label.setText(description)
        
        # Update sparkline if history provided
        if history is not None and len(history) > 1:
            self._history_data = history
            # Determine color based on price direction (up = green, down = red)
            if history[-1] >= history[0]:
                spark_color = COLORS['positive']
            else:
                spark_color = COLORS['negative']
            self.sparkline.set_data(history, spark_color, median)
    
    def scale_to(self, scale: float):
        """Scale the volatility card based on window size."""
        # Scale fonts
        name_font = max(11, int(13 * scale))
        value_font = max(14, int(18 * scale))
        pct_font = max(11, int(13 * scale))
        stat_font = max(10, int(12 * scale))
        desc_font = max(10, int(12 * scale))
        
        self.name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: {name_font}px; font-weight: 600; background: transparent; border: none;")
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: {value_font}px; font-family: 'JetBrains Mono', 'Consolas', monospace; font-weight: 600; background: transparent; border: none;")
        
        # Keep pct_label color
        current_style = self.pct_label.styleSheet()
        for color in [COLORS['positive'], COLORS['accent'], COLORS['warning'], COLORS['negative']]:
            if color in current_style:
                self.pct_label.setStyleSheet(f"color: {color}; font-size: {pct_font}px; background: transparent; border: none;")
                break
        
        self.median_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {stat_font}px; background: transparent; border: none;")
        self.mode_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {stat_font}px; background: transparent; border: none;")
        self.desc_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {desc_font}px; font-style: italic; background: transparent; border: none;")
        
        # Scale sparkline
        self.sparkline.scale_to(scale)

class SectionHeader(QLabel):
    """Section header label with amber accent - dynamically scalable."""
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._base_font = 13
        self.setObjectName("sectionHeader")
        self.setStyleSheet(f"""
            color: {COLORS['accent']}; 
            font-size: {TYPOGRAPHY['header_section']}px; 
            font-weight: 600; 
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 8px 0;
            background: transparent;
        """)
    
    def scale_to(self, scale: float):
        """Scale the section header based on window size."""
        font_size = max(10, int(self._base_font * scale))
        self.setStyleSheet(f"""
            color: {COLORS['accent']}; 
            font-size: {font_size}px; 
            font-weight: 600; 
            letter-spacing: 1.5px;
            text-transform: uppercase;
            padding: 8px 0;
            background: transparent;
        """)




class MarketDetailDialog(QDialog):
    """Popup showing 5-day chart and stats for a market instrument."""

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        name = data.get('name', '')
        symbol = data.get('symbol', '')
        self.setWindowTitle(f"{name} ({symbol})")
        self.setMinimumSize(520, 420)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                background: transparent;
                border: none;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel(f"{name}  ({symbol})")
        title.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        # Chart (pyqtgraph)
        history = data.get('history', [])
        pg_mod = get_pyqtgraph()
        if pg_mod and len(history) >= 2:
            plot_widget = pg_mod.PlotWidget()
            plot_widget.setBackground(COLORS['bg_card'])
            plot_widget.setMinimumHeight(220)
            plot_widget.showGrid(x=False, y=True, alpha=0.2)

            closes = [h[1] for h in history]
            xs = list(range(len(closes)))

            color = COLORS['positive'] if closes[-1] >= closes[0] else COLORS['negative']
            pen = pg_mod.mkPen(color=color, width=2)
            plot_widget.plot(xs, closes, pen=pen)

            # Date labels on x-axis
            ax = plot_widget.getAxis('bottom')
            step = max(1, len(history) // 5)
            ticks = [(i, history[i][0]) for i in range(0, len(history), step)]
            ax.setTicks([ticks])

            layout.addWidget(plot_widget)
        else:
            no_chart = QLabel("No chart data available")
            no_chart.setAlignment(Qt.AlignCenter)
            no_chart.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 13px;")
            layout.addWidget(no_chart)

        # Stats grid
        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            f"background: {COLORS['bg_elevated']}; "
            f"border: 1px solid {COLORS['border_default']}; border-radius: 4px;")
        stats_grid = QGridLayout(stats_frame)
        stats_grid.setContentsMargins(12, 10, 12, 10)
        stats_grid.setSpacing(8)

        price = data.get('price', 0)
        change_pct = data.get('change_pct', 0)
        market = data.get('market', '')

        if history and len(history) >= 2:
            closes = [h[1] for h in history]
            high_5d = max(closes)
            low_5d = min(closes)
            ret_5d = ((closes[-1] / closes[0]) - 1) * 100
        else:
            high_5d = low_5d = ret_5d = 0

        price_fmt = f"{price:,.2f}" if price >= 100 else f"{price:.4f}"
        stats = [
            ("Region", market),
            ("Current Price", price_fmt),
            ("Daily Change", f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"),
            ("5-Day High", f"{high_5d:,.2f}" if high_5d >= 100 else f"{high_5d:.4f}"),
            ("5-Day Low", f"{low_5d:,.2f}" if low_5d >= 100 else f"{low_5d:.4f}"),
            ("5-Day Return", f"{'+' if ret_5d >= 0 else ''}{ret_5d:.2f}%"),
        ]

        for i, (label_text, value_text) in enumerate(stats):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
            val = QLabel(str(value_text))

            if 'Change' in label_text or 'Return' in label_text:
                try:
                    num = float(value_text.replace('%', '').replace('+', ''))
                    c = COLORS['positive'] if num >= 0 else COLORS['negative']
                except ValueError:
                    c = COLORS['text_primary']
                val.setStyleSheet(
                    f"color: {c}; font-size: 14px; font-weight: 600; "
                    f"font-family: 'JetBrains Mono', monospace;")
            else:
                val.setStyleSheet(
                    f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: 600; "
                    f"font-family: 'JetBrains Mono', monospace;")

            stats_grid.addWidget(lbl, i, 0)
            stats_grid.addWidget(val, i, 1)

        layout.addWidget(stats_frame)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_default']};
                padding: 8px 24px;
                color: {COLORS['text_secondary']};
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background: {COLORS['bg_hover']};
                color: {COLORS['accent']};
            }}
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


class NewsItem(QFrame):
    """Compact single-row news item - [title] [TICKER badge] [time] on one line."""

    clicked = Signal(str)  # Emits URL when clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.url = ""
        self._base_font_size = 13

        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border: none;
                border-left: 2px solid {COLORS['border_subtle']};
                border-top: 2px solid {COLORS['accent']};
                border-radius: 2px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Title (single line, elided — full text in tooltip)
        self.title_label = QLabel("")
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; background: transparent; border: none;")
        self.title_label.setWordWrap(False)
        self.title_label.setMinimumWidth(0)
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.title_label, stretch=1)

        # Ticker badge — guaranteed minimum width so it's always visible
        self.ticker_label = QLabel("")
        self.ticker_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent']};
                border: 1px solid {COLORS['accent']};
                font-size: 11px;
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 3px;
            }}
        """)
        self.ticker_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.ticker_label.setMinimumWidth(50)
        self.ticker_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout.addWidget(self.ticker_label)

        # Time label — fixed width so it's always visible
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent; border: none;")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setMinimumWidth(40)
        self.time_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        layout.addWidget(self.time_label)

    def scale_to(self, scale: float):
        """Scale the news item based on window size."""
        title_font = max(12, int(self._base_font_size * scale))
        badge_font = max(10, int(11 * scale))
        h = max(32, int(36 * scale))
        self.setFixedHeight(h)

        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: {title_font}px; background: transparent; border: none;")
        self.ticker_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent']};
                border: 1px solid {COLORS['accent']};
                font-size: {badge_font}px;
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 3px;
            }}
        """)
        self.time_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {max(9, int(11 * scale))}px; background: transparent; border: none;")

    def set_news(self, title: str, time_str: str, url: str, ticker: str):
        """Set the news item data."""
        self.title_label.setText(title)
        self.title_label.setToolTip(title)
        self.time_label.setText(time_str)

        self.url = url
        self.ticker = ticker or ""

        # Show ticker up to 8 chars in badge, full in tooltip
        self.ticker_label.setText(self.ticker[:8])
        self.ticker_label.setToolTip(self.ticker)

    def mousePressEvent(self, event):
        """Handle click to open URL."""
        if self.url:
            self.clicked.emit(self.url)
        super().mousePressEvent(event)


# News cache file path
NEWS_CACHE_FILE = Paths.news_cache_file()


def load_news_cache() -> list:
    """Load cached news from file, filtering to last 24 hours."""
    try:
        if os.path.exists(NEWS_CACHE_FILE):
            with open(NEWS_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Filter to last 24 hours
            cutoff = datetime.now() - timedelta(hours=24)
            cutoff_ts = cutoff.timestamp()
            
            valid_news = [n for n in cached if n.get('timestamp', 0) > cutoff_ts]
            return valid_news
    except Exception as e:
        print(f"[NewsCache] Error loading cache: {e}")
    return []


def save_news_cache(news_items: list):
    """Save news items to cache file."""
    try:
        # Filter to last 24 hours before saving
        cutoff = datetime.now() - timedelta(hours=24)
        cutoff_ts = cutoff.timestamp()
        
        valid_news = [n for n in news_items if n.get('timestamp', 0) > cutoff_ts]
        
        with open(NEWS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(valid_news, f, ensure_ascii=False, indent=2)
        print(f"[NewsCache] Saved {len(valid_news)} news items to cache")
    except Exception as e:
        print(f"[NewsCache] Error saving cache: {e}")


class NewsFeedWorker(QObject):
    """Worker thread for fetching news from yfinance for all tickers.
    
    OPTIMIZED: Uses ThreadPoolExecutor for parallel fetching (~5-10x faster).
    """
    
    finished = Signal()
    result = Signal(list)  # List of news items
    error = Signal(str)
    status_message = Signal(str)
    
    def __init__(self, csv_path: str = None):
        super().__init__()
        self.csv_path = csv_path or SCHEDULED_CSV_PATH
    
    def _fetch_ticker_news(self, ticker_symbol: str, cutoff_ts: float) -> list:
        """Fetch news for a single ticker. Returns list of news items."""
        import yfinance as yf

        news_items = []
        try:
            ticker = yf.Ticker(ticker_symbol)
            news_list = ticker.news
            
            if not news_list:
                return []
            
            for item in news_list:
                # Handle both old and new yfinance formats
                if 'content' in item:
                    content = item['content']
                    content_type = content.get('contentType', '')
                    
                    # Filter: only STORY type
                    if content_type != 'STORY':
                        continue
                    
                    news_id = content.get('id', '')
                    title = content.get('title', 'No title')
                    
                    # Get URL from clickThroughUrl
                    click_url = content.get('clickThroughUrl', {})
                    url = click_url.get('url', '') if isinstance(click_url, dict) else ''
                    
                    # Get publish time
                    pub_date_str = content.get('pubDate', '')
                    if pub_date_str:
                        try:
                            dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            timestamp = dt.timestamp()
                        except (ValueError, OSError):
                            timestamp = 0
                    else:
                        timestamp = 0
                    
                    # Get provider/source
                    provider = content.get('provider', {})
                    source = provider.get('displayName', 'Unknown') if isinstance(provider, dict) else 'Unknown'
                else:
                    # Old yfinance format
                    news_id = item.get('uuid', item.get('id', ''))
                    title = item.get('title', 'No title')
                    url = item.get('link', '')
                    timestamp = item.get('providerPublishTime', 0)
                    source = item.get('publisher', 'Unknown')
                
                # Skip if older than 24h
                if timestamp < cutoff_ts:
                    continue
                
                # Format time string
                if timestamp > 0:
                    dt = datetime.fromtimestamp(timestamp)
                    if dt.date() == datetime.now().date():
                        time_str = dt.strftime('%H:%M')
                    else:
                        time_str = dt.strftime('%d %b %H:%M')
                else:
                    time_str = ""
                
                news_items.append({
                    'id': news_id,
                    'title': title,
                    'source': source,
                    'time': time_str,
                    'timestamp': timestamp,
                    'ticker': ticker_symbol,
                    'url': url
                })
                
        except Exception as e:
            pass  # Skip problematic tickers silently
        
        return news_items
    
    def run(self):
        """Fetch news for all tickers from CSV file using parallel execution."""
        try:
            # Load existing cache
            cached_news = load_news_cache()
            cached_ids = {n.get('id') for n in cached_news if n.get('id')}
            print(f"[NewsFeed] Loaded {len(cached_news)} cached news items")
            
            # Load tickers from CSV
            tickers = []
            if os.path.exists(self.csv_path):
                try:
                    df = pd.read_csv(self.csv_path, sep=';', encoding='utf-8-sig')
                    if 'Ticker' in df.columns:
                        tickers = df['Ticker'].dropna().tolist()
                    print(f"[NewsFeed] Loaded {len(tickers)} tickers from CSV")
                except Exception as e:
                    print(f"[NewsFeed] Error reading CSV: {e}")
            
            if not tickers:
                # Fallback to some default tickers
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
                print(f"[NewsFeed] Using default tickers: {tickers}")
            
            self.status_message.emit(f"Fetching news for {len(tickers)} tickers (parallel)...")
            
            # 24 hour cutoff
            cutoff = datetime.now() - timedelta(hours=24)
            cutoff_ts = cutoff.timestamp()
            
            all_news = list(cached_news)  # Start with cached
            new_count = 0
            
            # PARALLEL FETCH using ThreadPoolExecutor
            # Use 5 workers - reduced from 10 to avoid overwhelming yfinance API
            MAX_WORKERS = 5
            completed = 0
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(self._fetch_ticker_news, ticker, cutoff_ts): ticker
                    for ticker in tickers
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_ticker):
                    completed += 1
                    if completed % 20 == 0:
                        self.status_message.emit(f"Fetching news... {completed}/{len(tickers)} tickers")
                    
                    try:
                        news_items = future.result()
                        for news_item in news_items:
                            news_id = news_item.get('id')
                            if news_id and news_id not in cached_ids:
                                all_news.append(news_item)
                                cached_ids.add(news_id)
                                new_count += 1
                    except Exception as e:
                        pass  # Skip failed tickers
            
            # Sort by timestamp (newest first) and cap at 500 items
            all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            all_news = all_news[:500]

            # Save to cache
            save_news_cache(all_news)
            
            self.status_message.emit(f"Loaded {len(all_news)} news ({new_count} new)")
            self.result.emit(all_news)
            
        except Exception as e:
            print(f"[NewsFeed] ERROR: {e}")
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class NewsFeedWidget(QWidget):
    """News feed widget showing latest market news from yfinance."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._news_worker = None
        self._news_thread = None
        self._news_running = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header container with proper border styling
        header_widget = QFrame()
        header_widget.setStyleSheet(f"""
            QFrame {{
                background: transparent;
                border-left: none;
                border-right: none;
                border-bottom: none;
            }}
        """)
        
        header_row = QHBoxLayout(header_widget)
        header_row.setContentsMargins(6, 8, 6, 6)
        header_row.setSpacing(8)
        
        # Title
        header = QLabel("NEWS FEED")
        header.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent']};
                font-size: {TYPOGRAPHY['header_section']}px;
                letter-spacing: 1.5px;
                font-weight: 600;
                padding: 8px 0;
                background: transparent;
                border: none;
            }}
        """)
        
        # Refresh button with visible icon
        self.refresh_btn = QPushButton("↻")
        self._base_btn_size = 30
        self.refresh_btn.setFixedSize(30, 30)
        
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 4px;
                color: {COLORS['text_secondary']};
                font-size: 14px;
                font-weight: bold;
            }}

            QPushButton:hover {{
                background: {COLORS['bg_hover']};
                border-color: {COLORS['accent']};
                color: {COLORS['accent']};
            }}
        
            QPushButton:pressed {{
                background: {COLORS['bg_card']};
            }}
        """)
        
        self.refresh_btn.setToolTip("Refresh news")
        
        header.setAlignment(Qt.AlignCenter)
        
        header_row.addStretch()
        header_row.addWidget(header)
        header_row.addStretch()
        header_row.addWidget(self.refresh_btn)
        
        # Add header to main layout
        layout.addWidget(header_widget)
        
        # Scroll area for news items
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.news_container = QWidget()
        self.news_layout = QVBoxLayout(self.news_container)
        self.news_layout.setContentsMargins(0, 0, 0, 0)
        self.news_layout.setSpacing(3)
        self.news_layout.addStretch()
        
        scroll.setWidget(self.news_container)
        layout.addWidget(scroll)
        
        # Loading label
        self.loading_label = QLabel("Loading news...")
        self.loading_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; padding: 10px;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.news_layout.insertWidget(0, self.loading_label)
        
        # Store news items
        self.news_items = []
        
        # Load cached news immediately
        self._load_cached_news()
    
    def scale_to(self, scale: float):
        """Scale the news feed widget based on window size."""
        # Scale refresh button
        btn_size = max(28, int(self._base_btn_size * scale))
        self.refresh_btn.setFixedSize(btn_size, btn_size)
        
        # Scale all news items
        for item in self.news_items:
            if hasattr(item, 'scale_to'):
                item.scale_to(scale)
    
    def _load_cached_news(self):
        """Load and display cached news immediately."""
        cached = load_news_cache()
        if cached:
            self._display_news(cached)
            self.loading_label.hide()
    
    def refresh_news(self, csv_path: str = None):
        """Refresh news feed from CSV tickers."""
        if self._news_running:
            return
        
        self._news_running = True
        self.loading_label.setText("Updating news...")
        self.loading_label.show()
        
        # Start worker thread
        self._news_thread = QThread()
        self._news_worker = NewsFeedWorker(csv_path)
        self._news_worker.moveToThread(self._news_thread)
        
        self._news_thread.started.connect(self._news_worker.run)
        self._news_worker.finished.connect(self._news_thread.quit)
        self._news_worker.finished.connect(self._on_news_thread_finished)
        self._news_worker.result.connect(self._on_news_received)
        self._news_worker.error.connect(self._on_news_error)
        
        self._news_thread.start()
    
    def _on_news_thread_finished(self):
        """Handle thread completion."""
        self._news_running = False
        
        # Använd deleteLater() för säker cleanup
        if self._news_worker is not None:
            self._news_worker.deleteLater()
        if self._news_thread is not None:
            self._news_thread.deleteLater()
        
        self._news_worker = None
        self._news_thread = None
    
    def _on_news_received(self, news_data: list):
        """Handle received news data."""
        self.loading_label.hide()
        self._display_news(news_data)
    
    def _display_news(self, news_data: list):
        """Display news items in the feed."""
        # Clear old news items
        for item in self.news_items:
            item.deleteLater()
        self.news_items.clear()
        
        if not news_data:
            self.loading_label.setText("No news available")
            self.loading_label.show()
            return
       
        for news in news_data:
            item = NewsItem()
            item.set_news(
                title=news.get('title', 'No title'),
                time_str=news.get('time', ''),
                url=news.get('url', ''),
                ticker=news.get('ticker', '')
            )
            item.clicked.connect(self._open_url)
            self.news_items.append(item)
            # Insert before the stretch
            self.news_layout.insertWidget(self.news_layout.count() - 1, item)
    
    def _on_news_error(self, error: str):
        """Handle news fetch error."""
        self.loading_label.setText(f"Error: {error}")
        self.loading_label.show()
    
    def _open_url(self, url: str):
        """Open URL in default browser."""
        if url:
            QDesktopServices.openUrl(QUrl(url))


# ============================================================================
# MAIN WINDOW
# ============================================================================

class PairsTradingTerminal(QMainWindow):
    """Main application window matching Streamlit layout."""

    # All market instruments for treemap (used by WS startup + treemap rendering)
    MARKET_INSTRUMENTS = {
        # America
        '^GSPC': ('S&P 500', 'AMERICA'),
        '^NDX': ('NASDAQ 100', 'AMERICA'),
        '^DJI': ('Dow Jones', 'AMERICA'),
        '^RUT': ('Russell 2000', 'AMERICA'),
        '^GSPTSE': ('Toronto', 'AMERICA'),
        '^BVSP': ('São Paulo', 'AMERICA'),
        # Europe
        '^FTSE': ('London', 'EUROPE'),
        '^FCHI': ('Paris', 'EUROPE'),
        '^STOXX50E': ('Europe 50', 'EUROPE'),
        '^N100': ('Euronext 100', 'EUROPE'),
        '^GDAXI': ('Frankfurt', 'EUROPE'),
        '^OMX': ('Stockholm', 'EUROPE'),
        'OBX.OL': ('Oslo', 'EUROPE'),
        '^OMXC25': ('Copenhagen', 'EUROPE'),
        '^OMXH25': ('Helsinki', 'EUROPE'),
        # Asia
        '^N225': ('Tokyo', 'ASIA'),
        '^HSI': ('Hong Kong', 'ASIA'),
        '000001.SS': ('Shanghai', 'ASIA'),
        '^KS11': ('Seoul', 'ASIA'),
        '^TWII': ('Taipei', 'ASIA'),
        '399106.SZ': ('Shenzen', 'ASIA'),
        '^BSESN': ('India', 'ASIA'),
        # Oceania
        '^AXJO': ('Sydney', 'OCEANIA'),
        # Currencies
        'EURUSD=X': ('EUR/USD', 'CURRENCIES'),
        'EURSEK=X': ('EUR/SEK', 'CURRENCIES'),
        'GBPUSD=X': ('GBP/USD', 'CURRENCIES'),
        'USDJPY=X': ('USD/JPY', 'CURRENCIES'),
        'USDSEK=X': ('USD/SEK', 'CURRENCIES'),
        # Commodities
        'GC=F': ('Gold', 'COMMODITIES'),
        'SI=F': ('Silver', 'COMMODITIES'),
        'CL=F': ('Crude Oil', 'COMMODITIES'),
        'BZ=F': ('Brent Crude Oil', 'COMMODITIES'),
        'NG=F': ('Natural Gas', 'COMMODITIES'),
        'HG=F': ('Copper', 'COMMODITIES'),
        # Yields
        '^TNX': ('10Y Yield', 'YIELDS'),
        '^IRX': ('13W T-bill', 'YIELDS'),
        '^FVX': ('5Y Yield', 'YIELDS'),
        '^TYX': ('30Y Yield', 'YIELDS'),
        # Cryptocurrencies
        'BTC-USD': ('Bitcoin', 'CRYPTO'),
        'SOL-USD': ('Solana', 'CRYPTO'),
        'XRP-USD': ('XRP', 'CRYPTO'),
        'ETH-USD': ('Ethereum', 'CRYPTO'),
    }

    def __init__(self):
        super().__init__()
        
        self.setWindowIcon(QIcon(Paths.logo_icon()))
        self.setWindowTitle(" KLIPPINGE INVESTMENT TRADING TERMINAL")
        self.setGeometry(100, 50, 1600, 900)
        self.setMinimumSize(1200, 700)
        
        # Fix: Windows activation state tracking
        self._was_minimized = False
        
        # State
        self.engine: Optional[PairsTradingEngine] = None
        self.selected_pair: Optional[str] = None  # For analytics tab
        self.signal_selected_pair: Optional[str] = None  # For signals tab (separate to avoid conflicts)
        self.portfolio: List[Dict] = []
        self.trade_history: List[Dict] = []  # Closed positions history
        self.worker_thread: Optional[QThread] = None
        self.worker = None  # Fix #8: Track worker for cleanup
        self.current_mini_futures: Dict = {}
        self.current_strategy: Dict = {}  # Optimal SL params
        self.current_mf_positions: Dict = {}
        self._is_scheduled_scan: bool = False  # Track if current scan is scheduled
        self._scheduled_scan_running: bool = False  # Guard mot dubbla scheduled scans (race condition fix)
        # Portfolio history manager
        self.portfolio_history = None
        if PORTFOLIO_HISTORY_AVAILABLE:
            try:
                self.portfolio_history = PortfolioHistoryManager(PORTFOLIO_HISTORY_FILE)
                print(f"Portfolio history: {self.portfolio_history.get_snapshot_count()} snapshots")
            except Exception as e:
                print(f"Portfolio history error: {e}")

        self._portfolio_file_mtime: float = 0  # Track portfolio file modification time
        self._engine_cache_mtime: float = 0  # Track engine cache file modification time
        
        # Fix #2 & #8: Track async workers for cleanup
        self._sync_worker: Optional[SyncWorker] = None
        self._sync_thread: Optional[QThread] = None
        self._sync_running = False  # Säker flagga
        
        # OPTIMERING: Track async market data workers med säkra flaggor
        self._market_watch_worker: Optional[MarketWatchWorker] = None
        self._market_watch_thread: Optional[QThread] = None
        self._market_watch_running = False  # Säker flagga istället för isRunning()
        self._market_data_cache = {}  # Cache for treemap click detail popups
        self._startup_complete = False  # Förhindra market watch innan startup är klar

        # WebSocket live streaming
        self._ws_worker: Optional[MarketWatchWebSocket] = None
        self._ws_thread: Optional[QThread] = None
        self._ws_cache_dirty = False
        self._ws_changed_symbols = set()
        self._ws_tick_history = {}          # symbol → [(timestamp, price), ...]
        self._ws_treemap_rendered = False   # True after first treemap render from WS
        self._ws_extra_info = {}            # symbol → {day_high, day_low, volume, prev_close, open}
        self._intraday_ohlc_seed = {}       # symbol → [[ts, O, H, L, C], ...] from yf.download
        self._intraday_worker = None
        self._intraday_thread = None
        self._intraday_retry_count = 0     # Retry-räknare för ofullständig OHLC-data
        self._intraday_max_retries = 3     # Max antal retries

        # Per-pair z-score chart widgets in positions table (uses historical data)
        self._lm_pairs: Dict[str, dict] = {}   # pair -> {plot_widget, z_curve, entry_line}

        self._volatility_worker: Optional[VolatilityDataWorker] = None
        self._volatility_thread: Optional[QThread] = None
        self._volatility_running = False  # Säker flagga
        # Cachade historiska serier för live-percentilberäkning (sparas vid yf.download)
        self._vol_hist_cache: Dict[str, np.ndarray] = {}  # ticker → sorterad numpy-array
        self._vol_median_cache: Dict[str, float] = {}     # ticker → median
        self._vol_mode_cache: Dict[str, float] = {}       # ticker → mode
        self._vol_sparkline_cache: Dict[str, list] = {}   # ticker → sparkline-värden
        self._portfolio_refresh_worker: Optional[PortfolioRefreshWorker] = None
        self._portfolio_refresh_thread: Optional[QThread] = None
        self._portfolio_refresh_running = False  # Säker flagga

        # Markov chain analysis state
        self._markov_thread: Optional[QThread] = None
        self._markov_worker = None
        self._markov_running = False
        self._markov_result = None

        # OPTIMERING: Metrics initieras som None (skapas i lazy-loaded tabs)
        self.tickers_metric: Optional[QFrame] = None
        self.pairs_metric: Optional[QFrame] = None
        self.viable_metric: Optional[QFrame] = None
        self.positions_metric: Optional[QFrame] = None
        
        # OPTIMERING: Tab containers för lazy loading
        self._tabs_loaded: Dict[int, bool] = {}
        self._tab_containers: Dict[int, QWidget] = {}
        
        # Setup UI
        self.setup_ui()
        
        # Load cached data
        QTimer.singleShot(100, self.load_initial_data)
        
        # =====================================================================
        # INDEPENDENT REFRESH TIMERS
        # Treemap: WebSocket live (no periodic yf.download)
        # Portfolio: periodic refresh every 60 min
        # News: fully independent (uses Ticker.news endpoint), 15 min
        # Watchdog: force-resets stuck thread flags, 30s check interval
        # =====================================================================

        # Portfolio refresh: every 5 minutes (z-scores + MF prices)
        self.auto_refresh_timer = QTimer(self)
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_data)
        self.auto_refresh_timer.start(300000)  # 5 minutes
        
        # Portfolio & engine cache sync timer (90 seconds) - for Google Drive sync
        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self.sync_from_drive)
        self.sync_timer.start(90000)  # 90 seconds

        # Check for scheduled scan every minute (22:00 weekdays)
        self.schedule_timer = QTimer(self)
        self.schedule_timer.timeout.connect(self.check_scheduled_scan)
        self.schedule_timer.start(60000)  # 1 minute

        # News feed: every 15 minutes (defers if yfinance busy)
        self.news_refresh_timer = QTimer(self)
        self.news_refresh_timer.timeout.connect(self._refresh_news_feed_safe)
        self.news_refresh_timer.start(900000)  # 15 minutes
        
        # Watchdog: detect and recover from stuck threads (30s interval)
        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.timeout.connect(self._watchdog_check)
        self._watchdog_timer.start(30000)  # 30 seconds

        # WebSocket: batch-render treemap updates every 5 seconds
        self._ws_render_timer = QTimer(self)
        self._ws_render_timer.timeout.connect(self._render_ws_updates)
        self._ws_render_timer.start(5000)

        # Apply initial dynamic layout after window is shown
        QTimer.singleShot(200, self._apply_dynamic_layout)
    
    # =========================================================================
    # WINDOW ACTIVATION FIX (Windows-specific)
    # =========================================================================
    
    def changeEvent(self, event):
        """Handle window state changes - fixes Windows taskbar activation issues.
        
        On Windows, clicking the taskbar icon when the window is minimized doesn't
        always restore and activate the window properly. This handler ensures the
        window is properly restored and brought to front.
        """
        if event.type() == event.WindowStateChange:
            # Track if we were minimized
            if self.windowState() & Qt.WindowMinimized:
                self._was_minimized = True
            elif self._was_minimized:
                # We were minimized and now we're not - force activation
                self._was_minimized = False
                # Use timer to ensure this happens after the state change completes
                QTimer.singleShot(0, self._force_activate_window)
        
        super().changeEvent(event)

    def closeEvent(self, event):
        """Clean up WebSocket and threads on app close."""
        self._stop_market_websocket()
        if self._intraday_thread is not None and self._intraday_thread.isRunning():
            self._intraday_thread.quit()
            self._intraday_thread.wait(2000)
        super().closeEvent(event)

    def _force_activate_window(self):
        """Force window activation on Windows.
        
        Windows has quirks where activateWindow() alone doesn't work.
        This method uses multiple techniques to ensure the window is visible.
        """
        # Ensure window is in normal state (not minimized)
        if self.windowState() & Qt.WindowMinimized:
            self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        
        # Show the window
        self.showNormal()
        
        # Raise to top of window stack
        self.raise_()
        
        # Activate and give focus
        self.activateWindow()
        self.setFocus(Qt.ActiveWindowFocusReason)
    
    # =========================================================================
    # DYNAMIC LAYOUT - Resize handling for different window sizes
    # =========================================================================
    
    def resizeEvent(self, event):
        """Handle window resize - dynamically adjust layout elements."""
        super().resizeEvent(event)
        
        # Get new window size
        new_width = event.size().width()
        new_height = event.size().height()
        
        # Debounce: only update if size changed significantly (> 50px)
        if not hasattr(self, '_last_resize_width'):
            self._last_resize_width = 0
            self._last_resize_height = 0
        
        width_changed = abs(new_width - self._last_resize_width) > 50
        height_changed = abs(new_height - self._last_resize_height) > 50
        
        if width_changed or height_changed:
            self._last_resize_width = new_width
            self._last_resize_height = new_height
            
            # Use timer to debounce rapid resize events
            if not hasattr(self, '_resize_timer'):
                self._resize_timer = QTimer(self)
                self._resize_timer.setSingleShot(True)
                self._resize_timer.timeout.connect(self._apply_dynamic_layout)
            
            self._resize_timer.start(150)  # 150ms debounce
    
    def _apply_dynamic_layout(self):
        """Apply layout changes based on current window size.
        
        Dynamically adjusts:
        - News feed width
        - Volatility card heights
        - Table row heights and font sizes
        - Chart heights
        - Metric card sizes
        - General spacing and fonts
        """
        try:
            width = self.width()
            height = self.height()
            
            # Calculate scale factor based on window width
            # Reference: 1920px = 1.0 scale factor
            scale = width / 1920.0
            scale = max(0.7, min(1.4, scale))  # Clamp between 0.7 and 1.4
            
            # Define layout parameters for each category
            if width < 1400:
                category = "small"
                params = {
                    'news_min': 250, 'news_max': 300,
                    'vol_card_height': 145,
                    'table_row_height': 26,
                    'table_font': 11,
                    'header_font': 12,
                    'metric_value_font': 18,
                    'metric_label_font': 10,
                    'chart_height': 200,
                    'spacing': 4,
                }
            elif width < 2000:
                category = "medium"
                params = {
                    'news_min': 300, 'news_max': 380,
                    'vol_card_height': 165,
                    'table_row_height': 30,
                    'table_font': 13,
                    'header_font': 13,
                    'metric_value_font': 22,
                    'metric_label_font': 12,
                    'chart_height': 250,
                    'spacing': 5,
                }
            elif width < 3000:
                category = "large"
                params = {
                    'news_min': 350, 'news_max': 450,
                    'vol_card_height': 175,
                    'table_row_height': 34,
                    'table_font': 14,
                    'header_font': 14,
                    'metric_value_font': 26,
                    'metric_label_font': 13,
                    'chart_height': 300,
                    'spacing': 6,
                }
            else:
                category = "xlarge"
                params = {
                    'news_min': 400, 'news_max': 520,
                    'vol_card_height': 190,
                    'table_row_height': 38,
                    'table_font': 15,
                    'header_font': 15,
                    'metric_value_font': 28,
                    'metric_label_font': 14,
                    'chart_height': 350,
                    'spacing': 8,
                }
            
            # Store current params for other methods to use
            self._layout_params = params
            self._layout_scale = scale
            
            # ─── NEWS FEED ───
            if hasattr(self, 'news_feed') and self.news_feed:
                self.news_feed.setMinimumWidth(params['news_min'])
                self.news_feed.setMaximumWidth(params['news_max'])
            
            # ─── VOLATILITY CARDS ───
            for card_name in ['vix_card', 'vvix_card', 'skew_card', 'move_card']:
                if hasattr(self, card_name):
                    card = getattr(self, card_name)
                    if card:
                        card.setMinimumHeight(params['vol_card_height'])
            
            # ─── TABLES ───
            # Portfolio table
            if hasattr(self, 'portfolio_table') and self.portfolio_table:
                self._apply_table_scaling(self.portfolio_table, params)
            
            # Signals table
            if hasattr(self, 'signals_table') and self.signals_table:
                self._apply_table_scaling(self.signals_table, params)
            
            # Pairs table
            if hasattr(self, 'pairs_table') and self.pairs_table:
                self._apply_table_scaling(self.pairs_table, params)
            
            # Market watch tables
            for table_name in ['us_table', 'eu_table', 'asia_table', 'commodities_table', 'crypto_table', 'macro_table']:
                if hasattr(self, table_name):
                    table = getattr(self, table_name)
                    if table:
                        self._apply_table_scaling(table, params)
            
            # ─── METRIC CARDS ───
            self._apply_metric_scaling(params)
            
            # ─── SECTION HEADERS ───
            self._apply_header_scaling(params)
            
            # ─── CHARTS / PLOTS ───
            self._apply_chart_scaling(params)
            
            # ─── HEADER BAR (logo, title, clocks) ───
            self._apply_header_bar_scaling(scale)
            
            # ─── VOLATILITY SPARKLINES ───
            self._apply_volatility_sparkline_scaling(scale)
            
            # ─── NEWS ITEMS ───
            self._apply_news_item_scaling(scale)

            # ─── OU ANALYTICS METRIC CARDS ───
            self._apply_ou_metric_scaling(scale)
            
            # Log for debugging (only when category changes)
            if not hasattr(self, '_last_layout_category') or self._last_layout_category != category:
                self._last_layout_category = category
                print(f"[Layout] Window {width}x{height} -> {category} (scale: {scale:.2f})")
                
        except Exception as e:
            print(f"[Layout] Error applying dynamic layout: {e}")
            traceback.print_exc()
    
    def _apply_table_scaling(self, table: QTableWidget, params: dict):
        """Apply scaling to a table widget."""
        try:
            # Row height
            table.verticalHeader().setDefaultSectionSize(params['table_row_height'])
            
            # Font
            font = table.font()
            font.setPointSize(params['table_font'])
            table.setFont(font)
            
            # Header font
            header = table.horizontalHeader()
            header_font = header.font()
            header_font.setPointSize(params['header_font'])
            header.setFont(header_font)
            
        except Exception as e:
            pass  # Silently ignore table scaling errors
    
    def _apply_metric_scaling(self, params: dict):
        """Apply scaling to metric cards."""
        try:
            # Find all metric cards and update their font sizes
            metric_cards = [
                ('tickers_metric', 'Tickers'),
                ('pairs_metric', 'Pairs'),
                ('viable_metric', 'Viable'),
                ('positions_metric', 'Positions'),
            ]
            
            for attr_name, _ in metric_cards:
                if hasattr(self, attr_name):
                    card = getattr(self, attr_name)
                    if card:
                        # Find value and label widgets within the card
                        for child in card.findChildren(QLabel):
                            text = child.text()
                            # Value labels are typically numeric or short
                            if text.isdigit() or (len(text) < 5 and not text.islower()):
                                font = child.font()
                                font.setPointSize(params['metric_value_font'])
                                child.setFont(font)
                            else:
                                # Label
                                font = child.font()
                                font.setPointSize(params['metric_label_font'])
                                child.setFont(font)
        except Exception as e:
            pass  # Silently ignore metric scaling errors
    
    def _apply_header_scaling(self, params: dict):
        """Apply scaling to section headers."""
        try:
            scale = self._layout_scale if hasattr(self, '_layout_scale') else 1.0
            # Find SectionHeader widgets and update via scale_to
            for child in self.findChildren(QLabel):
                if child.objectName() == "sectionHeader":
                    if hasattr(child, 'scale_to'):
                        child.scale_to(scale)
                    else:
                        # Fallback for headers without scale_to
                        font = child.font()
                        font.setPointSize(params['header_font'])
                        child.setFont(font)
        except Exception as e:
            pass  # Silently ignore header scaling errors
    
    def _apply_chart_scaling(self, params: dict):
        """Apply scaling to pyqtgraph chart widgets."""
        try:
            # Main price/zscore plots
            chart_configs = [
                ('ou_price_plot', params['chart_height']),
                ('ou_zscore_plot', int(params['chart_height'] * 0.5)),
                ('ou_path_plot', int(params['chart_height'] * 0.5)),
                ('ou_dist_plot', int(params['chart_height'] * 0.5)),
                ('signal_price_plot', params['chart_height']),
                ('signal_zscore_plot', int(params['chart_height'] * 0.5)),
            ]
            
            for attr_name, height in chart_configs:
                if hasattr(self, attr_name):
                    chart = getattr(self, attr_name)
                    if chart:
                        chart.setMinimumHeight(height)
                        
        except Exception as e:
            pass  # Silently ignore chart scaling errors
    
    def _apply_header_bar_scaling(self, scale: float):
        """Apply scaling to header bar with logo, title and clocks."""
        try:
            if hasattr(self, 'header_bar') and self.header_bar:
                self.header_bar.scale_to(scale)
        except Exception as e:
            pass  # Silently ignore header bar scaling errors
    
    def _apply_volatility_sparkline_scaling(self, scale: float):
        """Apply scaling to volatility cards including their sparklines."""
        try:
            for card_name in ['vix_card', 'vvix_card', 'skew_card', 'move_card']:
                if hasattr(self, card_name):
                    card = getattr(self, card_name)
                    if card and hasattr(card, 'scale_to'):
                        card.scale_to(scale)
        except Exception as e:
            pass  # Silently ignore volatility card scaling errors
    
    def _apply_news_item_scaling(self, scale: float):
        """Apply scaling to the news feed widget and all news items."""
        try:
            if hasattr(self, 'news_feed') and self.news_feed:
                if hasattr(self.news_feed, 'scale_to'):
                    self.news_feed.scale_to(scale)
        except Exception as e:
            pass  # Silently ignore news item scaling errors
    
    def _apply_ou_metric_scaling(self, scale: float):
        """Apply scaling to OU analytics metric cards."""
        try:
            # OU parameter cards
            ou_cards = [
                'ou_theta_card', 'ou_mu_card', 'ou_halflife_card',
                'ou_zscore_card', 'ou_hedge_card', 'ou_status_card',
                # Expected move cards
                'exp_spread_change_card', 'exp_y_only_card', 'exp_x_only_card',
                'exp_y_half_card', 'exp_x_half_card',
            ]
            
            for card_name in ou_cards:
                if hasattr(self, card_name):
                    card = getattr(self, card_name)
                    if card and hasattr(card, 'scale_to'):
                        card.scale_to(scale)
        except Exception as e:
            pass  # Silently ignore OU metric scaling errors
    
    def auto_refresh_data(self):
        """Auto-refresh portfolio (every 60 min). Treemap uses WebSocket live data.
        Volatility percentiles refreshed independently at startup.
        """
        if not self._startup_complete:
            return

        try:
            self._auto_refresh_portfolio()
            self.last_updated_label.setText(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"[AutoRefresh] Error: {e}")
    
    def _watchdog_check(self):
        """Watchdog: reset stuck flags so the next timer cycle isn't blocked."""
        now = time.time()

        # Volatility: reset flag if thread died
        if self._volatility_running:
            elapsed = now - getattr(self, '_volatility_last_start', now)
            thread = self._volatility_thread
            if thread is not None and not thread.isRunning():
                print(f"[Watchdog] Volatility flag stuck (thread dead, {elapsed:.0f}s) — resetting")
                self._volatility_running = False
            elif elapsed > 120:
                print(f"[Watchdog] Volatility flag reset ({elapsed:.0f}s)")
                self._volatility_running = False
    
    def _auto_refresh_portfolio(self):
        """Auto-refresh Z-scores and MF prices asynchronously (no popups).
        
        OPTIMERING: Flyttar HTTP-anrop till bakgrundstrad for att undvika GUI-frysning.
        """
        if not self.portfolio:
            return
        
        # Don't start if already running - använd säker flagga
        if self._portfolio_refresh_running:
            return
        
        # Sätt flagga INNAN vi skapar tråden
        self._portfolio_refresh_running = True
        
        # Create and start worker
        self._portfolio_refresh_thread = QThread()
        self._portfolio_refresh_worker = PortfolioRefreshWorker(
            self.portfolio, 
            self.engine, 
            MF_PRICE_SCRAPING_AVAILABLE
        )
        self._portfolio_refresh_worker.moveToThread(self._portfolio_refresh_thread)
        
        self._portfolio_refresh_thread.started.connect(self._portfolio_refresh_worker.run)
        self._portfolio_refresh_worker.finished.connect(self._portfolio_refresh_thread.quit)
        # VIKTIGT: Återställ flagga och rensa referenser när tråden är klar
        self._portfolio_refresh_thread.finished.connect(self._on_portfolio_refresh_thread_finished)
        self._portfolio_refresh_worker.result.connect(self._on_portfolio_refresh_received)
        self._portfolio_refresh_worker.error.connect(self._on_portfolio_refresh_error)
        
        self._portfolio_refresh_thread.start()
    
    def _on_portfolio_refresh_thread_finished(self):
        """Handle portfolio refresh thread completion - clear references and flag."""
        self._portfolio_refresh_running = False
        
        # Använd deleteLater() för säker cleanup
        if self._portfolio_refresh_worker is not None:
            self._portfolio_refresh_worker.deleteLater()
        if self._portfolio_refresh_thread is not None:
            self._portfolio_refresh_thread.deleteLater()
        
        self._portfolio_refresh_worker = None
        self._portfolio_refresh_thread = None
    
    def _on_portfolio_refresh_received(self, updated_portfolio: list, z_count: int, mf_count: int):
        """Handle refreshed portfolio data - runs on GUI thread (safe).
        
        MERGE updates into current portfolio instead of replacing.
        Prevents race condition: worker snapshots portfolio, user closes position
        while worker runs, worker finishes → would re-add closed position.
        """
        try:
            # Build lookup from worker results
            updated_lookup = {p['pair']: p for p in updated_portfolio}
            
            # Only update z-scores/prices for positions that STILL exist
            for pos in self.portfolio:
                pair = pos['pair']
                if pair in updated_lookup:
                    up = updated_lookup[pair]
                    if 'current_z' in up:
                        pos['previous_z'] = pos.get('current_z', pos['entry_z'])
                        pos['current_z'] = up['current_z']
                    for key in ['mf_current_price_y', 'mf_current_price_x']:
                        if key in up and up[key]:
                            pos[key] = up[key]
            
            self._save_and_sync_portfolio()
            # OPTIMERING: Uppdatera endast om Portfolio-tabben är laddad
            if self._tabs_loaded.get(4, False):
                self.update_portfolio_display()
            
        except Exception as e:
            pass
    
    def _on_portfolio_refresh_error(self, error: str):
        """Handle portfolio refresh error."""
        pass
    
    def check_scheduled_scan(self):
        """Check if it's time to run scheduled scan (22:00 weekdays)."""
        now = datetime.now()
        
        # Logg varje timme för att bekräfta att timern körs
        if now.minute == 0 and now.second < 60:
            print(f"[SCHEDULE CHECK] Timer running at {now.strftime('%Y-%m-%d %H:%M:%S')} (weekday: {now.weekday()}, target hour: {SCHEDULED_HOUR})")
        
        # ROBUST GUARD: Förhindra dubbla körningar under samma scan-session
        # Denna flagga sätts FÖRST och kontrolleras FÖRST för att undvika race conditions
        # (QApplication.processEvents() kan trigga pending timer events under lazy loading)
        if getattr(self, '_scheduled_scan_running', False):
            return  # Scan redan igång, hoppa över
        
        # More robust check: trigger if we're AT or PAST the scheduled time (within a 5-minute window)
        # This handles cases where the timer misses exactly minute 0
        is_weekday = now.weekday() < 5
        is_scheduled_hour = now.hour == SCHEDULED_HOUR
        is_within_window = SCHEDULED_MINUTE <= now.minute < SCHEDULED_MINUTE + 5  # 5-minute window
        
        if is_weekday and is_scheduled_hour and is_within_window:
            if not hasattr(self, '_last_scheduled_run'):
                self._last_scheduled_run = None
            if self._last_scheduled_run != now.strftime('%Y-%m-%d'):
                # SÄTT FLAGGAN DIREKT INNAN NÅGOT ANNAT för att förhindra race conditions
                self._scheduled_scan_running = True
                print(f"[SCHEDULE] *** TRIGGERING SCHEDULED SCAN at {now.strftime('%H:%M:%S')} ***")
                print(f"[SCHEDULE] Weekday: {now.weekday()}, Hour: {now.hour}, Minute: {now.minute}")
                self._last_scheduled_run = now.strftime('%Y-%m-%d')
                self.statusBar().showMessage("Starting scheduled scan...")
                
                # Set flag for scheduled scan (used by callbacks)
                self._scheduled_snapshot_pending = True
                
                # First refresh MF prices asynchronously
                if MF_PRICE_SCRAPING_AVAILABLE and self.portfolio:
                    print("[SCHEDULE] Refreshing MF prices before scan...")
                    self.refresh_mf_prices()
                    # Snapshot will be taken in _on_mf_prices_received when prices arrive
                    # Then we start the scan after a short delay to let prices settle
                    QTimer.singleShot(3000, self._run_scheduled_scan_after_prices)
                else:
                    # No MF prices to fetch, take snapshot and run scan immediately
                    print("[SCHEDULE] No MF prices needed, running scan directly...")
                    self._take_daily_portfolio_snapshot()
                    self.run_scheduled_scan()
    
    def _run_scheduled_scan_after_prices(self):
        """Run scheduled scan after MF prices have been fetched."""
        # Take snapshot with updated prices
        if self._scheduled_snapshot_pending:
            self._take_daily_portfolio_snapshot()
            self._scheduled_snapshot_pending = False
        # Run the actual scan
        self.run_scheduled_scan()

    def _take_daily_portfolio_snapshot(self):
        """Take daily portfolio snapshot (async-safe version)."""
        if not PORTFOLIO_HISTORY_AVAILABLE or self.portfolio_history is None:
            return
        if not self.portfolio:
            return
        
        # If MF prices available, refresh them first (now async), then take snapshot via callback
        if MF_PRICE_SCRAPING_AVAILABLE and hasattr(self, '_price_worker'):
            # Prices are being fetched async - snapshot will be taken with current prices
            # For scheduled scans, we trigger a separate snapshot after prices arrive
            pass
        
        # Take snapshot with current prices (may be slightly stale, but won't block GUI)
        try:
            snapshot = self.portfolio_history.take_snapshot(self.portfolio)
            if snapshot:
                self.statusBar().showMessage(f"Snapshot: {snapshot.n_positions} pos, {snapshot.unrealized_pnl_pct:+.2f}%")
        except Exception as e:
            print(f"Snapshot error: {e}")

    def run_scheduled_scan(self):
        """Run the full scheduled scan: load CSV, analyze pairs, send to Discord."""
        try:
            print(f"[SCHEDULED SCAN] === Starting scheduled scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            # Check if CSV path exists
            if not os.path.exists(SCHEDULED_CSV_PATH):
                self.statusBar().showMessage(f"Scheduled scan failed: CSV not found at {SCHEDULED_CSV_PATH}")
                self.send_discord_notification(
                    title="⚠️ Scheduled Scan Failed",
                    description=f"CSV file not found: {SCHEDULED_CSV_PATH}",
                    color=0xff0000  # Red
                )
                return
            
            # Load tickers from CSV
            tickers = load_tickers_from_csv(SCHEDULED_CSV_PATH)
            if not tickers:
                self.statusBar().showMessage("Scheduled scan failed: No tickers in CSV")
                self.send_discord_notification(
                    title="⚠️ Scheduled Scan Failed",
                    description="No tickers found in CSV file",
                    color=0xff0000
                )
                return
            
            self.statusBar().showMessage(f"Scheduled scan: Loading {len(tickers)} tickers...")
            print(f"[SCHEDULED SCAN] Loaded {len(tickers)} tickers from CSV")
            
            # VIKTIGT: Se till att Arbitrage Scanner-tabben är laddad (lazy loading fix)
            # Förbättrad version med retry-logik
            max_retries = 3
            for attempt in range(max_retries):
                if not self._tabs_loaded.get(1, False):
                    print(f"[SCHEDULED SCAN] Arbitrage Scanner tab not loaded - loading (attempt {attempt + 1}/{max_retries})...")
                    self._on_tab_changed(1)
                    # Processa events flera gånger för att säkerställa widget-skapande
                    for _ in range(5):
                        QApplication.processEvents()
                        time.sleep(0.05)  # Kort paus för att låta widgets skapas
                    print(f"[SCHEDULED SCAN] Tab loaded status: {self._tabs_loaded.get(1, False)}")
                
                # Kontrollera om widgets finns
                has_tickers = hasattr(self, 'tickers_input') and self.tickers_input is not None
                has_btn = hasattr(self, 'run_btn') and self.run_btn is not None
                
                if has_tickers and has_btn:
                    print(f"[SCHEDULED SCAN] Widgets verified on attempt {attempt + 1}")
                    break
                elif attempt < max_retries - 1:
                    print(f"[SCHEDULED SCAN] Widgets not ready, waiting... (tickers_input: {has_tickers}, run_btn: {has_btn})")
                    time.sleep(0.5)  # Vänta lite längre innan nästa försök
            
            # Verify critical widgets exist after all attempts
            if not hasattr(self, 'tickers_input') or self.tickers_input is None:
                error_msg = "tickers_input widget not created after retries - lazy loading failed"
                print(f"[SCHEDULED SCAN ERROR] {error_msg}")
                self.send_discord_notification(
                    title="⚠️ Scheduled Scan Failed",
                    description=f"Internal error: {error_msg}",
                    color=0xff0000
                )
                return
            
            if not hasattr(self, 'run_btn') or self.run_btn is None:
                error_msg = "run_btn widget not created after retries - lazy loading failed"
                print(f"[SCHEDULED SCAN ERROR] {error_msg}")
                self.send_discord_notification(
                    title="⚠️ Scheduled Scan Failed", 
                    description=f"Internal error: {error_msg}",
                    color=0xff0000
                )
                return
            
            # Update the tickers input field
            self.tickers_input.setText(', '.join(tickers))

            # Force Extended preset for scheduled scans (max data points)
            self.lookback_combo.setCurrentText("Extended (500-2500)")
            print(f"[SCHEDULED SCAN] Tickers set in input field, using Extended preset, starting analysis...")

            # Store that this is a scheduled scan so we can send Discord after completion
            self._is_scheduled_scan = True

            # Run analysis (will call on_analysis_complete when done)
            self.run_analysis()
            print(f"[SCHEDULED SCAN] Analysis started successfully")
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"[SCHEDULED SCAN ERROR] {error_details}")
            self.statusBar().showMessage(f"Scheduled scan error: {e}")
            self._scheduled_scan_running = False  # Reset guard flag vid error
            self._is_scheduled_scan = False
            self.send_discord_notification(
                title="⚠️ Scheduled Scan Error",
                description=f"Error: {str(e)}\n\nDetails logged to console.",
                color=0xff0000
            )
    
    def send_discord_notification(self, title: str, description: str, color: int = 0xff6b00, 
                                   fields: list = None, footer: str = None):
        """Send notification to Discord webhook ASYNCHRONOUSLY (prevents GUI freeze)."""
        if not SCRAPING_AVAILABLE:
            print("Discord notification skipped: requests not available")
            return
        
        # Robust guard: skip if webhook is empty, placeholder, or not a valid URL
        if not DISCORD_WEBHOOK_URL or \
           "YOUR_WEBHOOK" in DISCORD_WEBHOOK_URL or \
           not DISCORD_WEBHOOK_URL.startswith("https://"):
            print("Discord notification skipped: webhook not configured")
            return
        
        # Build the embed payload
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": footer or "Klippinge Investment Trading Terminal"}
        }
        
        if fields:
            embed["fields"] = fields
        
        payload = {"embeds": [embed]}
        
        # Run in background thread to prevent GUI freeze
        self._discord_thread = QThread()
        self._discord_worker = DiscordWorker(DISCORD_WEBHOOK_URL, payload)
        self._discord_worker.moveToThread(self._discord_thread)
        
        self._discord_thread.started.connect(self._discord_worker.run)
        self._discord_worker.finished.connect(self._discord_thread.quit)
        self._discord_worker.finished.connect(self._discord_worker.deleteLater)
        self._discord_thread.finished.connect(self._discord_thread.deleteLater)
        self._discord_worker.success.connect(lambda msg: print(msg))
        self._discord_worker.error.connect(lambda msg: print(msg))
        
        self._discord_thread.start()
    
    def send_scan_results_to_discord(self):
        """Send pair scan results to Discord."""
        if self.engine is None:
            return
        
        try:
            # Build the message
            n_tickers = len(self.engine.price_data.columns)
            n_pairs = len(self.engine.pairs_stats) if self.engine.pairs_stats is not None else 0
            n_viable = len(self.engine.viable_pairs) if self.engine.viable_pairs is not None else 0
            
            # Build fields for embed
            fields = [
                {"name": "Tickers Analyzed", "value": str(n_tickers), "inline": True},
                {"name": "Pairs Tested", "value": str(n_pairs), "inline": True},
                {"name": "Viable Pairs", "value": str(n_viable), "inline": True},
            ]
            
            # === LIVE Portfolio status (calculated from current self.portfolio) ===
            if self.portfolio:
                live_pnl_fields = self._calculate_live_portfolio_for_discord()
                fields.extend(live_pnl_fields)

            # Calculate z-scores for viable pairs and find signals
            if self.engine.viable_pairs is not None and len(self.engine.viable_pairs) > 0:
                pairs_with_z = []
                
                # Fix #5: Use itertuples instead of iterrows
                for row in self.engine.viable_pairs.itertuples():
                    pair = row.pair
                    try:
                        ou, spread, z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                        if ou is not None:
                            pairs_with_z.append({
                                'pair': pair,
                                'z': z,
                                'half_life': getattr(row, 'half_life_days', 0)
                            })
                    except Exception as e:
                        print(f"Error getting z-score for {pair}: {e}")
                        continue
                
                # Filter pairs with |z| >= threshold and sort by |z| descending
                signal_pairs = [p for p in pairs_with_z if abs(p['z']) >= SIGNAL_TAB_THRESHOLD]
                signal_pairs.sort(key=lambda x: abs(x['z']), reverse=True)
                
                # Take top 5 signals
                top_signals = signal_pairs[:5]
                
                if top_signals:
                    pairs_text = ""
                    for p in top_signals:
                        z = p['z']
                        pairs_text += f"**{p['pair']}** | Spread Z-score: {z:.2f}\n"
                    
                    fields.append({
                        "name": f"🎯 Signals (|Z| ≥ {SIGNAL_TAB_THRESHOLD})",
                        "value": pairs_text,
                        "inline": False
                    })
                else:
                    fields.append({
                        "name": "🎯 Signals",
                        "value": f"No pairs with |Z| ≥ {SIGNAL_TAB_THRESHOLD}",
                        "inline": False
                    })
            
            # Send the notification
            self.send_discord_notification(
                title="📈 Daily Scan Complete",
                description=f"Analyzed {n_tickers} tickers → {n_viable} viable pairs found",
                color=0x3b82f6,
                fields=fields,
                footer="Klippinge Investment Trading Terminal"
            )
            
        except Exception as e:
            print(f"Error sending scan results to Discord: {e}")
            traceback.print_exc()
    
    def _calculate_live_portfolio_for_discord(self) -> list:
        """Calculate LIVE portfolio P&L from current self.portfolio for Discord.
        
        This ensures Discord shows the same data as the dashboard, not stale snapshot data.
        """
        fields = []
        
        try:
            total_pnl = 0.0
            total_invested = 0.0
            position_texts = []
            
            for pos in self.portfolio:
                if pos.get('status', 'OPEN') != 'OPEN':
                    continue
                
                pair = pos.get('pair', 'Unknown')
                direction = pos.get('direction', 'LONG')
                
                # FIX: Hämta AKTUELL z-score istället för lagrad (synkar med Signals)
                current_z = pos.get('current_z', 0.0)  # Fallback
                if self.engine is not None:
                    try:
                        _, _, live_z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                        current_z = live_z
                        # Uppdatera även positionen för konsistens
                        pos['current_z'] = current_z
                    except Exception as e:
                        print(f"[Discord] Could not get live z-score for {pair}: {e}")
                
                # Calculate P&L for Y leg (long leg)
                entry_y = pos.get('mf_entry_price_y', 0.0)
                current_y = pos.get('mf_current_price_y', entry_y)
                qty_y = pos.get('mf_qty_y', 0)
                
                pnl_y = 0.0
                invested_y = 0.0
                if entry_y > 0 and qty_y > 0:
                    pnl_y = (current_y - entry_y) * qty_y
                    invested_y = entry_y * qty_y
                
                # Calculate P&L for X leg (short/hedge leg)
                entry_x = pos.get('mf_entry_price_x', 0.0)
                current_x = pos.get('mf_current_price_x', entry_x)
                qty_x = pos.get('mf_qty_x', 0)
                
                pnl_x = 0.0
                invested_x = 0.0
                if entry_x > 0 and qty_x > 0:
                    pnl_x = (current_x - entry_x) * qty_x
                    invested_x = entry_x * qty_x
                
                # Total for this position
                pos_pnl = pnl_y + pnl_x
                pos_invested = invested_y + invested_x
                
                total_pnl += pos_pnl
                total_invested += pos_invested
                
                # Position summary text
                if pos_invested > 0:
                    pos_pnl_pct = (pos_pnl / pos_invested) * 100
                    position_texts.append(f"**{pair}**: {pos_pnl_pct:+.2f}% (Z-score: {current_z:.2f})")
            
            # Total P&L calculation
            if total_invested > 0:
                total_pnl_pct = (total_pnl / total_invested) * 100
                total_value = total_invested + total_pnl
                
                pnl_emoji = "📈" if total_pnl >= 0 else "📉"
                
                fields.append({
                    "name": f"{pnl_emoji} Portfolio Status",
                    "value": (
                        f"Total P&L: **{total_pnl_pct:+.2f}%** ({total_pnl:+,.0f} SEK)\n"
                        f"Positions: {len([p for p in self.portfolio if p.get('status') == 'OPEN'])}*2\n"
                        f"Value: {total_value:,.0f} SEK"
                    ),
                    "inline": False
                })
                
                # Open positions breakdown
                if position_texts:
                    fields.append({
                        "name": "📋 Open Positions",
                        "value": "\n".join(position_texts[:5]),  # Max 5 to keep Discord message short
                        "inline": False
                    })
            else:
                fields.append({
                    "name": "📋 Portfolio Status",
                    "value": "No open positions",
                    "inline": False
                })
                
        except Exception as e:
            print(f"Error calculating live portfolio for Discord: {e}")
            fields.append({
                "name": "📋 Portfolio Status",
                "value": f"Error: {str(e)[:50]}",
                "inline": False
            })
        
        return fields

    def _manual_portfolio_snapshot(self):
        if not PORTFOLIO_HISTORY_AVAILABLE or self.portfolio_history is None:
            QMessageBox.warning(self, "Not Available", "Portfolio history not available.")
            return
        if not self.portfolio:
            QMessageBox.information(self, "No Positions", "No positions to snapshot.")
            return
        if MF_PRICE_SCRAPING_AVAILABLE:
            self.refresh_mf_prices()
        snapshot = self.portfolio_history.take_snapshot(self.portfolio)
        if snapshot:
            QMessageBox.information(self, "Snapshot", f"Saved: {snapshot.n_positions} pos, {snapshot.total_value:,.0f} SEK, {snapshot.unrealized_pnl_pct:+.2f}%")
    
    def _show_portfolio_history(self):
        if not PORTFOLIO_HISTORY_AVAILABLE or self.portfolio_history is None:
            QMessageBox.warning(self, "Not Available", "Portfolio history not available.")
            return
        count = self.portfolio_history.get_snapshot_count()
        if count == 0:
            QMessageBox.information(self, "No History", "No snapshots yet. Take one from Analysis menu.")
            return
        perf = self.portfolio_history.get_performance_summary()
        if not perf:
            QMessageBox.information(self, "Insufficient Data", f"{count} snapshots, need 2+ for comparison.")
            return
        table = format_performance_table(perf)
        latest = self.portfolio_history.get_latest_snapshot()
        info = f"\n\nLatest: {latest.timestamp[:16]}, {latest.total_value:,.0f} SEK, {latest.unrealized_pnl_pct:+.2f}%" if latest else ""
        QMessageBox.information(self, "Performance", f"Portfolio vs S&P 500\n{'='*40}\n\n{table}\n\nSnapshots: {count}{info}")


    def _manual_portfolio_snapshot(self):
        if not PORTFOLIO_HISTORY_AVAILABLE or self.portfolio_history is None:
            QMessageBox.warning(self, "Not Available", "Portfolio history not available.")
            return
        if not self.portfolio:
            QMessageBox.information(self, "No Positions", "No positions to snapshot.")
            return
        if MF_PRICE_SCRAPING_AVAILABLE:
            self.refresh_mf_prices()
        snapshot = self.portfolio_history.take_snapshot(self.portfolio)
        if snapshot:
            QMessageBox.information(self, "Snapshot", f"Saved: {snapshot.n_positions} pos, {snapshot.total_value:,.0f} SEK, {snapshot.unrealized_pnl_pct:+.2f}%")

    def _show_portfolio_history(self):
        if not PORTFOLIO_HISTORY_AVAILABLE or self.portfolio_history is None:
            QMessageBox.warning(self, "Not Available", "Portfolio history not available.")
            return
        count = self.portfolio_history.get_snapshot_count()
        if count == 0:
            QMessageBox.information(self, "No History", "No snapshots yet. Take one from Analysis menu.")
            return
        perf = self.portfolio_history.get_performance_summary()
        if not perf:
            QMessageBox.information(self, "Insufficient Data", f"{count} snapshots, need 2+ for comparison.")
            return
        table = format_performance_table(perf)
        latest = self.portfolio_history.get_latest_snapshot()
        info = f"\n\nLatest: {latest.timestamp[:16]}, {latest.total_value:,.0f} SEK, {latest.unrealized_pnl_pct:+.2f}%" if latest else ""
        QMessageBox.information(self, "Performance", f"Portfolio vs S&P 500\n{'='*40}\n\n{table}\n\nSnapshots: {count}{info}")

    def test_discord_notification(self):
        """Send a test notification to Discord to verify webhook configuration."""
        self.send_discord_notification(
            title="🧪 Test Notification",
            description="This is a test notification from Klippinge Investment Terminal.",
            color=0x00c853,  # Green
            fields=[
                {"name": "Status", "value": "✅ Discord webhook is working!", "inline": False},
                {"name": "Configuration", "value": f"CSV: {SCHEDULED_CSV_PATH}", "inline": False},
                {"name": "Schedule", "value": f"Daily at {SCHEDULED_HOUR:02d}:{SCHEDULED_MINUTE:02d} (weekdays)", "inline": False},
            ],
            footer="Test notification"
        )
        self.statusBar().showMessage("Test notification sent to Discord")
    
    def setup_ui(self):
        """Setup the main UI."""
        # Menu bar with updated styling
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{ 
                background-color: {COLORS['bg_darkest']}; 
                border-bottom: 1px solid {COLORS['border_subtle']}; 
                padding: 2px;
            }}
            QMenuBar::item {{ 
                padding: 6px 16px; 
                color: {COLORS['text_secondary']}; 
                border-radius: 4px;
            }}
            QMenuBar::item:selected {{ 
                background-color: {COLORS['bg_hover']}; 
                color: {COLORS['accent']}; 
            }}
            QMenu {{ 
                background-color: {COLORS['bg_elevated']}; 
                border: 1px solid {COLORS['border_default']}; 
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{ 
                padding: 8px 24px; 
                border-radius: 4px;
            }}
            QMenu::item:selected {{ 
                background-color: rgba(212, 165, 116, 0.15); 
                color: {COLORS['accent']}; 
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {COLORS['border_subtle']};
                margin: 4px 8px;
            }}
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_csv_action = QAction("Load Tickers CSV", self)
        load_csv_action.triggered.connect(self.load_csv)
        file_menu.addAction(load_csv_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        
        run_analysis_action = QAction("Run Pair Analysis", self)
        run_analysis_action.triggered.connect(self.run_analysis)
        analysis_menu.addAction(run_analysis_action)
        
        analysis_menu.addSeparator()

        scheduled_scan_action = QAction("Run Scheduled Scan Now", self)
        scheduled_scan_action.triggered.connect(self.run_scheduled_scan)
        analysis_menu.addAction(scheduled_scan_action)
        
        test_discord_action = QAction("Test Discord Notification", self)
        test_discord_action.triggered.connect(self.test_discord_notification)
        analysis_menu.addAction(test_discord_action)
        
        snapshot_action = QAction("Take Portfolio Snapshot", self)
        snapshot_action.triggered.connect(self._manual_portfolio_snapshot)
        analysis_menu.addAction(snapshot_action)

        history_action = QAction("View Portfolio History", self)
        history_action.triggered.connect(self._show_portfolio_history)
        analysis_menu.addAction(history_action)
        
        # Data menu
        data_menu = menubar.addMenu("Data")
        
        refresh_market_action = QAction("Refresh Market Watch", self)
        refresh_market_action.setShortcut("F5")
        refresh_market_action.triggered.connect(self.refresh_market_watch)
        data_menu.addAction(refresh_market_action)
        
        refresh_vol_action = QAction("Refresh Volatility Data", self)
        refresh_vol_action.triggered.connect(self.refresh_market_data)
        data_menu.addAction(refresh_vol_action)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Professional Header Bar with clocks
        self.header_bar = HeaderBar()
        main_layout.addWidget(self.header_bar)
        
        # Content area with padding
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 15, 20, 15)
        content_layout.setSpacing(15)  # Ökat från 10 för mer luft
        
        # Main tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        
        # OPTIMERING: Lazy loading med container-widgets
        # Spara vilka tabbar som är laddade
        self._tabs_loaded = {
            0: True,   # Market Overview - laddas direkt (starttab)
            1: False,  # Arbitrage Scanner
            2: False,  # OU Analytics
            3: False,  # Pair Signals (tung - MS scraping)
            4: False,  # Portfolio
            5: False,  # Markov Chains
        }
        
        # Spara container-widgets för lazy loading
        self._tab_containers = {}
        
        # Tab 0: Market Overview - laddas direkt (starttab)
        self.tabs.addTab(self.create_market_overview_tab(), "|◎| MARKET OVERVIEW")
        
        # Tab 1-4: Containers med placeholders som fylls on-demand
        self._tab_containers[1] = self._create_lazy_container("ARBITRAGE SCANNER")
        self.tabs.addTab(self._tab_containers[1], "|◊| ARBITRAGE SCANNER")
        
        self._tab_containers[2] = self._create_lazy_container("OU ANALYTICS")
        self.tabs.addTab(self._tab_containers[2], "|∂x| OU ANALYTICS")
        
        self._tab_containers[3] = self._create_lazy_container("PAIR SIGNALS")
        self.tabs.addTab(self._tab_containers[3], "|⧗| PAIR SIGNALS")
        
        self._tab_containers[4] = self._create_lazy_container("PORTFOLIO")
        self.tabs.addTab(self._tab_containers[4], "|≡| PORTFOLIO")

        self._tab_containers[5] = self._create_lazy_container("MARKOV CHAINS")
        self.tabs.addTab(self._tab_containers[5], "|⛓| MARKOV CHAINS")

        # Koppla signal för lazy loading
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        content_layout.addWidget(self.tabs)
        main_layout.addWidget(content_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_darkest']};
                border-top: 1px solid {COLORS['border_subtle']};
                color: {COLORS['text_muted']};
                font-size: 10px;
                padding: 4px;
            }}
        """)
        self.status_time = QLabel(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.status_time.setStyleSheet(f"color: {COLORS['text_muted']}; font-family: 'JetBrains Mono', monospace;")
        self.statusBar().addPermanentWidget(self.status_time)
        
        # Update time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
    
    def _create_lazy_container(self, tab_name: str) -> QWidget:
        """Create a container widget for lazy-loaded tabs.
        
        OPTIMERING: Skapar en container med placeholder som ersätts med riktigt innehåll.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Placeholder som visas tills tabben laddas
        placeholder = QLabel(f"⏳ Loading {tab_name}...")
        placeholder.setStyleSheet(f"""
            color: {COLORS['text_muted']}; 
            font-size: 14px;
            padding: 40px;
        """)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setObjectName("placeholder")
        layout.addWidget(placeholder)
        
        return container
    
    def _create_compact_metric_card(self, label: str, value: str = "-") -> 'CompactMetricCard':
        """Create a compact metric card for dense layouts like Pair Signals."""
        return CompactMetricCard(label, value)
    
    def _on_tab_changed(self, index: int):
        """Handle tab change - load tab content on demand.
        
        OPTIMERING: Lazy loading - laddar tabbinnehåll i befintlig container.
        """
        if self._tabs_loaded.get(index, False):
            return  # Redan laddad
        
        if index not in self._tab_containers:
            return  # Inte en lazy-loaded tab
        
        self.statusBar().showMessage(f"Loading tab...")
        QApplication.processEvents()  # Visa statusmeddelande direkt
        
        # Skapa rätt innehåll baserat på index
        tab_creators = {
            1: self.create_arbitrage_scanner_tab,
            2: self.create_ou_analytics_tab,
            3: self.create_signals_tab,
            4: self.create_portfolio_tab,
            5: self.create_markov_chains_tab,
        }
        
        if index in tab_creators:
            container = self._tab_containers[index]
            layout = container.layout()
            
            # Ta bort placeholder
            placeholder = container.findChild(QLabel, "placeholder")
            if placeholder:
                placeholder.deleteLater()
            
            # Skapa det riktiga innehållet
            content = tab_creators[index]()
            
            # Flytta innehållet till containern
            # (content är en QWidget, vi tar dess layout och barn)
            content_layout = content.layout()
            if content_layout:
                # Flytta alla widgets från content till container
                while content_layout.count():
                    item = content_layout.takeAt(0)
                    if item.widget():
                        layout.addWidget(item.widget())
                    elif item.layout():
                        layout.addLayout(item.layout())
            
            # Markera som laddad
            self._tabs_loaded[index] = True
            
            # Ladda data för tabben om engine finns
            if self.engine is not None:
                if index == 1:  # Arbitrage Scanner
                    self.update_viable_table()
                    self.update_all_pairs_table()
                elif index == 2:  # OU Analytics
                    self.update_ou_pair_list()
                elif index == 3:  # Pair Signals
                    self.update_signals_list()
            
            # Portfolio behöver alltid uppdateras
            if index == 4:  # Portfolio
                self.update_portfolio_display()
            
            self.statusBar().showMessage("Tab loaded")
    
    # ========================================================================
    # TAB 1: MARKET OVERVIEW
    # ========================================================================
    
    def create_market_overview_tab(self) -> QWidget:
        """Create Market Overview tab with treemap heatmap + vol cards + news feed."""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # CENTER: Treemap heatmap + Volatility cards below
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)

        # Treemap container
        map_container = QFrame()
        map_container.setStyleSheet("background-color: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 4px;")
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(0)

        # Use QWebEngineView for Plotly treemap
        if WEBENGINE_AVAILABLE and QWebEngineView is not None:
            self.map_widget = QWebEngineView()
            # Enable local file:// pages to load remote https:// resources (CDN)
            settings = self.map_widget.settings()
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
            # Capture JS console messages and intercept treemap clicks via console.log
            main_window_ref = self

            class _TreemapPage(QWebEnginePage):
                def __init__(self, main_win, parent=None):
                    super().__init__(parent)
                    self._main_window = main_win

                def javaScriptConsoleMessage(self, level, msg, line, src):
                    # Intercept click signals sent via console.log
                    if msg.startswith('TREEMAP_CLICK:'):
                        ticker = msg[len('TREEMAP_CLICK:'):]
                        if hasattr(self._main_window, '_on_treemap_click'):
                            self._main_window._on_treemap_click(ticker)
                        return
                    print(f'[Treemap JS] {msg} (line {line})')

            treemap_page = _TreemapPage(main_window_ref, self.map_widget)
            self.map_widget.setPage(treemap_page)
            self.map_widget.setStyleSheet("background-color: #0a0a0a;")
            self.map_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.update_treemap_heatmap([])
            map_layout.addWidget(self.map_widget, stretch=1)
        else:
            map_placeholder = QLabel("Treemap Heatmap\n(Install PyQtWebEngine)")
            map_placeholder.setAlignment(Qt.AlignCenter)
            map_placeholder.setStyleSheet("background-color: #111; color: #444;")
            map_layout.addWidget(map_placeholder, stretch=1)

        # Last updated label
        self.last_updated_label = QLabel(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.last_updated_label.setStyleSheet("color: #444444; font-size: 10px; border: none; background: transparent; padding: 2px 5px;")
        self.last_updated_label.setAlignment(Qt.AlignRight)
        map_layout.addWidget(self.last_updated_label)

        center_layout.addWidget(map_container, stretch=1)

        # Volatility Cards (4 horizontal cards below treemap)
        vol_section = QWidget()
        vol_section_layout = QVBoxLayout(vol_section)
        vol_section_layout.setContentsMargins(0, 0, 0, 0)
        vol_section_layout.setSpacing(8)

        vol_header = QLabel("VOLATILITY & RISK")
        vol_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 14px; font-weight: 700; letter-spacing: 2px; padding-top: 6px; border-top: 2px solid {COLORS['accent']};")
        vol_section_layout.addWidget(vol_header)

        vol_cards_row = QHBoxLayout()
        vol_cards_row.setSpacing(10)

        self.vix_card = VolatilityCard("^VIX", "VIX", "Fear Index")
        vol_cards_row.addWidget(self.vix_card)

        self.vvix_card = VolatilityCard("^VVIX", "VVIX", "Vol of Vol")
        vol_cards_row.addWidget(self.vvix_card)

        self.skew_card = VolatilityCard("^SKEW", "SKEW", "Tail Risk")
        vol_cards_row.addWidget(self.skew_card)

        self.move_card = VolatilityCard("^MOVE", "MOVE", "Bond Vol")
        vol_cards_row.addWidget(self.move_card)

        vol_section_layout.addLayout(vol_cards_row)
        center_layout.addWidget(vol_section)

        center_panel.setMinimumWidth(600)
        main_layout.addWidget(center_panel, stretch=7)

        # RIGHT: News Feed
        self.news_feed = NewsFeedWidget()
        self.news_feed.setMinimumWidth(300)
        self.news_feed.setMaximumWidth(380)
        self.news_feed.refresh_btn.clicked.connect(self._refresh_news_feed)
        main_layout.addWidget(self.news_feed, stretch=3)
        
        return tab
    
    def _refresh_news_feed(self):
        """Refresh the news feed using tickers from CSV file."""
        if hasattr(self, 'news_feed') and self.news_feed:
            # Use SCHEDULED_CSV_PATH which contains all tickers
            self.news_feed.refresh_news(SCHEDULED_CSV_PATH)
    
    def _refresh_news_feed_safe(self):
        """Refresh news feed — defers if yfinance price operations are active.
        
        News uses yfinance Ticker.news with ThreadPoolExecutor(10 workers).
        Running that in parallel with yf.download(threads=10) = 20+ connections
        which overwhelms yfinance and causes massive timeouts.
        
        If busy, simply skips — the chain from _on_volatility_thread_finished
        or the next 15-min timer tick will retry.
        """
        if self._market_watch_running or self._volatility_running:
            print("[News] Skipped — yfinance busy (market watch or volatility running)")
            return
        
        # Also check threads physically alive
        if (self._market_watch_thread is not None and self._market_watch_thread.isRunning()):
            print("[News] Skipped — market watch thread still alive")
            return
        if (self._volatility_thread is not None and self._volatility_thread.isRunning()):
            print("[News] Skipped — volatility thread still alive")
            return
        
        self._last_news_fetch_time = time.time()
        self._refresh_news_feed()
    
    def _stat_label(self, text: str) -> QLabel:
        """Create a stat label."""
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent;")
        return lbl
    
    def _stat_value(self, text: str) -> QLabel:
        """Create a stat value label."""
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', monospace; background: transparent;")
        return lbl
    
    # ========================================================================
    # TAB 2: ARBITRAGE SCANNER
    # ========================================================================
    
    def create_arbitrage_scanner_tab(self) -> QWidget:
        """Create Arbitrage Scanner tab with professional layout."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # TOP: Configuration Panel
        config_frame = QFrame()
        config_frame.setStyleSheet("""
            QFrame {
                background-color: #111111;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
            }
        """)
        config_layout = QHBoxLayout(config_frame)
        config_layout.setContentsMargins(15, 12, 15, 12)
        config_layout.setSpacing(15)
        
        # Tickers input
        ticker_group = QVBoxLayout()
        ticker_label = QLabel("TICKERS:")
        ticker_label.setStyleSheet("color: #d4a574; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 6px; border: none;")
        ticker_group.addWidget(ticker_label)
        self.tickers_input = QLineEdit()
        self.tickers_input.setPlaceholderText("Enter tickers comma-separated or load from CSV")
        self.tickers_input.setText("PEP, KO, PG, CL, GS, XLF, BX, KKR, NVDA, TSM")
        self.tickers_input.setMinimumWidth(200)
        self.tickers_input.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; padding: 6px; border-radius: 4px;")
        ticker_group.addWidget(self.tickers_input)
        config_layout.addLayout(ticker_group)
        
        # Windows preset
        lookback_group = QVBoxLayout()
        lookback_label = QLabel("WINDOWS:")
        lookback_label.setStyleSheet("color: #d4a574; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 6px; border: none;")
        lookback_group.addWidget(lookback_label)
        self.lookback_combo = QComboBox()
        self.lookback_combo.addItems(["Standard (500-2000)", "Quick (750-1500)", "Extended (500-2500)"])
        self.lookback_combo.setCurrentIndex(0)
        self.lookback_combo.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; padding: 6px; min-width: 80px;")
        lookback_group.addWidget(self.lookback_combo)
        config_layout.addLayout(lookback_group)
        
        config_layout.addStretch()
        
        # Buttons
        self.load_csv_btn = QPushButton("📊 LOAD CSV")
        self.load_csv_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                border: 1px solid #333;
                padding: 15px 20px;
                color: #888;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
        """)
        self.load_csv_btn.clicked.connect(self.load_csv)
        config_layout.addWidget(self.load_csv_btn)
        
        self.run_btn = QPushButton("▶ RUN ANALYSIS")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #d4a574;
                border: none;
                padding: 15px 25px;
                color: #000;
                font-weight: 600;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #ff8533;
            }
        """)
        self.run_btn.clicked.connect(self.run_analysis)
        config_layout.addWidget(self.run_btn)
        
        main_layout.addWidget(config_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(True)
        self.progress_bar.setMinimumHeight(10)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: none;
            }
            QProgressBar::chunk {
                background-color: #d4a574;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # MIDDLE: Split layout - Metrics left, Results right
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # LEFT: Summary Statistics
        stats_panel = QFrame()
        stats_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
            }}
        """)
        stats_panel.setMinimumWidth(220)
        stats_panel.setMaximumWidth(280)
        stats_layout = QVBoxLayout(stats_panel)
        stats_layout.setContentsMargins(18, 18, 18, 18)
        stats_layout.setSpacing(12)
        
        stats_header = QLabel("SCAN SUMMARY")
        stats_header.setStyleSheet(f"color: {COLORS['accent']}; border: none; font-size: 16px; font-weight: 700; letter-spacing: 1.5px;")
        stats_layout.addWidget(stats_header)
        
        # Summary metrics (vertical stack)
        self.tickers_metric = self._create_summary_metric("TICKERS LOADED", "0")
        stats_layout.addWidget(self.tickers_metric)
        
        self.pairs_metric = self._create_summary_metric("PAIRS ANALYZED", "0")
        stats_layout.addWidget(self.pairs_metric)
        
        self.viable_metric = self._create_summary_metric("VIABLE PAIRS", "0")
        stats_layout.addWidget(self.viable_metric)
        
        self.positions_metric = self._create_summary_metric("OPEN POSITIONS", "0")
        stats_layout.addWidget(self.positions_metric)
        
        # Criteria info
        criteria_header = QLabel("VIABILITY CRITERIA")
        criteria_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 14px; border: none; font-weight: 700; letter-spacing: 1px; padding-top: 12px;")
        stats_layout.addWidget(criteria_header)
        
        criteria_frame = QFrame()
        criteria_frame.setStyleSheet(f"background-color: {COLORS['bg_elevated']}; border: none; border-radius: 4px;")
        criteria_inner = QVBoxLayout(criteria_frame)
        criteria_inner.setContentsMargins(12, 10, 12, 10)
        criteria_inner.setSpacing(6)
        
        criteria_items = [
            "• Half-life: 5-60 days",
            "• Engle-Granger p-value: \u2264 0.05",
            "• Johansen trace \u2265 15.4943",
            "• Hurst exponent: \u2264 0.50",
            "• Correlation: \u2265 0.70",
            "\u2500\u2500 Kalman Validation \u2500\u2500",
            "• Param stability: > 0.50",
            "• Innovation ratio: [0.5, 2.0]",
            "• Regime score: < 4.0",
            "• \u03b8 significant at 95% CI",
        ]
        for item in criteria_items:
            lbl = QLabel(item)
            lbl.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent;")
            criteria_inner.addWidget(lbl)
        
        stats_layout.addWidget(criteria_frame)
        stats_layout.addStretch()
        
        content_layout.addWidget(stats_panel)
        
        # RIGHT: Results tables
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(10)
        
        # Viable Pairs section
        viable_header_layout = QHBoxLayout()
        self.viable_header = QLabel("VIABLE PAIRS")
        self.viable_header.setStyleSheet("color: #d4a574; font-size: 15px; font-weight: 600; letter-spacing: 1px; padding-top: 10px;")
        viable_header_layout.addWidget(self.viable_header)
        viable_header_layout.addStretch()
        results_layout.addLayout(viable_header_layout)
        
        # Viable Pairs table
        self.viable_table = QTableWidget()
        self.viable_table.verticalHeader().setVisible(False)
        self.viable_table.verticalHeader().setDefaultSectionSize(44)  # Öka radhöjd ytterligare

        self.viable_table.setColumnCount(12)
        self.viable_table.setHorizontalHeaderLabels([
            "Pair", "Z-Score", "Half-life (days)", "EG p-value", "Johansen trace",
            "Hurst", "Correlation", "Robustness",
            "Kalman Stab.", "Innov. Ratio", "Regime Score", "\u03b8 Sig."
        ])
        # Tooltips for scanner columns
        _scanner_tips = [
            "Stock pair Y/X. Spread = Y - \u03b2\u00b7X - \u03b1",
            "Current Z-score of the spread.\n|Z| > 2.0 = signal (highlighted).\nComputed from shortest passing window.",
            "Days for spread to move halfway to equilibrium.\nln(2)/\u03b8 \u00d7 252. Ideal: 5-60 days.",
            "Engle-Granger cointegration p-value.\np < 0.05 = significant mean reversion.",
            "Johansen trace statistic.\nHigher = stronger cointegration evidence.",
            "Hurst exponent. H < 0.5 = mean-reverting (good).\nH \u2248 0.5 = random walk. H > 0.5 = trending.",
            "Pearson correlation between Y and X.\nHigher = more stable hedge. > 0.8 desirable.",
            "Windows passed / windows tested.\nHigher = more robust across time periods.",
            "Kalman parameter stability [0-1].\n1 = perfectly stable \u03b8. > 0.5 required.",
            "Normalized innovation ratio.\nShould be \u22481.0 (valid range [0.5, 2.0]).",
            "CUSUM regime change score.\n< 4.0 = no structural break detected.",
            "Is \u03b8 (mean-reversion speed)\nstatistically significant at 95% CI?",
        ]
        for col, tip in enumerate(_scanner_tips):
            item = self.viable_table.horizontalHeaderItem(col)
            if item:
                item.setToolTip(tip)
        self.viable_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.viable_table.horizontalHeader().setMinimumSectionSize(100)
        self.viable_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.viable_table.setMinimumHeight(150)
        self.viable_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                gridline-color: {COLORS['border_subtle']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: {TYPOGRAPHY['table_cell']}px;
            }}
            QTableWidget::item {{
                padding: 10px 12px;
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(212, 165, 116, 0.15);
                color: {COLORS['accent_bright']};
            }}
            QTableWidget::item:hover {{
                background-color: {COLORS['bg_hover']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_secondary']};
                padding: 12px 10px;
                border: none;
                border-bottom: 2px solid {COLORS['border_default']};
                font-weight: 600;
                text-transform: uppercase;
                font-size: {TYPOGRAPHY['table_header']}px;
                letter-spacing: 0.5px;
            }}
        """)
        self.viable_table.itemSelectionChanged.connect(self._on_scanner_pair_clicked)
        self.viable_table.doubleClicked.connect(self._on_scanner_pair_double_clicked)
        results_layout.addWidget(self.viable_table)

        # Window breakdown panel (hidden by default)
        self.window_detail_frame = QFrame()
        self.window_detail_frame.setVisible(False)
        self.window_detail_frame.setMaximumHeight(280)
        self.window_detail_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
            }}
        """)
        wd_layout = QVBoxLayout(self.window_detail_frame)
        wd_layout.setContentsMargins(10, 8, 10, 8)
        wd_layout.setSpacing(4)

        self.window_detail_header = QLabel("WINDOW BREAKDOWN")
        self.window_detail_header.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 12px; font-weight: 600; "
            f"letter-spacing: 1px; background: transparent; border: none;")
        wd_layout.addWidget(self.window_detail_header)

        self.window_detail_table = QTableWidget()
        self.window_detail_table.setColumnCount(9)
        self.window_detail_table.setHorizontalHeaderLabels([
            "Window", "Status", "Half-life", "EG p-val", "Johansen",
            "Hurst", "Corr", "Kalman Stab", "Failed At"
        ])
        self.window_detail_table.verticalHeader().setVisible(False)
        self.window_detail_table.verticalHeader().setDefaultSectionSize(28)
        self.window_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.window_detail_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.window_detail_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.window_detail_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                gridline-color: {COLORS['border_subtle']};
                border: none;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 4px 6px;
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_secondary']};
                padding: 6px 4px;
                border: none;
                border-bottom: 1px solid {COLORS['border_default']};
                font-weight: 600;
                font-size: 10px;
            }}
        """)
        wd_layout.addWidget(self.window_detail_table)
        results_layout.addWidget(self.window_detail_frame)

        # All Pairs expandable
        self.all_pairs_btn = QPushButton("▼ View All Analyzed Pairs")
        self.all_pairs_btn.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 8px 15px;
                background-color: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_default']};
                color: {COLORS['text_muted']};
                font-size: {TYPOGRAPHY['body_small']}px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.all_pairs_btn.clicked.connect(self.toggle_all_pairs)
        results_layout.addWidget(self.all_pairs_btn)
        
        # All pairs table (hidden by default)
        self.all_pairs_table = QTableWidget()
        self.all_pairs_table.setVisible(False)
        self.all_pairs_table.verticalHeader().setDefaultSectionSize(36)  # Öka radhöjd
        self.all_pairs_table.setColumnCount(6)
        headers = [
            "Pair", "Half-life (days)", "Engle-Granger p-value", "Johansen trace", "Hurst exponent", "Correlation"
                   ]
        self.all_pairs_table.setHorizontalHeaderLabels(headers)
        self.all_pairs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.all_pairs_table.setMaximumHeight(600)
        results_layout.addWidget(self.all_pairs_table)
        
        results_layout.addStretch()
        content_layout.addWidget(results_panel)
        
        main_layout.addLayout(content_layout)
        
        return tab
    
    def _create_summary_metric(self, label: str, value: str, highlight: bool = False) -> QFrame:
        """Create a summary metric widget for arbitrage scanner."""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {'#0f1a0f' if highlight else '#111111'};
                border: 1px solid {'#1a3a1a' if highlight else '#2a2a2a'};
                border-left: 3px solid {'#00c853' if highlight else '#d4a574'};
                border-radius: 4px;
            }}
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #666; font-size: 11px; letter-spacing: 1px; background: transparent; border: none")
        layout.addWidget(label_widget)
        
        value_widget = QLabel(value)
        value_widget.setObjectName("metricValue")
        value_widget.setStyleSheet(f"color: {'#00c853' if highlight else '#fff'}; border: none; font-size: 20px; font-weight: 600; background: transparent;")
        layout.addWidget(value_widget)
        
        return frame
    
    # ========================================================================
    # TAB 3: OU ANALYTICS
    # ========================================================================
    
    def create_ou_analytics_tab(self) -> QWidget:
        """Create OU Analytics tab with cards on left, plots on right."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Pair selector row (top)
        selector_layout = QHBoxLayout()
        selector_layout.setSpacing(15)
        selector_label = QLabel("Select Pair:")
        selector_label.setStyleSheet("color: #888; font-size: 11px;")
        selector_layout.addWidget(selector_label)
        
        self.ou_pair_combo = QComboBox()
        self.ou_pair_combo.setMinimumWidth(200)
        self.ou_pair_combo.setEditable(True)
        self.ou_pair_combo.setInsertPolicy(QComboBox.NoInsert)
        self.ou_pair_combo.completer().setFilterMode(Qt.MatchContains)
        self.ou_pair_combo.completer().setCompletionMode(QCompleter.PopupCompletion)
        self.ou_pair_combo.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; padding: 6px;")
        self.ou_pair_combo.lineEdit().setPlaceholderText("Search pair...")
        self.ou_pair_combo.currentTextChanged.connect(self.on_ou_pair_changed)
        selector_layout.addWidget(self.ou_pair_combo)
        
        self.viable_only_check = QCheckBox("Viable only")
        self.viable_only_check.setChecked(True)
        self.viable_only_check.stateChanged.connect(self.update_ou_pair_list)
        selector_layout.addWidget(self.viable_only_check)
        
        selector_layout.addStretch()
        main_layout.addLayout(selector_layout)
        
        # Main content: Splitter with cards (left) and plots (right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border_subtle']};
            }}
        """)
        
        # =====================================================================
        # LEFT SIDE: Metric Cards (one card per row)
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(5, 5, 10, 5)
        
        
        # =========================
        # OU Parameters
        # =========================
        left_layout.addWidget(SectionHeader("OU DETAILS"))
        
        self.ou_theta_card = MetricCard("MEAN REVERSION", "-",
            tooltip="θ — Speed of mean reversion (annualized).\nHigher = faster reversion. Typical: 5-100.")
        self.ou_mu_card = MetricCard("MEAN SPREAD", "-",
            tooltip="μ — Long-term equilibrium spread level.\nDerived from Kalman state: a/(1-b).")
        self.ou_halflife_card = MetricCard("HALF-LIFE", "-",
            tooltip="Trading days to move halfway to equilibrium.\nln(2)/θ × 252. Ideal for trading: 5-60 days.")
        self.ou_zscore_card = MetricCard("CURRENT Z-SCORE", "-",
            tooltip="Std deviations from equilibrium: (S-μ)/σ_eq.\n|z|>2 = entry signal. Positive = above mean.")
        self.ou_hedge_card = MetricCard("BETA", "-",
            tooltip="Hedge ratio β from EG regression: Y = α+β·X+ε.\nFor 1 share Y, short β shares X.")
        self.ou_status_card = MetricCard("STATUS", "-",
            tooltip="VIABLE = passes all tests:\n• EG p<0.05 • Hurst<0.5 • HL 1-252d")
        
        left_layout.addWidget(self.ou_theta_card)
        left_layout.addWidget(self.ou_mu_card)
        left_layout.addWidget(self.ou_halflife_card)
        left_layout.addWidget(self.ou_zscore_card)
        left_layout.addWidget(self.ou_hedge_card)
        left_layout.addWidget(self.ou_status_card)
        
        
        # =========================
        # Kalman Filter Diagnostics
        # =========================
        left_layout.addWidget(SectionHeader("KALMAN FILTER"))
        
        self.kalman_stability_card = MetricCard("PARAM STABILITY", "-",
            tooltip="θ stability over last 60 days (1-CV).\n>0.7 good, 0.4-0.7 caution, <0.4 unstable.")
        self.kalman_ess_card = MetricCard("EFFECTIVE N", "-",
            tooltip="Effective sample size adjusted for autocorrelation.\nLower than N = redundant observations.")
        self.kalman_theta_ci_card = MetricCard("θ (95% CI)", "-",
            tooltip="95% CI for half-life (days) from Kalman covariance.\n'inf' = cannot exclude random walk (θ≈0).")
        self.kalman_mu_ci_card = MetricCard("μ (95% CI)", "-",
            tooltip="95% CI for equilibrium spread level.\nWide CI = uncertain reversion target.")
        self.kalman_innovation_card = MetricCard("INNOV. RATIO", "-",
            tooltip="Actual/expected innovation variance. Should ≈ 1.0.\n>1.5 = model underestimates uncertainty.")
        self.kalman_regime_card = MetricCard("REGIME CHANGE", "-",
            tooltip="CUSUM on innovations. Threshold=4.0.\nHigh = structural break, parameters unreliable.")
        
        left_layout.addWidget(self.kalman_stability_card)
        left_layout.addWidget(self.kalman_ess_card)
        left_layout.addWidget(self.kalman_theta_ci_card)
        left_layout.addWidget(self.kalman_mu_ci_card)
        left_layout.addWidget(self.kalman_innovation_card)
        left_layout.addWidget(self.kalman_regime_card)
        
        
        # =========================
        # Expected Move
        # =========================
        left_layout.addWidget(SectionHeader("EXPECTED MOVE"))
        
        self.exp_spread_change_card = MetricCard("Δ SPREAD", "-",
            tooltip="Expected spread change over 1 half-life.\nBased on OU: E[S_t] = μ + (S₀-μ)·e^(-θt).")
        self.exp_y_only_card = MetricCard("Y (100%)", "-",
            tooltip="Y price move if Y absorbs 100% of convergence.")
        self.exp_x_only_card = MetricCard("X (100%)", "-",
            tooltip="X price move if X absorbs 100% of convergence.")
        
        left_layout.addWidget(self.exp_spread_change_card)
        left_layout.addWidget(self.exp_y_only_card)
        left_layout.addWidget(self.exp_x_only_card)

        # =========================
        # Window Robustness
        # =========================
        left_layout.addWidget(SectionHeader("WINDOW ROBUSTNESS"))

        self.ou_robustness_card = MetricCard("ROBUSTNESS SCORE", "-",
            tooltip="Windows passed / windows tested.\nGreen ≥70%, Yellow ≥50%, Red <50%.")
        left_layout.addWidget(self.ou_robustness_card)

        self.ou_window_container = QWidget()
        self.ou_window_container_layout = QVBoxLayout(self.ou_window_container)
        self.ou_window_container_layout.setContentsMargins(0, 0, 0, 0)
        self.ou_window_container_layout.setSpacing(2)
        left_layout.addWidget(self.ou_window_container)

        # Push content to top
        left_layout.addStretch()
                
        # Wrap in scroll area
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {COLORS['bg_dark']};
                border: none;
            }}
        """)
        
        # =====================================================================
        # RIGHT SIDE: Plots
        # =====================================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(10, 5, 5, 5)
        
        if ensure_pyqtgraph():
            DateAxisItem = get_date_axis_item_class()
            
            # ===== ROW 1: Price comparison (full width, stretch 2) =====
            price_container = QWidget()
            price_vlayout = QVBoxLayout(price_container)
            price_vlayout.setSpacing(2)
            price_vlayout.setContentsMargins(0, 0, 0, 0)
            price_vlayout.addWidget(SectionHeader("BETA ADJUSTED PRICE COMPARISON"))
            
            self.ou_price_date_axis = DateAxisItem(orientation='bottom')
            self.ou_price_plot = pg.PlotWidget(axisItems={'bottom': self.ou_price_date_axis})
            self.ou_price_plot.setLabel('left', 'Price')
            self.ou_price_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_price_plot.addLegend()
            self.ou_price_plot.setMinimumHeight(200)
            self.ou_price_plot.setMouseEnabled(x=True, y=True)
            self.ou_price_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            price_vlayout.addWidget(self.ou_price_plot)
            
            # ===== ROW 2: Z-score + Expected Path + Conditional Distribution (stretch 1) =====
            bottom_container = QWidget()
            bottom_hlayout = QHBoxLayout(bottom_container)
            bottom_hlayout.setSpacing(10)
            bottom_hlayout.setContentsMargins(0, 0, 0, 0)
            
            # Spread plot (wider — stretch 2 in row)
            zscore_col = QVBoxLayout()
            zscore_col.addWidget(SectionHeader("SPREAD (Y − β·X − α)"))
            
            self.ou_zscore_date_axis = DateAxisItem(orientation='bottom')
            self.ou_zscore_plot = pg.PlotWidget(axisItems={'bottom': self.ou_zscore_date_axis})
            self.ou_zscore_plot.setLabel('left', 'Spread')
            self.ou_zscore_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_zscore_plot.setMinimumHeight(100)
            self.ou_zscore_plot.setMouseEnabled(x=True, y=True)
            self.ou_zscore_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            
            # SYNC: Bidirectional x-axis sync between price and zscore
            self._syncing_plots = False
            self._ou_syncing_plots = False
            
            def sync_price_to_zscore(viewbox, range):
                if not self._syncing_plots and not self._ou_syncing_plots:
                    self._syncing_plots = True
                    self.ou_zscore_plot.setXRange(range[0][0], range[0][1], padding=0)
                    self._syncing_plots = False
            
            def sync_zscore_to_price(viewbox, range):
                if not self._syncing_plots and not self._ou_syncing_plots:
                    self._syncing_plots = True
                    self.ou_price_plot.setXRange(range[0][0], range[0][1], padding=0)
                    self._syncing_plots = False
            
            self.ou_price_plot.sigRangeChanged.connect(sync_price_to_zscore)
            self.ou_zscore_plot.sigRangeChanged.connect(sync_zscore_to_price)
            
            zscore_col.addWidget(self.ou_zscore_plot)
            bottom_hlayout.addLayout(zscore_col, stretch=2)
            
            # Expected Path plot
            path_col = QVBoxLayout()
            path_col.addWidget(SectionHeader("EXPECTED PATH"))
            self.ou_path_plot = pg.PlotWidget()
            self.ou_path_plot.setLabel('left', 'Spread')
            self.ou_path_plot.setLabel('bottom', 'Days')
            self.ou_path_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_path_plot.setMinimumHeight(100)
            path_col.addWidget(self.ou_path_plot)
            bottom_hlayout.addLayout(path_col, stretch=1)
            
            # Conditional Distribution plot
            dist_col = QVBoxLayout()
            dist_col.addWidget(SectionHeader("CONDITIONAL DISTRIBUTION"))
            self.ou_dist_plot = pg.PlotWidget()
            self.ou_dist_plot.setLabel('left', 'Density (×0.001)')
            self.ou_dist_plot.setLabel('bottom', 'Spread')
            self.ou_dist_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_dist_plot.setMinimumHeight(100)
            dist_col.addWidget(self.ou_dist_plot)
            bottom_hlayout.addLayout(dist_col, stretch=1)
            
            # Add both rows with 2:1 height ratio
            right_layout.addWidget(price_container, stretch=2)
            right_layout.addWidget(bottom_container, stretch=1)
            
            # Initialize crosshair managers
            self.price_crosshair = None
            self.zscore_crosshair = None
            
        else:
            right_layout.addWidget(QLabel("Install pyqtgraph for charts"))
        
        # Add to splitter
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        
        # Set ratio 1:3 (left:right)
        splitter.setSizes([125, 875])
        
        main_layout.addWidget(splitter)
        
        return tab
    
    # ========================================================================
    # TAB 4: PAIR SIGNALS
    # ========================================================================
    
    def create_signals_tab(self) -> QWidget:
        """Create Pair Signals tab with professional institutional layout."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 15, 20, 15)
        
        # ===== TOP: Signal selector row =====
        top_frame = QFrame()
        top_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
            }}
        """)
        top_layout = QHBoxLayout(top_frame)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(20)
        
        # Signal count indicator
        self.signal_count_label = QLabel(f"⚡ 0 viable pairs with |Z| ≥ {SIGNAL_TAB_THRESHOLD}")
        self.signal_count_label.setStyleSheet(f"color: {COLORS['positive']}; font-size: 13px; font-weight: 500;")
        top_layout.addWidget(self.signal_count_label)
        
        top_layout.addStretch()
        
        # Signal selector
        selector_label = QLabel("Select Signal:")
        selector_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; border: none;")
        top_layout.addWidget(selector_label)
        
        self.signal_combo = QComboBox()
        self.signal_combo.setMinimumWidth(180)
        self.signal_combo.setMaximumWidth(250)
        self.signal_combo.currentTextChanged.connect(self.on_signal_changed)
        top_layout.addWidget(self.signal_combo)
        
        # Open Position button
        self.open_position_btn = QPushButton("📊 Open Position")
        self.open_position_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['positive']}, stop:1 #1b8a42);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ed573, stop:1 {COLORS['positive']});
            }}
            QPushButton:disabled {{
                background: {COLORS['bg_hover']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.open_position_btn.clicked.connect(lambda: self.open_position("Balanced"))
        self.open_position_btn.setEnabled(False)
        top_layout.addWidget(self.open_position_btn)
        
        layout.addWidget(top_frame)
        
        # ===== MAIN CONTENT: Two-column layout (2:3 ratio) =====
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # ===== LEFT COLUMN: Signal State, Position Sizing, Derivatives =====
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        left_panel = QFrame()
        left_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
            }}
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 14, 16, 14)
        left_layout.setSpacing(12)
        
        # Current Signal State header
        state_header = QLabel("CURRENT SIGNAL STATE")
        state_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 700; letter-spacing: 1px; border: none;")
        left_layout.addWidget(state_header)
        
        # Signal state metrics - grid layout
        state_grid = QGridLayout()
        state_grid.setSpacing(8)
        
        self.signal_pair_card = self._create_compact_metric_card("PAIR", "-")
        state_grid.addWidget(self.signal_pair_card, 0, 0)
        
        self.signal_z_card = self._create_compact_metric_card("CURRENT Z-SCORE", "-")
        state_grid.addWidget(self.signal_z_card, 0, 1)
        
        self.signal_dir_card = self._create_compact_metric_card("DIRECTION", "-")
        state_grid.addWidget(self.signal_dir_card, 0, 2)
        
        self.signal_hedge_card = self._create_compact_metric_card("HEDGE RATIO (β)", "-")
        state_grid.addWidget(self.signal_hedge_card, 1, 0)
        
        self.signal_hl_card = self._create_compact_metric_card("HALF-LIFE", "-")
        state_grid.addWidget(self.signal_hl_card, 1, 1)
        
        left_layout.addLayout(state_grid)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        divider.setMaximumHeight(1)
        left_layout.addWidget(divider)
        
        # Position Sizing header
        sizing_header = QLabel("POSITION SIZING")
        sizing_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 700; letter-spacing: 1px; border: none;")
        left_layout.addWidget(sizing_header)
        
        # Position sizing inputs - compact row
        sizing_layout = QHBoxLayout()
        sizing_layout.setSpacing(10)
        
        notional_group = QVBoxLayout()
        notional_group.setSpacing(2)
        notional_label = QLabel("Notional (SEK)")
        notional_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; border: none;")
        notional_group.addWidget(notional_label)
        self.notional_spin = QSpinBox()
        self.notional_spin.setRange(100, 10000000)
        self.notional_spin.setValue(3000)
        self.notional_spin.setSingleStep(500)
        self.notional_spin.setMinimumWidth(90)
        self.notional_spin.valueChanged.connect(self.update_position_sizing)
        notional_group.addWidget(self.notional_spin)
        sizing_layout.addLayout(notional_group)
        
        lev_y_group = QVBoxLayout()
        lev_y_group.setSpacing(2)
        self.lev_y_label = QLabel("Leverage Y")
        self.lev_y_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; border: none;")
        lev_y_group.addWidget(self.lev_y_label)
        self.leverage_y_spin = QDoubleSpinBox()
        self.leverage_y_spin.setRange(0.1, 10)
        self.leverage_y_spin.setValue(1.0)
        self.leverage_y_spin.setSingleStep(0.1)
        self.leverage_y_spin.setMinimumWidth(60)
        self.leverage_y_spin.valueChanged.connect(self.update_position_sizing)
        lev_y_group.addWidget(self.leverage_y_spin)
        sizing_layout.addLayout(lev_y_group)
        
        lev_x_group = QVBoxLayout()
        lev_x_group.setSpacing(2)
        self.lev_x_label = QLabel("Leverage X")
        self.lev_x_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; border: none")
        lev_x_group.addWidget(self.lev_x_label)
        self.leverage_x_spin = QDoubleSpinBox()
        self.leverage_x_spin.setRange(0.1, 10)
        self.leverage_x_spin.setValue(1.0)
        self.leverage_x_spin.setSingleStep(0.1)
        self.leverage_x_spin.setMinimumWidth(60)
        self.leverage_x_spin.valueChanged.connect(self.update_position_sizing)
        lev_x_group.addWidget(self.leverage_x_spin)
        sizing_layout.addLayout(lev_x_group)
        
        sizing_layout.addStretch()
        left_layout.addLayout(sizing_layout)
        
        # Position cards - Buy/Sell/Capital
        pos_layout = QHBoxLayout()
        pos_layout.setSpacing(8)
        
        # Buy card
        self.buy_frame = QFrame()
        self.buy_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_elevated']};
                border: none;
                border-left: 2px solid {COLORS['positive']};
                border-radius: 4px;
            }}
        """)
        buy_layout = QVBoxLayout(self.buy_frame)
        buy_layout.setContentsMargins(10, 8, 10, 8)
        buy_layout.setSpacing(3)
        self.buy_action_label = QLabel("BUY")
        self.buy_action_label.setStyleSheet(f"color: {COLORS['positive']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        self.buy_shares_label = QLabel("-")
        self.buy_shares_label.setStyleSheet(f"color: {COLORS['positive']}; font-size: 16px; font-weight: 700; background: transparent; border: none;")
        self.buy_price_label = QLabel("@ -")
        self.buy_price_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        buy_layout.addWidget(self.buy_action_label)
        buy_layout.addWidget(self.buy_shares_label)
        buy_layout.addWidget(self.buy_price_label)
        pos_layout.addWidget(self.buy_frame)
        
        # Sell card
        self.sell_frame = QFrame()
        self.sell_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_elevated']};
                border: none;
                border-left: 2px solid {COLORS['negative']};
                border-radius: 4px;
            }}
        """)
        sell_layout = QVBoxLayout(self.sell_frame)
        sell_layout.setContentsMargins(10, 8, 10, 8)
        sell_layout.setSpacing(3)
        self.sell_action_label = QLabel("SELL")
        self.sell_action_label.setStyleSheet(f"color: {COLORS['negative']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        self.sell_shares_label = QLabel("-")
        self.sell_shares_label.setStyleSheet(f"color: {COLORS['negative']}; font-size: 16px; font-weight: 700; background: transparent; border: none;")
        self.sell_price_label = QLabel("@ -")
        self.sell_price_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        sell_layout.addWidget(self.sell_action_label)
        sell_layout.addWidget(self.sell_shares_label)
        sell_layout.addWidget(self.sell_price_label)
        pos_layout.addWidget(self.sell_frame)
        
        # Capital required
        self.capital_frame = QFrame()
        self.capital_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_elevated']};
                border: none;
                border-left: 2px solid {COLORS['accent']};
                border-radius: 4px;
            }}
        """)
        capital_layout = QVBoxLayout(self.capital_frame)
        capital_layout.setContentsMargins(10, 8, 10, 8)
        capital_layout.setSpacing(3)
        cap_title = QLabel("CAPITAL REQUIRED")
        cap_title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        capital_layout.addWidget(cap_title)
        self.capital_label = QLabel("-")
        self.capital_label.setStyleSheet(f"color: {COLORS['accent_bright']}; font-size: 16px; font-weight: 700; background: transparent; border: none;")
        capital_layout.addWidget(self.capital_label)
        self.capital_beta_label = QLabel("")
        self.capital_beta_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        capital_layout.addWidget(self.capital_beta_label)
        pos_layout.addWidget(self.capital_frame)
        
        left_layout.addLayout(pos_layout)
        
        # Divider before derivatives
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.HLine)
        divider2.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        divider2.setMaximumHeight(1)
        left_layout.addWidget(divider2)
        
        # Derivatives header
        deriv_header = QLabel("DERIVATIVES")
        deriv_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 600; letter-spacing: 1px; border: none;")
        left_layout.addWidget(deriv_header)

        # Knapp-stilar för produkttyp-toggle (per ben)
        self._deriv_btn_style_active = f"""
            QPushButton {{
                background: {COLORS['accent']}; color: #000000;
                border: none; border-radius: 3px; padding: 2px 8px;
                font-size: 10px; font-weight: 600;
            }}
        """
        self._deriv_btn_style_inactive = f"""
            QPushButton {{
                background: {COLORS['bg_elevated']}; color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border_subtle']}; border-radius: 3px; padding: 2px 8px;
                font-size: 10px; font-weight: 500;
            }}
            QPushButton:hover {{ background: {COLORS['bg_dark']}; color: {COLORS['text_primary']}; }}
        """

        # Per-ben produkttyp: 'mini' eller 'cert'
        self._active_product_type_y = 'mini'
        self._active_product_type_x = 'mini'

        # Y ticker mini future
        self.mini_y_frame = QFrame()
        self.mini_y_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_elevated']};
                border: none;
                border-left: 3px solid {COLORS['positive']};
                border-radius: 4px;
            }}
        """)
        mini_y_layout = QVBoxLayout(self.mini_y_frame)
        mini_y_layout.setContentsMargins(10, 8, 10, 8)
        mini_y_layout.setSpacing(2)
        
        self.mini_y_ticker = QLabel("POSITION IN Y - MINI LONG")
        self.mini_y_ticker.setStyleSheet(f"color: {COLORS['positive']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mini_y_layout.addWidget(self.mini_y_ticker)

        # Produkttyp-knappar för Y
        y_btn_row = QHBoxLayout()
        y_btn_row.setSpacing(4)
        y_btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_mini_y = QPushButton("Mini (0)")
        self.btn_mini_y.setStyleSheet(self._deriv_btn_style_active)
        self.btn_mini_y.setCursor(Qt.PointingHandCursor)
        self.btn_mini_y.clicked.connect(lambda: self._switch_product_type('mini', 'y'))
        y_btn_row.addWidget(self.btn_mini_y)
        self.btn_cert_y = QPushButton("Cert (0)")
        self.btn_cert_y.setStyleSheet(self._deriv_btn_style_inactive)
        self.btn_cert_y.setCursor(Qt.PointingHandCursor)
        self.btn_cert_y.clicked.connect(lambda: self._switch_product_type('cert', 'y'))
        y_btn_row.addWidget(self.btn_cert_y)
        y_btn_row.addStretch()
        mini_y_layout.addLayout(y_btn_row)

        self._instruments_y = []  # Filtrerade instrument för Y
        self.mini_y_combo = QComboBox()
        self.mini_y_combo.addItem("No instruments found")
        self.mini_y_combo.setStyleSheet(f"""
            QComboBox {{
                color: {COLORS['text_primary']}; font-size: 12px; font-weight: 500;
                background: {COLORS['bg_dark']}; border: 1px solid {COLORS['border_subtle']};
                border-radius: 3px; padding: 3px 6px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox::down-arrow {{ image: none; border: none; width: 12px; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['bg_dark']}; color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']}; border: 1px solid {COLORS['border_subtle']};
                font-size: 11px;
            }}
        """)
        self.mini_y_combo.currentIndexChanged.connect(lambda idx: self._on_instrument_changed(idx, 'y'))
        mini_y_layout.addWidget(self.mini_y_combo)
        
        self.mini_y_info = QLabel("")
        self.mini_y_info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        self.mini_y_info.setWordWrap(True)
        mini_y_layout.addWidget(self.mini_y_info)
        
        left_layout.addWidget(self.mini_y_frame)
        
        # X ticker mini future
        self.mini_x_frame = QFrame()
        self.mini_x_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_elevated']};
                border: none;
                border-left: 2px solid {COLORS['negative']};
                border-radius: 4px;
            }}
        """)
        mini_x_layout = QVBoxLayout(self.mini_x_frame)
        mini_x_layout.setContentsMargins(10, 8, 10, 8)
        mini_x_layout.setSpacing(2)
        
        self.mini_x_ticker = QLabel("POSITION IN X - MINI SHORT")
        self.mini_x_ticker.setStyleSheet(f"color: {COLORS['negative']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mini_x_layout.addWidget(self.mini_x_ticker)

        # Produkttyp-knappar för X
        x_btn_row = QHBoxLayout()
        x_btn_row.setSpacing(4)
        x_btn_row.setContentsMargins(0, 0, 0, 0)
        self.btn_mini_x = QPushButton("Mini (0)")
        self.btn_mini_x.setStyleSheet(self._deriv_btn_style_active)
        self.btn_mini_x.setCursor(Qt.PointingHandCursor)
        self.btn_mini_x.clicked.connect(lambda: self._switch_product_type('mini', 'x'))
        x_btn_row.addWidget(self.btn_mini_x)
        self.btn_cert_x = QPushButton("Cert (0)")
        self.btn_cert_x.setStyleSheet(self._deriv_btn_style_inactive)
        self.btn_cert_x.setCursor(Qt.PointingHandCursor)
        self.btn_cert_x.clicked.connect(lambda: self._switch_product_type('cert', 'x'))
        x_btn_row.addWidget(self.btn_cert_x)
        x_btn_row.addStretch()
        mini_x_layout.addLayout(x_btn_row)

        self._instruments_x = []  # Filtrerade instrument för X
        self.mini_x_combo = QComboBox()
        self.mini_x_combo.addItem("No instruments found")
        self.mini_x_combo.setStyleSheet(f"""
            QComboBox {{
                color: {COLORS['text_primary']}; font-size: 12px; font-weight: 500;
                background: {COLORS['bg_dark']}; border: 1px solid {COLORS['border_subtle']};
                border-radius: 3px; padding: 3px 6px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox::down-arrow {{ image: none; border: none; width: 12px; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['bg_dark']}; color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent']}; border: 1px solid {COLORS['border_subtle']};
                font-size: 11px;
            }}
        """)
        self.mini_x_combo.currentIndexChanged.connect(lambda idx: self._on_instrument_changed(idx, 'x'))
        mini_x_layout.addWidget(self.mini_x_combo)
        
        self.mini_x_info = QLabel("")
        self.mini_x_info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        self.mini_x_info.setWordWrap(True)
        mini_x_layout.addWidget(self.mini_x_info)
        
        left_layout.addWidget(self.mini_x_frame)
        
        # Divider before MF position sizing
        divider3 = QFrame()
        divider3.setFrameShape(QFrame.HLine)
        divider3.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        divider3.setMaximumHeight(1)
        left_layout.addWidget(divider3)
        
        # Mini Futures Position Sizing header
        mf_header = QLabel("POSITION SIZING")
        mf_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 600; letter-spacing: 1px; border: none;")
        left_layout.addWidget(mf_header)
        
        # MF Position sizing - compact cards
        mf_pos_layout = QHBoxLayout()
        mf_pos_layout.setSpacing(6)
        
        # Y leg position sizing
        self.mf_pos_y_frame = QFrame()
        self.mf_pos_y_frame.setStyleSheet(f"background: {COLORS['bg_elevated']}; border-radius: 4px; border-left: 2px solid {COLORS['positive']};")
        mf_pos_y_layout = QVBoxLayout(self.mf_pos_y_frame)
        mf_pos_y_layout.setContentsMargins(8, 6, 8, 6)
        mf_pos_y_layout.setSpacing(2)
        
        self.mf_pos_y_action = QLabel("BUY Y MINI")
        self.mf_pos_y_action.setStyleSheet(f"color: {COLORS['positive']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mf_pos_y_layout.addWidget(self.mf_pos_y_action)
        
        self.mf_pos_y_capital = QLabel("-")
        self.mf_pos_y_capital.setStyleSheet(f"color: {COLORS['positive']}; font-size: 12px; font-weight: 700; background: transparent; border: none;")
        mf_pos_y_layout.addWidget(self.mf_pos_y_capital)
        
        self.mf_pos_y_info = QLabel("capital allocation")
        self.mf_pos_y_info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent; border: none;")
        mf_pos_y_layout.addWidget(self.mf_pos_y_info)
        
        self.mf_pos_y_exposure = QLabel("")
        self.mf_pos_y_exposure.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px; background: transparent; border: none;")
        mf_pos_y_layout.addWidget(self.mf_pos_y_exposure)
        
        mf_pos_layout.addWidget(self.mf_pos_y_frame)
        
        # X leg position sizing
        self.mf_pos_x_frame = QFrame()
        self.mf_pos_x_frame.setStyleSheet(f"background: {COLORS['bg_elevated']}; border-radius: 4px; border-left: 2px solid {COLORS['negative']};")
        mf_pos_x_layout = QVBoxLayout(self.mf_pos_x_frame)
        mf_pos_x_layout.setContentsMargins(8, 6, 8, 6)
        mf_pos_x_layout.setSpacing(2)
        
        self.mf_pos_x_action = QLabel("SELL X MINI")
        self.mf_pos_x_action.setStyleSheet(f"color: {COLORS['negative']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mf_pos_x_layout.addWidget(self.mf_pos_x_action)
        
        self.mf_pos_x_capital = QLabel("-")
        self.mf_pos_x_capital.setStyleSheet(f"color: {COLORS['negative']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mf_pos_x_layout.addWidget(self.mf_pos_x_capital)
        
        self.mf_pos_x_info = QLabel("capital allocation")
        self.mf_pos_x_info.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent; border: none;")
        mf_pos_x_layout.addWidget(self.mf_pos_x_info)
        
        self.mf_pos_x_exposure = QLabel("")
        self.mf_pos_x_exposure.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px; background: transparent; border: none;")
        mf_pos_x_layout.addWidget(self.mf_pos_x_exposure)
        
        mf_pos_layout.addWidget(self.mf_pos_x_frame)
        
        # Total summary
        self.mf_pos_total_frame = QFrame()
        self.mf_pos_total_frame.setStyleSheet(f"background: {COLORS['bg_elevated']}; border-left: 2px solid {COLORS['accent']}; border-radius: 4px;")
        mf_pos_total_layout = QVBoxLayout(self.mf_pos_total_frame)
        mf_pos_total_layout.setContentsMargins(8, 6, 8, 6)
        mf_pos_total_layout.setSpacing(2)
        
        mf_pos_total_title = QLabel("TOTAL CAPITAL")
        mf_pos_total_title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mf_pos_total_layout.addWidget(mf_pos_total_title)
        
        self.mf_pos_total_capital = QLabel("-")
        self.mf_pos_total_capital.setStyleSheet(f"color: {COLORS['accent_bright']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
        mf_pos_total_layout.addWidget(self.mf_pos_total_capital)
        
        self.mf_pos_total_beta = QLabel("")
        self.mf_pos_total_beta.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent; border: none;")
        mf_pos_total_layout.addWidget(self.mf_pos_total_beta)
        
        self.mf_pos_total_exposure = QLabel("")
        self.mf_pos_total_exposure.setStyleSheet(f"color: {COLORS['positive']}; font-size: 11px; background: transparent; border: none;")
        mf_pos_total_layout.addWidget(self.mf_pos_total_exposure)
        
        self.mf_pos_eff_leverage = QLabel("")
        self.mf_pos_eff_leverage.setStyleSheet(f"color: #9c27b0; font-size: 11px; background: transparent; border: none;")
        mf_pos_total_layout.addWidget(self.mf_pos_eff_leverage)
        
        mf_pos_layout.addWidget(self.mf_pos_total_frame)
        
        left_layout.addLayout(mf_pos_layout)
        
        # Minimum capital info
        self.mf_min_cap_frame = QFrame()
        self.mf_min_cap_frame.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(0, 188, 212, 0.1);
                border: 1px solid {COLORS['info']};
                border-radius: 4px;
            }}
        """)
        mf_min_cap_layout = QHBoxLayout(self.mf_min_cap_frame)
        mf_min_cap_layout.setContentsMargins(8, 6, 8, 6)
        
        self.mf_min_cap_label = QLabel("ℹ Minimum capital info will appear here")
        self.mf_min_cap_label.setStyleSheet(f"color: {COLORS['info']}; font-size: 12px; background: transparent; border: none;")
        self.mf_min_cap_label.setWordWrap(True)
        mf_min_cap_layout.addWidget(self.mf_min_cap_label)
        
        left_layout.addWidget(self.mf_min_cap_frame)
        left_layout.addStretch()
        
        left_scroll.setWidget(left_panel)
        content_layout.addWidget(left_scroll, stretch=2)
        
        # ===== RIGHT COLUMN: Charts =====
        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 6px;
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 14, 16, 14)
        right_layout.setSpacing(10)
        
        # Check if pyqtgraph is available
        pg = get_pyqtgraph()
        if pg is not None:
            
            DateAxisItem = get_date_axis_item_class()
            
            # self.ou_price_date_axis = DateAxisItem(orientation='bottom')

            # Price comparison plot
            price_header = QLabel("BETA ADJUSTED PRICE COMPARISON")
            price_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
            right_layout.addWidget(price_header)
            
            # Create date axis for price plot
            self.signal_price_date_axis = DateAxisItem(orientation='bottom')
            self.signal_price_plot = pg.PlotWidget(axisItems={'bottom': self.signal_price_date_axis})
            self.signal_price_plot.setLabel('left', 'Price')
            self.signal_price_plot.showGrid(x=False, y=False, alpha=0.3)
            self.signal_price_plot.addLegend()
            self.signal_price_plot.setMinimumHeight(180)
            self.signal_price_plot.setMouseEnabled(x=True, y=True)
            self.signal_price_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            right_layout.addWidget(self.signal_price_plot, stretch=2)
            
            # Spread plot
            zscore_header = QLabel("SPREAD (Y − β·X − α)")
            zscore_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
            right_layout.addWidget(zscore_header)
            
            # Create date axis for spread plot
            self.signal_zscore_date_axis = DateAxisItem(orientation='bottom')
            self.signal_zscore_plot = pg.PlotWidget(axisItems={'bottom': self.signal_zscore_date_axis})
            self.signal_zscore_plot.setLabel('left', 'Spread')
            self.signal_zscore_plot.showGrid(x=False, y=False, alpha=0.3)
            self.signal_zscore_plot.setMinimumHeight(120)
            self.signal_zscore_plot.setMouseEnabled(x=True, y=True)
            self.signal_zscore_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            right_layout.addWidget(self.signal_zscore_plot, stretch=1)
            
            # Synchronize X-axis zoom between plots
            self._signal_syncing_plots = False
            
            def sync_price_to_zscore(viewbox, range):
                if not self._signal_syncing_plots:
                    self._signal_syncing_plots = True
                    self.signal_zscore_plot.setXRange(range[0][0], range[0][1], padding=0)
                    self._signal_syncing_plots = False
            
            def sync_zscore_to_price(viewbox, range):
                if not self._signal_syncing_plots:
                    self._signal_syncing_plots = True
                    self.signal_price_plot.setXRange(range[0][0], range[0][1], padding=0)
                    self._signal_syncing_plots = False
            
            self.signal_price_plot.sigRangeChanged.connect(sync_price_to_zscore)
            self.signal_zscore_plot.sigRangeChanged.connect(sync_zscore_to_price)
            
            # Initialize crosshair managers (will be populated when signal changes)
            self.signal_price_crosshair = None
            self.signal_zscore_crosshair = None
            
        else:
            no_pg_label = QLabel("Install pyqtgraph for charts")
            no_pg_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
            no_pg_label.setAlignment(Qt.AlignCenter)
            right_layout.addWidget(no_pg_label)
        
        content_layout.addWidget(right_panel, stretch=3)
        
        layout.addLayout(content_layout)
        
        return tab
    
    # ========================================================================
    # TAB 5: PORTFOLIO
    # ========================================================================
    
    def create_portfolio_tab(self) -> QWidget:
        """Create Portfolio tab with compact design."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)  # Ökat från 10
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Sub-tabs
        self.portfolio_subtabs = QTabWidget()
        self.portfolio_subtabs.setDocumentMode(True)
        
        # Current Positions sub-tab
        current_tab = QWidget()
        current_layout = QVBoxLayout(current_tab)
        current_layout.setSpacing(12)  # Ökat från 10
        
        # Compact summary metrics row
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(10, 6, 10, 6)
        summary_layout.setSpacing(20)
        
        # Compact inline metrics
        self.open_pos_label = QLabel("Open: <span style='color:#22c55e; font-weight:600;'>0</span>")
        self.open_pos_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.open_pos_label)
        
        self.closed_pos_label = QLabel("Closed: <span style='color:#888;'>0</span>")
        self.closed_pos_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.closed_pos_label)
        
        self.unrealized_pnl_label = QLabel("Unrealized: <span style='color:#22c55e;'>+0.00%</span>")
        self.unrealized_pnl_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.unrealized_pnl_label)
        
        self.realized_pnl_label = QLabel("Realized: <span style='color:#888;'>+0.00%</span>")
        self.realized_pnl_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.realized_pnl_label)
        
        summary_layout.addStretch()
        
        # Refresh button
        refresh_btn = QPushButton("Update Z-Scores")
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_hover']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: {COLORS['accent_dark']};
                color: {COLORS['accent']};
            }}
        """)
        refresh_btn.clicked.connect(self.refresh_portfolio_zscores)
        summary_layout.addWidget(refresh_btn)
        
        # Refresh mini futures prices button
        refresh_mf_btn = QPushButton("Update Derivatives Prices")
        refresh_mf_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_hover']};
                color: {COLORS['accent']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background: #1a2a1a;
                color: #2ed573;
            }}
        """)
        refresh_mf_btn.clicked.connect(self.refresh_mf_prices)
        summary_layout.addWidget(refresh_mf_btn)
        
        current_layout.addWidget(summary_frame)
        
        # Positions table with dynamic columns + inline z-chart
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(14)
        self.positions_table.setHorizontalHeaderLabels([
            "PAIR", "DIRECTION", "Z-SCORE", "STATUS",
            "Y LEG", "ENTRY PRICE", "QUANTITY", "P/L",
            "X LEG", "ENTRY PRICE", "QUANTITY", "P/L",
            "CLOSE", "Z-CHART"
        ])

        # Row height 3x for inline chart visibility
        self.positions_table.verticalHeader().setDefaultSectionSize(110)
        self.positions_table.verticalHeader().setVisible(False)

        # Dynamic column widths
        header = self.positions_table.horizontalHeader()
        header.setStretchLastSection(True)  # Z-CHART stretches to fill

        self.positions_table.setColumnWidth(0, 140)   # PAIR
        self.positions_table.setColumnWidth(1, 100)    # DIR
        self.positions_table.setColumnWidth(2, 90)     # Z
        self.positions_table.setColumnWidth(3, 90)     # STATUS
        self.positions_table.setColumnWidth(4, 220)   # MINI Y
        self.positions_table.setColumnWidth(5, 100)   # ENTRY Y
        self.positions_table.setColumnWidth(6, 90)     # QTY Y
        self.positions_table.setColumnWidth(7, 110)   # P/L Y
        self.positions_table.setColumnWidth(8, 220)   # MINI X
        self.positions_table.setColumnWidth(9, 100)   # ENTRY X
        self.positions_table.setColumnWidth(10, 90)    # QTY X
        self.positions_table.setColumnWidth(11, 110)   # P/L X
        self.positions_table.setColumnWidth(12, 70)    # CLOSE
        # col 13 (Z-CHART) stretches via setStretchLastSection

        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(13, QHeaderView.Stretch)
        header.setMinimumSectionSize(45)

        self.positions_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.positions_table.setAlternatingRowColors(True)
        self.positions_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                alternate-background-color: {COLORS['bg_elevated']};
                gridline-color: {COLORS['border_subtle']};
                font-size: 13px;
            }}
            QTableWidget::item {{
                padding: 8px 12px;
                font-size: 13px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['accent']};
                font-weight: 600;
                font-size: 12px;
                padding: 10px 8px;
                border: none;
                border-bottom: 2px solid {COLORS['accent']};
            }}
        """)
        current_layout.addWidget(self.positions_table)

        # Clear all button
        btn_layout = QHBoxLayout()
        self.clear_all_btn = QPushButton("🗑 Clear All")
        self.clear_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['negative']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #ff3d5a;
            }}
        """)
        self.clear_all_btn.clicked.connect(self.clear_all_positions)
        self.clear_all_btn.setMaximumWidth(120)
        btn_layout.addWidget(self.clear_all_btn)
        btn_layout.addStretch()
        current_layout.addLayout(btn_layout)
        
        self.portfolio_subtabs.addTab(current_tab, "📋 CURRENT POSITIONS")
        
        # Trade History sub-tab
        history_tab = self._create_trade_history_subtab()
        self.portfolio_subtabs.addTab(history_tab, "📜 TRADE HISTORY")
        
        # Benchmark Analysis sub-tab
        benchmark_tab = self._create_benchmark_analysis_subtab()
        self.portfolio_subtabs.addTab(benchmark_tab, "📈 BENCHMARK ANALYSIS")

        # Auto-load benchmark when switching to that tab
        self.portfolio_subtabs.currentChanged.connect(self._on_portfolio_subtab_changed)
        self._benchmark_loaded = False  # Track if benchmark data has been loaded
        
        layout.addWidget(self.portfolio_subtabs)
        
        return tab
    
    def _on_portfolio_subtab_changed(self, index: int):
        """Handle portfolio sub-tab change."""
        if index == 1:
            self._update_trade_history_display()
        elif index == 2:
            self._load_or_update_benchmark()
    
    # -----------------------------------------------------------------------
    # Z-CHART — per-pair z-score sparklines in positions table
    # -----------------------------------------------------------------------

    def _lm_get_or_create_plot(self, pair: str) -> Optional[dict]:
        """Return cached {plot_widget, z_curve, entry_line} or create a new one."""
        if pair in self._lm_pairs:
            return self._lm_pairs[pair]

        pg_mod = get_pyqtgraph()
        EpochAxis = get_epoch_axis_item_class()
        if pg_mod is None or EpochAxis is None:
            return None

        bottom_axis = EpochAxis(orientation='bottom')
        pw = pg_mod.PlotWidget(axisItems={'bottom': bottom_axis})
        pw.setBackground('#0a0a0a')
        pw.showGrid(x=False, y=True, alpha=0.15)
        pw.setLabel('left', 'Z', color='#888888', size='8pt')
        pw.getAxis('left').setWidth(30)
        pw.getAxis('bottom').setHeight(20)
        pw.setContentsMargins(0, 0, 0, 0)
        pw.getPlotItem().setContentsMargins(0, 0, 0, 0)

        # Reference lines
        pw.addLine(y=0,  pen=pg_mod.mkPen('#ffffff', width=1, style=Qt.DashLine))
        pw.addLine(y=2,  pen=pg_mod.mkPen('#22c55e', width=1, style=Qt.DashLine))
        pw.addLine(y=-2, pen=pg_mod.mkPen('#22c55e', width=1, style=Qt.DashLine))

        z_curve = pw.plot([], [], pen=pg_mod.mkPen('#d4a574', width=2))
        entry_line = None  # created dynamically

        self._lm_pairs[pair] = {
            'plot_widget': pw,
            'z_curve': z_curve,
            'entry_line': entry_line,
        }
        return self._lm_pairs[pair]

    def _lm_update_chart(self, pair: str, entry_z: float = 0.0,
                         entry_date: str = None, window_size: int = None):
        """Fetch spread from engine and refresh the z-score chart for *pair*."""
        if self.engine is None:
            return
        pd_data = self._lm_get_or_create_plot(pair)
        if pd_data is None:
            return

        try:
            ou, spread_series, current_z = self.engine.get_pair_ou_params(
                pair, use_raw_data=True, window_size=window_size)
        except Exception:
            return

        eq_std = ou.eq_std if ou.eq_std > 0 else 1.0
        spread = spread_series.dropna()

        # Filter from entry_date - 1 trading day to present
        if entry_date and len(spread) > 5:
            try:
                entry_ts = pd.Timestamp(entry_date)
                # Go back 1 trading day from entry
                mask = spread.index >= (entry_ts - pd.tseries.offsets.BDay(1))
                if mask.any() and mask.sum() >= 5:
                    spread = spread[mask]
            except Exception:
                spread = spread.iloc[-60:]
        else:
            spread = spread.iloc[-60:]

        if spread.empty:
            return

        epochs = [pd.Timestamp(ts).timestamp() for ts in spread.index]
        zvals  = [(s - ou.mu) / eq_std for s in spread.values]

        pd_data['z_curve'].setData(epochs, zvals)

        pg_mod = get_pyqtgraph()
        pw = pd_data['plot_widget']

        # Update entry-date vertical line
        if entry_date:
            try:
                entry_epoch = pd.Timestamp(entry_date).timestamp()
                if pd_data.get('entry_date_line') is not None:
                    try:
                        pw.removeItem(pd_data['entry_date_line'])
                    except Exception:
                        pass
                pd_data['entry_date_line'] = pg_mod.InfiniteLine(
                    pos=entry_epoch, angle=90, movable=False,
                    pen=pg_mod.mkPen('#555555', width=1, style=Qt.DashLine))
                pw.addItem(pd_data['entry_date_line'])
            except Exception:
                pass

        # Update entry-z horizontal line
        ez = entry_z if entry_z else current_z
        if pd_data['entry_line'] is not None:
            try:
                pw.removeItem(pd_data['entry_line'])
            except Exception:
                pass
        pd_data['entry_line'] = pg_mod.InfiniteLine(
            pos=ez, angle=0, movable=False,
            pen=pg_mod.mkPen('#f59e0b', width=1, style=Qt.DotLine),
            label=f'Entry {ez:.2f}',
            labelOpts={'color': '#f59e0b', 'position': 0.05})
        pw.addItem(pd_data['entry_line'])

    def _lm_cleanup_stale_pairs(self):
        """Remove chart data for pairs no longer in portfolio."""
        active = {pos['pair'] for pos in self.portfolio}
        stale = [p for p in self._lm_pairs if p not in active]
        for p in stale:
            pw = self._lm_pairs[p].get('plot_widget')
            if pw is not None:
                pw.setParent(None)
                pw.deleteLater()
            del self._lm_pairs[p]

    def _create_trade_history_subtab(self) -> QWidget:
        """Create Trade History sub-tab showing closed positions."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Summary bar
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(10, 6, 10, 6)
        summary_layout.setSpacing(20)
        
        self.th_total_label = QLabel("Total Trades: <span style='color:#e8e8e8; font-weight:600;'>0</span>")
        self.th_total_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.th_total_label)
        
        self.th_winrate_label = QLabel("Win Rate: <span style='color:#888;'>-</span>")
        self.th_winrate_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.th_winrate_label)
        
        self.th_realized_label = QLabel("Total Realized: <span style='color:#888;'>0 SEK</span>")
        self.th_realized_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        summary_layout.addWidget(self.th_realized_label)
        
        summary_layout.addStretch()
        layout.addWidget(summary_frame)
        
        # Trade history table
        self.trade_history_table = QTableWidget()
        self.trade_history_table.verticalHeader().setVisible(False)
        self.trade_history_table.verticalHeader().setDefaultSectionSize(40)
        self.trade_history_table.setColumnCount(10)
        self.trade_history_table.setHorizontalHeaderLabels([
            "PAIR", "DIRECTION", "ENTRY DATE", "CLOSE DATE",
            "ENTRY Z", "CLOSE Z", "RESULT",
            "REALIZED P/L (SEK)", "REALIZED P/L (%)", "CAPITAL"
        ])
        header = self.trade_history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.trade_history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.trade_history_table.setAlternatingRowColors(True)
        self.trade_history_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                alternate-background-color: {COLORS['bg_elevated']};
                gridline-color: {COLORS['border_subtle']};
                font-size: 13px;
            }}
            QTableWidget::item {{
                padding: 8px 12px;
                font-size: 13px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['accent']};
                font-weight: 600;
                font-size: 12px;
                padding: 10px 8px;
                border: none;
                border-bottom: 2px solid {COLORS['accent']};
            }}
        """)
        layout.addWidget(self.trade_history_table)
        
        return tab
    
    def _update_trade_history_display(self):
        """Update the trade history table with closed positions."""
        self.trade_history_table.setRowCount(len(self.trade_history))
        
        total_realized_sek = 0
        wins = 0
        
        for i, trade in enumerate(reversed(self.trade_history)):  # Newest first
            # PAIR
            pair_item = QTableWidgetItem(trade.get('pair', ''))
            pair_item.setForeground(QColor(COLORS['accent']))
            self.trade_history_table.setItem(i, 0, pair_item)
            
            # DIRECTION
            d = trade.get('direction', '')
            dir_item = QTableWidgetItem(d)
            dir_item.setForeground(QColor("#22c55e" if d == 'LONG' else "#ef4444"))
            self.trade_history_table.setItem(i, 1, dir_item)
            
            # ENTRY DATE
            self.trade_history_table.setItem(i, 2, QTableWidgetItem(trade.get('entry_date', '')))
            
            # CLOSE DATE
            self.trade_history_table.setItem(i, 3, QTableWidgetItem(trade.get('close_date', '')))
            
            # ENTRY Z
            ez = trade.get('entry_z', 0)
            ez_item = QTableWidgetItem(f"{ez:.2f}")
            ez_item.setForeground(QColor("#e8e8e8"))
            self.trade_history_table.setItem(i, 4, ez_item)
            
            # CLOSE Z
            cz = trade.get('close_z', 0)
            cz_item = QTableWidgetItem(f"{cz:.2f}")
            cz_item.setForeground(QColor("#e8e8e8"))
            self.trade_history_table.setItem(i, 5, cz_item)
            
            # RESULT
            result = trade.get('result', '')
            is_profit = result == 'PROFIT'
            if is_profit:
                wins += 1
            result_item = QTableWidgetItem(result)
            result_item.setForeground(QColor("#22c55e" if is_profit else "#ef4444"))
            self.trade_history_table.setItem(i, 6, result_item)
            
            # REALIZED P/L (SEK)
            pnl_sek = trade.get('realized_pnl_sek', 0)
            total_realized_sek += pnl_sek
            pnl_sek_item = QTableWidgetItem(f"{pnl_sek:+.0f}")
            pnl_sek_item.setForeground(QColor("#22c55e" if pnl_sek >= 0 else "#ef4444"))
            self.trade_history_table.setItem(i, 7, pnl_sek_item)
            
            # REALIZED P/L (%)
            pnl_pct = trade.get('realized_pnl_pct', 0)
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.1f}%")
            pnl_pct_item.setForeground(QColor("#22c55e" if pnl_pct >= 0 else "#ef4444"))
            self.trade_history_table.setItem(i, 8, pnl_pct_item)
            
            # CAPITAL
            cap = trade.get('capital', 0)
            cap_item = QTableWidgetItem(f"{cap:,.0f}")
            cap_item.setForeground(QColor("#e8e8e8"))
            self.trade_history_table.setItem(i, 9, cap_item)
        
        # Update summary
        n = len(self.trade_history)
        self.th_total_label.setText(f"Total Trades: <span style='color:#e8e8e8; font-weight:600;'>{n}</span>")
        
        if n > 0:
            wr = wins / n * 100
            wr_color = "#22c55e" if wr >= 50 else "#ef4444"
            self.th_winrate_label.setText(f"Win Rate: <span style='color:{wr_color}; font-weight:600;'>{wr:.0f}%</span> ({wins}/{n})")
        else:
            self.th_winrate_label.setText("Win Rate: <span style='color:#888;'>-</span>")
        
        r_color = "#22c55e" if total_realized_sek >= 0 else "#ef4444"
        self.th_realized_label.setText(f"Total Realized: <span style='color:{r_color}; font-weight:600;'>{total_realized_sek:+,.0f} SEK</span>")
    
    # ========================================================================
    # BENCHMARK ANALYSIS - Faktisk P&L baserad på positioner
    # ========================================================================
    
    def _create_benchmark_analysis_subtab(self) -> QWidget:
        """Create the Benchmark Analysis sub-tab for portfolio performance comparison."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Header with controls
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        # Title
        title = QLabel("PORTFOLIO VS BENCHMARK")
        title.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 1.5px;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Benchmark label (always S&P 500)
        bench_label = QLabel("Benchmark: S&P 500")
        bench_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        header_layout.addWidget(bench_label)
        
        # Refresh button
        refresh_bench_btn = QPushButton("⟳ Update Analysis")
        refresh_bench_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent_dark']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS['accent']};
            }}
        """)
        refresh_bench_btn.clicked.connect(self._update_benchmark_analysis)
        header_layout.addWidget(refresh_bench_btn)
        
        layout.addWidget(header_frame)
        
        # Main statistics table
        self.benchmark_stats_table = QTableWidget()
        self.benchmark_stats_table.setColumnCount(10)
        self.benchmark_stats_table.setHorizontalHeaderLabels([
            "METRIC", "1v", "1mo", "3mo", "6mo", "YTD", "1y", "3y", "5y", "Sen start"
        ])
        
        # Metrics to display
        metrics = [
            "Portfolio Return",
            "Benchmark Return",
            "Alpha",
            "Correlation",
            "Beta",
            "Sharpe (Portfolio)",
            "Sharpe (Benchmark)",
            "Tracking Error",
            "Information Ratio",
            "Max DD (Portfolio)",
            "Max DD (Benchmark)",
            "Sortino Ratio",
            "Win Rate",
            "Profit Factor"
        ]
        
        self.benchmark_stats_table.setRowCount(len(metrics))
        
        # Style table
        self.benchmark_stats_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                alternate-background-color: {COLORS['bg_elevated']};
                gridline-color: {COLORS['border_subtle']};
                font-family: 'JetBrains Mono', 'Fira Code', monospace;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 8px 12px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_elevated']};
                color: {COLORS['accent']};
                font-weight: 600;
                font-size: 11px;
                padding: 10px 8px;
                border: none;
                border-bottom: 2px solid {COLORS['accent']};
            }}
        """)
        
        # Dynamic column widths
        header = self.benchmark_stats_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 10):
            header.setSectionResizeMode(i, QHeaderView.Stretch)
        
        self.benchmark_stats_table.verticalHeader().setVisible(False)
        self.benchmark_stats_table.setAlternatingRowColors(True)
        self.benchmark_stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Tooltips for each metric
        metric_tooltips = [
            "Total return of your portfolio over the period",
            "Total return of S&P 500 (SPY) over the same period",
            "Excess return vs benchmark (Portfolio Return - Benchmark Return)",
            "Pearson correlation between daily portfolio and benchmark returns",
            "Portfolio sensitivity to benchmark moves (covariance / variance)",
            "Risk-adjusted return of portfolio (excess return / std dev, annualized)",
            "Risk-adjusted return of benchmark (excess return / std dev, annualized)",
            "Std dev of return difference between portfolio and benchmark (annualized)",
            "Risk-adjusted excess return vs benchmark (alpha / tracking error)",
            "Largest peak-to-trough decline in portfolio value",
            "Largest peak-to-trough decline in benchmark value",
            "Like Sharpe but only penalizes downside volatility",
            "Percentage of days with positive returns",
            "Ratio of gross profits to gross losses",
        ]

        # Initialize with metric names
        for i, metric in enumerate(metrics):
            item = QTableWidgetItem(metric)
            item.setForeground(QColor(COLORS['text_primary']))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if i < len(metric_tooltips):
                item.setToolTip(metric_tooltips[i])
            self.benchmark_stats_table.setItem(i, 0, item)

            # Initialize other columns with "-"
            for j in range(1, 10):
                dash_item = QTableWidgetItem("-")
                dash_item.setForeground(QColor(COLORS['text_muted']))
                dash_item.setTextAlignment(Qt.AlignCenter)
                dash_item.setFlags(dash_item.flags() & ~Qt.ItemIsEditable)
                self.benchmark_stats_table.setItem(i, j, dash_item)
        
        layout.addWidget(self.benchmark_stats_table)
        
        # Cumulative returns chart (if pyqtgraph available)
        if ensure_pyqtgraph():
            chart_frame = QFrame()
            chart_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['bg_card']};
                    border: 1px solid {COLORS['border_subtle']};
                    border-radius: 6px;
                }}
            """)
            chart_layout = QVBoxLayout(chart_frame)
            chart_layout.setContentsMargins(10, 10, 10, 10)
            
            chart_title = QLabel("CUMULATIVE RETURNS (from earliest position entry)")
            chart_title.setStyleSheet(f"""
                color: {COLORS['accent']};
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 1px;
            """)
            chart_layout.addWidget(chart_title)
            
            self.benchmark_chart = pg.PlotWidget()
            self.benchmark_chart.setBackground(COLORS['bg_card'])
            self.benchmark_chart.showGrid(x=True, y=True, alpha=0.2)
            self.benchmark_chart.setLabel('left', 'Cumulative Return', '%')
            self.benchmark_chart.setLabel('bottom', 'Date')
            self.benchmark_chart.addLegend(offset=(10, 10))
            self.benchmark_chart.setMinimumHeight(250)
            self.benchmark_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            chart_layout.addWidget(self.benchmark_chart)

            chart_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(chart_frame)
        
        # Position summary
        self.position_summary_label = QLabel("")
        self.position_summary_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self.position_summary_label.setWordWrap(True)
        layout.addWidget(self.position_summary_label)
        
        # Status label
        self.benchmark_status_label = QLabel("Klicka 'Update Analysis' för att beräkna statistik")
        self.benchmark_status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; font-style: italic;")
        layout.addWidget(self.benchmark_status_label)
        
        return tab
    

    def _calculate_returns_from_history(self):
        """Calculate returns from portfolio history snapshots.
        
        COMPLETELY REWRITTEN:
        - Uses unrealized_pnl_pct directly from snapshots (already correct)
        - Does NOT use pct_change() which breaks with capital additions
        - Returns simple, correct data for display
        """
        if not PORTFOLIO_HISTORY_AVAILABLE or self.portfolio_history is None:
            print("DEBUG: portfolio_history not available")
            return None
        
        snapshots = self.portfolio_history.snapshots
        if len(snapshots) < 1:
            print("DEBUG: No snapshots found")
            return None
        
        print(f"DEBUG: Found {len(snapshots)} snapshots")
        
        sorted_snaps = sorted(snapshots, key=lambda s: s.timestamp)
        
        # Build simple lists from snapshots
        data_points = []
        for s in sorted_snaps:
            try:
                dt = datetime.fromisoformat(s.timestamp)
                date_key = pd.Timestamp(dt.date())
                
                # Handle both object attributes and dict keys
                if hasattr(s, 'unrealized_pnl_pct'):
                    pnl_pct = s.unrealized_pnl_pct
                elif isinstance(s, dict):
                    pnl_pct = s.get('unrealized_pnl_pct', 0)
                else:
                    pnl_pct = 0
                    
                if hasattr(s, 'benchmark_price'):
                    bench_price = s.benchmark_price
                elif isinstance(s, dict):
                    bench_price = s.get('benchmark_price')
                else:
                    bench_price = None
                
                print(f"DEBUG: {date_key.date()} -> pnl_pct={pnl_pct}, bench={bench_price}")
                
                data_points.append({
                    'date': date_key,
                    'pnl_pct': pnl_pct,
                    'benchmark': bench_price,
                    'total_invested': getattr(s, 'total_invested', None) if hasattr(s, 'total_invested') else s.get('total_invested') if isinstance(s, dict) else None
                })
            except Exception as e:
                print(f"DEBUG: Error processing snapshot: {e}")
                continue
        
        if len(data_points) < 1:
            print("DEBUG: No valid data points extracted")
            return None
        
        # Keep last value per day
        by_date = {}
        for dp in data_points:
            by_date[dp['date']] = dp
        
        sorted_dates = sorted(by_date.keys())
        
        # Build series
        pnl_pcts = [by_date[d]['pnl_pct'] for d in sorted_dates]
        benchmarks = [by_date[d]['benchmark'] for d in sorted_dates]
        
        print(f"DEBUG: pnl_pcts = {pnl_pcts}")
        print(f"DEBUG: benchmarks = {benchmarks}")
        
        pnl_series = pd.Series(pnl_pcts, index=sorted_dates)
        bench_series = pd.Series(benchmarks, index=sorted_dates).dropna()
        
        # Find earliest entry date from portfolio positions
        earliest_entry = None
        if self.portfolio:
            for pos in self.portfolio:
                entry_str = pos.get('entry_date')
                if entry_str:
                    try:
                        entry_dt = datetime.strptime(entry_str, '%Y-%m-%d %H:%M')
                        if earliest_entry is None or entry_dt < earliest_entry:
                            earliest_entry = entry_dt
                    except (ValueError, TypeError):
                        pass
        
        return {
            'pnl_series': pnl_series,           # Portfolio P&L % per day
            'benchmark_series': bench_series,    # Benchmark prices per day
            'first_date': sorted_dates[0], 
            'last_date': sorted_dates[-1],
            'earliest_entry': earliest_entry
        }
    
    def _set_benchmark_cell(self, row, col, value, is_return=False, is_ratio=False, is_pct=False, invert=False):
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            item = QTableWidgetItem("-")
            item.setForeground(QColor(COLORS['text_muted']))
        else:
            fmt = f"{value:+.2f}%" if (is_return or is_pct) else f"{value:.2f}"
            item = QTableWidgetItem(fmt)
            if is_return or is_ratio:
                if invert:
                    color = COLORS['negative'] if value < -5 else COLORS['text_primary']
                else:
                    color = COLORS['positive'] if value > 0 else COLORS['negative'] if value < 0 else COLORS['text_primary']
                item.setForeground(QColor(color))
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.benchmark_stats_table.setItem(row, col, item)
    
    def _update_pnl_chart(self, pnl_series, bench_series, earliest_entry=None):
        """Update cumulative returns chart - SIMPLE VERSION.
        
        Args:
            pnl_series: Series of P&L % (unrealized_pnl_pct from snapshots)
            bench_series: Series of benchmark prices
            earliest_entry: Earliest position entry datetime
        """
        self.benchmark_chart.clear()
        
        if len(pnl_series) == 0:
            return
        
        # Create entry date: day before first snapshot
        first_date = pnl_series.index[0]
        if earliest_entry is not None:
            entry_date = pd.Timestamp(earliest_entry.date())
            if entry_date >= first_date:
                entry_date = first_date - pd.Timedelta(days=1)
        else:
            entry_date = first_date - pd.Timedelta(days=1)
        
        # Build dates with entry point at 0%
        dates = [entry_date] + list(pnl_series.index)
        port_y = [0.0] + list(pnl_series.values)
        x_vals = list(range(len(dates)))
        
        # Plot portfolio P&L
        self.benchmark_chart.plot(
            x_vals, 
            port_y,
            pen=pg.mkPen(color=COLORS['accent'], width=2), 
            name='Portfolio'
        )
        
        # Benchmark cumulative return (normalized to same start)
        if bench_series is not None and len(bench_series) >= 2:
            common_dates = pnl_series.index.intersection(bench_series.index)
            if len(common_dates) >= 1:
                bench_aligned = bench_series.loc[common_dates]
                bench_baseline = bench_aligned.iloc[0]
                bench_cum = ((bench_aligned / bench_baseline) - 1) * 100
                
                # Build benchmark with 0% at entry
                bench_x = [0]
                bench_y = [0.0]
                
                for d in common_dates:
                    try:
                        idx = dates.index(d)
                        bench_x.append(idx)
                        bench_y.append(bench_cum.loc[d])
                    except (ValueError, KeyError):
                        continue
                
                if len(bench_x) >= 2:
                    self.benchmark_chart.plot(
                        bench_x,
                        bench_y,
                        pen=pg.mkPen(color=COLORS['info'], width=2), 
                        name='S&P 500'
                    )
        
        # Add zero line
        self.benchmark_chart.addLine(y=0, pen=pg.mkPen(color='#444', width=1, style=Qt.DashLine))
        
        # Set x-axis labels
        axis = self.benchmark_chart.getAxis('bottom')
        n = min(6, len(dates))
        idx = np.linspace(0, len(dates)-1, n, dtype=int)
        axis.setTicks([[(int(i), dates[i].strftime('%Y-%m-%d')) for i in idx]])
    
    def _update_cumulative_chart_from_pnl(self, port_pnl, bench_values, earliest_entry=None):
        """Update cumulative returns chart using P&L % directly.
        
        Args:
            port_pnl: Series of cumulative P&L % (unrealized_pnl_pct from snapshots)
            bench_values: Series of benchmark prices
            earliest_entry: Earliest position entry datetime
        """
        self.benchmark_chart.clear()
        
        if len(port_pnl) == 0:
            return
        
        # Portfolio P&L is already in decimal form, convert to %
        port_cum = port_pnl * 100
        
        # Create entry date: day before first snapshot
        first_date = port_pnl.index[0]
        if earliest_entry is not None:
            entry_date = pd.Timestamp(earliest_entry.date())
            # Make sure entry is before first data point
            if entry_date >= first_date:
                entry_date = first_date - pd.Timedelta(days=1)
        else:
            entry_date = first_date - pd.Timedelta(days=1)
        
        # Build dates with entry point
        dates = [entry_date] + list(port_pnl.index)
        
        # Build portfolio y-values with 0% at entry
        port_y = [0.0] + list(port_cum.values)
        
        # X-axis positions
        x_vals = list(range(len(dates)))
        
        # Plot portfolio
        self.benchmark_chart.plot(
            x_vals, 
            port_y,
            pen=pg.mkPen(color=COLORS['accent'], width=2), 
            name='Portfolio'
        )
        
        # Benchmark cumulative return
        if bench_values is not None and len(bench_values) >= 2:
            # Align benchmark to portfolio dates
            common_dates = port_pnl.index.intersection(bench_values.index)
            if len(common_dates) >= 1:
                bench_aligned = bench_values.loc[common_dates]
                
                # Benchmark baseline = value at first common date
                bench_baseline = bench_aligned.iloc[0]
                bench_cum = ((bench_aligned / bench_baseline) - 1) * 100
                
                # Build benchmark x-positions (offset by 1 for entry point)
                bench_x = [0]  # Start at entry point (0%)
                bench_y = [0.0]  # 0% at entry
                
                for d in common_dates:
                    try:
                        # Find position in dates list
                        idx = dates.index(d)
                        bench_x.append(idx)
                        bench_y.append(bench_cum.loc[d])
                    except (ValueError, KeyError):
                        continue
                
                if len(bench_x) >= 2:
                    self.benchmark_chart.plot(
                        bench_x,
                        bench_y,
                        pen=pg.mkPen(color=COLORS['info'], width=2), 
                        name='S&P 500'
                    )
        
        # Add zero line
        self.benchmark_chart.addLine(y=0, pen=pg.mkPen(color='#444', width=1, style=Qt.DashLine))
        
        # Set x-axis labels (show dates)
        axis = self.benchmark_chart.getAxis('bottom')
        n = min(6, len(dates))
        idx = np.linspace(0, len(dates)-1, n, dtype=int)
        axis.setTicks([[(int(i), dates[i].strftime('%Y-%m-%d')) for i in idx]])
    
    def _update_cumulative_chart_from_history(self, port_values, bench_values, total_invested=None, earliest_entry=None):
        """Update cumulative returns chart.
        
        FIXED: 
        - Always starts at 0% on the day BEFORE first snapshot
        - Uses total_invested as baseline
        - Benchmark aligned to same entry point
        """
        self.benchmark_chart.clear()
        
        if len(port_values) == 0:
            return
        
        # Use total_invested as baseline, fallback to first value if not available
        baseline = total_invested if total_invested and total_invested > 0 else port_values.iloc[0]
        
        # Portfolio cumulative return from INVESTED CAPITAL
        port_cum = ((port_values / baseline) - 1) * 100
        
        # Create date labels - ALWAYS add a day 0 at entry (before first snapshot)
        first_date = port_values.index[0]
        
        # Create entry date: either from earliest_entry or day before first snapshot
        if earliest_entry is not None:
            entry_date = pd.Timestamp(earliest_entry.date())
        else:
            # Use day before first snapshot as entry
            entry_date = first_date - pd.Timedelta(days=1)
        
        # Build x-axis dates with entry point at start
        dates = [entry_date] + list(port_values.index)
        
        # Build portfolio y-values with 0% at entry
        port_y = [0.0] + list(port_cum.values)
        
        # X-axis positions
        x_vals = list(range(len(dates)))
        
        # Plot portfolio
        self.benchmark_chart.plot(
            x_vals, 
            port_y,
            pen=pg.mkPen(color=COLORS['accent'], width=2), 
            name='Portfolio'
        )
        
        # Benchmark cumulative return
        if bench_values is not None and len(bench_values) >= 2:
            # Align benchmark to portfolio dates
            common_dates = port_values.index.intersection(bench_values.index)
            if len(common_dates) >= 1:
                bench_aligned = bench_values.loc[common_dates]
                
                # Benchmark baseline = value at first common date
                bench_baseline = bench_aligned.iloc[0]
                bench_cum = ((bench_aligned / bench_baseline) - 1) * 100
                
                # Build benchmark x-positions (offset by 1 for entry point)
                bench_x = [0]  # Start at entry point (0%)
                bench_y = [0.0]  # 0% at entry
                
                for d in common_dates:
                    try:
                        # Find position in dates list
                        idx = dates.index(d)
                        bench_x.append(idx)
                        bench_y.append(bench_cum.loc[d])
                    except (ValueError, KeyError):
                        continue
                
                if len(bench_x) >= 2:
                    self.benchmark_chart.plot(
                        bench_x,
                        bench_y,
                        pen=pg.mkPen(color=COLORS['info'], width=2), 
                        name='S&P 500'
                    )
        
        # Add zero line
        self.benchmark_chart.addLine(y=0, pen=pg.mkPen(color='#444', width=1, style=Qt.DashLine))
        
        # Set x-axis labels (show dates)
        axis = self.benchmark_chart.getAxis('bottom')
        n = min(6, len(dates))
        idx = np.linspace(0, len(dates)-1, n, dtype=int)
        axis.setTicks([[(int(i), dates[i].strftime('%Y-%m-%d')) for i in idx]])
    
    def _update_benchmark_from_history(self, data):
        """Update benchmark stats table and chart from history data.
        
        COMPLETELY REWRITTEN:
        - Uses pnl_series directly (unrealized_pnl_pct from snapshots)
        - Calculates benchmark return over same period
        - Simple and correct
        """
        pnl_series = data['pnl_series']          # Portfolio P&L % per day
        bench_series = data['benchmark_series']   # Benchmark prices per day
        earliest_entry = data.get('earliest_entry')
        
        if len(pnl_series) < 1:
            self.benchmark_status_label.setText("No data in snapshots")
            return
        
        n_days = len(pnl_series)
        
        # Current portfolio P&L (latest snapshot)
        current_pnl = pnl_series.iloc[-1]
        
        # Benchmark total return from first to last
        if len(bench_series) >= 2:
            bench_first = bench_series.iloc[0]
            bench_last = bench_series.iloc[-1]
            bench_total_return = ((bench_last / bench_first) - 1) * 100
        else:
            bench_total_return = 0
        
        # Calculate metrics for each period
        # Trading days: 1v=5, 1mo=21, 3mo=63, 6mo=126, 1y=252, 3y=756, 5y=1260
        periods = {
            '1v': (5, 1),           # 5 trading days (1 week)
            '1mo': (21, 2),         # ~21 trading days (1 month)
            '3mo': (63, 3),         # ~63 trading days (3 months)
            '6mo': (126, 4),        # ~126 trading days (6 months)
            'YTD': (self._calculate_ytd_days(), 5),
            '1y': (252, 6),         # 252 trading days (1 year)
            '3y': (756, 7),         # 756 trading days (3 years)
            '5y': (1260, 8),        # 1260 trading days (5 years)
            'Sen start': (999999, 9)  # All available data
        }
        
        rf_daily = 0.04 / 252  # 4% annual risk-free rate

        for name, (days, col) in periods.items():
            n_use = min(days, n_days)

            if n_use < 1:
                continue

            # Portfolio: change in P&L% over the period
            pnl_window = pnl_series.tail(n_use)
            if len(pnl_window) >= 2:
                port_return = pnl_window.iloc[-1] - pnl_window.iloc[0]
            else:
                port_return = pnl_window.iloc[-1]  # Just show current P&L
            
            # Benchmark: pct change over same period
            bench_window = bench_series.tail(n_use)
            if len(bench_window) >= 2:
                bench_return = ((bench_window.iloc[-1] / bench_window.iloc[0]) - 1) * 100
            else:
                bench_return = 0
            
            excess = port_return - bench_return
            
            # Set table cells
            self._set_benchmark_cell(0, col, port_return, is_return=True)
            self._set_benchmark_cell(1, col, bench_return, is_return=True)
            self._set_benchmark_cell(2, col, excess, is_return=True)
            
            # Advanced metrics only if enough data (and only for larger windows)
            if n_use >= 5 and len(pnl_window) >= 3:
                # Calculate daily changes for correlation etc
                pnl_changes = pnl_window.diff().dropna()
                bench_returns = bench_window.pct_change().dropna() * 100
                
                # Align
                common = pnl_changes.index.intersection(bench_returns.index)
                if len(common) >= 3:
                    pc = pnl_changes.loc[common]
                    br = bench_returns.loc[common]
                    
                    # Correlation
                    if pc.std() > 0 and br.std() > 0:
                        corr = pc.corr(br)
                        self._set_benchmark_cell(3, col, corr, is_ratio=True)
                    
                    # Beta
                    if br.var() > 0:
                        beta = pc.cov(br) / br.var()
                        self._set_benchmark_cell(4, col, beta, is_ratio=True)
                    
                    # Sharpe (annualized, with risk-free rate)
                    if pc.std() > 0:
                        sharpe = ((pc.mean() - rf_daily) * np.sqrt(252)) / pc.std()
                        self._set_benchmark_cell(5, col, sharpe, is_ratio=True)

                    if br.std() > 0:
                        bench_sharpe = ((br.mean() - rf_daily) * np.sqrt(252)) / br.std()
                        self._set_benchmark_cell(6, col, bench_sharpe, is_ratio=True)
                    
                    # Tracking Error
                    diff = pc - br
                    te = diff.std() * np.sqrt(252)
                    self._set_benchmark_cell(7, col, te, is_return=True)
                    
                    # Information Ratio
                    if te > 0:
                        ir = (diff.mean() * 252) / te
                        self._set_benchmark_cell(8, col, ir, is_ratio=True)
                    
                    # Max Drawdown (portfolio)
                    cum = (1 + pnl_window / 100).cumprod()
                    running_max = cum.cummax()
                    drawdown = (cum - running_max) / running_max
                    max_dd = drawdown.min() * 100
                    self._set_benchmark_cell(9, col, max_dd, is_return=True, invert=True)
                    
                    # Max Drawdown (benchmark)
                    bench_cum = (1 + br / 100).cumprod()
                    bench_running_max = bench_cum.cummax()
                    bench_drawdown = (bench_cum - bench_running_max) / bench_running_max
                    bench_max_dd = bench_drawdown.min() * 100
                    self._set_benchmark_cell(10, col, bench_max_dd, is_return=True, invert=True)
                    
                    # Sortino Ratio (portfolio) - uses downside deviation
                    downside = pc[pc < 0]
                    if len(downside) > 0:
                        downside_std = downside.std()
                        if downside_std > 0:
                            sortino = ((pc.mean() - rf_daily) * np.sqrt(252)) / downside_std
                            self._set_benchmark_cell(11, col, sortino, is_ratio=True)
                    
                    # Win Rate
                    wins = (pc > 0).sum()
                    total = len(pc)
                    win_rate = (wins / total) * 100 if total > 0 else None
                    self._set_benchmark_cell(12, col, win_rate, is_pct=True)
                    
                    # Profit Factor (sum of gains / sum of losses)
                    gains = pc[pc > 0].sum()
                    losses = abs(pc[pc < 0].sum())
                    if losses > 0:
                        profit_factor = gains / losses
                        self._set_benchmark_cell(13, col, profit_factor, is_ratio=True)
        
        # Update chart
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'benchmark_chart'):
            self._update_pnl_chart(pnl_series, bench_series, earliest_entry)
        
        n = self.portfolio_history.get_snapshot_count() if self.portfolio_history else 0
        self.benchmark_status_label.setText(
            f"✓ {n} snapshots | {data['first_date'].strftime('%Y-%m-%d')} to "
            f"{data['last_date'].strftime('%Y-%m-%d')} | Portfolio P&L: {current_pnl:.2f}%"
        )
        
        # Mark as loaded and save cache
        self._benchmark_loaded = True
        self._save_benchmark_cache()


    def _get_benchmark_ticker(self) -> str:
        """Get the yfinance ticker for the benchmark (always S&P 500)."""
        return "SPY"
    
    def _calculate_portfolio_returns_actual(self) -> Optional[pd.DataFrame]:
        """
        Beräkna faktisk portfölj-P&L baserat på positionernas entry points.
        
        Matematisk modell:
        ==================
        För varje position från entry_date:
        
        1. Hämta underliggande priser P_Y(t) och P_X(t)
        2. Entry priser: P_Y0, P_X0 (första dagen efter entry)
        3. Position sizing från mf_qty_y, mf_qty_x eller beräknat från notional
        
        Daglig P&L för LONG spread (long Y, short X):
            PnL_t = qty_Y × (P_Y,t - P_Y,t-1) - qty_X × (P_X,t - P_X,t-1)
        
        För SHORT spread (short Y, long X):
            PnL_t = -qty_Y × (P_Y,t - P_Y,t-1) + qty_X × (P_X,t - P_X,t-1)
        
        Daglig avkastning:
            r_t = PnL_t / Investerat_Kapital
        
        där Investerat_Kapital = entry_price_Y × qty_Y + entry_price_X × qty_X
        
        Returns:
            DataFrame med columns ['date', 'daily_return', 'daily_pnl', 'capital']
        """
        if not self.portfolio or self.engine is None:
            return None
        
        # Samla alla öppna positioner
        open_positions = [p for p in self.portfolio if p.get('status', 'OPEN') == 'OPEN']
        
        if not open_positions:
            return None
        
        # Hitta tidigaste entry date
        earliest_entry = None
        position_data = []
        
        for pos in open_positions:
            entry_date_str = pos.get('entry_date')
            if not entry_date_str:
                continue
            
            try:
                # Parse entry date (format: 'YYYY-MM-DD HH:MM')
                entry_dt = datetime.strptime(entry_date_str, '%Y-%m-%d %H:%M')
                entry_date = pd.Timestamp(entry_dt.date())
            except (ValueError, TypeError):
                continue
            
            pair = pos['pair']
            y_ticker, x_ticker = pair.split('/')
            
            # Kontrollera att vi har prisdata
            if y_ticker not in self.engine.price_data or x_ticker not in self.engine.price_data:
                continue
            
            # Hämta position sizing
            # Prioritet: mf_qty > beräknat från notional
            qty_y = pos.get('mf_qty_y', 0)
            qty_x = pos.get('mf_qty_x', 0)
            
            # Om inga MF quantities, beräkna från notional och aktiepriser
            if qty_y == 0 or qty_x == 0:
                notional = pos.get('notional', 10000)
                hedge_ratio = pos.get('hedge_ratio', 1.0)
                
                y_prices = self.engine.price_data[y_ticker]
                x_prices = self.engine.price_data[x_ticker]
                
                # Hitta entry price (första tillgängliga efter entry_date)
                y_after_entry = y_prices[y_prices.index >= entry_date]
                x_after_entry = x_prices[x_prices.index >= entry_date]
                
                if len(y_after_entry) == 0 or len(x_after_entry) == 0:
                    continue
                
                entry_y = y_after_entry.iloc[0]
                entry_x = x_after_entry.iloc[0]
                
                # Beräkna teoretiskt antal aktier baserat på notional
                # qty_y × P_Y + qty_x × P_X = notional
                # qty_x = qty_y × hedge_ratio
                # qty_y × P_Y + qty_y × hedge_ratio × P_X = notional
                # qty_y = notional / (P_Y + hedge_ratio × P_X)
                qty_y = notional / (entry_y + hedge_ratio * entry_x)
                qty_x = qty_y * hedge_ratio
            
            if earliest_entry is None or entry_date < earliest_entry:
                earliest_entry = entry_date
            
            position_data.append({
                'pair': pair,
                'y_ticker': y_ticker,
                'x_ticker': x_ticker,
                'entry_date': entry_date,
                'direction': pos.get('direction', 'LONG'),
                'qty_y': qty_y,
                'qty_x': qty_x,
                'hedge_ratio': pos.get('hedge_ratio', 1.0),
                'notional': pos.get('notional', 10000),
                'mf_entry_y': pos.get('mf_entry_price_y', 0),
                'mf_entry_x': pos.get('mf_entry_price_x', 0),
            })
        
        if not position_data or earliest_entry is None:
            return None
        
        # Bygg daglig avkastningsserie
        # Hitta senaste gemensamma datum
        all_dates = None
        for pdata in position_data:
            y_prices = self.engine.price_data[pdata['y_ticker']]
            x_prices = self.engine.price_data[pdata['x_ticker']]
            
            # Filtrera från entry date
            y_valid = y_prices[y_prices.index >= pdata['entry_date']]
            x_valid = x_prices[x_prices.index >= pdata['entry_date']]
            
            common = y_valid.index.intersection(x_valid.index)
            
            if all_dates is None:
                all_dates = set(common)
            else:
                all_dates = all_dates.union(set(common))
        
        if not all_dates:
            return None
        
        all_dates = sorted(all_dates)
        
        # Beräkna daglig P&L
        daily_results = []
        
        for date in all_dates:
            total_pnl = 0.0
            total_capital = 0.0
            
            for pdata in position_data:
                # Hoppa över om datum är före position entry
                if date < pdata['entry_date']:
                    continue
                
                y_prices = self.engine.price_data[pdata['y_ticker']]
                x_prices = self.engine.price_data[pdata['x_ticker']]
                
                # Kontrollera att vi har data för detta datum
                if date not in y_prices.index or date not in x_prices.index:
                    continue
                
                # Hitta föregående datum
                valid_dates = y_prices.index[y_prices.index <= date]
                if len(valid_dates) < 2:
                    continue
                
                prev_date = valid_dates[-2]
                
                # Hämta priser
                y_today = y_prices.loc[date]
                y_prev = y_prices.loc[prev_date]
                x_today = x_prices.loc[date]
                x_prev = x_prices.loc[prev_date]
                
                # Beräkna P&L
                # LONG spread: long Y, short X
                # SHORT spread: short Y, long X
                qty_y = pdata['qty_y']
                qty_x = pdata['qty_x']
                
                if pdata['direction'] == 'LONG':
                    # Long Y (profit when Y increases), Short X (profit when X decreases)
                    pnl = qty_y * (y_today - y_prev) - qty_x * (x_today - x_prev)
                else:
                    # Short Y (profit when Y decreases), Long X (profit when X increases)
                    pnl = -qty_y * (y_today - y_prev) + qty_x * (x_today - x_prev)
                
                total_pnl += pnl
                
                # Investerat kapital
                # Använd MF entry prices om tillgängliga, annars underliggande entry
                if pdata['mf_entry_y'] > 0 and pdata['mf_entry_x'] > 0:
                    capital = pdata['mf_entry_y'] * qty_y + pdata['mf_entry_x'] * qty_x
                else:
                    # Använd första tillgängliga priser efter entry som proxy
                    y_entry_prices = y_prices[y_prices.index >= pdata['entry_date']]
                    x_entry_prices = x_prices[x_prices.index >= pdata['entry_date']]
                    if len(y_entry_prices) > 0 and len(x_entry_prices) > 0:
                        capital = y_entry_prices.iloc[0] * qty_y + x_entry_prices.iloc[0] * qty_x
                    else:
                        capital = pdata['notional']
                
                total_capital += capital
            
            if total_capital > 0:
                daily_return = total_pnl / total_capital
            else:
                daily_return = 0.0
            
            daily_results.append({
                'date': date,
                'daily_pnl': total_pnl,
                'capital': total_capital,
                'daily_return': daily_return
            })
        
        if not daily_results:
            return None
        
        df = pd.DataFrame(daily_results)
        df.set_index('date', inplace=True)
        
        return df
    
    def _get_benchmark_returns(self, ticker: str, start_date: pd.Timestamp) -> Optional[pd.Series]:
        """Fetch benchmark returns from yfinance starting from a specific date."""
        try:
            import yfinance as yf
            
            # Add buffer
            end_date = pd.Timestamp.now()
            
            bench = yf.Ticker(ticker)
            hist = bench.history(start=start_date - pd.Timedelta(days=5), end=end_date)
            
            if hist.empty:
                return None
            
            returns = hist['Close'].pct_change().dropna()
            returns.index = returns.index.tz_localize(None)  # Remove timezone
            
            return returns
            
        except Exception as e:
            print(f"Error fetching benchmark {ticker}: {e}")
            return None
    
    def _calculate_ytd_days(self) -> int:
        """Calculate number of trading days YTD."""
        today = datetime.now()
        year_start = datetime(today.year, 1, 1)
        # Approximately 252 trading days per year
        days_elapsed = (today - year_start).days
        return int(days_elapsed * 252 / 365)
    
    def _calculate_metrics(self, portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series, 
                          window_days: int) -> Dict:
        """
        Beräkna alla performance-metriker för ett givet fönster.
        
        FIXED: Annualiserar INTE korta perioder (< 60 dagar) - visar enkla kumulativa returns.
        """
        # Align series by index
        common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
        
        if len(common_idx) < 1:
            return self._empty_metrics()
        
        # Get last N days (or all if fewer available)
        port_ret = portfolio_returns.loc[common_idx].tail(window_days)
        bench_ret = benchmark_returns.loc[common_idx].tail(window_days)
        
        actual_days = len(port_ret)
        if actual_days < 1:
            return self._empty_metrics()
        
        try:
            trading_days = 252
            
            # 1. Total returns (SIMPLE cumulative, no annualization for short periods)
            port_total = (1 + port_ret).prod() - 1
            bench_total = (1 + bench_ret).prod() - 1
            
            # Only annualize if we have >= 60 days of data
            if actual_days >= 60:
                port_return = ((1 + port_total) ** (trading_days / actual_days)) - 1
                bench_return = ((1 + bench_total) ** (trading_days / actual_days)) - 1
            else:
                # For short periods, show SIMPLE cumulative return (not annualized)
                port_return = port_total
                bench_return = bench_total
            
            # 2. Excess return
            excess_return = port_return - bench_return
            
            # 3-11: Only calculate advanced metrics if we have enough data
            if actual_days >= 3:
                # 3. Correlation
                if port_ret.std() > 0 and bench_ret.std() > 0:
                    correlation = port_ret.corr(bench_ret)
                else:
                    correlation = None
                
                # 4. Beta
                bench_var = bench_ret.var()
                if bench_var > 0:
                    beta = port_ret.cov(bench_ret) / bench_var
                else:
                    beta = None
                
                # 5. Sharpe Ratio (rf = 4% annual)
                rf_daily = 0.04 / trading_days
                port_excess_mean = port_ret.mean() - rf_daily
                bench_excess_mean = bench_ret.mean() - rf_daily
                
                port_std = port_ret.std()
                bench_std = bench_ret.std()
                
                if port_std > 0:
                    port_sharpe = (port_excess_mean * np.sqrt(trading_days)) / port_std
                else:
                    port_sharpe = None
                
                if bench_std > 0:
                    bench_sharpe = (bench_excess_mean * np.sqrt(trading_days)) / bench_std
                else:
                    bench_sharpe = None
                
                # 6. Tracking Error
                tracking_diff = port_ret - bench_ret
                tracking_error = tracking_diff.std() * np.sqrt(trading_days)
                
                # 7. Information Ratio
                if tracking_error > 0:
                    info_ratio = (tracking_diff.mean() * trading_days) / tracking_error
                else:
                    info_ratio = None
                
                # 8. Max Drawdown
                port_cumulative = (1 + port_ret).cumprod()
                port_running_max = port_cumulative.cummax()
                port_drawdown = (port_cumulative - port_running_max) / port_running_max
                port_max_dd = port_drawdown.min()
                
                bench_cumulative = (1 + bench_ret).cumprod()
                bench_running_max = bench_cumulative.cummax()
                bench_drawdown = (bench_cumulative - bench_running_max) / bench_running_max
                bench_max_dd = bench_drawdown.min()
                
                # 9. Sortino Ratio
                downside_returns = port_ret[port_ret < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * np.sqrt(trading_days)
                else:
                    downside_std = port_std * np.sqrt(trading_days) if port_std else 0
                
                if downside_std > 0:
                    sortino = ((port_ret.mean() - rf_daily) * trading_days) / downside_std
                else:
                    sortino = None
                
                # 10. Win Rate
                wins = (port_ret > 0).sum()
                total = len(port_ret)
                win_rate = (wins / total) if total > 0 else None
                
                # 11. Profit Factor
                gross_profit = port_ret[port_ret > 0].sum()
                gross_loss = abs(port_ret[port_ret < 0].sum())
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                elif gross_profit > 0:
                    profit_factor = float('inf')
                else:
                    profit_factor = None
            else:
                # Not enough data for advanced metrics
                correlation = None
                beta = None
                port_sharpe = None
                bench_sharpe = None
                tracking_error = None
                info_ratio = None
                port_max_dd = None
                bench_max_dd = None
                sortino = None
                win_rate = None
                profit_factor = None
            
            return {
                'port_return': port_return * 100,
                'bench_return': bench_return * 100,
                'excess_return': excess_return * 100,
                'correlation': correlation,
                'beta': beta,
                'port_sharpe': port_sharpe,
                'bench_sharpe': bench_sharpe,
                'tracking_error': tracking_error * 100 if tracking_error is not None else None,
                'info_ratio': info_ratio,
                'port_max_dd': port_max_dd * 100 if port_max_dd is not None else None,
                'bench_max_dd': bench_max_dd * 100 if bench_max_dd is not None else None,
                'sortino': sortino,
                'win_rate': win_rate * 100 if win_rate is not None else None,
                'profit_factor': profit_factor,
                'actual_days': actual_days
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict."""
        return {
            'port_return': None,
            'bench_return': None,
            'excess_return': None,
            'correlation': None,
            'beta': None,
            'port_sharpe': None,
            'bench_sharpe': None,
            'tracking_error': None,
            'info_ratio': None,
            'port_max_dd': None,
            'bench_max_dd': None,
            'sortino': None,
            'win_rate': None,
            'profit_factor': None,
            'actual_days': 0
        }
    
    def _update_benchmark_analysis(self):
        """Update all benchmark analysis metrics and charts."""
        self.benchmark_status_label.setText("Beräknar faktisk P&L från positioner...")
        QApplication.processEvents()
        
        # Try history first (uses actual MF prices - more accurate)
        if PORTFOLIO_HISTORY_AVAILABLE and self.portfolio_history is not None:
            data = self._calculate_returns_from_history()
            # FIXED: Use correct key 'pnl_series' instead of 'portfolio_returns'
            if data and data.get('pnl_series') is not None and len(data['pnl_series']) >= 1:
                self._update_benchmark_from_history(data)
                return
        
        # Fallback to position-based calculation
        portfolio_df = self._calculate_portfolio_returns_actual()
        
        if portfolio_df is None or len(portfolio_df) < 2:
            self.benchmark_status_label.setText(
                "⚠️ Ingen tillräcklig positionsdata. Se till att positioner har entry_date och prisdata finns."
            )
            return
        
        portfolio_returns = portfolio_df['daily_return']
        earliest_date = portfolio_returns.index.min()
        
        # Get benchmark returns from same start date
        bench_ticker = self._get_benchmark_ticker()
        self.benchmark_status_label.setText(f"Hämtar benchmark-data ({bench_ticker})...")
        QApplication.processEvents()
        
        benchmark_returns = self._get_benchmark_returns(bench_ticker, earliest_date)
        
        if benchmark_returns is None or len(benchmark_returns) < 2:
            self.benchmark_status_label.setText(f"⚠️ Kunde inte hämta data för {bench_ticker}")
            return
        
        # Define periods using trading days: 1v=5, 1mo=21, 3mo=63, 6mo=126, 1y=252, 3y=756, 5y=1260
        ytd_days = self._calculate_ytd_days()
        
        periods = {
            '1v': 5,           # 5 trading days (1 week)
            '1mo': 21,         # ~21 trading days (1 month)
            '3mo': 63,         # ~63 trading days (3 months)
            '6mo': 126,        # ~126 trading days (6 months)
            'YTD': ytd_days,
            '1y': 252,         # 252 trading days (1 year)
            '3y': 756,         # 756 trading days (3 years)
            '5y': 1260,        # 1260 trading days (5 years)
            'Sen start': 999999  # All available data
        }
        
        # Calculate metrics for each period
        all_metrics = {}
        for period_name, days in periods.items():
            all_metrics[period_name] = self._calculate_metrics(
                portfolio_returns, benchmark_returns, days
            )
        
        # Update table
        metric_keys = [
            ('port_return', '%.2f%%', True),
            ('bench_return', '%.2f%%', True),
            ('excess_return', '%+.2f%%', True),
            ('correlation', '%.3f', False),
            ('beta', '%.3f', False),
            ('port_sharpe', '%.2f', True),
            ('bench_sharpe', '%.2f', True),
            ('tracking_error', '%.2f%%', False),
            ('info_ratio', '%.2f', True),
            ('port_max_dd', '%.2f%%', True),
            ('bench_max_dd', '%.2f%%', False),
            ('sortino', '%.2f', True),
            ('win_rate', '%.1f%%', True),
            ('profit_factor', '%.2f', True)
        ]
        
        period_cols = {'1v': 1, '1mo': 2, '3mo': 3, '6mo': 4, 'YTD': 5, '1y': 6, '3y': 7, '5y': 8, 'Sen start': 9}
        
        for row, (key, fmt, color_code) in enumerate(metric_keys):
            for period_name, col in period_cols.items():
                value = all_metrics[period_name].get(key)
                
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    item = QTableWidgetItem("-")
                    item.setForeground(QColor(COLORS['text_muted']))
                else:
                    try:
                        if value == float('inf'):
                            text = "∞"
                        else:
                            text = fmt % value
                    except (TypeError, ValueError):
                        text = str(value)
                    
                    item = QTableWidgetItem(text)
                    
                    # Color coding
                    if color_code and value != float('inf'):
                        if key in ['port_return', 'excess_return', 'port_sharpe', 'sortino', 'info_ratio', 'profit_factor']:
                            if value > 0:
                                item.setForeground(QColor(COLORS['positive']))
                            elif value < 0:
                                item.setForeground(QColor(COLORS['negative']))
                            else:
                                item.setForeground(QColor(COLORS['text_secondary']))
                        elif key in ['port_max_dd', 'bench_max_dd']:
                            item.setForeground(QColor(COLORS['negative']))
                        elif key == 'win_rate':
                            if value >= 50:
                                item.setForeground(QColor(COLORS['positive']))
                            else:
                                item.setForeground(QColor(COLORS['negative']))
                        elif key == 'bench_return':
                            if value > 0:
                                item.setForeground(QColor(COLORS['positive']))
                            elif value < 0:
                                item.setForeground(QColor(COLORS['negative']))
                            else:
                                item.setForeground(QColor(COLORS['text_secondary']))
                        else:
                            item.setForeground(QColor(COLORS['text_primary']))
                    elif key == 'correlation':
                        # Low correlation = good for market neutral
                        if abs(value) < 0.3:
                            item.setForeground(QColor(COLORS['positive']))
                        elif abs(value) > 0.7:
                            item.setForeground(QColor(COLORS['warning']))
                        else:
                            item.setForeground(QColor(COLORS['text_primary']))
                    elif key == 'beta':
                        # Low beta = good for pairs trading
                        if abs(value) < 0.3:
                            item.setForeground(QColor(COLORS['positive']))
                        elif abs(value) > 0.7:
                            item.setForeground(QColor(COLORS['warning']))
                        else:
                            item.setForeground(QColor(COLORS['text_primary']))
                    else:
                        item.setForeground(QColor(COLORS['text_primary']))
                
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.benchmark_stats_table.setItem(row, col, item)
        
        # Update chart
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'benchmark_chart'):
            self._update_benchmark_chart(portfolio_returns, benchmark_returns, bench_ticker)
        
        # Update position summary
        open_positions = [p for p in self.portfolio if p.get('status', 'OPEN') == 'OPEN']
        total_capital = portfolio_df['capital'].iloc[-1] if len(portfolio_df) > 0 else 0
        
        position_info = []
        for p in open_positions:
            pair = p['pair']
            direction = p['direction']
            entry_date = p.get('entry_date', 'N/A')
            position_info.append(f"{pair} ({direction}, entry: {entry_date})")
        
        self.position_summary_label.setText(
            f"<b>Positioner i analys:</b> {', '.join(position_info) if position_info else 'Inga'}<br>"
            f"<b>Total investerat kapital:</b> {total_capital:,.0f} SEK"
        )
        
        # Update status
        data_days = len(portfolio_returns)
        self.benchmark_status_label.setText(
            f"✓ Analys uppdaterad | {len(open_positions)} positioner | "
            f"{data_days} handelsdagar | Benchmark: {bench_ticker} | "
            f"Från: {earliest_date.strftime('%Y-%m-%d')}"
        )
        
        # Save to cache
        self._benchmark_loaded = True
    
    def _get_benchmark_cache_file(self) -> str:
        """Get path to benchmark cache file."""
        return Paths.benchmark_cache_file()
    
    def _load_or_update_benchmark(self):
        """Load benchmark from cache if recent, otherwise update from history.
        
        Always updates the chart, but may use cached table data if recent.
        """
        cache_file = self._get_benchmark_cache_file()
        
        # Get current snapshot count
        current_snapshots = 0
        if PORTFOLIO_HISTORY_AVAILABLE and self.portfolio_history:
            current_snapshots = self.portfolio_history.get_snapshot_count()
        
        # Check if cache exists and is from today with matching snapshot count
        use_cache = False
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                
                cache_date = datetime.fromisoformat(cache['timestamp']).date()
                today = datetime.now().date()
                cache_snapshots = cache.get('n_snapshots', 0)
                
                # Use cache only for table data if from today AND snapshot count matches
                if cache_date == today and cache_snapshots == current_snapshots and current_snapshots > 0:
                    use_cache = True
                    self._apply_benchmark_cache(cache)
                    self._benchmark_loaded = True
            except Exception as e:
                print(f"Could not load benchmark cache: {e}")
        
        # Always run the full analysis to update the chart (even if table was cached)
        # This ensures the chart is always displayed
        self._update_benchmark_analysis()
        
        # Save fresh data to cache if not using cached version
        if not use_cache:
            self._save_benchmark_cache()
    
    def _save_benchmark_cache(self):
        """Save current benchmark table data to cache file."""
        cache_file = self._get_benchmark_cache_file()
        if not cache_file:
            return
        
        try:
            # Extract table data
            table_data = {}
            for row in range(self.benchmark_stats_table.rowCount()):
                metric_item = self.benchmark_stats_table.item(row, 0)
                if not metric_item:
                    continue
                metric_name = metric_item.text()
                row_data = {}
                for col in range(1, self.benchmark_stats_table.columnCount()):
                    header = self.benchmark_stats_table.horizontalHeaderItem(col)
                    if header:
                        cell_item = self.benchmark_stats_table.item(row, col)
                        if cell_item:
                            row_data[header.text()] = cell_item.text()
                table_data[metric_name] = row_data
            
            # Get snapshot count
            n_snapshots = 0
            if PORTFOLIO_HISTORY_AVAILABLE and self.portfolio_history:
                n_snapshots = self.portfolio_history.get_snapshot_count()
            
            cache = {
                'timestamp': datetime.now().isoformat(),
                'n_snapshots': n_snapshots,
                'table_data': table_data
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
            
            print(f"Saved benchmark cache to {cache_file}")
        except Exception as e:
            print(f"Error saving benchmark cache: {e}")
    
    def _apply_benchmark_cache(self, cache: dict):
        """Apply cached benchmark data to the table."""
        table_data = cache.get('table_data', {})
        
        for row in range(self.benchmark_stats_table.rowCount()):
            metric_item = self.benchmark_stats_table.item(row, 0)
            if not metric_item:
                continue
            metric_name = metric_item.text()
            
            if metric_name in table_data:
                row_data = table_data[metric_name]
                for col in range(1, self.benchmark_stats_table.columnCount()):
                    header = self.benchmark_stats_table.horizontalHeaderItem(col)
                    if header and header.text() in row_data:
                        value_text = row_data[header.text()]
                        item = QTableWidgetItem(value_text)
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        
                        # Color coding for returns
                        if value_text != "-":
                            try:
                                # Parse value and apply color
                                val_str = value_text.replace('%', '').replace('+', '')
                                val = float(val_str)
                                if metric_name in ['Portfolio Return', 'Excess Return (Alpha)',
                                                   'Sharpe Ratio (Portfolio)', 'Sortino Ratio']:
                                    if val > 0:
                                        item.setForeground(QColor(COLORS['positive']))
                                    elif val < 0:
                                        item.setForeground(QColor(COLORS['negative']))
                                elif metric_name == 'Benchmark Return':
                                    if val > 0:
                                        item.setForeground(QColor(COLORS['positive']))
                                    elif val < 0:
                                        item.setForeground(QColor(COLORS['negative']))
                            except (ValueError, TypeError):
                                pass
                        else:
                            item.setForeground(QColor(COLORS['text_muted']))
                        
                        self.benchmark_stats_table.setItem(row, col, item)
    
    def _update_benchmark_chart(self, portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series, 
                                bench_name: str):
        """Update the cumulative returns chart."""
        try:
            self.benchmark_chart.clear()
            
            # Align series
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_idx]
            bench_ret = benchmark_returns.loc[common_idx]
            
            if len(port_ret) < 2:
                return
            
            # Calculate cumulative returns (starting from 0%)
            port_cum = ((1 + port_ret).cumprod() - 1) * 100
            bench_cum = ((1 + bench_ret).cumprod() - 1) * 100
            
            # Convert to numeric x-axis
            x = np.arange(len(port_cum))
            
            # Plot portfolio
            port_pen = pg.mkPen(color=COLORS['accent'], width=2)
            self.benchmark_chart.plot(x, port_cum.values, pen=port_pen, name='Portfolio')
            
            # Plot benchmark  
            bench_pen = pg.mkPen(color=COLORS['info'], width=2)
            self.benchmark_chart.plot(x, bench_cum.values, pen=bench_pen, name=bench_name)
            
            # Zero line
            zero_pen = pg.mkPen(color=COLORS['text_muted'], width=1, style=Qt.DashLine)
            self.benchmark_chart.addLine(y=0, pen=zero_pen)
            
            # Set axis labels (show dates at intervals)
            date_labels = port_cum.index.strftime('%Y-%m-%d').tolist()
            n_ticks = min(6, len(date_labels))
            if n_ticks > 1:
                tick_positions = np.linspace(0, len(date_labels)-1, n_ticks, dtype=int)
                ticks = [(int(pos), date_labels[pos]) for pos in tick_positions]
                
                axis = self.benchmark_chart.getAxis('bottom')
                axis.setTicks([ticks])
            
        except Exception as e:
            print(f"Error updating benchmark chart: {e}")
    
    # ========================================================================
    # TAB 6: MARKOV CHAINS
    # ========================================================================

    def create_markov_chains_tab(self) -> QWidget:
        """Create Markov Chains tab with ticker selector, transition matrix, forecasts, and charts."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 10, 15, 10)

        # ── Top bar ───────────────────────────────────────────────────────
        top_bar = QFrame()
        top_bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {COLORS['bg_elevated']}, stop:1 {COLORS['bg_card']});
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
            }}
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(10)

        lbl = QLabel("TICKER")
        lbl.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;text-transform:uppercase;letter-spacing:1px;background:transparent;border:none;")
        top_layout.addWidget(lbl)

        self.markov_ticker_combo = QComboBox()
        self.markov_ticker_combo.setEditable(True)
        self.markov_ticker_combo.setInsertPolicy(QComboBox.NoInsert)
        self.markov_ticker_combo.setMinimumWidth(280)
        self.markov_ticker_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 4px;
                padding: 5px 10px;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 12px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['accent_dark']};
                border: 1px solid {COLORS['border_default']};
            }}
        """)
        self.markov_ticker_combo.setToolTip("Select or type a ticker symbol to analyze")
        top_layout.addWidget(self.markov_ticker_combo)

        # Populate from CSV
        self._populate_markov_tickers()

        self.markov_analyze_btn = QPushButton("ANALYZE")
        self.markov_analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent']};
                color: {COLORS['bg_darkest']};
                border: none;
                border-radius: 4px;
                padding: 6px 18px;
                font-weight: 600;
                font-size: 12px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {COLORS['accent_bright']}; }}
            QPushButton:pressed {{ background: {COLORS['accent_dark']}; }}
            QPushButton:disabled {{ background: {COLORS['text_disabled']}; color: {COLORS['text_muted']}; }}
        """)
        self.markov_analyze_btn.setToolTip("Run Markov chain analysis on selected ticker")
        self.markov_analyze_btn.clicked.connect(self.run_markov_analysis)
        top_layout.addWidget(self.markov_analyze_btn)

        self.markov_status_label = QLabel("")
        self.markov_status_label.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;background:transparent;border:none;")
        top_layout.addWidget(self.markov_status_label)

        top_layout.addStretch()

        self.markov_summary_state = CompactMetricCard("CURRENT STATE", "-")
        self.markov_summary_state.setToolTip("Current Markov state based on last completed week's return")
        self.markov_summary_expected = CompactMetricCard("E[RETURN]", "-")
        self.markov_summary_expected.setToolTip("Expected next-week return (probability-weighted average)")
        self.markov_summary_mixing = CompactMetricCard("MIXING TIME", "-")
        self.markov_summary_mixing.setToolTip("Weeks for chain to forget initial state. Short = memoryless")
        self.markov_summary_obs = CompactMetricCard("OBSERVATIONS", "-")
        self.markov_summary_obs.setToolTip("Number of weekly return observations used")
        for card in [self.markov_summary_state, self.markov_summary_expected,
                     self.markov_summary_mixing, self.markov_summary_obs]:
            card.setFixedWidth(140)
            top_layout.addWidget(card)

        main_layout.addWidget(top_bar)

        # ── Splitter: left panel + right charts ───────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['border_subtle']}; }}")

        # ── LEFT PANEL (scroll area) ─────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {COLORS['bg_dark']};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_default']};
                border-radius: 4px;
                min-height: 30px;
            }}
        """)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 10, 5)
        left_layout.setSpacing(8)

        # -- TRANSITION MATRIX --
        left_layout.addWidget(SectionHeader("TRANSITION MATRIX"))
        self.markov_matrix_frame = QFrame()
        self.markov_matrix_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)
        self.markov_matrix_grid = QGridLayout(self.markov_matrix_frame)
        self.markov_matrix_grid.setSpacing(2)
        self.markov_matrix_grid.setContentsMargins(6, 6, 6, 6)
        self.markov_matrix_labels = {}
        # Create 6×6 grid (header row + header col + 5×5 cells)
        state_shorts = ['SD', 'D', 'F', 'U', 'SU']
        # Corner
        corner = QLabel("")
        corner.setStyleSheet(f"background:transparent;border:none;")
        self.markov_matrix_grid.addWidget(corner, 0, 0)
        # Column headers
        for j, s in enumerate(state_shorts):
            h = QLabel(s)
            h.setAlignment(Qt.AlignCenter)
            h.setStyleSheet(f"color:{COLORS['accent']};font-size:11px;font-weight:600;background:transparent;border:none;")
            self.markov_matrix_grid.addWidget(h, 0, j + 1)
        # Row headers + cells
        for i, s in enumerate(state_shorts):
            rh = QLabel(s)
            rh.setAlignment(Qt.AlignCenter)
            rh.setStyleSheet(f"color:{COLORS['accent']};font-size:11px;font-weight:600;background:transparent;border:none;")
            self.markov_matrix_grid.addWidget(rh, i + 1, 0)
            for j in range(5):
                cell = QLabel("-")
                cell.setAlignment(Qt.AlignCenter)
                cell.setFixedSize(52, 28)
                cell.setStyleSheet(f"""
                    color: {COLORS['text_primary']};
                    font-size: 11px;
                    font-family: 'JetBrains Mono', monospace;
                    background: {COLORS['bg_elevated']};
                    border-radius: 3px;
                    border: none;
                """)
                cell.setToolTip(f"P({state_shorts[i]} → {state_shorts[j]})")
                self.markov_matrix_grid.addWidget(cell, i + 1, j + 1)
                self.markov_matrix_labels[(i, j)] = cell
        left_layout.addWidget(self.markov_matrix_frame)

        # -- NEXT WEEK FORECAST --
        left_layout.addWidget(SectionHeader("NEXT WEEK FORECAST"))
        forecast_row = QHBoxLayout()
        forecast_row.setSpacing(4)
        self.markov_forecast_cards = {}
        if MARKOV_AVAILABLE:
            for s_id in range(N_MARKOV_STATES):
                info = MARKOV_STATES[s_id]
                card = CompactMetricCard(f"P({info['short']})", "-")
                card.setToolTip(f"Probability of {info['name']} next week, based on transition matrix row for current state")
                card.setFixedWidth(80)
                forecast_row.addWidget(card)
                self.markov_forecast_cards[s_id] = card
        else:
            for s_id in range(5):
                names = ['SD', 'D', 'F', 'U', 'SU']
                card = CompactMetricCard(f"P({names[s_id]})", "-")
                card.setFixedWidth(80)
                forecast_row.addWidget(card)
                self.markov_forecast_cards[s_id] = card
        forecast_row.addStretch()
        left_layout.addLayout(forecast_row)

        # -- INTRAWEEK TRACKING --
        left_layout.addWidget(SectionHeader("INTRAWEEK TRACKING"))
        self.markov_tracking_frame = QFrame()
        self.markov_tracking_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)
        tracking_layout = QVBoxLayout(self.markov_tracking_frame)
        tracking_layout.setContentsMargins(10, 8, 10, 8)
        tracking_layout.setSpacing(4)
        self.markov_track_forecast_lbl = QLabel("Forecast: —")
        self.markov_track_forecast_lbl.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;background:transparent;border:none;")
        self.markov_track_actual_lbl = QLabel("Actual so far: —")
        self.markov_track_actual_lbl.setStyleSheet(f"color:{COLORS['text_secondary']};font-size:12px;background:transparent;border:none;")
        self.markov_track_status_lbl = QLabel("")
        self.markov_track_status_lbl.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;background:transparent;border:none;")
        tracking_layout.addWidget(self.markov_track_forecast_lbl)
        tracking_layout.addWidget(self.markov_track_actual_lbl)
        tracking_layout.addWidget(self.markov_track_status_lbl)
        left_layout.addWidget(self.markov_tracking_frame)

        # -- STATE STATISTICS --
        left_layout.addWidget(SectionHeader("STATE STATISTICS"))
        self.markov_stats_frame = QFrame()
        self.markov_stats_frame.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)
        stats_grid = QGridLayout(self.markov_stats_frame)
        stats_grid.setContentsMargins(8, 6, 8, 6)
        stats_grid.setSpacing(3)
        # Headers
        for col_idx, hdr in enumerate(["State", "Avg Ret", "Vol", "Freq", "Avg Dur"]):
            h = QLabel(hdr)
            h.setAlignment(Qt.AlignCenter)
            h.setStyleSheet(f"color:{COLORS['text_muted']};font-size:10px;font-weight:600;text-transform:uppercase;background:transparent;border:none;")
            stats_grid.addWidget(h, 0, col_idx)
        self.markov_stats_labels = {}
        state_shorts_full = ['SD', 'D', 'F', 'U', 'SU']
        for row in range(5):
            name_lbl = QLabel(state_shorts_full[row])
            name_lbl.setAlignment(Qt.AlignCenter)
            clr = MARKOV_STATES[row]['color'] if MARKOV_AVAILABLE else '#888'
            name_lbl.setStyleSheet(f"color:{clr};font-size:11px;font-weight:600;background:transparent;border:none;")
            stats_grid.addWidget(name_lbl, row + 1, 0)
            for col in range(4):
                v = QLabel("-")
                v.setAlignment(Qt.AlignCenter)
                v.setStyleSheet(f"color:{COLORS['text_primary']};font-size:11px;font-family:'JetBrains Mono',monospace;background:transparent;border:none;")
                stats_grid.addWidget(v, row + 1, col + 1)
                self.markov_stats_labels[(row, col)] = v
        left_layout.addWidget(self.markov_stats_frame)

        # -- DIAGNOSTICS --
        left_layout.addWidget(SectionHeader("DIAGNOSTICS"))
        diag_row = QHBoxLayout()
        diag_row.setSpacing(6)
        self.markov_diag_gap = CompactMetricCard("SPECTRAL GAP", "-")
        self.markov_diag_gap.setToolTip("1 − |λ₂|. Larger gap = faster mixing, less predictive. Smaller = more persistent")
        self.markov_diag_mix = CompactMetricCard("MIXING TIME", "-")
        self.markov_diag_mix.setToolTip("Weeks for chain to forget initial state")
        self.markov_diag_range = CompactMetricCard("DATA RANGE", "-")
        self.markov_diag_range.setToolTip("Date range of weekly return observations")
        self.markov_diag_stat = CompactMetricCard("STATIONARY DIST", "-")
        self.markov_diag_stat.setToolTip("Long-run equilibrium probability for each state")
        for d in [self.markov_diag_gap, self.markov_diag_mix, self.markov_diag_range, self.markov_diag_stat]:
            diag_row.addWidget(d)
        left_layout.addLayout(diag_row)

        left_layout.addStretch()
        scroll.setWidget(left_widget)
        splitter.addWidget(scroll)

        # ── RIGHT PANEL (charts) ─────────────────────────────────────────
        pg = get_pyqtgraph()
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        if pg is not None:
            # Chart 1: Weekly returns scatter
            DateAxisItem = get_date_axis_item_class()
            self.markov_returns_date_axis = DateAxisItem(orientation='bottom')
            self.markov_returns_plot = pg.PlotWidget(
                axisItems={'bottom': self.markov_returns_date_axis},
                title="WEEKLY RETURNS (STATE CLASSIFIED)"
            )
            self.markov_returns_plot.setBackground(COLORS['bg_card'])
            self.markov_returns_plot.showGrid(x=False, y=False, alpha=0.3)
            self.markov_returns_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))
            self.markov_returns_plot.setMouseEnabled(x=True, y=True)
            self.markov_returns_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            right_layout.addWidget(self.markov_returns_plot, stretch=3)

            # Chart 2: Transition heatmap
            self.markov_heatmap_plot = pg.PlotWidget(title="TRANSITION HEATMAP")
            self.markov_heatmap_plot.setBackground(COLORS['bg_card'])
            self.markov_heatmap_plot.setMouseEnabled(x=False, y=False)
            self.markov_heatmap_img = pg.ImageItem()
            self.markov_heatmap_plot.addItem(self.markov_heatmap_img)
            self.markov_heatmap_texts = []
            right_layout.addWidget(self.markov_heatmap_plot, stretch=2)

            # Bottom row: forecast bar + forecast vs actual
            bottom_row = QHBoxLayout()
            bottom_row.setSpacing(6)

            # Chart 3: Forecast bars
            self.markov_forecast_plot = pg.PlotWidget(title="NEXT WEEK FORECAST")
            self.markov_forecast_plot.setBackground(COLORS['bg_card'])
            self.markov_forecast_plot.setMouseEnabled(x=False, y=False)
            self.markov_forecast_plot.setLabel('left', 'Probability %')
            bottom_row.addWidget(self.markov_forecast_plot)

            # Chart 4: Multi-horizon forecast
            self.markov_horizon_plot = pg.PlotWidget(title="MULTI-HORIZON FORECAST")
            self.markov_horizon_plot.setBackground(COLORS['bg_card'])
            self.markov_horizon_plot.setMouseEnabled(x=False, y=False)
            self.markov_horizon_plot.setLabel('left', 'Probability %')
            self.markov_horizon_plot.addLegend(offset=(10, 10))
            bottom_row.addWidget(self.markov_horizon_plot)

            right_layout.addLayout(bottom_row, stretch=2)
        else:
            no_pg = QLabel("pyqtgraph not available — charts disabled")
            no_pg.setAlignment(Qt.AlignCenter)
            no_pg.setStyleSheet(f"color:{COLORS['text_muted']};font-size:13px;")
            right_layout.addWidget(no_pg)

        splitter.addWidget(right_widget)
        splitter.setSizes([280, 700])
        main_layout.addWidget(splitter, stretch=1)

        return tab

    def _populate_markov_tickers(self):
        """Populate the Markov ticker combo from the matched tickers CSV."""
        try:
            csv_path = find_matched_tickers_csv()
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
            items = []
            if 'Ticker' in df.columns and 'Underliggande tillgång' in df.columns:
                for _, row in df.iterrows():
                    ticker = str(row['Ticker']).strip()
                    name = str(row['Underliggande tillgång']).strip()
                    if ticker and ticker.lower() != 'ticker':
                        items.append(f"{ticker} — {name}")
            elif 'Ticker' in df.columns:
                for t in df['Ticker'].dropna().astype(str).str.strip():
                    if t and t.lower() != 'ticker':
                        items.append(t)
            if items:
                self.markov_ticker_combo.addItems(sorted(items))
        except Exception as e:
            print(f"[Markov] Error loading tickers: {e}")

    # ── Analysis runner ───────────────────────────────────────────────────

    def run_markov_analysis(self):
        """Start Markov chain analysis for the selected ticker."""
        if not MARKOV_AVAILABLE:
            self.markov_status_label.setText("markov_chain.py not found")
            return
        if self._markov_running:
            self.markov_status_label.setText("Analysis already running...")
            return

        raw = self.markov_ticker_combo.currentText().strip()
        if not raw:
            self.markov_status_label.setText("Select a ticker first")
            return
        ticker = raw.split(" — ")[0].strip()

        self._markov_running = True
        self.markov_analyze_btn.setEnabled(False)
        self.markov_status_label.setText(f"Analyzing {ticker}...")

        self._markov_thread = QThread()
        self._markov_worker = MarkovChainWorker(ticker)
        self._markov_worker.moveToThread(self._markov_thread)

        self._markov_thread.started.connect(self._markov_worker.run)
        self._markov_worker.progress.connect(self._on_markov_progress)
        self._markov_worker.result.connect(self._on_markov_result)
        self._markov_worker.error.connect(self._on_markov_error)
        self._markov_worker.finished.connect(self._on_markov_finished)
        self._markov_worker.finished.connect(self._markov_thread.quit)

        self._markov_thread.start()

    def _on_markov_progress(self, pct: int, msg: str):
        self.markov_status_label.setText(f"[{pct}%] {msg}")

    def _on_markov_error(self, msg: str):
        self.markov_status_label.setText(f"Error: {msg}")
        self.markov_status_label.setStyleSheet(f"color:{COLORS['negative']};font-size:11px;background:transparent;border:none;")

    def _on_markov_finished(self):
        self._markov_running = False
        self.markov_analyze_btn.setEnabled(True)

    def _on_markov_result(self, result):
        """Handle completed Markov analysis — update all UI elements."""
        self._markov_result = result
        self.markov_status_label.setText(f"Analysis complete — {result.ticker}")
        self.markov_status_label.setStyleSheet(f"color:{COLORS['positive']};font-size:11px;background:transparent;border:none;")
        self._update_markov_summary(result)
        self._update_markov_matrix(result)
        self._update_markov_forecast(result)
        self._update_markov_tracking(result)
        self._update_markov_state_stats(result)
        self._update_markov_diagnostics(result)
        self._update_markov_charts(result)

    # ── UI updaters ───────────────────────────────────────────────────────

    def _update_markov_summary(self, r):
        """Update top bar summary cards."""
        if MARKOV_AVAILABLE:
            state_info = MARKOV_STATES[r.current_state]
            self.markov_summary_state.set_value(state_info['short'], state_info['color'])
        else:
            self.markov_summary_state.set_value(str(r.current_state))

        er_color = COLORS['positive'] if r.expected_return >= 0 else COLORS['negative']
        self.markov_summary_expected.set_value(f"{r.expected_return * 100:+.2f}%", er_color)
        self.markov_summary_mixing.set_value(f"{r.mixing_time:.1f}w")
        self.markov_summary_obs.set_value(str(r.n_observations))

    def _update_markov_matrix(self, r):
        """Update transition matrix grid cells with intensity coloring."""
        T = r.transition_matrix
        for i in range(5):
            for j in range(5):
                val = T[i, j]
                cell = self.markov_matrix_labels.get((i, j))
                if cell is None:
                    continue
                cell.setText(f"{val:.0%}")
                # Intensity: 0→bg_elevated, high→amber
                intensity = min(val * 2.5, 1.0)
                bg_r = int(13 + intensity * (212 - 13))
                bg_g = int(13 + intensity * (165 - 13))
                bg_b = int(13 + intensity * (116 - 13))
                text_color = COLORS['text_primary'] if intensity < 0.6 else COLORS['bg_darkest']
                cell.setStyleSheet(f"""
                    color: {text_color};
                    font-size: 11px;
                    font-family: 'JetBrains Mono', monospace;
                    background: rgb({bg_r},{bg_g},{bg_b});
                    border-radius: 3px;
                    border: none;
                """)

    def _update_markov_forecast(self, r):
        """Update forecast probability cards."""
        for s_id in range(5):
            card = self.markov_forecast_cards.get(s_id)
            if card is None:
                continue
            prob = r.forecast_probs[s_id]
            clr = MARKOV_STATES[s_id]['color'] if MARKOV_AVAILABLE else COLORS['text_primary']
            card.set_value(f"{prob:.0%}", clr)

    def _update_markov_tracking(self, r):
        """Update intraweek tracking labels."""
        # Forecast from current state
        if MARKOV_AVAILABLE:
            most_likely = int(np.argmax(r.forecast_probs))
            ml_info = MARKOV_STATES[most_likely]
            self.markov_track_forecast_lbl.setText(
                f"Forecast: Most likely → {ml_info['name']} ({r.forecast_probs[most_likely]:.0%})"
            )
        else:
            self.markov_track_forecast_lbl.setText("Forecast: —")

        # Actual intraweek
        iwr = r.current_intraweek_return
        iw_color = COLORS['positive'] if iwr >= 0 else COLORS['negative']
        if MARKOV_AVAILABLE:
            iw_state_info = MARKOV_STATES[r.current_intraweek_state]
            self.markov_track_actual_lbl.setText(
                f"Actual so far: {iwr * 100:+.2f}% ({iw_state_info['short']})"
            )
            self.markov_track_actual_lbl.setStyleSheet(
                f"color:{iw_color};font-size:12px;font-weight:600;background:transparent;border:none;"
            )
            # Tracking indicator
            if r.current_intraweek_state == int(np.argmax(r.forecast_probs)):
                self.markov_track_status_lbl.setText("● Tracking forecast")
                self.markov_track_status_lbl.setStyleSheet(f"color:{COLORS['positive']};font-size:11px;background:transparent;border:none;")
            else:
                self.markov_track_status_lbl.setText("○ Diverging from forecast")
                self.markov_track_status_lbl.setStyleSheet(f"color:{COLORS['warning']};font-size:11px;background:transparent;border:none;")
        else:
            self.markov_track_actual_lbl.setText(f"Actual so far: {iwr * 100:+.2f}%")

    def _update_markov_state_stats(self, r):
        """Update state statistics table."""
        for s_id in range(5):
            st = r.state_stats.get(s_id, {})
            avg_ret = st.get('avg_return', 0)
            vol = st.get('volatility', 0)
            freq = st.get('frequency', 0)
            avg_dur = st.get('avg_duration', 0)

            vals = [
                (f"{avg_ret * 100:+.2f}%", COLORS['positive'] if avg_ret >= 0 else COLORS['negative']),
                (f"{vol * 100:.2f}%", COLORS['text_primary']),
                (f"{freq:.0%}", COLORS['text_primary']),
                (f"{avg_dur:.1f}w", COLORS['text_primary']),
            ]
            for col, (txt, clr) in enumerate(vals):
                lbl = self.markov_stats_labels.get((s_id, col))
                if lbl:
                    lbl.setText(txt)
                    lbl.setStyleSheet(f"color:{clr};font-size:11px;font-family:'JetBrains Mono',monospace;background:transparent;border:none;")

    def _update_markov_diagnostics(self, r):
        """Update diagnostic cards."""
        self.markov_diag_gap.set_value(f"{r.eigenvalue_gap:.4f}")
        self.markov_diag_mix.set_value(f"{r.mixing_time:.1f} weeks")
        self.markov_diag_range.set_value(f"{r.data_start}\n{r.data_end}")
        # Stationary dist compact
        if MARKOV_AVAILABLE:
            parts = [f"{MARKOV_STATES[i]['short']}:{r.stationary_dist[i]:.0%}" for i in range(5)]
        else:
            parts = [f"S{i}:{r.stationary_dist[i]:.0%}" for i in range(5)]
        self.markov_diag_stat.set_value(" ".join(parts))

    # ── Chart rendering ───────────────────────────────────────────────────

    def _update_markov_charts(self, r):
        """Render all 4 Markov chain pyqtgraph charts."""
        pg = get_pyqtgraph()
        if pg is None:
            return

        self._render_markov_returns_chart(r, pg)
        self._render_markov_heatmap(r, pg)
        self._render_markov_forecast_bars(r, pg)
        self._render_markov_horizon_chart(r, pg)

    def _render_markov_returns_chart(self, r, pg):
        """Chart 1: Weekly returns scatter colored by state."""
        plot = self.markov_returns_plot
        plot.clear()
        plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))

        dates = r.weekly_dates
        returns = r.weekly_returns
        states = r.state_sequence
        n = len(returns)

        # Add threshold lines
        for th in r.thresholds:
            plot.addLine(y=th, pen=pg.mkPen('#444444', width=1, style=Qt.DashLine))

        # Set date axis
        self.markov_returns_date_axis.set_dates(
            pd.DatetimeIndex(dates) if not isinstance(dates, pd.DatetimeIndex) else dates
        )

        # Scatter by state
        state_colors = {
            0: '#ef4444', 1: '#f59e0b', 2: '#a0a0a0', 3: '#22c55e', 4: '#3b82f6'
        }
        if MARKOV_AVAILABLE:
            state_colors = {k: v['color'] for k, v in MARKOV_STATES.items()}

        for s_id in range(5):
            mask = states == s_id
            if not mask.any():
                continue
            x = np.where(mask)[0].astype(float)
            y = returns[mask]
            short = MARKOV_STATES[s_id]['short'] if MARKOV_AVAILABLE else f"S{s_id}"
            color = state_colors.get(s_id, '#888888')
            scatter = pg.ScatterPlotItem(
                x=x, y=y, size=6,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(color),
                name=short
            )
            plot.addItem(scatter)

        plot.addLegend(offset=(10, 10))

    def _render_markov_heatmap(self, r, pg):
        """Chart 2: Transition matrix as heatmap."""
        plot = self.markov_heatmap_plot
        plot.clear()
        self.markov_heatmap_img = pg.ImageItem()
        plot.addItem(self.markov_heatmap_img)

        T = r.transition_matrix

        # Create amber colormap: dark → amber
        # Transpose for image orientation (row=y, col=x)
        img_data = T.copy()

        # Apply colormap manually: map 0→dark, 1→amber
        # ImageItem expects [x, y] so transpose
        self.markov_heatmap_img.setImage(img_data.T, levels=[0, np.max(T) + 0.01])

        # Create amber colormap
        colors = [
            (10, 10, 10),       # dark (0)
            (80, 60, 30),       # dark amber (0.25)
            (170, 130, 70),     # mid amber (0.5)
            (212, 165, 116),    # accent amber (0.75)
            (232, 184, 109),    # bright amber (1.0)
        ]
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        cmap = pg.ColorMap(positions, colors)
        lut = cmap.getLookupTable(nPts=256)
        self.markov_heatmap_img.setLookupTable(lut)

        # Remove old text items
        for txt in self.markov_heatmap_texts:
            plot.removeItem(txt)
        self.markov_heatmap_texts = []

        # Add text annotations
        for i in range(5):
            for j in range(5):
                txt = pg.TextItem(f"{T[i, j]:.0%}", color='#ffffff', anchor=(0.5, 0.5))
                txt.setPos(j + 0.5, i + 0.5)
                txt.setFont(QFont('JetBrains Mono', 9))
                plot.addItem(txt)
                self.markov_heatmap_texts.append(txt)

        # Axis labels
        state_shorts = ['SD', 'D', 'F', 'U', 'SU']
        x_axis = plot.getAxis('bottom')
        x_axis.setTicks([[(i + 0.5, s) for i, s in enumerate(state_shorts)]])
        y_axis = plot.getAxis('left')
        y_axis.setTicks([[(i + 0.5, s) for i, s in enumerate(state_shorts)]])
        plot.setXRange(0, 5)
        plot.setYRange(0, 5)

    def _render_markov_forecast_bars(self, r, pg):
        """Chart 3: Next week forecast bar chart."""
        plot = self.markov_forecast_plot
        plot.clear()

        probs = r.forecast_probs * 100  # percent
        state_colors = {
            0: '#ef4444', 1: '#f59e0b', 2: '#a0a0a0', 3: '#22c55e', 4: '#3b82f6'
        }
        if MARKOV_AVAILABLE:
            state_colors = {k: v['color'] for k, v in MARKOV_STATES.items()}

        for i in range(5):
            bar = pg.BarGraphItem(
                x=[i], height=[probs[i]], width=0.6,
                brush=pg.mkBrush(state_colors.get(i, '#888')),
                pen=pg.mkPen(None)
            )
            plot.addItem(bar)
            # Label on top
            txt = pg.TextItem(f"{probs[i]:.1f}%", color='#e8e8e8', anchor=(0.5, 1.0))
            txt.setPos(i, probs[i])
            txt.setFont(QFont('JetBrains Mono', 9))
            plot.addItem(txt)

        # Baseline at 20% (uniform)
        plot.addLine(y=20, pen=pg.mkPen('#666666', width=1, style=Qt.DashLine))

        state_shorts = ['SD', 'D', 'F', 'U', 'SU']
        x_axis = plot.getAxis('bottom')
        x_axis.setTicks([[(i, s) for i, s in enumerate(state_shorts)]])
        plot.setXRange(-0.5, 4.5)
        plot.setYRange(0, max(probs) * 1.2 + 5)

    def _render_markov_horizon_chart(self, r, pg):
        """Chart 4: Multi-horizon forecast (1w, 2w, 4w) as grouped bars."""
        plot = self.markov_horizon_plot
        plot.clear()

        horizons = [
            ('1W', r.forecast_probs),
            ('2W', r.forecast_2w),
            ('4W', r.forecast_4w),
        ]
        bar_width = 0.25
        offsets = [-bar_width, 0, bar_width]
        horizon_colors = [COLORS['accent'], COLORS['info'], COLORS['text_muted']]

        for h_idx, (label, probs) in enumerate(horizons):
            x = np.arange(5) + offsets[h_idx]
            bar = pg.BarGraphItem(
                x=x, height=probs * 100, width=bar_width,
                brush=pg.mkBrush(horizon_colors[h_idx]),
                pen=pg.mkPen(None),
                name=label
            )
            plot.addItem(bar)

        state_shorts = ['SD', 'D', 'F', 'U', 'SU']
        x_axis = plot.getAxis('bottom')
        x_axis.setTicks([[(i, s) for i, s in enumerate(state_shorts)]])
        plot.setXRange(-0.5, 4.5)
        max_val = max(r.forecast_probs.max(), r.forecast_2w.max(), r.forecast_4w.max()) * 100
        plot.setYRange(0, max_val * 1.2 + 5)
        plot.addLegend(offset=(10, 10))

    # ========================================================================
    # WORLD MAP
    # ========================================================================

    def update_treemap_heatmap(self, items: list):
        """Render Finviz-style treemap heatmap using Plotly in QWebEngineView.

        Args:
            items: List of dicts with {market, symbol, name, price, change, change_pct}
        """
        if not WEBENGINE_AVAILABLE or not hasattr(self, 'map_widget'):
            return

        # Market display names
        MARKET_NAMES = {
            'AMERICA': 'America',
            'EUROPE': 'Europe',
            'MIDDLE EAST': 'Middle East',
            'AFRICA': 'Africa',
            'ASIA': 'Asia',
            'OCEANIA': 'Oceania',
            'CURRENCIES': 'Currencies',
            'COMMODITIES': 'Commodities',
            'YIELDS': 'Yields',
        }

        # Approximate market cap / importance weights ($T) for tile sizing
        MARKET_WEIGHTS = {
            # America
            '^GSPC': 50, '^NDX': 25, '^DJI': 15, '^RUT': 3,
            '^GSPTSE': 3, '^BVSP': 1,
            # Europe
            '^FTSE': 3, '^FCHI': 3.5, '^STOXX50E': 10, '^N100': 5,
            '^GDAXI': 2.5, '^OMX': 1,
            'OBX.OL': 0.4, '^OMXC25': 0.6, '^OMXH25': 0.3,
            # Asia
            '^N225': 6, '^HSI': 4, '000001.SS': 8, '^KS11': 2,
            '^TWII': 2, '399106.SZ': 2, '^BSESN': 4,
            # Oceania
            '^AXJO': 1.5,
            # Currencies
            'EURUSD=X': 2, 'EURSEK=X': 0.5, 'GBPUSD=X': 1,
            'USDJPY=X': 1, 'USDSEK=X': 0.3,
            # Commodities
            'GC=F': 5, 'SI=F': 1, 'CL=F': 4, 'BZ=F': 3, 'NG=F': 1.5, 'HG=F': 1.5,
            # Yields
            '^TNX': 3, '^FVX': 2, '^TYX': 2, '^IRX': 1,
            # Crypto
            'BTC-USD': 3, 'SOL-USD': 1, 'XRP-USD': 1, 'ETH-USD': 1,
        }

        # Build treemap arrays with unique IDs (no root node - regions are top-level)
        ids = []
        labels = []
        names_arr = []  # Rena namn för JS-uppdatering av labels
        parents = []
        values = []
        text = []
        colors = []
        font_sizes = []
        line_widths = []   # Per-tile border width (tjockare för regioner)
        line_colors = []   # Per-tile border color

        # Extra data for overlay & tooltip
        histories = {}       # symbol → [[date, close], ...]
        ohlc_histories = {}  # symbol → [[date, O, H, L, C], ...]
        region_items = {}    # region_key → [{name, symbol, price_str, change_pct}, ...]
        item_regions = {}    # symbol → region display name

        # Group items by market
        markets = {}
        for item in items:
            m = item.get('market', 'OTHER')
            if m not in markets:
                markets[m] = []
            markets[m].append(item)

        # Add market categories and instruments
        for market_key, market_items in markets.items():
            display_name = MARKET_NAMES.get(market_key, market_key)
            region_id = f'region:{market_key}'
            region_items[market_key] = []

            # Placeholder for region - value will be set to sum of children
            region_idx = len(ids)
            ids.append(region_id)
            labels.append(f'<br>{display_name.upper()}')
            names_arr.append('')  # Regioner har inga individuella namn
            parents.append('')
            values.append(0)  # Updated below
            text.append('')
            colors.append(0)
            font_sizes.append(16)
            line_widths.append(2)
            line_colors.append('#444444')

            # Add individual instruments
            region_total = 0
            for item in market_items:
                name = item.get('name', item.get('symbol', ''))
                symbol = item.get('symbol', '')
                price = item.get('price', 0)
                change_pct = item.get('change_pct', 0)
                change_price = item.get('change', 0)
                weight = MARKET_WEIGHTS.get(symbol, 0.5)
                region_total += weight

                # Format price with appropriate precision
                if price >= 100:
                    price_str = f'{price:,.2f}'
                elif price > 0:
                    price_str = f'{price:.4f}'
                else:
                    price_str = 'N/A'

                change_sign = '+' if change_pct >= 0 else ''
                display_ticker = symbol.lstrip('^').replace('=X', '').replace('=F', '')
                # Rad 3: prisförändring + %
                change_str = f'{change_price:+.2f}' if change_price != 0 else ''
                tile_text = f'{change_str} ({change_sign}{change_pct:.2f}%)'

                ids.append(f'item:{symbol}')

                # Rad 1: Namn | Rad 2: Pris (via label) | Rad 3: Förändring (via text)
                label_text = f"<b>{name}</b><br>{price_str}<br>"
                labels.append(label_text)
                names_arr.append(name)

                parents.append(region_id)
                values.append(weight)
                text.append(tile_text)
                colors.append(change_pct)
                # Fontstorlek proportionell mot market cap-vikt (log-skala)
                fs = max(7, min(22, round(10 + 2.5 * math.log2(max(weight, 0.1)))))
                font_sizes.append(fs)
                line_widths.append(0.5)
                line_colors.append('#0a0a0a')

                # Collect history for sparklines & candlestick charts
                hist = item.get('history', [])
                if hist:
                    histories[symbol] = hist
                ohlc = item.get('ohlc_history', [])
                if ohlc:
                    ohlc_histories[symbol] = ohlc
                # Region item data for overlay table
                region_items[market_key].append({
                    'name': name, 'symbol': symbol,
                    'price_str': price_str, 'change_pct': change_pct,
                })
                item_regions[symbol] = {'display': display_name.upper(), 'key': market_key}

            # Set region value to exact sum of children (required for branchvalues: 'total')
            values[region_idx] = region_total

        # If empty, show placeholder
        if len(ids) == 0:
            html = '''<!DOCTYPE html><html><body style="background:#0a0a0a;color:#444;display:flex;align-items:center;justify-content:center;height:100%;margin:0;font-family:monospace;font-size:14px;">
            <div>Loading market data...</div></body></html>'''
            self.map_widget.setHtml(html)
            return

        # Sanitize any inf/nan in numeric arrays before JSON serialization
        colors = [c if math.isfinite(c) else 0.0 for c in colors]
        values = [v if math.isfinite(v) else 0.5 for v in values]

        # Build JSON for JS
        try:
            data_json = json.dumps({
                'ids': ids,
                'labels': labels,
                'names': names_arr,
                'parents': parents,
                'values': values,
                'text': text,
                'colors': colors,
                'font_sizes': font_sizes,
                'line_widths': line_widths,
                'line_colors': line_colors,
                'histories': histories,
                'ohlc_histories': ohlc_histories,
                'region_items': region_items,
                'item_regions': item_regions,
            })
        except (ValueError, TypeError) as e:
            print(f"[MarketWatch] JSON serialization error: {e}, falling back without OHLC")
            data_json = json.dumps({
                'ids': ids, 'labels': labels, 'names': names_arr,
                'parents': parents,
                'values': values, 'text': text, 'colors': colors,
                'font_sizes': font_sizes,
                'line_widths': line_widths, 'line_colors': line_colors,
                'histories': {},
                'ohlc_histories': {}, 'region_items': region_items,
                'item_regions': item_regions,
            }, default=str)

        html = f'''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            background: #0a0a0a;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: Arial, Helvetica, sans-serif;
        }}
        #map {{ width: 100%; height: 100%; }}
        #loading {{
            color: #555;
            font-family: monospace;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }}
        /* Hover tooltip */
        #tooltip {{
            display: none;
            position: fixed;
            z-index: 1000;
            pointer-events: none;
            background: #111111;
            border: 1px solid #444;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 12px;
            color: #e8e8e8;
            max-width: 300px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.6);
        }}
        #tooltip .tt-region {{ color: #888; font-size: 10px; text-transform: uppercase; margin-bottom: 4px; }}
        #tooltip .tt-name {{ font-weight: 700; font-size: 13px; margin-bottom: 4px; }}
        #tooltip .tt-spark {{ margin: 4px 0; }}
        #tooltip .tt-price {{ font-size: 13px; font-weight: 600; }}
        #tooltip .tt-change {{ font-size: 12px; font-weight: 600; }}
        .tt-pos {{ color: #33ff33; }}
        .tt-neg {{ color: #ff3333; }}
        /* Detail overlay */
        #overlay-backdrop {{
            display: none;
            position: fixed;
            z-index: 998;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5);
        }}
        #detail-overlay {{
            display: none;
            position: fixed;
            z-index: 999;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: #111;
            border: 1px solid #333;
            border-radius: 6px;
            min-width: 380px;
            max-width: 520px;
            max-height: 80vh;
            overflow-y: auto;
            color: #e8e8e8;
            box-shadow: 0 8px 32px rgba(0,0,0,0.8);
        }}
        #detail-overlay::-webkit-scrollbar {{ width: 6px; }}
        #detail-overlay::-webkit-scrollbar-track {{ background: #111; }}
        #detail-overlay::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}
        .ov-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid #333;
            background: #0d0d0d;
        }}
        .ov-region {{ font-size: 14px; font-weight: 700; color: #888; text-transform: uppercase; }}
        .ov-close {{
            cursor: pointer;
            color: #666;
            font-size: 18px;
            width: 28px; height: 28px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 4px;
            border: none;
            background: transparent;
        }}
        .ov-close:hover {{ color: #e8e8e8; background: #222; }}
        .ov-main {{
            padding: 12px 16px;
            border-bottom: 1px solid #222;
        }}
        .ov-main-name {{ font-size: 15px; font-weight: 700; color: #e8e8e8; }}
        .ov-main-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 6px;
        }}
        .ov-main-price {{ font-size: 16px; font-weight: 700; }}
        .ov-main-change {{ font-size: 14px; font-weight: 600; }}
        .ov-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .ov-table tr {{ border-bottom: 1px solid #1a1a1a; }}
        .ov-table tr:hover {{ background: #1a1a1a; }}
        .ov-table td {{
            padding: 8px 16px;
            font-size: 12px;
        }}
        .ov-table .ov-tname {{ color: #ccc; font-weight: 600; }}
        .ov-table .ov-tprice {{ color: #aaa; text-align: right; }}
        .ov-table .ov-tchange {{ text-align: right; font-weight: 600; min-width: 70px; }}
    </style>
</head>
<body>
    <div id="map"><div id="loading">Loading market heatmap...</div></div>
    <div id="tooltip"></div>
    <div id="overlay-backdrop" onclick="hideOverlay()"></div>
    <div id="detail-overlay"><div id="overlay-content"></div></div>
    <script>
        console.log('Treemap script starting...');
        const d = {data_json};
        console.log('Data loaded: ' + d.labels.length + ' labels');

        /* SVG sparkline from history data; color based on daily change_pct */
        function makeSpark(hist, w, h, changePct) {{
            if (!hist || hist.length < 2) return '';
            var closes = hist.map(function(p) {{ return p[1]; }});
            var mn = Math.min.apply(null, closes);
            var mx = Math.max.apply(null, closes);
            var rng = mx - mn || 1;
            var pts = closes.map(function(v, i) {{
                var x = (i / (closes.length - 1)) * w;
                var y = h - ((v - mn) / rng) * (h - 2) - 1;
                return x.toFixed(1) + ',' + y.toFixed(1);
            }}).join(' ');
            var clr = (changePct !== undefined ? changePct >= 0 : closes[closes.length - 1] >= closes[0]) ? '#33ff33' : '#ff3333';
            return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">' +
                '<polyline points="' + pts + '" fill="none" stroke="' + clr + '" stroke-width="1.5"/></svg>';
        }}

        /* SVG candlestick chart from OHLC data: [[date, O, H, L, C], ...] */
        function makeCandle(ohlc, w, h) {{
            if (!ohlc || ohlc.length < 2) return '';
            var allH = ohlc.map(function(c) {{ return c[2]; }});
            var allL = ohlc.map(function(c) {{ return c[3]; }});
            var mn = Math.min.apply(null, allL);
            var mx = Math.max.apply(null, allH);
            var rng = mx - mn || 1;
            var pad = 2;
            var cw = Math.max(1, ((w - pad * 2) / ohlc.length) * 0.7);
            var gap = (w - pad * 2) / ohlc.length;
            var svg = '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '">';
            /* Grid lines */
            for (var g = 0; g < 4; g++) {{
                var gy = Math.round(h * g / 4);
                svg += '<line x1="0" y1="' + gy + '" x2="' + w + '" y2="' + gy + '" stroke="#222" stroke-width="0.5"/>';
            }}
            for (var i = 0; i < ohlc.length; i++) {{
                var o = ohlc[i][1], hi = ohlc[i][2], lo = ohlc[i][3], c = ohlc[i][4];
                var x = pad + i * gap + gap / 2;
                var yH = h - ((hi - mn) / rng) * (h - 4) - 2;
                var yL = h - ((lo - mn) / rng) * (h - 4) - 2;
                var yO = h - ((o - mn) / rng) * (h - 4) - 2;
                var yC = h - ((c - mn) / rng) * (h - 4) - 2;
                var bull = c >= o;
                var clr = bull ? '#33ff33' : '#ff3333';
                var bodyTop = Math.min(yO, yC);
                var bodyH = Math.max(1, Math.abs(yC - yO));
                /* Wick */
                svg += '<line x1="' + x.toFixed(1) + '" y1="' + yH.toFixed(1) + '" x2="' + x.toFixed(1) + '" y2="' + yL.toFixed(1) + '" stroke="' + clr + '" stroke-width="0.8"/>';
                /* Body */
                svg += '<rect x="' + (x - cw / 2).toFixed(1) + '" y="' + bodyTop.toFixed(1) + '" width="' + cw.toFixed(1) + '" height="' + bodyH.toFixed(1) + '" fill="' + (bull ? clr : clr) + '" stroke="' + clr + '" stroke-width="0.5"/>';
            }}
            svg += '</svg>';
            return svg;
        }}

        /* Format change with sign and color class */
        function fmtChange(pct) {{
            var sign = pct >= 0 ? '+' : '';
            var cls = pct >= 0 ? 'tt-pos' : 'tt-neg';
            return '<span class="' + cls + '">' + sign + pct.toFixed(2) + '%</span>';
        }}

        /* Show hover tooltip near cursor */
        function showTooltip(evt, ticker) {{
            var info = d.item_regions[ticker];
            if (!info) return;
            var region = info.display;
            var key = info.key;
            var items = d.region_items[key] || [];
            var item = null;
            for (var i = 0; i < items.length; i++) {{
                if (items[i].symbol === ticker) {{ item = items[i]; break; }}
            }}
            if (!item) return;
            /* Use intraday ticks for sparkline, fallback to d.histories */
            var ohlc = (d.intraday_ohlc || {{}})[ticker];
            var spark = '';
            if (ohlc && ohlc.length >= 2) {{
                var tickHist = ohlc.map(function(b) {{ return [b[0], b[4]]; }});
                spark = makeSpark(tickHist, 120, 30, item.change_pct);
            }} else {{
                var hist = d.histories[ticker];
                spark = makeSpark(hist, 120, 30, item.change_pct);
            }}
            var tt = document.getElementById('tooltip');
            tt.innerHTML =
                '<div class="tt-region">' + region + '</div>' +
                '<div class="tt-name">' + item.name + '</div>' +
                (spark ? '<div class="tt-spark">' + spark + '</div>' : '') +
                '<div class="tt-price">' + item.price_str + '</div>' +
                '<div class="tt-change">' + fmtChange(item.change_pct) + '</div>';
            tt.style.display = 'block';
            /* Position near cursor */
            var x = (evt.clientX || evt.pageX || 0) + 14;
            var y = (evt.clientY || evt.pageY || 0) + 14;
            if (x + 310 > window.innerWidth) x = x - 330;
            if (y + 160 > window.innerHeight) y = y - 170;
            tt.style.left = x + 'px';
            tt.style.top = y + 'px';
        }}
        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        /* Show detail overlay */
        /* Format large numbers with K/M/B suffixes */
        function fmtVol(v) {{
            if (!v || v === 0) return '-';
            if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
            if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
            if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
            return v.toString();
        }}
        function fmtP(v) {{
            if (!v || v === 0) return '-';
            return v >= 100 ? v.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}}) : v.toFixed(4);
        }}

        function showOverlay(ticker) {{
            var info = d.item_regions[ticker];
            if (!info) return;
            var region = info.display;
            var key = info.key;
            var items = d.region_items[key] || [];
            var clickedItem = null;
            for (var i = 0; i < items.length; i++) {{
                if (items[i].symbol === ticker) {{ clickedItem = items[i]; break; }}
            }}
            if (!clickedItem) return;

            /* Prefer intraday candle chart from WS ticks, fallback to historic OHLC/sparkline */
            var intradayOhlc = (d.intraday_ohlc || {{}})[ticker];
            var chart = '';
            if (intradayOhlc && intradayOhlc.length >= 2) {{
                chart = makeCandle(intradayOhlc, 460, 120);
            }}
            if (!chart) {{
                var ohlc = d.ohlc_histories[ticker];
                chart = ohlc ? makeCandle(ohlc, 460, 120) : '';
            }}
            if (!chart) {{
                var hist = d.histories[ticker];
                chart = makeSpark(hist, 460, 60, clickedItem.change_pct);
            }}

            var html = '<div class="ov-header">' +
                '<span class="ov-region">' + region + '</span>' +
                '<button class="ov-close" onclick="hideOverlay()">&times;</button></div>';
            html += '<div class="ov-main">';
            html += '<div class="ov-main-name">' + clickedItem.name + '</div>';
            if (chart) html += '<div style="margin:8px 0;">' + chart + '</div>';
            html += '<div class="ov-main-row">';
            html += '<span class="ov-main-price">' + clickedItem.price_str + '</span>';
            html += '<span class="ov-main-change">' + fmtChange(clickedItem.change_pct) + '</span>';
            html += '</div>';

            /* WS extra data row: Open | Day High | Day Low | Prev Close | Volume */
            var wi = (d.ws_info || {{}})[ticker];
            if (wi) {{
                html += '<div style="display:flex;gap:6px;margin-top:8px;flex-wrap:wrap;">';
                var fields = [
                    ['Open', wi.open_price],
                    ['High', wi.day_high],
                    ['Low', wi.day_low],
                    ['Prev', wi.previous_close],
                    ['Vol', wi.day_volume]
                ];
                for (var f = 0; f < fields.length; f++) {{
                    var lbl = fields[f][0];
                    var val = fields[f][1];
                    var vs = (lbl === 'Vol') ? fmtVol(val) : fmtP(val);
                    html += '<div style="background:#1a1a1a;border-radius:4px;padding:4px 8px;font-size:11px;text-align:center;">' +
                        '<div style="color:#666;font-size:9px;text-transform:uppercase;">' + lbl + '</div>' +
                        '<div style="color:#ccc;font-weight:600;">' + vs + '</div></div>';
                }}
                html += '</div>';
            }}
            html += '</div>';

            html += '<table class="ov-table">';
            for (var i = 0; i < items.length; i++) {{
                var it = items[i];
                if (it.symbol === ticker) continue;
                html += '<tr>' +
                    '<td class="ov-tname">' + it.name + '</td>' +
                    '<td class="ov-tprice">' + it.price_str + '</td>' +
                    '<td class="ov-tchange">' + fmtChange(it.change_pct) + '</td>' +
                    '</tr>';
            }}
            html += '</table>';
            document.getElementById('overlay-content').innerHTML = html;
            document.getElementById('overlay-backdrop').style.display = 'block';
            document.getElementById('detail-overlay').style.display = 'block';
            hideTooltip();
        }}
        function hideOverlay() {{
            document.getElementById('overlay-backdrop').style.display = 'none';
            document.getElementById('detail-overlay').style.display = 'none';
        }}

        function renderPlot() {{
            console.log('renderPlot() called, Plotly version: ' + (typeof Plotly !== 'undefined' ? Plotly.version : 'NOT LOADED'));
            try {{
            var mapDiv = document.getElementById('map');
            Plotly.newPlot(mapDiv, [{{
                type: 'treemap',
                ids: d.ids,
                labels: d.labels,
                parents: d.parents,
                values: d.values,
                text: d.text,
                marker: {{
                    colors: d.colors,
                    colorscale: [
                        [0,    '#ff3333'],
                        [0.35, '#8b0000'],
                        [0.5,  '#262626'],
                        [0.65, '#006400'],
                        [1,    '#33ff33']
                    ],
                    cmid: 0,
                    cmin: -3,
                    cmax: 3,
                    line: {{ width: d.line_widths, color: d.line_colors }},
                    pad: {{ t: 25, l: 3, r: 3, b: 3 }}
                }},
                textinfo: 'label+text',
                textposition: 'middle center',
                hoverinfo: 'none',
                insidetextfont: {{ color: '#e8e8e8', size: d.font_sizes }},
                pathbar: {{ visible: false }},
                tiling: {{ pad: 1 }},
                branchvalues: 'total',
                maxdepth: 2,
                uniformtext: {{ minsize: 5, mode: 'show' }}
            }}], {{
                margin: {{ l: 0, r: 0, t: 0, b: 0 }},
                paper_bgcolor: '#0a0a0a',
                plot_bgcolor: '#0a0a0a',
                font: {{ color: '#e8e8e8', family: 'Arial, Helvetica, sans-serif' }},
                autosize: true
            }}, {{
                displayModeBar: false,
                responsive: true
            }});

            /* Click handler: overlay for items, drill-down for regions */
            mapDiv.on('plotly_treemapclick', function(data) {{
                if (data && data.points && data.points.length > 0) {{
                    var point = data.points[0];
                    var id = point.id || '';
                    if (id.indexOf('item:') === 0) {{
                        var ticker = id.replace('item:', '');
                        console.log('TREEMAP_CLICK:' + ticker);
                        showOverlay(ticker);
                        return false;  /* Prevent drill-down for items */
                    }}
                    /* Region clicks: allow drill-down (don't return false) */
                }}
            }});

            /* Hover handler: custom tooltip */
            mapDiv.on('plotly_hover', function(data) {{
                if (data && data.points && data.points.length > 0) {{
                    var point = data.points[0];
                    var id = point.id || '';
                    if (id.indexOf('item:') === 0) {{
                        var ticker = id.replace('item:', '');
                        showTooltip(data.event, ticker);
                    }}
                }}
            }});
            mapDiv.on('plotly_unhover', function() {{
                hideTooltip();
            }});

            window.addEventListener('resize', function() {{
                Plotly.Plots.resize('map');
            }});
            /* Close overlay on Escape */
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') hideOverlay();
            }});
            console.log('renderPlot() completed successfully');
            }} catch(err) {{
                console.log('renderPlot() ERROR: ' + err.message);
                document.getElementById('map').innerHTML =
                    '<div style="color:#ef4444;font-family:monospace;font-size:13px;display:flex;align-items:center;justify-content:center;height:100%;">' +
                    'Plotly error: ' + err.message + '</div>';
            }}
        }}

        /* ---- Live WebSocket price updates (called from Python via runJavaScript) ---- */
        var _flashTimers = {{}};
        d.intraday_ohlc = d.intraday_ohlc || {{}};
        d.ws_info = d.ws_info || {{}};

        window.updatePrices = function(payload) {{
            var updates = payload.updates || {{}};
            var changedTickers = [];
            var changedDirs = {{}};

            /* Store intraday OHLC and extra WS info */
            if (payload.intraday) {{
                for (var sym in payload.intraday) d.intraday_ohlc[sym] = payload.intraday[sym];
            }}
            if (payload.ws_info) {{
                for (var sym in payload.ws_info) d.ws_info[sym] = payload.ws_info[sym];
            }}

            for (var symbol in updates) {{
                var itemId = 'item:' + symbol;
                var idx = d.ids.indexOf(itemId);
                if (idx === -1) continue;

                var upd = updates[symbol];
                var oldPct = d.colors[idx];
                var newPct = upd.change_pct;

                var p = upd.price || 0;
                var c = upd.change || 0;
                var sign = newPct >= 0 ? '+' : '';
                var cStr = c !== 0 ? (c >= 0 ? '+' : '') + c.toFixed(2) : '';
                d.text[idx] = cStr + ' (' + sign + newPct.toFixed(2) + '%)';
                d.colors[idx] = newPct;

                /* Uppdatera label: rad 1 = namn, rad 2 = pris */
                var nm = d.names[idx] || '';
                if (nm && p > 0) {{
                    var pStr = p >= 100
                        ? p.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}})
                        : (p >= 1 ? p.toFixed(2) : p.toFixed(4));
                    d.labels[idx] = '<b>' + nm + '</b><br>' + pStr + '<br>';
                }}

                /* Update region data for tooltips/overlays */
                var info = d.item_regions[symbol];
                if (info) {{
                    var ritems = d.region_items[info.key] || [];
                    for (var j = 0; j < ritems.length; j++) {{
                        if (ritems[j].symbol === symbol) {{
                            ritems[j].change_pct = newPct;
                            var p = upd.price;
                            ritems[j].price_str = p >= 100
                                ? p.toLocaleString(undefined, {{minimumFractionDigits:2, maximumFractionDigits:2}})
                                : (p > 0 ? p.toFixed(4) : 'N/A');
                            break;
                        }}
                    }}
                }}

                var dt = symbol.replace(/^\\^/, '').replace(/=X$/, '').replace(/=F$/, '');
                changedTickers.push(dt);
                changedDirs[dt] = newPct > oldPct ? 'up' : (newPct < oldPct ? 'down' : 'same');
            }}

            if (changedTickers.length === 0) return;

            var mapDiv = document.getElementById('map');
            /* Preserve current drill-down level (e.g. 'region:EUROPE') */
            var curLevel = (mapDiv.data && mapDiv.data[0]) ? mapDiv.data[0].level : undefined;
            var trace = {{
                type: 'treemap',
                ids: d.ids, labels: d.labels, parents: d.parents,
                values: d.values, text: d.text,
                marker: {{
                    colors: d.colors,
                    colorscale: [[0,'#ff3333'],[0.35,'#8b0000'],[0.5,'#262626'],[0.65,'#006400'],[1,'#33ff33']],
                    cmid: 0, cmin: -3, cmax: 3,
                    line: {{ width: d.line_widths, color: d.line_colors }},
                    pad: {{ t: 25, l: 3, r: 3, b: 3 }}
                }},
                textinfo: 'label+text',
                textposition: 'middle center',
                hoverinfo: 'none',
                insidetextfont: {{ color: '#e8e8e8', size: d.font_sizes }},
                pathbar: {{ visible: false }},
                tiling: {{ pad: 1 }},
                branchvalues: 'total',
                maxdepth: 2,
                uniformtext: {{ minsize: 5, mode: 'show' }}
            }};
            if (curLevel) trace.level = curLevel;
            Plotly.react(mapDiv, [trace], {{
                margin: {{ l: 0, r: 0, t: 0, b: 0 }},
                paper_bgcolor: '#0a0a0a',
                plot_bgcolor: '#0a0a0a',
                font: {{ color: '#e8e8e8', family: 'Arial, Helvetica, sans-serif' }},
                autosize: true
            }});

            /* Flash the %-text on changed tiles */
            setTimeout(function() {{
                var allText = document.querySelectorAll('#map text');
                allText.forEach(function(textEl) {{
                    var tspans = textEl.querySelectorAll('tspan');
                    if (tspans.length < 1) return;
                    var label = (tspans[0].textContent || '').trim();
                    var dir = changedDirs[label];
                    if (!dir || dir === 'same') return;

                    var target = tspans.length > 1 ? tspans[tspans.length - 1] : tspans[0];
                    var flashColor = dir === 'up' ? '#00ff00' : '#ff4444';

                    if (_flashTimers[label]) clearTimeout(_flashTimers[label]);
                    target.setAttribute('fill', flashColor);
                    _flashTimers[label] = setTimeout(function() {{
                        target.setAttribute('fill', '#e8e8e8');
                    }}, 2000);
                }});
            }}, 100);
        }};

        // Wait for Plotly to be available before rendering
        if (typeof Plotly !== 'undefined') {{
            console.log('Plotly already available, rendering immediately');
            renderPlot();
        }} else {{
            console.log('Plotly not yet loaded, waiting for script...');
            var plotlyScript = document.querySelector('script[src*="plotly"]');
            if (plotlyScript) {{
                plotlyScript.addEventListener('load', function() {{
                    console.log('Plotly script loaded via event');
                    renderPlot();
                }});
                plotlyScript.addEventListener('error', function(e) {{
                    console.log('Plotly script FAILED to load: ' + e.type);
                    document.getElementById('map').innerHTML =
                        '<div style="color:#ef4444;font-family:monospace;font-size:13px;display:flex;align-items:center;justify-content:center;height:100%;">' +
                        'Failed to load Plotly library. Check internet connection.</div>';
                }});
            }}
        }}
    </script>
</body>
</html>
'''
        # Write to temp file and load via file:// URL so external scripts can load
        try:
            treemap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trading')
            os.makedirs(treemap_dir, exist_ok=True)
            tmp_path = os.path.join(treemap_dir, 'treemap.html')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(html)
            file_url = QUrl.fromLocalFile(tmp_path)
            print(f'[MarketWatch] Treemap HTML written to: {tmp_path} ({len(html)} bytes)')
            self.map_widget.load(file_url)
        except Exception as e:
            print(f'[MarketWatch] Treemap file write error: {e}')
            self.map_widget.setHtml(html)

    # ========================================================================
    # ACTIONS & SLOTS
    # ========================================================================
    
    def update_time(self):
        """Update status bar time."""
        self.status_time.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def load_initial_data(self):
        """Load initial data on startup using background thread.
        
        OPTIMERING: All disk I/O körs i bakgrundstråd för att undvika GUI-frys.
        """
        # Track startup worker
        self._startup_worker = None
        self._startup_thread = None
        
        # Create and start startup worker
        self._startup_thread = QThread()
        self._startup_worker = StartupWorker(PORTFOLIO_FILE, ENGINE_CACHE_FILE)
        self._startup_worker.moveToThread(self._startup_thread)
        
        # Connect signals
        self._startup_thread.started.connect(self._startup_worker.run)
        self._startup_worker.finished.connect(self._startup_thread.quit)
        self._startup_worker.finished.connect(self._startup_worker.deleteLater)
        self._startup_thread.finished.connect(self._startup_thread.deleteLater)
        self._startup_thread.finished.connect(self._on_startup_finished)
        
        # Connect data signals to handlers
        self._startup_worker.portfolio_loaded.connect(self._on_startup_portfolio_loaded)
        self._startup_worker.engine_loaded.connect(self._on_startup_engine_loaded)
        self._startup_worker.status_message.connect(self.statusBar().showMessage)
        
        # Start loading
        self.statusBar().showMessage("Loading cached data...")
        self._startup_thread.start()
    
    def _on_startup_portfolio_loaded(self, positions: list):
        """Handle portfolio loaded from startup worker."""
        self.portfolio = positions
        self.trade_history = load_trade_history(PORTFOLIO_FILE)
        self._migrate_trade_history()
        if self._tabs_loaded.get(4, False):
            self.update_portfolio_display()
        self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
        if os.path.exists(PORTFOLIO_FILE):
            self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
    
    def _migrate_trade_history(self):
        """Migrate trade history: recalculate capital and result from MF data."""
        changed = False
        for trade in self.trade_history:
            entry_y = trade.get('mf_entry_price_y')
            entry_x = trade.get('mf_entry_price_x')
            qty_y = trade.get('mf_qty_y', 0)
            qty_x = trade.get('mf_qty_x', 0)

            # Recalculate capital from actual invested amount
            actual_capital = 0
            if entry_y and qty_y:
                actual_capital += entry_y * qty_y
            if entry_x and qty_x:
                actual_capital += entry_x * qty_x

            if actual_capital > 0 and trade.get('capital', 0) != actual_capital:
                trade['capital'] = actual_capital
                # Recalculate P/L percentage with correct capital
                pnl_sek = trade.get('realized_pnl_sek', 0)
                trade['realized_pnl_pct'] = round(pnl_sek / actual_capital * 100, 2)
                changed = True

            # Fix result based on actual P/L (not Z-score direction)
            close_y = trade.get('mf_close_price_y')
            close_x = trade.get('mf_close_price_x')
            has_mf = (entry_y and close_y and qty_y) or (entry_x and close_x and qty_x)
            if has_mf:
                pnl_sek = trade.get('realized_pnl_sek', 0)
                correct_result = "PROFIT" if pnl_sek >= 0 else "LOSS"
                if trade.get('result') != correct_result:
                    trade['result'] = correct_result
                    changed = True

        if changed:
            save_portfolio(self.portfolio, trade_history=self.trade_history)
            print("[Migration] Trade history capital and result fields updated")

    def _on_startup_engine_loaded(self, cache_data: dict):
        """Handle engine cache loaded from startup worker."""
        self._apply_engine_cache(cache_data)
        if os.path.exists(ENGINE_CACHE_FILE):
            self._engine_cache_mtime = os.path.getmtime(ENGINE_CACHE_FILE)
    
    def _on_startup_finished(self):
        """Handle startup worker finished - start market data fetching.
        
        NOTE: Uses one-shot flag because QThread.finished can fire twice
        (once from quit(), once from deleteLater destruction).
        """
        if getattr(self, '_startup_finished_done', False):
            print("[Startup] _on_startup_finished already ran, skipping duplicate")
            return
        self._startup_finished_done = True
        
        # Markera att startup är klar
        self._startup_complete = True

        # WS-first: start WebSocket for all market tickers (replaces yf.download for treemap)
        QTimer.singleShot(500, self._start_ws_market_feed)
    
    def load_saved_portfolio(self):
        """Load saved portfolio positions from file."""
        try:
            saved_positions = load_portfolio(PORTFOLIO_FILE)
            if saved_positions:
                self.portfolio = saved_positions
                # OPTIMERING: Uppdatera endast om Portfolio-tabben är laddad
                if self._tabs_loaded.get(4, False):
                    self.update_portfolio_display()
                self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
                self.statusBar().showMessage(f"Loaded {len(self.portfolio)} saved position(s)")
            
            # Spara filens mtime för sync-tracking
            if os.path.exists(PORTFOLIO_FILE):
                self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
        except Exception as e:
            print(f"Error loading saved portfolio: {e}")
    
    def load_saved_engine_cache(self):
        """Load saved engine cache from file (from another computer's scan)."""
        try:
            cache_data = load_engine_cache(ENGINE_CACHE_FILE)
            if cache_data and cache_data.get('price_data') is not None:
                self._apply_engine_cache(cache_data)
                
                # Spara mtime för sync-tracking
                if os.path.exists(ENGINE_CACHE_FILE):
                    self._engine_cache_mtime = os.path.getmtime(ENGINE_CACHE_FILE)
                    
                scan_time = cache_data.get('scan_timestamp', 'unknown')
                scanned_by = cache_data.get('scanned_by', 'unknown')
                self.statusBar().showMessage(f"Loaded engine cache from {scanned_by} at {scan_time[:16]}")
        except Exception as e:
            print(f"Error loading saved engine cache: {e}")
            traceback.print_exc()
    
    def _apply_engine_cache(self, cache_data: dict):
        """Apply loaded engine cache to restore full functionality."""
        price_data = cache_data.get('price_data')
        raw_price_data = cache_data.get('raw_price_data')
        viable_pairs = cache_data.get('viable_pairs')
        pairs_stats = cache_data.get('pairs_stats')
        tickers = cache_data.get('tickers', [])
        config = cache_data.get('config', {})
        ou_models = cache_data.get('ou_models', {})
        
        if price_data is None or len(price_data.columns) == 0:
            print("[Engine Cache] No price data in cache")
            return
        
        # Skapa en riktig PairsTradingEngine med cached data
        try:
            # Använd standardconfig om ingen finns
            if not config:
                config = {
                    'min_half_life': 1,
                    'max_half_life': 60,
                    'max_adf_pvalue': 0.05,
                    'min_hurst': 0.0,
                    'max_hurst': 0.5,
                    'min_correlation': 0.7,
                }
            
            # Skapa engine med config
            engine = PairsTradingEngine(config=config)
            
            # Sätt cached data direkt
            engine.price_data = price_data
            engine.raw_price_data = raw_price_data if raw_price_data is not None else price_data.copy()
            engine.viable_pairs = viable_pairs
            engine.pairs_stats = pairs_stats if pairs_stats is not None else []
            engine.ou_models = ou_models
            engine._window_details = cache_data.get('window_details', {})

            # Rebuild _pair_index from pairs_stats for O(1) lookups
            # CRITICAL: Without this, get_pair_ou_params falls back to fresh OLS
            # which gives different β/α than the cached screening, causing
            # spread/z-score to be inconsistent with the price comparison chart.
            engine._pair_index = {}
            if pairs_stats is not None and hasattr(pairs_stats, 'itertuples'):
                for row in pairs_stats.itertuples():
                    if hasattr(row, 'pair'):
                        engine._pair_index[row.pair] = row

            self.engine = engine
            
            # Uppdatera metrics
            n_tickers = len(price_data.columns)
            n_pairs = len(pairs_stats) if pairs_stats is not None else 0
            n_viable = len(viable_pairs) if viable_pairs is not None and len(viable_pairs) > 0 else 0
            
            self._update_metric_value(self.tickers_metric, str(n_tickers))
            self._update_metric_value(self.pairs_metric, str(n_pairs))
            self._update_metric_value(self.viable_metric, str(n_viable))
            
            if hasattr(self, 'viable_count_label'):
                self.viable_count_label.setText(f"({n_viable})")
            
            # OPTIMERING: Uppdatera endast laddade tabbar (lazy loading)
            if self._tabs_loaded.get(1, False):  # Arbitrage Scanner
                self.update_viable_table()
                self.update_all_pairs_table()
            if self._tabs_loaded.get(2, False):  # OU Analytics
                self.update_ou_pair_list()
            if self._tabs_loaded.get(3, False):  # Pair Signals
                self.update_signals_list()
            
            print(f"[Engine Cache] Applied {n_tickers} tickers, {n_viable} viable pairs - full functionality restored")
            
        except Exception as e:
            print(f"[Engine Cache] Error applying cache: {e}")
            traceback.print_exc()
    
    # Legacy compatibility
    def load_saved_scan_results(self):
        """Legacy method - redirects to load_saved_engine_cache."""
        self.load_saved_engine_cache()
    
    def sync_from_drive(self):
        """Check if portfolio or engine cache changed (e.g. from another computer) and reload.
        
        Fix #2: Runs in background thread to prevent UI blocking.
        """
        # Don't start new sync if one is already running - använd säker flagga
        if self._sync_running:
            return
        
        # Sätt flagga
        self._sync_running = True
        
        # Create new worker and thread
        self._sync_thread = QThread()
        self._sync_worker = SyncWorker(
            PORTFOLIO_FILE, 
            ENGINE_CACHE_FILE,
            self._portfolio_file_mtime,
            self._engine_cache_mtime
        )
        self._sync_worker.moveToThread(self._sync_thread)
        
        # Connect signals
        self._sync_thread.started.connect(self._sync_worker.run)
        self._sync_worker.finished.connect(self._sync_thread.quit)
        self._sync_thread.finished.connect(self._on_sync_thread_finished)
        self._sync_worker.portfolio_changed.connect(self._on_portfolio_synced)
        self._sync_worker.engine_changed.connect(self._on_engine_synced)
        self._sync_worker.status_message.connect(lambda msg: self.statusBar().showMessage(msg))
        
        # Start the thread
        self._sync_thread.start()
    
    def _on_sync_thread_finished(self):
        """Handle sync thread completion - clear references and flag."""
        self._sync_running = False
        
        # Använd deleteLater() för säker cleanup
        if self._sync_worker is not None:
            self._sync_worker.deleteLater()
        if self._sync_thread is not None:
            self._sync_thread.deleteLater()
        
        self._sync_worker = None
        self._sync_thread = None
    
    def _on_portfolio_synced(self, new_positions: list):
        """Handle portfolio sync from background thread.
        
        Protects trade_history: positions closed locally won't be re-added
        if another computer syncs an older version of the file.
        """
        # Merge trade_history: keep local entries + any new ones from file
        file_history = load_trade_history(PORTFOLIO_FILE)
        merged_history = list(self.trade_history)  # Start with local
        local_keys = {(t['pair'], t.get('entry_date', '')) for t in merged_history}
        for fh in file_history:
            key = (fh['pair'], fh.get('entry_date', ''))
            if key not in local_keys:
                merged_history.append(fh)
                local_keys.add(key)
        self.trade_history = merged_history
        
        # Filter out positions that are in trade_history (already closed)
        closed_keys = {(t['pair'], t.get('entry_date', '')) for t in self.trade_history}
        filtered = [p for p in new_positions 
                    if (p['pair'], p.get('entry_date', '')) not in closed_keys]
        self.portfolio = filtered
        
        if os.path.exists(PORTFOLIO_FILE):
            self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
        
        # Re-save with correct state (so other computers pick up the close)
        self._save_and_sync_portfolio()
        
        if self._tabs_loaded.get(4, False):
            self.update_portfolio_display()
        self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
    
    def _on_engine_synced(self, cache_data: dict):
        """Handle engine cache sync from background thread."""
        self._apply_engine_cache(cache_data)
        if os.path.exists(ENGINE_CACHE_FILE):
            self._engine_cache_mtime = os.path.getmtime(ENGINE_CACHE_FILE)
    
    def _sync_portfolio(self):
        """Sync portfolio from file."""
        try:
            if not os.path.exists(PORTFOLIO_FILE):
                return
            
            current_mtime = os.path.getmtime(PORTFOLIO_FILE)
            
            # Om filen har ändrats sedan vi senast laddade/sparade
            if current_mtime > self._portfolio_file_mtime + 1:  # +1 sek marginal
                print(f"[Portfolio Sync] File changed externally, reloading...")
                
                # Ladda nya positioner
                new_positions = load_portfolio(PORTFOLIO_FILE)
                
                if new_positions is not None:
                    # Merge trade_history: keep local + file entries
                    file_history = load_trade_history(PORTFOLIO_FILE)
                    merged_history = list(self.trade_history)
                    local_keys = {(t['pair'], t.get('entry_date', '')) for t in merged_history}
                    for fh in file_history:
                        key = (fh['pair'], fh.get('entry_date', ''))
                        if key not in local_keys:
                            merged_history.append(fh)
                            local_keys.add(key)
                    self.trade_history = merged_history
                    
                    # Filter out positions that are CLOSED in trade_history
                    closed_keys = {(t['pair'], t.get('entry_date', ''))
                                   for t in self.trade_history
                                   if t.get('status') not in ('OPEN', None)
                                   and t.get('close_date')}
                    filtered = [p for p in new_positions
                                if (p['pair'], p.get('entry_date', '')) not in closed_keys]
                    self.portfolio = filtered
                    self._portfolio_file_mtime = current_mtime
                    
                    # Re-save so other computers see the close
                    self._save_and_sync_portfolio()
                    self.update_portfolio_display()
                    self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
                    self.statusBar().showMessage(f"Portfolio synced: {len(self.portfolio)} position(s) from another device")
                    
        except Exception as e:
            print(f"[Portfolio Sync] Error: {e}")
    
    def _sync_engine_cache(self):
        """Sync engine cache from file."""
        try:
            if not os.path.exists(ENGINE_CACHE_FILE):
                return
            
            current_mtime = os.path.getmtime(ENGINE_CACHE_FILE)
            
            # Om filen har ändrats sedan vi senast laddade
            if current_mtime > self._engine_cache_mtime + 1:  # +1 sek marginal
                print(f"[Engine Cache Sync] File changed externally, reloading...")
                
                cache_data = load_engine_cache(ENGINE_CACHE_FILE)
                
                if cache_data and cache_data.get('price_data') is not None:
                    self._apply_engine_cache(cache_data)
                    self._engine_cache_mtime = current_mtime
                    
                    scan_time = cache_data.get('scan_timestamp', 'unknown')[:16]
                    scanned_by = cache_data.get('scanned_by', 'unknown')
                    n_pairs = len(cache_data.get('viable_pairs', [])) if cache_data.get('viable_pairs') is not None else 0
                    self.statusBar().showMessage(f"Engine synced: {n_pairs} pairs from {scanned_by} at {scan_time}")
                    
        except Exception as e:
            print(f"[Engine Cache Sync] Error: {e}")
    
    def sync_portfolio_from_file(self):
        """Legacy method - redirects to sync_from_drive."""
        self.sync_from_drive()
    
    def _save_and_sync_portfolio(self):
        """Save portfolio + trade history, update mtime."""
        if save_portfolio(self.portfolio, trade_history=self.trade_history):
            if os.path.exists(PORTFOLIO_FILE):
                self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
    
    def refresh_market_watch(self):
        """Restart WebSocket market feed (manual refresh). No yf.download."""
        if not self._startup_complete:
            return
        print("[MarketWatch] Manual refresh — restarting WebSocket feed")
        self._ws_treemap_rendered = False
        self._start_ws_market_feed(trigger_volatility=False)
    
    def _start_volatility_refresh_safe(self):
        """Safely start volatility refresh after a delay."""
        print("[Volatility] Starting safe volatility refresh...")
        try:
            self.refresh_market_data()
        except Exception as e:
            print(f"[Volatility] Error starting refresh: {e}")
            traceback.print_exc()

    def _load_or_fetch_volatility(self):
        """Ladda percentilcache från disk om <24h gammal, annars yf.download."""
        cache = load_volatility_cache()
        if cache is not None:
            try:
                # Ladda alltid in cachad data i minne (fallback för SKEW/MOVE)
                for ticker, values in cache.get('hist', {}).items():
                    self._vol_hist_cache[ticker] = np.array(values)
                self._vol_median_cache = cache.get('median', {})
                self._vol_mode_cache = cache.get('mode', {})
                self._vol_sparkline_cache = cache.get('sparkline', {})

                saved_str = cache.get('saved_at', '')
                saved_dt = datetime.fromisoformat(saved_str)
                age_hours = (datetime.now() - saved_dt).total_seconds() / 3600
                if age_hours < 24:
                    # Cache är färsk — visa cachad data och skippa yf.download
                    self._apply_cached_volatility_cards()
                    print(f"[VolCache] Using disk cache ({age_hours:.1f}h old) — skipping yf.download")
                    self.statusBar().showMessage("Volatility percentiles loaded from cache")
                    return
                else:
                    # Visa cachad data direkt, sedan uppdatera med yf.download
                    self._apply_cached_volatility_cards()
                    print(f"[VolCache] Cache is {age_hours:.1f}h old — showing cached, then refreshing via yf.download")
            except Exception as e:
                print(f"[VolCache] Could not parse cache: {e}")

        # Hämta färsk data via yf.download (cachad data visas som fallback)
        self._start_volatility_refresh_safe()

    def _apply_cached_volatility_cards(self):
        """Uppdatera volatility cards med cachad percentildata (utan yf.download)."""
        vol_descs = {
            '^VIX': [
                (10, "Extreme complacency in markets"), (25, "Low fear - calm markets"),
                (50, "Relatively calm conditions"), (75, "Slightly elevated uncertainty"),
                (90, "Elevated fear & uncertainty"), (95, "Significant market stress"),
                (100, "Extreme fear - crisis levels"),
            ],
            '^VVIX': [
                (10, "Very stable VIX expectations"), (25, "Stable VIX outlook"),
                (50, "Below median uncertainty about VIX"), (75, "Somewhat uncertain VIX outlook"),
                (90, "Potential for volatility spikes"), (95, "VIX itself highly volatile"),
                (100, "Crisis-level VIX uncertainty"),
            ],
            '^SKEW': [
                (10, "Very low tail risk pricing"), (25, "Limited crash protection demand"),
                (50, "Below median tail risk perception"), (75, "Modest downside protection demand"),
                (90, "Increased put demand"), (95, "Significant crash hedging"),
                (100, "Major crash protection demand"),
            ],
            '^MOVE': [
                (10, "Low rate volatility"), (25, "Stable rate environment"),
                (50, "Relatively stable rates"), (75, "Some rate uncertainty"),
                (90, "Significant rate uncertainty"), (95, "Major rate movements expected"),
                (100, "Crisis-level rate uncertainty"),
            ],
        }
        card_map = {'^VIX': 'vix_card', '^VVIX': 'vvix_card',
                    '^SKEW': 'skew_card', '^MOVE': 'move_card'}

        for ticker, attr in card_map.items():
            card = getattr(self, attr, None)
            if card is None or ticker not in self._vol_hist_cache:
                continue
            hist = self._vol_hist_cache[ticker]
            median = self._vol_median_cache.get(ticker, 0)
            mode = self._vol_mode_cache.get(ticker, 0)
            sparkline = self._vol_sparkline_cache.get(ticker)

            # Använd senaste värdet från sparkline som "current value"
            if sparkline and len(sparkline) > 0:
                val = sparkline[-1]
                prev = sparkline[-2] if len(sparkline) > 1 else val
                chg = ((val / prev) - 1) * 100 if prev != 0 else 0
                pct = np.searchsorted(hist, val) / len(hist) * 100

                desc = ""
                for threshold, text in vol_descs.get(ticker, []):
                    if pct < threshold:
                        desc = text
                        break

                card.update_data(val, chg, pct, median, mode, desc, history=sparkline)

    def _apply_vol_card_fallback(self, ticker: str, card):
        """Fallback: visa senaste cachade data när yfinance inte returnerar ticker.

        Används för SKEW/MOVE som ofta uppdateras med fördröjning (efter midnatt).
        Visar senaste kända pris med percentil och sparkline från cachen.
        """
        vol_descs = {
            '^VIX': [
                (10, "Extreme complacency in markets"), (25, "Low fear - calm markets"),
                (50, "Relatively calm conditions"), (75, "Slightly elevated uncertainty"),
                (90, "Elevated fear & uncertainty"), (95, "Significant market stress"),
                (100, "Extreme fear - crisis levels"),
            ],
            '^VVIX': [
                (10, "Very stable VIX expectations"), (25, "Stable VIX outlook"),
                (50, "Below median uncertainty about VIX"), (75, "Somewhat uncertain VIX outlook"),
                (90, "Potential for volatility spikes"), (95, "VIX itself highly volatile"),
                (100, "Crisis-level VIX uncertainty"),
            ],
            '^SKEW': [
                (10, "Very low tail risk pricing"), (25, "Limited crash protection demand"),
                (50, "Below median tail risk perception"), (75, "Modest downside protection demand"),
                (90, "Increased put demand"), (95, "Significant crash hedging"),
                (100, "Major crash protection demand"),
            ],
            '^MOVE': [
                (10, "Low rate volatility"), (25, "Stable rate environment"),
                (50, "Relatively stable rates"), (75, "Some rate uncertainty"),
                (90, "Significant rate uncertainty"), (95, "Major rate movements expected"),
                (100, "Crisis-level rate uncertainty"),
            ],
        }
        sparkline = self._vol_sparkline_cache.get(ticker)
        hist = self._vol_hist_cache.get(ticker)
        if sparkline and len(sparkline) > 0 and hist is not None and len(hist) > 0:
            val = sparkline[-1]
            prev = sparkline[-2] if len(sparkline) > 1 else val
            chg = ((val / prev) - 1) * 100 if prev != 0 else 0
            pct = np.searchsorted(hist, val) / len(hist) * 100
            median = self._vol_median_cache.get(ticker, 0)
            mode = self._vol_mode_cache.get(ticker, 0)

            desc = ""
            for threshold, text in vol_descs.get(ticker, []):
                if pct < threshold:
                    desc = text
                    break

            card.update_data(val, chg, pct, median, mode, desc, history=sparkline)
            print(f"[Volatility] {ticker}: yfinance data unavailable, using cached fallback (last: {val:.2f})")
        else:
            card.value_label.setText("N/A")
            card.desc_label.setText(f"No {ticker.replace('^', '')} data available")
            print(f"[Volatility] {ticker}: no data from yfinance and no cache available")

    def _start_ws_market_feed(self, trigger_volatility=True):
        """Start WebSocket for all market instruments (replaces yf.download for treemap)."""
        tickers = list(self.MARKET_INSTRUMENTS.keys())
        # Add VIX/VVIX for live volatility card updates (SKEW/MOVE via yf.download only)
        for extra in ['^VIX', '^VVIX']:
            if extra not in tickers:
                tickers.append(extra)
        # Clean up old refs
        self._stop_market_websocket()

        self._ws_thread = QThread()
        self._ws_worker = MarketWatchWebSocket(tickers)
        self._ws_worker.moveToThread(self._ws_thread)
        self._ws_thread.started.connect(self._ws_worker.run)
        self._ws_worker.price_update.connect(self._on_ws_price_update)
        self._ws_worker.connected.connect(self._on_ws_connected)
        self._ws_worker.status_message.connect(self.statusBar().showMessage)
        self._ws_worker.error.connect(lambda e: print(f"[WS] {e}"))
        self._ws_thread.start()
        print(f"[WS] Starting WebSocket feed for {len(tickers)} tickers (WS-first mode)")

        # Volatilitetspercentiler: ladda från disk-cache om <24h gammal, annars yf.download
        if trigger_volatility:
            QTimer.singleShot(2000, self._load_or_fetch_volatility)

        # Fetch intraday OHLC (1d/5m) to seed overlay candlestick charts
        QTimer.singleShot(5000, self._fetch_intraday_ohlc)

    def _on_ws_connected(self):
        """WS connected — schedule first treemap render after initial snapshots arrive."""
        if not self._ws_treemap_rendered:
            # Wait 4s for WS snapshots to accumulate, then render all instruments
            QTimer.singleShot(4000, self._render_initial_treemap_from_ws)

    def _render_initial_treemap_from_ws(self):
        """Build treemap from ALL instruments, using WS data where available.

        Renders all 71+ instruments immediately — instruments without WS
        snapshots yet show as gray (0% change) and update via the 5s timer.
        """
        if self._ws_treemap_rendered:
            return

        self._ws_treemap_rendered = True
        ws_count = 0
        items = []
        # Iterate ALL instruments — use WS cache if available, placeholder if not
        for symbol, (name, market) in self.MARKET_INSTRUMENTS.items():
            cached = self._market_data_cache.get(symbol)
            if cached:
                ws_count += 1
                items.append({
                    'market': market,
                    'symbol': symbol,
                    'name': name,
                    'price': cached.get('price', 0),
                    'change': cached.get('change', 0),
                    'change_pct': cached.get('change_pct', 0),
                })
            else:
                # Placeholder — will be updated by WS + 5s render timer
                items.append({
                    'market': market,
                    'symbol': symbol,
                    'name': name,
                    'price': 0,
                    'change': 0,
                    'change_pct': 0,
                })

        if items and hasattr(self, 'map_widget'):
            self.update_treemap_heatmap(items)
            print(f"[WS] Initial treemap rendered: {len(items)} instruments ({ws_count} with WS data)")
            self.statusBar().showMessage(f"Market data: {len(items)} instruments ({ws_count} live)")
        else:
            print(f"[WS] No items for treemap")

    def _fetch_intraday_ohlc(self):
        """Fetch today's intraday 5-min OHLC for all instruments (background)."""
        tickers = list(self.MARKET_INSTRUMENTS.keys())
        self._intraday_thread = QThread()
        self._intraday_worker = IntradayOHLCWorker(tickers, self.MARKET_INSTRUMENTS)
        self._intraday_worker.moveToThread(self._intraday_thread)
        self._intraday_thread.started.connect(self._intraday_worker.run)
        self._intraday_worker.finished.connect(self._intraday_thread.quit)
        self._intraday_worker.result.connect(self._on_intraday_ohlc_received)
        self._intraday_worker.error.connect(self._on_intraday_ohlc_error)
        self._intraday_thread.start()

    def _on_intraday_ohlc_received(self, data: dict):
        """Handle intraday OHLC data — seed charts, extra info, och uppdatera tiles."""
        count = 0
        updates_js = {}  # pris/change-uppdateringar för tiles utan WS-data

        for symbol, payload in data.items():
            ohlc = payload.get('ohlc', [])
            info = payload.get('info', {})
            if ohlc:
                self._intraday_ohlc_seed[symbol] = ohlc
                count += 1

            # Fyll i ws_extra_info med nedladdad data
            if info and symbol not in self._ws_extra_info:
                self._ws_extra_info[symbol] = {
                    'day_high': info.get('day_high', 0),
                    'day_low': info.get('day_low', 0),
                    'day_volume': info.get('day_volume', 0),
                    'open_price': info.get('open_price', 0),
                    'previous_close': info.get('previous_close', 0),
                    'short_name': '',
                }
            elif info:
                ws = self._ws_extra_info[symbol]
                if not ws.get('day_high'):
                    ws['day_high'] = info.get('day_high', 0)
                if not ws.get('day_low'):
                    ws['day_low'] = info.get('day_low', 0)
                if not ws.get('day_volume'):
                    ws['day_volume'] = info.get('day_volume', 0)
                if not ws.get('open_price'):
                    ws['open_price'] = info.get('open_price', 0)
                if not ws.get('previous_close'):
                    ws['previous_close'] = info.get('previous_close', 0)

            # Uppdatera treemap-tiles för instrument utan WS-data
            if info and symbol in self.MARKET_INSTRUMENTS:
                close_price = info.get('close_price', 0)
                change_pct = info.get('change_pct', 0)
                cached = self._market_data_cache.get(symbol, {})
                # Bara uppdatera om WS inte redan gett oss riktiga värden
                if close_price > 0 and not cached.get('price'):
                    name, market = self.MARKET_INSTRUMENTS[symbol]
                    prev_close = info.get('previous_close', 0)
                    change = close_price - prev_close if prev_close > 0 else 0
                    self._market_data_cache[symbol] = {
                        'symbol': symbol, 'name': name, 'market': market,
                        'price': close_price,
                        'change': round(change, 4),
                        'change_pct': change_pct,
                    }
                    updates_js[symbol] = {
                        'price': close_price,
                        'change': round(change, 4),
                        'change_pct': change_pct,
                    }

        print(f"[Intraday] Seeded OHLC charts for {count} instruments, "
              f"tile updates for {len(updates_js)} (closed/no-WS)")

        # Skicka allt till JS: OHLC-charts + ws_info + tile-uppdateringar
        if (count > 0 or updates_js) and hasattr(self, 'map_widget'):
            intraday_js = {}
            ws_info_js = {}
            for symbol in data:
                ohlc = self._intraday_ohlc_seed.get(symbol)
                if ohlc:
                    intraday_js[symbol] = ohlc
                if symbol in self._ws_extra_info:
                    ws_info_js[symbol] = self._ws_extra_info[symbol]
            payload = json.dumps({
                'updates': updates_js,
                'intraday': intraday_js,
                'ws_info': ws_info_js,
            })
            js = f'if(window.updatePrices)window.updatePrices({payload});'
            self.map_widget.page().runJavaScript(js)

        # Retry om vi fick data för mindre än 50% av tickrarna
        total_tickers = len(self.MARKET_INSTRUMENTS)
        if count < total_tickers * 0.5 and self._intraday_retry_count < self._intraday_max_retries:
            self._intraday_retry_count += 1
            delay = self._intraday_retry_count * 15000  # 15s, 30s, 45s
            print(f"[Intraday] Only {count}/{total_tickers} tickers — retry {self._intraday_retry_count}/{self._intraday_max_retries} in {delay//1000}s")
            QTimer.singleShot(delay, self._fetch_intraday_ohlc)

    def _on_intraday_ohlc_error(self, error_msg: str):
        """Handle intraday OHLC error — retry med backoff."""
        print(f"[Intraday] Error: {error_msg}")
        if self._intraday_retry_count < self._intraday_max_retries:
            self._intraday_retry_count += 1
            delay = self._intraday_retry_count * 15000
            print(f"[Intraday] Retry {self._intraday_retry_count}/{self._intraday_max_retries} in {delay//1000}s")
            QTimer.singleShot(delay, self._fetch_intraday_ohlc)

    def _on_treemap_click(self, ticker: str):
        """Handle treemap tile click — now handled by JS overlay in treemap."""
        print(f"[Treemap] Click on {ticker} (handled by JS overlay)")

    def _on_ws_price_update(self, update: dict):
        """Handle a single live price update from WebSocket."""
        try:
            self._on_ws_price_update_inner(update)
        except Exception as e:
            print(f"[WS] Price update error for {update.get('symbol', '?')}: {e}")

    def _on_ws_price_update_inner(self, update: dict):
        """Inner handler — all WS price update logic."""
        symbol = update['symbol']

        # Accumulate tick history for intraday charts
        ts = update.get('timestamp', time.time())
        if symbol not in self._ws_tick_history:
            self._ws_tick_history[symbol] = []
        self._ws_tick_history[symbol].append((ts, update['price']))

        # Store extra WS info for overlay
        self._ws_extra_info[symbol] = {
            'day_high': update.get('day_high', 0),
            'day_low': update.get('day_low', 0),
            'day_volume': update.get('day_volume', 0),
            'previous_close': update.get('previous_close', 0),
            'open_price': update.get('open_price', 0),
            'short_name': update.get('short_name', ''),
        }

        # VIX/VVIX live updates med percentilberäkning från cachad historik
        # (SKEW/MOVE uppdateras enbart via yf.download — inte WebSocket)
        if symbol in ('^VIX', '^VVIX') and symbol in self._vol_hist_cache:
            price = update['price']
            hist = self._vol_hist_cache[symbol]
            pct = np.searchsorted(hist, price) / len(hist) * 100
            median = self._vol_median_cache.get(symbol, 0)
            mode = self._vol_mode_cache.get(symbol, 0)
            change_pct = update.get('change_pct', 0)

            if symbol == '^VIX':
                if pct < 10:
                    desc = "Extreme complacency in markets"
                elif pct < 25:
                    desc = "Low fear - calm markets"
                elif pct < 50:
                    desc = "Relatively calm conditions"
                elif pct < 75:
                    desc = "Slightly elevated uncertainty"
                elif pct < 90:
                    desc = "Elevated fear & uncertainty"
                elif pct < 95:
                    desc = "Significant market stress"
                else:
                    desc = "Extreme fear - crisis levels"
                self.vix_card.update_data(price, change_pct, pct, median, mode, desc)
            elif symbol == '^VVIX':
                if pct < 10:
                    desc = "Very stable VIX expectations"
                elif pct < 25:
                    desc = "Stable VIX outlook"
                elif pct < 50:
                    desc = "Below median uncertainty about VIX"
                elif pct < 75:
                    desc = "Somewhat uncertain VIX outlook"
                elif pct < 90:
                    desc = "Potential for volatility spikes"
                elif pct < 95:
                    desc = "VIX itself highly volatile"
                else:
                    desc = "Crisis-level VIX uncertainty"
                self.vvix_card.update_data(price, change_pct, pct, median, mode, desc)
        elif symbol == '^VIX' and hasattr(self, 'vix_card'):
            self.vix_card.value_label.setText(f"{update['price']:.2f}")
        elif symbol == '^VVIX' and hasattr(self, 'vvix_card'):
            self.vvix_card.value_label.setText(f"{update['price']:.2f}")

        # Build/update market data cache entry for treemap
        if symbol in self.MARKET_INSTRUMENTS:
            name, market = self.MARKET_INSTRUMENTS[symbol]
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {
                    'symbol': symbol, 'name': name, 'market': market,
                }
            cached = self._market_data_cache[symbol]
            cached['price'] = update['price']
            cached['change'] = update['change']
            cached['change_pct'] = update['change_pct']
            self._ws_cache_dirty = True
            self._ws_changed_symbols.add(symbol)


    def _render_ws_updates(self):
        """Batch update treemap via JS injection if WebSocket updated prices (every 5s)."""
        if not self._ws_cache_dirty:
            return
        self._ws_cache_dirty = False
        changed = self._ws_changed_symbols.copy()
        self._ws_changed_symbols.clear()

        if not changed or not hasattr(self, 'map_widget'):
            return

        updates = {}
        intraday = {}
        ws_info = {}
        for symbol in changed:
            if symbol in self._market_data_cache:
                c = self._market_data_cache[symbol]
                updates[symbol] = {
                    'price': round(c.get('price', 0), 4),
                    'change': round(c.get('change', 0), 4),
                    'change_pct': round(c.get('change_pct', 0), 2),
                }
            if symbol in self._ws_tick_history:
                intraday[symbol] = self._build_intraday_ohlc(symbol)
            if symbol in self._ws_extra_info:
                ws_info[symbol] = self._ws_extra_info[symbol]

        if updates:
            payload = json.dumps({'updates': updates, 'intraday': intraday, 'ws_info': ws_info})
            js = f'if(window.updatePrices)window.updatePrices({payload});'
            self.map_widget.page().runJavaScript(js)

    def _build_intraday_ohlc(self, symbol, interval_min=5):
        """Build intraday OHLC: seeded history + WS tick-derived bars merged."""
        # Start with seeded bars from yf.download (keyed by bucket timestamp)
        bars = {}
        seed = self._intraday_ohlc_seed.get(symbol, [])
        for bar in seed:
            ts = int(bar[0])
            bars[ts] = {'o': bar[1], 'h': bar[2], 'l': bar[3], 'c': bar[4]}

        # Layer WS ticks on top (extends chart beyond seed data)
        ticks = self._ws_tick_history.get(symbol, [])
        for ts, price in ticks:
            bucket = int(ts // (interval_min * 60)) * (interval_min * 60)
            if bucket not in bars:
                bars[bucket] = {'o': price, 'h': price, 'l': price, 'c': price}
            else:
                b = bars[bucket]
                b['h'] = max(b['h'], price)
                b['l'] = min(b['l'], price)
                b['c'] = price

        if not bars:
            return []
        return [[t, b['o'], b['h'], b['l'], b['c']] for t, b in sorted(bars.items())]

    def _stop_market_websocket(self):
        """Stop the WebSocket connection and thread."""
        if self._ws_worker is not None:
            self._ws_worker.stop()
        if self._ws_thread is not None and self._ws_thread.isRunning():
            self._ws_thread.quit()
            self._ws_thread.wait(3000)
        self._ws_worker = None
        self._ws_thread = None

    def refresh_market_data(self):
        """Refresh volatility data (VIX, VVIX, SKEW, MOVE) asynchronously.

        OPTIMERING: Flyttar tung yfinance.download till bakgrundstrad.
        First load uses period='max' for full history. Subsequent refreshes
        use period='1y' to avoid re-downloading decades of data every 5 min.
        Cooldown of 5 minutes prevents excessive fetching.
        """
        print("[Volatility] refresh_market_data called")

        # Don't start if already running - använd säker flagga
        if self._volatility_running:
            print("[Volatility] Already running (flag), skipping")
            return

        # Also check if old thread is still alive
        if self._volatility_thread is not None and self._volatility_thread.isRunning():
            print("[Volatility] Old thread still alive, skipping")
            return

        # Cooldown: skip if last fetch was less than 5 minutes ago
        now = time.time()
        last_vol = getattr(self, '_volatility_last_start', 0)
        if last_vol > 0 and (now - last_vol) < 300:
            print(f"[Volatility] Skipped - cooldown ({now - last_vol:.0f}s < 300s)")
            return

        tickers = ['^VIX', '^VVIX', '^SKEW', '^MOVE']

        # Always fetch full history for accurate percentile calculations
        vol_period = 'max'

        # Sätt flagga INNAN vi skapar tråden
        self._volatility_running = True
        self._volatility_last_start = now
        print(f"[Volatility] Creating thread for {tickers} (period={vol_period})")
        
        try:
            # Create and start worker
            self._volatility_thread = QThread()
            self._volatility_worker = VolatilityDataWorker(tickers, period=vol_period)
            self._volatility_worker.moveToThread(self._volatility_thread)
            
            self._volatility_thread.started.connect(self._volatility_worker.run)
            self._volatility_worker.finished.connect(self._volatility_thread.quit)
            # VIKTIGT: Återställ flagga och rensa referenser när tråden är klar
            self._volatility_thread.finished.connect(self._on_volatility_thread_finished)
            self._volatility_worker.result.connect(self._on_volatility_data_received)
            self._volatility_worker.error.connect(self._on_volatility_data_error)
            self._volatility_worker.status_message.connect(self.statusBar().showMessage)
            
            print("[Volatility] Starting thread...")
            self._volatility_thread.start()
            self.statusBar().showMessage("Fetching volatility data in background...")
            print("[Volatility] Thread started successfully")
        except Exception as e:
            print(f"[Volatility] ERROR creating/starting thread: {e}")
            traceback.print_exc()
            self._volatility_running = False
    
    def _on_volatility_thread_finished(self):
        """Handle volatility thread completion — clean up and chain to news.
        
        Always cleans up refs. Only chains to news if this was expected completion.
        """
        was_running = self._volatility_running
        self._volatility_running = False
        
        elapsed = time.time() - getattr(self, '_volatility_last_start', 0)
        print(f"[Volatility] Thread finished ({elapsed:.1f}s, was_running={was_running})")
        
        if self._volatility_worker is not None:
            self._volatility_worker.deleteLater()
        if self._volatility_thread is not None:
            self._volatility_thread.deleteLater()
        
        self._volatility_worker = None
        self._volatility_thread = None
        
        # Chain: start news refresh if 15+ min since last fetch
        # Only if this was expected completion (not stale thread after watchdog)
        if was_running:
            last_news = getattr(self, '_last_news_fetch_time', 0)
            if time.time() - last_news >= 900:  # 900s = 15 min
                QTimer.singleShot(2000, self._refresh_news_feed_safe)
    
    def _on_volatility_data_received(self, close):
        """Handle received volatility data - runs on GUI thread (safe)."""
        try:
            if len(close) == 0:
                print("No volatility data returned")
                return

            # Antal dagar för sparkline (senaste ~30 handelsdagar)
            SPARKLINE_DAYS = 30
            
            # VIX
            try:
                if '^VIX' in close.columns:
                    vix_series = close['^VIX'].dropna()
                    if len(vix_series) > 0:
                        vix_val = vix_series.iloc[-1]
                        vix_prev = vix_series.iloc[-2] if len(vix_series) > 1 else vix_val
                        vix_chg = ((vix_val / vix_prev) - 1) * 100
                        vix_pct = (vix_series < vix_val).sum() / len(vix_series) * 100
                        median = vix_series.median()
                        mode_series = vix_series.mode()
                        mode = mode_series.iloc[0] if len(mode_series) > 0 else median
                        
                        if vix_pct < 10:
                            desc = "Extreme complacency in markets"
                        elif vix_pct < 25:
                            desc = "Low fear - calm markets"
                        elif vix_pct < 50:
                            desc = "Relatively calm conditions"
                        elif vix_pct < 75:
                            desc = "Slightly elevated uncertainty"
                        elif vix_pct < 90:
                            desc = "Elevated fear & uncertainty"
                        elif vix_pct < 95:
                            desc = "Significant market stress"
                        else:
                            desc = "Extreme fear - crisis levels"
                        
                        # Hämta senaste dagarna för sparkline
                        history = vix_series.tail(SPARKLINE_DAYS).tolist()

                        self.vix_card.update_data(vix_val, vix_chg, vix_pct, median, mode, desc, history=history)
                        # Cacha sorterad serie för live-percentilberäkning via WebSocket
                        self._vol_hist_cache['^VIX'] = np.sort(vix_series.values)
                        self._vol_median_cache['^VIX'] = median
                        self._vol_mode_cache['^VIX'] = mode
                        self._vol_sparkline_cache['^VIX'] = history
            except Exception as e:
                print(f"VIX error: {e}")

            # VVIX
            try:
                if '^VVIX' in close.columns:
                    vvix_series = close['^VVIX'].dropna()
                    if len(vvix_series) > 0:
                        vvix_val = vvix_series.iloc[-1]
                        vvix_prev = vvix_series.iloc[-2] if len(vvix_series) > 1 else vvix_val
                        vvix_chg = ((vvix_val / vvix_prev) - 1) * 100
                        vvix_pct = (vvix_series < vvix_val).sum() / len(vvix_series) * 100
                        median = vvix_series.median()
                        mode_series = vvix_series.mode()
                        mode = mode_series.iloc[0] if len(mode_series) > 0 else median
                        
                        if vvix_pct < 10:
                            desc = "Very stable VIX expectations"
                        elif vvix_pct < 25:
                            desc = "Stable VIX outlook"
                        elif vvix_pct < 50:
                            desc = "Below median uncertainty about VIX"
                        elif vvix_pct < 75:
                            desc = "Somewhat uncertain VIX outlook"
                        elif vvix_pct < 90:
                            desc = "Potential for volatility spikes"
                        elif vvix_pct < 95:
                            desc = "VIX itself highly volatile"
                        else:
                            desc = "Crisis-level VIX uncertainty"
                        
                        # Hämta senaste dagarna för sparkline
                        history = vvix_series.tail(SPARKLINE_DAYS).tolist()

                        self.vvix_card.update_data(vvix_val, vvix_chg, vvix_pct, median, mode, desc, history=history)
                        # Cacha sorterad serie för live-percentilberäkning via WebSocket
                        self._vol_hist_cache['^VVIX'] = np.sort(vvix_series.values)
                        self._vol_median_cache['^VVIX'] = median
                        self._vol_mode_cache['^VVIX'] = mode
                        self._vol_sparkline_cache['^VVIX'] = history
                    else:
                        self.vvix_card.value_label.setText("N/A")
                        self.vvix_card.desc_label.setText("No VVIX data available")
            except Exception as e:
                print(f"VVIX error: {e}")
                self.vvix_card.value_label.setText("Error")
            
            # SKEW
            try:
                skew_ok = False
                if '^SKEW' in close.columns:
                    skew_series = close['^SKEW'].dropna()
                    if len(skew_series) > 0:
                        skew_val = skew_series.iloc[-1]
                        skew_prev = skew_series.iloc[-2] if len(skew_series) > 1 else skew_val
                        skew_chg = ((skew_val / skew_prev) - 1) * 100
                        skew_pct = (skew_series < skew_val).sum() / len(skew_series) * 100
                        median = skew_series.median()
                        mode_series = skew_series.mode()
                        mode = mode_series.iloc[0] if len(mode_series) > 0 else median

                        if skew_pct < 10:
                            desc = "Very low tail risk pricing"
                        elif skew_pct < 25:
                            desc = "Limited crash protection demand"
                        elif skew_pct < 50:
                            desc = "Below median tail risk perception"
                        elif skew_pct < 75:
                            desc = "Modest downside protection demand"
                        elif skew_pct < 90:
                            desc = "Increased put demand"
                        elif skew_pct < 95:
                            desc = "Significant crash hedging"
                        else:
                            desc = "Major crash protection demand"

                        # Hämta senaste dagarna för sparkline
                        history = skew_series.tail(SPARKLINE_DAYS).tolist()

                        self.skew_card.update_data(skew_val, skew_chg, skew_pct, median, mode, desc, history=history)
                        # Cacha sorterad serie för live-percentilberäkning via WebSocket
                        self._vol_hist_cache['^SKEW'] = np.sort(skew_series.values)
                        self._vol_median_cache['^SKEW'] = median
                        self._vol_mode_cache['^SKEW'] = mode
                        self._vol_sparkline_cache['^SKEW'] = history
                        skew_ok = True
                # Fallback: använd cachad data om yfinance inte returnerade SKEW
                if not skew_ok:
                    self._apply_vol_card_fallback('^SKEW', self.skew_card)
            except Exception as e:
                print(f"SKEW error: {e}")
                self._apply_vol_card_fallback('^SKEW', self.skew_card)
            
            # MOVE (Bond Market Volatility)
            try:
                move_ok = False
                if '^MOVE' in close.columns:
                    move_series = close['^MOVE'].dropna()
                    if len(move_series) > 0:
                        move_val = move_series.iloc[-1]
                        move_prev = move_series.iloc[-2] if len(move_series) > 1 else move_val
                        move_chg = ((move_val / move_prev) - 1) * 100
                        move_pct = (move_series < move_val).sum() / len(move_series) * 100
                        median = move_series.median()
                        mode_series = move_series.mode()
                        mode = mode_series.iloc[0] if len(mode_series) > 0 else median

                        if move_pct < 10:
                            desc = "Low rate volatility"
                        elif move_pct < 25:
                            desc = "Stable rate environment"
                        elif move_pct < 50:
                            desc = "Relatively stable rates"
                        elif move_pct < 75:
                            desc = "Some rate uncertainty"
                        elif move_pct < 90:
                            desc = "Significant rate uncertainty"
                        elif move_pct < 95:
                            desc = "Major rate movements expected"
                        else:
                            desc = "Crisis-level rate uncertainty"

                        # Hämta senaste dagarna för sparkline
                        history = move_series.tail(SPARKLINE_DAYS).tolist()

                        self.move_card.update_data(move_val, move_chg, move_pct, median, mode, desc, history=history)
                        # Cacha sorterad serie för live-percentilberäkning via WebSocket
                        self._vol_hist_cache['^MOVE'] = np.sort(move_series.values)
                        self._vol_median_cache['^MOVE'] = median
                        self._vol_mode_cache['^MOVE'] = mode
                        self._vol_sparkline_cache['^MOVE'] = history
                        move_ok = True
                # Fallback: använd cachad data om yfinance inte returnerade MOVE
                if not move_ok:
                    self._apply_vol_card_fallback('^MOVE', self.move_card)
            except Exception as e:
                print(f"MOVE error: {e}")
                self._apply_vol_card_fallback('^MOVE', self.move_card)
            
            # Spara percentildata till disk (laddas vid nästa uppstart istället för yf.download)
            if self._vol_hist_cache:
                save_volatility_cache(
                    self._vol_hist_cache, self._vol_median_cache,
                    self._vol_mode_cache, self._vol_sparkline_cache)

            self.last_updated_label.setText(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            self.statusBar().showMessage("Volatility data updated")

        except Exception as e:
            print(f"Volatility data UI error: {e}")
            traceback.print_exc()
    
    def _on_volatility_data_error(self, error: str):
        """Handle volatility data fetch error."""
        print(f"Volatility data error: {error}")
        self.statusBar().showMessage(f"Volatility data error: {error}")


    def load_csv(self):
        """Load tickers from CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Tickers CSV", "", "CSV Files (*.csv)")
        if filepath:
            try:
                tickers = load_tickers_from_csv(filepath)
                self.tickers_input.setText(", ".join(tickers))
                self.statusBar().showMessage(f"Loaded {len(tickers)} tickers from CSV")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load CSV:\n{e}")
    
    def run_analysis(self):
        """Run pair analysis."""
        ticker_text = self.tickers_input.text()
        tickers = [t.strip().upper() for t in ticker_text.replace('\n', ',').split(',') if t.strip()]
        
        if not tickers:
            QMessageBox.warning(self, "No Tickers", "Please enter some tickers to analyze.")
            return
        
        # Fix #8: Clean up previous worker to prevent memory leak
        self._cleanup_analysis_worker()
        
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Map window preset to robustness_windows list
        window_presets = {
            "Standard (500-2000)": [500, 750, 1000, 1250, 1500, 1750, 2000],
            "Quick (750-1500)": [750, 1000, 1250, 1500],
            "Extended (500-2500)": [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500],
        }
        preset = self.lookback_combo.currentText()
        windows = window_presets.get(preset, [500, 750, 1000, 1250, 1500, 1750, 2000])
        max_pts = max(windows) + 500 if windows else 2500

        config = {
            'lookback_period': 'max',
            'min_half_life': 1,
            'max_half_life': 60,
            'max_adf_pvalue': 0.05,
            'min_correlation': 0.70,
            'robustness_windows': windows,
            'min_windows_passed': 4,
            'max_data_points': max_pts,
        }

        self.worker_thread = QThread()
        self.worker = AnalysisWorker(tickers, 'max', config)
        self.worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        # Fix #8: Schedule cleanup after thread finishes
        self.worker.finished.connect(lambda: QTimer.singleShot(100, self._cleanup_analysis_worker))
        self.worker.progress.connect(self.on_analysis_progress)
        self.worker.result.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        
        self.worker_thread.start()
    
    def _cleanup_analysis_worker(self):
        """Clean up analysis worker resources (Fix #8)."""
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.deleteLater()
            except RuntimeError:
                pass  # Already deleted
            self.worker = None
        if self.worker_thread is not None:
            try:
                if self.worker_thread.isRunning():
                    self.worker_thread.quit()
                    self.worker_thread.wait(1000)
                self.worker_thread.deleteLater()
            except RuntimeError:
                pass  # Already deleted
            self.worker_thread = None
    
    def on_analysis_progress(self, progress: int, message: str):
        """Update progress bar."""
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(message)
    
    def on_analysis_complete(self, engine: PairsTradingEngine):
        """Handle analysis completion."""
        self.engine = engine
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Update metrics
        n_tickers = len(engine.price_data.columns)
        n_pairs = len(engine.pairs_stats) if engine.pairs_stats is not None else 0
        n_viable = len(engine.viable_pairs) if engine.viable_pairs is not None else 0
        
        # Update metric values (find the value label inside each frame)
        self._update_metric_value(self.tickers_metric, str(n_tickers))
        self._update_metric_value(self.pairs_metric, str(n_pairs))
        self._update_metric_value(self.viable_metric, str(n_viable))
        self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
        
        # Update viable count label
        if hasattr(self, 'viable_count_label'):
            self.viable_count_label.setText(f"({n_viable})")
        
        # OPTIMERING: Uppdatera endast laddade tabbar (lazy loading)
        if self._tabs_loaded.get(1, False):  # Arbitrage Scanner
            self.update_viable_table()
            self.update_all_pairs_table()
        if self._tabs_loaded.get(2, False):  # OU Analytics
            self.update_ou_pair_list()
        if self._tabs_loaded.get(3, False):  # Pair Signals
            self.update_signals_list()
        
        # Save engine cache to file (for sync between computers)
        save_engine_cache(engine)
        if os.path.exists(ENGINE_CACHE_FILE):
            self._engine_cache_mtime = os.path.getmtime(ENGINE_CACHE_FILE)
        
        self.statusBar().showMessage(f"Analysis complete: {n_viable} viable pairs found")
        
        # If this was a scheduled scan, send Discord and refresh volatility cache
        if self._is_scheduled_scan:
            self._is_scheduled_scan = False
            self._scheduled_scan_running = False
            self.send_scan_results_to_discord()
            # Uppdatera volatilitets-percentilcache (daglig refresh)
            QTimer.singleShot(5000, self._start_volatility_refresh_safe)
            self.statusBar().showMessage("Scheduled scan complete - results sent to Discord")
    
    def _update_metric_value(self, frame: Optional[QFrame], value: str):
        """Update the value label inside a metric frame.
        
        OPTIMERING: Hanterar None-frames för lazy loading.
        """
        if frame is None:
            return  # Tab inte laddad ännu
        for child in frame.findChildren(QLabel):
            if child.objectName() == "metricValue":
                child.setText(value)
                break
    
    def on_analysis_error(self, error: str):
        """Handle analysis error."""
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", error)
        
        # If this was part of a scheduled scan, send Discord notification with error
        if hasattr(self, '_is_scheduled_scan') and self._is_scheduled_scan:
            self._is_scheduled_scan = False  # Reset flag
            self._scheduled_scan_running = False  # Reset guard flag
            self.send_discord_notification(
                title="⚠️ Scheduled Scan - Analysis Error",
                description=f"Pair analysis failed:\n{error}",
                color=0xff0000  # Red
            )
            self.statusBar().showMessage("Scheduled scan: Analysis failed - error sent to Discord")
    
    def update_viable_table(self):
        """Update viable pairs table."""
        # OPTIMERING: Guard för lazy loading
        if not hasattr(self, 'viable_table') or self.viable_table is None:
            return
        
        if self.engine is None or self.engine.viable_pairs is None:
            self.viable_table.setRowCount(0)
            if hasattr(self, 'viable_count_label'):
                self.viable_count_label.setText("(0)")
            return
        
        df = self.engine.viable_pairs
        # Sort by robustness_score descending
        if 'robustness_score' in df.columns:
            df = df.sort_values('robustness_score', ascending=False)
        if hasattr(self, 'viable_count_label'):
            self.viable_count_label.setText(f"({len(df)})")

        # Fix #1: Disable updates during batch operations
        self.viable_table.setUpdatesEnabled(False)
        try:
            self.viable_table.setRowCount(len(df))
            # Fix #5: Use itertuples instead of iterrows (10-100x faster)
            for i, row in enumerate(df.itertuples()):
                self.viable_table.setItem(i, 0, QTableWidgetItem(row.pair))
                # Z-Score from shortest passing window
                try:
                    _, _, z_val = self.engine.get_pair_ou_params(row.pair, use_raw_data=True)
                    z_item = QTableWidgetItem(f"{z_val:+.2f}")
                    if abs(z_val) >= SIGNAL_TAB_THRESHOLD:
                        z_item.setForeground(QColor(COLORS['positive'] if z_val < 0 else COLORS['negative']))
                        z_item.setBackground(QColor(COLORS['positive_bg'] if z_val < 0 else COLORS['negative_bg']))
                    self.viable_table.setItem(i, 1, z_item)
                except Exception:
                    self.viable_table.setItem(i, 1, QTableWidgetItem("-"))
                self.viable_table.setItem(i, 2, QTableWidgetItem(f"{row.half_life_days:.2f}"))
                self.viable_table.setItem(i, 3, QTableWidgetItem(f"{row.eg_pvalue:.4f}"))
                self.viable_table.setItem(i, 4, QTableWidgetItem(f"{row.johansen_trace:.2f}"))
                self.viable_table.setItem(i, 5, QTableWidgetItem(f"{row.hurst_exponent:.2f}"))
                self.viable_table.setItem(i, 6, QTableWidgetItem(f"{row.correlation:.2f}"))
                # Robustness score — colored dots + text
                wp = getattr(row, 'windows_passed', 0)
                wt = getattr(row, 'windows_tested', 0)
                pair_name = row.pair
                window_details = self.engine.get_window_details(pair_name) if self.engine else []
                rob_widget = QWidget()
                rob_layout = QHBoxLayout(rob_widget)
                rob_layout.setContentsMargins(6, 0, 6, 0)
                rob_layout.setSpacing(3)
                if window_details:
                    for wd in window_details:
                        dot = QLabel()
                        dot.setFixedSize(10, 10)
                        passed = wd.get('passed', False)
                        color = COLORS['positive'] if passed else COLORS['negative']
                        dot.setStyleSheet(
                            f"background-color: {color}; border-radius: 5px; border: none;")
                        ws = wd.get('window_size', '?')
                        status = "PASS" if passed else "FAIL"
                        failed_at = wd.get('failed_at', '')
                        tip = f"Window {ws}: {status}"
                        if failed_at:
                            tip += f" (failed at {failed_at})"
                        dot.setToolTip(tip)
                        rob_layout.addWidget(dot)
                rob_label = QLabel(f"{wp}/{wt}" if wt > 0 else "N/A")
                rob_label.setStyleSheet(
                    f"color: {COLORS['text_secondary']}; font-size: 11px; "
                    f"background: transparent; border: none; margin-left: 4px;")
                rob_layout.addWidget(rob_label)
                rob_layout.addStretch()
                self.viable_table.setCellWidget(i, 7, rob_widget)
                # Kalman diagnostics
                ks = getattr(row, 'kalman_stability', None)
                self.viable_table.setItem(i, 8, QTableWidgetItem(
                    f"{ks:.2f}" if ks is not None else "N/A"))
                ir = getattr(row, 'kalman_innovation_ratio', None)
                self.viable_table.setItem(i, 9, QTableWidgetItem(
                    f"{ir:.2f}" if ir is not None else "N/A"))
                rs = getattr(row, 'kalman_regime_score', None)
                self.viable_table.setItem(i, 10, QTableWidgetItem(
                    f"{rs:.2f}" if rs is not None else "N/A"))
                ts = getattr(row, 'kalman_theta_significant', None)
                self.viable_table.setItem(i, 11, QTableWidgetItem(
                    "Yes" if ts else ("No" if ts is not None else "N/A")))
        finally:
            self.viable_table.setUpdatesEnabled(True)
    
    def update_all_pairs_table(self):
        """Update all pairs table."""
        if self.engine is None or self.engine.pairs_stats is None:
            return
        
        df = self.engine.pairs_stats
        self.all_pairs_btn.setText(f"▼ View All Analyzed Pairs ({len(df)})")
        
        # Fix #1: Disable updates during batch operations
        self.all_pairs_table.setUpdatesEnabled(False)
        try:
            self.all_pairs_table.setRowCount(len(df))
            cols = ["pair", "half_life_days", "eg_pvalue", "johansen_trace", "hurst_exponent", "correlation"]
            # Fix #5: Use itertuples instead of iterrows
            for i, row in enumerate(df.itertuples()):
                for j, col in enumerate(cols):
                    val = getattr(row, col, None)
                    if val is not None:
                        if isinstance(val, float):
                            text = f"{val:.4f}"
                        elif isinstance(val, bool):
                            text = "✓" if val else ""
                        else:
                            text = str(val)
                        self.all_pairs_table.setItem(i, j, QTableWidgetItem(text))
        finally:
            self.all_pairs_table.setUpdatesEnabled(True)
    
    def toggle_all_pairs(self):
        """Toggle all pairs table visibility."""
        visible = not self.all_pairs_table.isVisible()
        self.all_pairs_table.setVisible(visible)
        
        if visible:
            self.all_pairs_btn.setText(self.all_pairs_btn.text().replace("▼", "▲"))
        else:
            self.all_pairs_btn.setText(self.all_pairs_btn.text().replace("▲", "▼"))
    
    def on_viable_pair_selected(self):
        """Legacy — redirects to single-click handler."""
        self._on_scanner_pair_clicked()

    def _on_scanner_pair_clicked(self):
        """Single-click: show window breakdown panel inline."""
        selected = self.viable_table.selectedItems()
        if not selected:
            self.window_detail_frame.setVisible(False)
            return
        row_idx = selected[0].row()
        pair_item = self.viable_table.item(row_idx, 0)
        if pair_item is None:
            return
        pair = pair_item.text()
        self.selected_pair = pair

        # Populate window breakdown panel
        details = self.engine.get_window_details(pair) if self.engine else []
        self.window_detail_header.setText(f"WINDOW BREAKDOWN — {pair}")
        self.window_detail_table.setRowCount(len(details))
        for i, wd in enumerate(details):
            passed = wd.get('passed', False)
            bg = "rgba(34, 197, 94, 0.10)" if passed else "rgba(239, 68, 68, 0.10)"
            status_color = COLORS['positive'] if passed else COLORS['negative']

            def _item(text, color=None):
                item = QTableWidgetItem(str(text))
                if color:
                    item.setForeground(QColor(color))
                item.setBackground(QColor(bg))
                return item

            self.window_detail_table.setItem(i, 0, _item(wd.get('window_size', '?')))
            self.window_detail_table.setItem(i, 1, _item(
                "PASS" if passed else "FAIL", status_color))
            hl = wd.get('half_life_days')
            self.window_detail_table.setItem(i, 2, _item(
                f"{hl:.1f}" if hl is not None else "-"))
            eg = wd.get('eg_pvalue')
            self.window_detail_table.setItem(i, 3, _item(
                f"{eg:.4f}" if eg is not None else "-"))
            jt = wd.get('johansen_trace')
            self.window_detail_table.setItem(i, 4, _item(
                f"{jt:.1f}" if jt is not None else "-"))
            h = wd.get('hurst_exponent')
            self.window_detail_table.setItem(i, 5, _item(
                f"{h:.3f}" if h is not None else "-"))
            c = wd.get('correlation')
            self.window_detail_table.setItem(i, 6, _item(
                f"{c:.3f}" if c is not None else "-"))
            ks = wd.get('kalman_stability')
            self.window_detail_table.setItem(i, 7, _item(
                f"{ks:.3f}" if ks is not None else "-"))
            fa = wd.get('failed_at', '')
            self.window_detail_table.setItem(i, 8, _item(
                fa if fa else "", COLORS['warning'] if fa else None))

        self.window_detail_frame.setVisible(len(details) > 0)

    def _on_scanner_pair_double_clicked(self):
        """Double-click: navigate to OU Analytics tab."""
        selected = self.viable_table.selectedItems()
        if not selected:
            return
        pair_item = self.viable_table.item(selected[0].row(), 0)
        if pair_item is None:
            return
        pair = pair_item.text()
        self.selected_pair = pair
        # Switch to OU tab first (triggers lazy load if needed), then select pair
        self.tabs.setCurrentIndex(2)
        self.ou_pair_combo.setCurrentText(pair)
    
    def update_ou_pair_list(self):
        """Update OU analytics pair dropdown.
        
        When 'Viable only' is checked, re-validates each pair using current
        Kalman half-life (not just scan-time OLS half-life).
        """
        self.ou_pair_combo.clear()
        
        if self.engine is None:
            return
        
        if self.viable_only_check.isChecked():
            if self.engine.viable_pairs is not None:
                min_hl = self.engine.config.get('min_half_life', 1)
                max_hl = self.engine.config.get('max_half_life', 60)
                pairs = []
                for p in self.engine.viable_pairs['pair'].tolist():
                    try:
                        ou, _, _ = self.engine.get_pair_ou_params(p, use_raw_data=True)
                        hl = ou.half_life_days()
                        if min_hl <= hl <= max_hl:
                            pairs.append(p)
                    except Exception:
                        pass
            else:
                pairs = []
        else:
            if self.engine.pairs_stats is not None:
                pairs = self.engine.pairs_stats['pair'].tolist()
            else:
                pairs = []
        
        self.ou_pair_combo.addItems(pairs)
    
    def on_ou_pair_changed(self, pair: str):
        """Handle OU pair selection change."""
        if not pair or self.engine is None:
            return

        self.selected_pair = pair
        self.update_ou_display(pair)

    def update_ou_display(self, pair: str):
        """Update OU analytics display for selected pair."""
        if self.engine is None:
            return

        try:
            ou, spread, z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
            
            # Get pair stats
            if self.engine.viable_pairs is not None and pair in self.engine.viable_pairs['pair'].values:
                pair_stats = self.engine.viable_pairs[self.engine.viable_pairs['pair'] == pair].iloc[0]
                # Re-check viability using CURRENT Kalman half-life (scan used OLS)
                current_hl = ou.half_life_days()
                min_hl = self.engine.config.get('min_half_life', 1)
                max_hl = self.engine.config.get('max_half_life', 60)
                is_viable = (min_hl <= current_hl <= max_hl)
            else:
                pair_stats = self.engine.pairs_stats[self.engine.pairs_stats['pair'] == pair].iloc[0]
                is_viable = pair_stats.get('is_viable', False)
            
            # Update metric cards (using Kalman-filtered estimates from ou object)
            self.ou_theta_card.set_value(f"{ou.theta:.2f}")
            self.ou_mu_card.set_value(f"{ou.mu:.2f}")
            self.ou_halflife_card.set_value(f"{ou.half_life_days():.2f} days")
            
            z_color = "#ff1744" if z > 2 else ("#00c853" if z < -2 else "#ffffff")
            self.ou_zscore_card.set_value(f"{z:.2f}", z_color)
            
            self.ou_hedge_card.set_value(f"{pair_stats['hedge_ratio']:.4f}")
            
            status_text = "✅ VIABLE" if is_viable else "⚠️ NON-VIABLE"
            status_color = "#00c853" if is_viable else "#ffc107"
            self.ou_status_card.set_value(status_text, status_color)
            
            # Update Kalman diagnostics (if available)
            kalman = getattr(ou, '_kalman', None)
            if kalman is not None:
                # Stability: 0-1 scale, higher = more stable parameters
                stab = kalman.param_stability
                stab_color = "#00c853" if stab > 0.7 else ("#ffc107" if stab > 0.4 else "#ff1744")
                self.kalman_stability_card.set_value(f"{stab:.1%}", stab_color)
                
                # Effective sample size
                self.kalman_ess_card.set_value(f"{kalman.effective_sample_size:.0f}")
                
                # θ confidence interval
                theta_lo = max(0, kalman.theta - 1.96 * kalman.theta_std)
                theta_hi = kalman.theta + 1.96 * kalman.theta_std
                hl_lo = np.log(2) / theta_hi * 252 if theta_hi > 0 else np.inf
                hl_hi = np.log(2) / theta_lo * 252 if theta_lo > 0 else np.inf
                self.kalman_theta_ci_card.set_value(f"{hl_lo:.0f}-{hl_hi:.0f}d")
                
                # μ confidence interval
                mu_lo = kalman.mu - 1.96 * kalman.mu_std
                mu_hi = kalman.mu + 1.96 * kalman.mu_std
                self.kalman_mu_ci_card.set_value(f"[{mu_lo:.2f}, {mu_hi:.2f}]")
                
                # Innovation ratio (should be ~1.0 if filter is well-calibrated)
                ir = kalman.innovation_ratio
                ir_color = "#00c853" if 0.5 < ir < 2.0 else "#ff1744"
                self.kalman_innovation_card.set_value(f"{ir:.2f}", ir_color)
                
                # Regime change CUSUM score
                rc = kalman.regime_change_score
                rc_color = "#ff1744" if rc > 4.0 else ("#ffc107" if rc > 2.0 else "#00c853")
                rc_text = f"{rc:.1f}" + (" ⚠" if rc > 4.0 else "")
                self.kalman_regime_card.set_value(rc_text, rc_color)
            else:
                # Kalman not available (fallback method used)
                for card in [self.kalman_stability_card, self.kalman_ess_card,
                             self.kalman_theta_ci_card, self.kalman_mu_ci_card,
                             self.kalman_innovation_card, self.kalman_regime_card]:
                    card.set_value("N/A", "#666666")
            
            # Calculate Expected Move to Mean
            # Spread: S = Y - β*X - α
            # Z = (S - μ) / σ_eq
            # To reach Z=0: ΔS = -Z * σ_eq
            hedge_ratio = pair_stats['hedge_ratio']
            delta_spread = -z * ou.eq_std  # Spread change needed to reach mean
            
            # Get current prices
            y_ticker, x_ticker = pair.split('/')
            prices = self.engine.price_data[[y_ticker, x_ticker]].dropna()
            y_price = prices[y_ticker].iloc[-1]
            x_price = prices[x_ticker].iloc[-1]
            
            # Scenario 1: Only Y moves (100%) -> ΔY = ΔS
            delta_y_full = delta_spread
            delta_y_full_pct = (delta_y_full / y_price) * 100
            
            # Scenario 2: Only X moves (100%) -> ΔX = -ΔS / β
            delta_x_full = -delta_spread / hedge_ratio if hedge_ratio != 0 else 0
            delta_x_full_pct = (delta_x_full / x_price) * 100
            
            # Scenario 3: 50/50 split
            delta_y_half = delta_spread / 2
            delta_y_half_pct = (delta_y_half / y_price) * 100
            delta_x_half = -delta_spread / (2 * hedge_ratio) if hedge_ratio != 0 else 0
            delta_x_half_pct = (delta_x_half / x_price) * 100
            
            # Update Expected Move cards
            spread_color = "#00c853" if delta_spread > 0 else "#ff1744"
            self.exp_spread_change_card.set_value(f"{delta_spread:+.2f}", spread_color)
            
            # Y only (100%)
            y_color = "#00c853" if delta_y_full > 0 else "#ff1744"
            self.exp_y_only_card.set_title(f"{display_ticker(y_ticker)} (100% of the move)")
            self.exp_y_only_card.set_value(f"{delta_y_full:+.2f} ({delta_y_full_pct:+.2f}%)", y_color)

            # X only (100%)
            x_color = "#00c853" if delta_x_full > 0 else "#ff1744"
            self.exp_x_only_card.set_title(f"{display_ticker(x_ticker)} (100% of the move)")
            self.exp_x_only_card.set_value(f"{delta_x_full:+.2f} ({delta_x_full_pct:+.2f}%)", x_color)
           
            # Update window robustness section
            self._update_window_robustness(pair, pair_stats)

            # Update charts
            if ensure_pyqtgraph():
                self.update_ou_charts(pair, ou, spread, z, pair_stats)

        except Exception as e:
            print(f"OU display error: {e}")

    def _update_window_robustness(self, pair: str, pair_stats):
        """Update the WINDOW ROBUSTNESS section in the OU analytics left panel."""
        try:
            # Update robustness score card
            wp = pair_stats.get('windows_passed', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'windows_passed', 0)
            wt = pair_stats.get('windows_tested', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'windows_tested', 0)
            if wt > 0:
                pct = wp / wt * 100
                score_text = f"{wp}/{wt} ({pct:.0f}%)"
                if pct >= 70:
                    score_color = "#00c853"
                elif pct >= 50:
                    score_color = "#ffc107"
                else:
                    score_color = "#ff1744"
            else:
                score_text = "N/A"
                score_color = COLORS['text_muted']
            self.ou_robustness_card.set_value(score_text, score_color)

            details = self.engine.get_window_details(pair) if self.engine else []

            # Clear old window rows
            while self.ou_window_container_layout.count():
                child = self.ou_window_container_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Add new window rows
            for wd in details:
                row = WindowResultRow(wd)
                self.ou_window_container_layout.addWidget(row)
        except Exception as e:
            print(f"Window robustness update error: {e}")

    def update_ou_charts(self, pair: str, ou, spread: pd.Series, z: float, pair_stats):
        """Update OU analytics charts with dates, crosshairs, and synchronized zoom."""
        tickers = pair.split('/')
        if len(tickers) != 2:
            return

        y_ticker, x_ticker = tickers
        hedge_ratio = pair_stats['hedge_ratio']
        intercept = pair_stats.get('intercept', 0)

        # Align price data to spread's date range so both charts use same x-axis
        prices = self.engine.price_data[[y_ticker, x_ticker]].dropna()
        common_dates = prices.index.intersection(spread.index)
        if len(common_dates) == 0:
            return
        prices = prices.loc[common_dates]
        spread = spread.loc[common_dates]

        x_axis = np.arange(len(prices))
        dates = prices.index  # DatetimeIndex for display

        # ===== UPDATE DATE AXES =====
        if hasattr(self, 'ou_price_date_axis'):
            self.ou_price_date_axis.set_dates(dates)
        if hasattr(self, 'ou_zscore_date_axis'):
            self.ou_zscore_date_axis.set_dates(dates)

        # ===== PRICE COMPARISON =====
        self.ou_price_plot.clear()
        self.ou_price_plot.addLegend()
        y_series = prices[y_ticker].values
        x_adjusted = (hedge_ratio * prices[x_ticker] + intercept).values
        self.ou_price_plot.plot(x_axis, y_series, pen=pg.mkPen('#d4a574', width=2), name=f"{display_ticker(y_ticker)} (Y)")
        self.ou_price_plot.plot(x_axis, x_adjusted, pen=pg.mkPen('#2196f3', width=2), name=f"β·{display_ticker(x_ticker)} + α")

        # ===== SPREAD with μ and ±2σ BANDS =====
        self.ou_zscore_plot.clear()

        spread_values = spread.values

        # Use Kalman time-varying μ and σ if available
        kalman = getattr(ou, '_kalman', None)
        if kalman is not None and kalman.mu_history is not None and kalman.eq_std_history is not None:
            mu_hist = kalman.mu_history
            std_hist = kalman.eq_std_history

            # Pad front to align with x_axis (Kalman history has n-1 points)
            n_pad = len(x_axis) - len(mu_hist)
            if n_pad > 0:
                mu_vals = np.concatenate([np.full(n_pad, np.nan), mu_hist])
                std_vals = np.concatenate([np.full(n_pad, np.nan), std_hist])
            else:
                mu_vals = mu_hist[-len(x_axis):]
                std_vals = std_hist[-len(x_axis):]

            upper_2 = mu_vals + 2 * std_vals
            lower_2 = mu_vals - 2 * std_vals

            # Plot ±2σ band as fill
            try:
                upper_curve = self.ou_zscore_plot.plot(x_axis, upper_2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
                lower_curve = self.ou_zscore_plot.plot(x_axis, lower_2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
                fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(255, 255, 255, 20))
                self.ou_zscore_plot.addItem(fill)
            except Exception:
                pass

            # Plot μ line
            self.ou_zscore_plot.plot(x_axis, mu_vals, pen=pg.mkPen('#ffffff', width=1), name='μ')
        else:
            # Fallback: static μ and σ from final OU params
            mu_val = ou.mu
            eq_std_val = ou.eq_std
            self.ou_zscore_plot.addLine(y=mu_val, pen=pg.mkPen('#ffffff', width=1))
            self.ou_zscore_plot.addLine(y=mu_val + 2*eq_std_val, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.ou_zscore_plot.addLine(y=mu_val - 2*eq_std_val, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))

        # Plot the raw spread
        self.ou_zscore_plot.plot(x_axis, spread_values, pen=pg.mkPen('#00e5ff', width=2), name='Spread')

        # Keep zscore Series for crosshair (show spread value on hover)
        zscore = spread
        
        # ===== AUTO-RANGE: Reset view to show full data after plotting =====
        self._ou_syncing_plots = True
        self.ou_price_plot.enableAutoRange()
        self.ou_zscore_plot.enableAutoRange()
        QTimer.singleShot(50, lambda: setattr(self, '_ou_syncing_plots', False))
        
        # ===== SETUP CROSSHAIRS FOR INTERACTIVITY =====
        # Remove old crosshairs if they exist
        if hasattr(self, 'price_crosshair') and self.price_crosshair is not None:
            self.price_crosshair.cleanup()
        
        if hasattr(self, 'zscore_crosshair') and self.zscore_crosshair is not None:
            self.zscore_crosshair.cleanup()
        
        # Create new synchronized crosshairs (lazy load class)
        CrosshairManager = get_crosshair_manager_class()
        if CrosshairManager:
            self.price_crosshair = CrosshairManager(
                self.ou_price_plot, 
                dates=dates,
                data_series={display_ticker(y_ticker): y_series, f"β·{display_ticker(x_ticker)}+α": x_adjusted},
                label_format="{:.2f}"
            )
            
            self.zscore_crosshair = CrosshairManager(
                self.ou_zscore_plot,
                dates=dates,
                data_series={'Spread': zscore.values},
                label_format="{:.2f}"
            )
            
            # Link crosshairs for synchronization
            self.price_crosshair.add_synced_manager(self.zscore_crosshair)
            self.zscore_crosshair.add_synced_manager(self.price_crosshair)
        
        # ===== CONDITIONAL DISTRIBUTION =====
        self.ou_dist_plot.clear()
        from scipy import stats
        x_range = np.linspace(ou.mu - 4*ou.eq_std, ou.mu + 4*ou.eq_std, 200)
        current_spread = spread.iloc[-1]
        
        colors = ['#FF9900', '#66BB6A', '#00BCD4', '#9C27B0', '#CDDC39', '#3F51B5']
        horizons = [5, 10, 20, 50, 100, 200]
        
        for i, days in enumerate(horizons):
            tau = days / 252
            mean = ou.conditional_mean(current_spread, tau)
            std = ou.conditional_std(tau)
            pdf = stats.norm.pdf(x_range, loc=mean, scale=std)
            self.ou_dist_plot.plot(x_range, pdf, pen=pg.mkPen(colors[i], width=2), name=f'{days}d')
        
        self.ou_dist_plot.addLine(x=current_spread, pen=pg.mkPen('#ff1744', width=2, style=Qt.DashLine))
        self.ou_dist_plot.addLine(x=ou.mu, pen=pg.mkPen('#ffffff', width=1))
        
        # ===== EXPECTED PATH =====
        self.ou_path_plot.clear()
        hl_days = ou.half_life_days()
        path_end = max(30, int(hl_days + 10)) if hl_days < 500 else 70
        days_range = np.arange(0, path_end + 1)
        taus = days_range / 252
        expected_path = np.array([ou.conditional_mean(current_spread, t) for t in taus])
        ci_results = [ou.confidence_interval(current_spread, t, 0.90) for t in taus]
        ci_low = np.array([c[0] for c in ci_results])
        ci_high = np.array([c[1] for c in ci_results])
        
        # OU expected path (orange)
        self.ou_path_plot.plot(days_range, expected_path, pen=pg.mkPen('#d4a574', width=2))
        self.ou_path_plot.plot(days_range, ci_high, pen=pg.mkPen('#d4a574', width=1, style=Qt.DashLine))
        self.ou_path_plot.plot(days_range, ci_low, pen=pg.mkPen('#d4a574', width=1, style=Qt.DashLine))
        
        self.ou_path_plot.addLine(y=ou.mu, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
        self.ou_path_plot.addLine(x=ou.half_life_days(), pen=pg.mkPen('#ffaa00', width=1, style=Qt.DashLine))
        
    def update_signals_list(self):
        """Update signals dropdown."""
        self.signal_combo.clear()
        
        if self.engine is None or self.engine.viable_pairs is None:
            self.signal_count_label.setText(f"⚡ 0 viable pairs with |Z| ≥ {SIGNAL_TAB_THRESHOLD}")
            return
        
        signals = []
        min_hl = self.engine.config.get('min_half_life', 1)
        max_hl = self.engine.config.get('max_half_life', 60)
        
        for row in self.engine.viable_pairs.itertuples():
            try:
                ou, spread, z = self.engine.get_pair_ou_params(row.pair, use_raw_data=True)

                # Re-check viability with CURRENT Kalman half-life
                current_hl = ou.half_life_days()
                if not (min_hl <= current_hl <= max_hl):
                    continue

                if abs(z) >= SIGNAL_TAB_THRESHOLD:
                    signals.append((row.pair, z))
            except (ValueError, KeyError, Exception):
                pass
        
        self.signal_count_label.setText(f"⚡ {len(signals)} viable pairs with |Z| ≥ {SIGNAL_TAB_THRESHOLD}")
        
        for pair, z in signals:
            self.signal_combo.addItem(pair)
    
    def on_signal_changed(self, pair: str):
        """Handle signal selection change."""
        if not pair or self.engine is None:
            if hasattr(self, 'open_position_btn'):
                self.open_position_btn.setEnabled(False)
            return
        
        try:
            ou, spread, z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
            pair_stats = self.engine.viable_pairs[self.engine.viable_pairs['pair'] == pair].iloc[0]
            
            # Update signal state cards
            self.signal_pair_card.set_value(pair)
            
            z_color = "#ff1744" if z > 0 else "#00c853"
            self.signal_z_card.set_value(f"{z:.2f}", z_color)
            
            direction = "LONG SPREAD" if z < 0 else "SHORT SPREAD"
            self.signal_dir_card.set_value(direction, z_color)
            
            self.signal_hedge_card.set_value(f"{pair_stats['hedge_ratio']:.4f}")
            self.signal_hl_card.set_value(f"{pair_stats['half_life_days']:.2f}d")
            
            # Enable Open Position button
            if hasattr(self, 'open_position_btn'):
                self.open_position_btn.setEnabled(True)
            
            # Update selected pair and clear old mini futures data
            self.signal_selected_pair = pair  # Use separate variable for signals tab
            self.current_mini_futures = {}  # Clear old data to prevent mismatch
            
            # Update leverage labels with actual ticker names
            tickers = pair.split('/')
            if len(tickers) == 2:
                y_ticker, x_ticker = tickers
                if hasattr(self, 'lev_y_label'):
                    self.lev_y_label.setText(f"Leverage {display_ticker(y_ticker)}")
                if hasattr(self, 'lev_x_label'):
                    self.lev_x_label.setText(f"Leverage {display_ticker(x_ticker)}")
            
            # Update mini-futures FIRST (this sets current_mini_futures)
            self.update_mini_futures(pair, z)
            
            # Then update position sizing (uses current_mini_futures)
            self.update_position_sizing()
            
            # Update signal tab plots
            self._update_signal_plots(pair, pair_stats, ou, spread, z)
            
        except Exception as e:
            print(f"Signal display error: {e}")
    
    def _update_signal_plots(self, pair: str, pair_stats, ou, spread, z: float):
        """Update the price comparison and z-score plots in the signals tab."""
        pg = get_pyqtgraph()
        if pg is None:
            return
        
        if not hasattr(self, 'signal_price_plot') or not hasattr(self, 'signal_zscore_plot'):
            return
        
        try:
            tickers = pair.split('/')
            if len(tickers) != 2:
                return
            
            y_ticker, x_ticker = tickers
            hedge_ratio = pair_stats['hedge_ratio']
            intercept = pair_stats.get('intercept', 0)

            # Align price data to spread's date range so both charts
            # use the same x-axis (consistent with OU Analytics tab)
            prices = self.engine.price_data[[y_ticker, x_ticker]].dropna()
            common_dates = prices.index.intersection(spread.index)
            if len(common_dates) == 0:
                return
            prices = prices.loc[common_dates]
            spread = spread.loc[common_dates]

            x_axis = np.arange(len(prices))
            dates = prices.index

            # Update date axes
            if hasattr(self, 'signal_price_date_axis'):
                self.signal_price_date_axis.set_dates(dates)
            if hasattr(self, 'signal_zscore_date_axis'):
                self.signal_zscore_date_axis.set_dates(dates)

            # ===== PRICE COMPARISON =====
            self.signal_price_plot.clear()
            self.signal_price_plot.addLegend()
            y_series = prices[y_ticker].values
            x_adjusted = (hedge_ratio * prices[x_ticker] + intercept).values
            self.signal_price_plot.plot(x_axis, y_series, pen=pg.mkPen('#d4a574', width=2), name=f"{display_ticker(y_ticker)} (Y)")
            self.signal_price_plot.plot(x_axis, x_adjusted, pen=pg.mkPen('#2196f3', width=2), name=f"β·{display_ticker(x_ticker)} + α")

            # ===== SPREAD with μ and ±2σ BANDS =====
            self.signal_zscore_plot.clear()

            spread_values = spread.values

            # Use Kalman time-varying μ and σ if available
            kalman = getattr(ou, '_kalman', None)
            if kalman is not None and kalman.mu_history is not None and kalman.eq_std_history is not None:
                mu_hist = kalman.mu_history
                std_hist = kalman.eq_std_history
                
                n_pad = len(x_axis) - len(mu_hist)
                if n_pad > 0:
                    mu_vals = np.concatenate([np.full(n_pad, np.nan), mu_hist])
                    std_vals = np.concatenate([np.full(n_pad, np.nan), std_hist])
                else:
                    mu_vals = mu_hist[-len(x_axis):]
                    std_vals = std_hist[-len(x_axis):]
                
                upper_2 = mu_vals + 2 * std_vals
                lower_2 = mu_vals - 2 * std_vals
                
                try:
                    upper_curve = self.signal_zscore_plot.plot(x_axis, upper_2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
                    lower_curve = self.signal_zscore_plot.plot(x_axis, lower_2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
                    fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(255, 255, 255, 20))
                    self.signal_zscore_plot.addItem(fill)
                except Exception:
                    pass
                
                self.signal_zscore_plot.plot(x_axis, mu_vals, pen=pg.mkPen('#ffffff', width=1))
            else:
                mu_val = ou.mu
                eq_std_val = ou.eq_std
                self.signal_zscore_plot.addLine(y=mu_val, pen=pg.mkPen('#ffffff', width=1))
                self.signal_zscore_plot.addLine(y=mu_val + 2*eq_std_val, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
                self.signal_zscore_plot.addLine(y=mu_val - 2*eq_std_val, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            
            self.signal_zscore_plot.plot(x_axis, spread_values, pen=pg.mkPen('#00e5ff', width=2))
            zscore = spread
            
            # ===== AUTO-RANGE: Reset view to show full data after plotting =====
            # Temporarily block sync signals to avoid recursive range triggering
            self._signal_syncing_plots = True
            self.signal_price_plot.enableAutoRange()
            self.signal_zscore_plot.enableAutoRange()
            # Use QTimer to unblock after the event loop processes the range changes
            QTimer.singleShot(50, lambda: setattr(self, '_signal_syncing_plots', False))
            
            # ===== SETUP SYNCHRONIZED CROSSHAIRS =====
            # Remove old crosshairs if they exist
            if hasattr(self, 'signal_price_crosshair') and self.signal_price_crosshair is not None:
                self.signal_price_crosshair.cleanup()
            
            if hasattr(self, 'signal_zscore_crosshair') and self.signal_zscore_crosshair is not None:
                self.signal_zscore_crosshair.cleanup()
            
            # Create new synchronized crosshairs
            CrosshairManager = get_crosshair_manager_class()
            if CrosshairManager:
                self.signal_price_crosshair = CrosshairManager(
                    self.signal_price_plot, 
                    dates=dates,
                    data_series={display_ticker(y_ticker): y_series, f"β·{display_ticker(x_ticker)}+α": x_adjusted},
                    label_format="{:.2f}"
                )
                
                self.signal_zscore_crosshair = CrosshairManager(
                    self.signal_zscore_plot,
                    dates=dates,
                    data_series={'Spread': zscore.values},
                    label_format="{:.2f}"
                )
                
                # Link crosshairs for synchronization
                self.signal_price_crosshair.add_synced_manager(self.signal_zscore_crosshair)
                self.signal_zscore_crosshair.add_synced_manager(self.signal_price_crosshair)
            
        except Exception as e:
            print(f"Signal plot update error: {e}")
    
    def update_mini_futures(self, pair: str, z: float):
        """Update mini-futures suggestions by scraping Morgan Stanley.

        Hämtar ALLA tillgängliga instrument (mini futures + certifikat) och
        populerar dropdown-menyer så att användaren kan välja instrument.
        """
        tickers = pair.split('/')
        if len(tickers) != 2:
            return

        y_ticker, x_ticker = tickers

        # Determine direction for each leg
        # LONG SPREAD = Long Y, Short X
        # SHORT SPREAD = Short Y, Long X
        if z < 0:  # Long spread
            dir_y, dir_x = 'Long', 'Short'
            direction = 'LONG'
        else:  # Short spread
            dir_y, dir_x = 'Short', 'Long'
            direction = 'SHORT'

        # Spara riktningar för _recalculate_mf_position_sizing
        self._current_dir_y = dir_y
        self._current_dir_x = dir_x
        self._current_direction = direction

        # VIKTIGT: Rensa BÅDA korten FÖRST innan vi hämtar nya produkter
        self._clear_mini_future_card('y', y_ticker, dir_y)
        self._clear_mini_future_card('x', x_ticker, dir_x)
        self._clear_mf_position_sizing()
        self.current_mini_futures = {'y': None, 'x': None}
        QApplication.processEvents()

        # Load ticker mapping
        ticker_to_ms, ms_to_ticker, ticker_to_ms_asset = load_ticker_mapping(force_reload=True)

        if not ticker_to_ms:
            self.mini_y_ticker.setText(f"{display_ticker(y_ticker)} - MAPPING UNAVAILABLE")
            self.mini_y_combo.blockSignals(True)
            self.mini_y_combo.clear()
            self.mini_y_combo.addItem("Ticker mapping file not found")
            self.mini_y_combo.blockSignals(False)
            self.mini_y_info.setText("")

            self.mini_x_ticker.setText(f"{display_ticker(x_ticker)} - MAPPING UNAVAILABLE")
            self.mini_x_combo.blockSignals(True)
            self.mini_x_combo.clear()
            self.mini_x_combo.addItem("Ticker mapping file not found")
            self.mini_x_combo.blockSignals(False)
            self.mini_x_info.setText("")
            self._clear_mf_position_sizing()
            return

        self.statusBar().showMessage("Fetching all instruments from Morgan Stanley...")
        QApplication.processEvents()

        # Hämta ALLA instrument för varje ben (alla typer)
        all_instruments_y = fetch_all_instruments_for_ticker(y_ticker, dir_y, ticker_to_ms, ticker_to_ms_asset)
        all_instruments_x = fetch_all_instruments_for_ticker(x_ticker, dir_x, ticker_to_ms, ticker_to_ms_asset)

        # Spara ofiltrerade listor + mapping för produkttyp-toggle
        self._all_instruments_y = all_instruments_y
        self._all_instruments_x = all_instruments_x
        self._last_ticker_to_ms = ticker_to_ms

        # Räkna antal per typ och uppdatera knappar per ben
        n_mini_y = sum(1 for i in all_instruments_y if i.get('product_type') != 'Certificate')
        n_cert_y = sum(1 for i in all_instruments_y if i.get('product_type') == 'Certificate')
        n_mini_x = sum(1 for i in all_instruments_x if i.get('product_type') != 'Certificate')
        n_cert_x = sum(1 for i in all_instruments_x if i.get('product_type') == 'Certificate')
        self.btn_mini_y.setText(f"Mini ({n_mini_y})")
        self.btn_cert_y.setText(f"Cert ({n_cert_y})")
        self.btn_mini_x.setText(f"Mini ({n_mini_x})")
        self.btn_cert_x.setText(f"Cert ({n_cert_x})")

        # Auto-välj produkttyp per ben: om inga mini futures finns, välj cert automatiskt
        if n_mini_y == 0 and n_cert_y > 0:
            self._active_product_type_y = 'cert'
            self.btn_mini_y.setStyleSheet(self._deriv_btn_style_inactive)
            self.btn_cert_y.setStyleSheet(self._deriv_btn_style_active)
        else:
            self._active_product_type_y = 'mini'
            self.btn_mini_y.setStyleSheet(self._deriv_btn_style_active)
            self.btn_cert_y.setStyleSheet(self._deriv_btn_style_inactive)

        if n_mini_x == 0 and n_cert_x > 0:
            self._active_product_type_x = 'cert'
            self.btn_mini_x.setStyleSheet(self._deriv_btn_style_inactive)
            self.btn_cert_x.setStyleSheet(self._deriv_btn_style_active)
        else:
            self._active_product_type_x = 'mini'
            self.btn_mini_x.setStyleSheet(self._deriv_btn_style_active)
            self.btn_cert_x.setStyleSheet(self._deriv_btn_style_inactive)

        # Filtrera per ben baserat på aktiv produkttyp
        target_y = 'Certificate' if self._active_product_type_y == 'cert' else 'Mini Future'
        target_x = 'Certificate' if self._active_product_type_x == 'cert' else 'Mini Future'
        instruments_y = [i for i in all_instruments_y if i.get('product_type', 'Mini Future') == target_y]
        instruments_x = [i for i in all_instruments_x if i.get('product_type', 'Mini Future') == target_x]

        self._instruments_y = instruments_y
        self._instruments_x = instruments_x

        # Populera Y-dropdown (utan position sizing-beräkning)
        self._populate_instrument_combo('y', y_ticker, dir_y, instruments_y, ticker_to_ms)

        # Populera X-dropdown (utan position sizing-beräkning)
        self._populate_instrument_combo('x', x_ticker, dir_x, instruments_x, ticker_to_ms)

        # Räkna om position sizing EFTER att båda benen är klara
        self._recalculate_mf_position_sizing()

        self.statusBar().showMessage(
            f"Loaded {len(all_instruments_y)} instruments for {display_ticker(y_ticker)}, "
            f"{len(all_instruments_x)} for {display_ticker(x_ticker)}"
        )
    
    def _update_mf_position_sizing(self, pair: str, y_ticker: str, x_ticker: str,
                                    dir_y: str, dir_x: str, mf_y: dict, mf_x: dict, direction: str):
        """Beräkna och visa mini futures position sizing med korrekt exponering."""
        try:
            # Hämta hedge ratio
            pair_stats = self.engine.viable_pairs[self.engine.viable_pairs['pair'] == pair].iloc[0]
            hedge_ratio = pair_stats['hedge_ratio']
            beta = abs(hedge_ratio)

            # Beräkna minimum-enheter via exponeringsbaserad funktion
            min_result = calculate_minifuture_minimum_units(hedge_ratio, mf_y, mf_x, dir_y, dir_x)

            if not min_result:
                self._clear_mf_position_sizing()
                return

            # Hämta priser och hävstång
            inst_price_y = min_result['price_y']
            inst_price_x = min_result['price_x']
            leverage_y = min_result['leverage_y']
            leverage_x = min_result['leverage_x']
            exp_per_unit_y = inst_price_y * leverage_y
            exp_per_unit_x = inst_price_x * leverage_x

            # Auto-uppdatera notional till minst minimum
            min_capital = math.ceil(min_result['total_capital'])
            self.notional_spin.blockSignals(True)
            self.notional_spin.setValue(max(min_capital, self.notional_spin.value()))
            self.notional_spin.blockSignals(False)

            notional = self.notional_spin.value()

            # Skala: hitta största X-enheter som ryms inom notional
            units_y = min_result['units_y']
            units_x = min_result['units_x']
            if notional > min_result['total_capital']:
                ux = units_x
                while True:
                    next_x = ux + 1
                    next_y = max(1, math.ceil(
                        next_x * exp_per_unit_x / (beta * exp_per_unit_y)
                    )) if beta > 0 else 1
                    if next_y * inst_price_y + next_x * inst_price_x > notional:
                        break
                    units_x = next_x
                    units_y = next_y
                    ux = next_x

            # Beräkna verkliga värden
            capital_y = units_y * inst_price_y
            capital_x = units_x * inst_price_x
            total_capital = capital_y + capital_x
            exposure_y = units_y * exp_per_unit_y
            exposure_x = units_x * exp_per_unit_x
            total_exposure = exposure_y + exposure_x
            effective_leverage = total_exposure / total_capital if total_capital > 0 else 1.0

            # Spara för export/annan användning
            self.current_mf_positions = {
                'capital_y': capital_y, 'capital_x': capital_x,
                'total_capital': total_capital,
                'exposure_y': exposure_y, 'exposure_x': exposure_x,
                'total_exposure': total_exposure,
                'leverage_y': leverage_y, 'leverage_x': leverage_x,
                'effective_leverage': effective_leverage,
                'units_y': units_y, 'units_x': units_x,
                'price_y': inst_price_y, 'price_x': inst_price_x,
            }

            # ── Y-ben ──
            action_y = "LONG POSITION:" if dir_y == "Long" else "SHORT POSITION:"
            color_y = "#00c853" if dir_y == "Long" else "#ff1744"

            if mf_y:
                self.mf_pos_y_action.setText(f"{action_y} {display_ticker(y_ticker)}")
                self.mf_pos_y_action.setStyleSheet(f"color: {color_y}; font-size: 11px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_y_capital.setText(f"{capital_y:,.0f} SEK")
                self.mf_pos_y_capital.setStyleSheet(f"color: {color_y}; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_y_info.setText(f"{units_y} units @ {inst_price_y:.2f} SEK")
                self.mf_pos_y_info.setStyleSheet("color: #888888; font-size: 10px; background: transparent; border: none;")
                self.mf_pos_y_exposure.setText(f"\u2192 {exposure_y:,.0f} SEK exposure ({leverage_y:.2f}x)")
            else:
                self.mf_pos_y_action.setText(f"{display_ticker(y_ticker)}")
                self.mf_pos_y_action.setStyleSheet("color: #888888; font-size: 11px; background: transparent; border: none;")
                self.mf_pos_y_capital.setText("N/A")
                self.mf_pos_y_capital.setStyleSheet("color: #555555; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_y_info.setText("No mini future available")
                self.mf_pos_y_exposure.setText("Use regular shares")

            # ── X-ben ──
            action_x = "LONG POSITION:" if dir_x == "Long" else "SHORT POSITION:"
            color_x = "#00c853" if dir_x == "Long" else "#ff1744"

            if mf_x:
                self.mf_pos_x_action.setText(f"{action_x} {display_ticker(x_ticker)}")
                self.mf_pos_x_action.setStyleSheet(f"color: {color_x}; font-size: 11px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_x_capital.setText(f"{capital_x:,.0f} SEK")
                self.mf_pos_x_capital.setStyleSheet(f"color: {color_x}; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_x_info.setText(f"{units_x} units @ {inst_price_x:.2f} SEK")
                self.mf_pos_x_info.setStyleSheet("color: #888888; font-size: 10px; background: transparent; border: none;")
                self.mf_pos_x_exposure.setText(f"\u2192 {exposure_x:,.0f} SEK exposure ({leverage_x:.2f}x)")
            else:
                self.mf_pos_x_action.setText(f"{display_ticker(x_ticker)}")
                self.mf_pos_x_action.setStyleSheet("color: #888888; font-size: 11px; background: transparent; border: none;")
                self.mf_pos_x_capital.setText("N/A")
                self.mf_pos_x_capital.setStyleSheet("color: #555555; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_x_info.setText("No mini future available")
                self.mf_pos_x_exposure.setText("Use regular shares")

            # ── Totalt ──
            self.mf_pos_total_capital.setText(f"{total_capital:,.0f} SEK")
            self.mf_pos_total_beta.setText(f"\u03b2 = {hedge_ratio:.4f}")
            self.mf_pos_total_exposure.setText(f"Total exposure: {total_exposure:,.0f} SEK")
            self.mf_pos_eff_leverage.setText(f"Effective leverage: {effective_leverage:.2f}x")

            # Verifiera att faktisk hedge ratio stämmer med target
            actual_hedge = (units_x * exp_per_unit_x) / (units_y * exp_per_unit_y) if (units_y * exp_per_unit_y) > 0 else 0
            hedge_dev = abs(actual_hedge - beta) / beta * 100 if beta > 0 else 0

            if hedge_dev > 10:
                info_text = f"\u26a0\ufe0f Hedge ratio: {actual_hedge:.4f} (target \u03b2={beta:.4f}, {hedge_dev:.1f}% avvikelse)"
            else:
                info_text = f"\u2713 Hedge ratio: {actual_hedge:.4f} (target \u03b2={beta:.4f})"

            self.mf_min_cap_label.setText(info_text)

        except Exception as e:
            print(f"Error calculating MF position sizing: {e}")
            traceback.print_exc()
            self._clear_mf_position_sizing()
    
    def _clear_mf_position_sizing(self):
        """Clear mini futures position sizing display."""
        self.mf_pos_y_action.setText("LEG Y")
        self.mf_pos_y_action.setStyleSheet("color: #888888; font-size: 11px; background: transparent;")
        self.mf_pos_y_capital.setText("-")
        self.mf_pos_y_capital.setStyleSheet("color: #555555; font-size: 20px; font-weight: 600; background: transparent;")
        self.mf_pos_y_info.setText("")
        self.mf_pos_y_exposure.setText("")
        
        self.mf_pos_x_action.setText("LEG X")
        self.mf_pos_x_action.setStyleSheet("color: #888888; font-size: 11px; background: transparent;")
        self.mf_pos_x_capital.setText("-")
        self.mf_pos_x_capital.setStyleSheet("color: #555555; font-size: 20px; font-weight: 600; background: transparent;")
        self.mf_pos_x_info.setText("")
        self.mf_pos_x_exposure.setText("")
        
        self.mf_pos_total_capital.setText("-")
        self.mf_pos_total_beta.setText("")
        self.mf_pos_total_exposure.setText("")
        self.mf_pos_eff_leverage.setText("")
        
        self.mf_min_cap_label.setText("ℹ️ Position sizing info will appear here")
    
    def _switch_product_type(self, ptype: str, leg: str):
        """Byt mellan 'mini' och 'cert' för ett specifikt ben (y/x)."""
        current = self._active_product_type_y if leg == 'y' else self._active_product_type_x
        if ptype == current:
            return

        if leg == 'y':
            self._active_product_type_y = ptype
            if ptype == 'mini':
                self.btn_mini_y.setStyleSheet(self._deriv_btn_style_active)
                self.btn_cert_y.setStyleSheet(self._deriv_btn_style_inactive)
            else:
                self.btn_mini_y.setStyleSheet(self._deriv_btn_style_inactive)
                self.btn_cert_y.setStyleSheet(self._deriv_btn_style_active)
        else:
            self._active_product_type_x = ptype
            if ptype == 'mini':
                self.btn_mini_x.setStyleSheet(self._deriv_btn_style_active)
                self.btn_cert_x.setStyleSheet(self._deriv_btn_style_inactive)
            else:
                self.btn_mini_x.setStyleSheet(self._deriv_btn_style_inactive)
                self.btn_cert_x.setStyleSheet(self._deriv_btn_style_active)

        self._repopulate_combo_for_leg(leg)

    def _repopulate_combo_for_leg(self, leg: str):
        """Filtrera instrument för ett ben baserat på dess aktiva produkttyp."""
        all_insts = getattr(self, f'_all_instruments_{leg}', None)
        if all_insts is None:
            return

        ptype = self._active_product_type_y if leg == 'y' else self._active_product_type_x
        target_type = 'Certificate' if ptype == 'cert' else 'Mini Future'
        filtered = [i for i in all_insts if i.get('product_type', 'Mini Future') == target_type]

        if leg == 'y':
            self._instruments_y = filtered
        else:
            self._instruments_x = filtered

        ticker = filtered[0]['ticker'] if filtered else (all_insts[0]['ticker'] if all_insts else '')
        direction = filtered[0]['direction'] if filtered else (all_insts[0]['direction'] if all_insts else 'Long')

        ticker_to_ms = getattr(self, '_last_ticker_to_ms', {})
        self._populate_instrument_combo(leg, ticker, direction, filtered, ticker_to_ms)

        self._recalculate_mf_position_sizing()

    def _populate_instrument_combo(self, leg: str, ticker: str, direction: str,
                                    instruments: list, ticker_to_ms: dict):
        """Populera instrument-dropdown för ett ben och auto-välj bästa instrument."""
        combo = self.mini_y_combo if leg == 'y' else self.mini_x_combo
        ticker_label = self.mini_y_ticker if leg == 'y' else self.mini_x_ticker
        info_label = self.mini_y_info if leg == 'y' else self.mini_x_info
        frame = self.mini_y_frame if leg == 'y' else self.mini_x_frame

        action = "BUY" if direction == "Long" else "SELL"
        color = "#00c853" if direction == "Long" else "#ff1744"
        dt = display_ticker(ticker)

        combo.blockSignals(True)
        combo.clear()

        if not instruments:
            active_ptype = self._active_product_type_y if leg == 'y' else self._active_product_type_x
            ptype_name = "certificates" if active_ptype == 'cert' else "mini futures"
            combo.addItem(f"No {ptype_name} found")
            ticker_label.setText(f"{dt} - NO {ptype_name.upper()} FOUND")
            ticker_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
            info_label.setText(f"Searched for: {ticker_to_ms.get(ticker, ticker)} ({direction})")
            self.current_mini_futures[leg] = None
            combo.blockSignals(False)
            return

        for inst in instruments:
            product_type = inst.get('product_type', 'Mini Future')
            if product_type == 'Certificate':
                label = f"{inst['name']} ({inst['leverage']:.0f}x)"
            else:
                label = f"{inst['name']} ({inst['leverage']:.2f}x)"
            combo.addItem(label)

        combo.setCurrentIndex(0)
        combo.blockSignals(False)

        best = instruments[0]
        product_type = best.get('product_type', 'Mini Future')
        if product_type == 'Certificate':
            ticker_label.setText(f"{action} {dt} CERTIFICATE {direction.upper()}")
        else:
            ticker_label.setText(f"{action} {dt} MINI {direction.upper()}")
        ticker_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600; background: transparent; border: none;")

        self.current_mini_futures[leg] = best
        self._update_instrument_info(best, leg)

        if best.get('avanza_link'):
            frame.setCursor(Qt.PointingHandCursor)
            frame.mousePressEvent = lambda e, url=best['avanza_link']: (
                QDesktopServices.openUrl(QUrl(url)) and None)

    def _on_instrument_changed(self, index: int, leg: str):
        """Hantera användarens val av ett annat instrument i dropdown."""
        try:
            instruments = self._instruments_y if leg == 'y' else self._instruments_x
            if not instruments or index < 0 or index >= len(instruments):
                return

            selected = instruments[index]
            self.current_mini_futures[leg] = selected
            self._update_instrument_info(selected, leg)

            # Uppdatera Avanza-klicklänk
            frame = self.mini_y_frame if leg == 'y' else self.mini_x_frame
            if selected.get('avanza_link'):
                frame.setCursor(Qt.PointingHandCursor)
                # OBS: lambda får INTE returnera värde — mousePressEvent är void
                frame.mousePressEvent = lambda e, url=selected['avanza_link']: (
                    QDesktopServices.openUrl(QUrl(url)) and None)

            # Uppdatera ticker-label med produkttyp
            ticker_label = self.mini_y_ticker if leg == 'y' else self.mini_x_ticker
            action = "BUY" if selected['direction'] == "Long" else "SELL"
            color = "#00c853" if selected['direction'] == "Long" else "#ff1744"
            product_type = selected.get('product_type', 'Mini Future')
            dt = display_ticker(selected['ticker'])
            if product_type == 'Certificate':
                ticker_label.setText(f"{action} {dt} CERTIFICATE {selected['direction'].upper()}")
            else:
                ticker_label.setText(f"{action} {dt} MINI {selected['direction'].upper()}")
            ticker_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600; background: transparent; border: none;")

            # Räkna om position sizing
            self._recalculate_mf_position_sizing()
        except Exception as e:
            print(f"[Instrument] Selection error ({leg}): {e}")

    def _update_instrument_info(self, mf_data: dict, leg: str):
        """Uppdatera info-label med detaljer om valt instrument."""
        info_label = self.mini_y_info if leg == 'y' else self.mini_x_info

        product_type = mf_data.get('product_type', 'Mini Future')
        is_certificate = product_type == 'Certificate'

        info_text = f"Underlying: {mf_data['underlying']}\n"

        if is_certificate:
            info_text += f"Product Type: Bull/Bear Certificate\n"
            if mf_data.get('spot_price'):
                info_text += f"Spot Price: {mf_data['spot_price']:,.2f}\n"
            daily_lev = mf_data.get('daily_leverage', mf_data.get('leverage', 0))
            info_text += f"Daily Leverage: {daily_lev:.0f}x"
            if mf_data.get('instrument_price'):
                info_text += f"\nInstrument Price: {mf_data['instrument_price']:.2f} SEK"
        else:
            if mf_data.get('financing_level'):
                info_text += f"Financing Level: {mf_data['financing_level']:,.2f}\n"
            if mf_data.get('spot_price'):
                info_text += f"Spot Price: {mf_data['spot_price']:,.2f}\n"
            info_text += f"Leverage: {mf_data['leverage']:.2f}x"

        if mf_data.get('isin') and mf_data['isin'] != 'N/A':
            info_text += f"\nISIN: {mf_data['isin']}"

        info_label.setText(info_text)

    def _recalculate_mf_position_sizing(self):
        """Räkna om position sizing med aktuellt valda instrument."""
        if not hasattr(self, 'signal_selected_pair') or self.signal_selected_pair is None:
            return
        if not self.current_mini_futures:
            return

        pair = self.signal_selected_pair
        tickers = pair.split('/')
        if len(tickers) != 2:
            return

        y_ticker, x_ticker = tickers
        dir_y = getattr(self, '_current_dir_y', 'Long')
        dir_x = getattr(self, '_current_dir_x', 'Short')
        direction = getattr(self, '_current_direction', 'LONG')

        mf_y = self.current_mini_futures.get('y')
        mf_x = self.current_mini_futures.get('x')

        self._update_mf_position_sizing(pair, y_ticker, x_ticker, dir_y, dir_x, mf_y, mf_x, direction)

    def _update_mini_future_card(self, ticker: str, direction: str, mf_data: dict, ticker_to_ms: dict, leg: str):
        """Update instrument info label (combo is populated separately)."""
        if mf_data:
            self._update_instrument_info(mf_data, leg)
        else:
            self._update_mini_future_card_not_found(ticker, direction, ticker_to_ms, leg)
    
    def _clear_mini_future_card(self, leg: str, ticker: str, direction: str):
        """Clear a mini future card to 'Searching...' state before fetching new data."""
        action = "BUY" if direction == "Long" else "SELL"
        dt = display_ticker(ticker)

        if leg == 'y':
            self.mini_y_ticker.setText(f"{action} {dt} MINI {direction.upper()}")
            self.mini_y_ticker.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
            self.mini_y_combo.blockSignals(True)
            self.mini_y_combo.clear()
            self.mini_y_combo.addItem("Searching Morgan Stanley...")
            self.mini_y_combo.blockSignals(False)
            self.mini_y_info.setText("")
            self._instruments_y = []
            self.mini_y_frame.setCursor(Qt.ArrowCursor)
            self.mini_y_frame.mousePressEvent = lambda e: None
        else:
            self.mini_x_ticker.setText(f"{action} {dt} MINI {direction.upper()}")
            self.mini_x_ticker.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
            self.mini_x_combo.blockSignals(True)
            self.mini_x_combo.clear()
            self.mini_x_combo.addItem("Searching Morgan Stanley...")
            self.mini_x_combo.blockSignals(False)
            self.mini_x_info.setText("")
            self._instruments_x = []
            self.mini_x_frame.setCursor(Qt.ArrowCursor)
            self.mini_x_frame.mousePressEvent = lambda e: None
    
    def _update_mini_future_card_not_found(self, ticker: str, direction: str, ticker_to_ms: dict, leg: str):
        """Update mini future card when not found."""
        ms_name = ticker_to_ms.get(ticker, ticker)
        dt = display_ticker(ticker)

        if leg == 'y':
            self.mini_y_ticker.setText(f"{dt} - NO PRODUCT FOUND")
            self.mini_y_ticker.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {TYPOGRAPHY['body_small']}px; background: transparent; border: none")
            self.mini_y_combo.blockSignals(True)
            self.mini_y_combo.clear()
            self.mini_y_combo.addItem(f"No instruments for {dt}")
            self.mini_y_combo.blockSignals(False)
            self.mini_y_info.setText(f"Searched for: {ms_name} ({direction})")
        else:
            self.mini_x_ticker.setText(f"{dt} - NO PRODUCT FOUND")
            self.mini_x_ticker.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {TYPOGRAPHY['body_small']}px; background: transparent; border: none")
            self.mini_x_combo.blockSignals(True)
            self.mini_x_combo.clear()
            self.mini_x_combo.addItem(f"No instruments for {dt}")
            self.mini_x_combo.blockSignals(False)
            self.mini_x_info.setText(f"Searched for: {ms_name} ({direction})")
    
    def update_position_sizing(self):
        """Update position sizing display."""
        if self.signal_selected_pair is None or self.engine is None:
            return
        
        try:
            pair = self.signal_selected_pair
            tickers = pair.split('/')
            if len(tickers) != 2:
                return
            
            y_ticker, x_ticker = tickers
            
            # Try to get prices - handle missing tickers (e.g. indices like ^DJI)
            try:
                y_price = self.engine.price_data[y_ticker].iloc[-1]
            except KeyError:
                y_price = None
                
            try:
                x_price = self.engine.price_data[x_ticker].iloc[-1]
            except KeyError:
                x_price = None
            
            pair_stats = self.engine.viable_pairs[self.engine.viable_pairs['pair'] == pair].iloc[0]
            hedge_ratio = pair_stats['hedge_ratio']
            
            ou, spread, z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
            
            # Always update labels with correct tickers first
            if z < 0:  # Long spread: buy Y, sell X
                self.buy_action_label.setText(f"BUY {display_ticker(y_ticker)}")
                self.sell_action_label.setText(f"SELL {display_ticker(x_ticker)}")
                dir_y, dir_x = 'Long', 'Short'
                direction = 'LONG'
            else:  # Short spread: sell Y, buy X
                self.sell_action_label.setText(f"SELL {display_ticker(y_ticker)}")
                self.buy_action_label.setText(f"BUY {display_ticker(x_ticker)}")
                dir_y, dir_x = 'Short', 'Long'
                direction = 'SHORT'
            
            # If price data is missing, show N/A for shares but keep correct tickers
            if y_price is None or x_price is None:
                self.buy_shares_label.setText("N/A (index)")
                self.buy_price_label.setText("")
                self.sell_shares_label.setText("N/A (index)")
                self.sell_price_label.setText("")
                self.capital_label.setText("N/A")
                self.capital_beta_label.setText(f"β = {hedge_ratio:.3f}")
                
                # Still update mini futures if available
                if self.current_mini_futures:
                    mf_y = self.current_mini_futures.get('y')
                    mf_x = self.current_mini_futures.get('x')
                    self._update_mf_position_sizing(pair, y_ticker, x_ticker, dir_y, dir_x, mf_y, mf_x, direction)
                return
            
            notional = self.notional_spin.value()
            lev_y = self.leverage_y_spin.value()
            lev_x = self.leverage_x_spin.value()
            
            y_shares = max(1, int((notional * lev_y) / (y_price + hedge_ratio * x_price)))
            x_shares = max(1, int(y_shares * hedge_ratio * lev_x))
            
            if z < 0:  # Long spread: buy Y, sell X
                self.buy_shares_label.setText(f"{y_shares} shares")
                self.buy_price_label.setText(f"@ {y_price:.2f}")
                self.sell_shares_label.setText(f"{x_shares} shares")
                self.sell_price_label.setText(f"@ {x_price:.2f}")
            else:  # Short spread: sell Y, buy X
                self.sell_shares_label.setText(f"{y_shares} shares")
                self.sell_price_label.setText(f"@ {y_price:.2f}")
                self.buy_shares_label.setText(f"{x_shares} shares")
                self.buy_price_label.setText(f"@ {x_price:.2f}")
            
            capital = y_shares * y_price + x_shares * x_price
            self.capital_label.setText(f"{capital:,.0f} SEK")
            self.capital_beta_label.setText(f"β = {hedge_ratio:.3f}")
            
            # Also update mini futures position sizing if we have mini futures data
            if self.current_mini_futures:
                mf_y = self.current_mini_futures.get('y')
                mf_x = self.current_mini_futures.get('x')
                self._update_mf_position_sizing(pair, y_ticker, x_ticker, dir_y, dir_x, mf_y, mf_x, direction)
            
        except Exception as e:
            print(f"Position sizing error: {e}")
    
    def calculate_strategy(self, ou, z: float):
        """Calculate optimal stop-loss for current signal."""
        try:
            entry_z_abs = abs(z)
            self.current_strategy = calculate_optimal_stop_loss(ou, entry_z_abs, z)
        except Exception as e:
            print(f"Strategy calculation error: {e}")
            self.current_strategy = {
                'stop_z': 3.0,
                'exit_z': 0.0,
                'rr': 0,
                'exp_pnl_pct': 0,
                'win_prob': 0,
                'kelly': 0
            }
    
    def open_position(self, strategy_type: str = "Balanced"):
        """Open a new position."""
        if self.signal_selected_pair is None or self.engine is None:
            return
        
        try:
            pair = self.signal_selected_pair
            ou, spread, z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
            pair_stats = self.engine.viable_pairs[self.engine.viable_pairs['pair'] == pair].iloc[0]
            
            # Calculate optimal strategy
            self.calculate_strategy(ou, z)
            strategy = self.current_strategy
            
            # Get position sizing
            tickers = pair.split('/')
            y_ticker, x_ticker = tickers
            y_price = self.engine.price_data[y_ticker].iloc[-1]
            x_price = self.engine.price_data[x_ticker].iloc[-1]
            hedge_ratio = pair_stats['hedge_ratio']
            
            notional = self.notional_spin.value()
            lev_y = self.leverage_y_spin.value()
            lev_x = self.leverage_x_spin.value()
            
            y_shares = max(1, int((notional * lev_y) / (y_price + hedge_ratio * x_price)))
            x_shares = max(1, int(y_shares * hedge_ratio * lev_x))
            
            position = {
                'pair': pair,
                'direction': 'LONG' if z < 0 else 'SHORT',
                'entry_z': z,
                'current_z': z,
                'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'hedge_ratio': hedge_ratio,
                'notional': notional,
                'shares_y': y_shares,
                'shares_x': x_shares,
                'leverage_y': lev_y,
                'leverage_x': lev_x,
                'exit_z': 0.0,  # TP is always 0 (mean reversion)
                'stop_z': strategy.get('stop_z', 3.0),
                'win_prob': strategy.get('win_prob', 0),
                'rr': strategy.get('rr', 0),
                'exp_pnl_pct': strategy.get('exp_pnl_pct', 0),
                'kelly': strategy.get('kelly', 0),
                'status': 'OPEN',
                # Mini futures info
                'mini_y_name': self.current_mini_futures.get('y', {}).get('name') if self.current_mini_futures.get('y') else None,
                'mini_y_isin': self.current_mini_futures.get('y', {}).get('isin') if self.current_mini_futures.get('y') else None,
                'mini_x_name': self.current_mini_futures.get('x', {}).get('name') if self.current_mini_futures.get('x') else None,
                'mini_x_isin': self.current_mini_futures.get('x', {}).get('isin') if self.current_mini_futures.get('x') else None,
                'mf_capital_y': self.current_mf_positions.get('capital_y') if self.current_mf_positions else None,
                'mf_capital_x': self.current_mf_positions.get('capital_x') if self.current_mf_positions else None,
                'mf_exposure_y': self.current_mf_positions.get('exposure_y') if self.current_mf_positions else None,
                'mf_exposure_x': self.current_mf_positions.get('exposure_x') if self.current_mf_positions else None,
                'mf_total_capital': self.current_mf_positions.get('total_capital') if self.current_mf_positions else None,
                'mf_effective_leverage': self.current_mf_positions.get('effective_leverage') if self.current_mf_positions else None,
                'window_size': min(pair_stats['passing_windows']) if pair_stats['passing_windows'] else None,
            }
            
            self.portfolio.append(position)
            self.update_portfolio_display()
            self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
            
            # Auto-save portfolio
            self._save_and_sync_portfolio()
            
            self.statusBar().showMessage(f"Opened position for {pair} | Win: {strategy.get('win_prob', 0)*100:.0f}%")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open position:\n{e}")
            traceback.print_exc()
    
    def update_portfolio_display(self):
        """Update portfolio positions table."""
        # OPTIMERING: Guard för lazy loading - hoppa över om Portfolio-tab inte är laddad
        if not hasattr(self, 'positions_table') or self.positions_table is None:
            return

        # Batch-uppdatering: stäng av repaints under modifiering
        self.positions_table.setUpdatesEnabled(False)
        try:
            self._update_portfolio_display_inner()
        finally:
            self.positions_table.setUpdatesEnabled(True)

        # Clean up stale pair monitors and init new ones
        self._lm_cleanup_stale_pairs()

    def _update_portfolio_display_inner(self):
        """Inner portfolio display update (called with setUpdatesEnabled=False)."""
        self.positions_table.setRowCount(len(self.portfolio))

        # Update summary labels
        open_count = len(self.portfolio)
        closed_count = len(self.trade_history)

        self.open_pos_label.setText(f"Open: <span style='color:#22c55e; font-weight:600;'>{open_count}</span>")
        self.closed_pos_label.setText(f"Closed: <span style='color:#888;'>{closed_count}</span>")

        # Calculate total P/L
        total_pnl = 0.0
        total_invested = 0.0

        for i, pos in enumerate(self.portfolio):
            # PAIR (col 0)
            pair_item = QTableWidgetItem(pos['pair'])
            pair_item.setForeground(_QCOLOR_TEXT)
            pair_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 0, pair_item)

            # DIRECTION (col 1)
            dir_item = QTableWidgetItem(pos['direction'])
            dir_item.setForeground(_QCOLOR_TEXT)
            dir_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 1, dir_item)

            # CURRENT Z (col 2)
            current_z = pos.get('current_z', pos['entry_z'])
            z_item = QTableWidgetItem(f"{current_z:.2f}")
            z_item.setForeground(_QCOLOR_TEXT)
            z_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 2, z_item)

            # STATUS (col 3) - Purely dynamic: SELL when z crosses zero toward profit
            direction = pos['direction']

            if direction == 'LONG' and current_z > 0:
                status_text = 'SELL'
                status_qcolor = _QCOLOR_AMBER
            elif direction == 'SHORT' and current_z < 0:
                status_text = 'SELL'
                status_qcolor = _QCOLOR_AMBER
            else:
                status_text = 'OPEN'
                status_qcolor = _QCOLOR_TEXT

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(status_qcolor)
            status_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 3, status_item)

            # MINI Y (col 4) - Show NAME (ISIN in tooltip)
            mini_y_isin = pos.get('mini_y_isin', '')
            mini_y_name = pos.get('mini_y_name', '')
            if mini_y_name:
                mini_y_item = QTableWidgetItem(mini_y_name)
                mini_y_item.setToolTip(f"{mini_y_name}\nISIN: {mini_y_isin}")
                mini_y_item.setForeground(_QCOLOR_TEXT)
            elif mini_y_isin:
                mini_y_item = QTableWidgetItem(mini_y_isin[:12])
                mini_y_item.setToolTip(f"ISIN: {mini_y_isin}")
                mini_y_item.setForeground(_QCOLOR_TEXT)
            else:
                mini_y_item = QTableWidgetItem("-")
                mini_y_item.setForeground(_QCOLOR_MUTED)
            mini_y_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 4, mini_y_item)

            # ENTRY Y (col 5) - Editable entry price
            entry_y_widget = QDoubleSpinBox()
            entry_y_widget.setRange(0, 1000000)
            entry_y_widget.setDecimals(2)
            entry_y_widget.setSingleStep(1.0)
            entry_y_widget.setValue(pos.get('mf_entry_price_y', 0.0))
            entry_y_widget.setFixedHeight(36)
            entry_y_widget.setAlignment(Qt.AlignCenter)
            entry_y_widget.setStyleSheet(_DSPINBOX_STYLESHEET)
            entry_y_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_entry_price(idx, 'y', val))
            self.positions_table.setCellWidget(i, 5, entry_y_widget)

            # QTY Y (col 6) - Editable quantity
            qty_y_widget = QSpinBox()
            qty_y_widget.setRange(0, 100000)
            qty_y_widget.setValue(pos.get('mf_qty_y', 0))
            qty_y_widget.setFixedHeight(36)
            qty_y_widget.setAlignment(Qt.AlignCenter)
            qty_y_widget.setStyleSheet(_SPINBOX_STYLESHEET)
            qty_y_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_qty(idx, 'y', val))
            self.positions_table.setCellWidget(i, 6, qty_y_widget)

            # P/L Y (col 7) - Calculated P/L for Y leg
            entry_price_y = pos.get('mf_entry_price_y', 0.0)
            current_price_y = pos.get('mf_current_price_y', 0.0)
            qty_y = pos.get('mf_qty_y', 0)

            if entry_price_y > 0 and qty_y > 0:
                pnl_y = (current_price_y - entry_price_y) * qty_y
                pnl_y_pct = ((current_price_y / entry_price_y) - 1) * 100 if entry_price_y > 0 else 0
                pnl_y_text = f"{pnl_y:+,.0f} ({pnl_y_pct:+.1f}%)"
                pnl_y_qcolor = _QCOLOR_POSITIVE if pnl_y >= 0 else _QCOLOR_NEGATIVE
                total_pnl += pnl_y
                total_invested += entry_price_y * qty_y
            else:
                pnl_y_text = "-"
                pnl_y_qcolor = _QCOLOR_MUTED

            pnl_y_item = QTableWidgetItem(pnl_y_text)
            pnl_y_item.setForeground(pnl_y_qcolor)
            pnl_y_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 7, pnl_y_item)

            # MINI X (col 8) - Show NAME (ISIN in tooltip)
            mini_x_isin = pos.get('mini_x_isin', '')
            mini_x_name = pos.get('mini_x_name', '')
            if mini_x_name:
                mini_x_item = QTableWidgetItem(mini_x_name)
                mini_x_item.setToolTip(f"{mini_x_name}\nISIN: {mini_x_isin}")
                mini_x_item.setForeground(_QCOLOR_TEXT)
            elif mini_x_isin:
                mini_x_item = QTableWidgetItem(mini_x_isin[:12])
                mini_x_item.setToolTip(f"ISIN: {mini_x_isin}")
                mini_x_item.setForeground(_QCOLOR_TEXT)
            else:
                mini_x_item = QTableWidgetItem("-")
                mini_x_item.setForeground(_QCOLOR_MUTED)
            mini_x_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 8, mini_x_item)

            # ENTRY X (col 9) - Editable entry price
            entry_x_widget = QDoubleSpinBox()
            entry_x_widget.setRange(0, 1000000)
            entry_x_widget.setDecimals(2)
            entry_x_widget.setSingleStep(1.0)
            entry_x_widget.setValue(pos.get('mf_entry_price_x', 0.0))
            entry_x_widget.setFixedHeight(36)
            entry_x_widget.setAlignment(Qt.AlignCenter)
            entry_x_widget.setStyleSheet(_DSPINBOX_STYLESHEET)
            entry_x_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_entry_price(idx, 'x', val))
            self.positions_table.setCellWidget(i, 9, entry_x_widget)

            # QTY X (col 10) - Editable quantity
            qty_x_widget = QSpinBox()
            qty_x_widget.setRange(0, 100000)
            qty_x_widget.setValue(pos.get('mf_qty_x', 0))
            qty_x_widget.setFixedHeight(36)
            qty_x_widget.setAlignment(Qt.AlignCenter)
            qty_x_widget.setStyleSheet(_SPINBOX_STYLESHEET)
            qty_x_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_qty(idx, 'x', val))
            self.positions_table.setCellWidget(i, 10, qty_x_widget)

            # P/L X (col 11) - Calculated P/L for X leg
            entry_price_x = pos.get('mf_entry_price_x', 0.0)
            current_price_x = pos.get('mf_current_price_x', 0.0)
            qty_x = pos.get('mf_qty_x', 0)

            if entry_price_x > 0 and qty_x > 0:
                pnl_x = (current_price_x - entry_price_x) * qty_x
                pnl_x_pct = ((current_price_x / entry_price_x) - 1) * 100 if entry_price_x > 0 else 0
                pnl_x_text = f"{pnl_x:+,.0f} ({pnl_x_pct:+.1f}%)"
                pnl_x_qcolor = _QCOLOR_POSITIVE if pnl_x >= 0 else _QCOLOR_NEGATIVE
                total_pnl += pnl_x
                total_invested += entry_price_x * qty_x
            else:
                pnl_x_text = "-"
                pnl_x_qcolor = _QCOLOR_MUTED

            pnl_x_item = QTableWidgetItem(pnl_x_text)
            pnl_x_item.setForeground(pnl_x_qcolor)
            pnl_x_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 11, pnl_x_item)

            # CLOSE button (col 12)
            close_btn = QPushButton("✕")
            close_btn.setStyleSheet("""
                QPushButton {
                    background: #d32f2f;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 4px 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #ff1744;
                }
            """)
            close_btn.clicked.connect(lambda checked, idx=i: self.close_position(idx))
            self.positions_table.setCellWidget(i, 12, close_btn)

            # Z-CHART (col 13) — z-score sparkline from historical data
            pair = pos['pair']
            entry_z_val = pos.get('entry_z', pos.get('current_z', 0))
            self._lm_update_chart(pair, entry_z=entry_z_val,
                                  entry_date=pos.get('entry_date'),
                                  window_size=pos.get('window_size'))
            pd_data = self._lm_pairs.get(pair)
            if pd_data and pd_data.get('plot_widget'):
                self.positions_table.setCellWidget(i, 13, pd_data['plot_widget'])

        # Update total P/L summary
        if total_invested > 0:
            total_pnl_pct = (total_pnl / total_invested) * 100
            pnl_color = "#22c55e" if total_pnl >= 0 else "#ff1744"
            self.unrealized_pnl_label.setText(f"Unrealized: <span style='color:{pnl_color};'>{total_pnl:+,.0f} SEK ({total_pnl_pct:+.2f}%)</span>")
        else:
            self.unrealized_pnl_label.setText("Unrealized: <span style='color:#888;'>+0.00%</span>")
        
        # Update realized P/L from trade history
        total_realized = sum(t.get('realized_pnl_sek', 0) for t in self.trade_history)
        if total_realized != 0:
            r_color = "#22c55e" if total_realized >= 0 else "#ff1744"
            self.realized_pnl_label.setText(f"Realized: <span style='color:{r_color};'>{total_realized:+,.0f} SEK</span>")
        else:
            self.realized_pnl_label.setText("Realized: <span style='color:#888;'>+0.00%</span>")
        
        # Update open/closed counts
        n_open = len(self.portfolio)
        n_closed = len(self.trade_history)
        self.open_pos_label.setText(f"Open: <span style='color:#22c55e; font-weight:600;'>{n_open}</span>")
        self.closed_pos_label.setText(f"Closed: <span style='color:#888;'>{n_closed}</span>")
    
    def _update_mf_entry_price(self, idx: int, leg: str, value: float):
        """Update mini future entry price for a position."""
        if 0 <= idx < len(self.portfolio):
            key = f'mf_entry_price_{leg}'
            self.portfolio[idx][key] = value
            self._save_and_sync_portfolio()
            # Recalculate P/L for this row
            self._recalculate_row_pnl(idx)
    
    def _update_mf_qty(self, idx: int, leg: str, value: int):
        """Update mini future quantity for a position."""
        if 0 <= idx < len(self.portfolio):
            key = f'mf_qty_{leg}'
            self.portfolio[idx][key] = value
            self._save_and_sync_portfolio()
            # Recalculate P/L for this row
            self._recalculate_row_pnl(idx)
    
    def _recalculate_row_pnl(self, idx: int):
        """Recalculate P/L for a specific row without full table refresh."""
        if idx < 0 or idx >= len(self.portfolio):
            return

        pos = self.portfolio[idx]

        # P/L Y
        entry_price_y = pos.get('mf_entry_price_y', 0.0)
        current_price_y = pos.get('mf_current_price_y', 0.0)
        qty_y = pos.get('mf_qty_y', 0)

        if entry_price_y > 0 and qty_y > 0 and current_price_y > 0:
            pnl_y = (current_price_y - entry_price_y) * qty_y
            pnl_y_pct = ((current_price_y / entry_price_y) - 1) * 100
            pnl_y_text = f"{pnl_y:+,.0f} ({pnl_y_pct:+.1f}%)"
            pnl_y_qcolor = _QCOLOR_POSITIVE if pnl_y >= 0 else _QCOLOR_NEGATIVE
        else:
            pnl_y_text = "-"
            pnl_y_qcolor = _QCOLOR_MUTED

        pnl_y_item = QTableWidgetItem(pnl_y_text)
        pnl_y_item.setForeground(pnl_y_qcolor)
        self.positions_table.setItem(idx, 7, pnl_y_item)

        # P/L X
        entry_price_x = pos.get('mf_entry_price_x', 0.0)
        current_price_x = pos.get('mf_current_price_x', 0.0)
        qty_x = pos.get('mf_qty_x', 0)

        if entry_price_x > 0 and qty_x > 0 and current_price_x > 0:
            pnl_x = (current_price_x - entry_price_x) * qty_x
            pnl_x_pct = ((current_price_x / entry_price_x) - 1) * 100
            pnl_x_text = f"{pnl_x:+,.0f} ({pnl_x_pct:+.1f}%)"
            pnl_x_qcolor = _QCOLOR_POSITIVE if pnl_x >= 0 else _QCOLOR_NEGATIVE
        else:
            pnl_x_text = "-"
            pnl_x_qcolor = _QCOLOR_MUTED

        pnl_x_item = QTableWidgetItem(pnl_x_text)
        pnl_x_item.setForeground(pnl_x_qcolor)
        self.positions_table.setItem(idx, 11, pnl_x_item)

        # Update summary
        self._update_total_pnl_summary()
    
    def _update_total_pnl_summary(self):
        """Update total P/L in summary bar."""
        total_pnl = 0.0
        total_invested = 0.0
        
        for pos in self.portfolio:
            entry_y = pos.get('mf_entry_price_y', 0.0)
            current_y = pos.get('mf_current_price_y', 0.0)
            qty_y = pos.get('mf_qty_y', 0)
            
            if entry_y > 0 and qty_y > 0 and current_y > 0:
                total_pnl += (current_y - entry_y) * qty_y
                total_invested += entry_y * qty_y
            
            entry_x = pos.get('mf_entry_price_x', 0.0)
            current_x = pos.get('mf_current_price_x', 0.0)
            qty_x = pos.get('mf_qty_x', 0)
            
            if entry_x > 0 and qty_x > 0 and current_x > 0:
                total_pnl += (current_x - entry_x) * qty_x
                total_invested += entry_x * qty_x
        
        if total_invested > 0:
            total_pnl_pct = (total_pnl / total_invested) * 100
            pnl_color = "#22c55e" if total_pnl >= 0 else "#ff1744"
            self.unrealized_pnl_label.setText(f"Unrealized: <span style='color:{pnl_color};'>{total_pnl:+,.0f} SEK ({total_pnl_pct:+.2f}%)</span>")
        else:
            self.unrealized_pnl_label.setText("Unrealized: <span style='color:#888;'>+0.00%</span>")
    
    def refresh_mf_prices(self):
        """Refresh current mini futures prices from Morgan Stanley ASYNCHRONOUSLY."""
        if not MF_PRICE_SCRAPING_AVAILABLE:
            QMessageBox.warning(self, "Not Available", 
                "Mini futures price scraping is not available.\n"
                "Make sure scrape_prices_MS.py is in the same directory.")
            return
        
        if not self.portfolio:
            self.statusBar().showMessage("No positions to update")
            return
        
        self.statusBar().showMessage("Fetching mini futures prices...")
        
        # Collect all ISINs and store mapping for callback
        isins_to_fetch = []
        self._mf_isin_to_positions = {}  # Store for callback use
        
        for idx, pos in enumerate(self.portfolio):
            mini_y_isin = pos.get('mini_y_isin')
            mini_x_isin = pos.get('mini_x_isin')
            
            if mini_y_isin:
                if mini_y_isin not in self._mf_isin_to_positions:
                    self._mf_isin_to_positions[mini_y_isin] = []
                    isins_to_fetch.append(mini_y_isin)
                self._mf_isin_to_positions[mini_y_isin].append((idx, 'y'))
            
            if mini_x_isin:
                if mini_x_isin not in self._mf_isin_to_positions:
                    self._mf_isin_to_positions[mini_x_isin] = []
                    isins_to_fetch.append(mini_x_isin)
                self._mf_isin_to_positions[mini_x_isin].append((idx, 'x'))
        
        if not isins_to_fetch:
            self.statusBar().showMessage("No mini futures ISINs to update")
            return
        
        # Run in background thread to prevent GUI freeze
        self._price_thread = QThread()
        self._price_worker = PriceFetchWorker(isins_to_fetch)
        self._price_worker.moveToThread(self._price_thread)
        
        self._price_thread.started.connect(self._price_worker.run)
        self._price_worker.finished.connect(self._price_thread.quit)
        self._price_worker.finished.connect(self._price_worker.deleteLater)
        self._price_thread.finished.connect(self._price_thread.deleteLater)
        self._price_worker.result.connect(self._on_mf_prices_received)
        self._price_worker.error.connect(self._on_mf_prices_error)
        self._price_worker.status_message.connect(self.statusBar().showMessage)
        
        self._price_thread.start()
    
    def _on_mf_prices_received(self, quotes: dict):
        """Handle fetched mini futures prices - runs on GUI thread (safe)."""
        try:
            updated_count = 0
            for isin, quote in quotes.items():
                if isin in self._mf_isin_to_positions:
                    for pos_idx, leg in self._mf_isin_to_positions[isin]:
                        if pos_idx < len(self.portfolio) and quote.buy_price is not None:
                            key = f'mf_current_price_{leg}'
                            self.portfolio[pos_idx][key] = quote.buy_price
                            updated_count += 1
            
            # Save and refresh display
            self._save_and_sync_portfolio()
            self.update_portfolio_display()
            
            self.statusBar().showMessage(f"Updated {updated_count} mini futures prices from {len(quotes)} instruments")
            
        except Exception as e:
            print(f"Error processing MF prices: {e}")
            self.statusBar().showMessage(f"Price processing error: {e}")
    
    def _on_mf_prices_error(self, error: str):
        """Handle mini futures price fetch error."""
        print(f"MF price fetch error: {error}")
        self.statusBar().showMessage(f"Price fetch error: {error}")
    
    def refresh_portfolio_zscores(self):
        """Refresh current Z-scores for all open positions."""
        if not self.portfolio or self.engine is None:
            self.statusBar().showMessage("No positions to update or engine not initialized")
            return
        
        updated_count = 0
        for pos in self.portfolio:
            try:
                pair = pos['pair']
                ou, spread, current_z = self.engine.get_pair_ou_params(
                    pair, use_raw_data=True,
                    window_size=pos.get('window_size'))

                # Store previous z for comparison
                pos['previous_z'] = pos.get('current_z', pos['entry_z'])
                pos['current_z'] = current_z
                
                updated_count += 1
                
            except Exception as e:
                print(f"Error updating Z for {pos['pair']}: {e}")
        
        self.update_portfolio_display()
        
        # Auto-save portfolio with updated Z-scores
        self._save_and_sync_portfolio()
        
        self.statusBar().showMessage(f"Updated Z-scores for {updated_count} positions")
    
    def close_position(self, index: int):
        """Close a position with P&L calculation and move to trade history."""
        if 0 <= index < len(self.portfolio):
            pos = self.portfolio[index]
            
            entry_z = pos['entry_z']
            current_z = pos.get('current_z', entry_z)
            z_change = current_z - entry_z
            direction = pos['direction']

            # Calculate realized P/L from MF prices
            pnl_y = 0
            pnl_x = 0

            entry_y = pos.get('mf_entry_price_y')
            current_y = pos.get('mf_current_price_y')
            qty_y = pos.get('mf_qty_y', 0)
            if entry_y and current_y and qty_y:
                pnl_y = (current_y - entry_y) * qty_y

            entry_x = pos.get('mf_entry_price_x')
            current_x = pos.get('mf_current_price_x')
            qty_x = pos.get('mf_qty_x', 0)
            if entry_x and current_x and qty_x:
                pnl_x = (current_x - entry_x) * qty_x

            realized_pnl_sek = pnl_y + pnl_x

            # Capital = actual invested amount (entry_price * qty per leg)
            total_capital = 0
            if entry_y and qty_y:
                total_capital += entry_y * qty_y
            if entry_x and qty_x:
                total_capital += entry_x * qty_x
            # Fallback if no MF data entered
            if total_capital == 0:
                total_capital = pos.get('mf_total_capital', pos.get('notional', 0))

            realized_pnl_pct = (realized_pnl_sek / total_capital * 100) if total_capital > 0 else 0

            # Determine profit/loss from actual realized P/L (not Z-score direction)
            has_mf_data = (entry_y and current_y and qty_y) or (entry_x and current_x and qty_x)
            if has_mf_data:
                profit = realized_pnl_sek >= 0
            else:
                # Fallback to Z-score direction when no MF prices available
                profit = (z_change > 0) if direction == 'LONG' else (z_change < 0)
            
            status = pos.get('status', 'MANUAL CLOSE')
            
            # Confirmation dialog
            msg = f"Close position for {pos['pair']}?\n\n"
            msg += f"Direction: {direction}\n"
            msg += f"Entry Z: {entry_z:.2f}\n"
            msg += f"Current Z: {current_z:.2f}\n"
            msg += f"Z Change: {z_change:+.2f}\n"
            msg += f"Realized P/L: {realized_pnl_sek:+.0f} SEK ({realized_pnl_pct:+.1f}%)\n"
            msg += f"Status: {status}"
            
            reply = QMessageBox.question(self, "Close Position", msg,
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                pos = self.portfolio.pop(index)
                
                # Create trade history record
                result = "PROFIT" if profit else "LOSS"
                trade_record = {
                    'pair': pos['pair'],
                    'direction': direction,
                    'entry_date': pos.get('entry_date', ''),
                    'close_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'entry_z': entry_z,
                    'close_z': current_z,
                    'result': result,
                    'realized_pnl_sek': round(realized_pnl_sek, 2),
                    'realized_pnl_pct': round(realized_pnl_pct, 2),
                    'capital': total_capital,
                    'status': status,
                    # Preserve MF details for reference
                    'mini_y_name': pos.get('mini_y_name', ''),
                    'mini_x_name': pos.get('mini_x_name', ''),
                    'mf_entry_price_y': entry_y,
                    'mf_entry_price_x': entry_x,
                    'mf_close_price_y': current_y,
                    'mf_close_price_x': current_x,
                    'mf_qty_y': qty_y,
                    'mf_qty_x': qty_x,
                }
                self.trade_history.append(trade_record)
                
                self.update_portfolio_display()
                self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
                self._save_and_sync_portfolio()
                
                self.statusBar().showMessage(
                    f"Closed {pos['pair']} - {result} (Z: {entry_z:.2f} → {current_z:.2f}, "
                    f"P/L: {realized_pnl_sek:+.0f} SEK)"
                )
    
    def clear_all_positions(self):
        """Clear all positions."""
        if not self.portfolio:
            return
        
        reply = QMessageBox.question(self, "Clear All",
            "Are you sure you want to clear all positions?",
            QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.portfolio.clear()
            self.update_portfolio_display()
            self._update_metric_value(self.positions_metric, "0")
            
            # Auto-save portfolio (empty)
            self._save_and_sync_portfolio()


# ============================================================================
# MAIN
# ============================================================================

def main():
    # CRITICAL: Required for PyInstaller on Windows when multiprocessing is used
    import multiprocessing
    multiprocessing.freeze_support()

    # Print configuration info (shows paths, frozen status, etc.)
    print_config()

    # Initialize user data directory (first run: copies default files)
    initialize_user_data()

    # Setup file-based logging
    logger = setup_logging()
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")

    # Apply screen scaling BEFORE creating QApplication
    if SCREEN_SCALING_AVAILABLE:
        try:
            apply_screen_scaling()
            print_screen_info()
        except Exception as e:
            print(f"Screen scaling setup failed: {e}")
    else:
        # Fallback to standard High DPI settings
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    # Scale base font size based on screen
    if SCREEN_SCALING_AVAILABLE:
        try:
            scale = get_scale_factor()
            base_font_size = max(9, round(10 * scale))
        except (ValueError, RuntimeError, Exception):
            base_font_size = 10
    else:
        base_font_size = 10
    
    font = QFont("Segoe UI", base_font_size)
    app.setFont(font)
        
    window = PairsTradingTerminal()
    window.show()

    # Start auto-updater (checks at startup + every 2 hours)
    try:
        from auto_updater import setup_auto_updater
        window._updater = setup_auto_updater(window)
    except Exception as e:
        print(f"Auto-updater disabled: {e}")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    
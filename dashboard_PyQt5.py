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
    QFileDialog, QCheckBox, QAction, QStackedWidget, QCompleter,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QThread, QObject, pyqtSlot as Slot, QSize, QUrl, QPointF, QRectF
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter, QBrush, QPen, QPolygonF, QDesktopServices, QCursor

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

# Portfolio history tracking
try:
    from portfolio_history import PortfolioHistoryManager, format_performance_table
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

# EPS Mean Reversion strategy
try:
    from eps_mean_reversion import fetch_eps_and_price, screen_tickers as eps_screen_tickers, analyze_pe_mean_reversion, analyze_spread_mean_reversion
    EPS_MR_AVAILABLE = True
except ImportError:
    EPS_MR_AVAILABLE = False

# TTM Squeeze strategy
try:
    from squeeze import SqueezeAnalyzer, SqueezeResult, TickerSqueezeResult
    SQUEEZE_AVAILABLE = True
except ImportError:
    SQUEEZE_AVAILABLE = False
    print("Warning: squeeze.py not found.")
    print("Warning: eps_mean_reversion.py not found.")

# Avanza Options
try:
    from avanza_options import build_ticker_mapping, get_straddle_summary
    OPTIONS_AVAILABLE = True
except ImportError:
    OPTIONS_AVAILABLE = False

# Volatility Analytics
try:
    from vol_analytics import analyze_ticker as vol_analyze_ticker
    VOL_ANALYTICS_AVAILABLE = True
except ImportError:
    VOL_ANALYTICS_AVAILABLE = False
    print("Warning: avanza_options.py not found.")

# Import trading engine
from pairs_engine import (PairsTradingEngine, OUProcess, load_tickers_from_csv,
                          KalmanHedgeRatio, KalmanOUEstimator)


# ── Application configuration (portable paths for distribution) ──
from app_config import (
    Paths, APP_VERSION, APP_NAME,
    get_discord_webhook_url, save_discord_webhook_url,
    get_email_config, save_email_config,
    initialize_user_data, resource_path, get_user_data_dir,
    find_ticker_csv, find_matched_tickers_csv, setup_logging,
    print_config, _is_frozen
)
from daily_email import build_daily_summary_html, EmailWorker

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
SCHEDULED_HOUR = 23
SCHEDULED_MINUTE = 0

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

    if os.path.exists(_gdrive_history) or not os.path.exists(PORTFOLIO_HISTORY_FILE):
        PORTFOLIO_HISTORY_FILE = _gdrive_history

    if os.path.exists(_gdrive_engine) or not os.path.exists(ENGINE_CACHE_FILE):
        ENGINE_CACHE_FILE = _gdrive_engine

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
                    # Stale lock — ta bort
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
            return []

        # Vänta kort om filen nyligen ändrades (Google Drive sync)
        file_age = time.time() - os.path.getmtime(filepath)
        if file_age < 2:
            time.sleep(2)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        positions = data.get("positions", [])
        
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
        # No engine to save
        return False
    
    if not _acquire_lock(filepath):
        print("[Engine Cache] Could not save - file is locked")
        return False
    
    try:
        # Konvertera bort PyArrow-backade dtypes innan pickle (miljö-oberoende)
        def _strip_arrow(df):
            if df is None:
                return None
            try:
                has_arrow = any(
                    type(dt).__name__ == 'ArrowDtype' or
                    (hasattr(dt, 'storage') and getattr(dt, 'storage', '') == 'pyarrow')
                    for dt in df.dtypes
                )
                return df.convert_dtypes(dtype_backend='numpy_nullable') if has_arrow else df
            except Exception:
                return df

        # Hämta tickers från price_data kolumner
        tickers = list(engine.price_data.columns) if engine.price_data is not None else []

        # Extrahera relevant data från engine
        cache_data = {
            'price_data': _strip_arrow(engine.price_data),
            'raw_price_data': _strip_arrow(getattr(engine, 'raw_price_data', None)),
            'viable_pairs': _strip_arrow(engine.viable_pairs),
            'pairs_stats': _strip_arrow(engine.pairs_stats),
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
        
        # Engine cache saved
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
            return None

        # Vänta kort om filen nyligen ändrades (Google Drive sync)
        file_age = time.time() - os.path.getmtime(filepath)
        if file_age < 2:
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
        
        print(f"[Engine Cache] {n_tickers} tickers, {n_pairs} viable pairs loaded")
        
        return cache_data
        
    except ImportError as e:
        if 'pyarrow' in str(e).lower():
            print(f"[Engine Cache] Cache inkompatibel (sparad med pyarrow, saknas i denna miljö).")
            print(f"[Engine Cache] Tar bort gammal cache — kör ny scan för att bygga om.")
            try:
                os.remove(filepath)
            except OSError:
                pass
        else:
            print(f"[Engine Cache] ImportError vid laddning: {e}")
            traceback.print_exc()
        return None
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
        # VolCache saved OK
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
            pass  # Cache exists but no MS_Asset data, reloading
    
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
                df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
                
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
                    # Required columns not found, skipping
                    continue
                
                # Load MS_Asset mapping if available
                ticker_to_ms_asset = {}
                if 'MS_Asset' in df.columns:
                    ticker_to_ms_asset = {
                        k: v for k, v in zip(df['Ticker'], df['MS_Asset']) 
                        if pd.notna(v) and v != ''
                    }
                    if ticker_to_ms_asset:
                        # Found file with MS_Asset - use it!
                        _ticker_mapping_cache = (ticker_to_ms, ms_to_ticker, ticker_to_ms_asset)
                        return _ticker_mapping_cache
                else:
                    pass  # No MS_Asset column
                
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


# Bloomberg futures month codes
_FUTURES_MONTH_CODES = 'FGHJKMNQUVXZ'  # Jan-Dec
_FUTURES_MONTH_SET = set(_FUTURES_MONTH_CODES)
_commodity_roll_cache = {}  # {stale_asset: rolled_asset}


_MONTH_NAME_TO_CODE = {
    'jan': 'F', 'feb': 'G', 'mar': 'H', 'apr': 'J', 'may': 'K', 'jun': 'M',
    'jul': 'N', 'aug': 'Q', 'sep': 'U', 'oct': 'V', 'nov': 'X', 'dec': 'Z',
}


def _roll_commodity_asset(ms_asset: str, ms_name: str = None, session=None) -> str:
    """Auto-roll stale commodity contract codes (e.g. GCJ6 → GCM6).

    Strategy:
    1. If ms_name contains a month (e.g. "Gold Jun26"), derive the correct code directly.
    2. Fallback: probe current month + next 12 months sequentially.
    """
    if not ms_asset or not ms_asset.endswith('_Comdty') or not SCRAPING_AVAILABLE:
        return ms_asset

    # Check cache first
    cache_key = (ms_asset, ms_name or '')
    if cache_key in _commodity_roll_cache:
        cached = _commodity_roll_cache[cache_key]
        print(f"[ROLL] Using cached: {ms_asset} → {cached}")
        return cached

    # Parse asset code: base + month_letter + year_digits + _Comdty
    m = re.match(r'^(.+?)([FGHJKMNQUVXZ])(\d{1,2})_Comdty$', ms_asset)
    if not m:
        return ms_asset

    base = m.group(1)       # e.g. "GC", "CL", "C_"
    year_str = m.group(3)   # e.g. "6" or "26"
    uses_2digit_year = len(year_str) == 2

    # ── Strategy 1: Derive from ms_name (no HTTP needed) ──
    if ms_name:
        name_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{2})\b',
                               ms_name, re.IGNORECASE)
        if name_match:
            month_code = _MONTH_NAME_TO_CODE[name_match.group(1).lower()]
            year_2d = name_match.group(2)  # e.g. "26"
            if uses_2digit_year:
                year_code = year_2d
            else:
                year_code = year_2d[-1]  # "26" → "6"
            corrected = f"{base}{month_code}{year_code}_Comdty"
            if corrected != ms_asset:
                print(f"[ROLL] {ms_asset} → {corrected} (derived from name '{ms_name}')")
                _commodity_roll_cache[cache_key] = corrected
                return corrected

    # ── Strategy 2: Sequential probe from current month forward ──
    now = datetime.now()
    current_month = now.month
    current_year = now.year

    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

    for offset in range(18):
        trial_month = ((current_month - 1 + offset) % 12) + 1
        trial_year = current_year + (current_month - 1 + offset) // 12
        month_code = _FUTURES_MONTH_CODES[trial_month - 1]

        if uses_2digit_year:
            year_code = str(trial_year % 100)
        else:
            year_code = str(trial_year % 10)

        trial_asset = f"{base}{month_code}{year_code}_Comdty"
        if trial_asset == ms_asset:
            continue

        try:
            params = {
                "f_pc": "LeverageProducts",
                "f_pt": "MiniFuture",
                "f_asset": trial_asset,
                "p_s": 1,
                "p_n": 1
            }
            _, rows, total = _scrape_ms_table_page(
                "https://etp.morganstanley.com/se/sv/produkter", params, session)
            count = total if total else len(rows) if rows else 0
            # Only accept if it has substantial products (>20), otherwise it's likely
            # a stale contract or MS returning "popular products" fallback
            if count > 20:
                print(f"[ROLL] {ms_asset} → {trial_asset} (probed, found {count} products)")
                _commodity_roll_cache[cache_key] = trial_asset
                return trial_asset
        except Exception:
            continue

    print(f"[ROLL] No active contract found for {ms_asset}, keeping original")
    _commodity_roll_cache[cache_key] = ms_asset
    return ms_asset


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


def _parse_ms_ask_price(val):
    """Parse ask (sälj) price from MS Köp/Sälj column: '165,530SEK166,730SEK' → 166.73.

    Returns the ask price (what you pay when buying).  Falls back to bid if
    ask is missing (e.g. '194,900SEK-SEK').
    """
    if not isinstance(val, str):
        return None
    prices = re.findall(r'([\d,]+)SEK', val)
    if not prices:
        return None
    try:
        bid = float(prices[0].replace(',', '.'))
        if len(prices) > 1:
            ask = float(prices[1].replace(',', '.'))
            return ask  # prefer ask (the price you buy at)
        return bid  # fallback to bid if ask missing
    except (ValueError, IndexError):
        return None


def _fetch_ms_product_price(isin: str, session=None) -> float:
    """Fetch ask price from MS product detail page as fallback.

    Used when the Köp/Sälj column in the table listing is empty
    (market maker wasn't quoting at that moment).
    """
    if not SCRAPING_AVAILABLE or not isin or isin == 'N/A':
        return None
    try:
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0"})
        url = f"https://etp.morganstanley.com/se/sv/product-details/{isin.lower()}"
        r = session.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Look for ask price box (class contains 'ask')
        ask_div = soup.find('div', class_=lambda c: c and 'ask' in c.lower())
        if ask_div:
            text = ask_div.get_text(strip=True)
            prices = re.findall(r'([\d,]+)SEK', text)
            if prices:
                return float(prices[0].replace(',', '.'))
        # Fallback: look for bid price box
        bid_div = soup.find('div', class_=lambda c: c and 'bid' in c.lower())
        if bid_div:
            text = bid_div.get_text(strip=True)
            prices = re.findall(r'([\d,]+)SEK', text)
            if prices:
                return float(prices[0].replace(',', '.'))
    except Exception:
        pass
    return None


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

    # Parse Köp/Sälj column → actual instrument price in SEK
    bid_ask_col = None
    for col in df.columns:
        if 'köp' in col.lower() or 'sälj' in col.lower() or 'bid' in col.lower() or 'ask' in col.lower():
            bid_ask_col = col
            break
    if bid_ask_col:
        df['InstrumentPriceSEK'] = df[bid_ask_col].apply(_parse_ms_ask_price)

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

    # Parse Köp/Sälj column → actual instrument price in SEK
    bid_ask_col = None
    for col in df.columns:
        if 'köp' in col.lower() or 'sälj' in col.lower() or 'bid' in col.lower() or 'ask' in col.lower():
            bid_ask_col = col
            break
    if bid_ask_col:
        df['InstrumentPriceSEK'] = df[bid_ask_col].apply(_parse_ms_ask_price)

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

    # Auto-roll stale commodity contracts
    if ms_asset.endswith('_Comdty'):
        ms_asset = _roll_commodity_asset(ms_asset, ms_name=ms_name)

    # Fetch certificates for this asset
    df_certs = fetch_certificates_for_asset(ms_asset)

    if df_certs.empty:
        return None

    # VIKTIGT: Verifiera att de hämtade certifikaten faktiskt är för rätt underliggande!
    # Morgan Stanley kan returnera "populära produkter" om tillgången inte finns.
    if 'Underliggande tillgång' in df_certs.columns:
        # Filtrera på underliggande som matchar ms_name (case-insensitive)
        mask = _match_underlying(df_certs['Underliggande tillgång'], ms_name)
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


def _match_underlying(df_col: pd.Series, ms_name: str) -> pd.Series:
    """Matcha underliggande kolumn mot MS-namn, hanterar kontraktsmanad och sprakvarianter."""
    # Rensa kontraktsmanad: "Gold Jun26" -> "Gold"
    clean_name = re.sub(r'\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}$',
                        '', ms_name, flags=re.IGNORECASE).strip()
    search_terms = [clean_name.lower()]
    print(f"[MATCH] ms_name={ms_name!r} → clean={clean_name!r}, terms will include aliases")
    # Svenska/engelska commodity-alias
    _ALIASES = {
        'gold': ['guld'], 'guld': ['gold'],
        'silver': ['silver'], 'copper': ['koppar'], 'koppar': ['copper'],
        'brent crude oil': ['brent', 'brentolja'],
        'wti crude oil': ['wti', 'olja', 'råolja'],
        'natural gas': ['naturgas', 'natgas'],
        'platinum': ['platina'], 'palladium': ['palladium'],
        'wheat': ['vete'], 'corn': ['majs'],
        'sugar': ['socker'], 'coffee': ['kaffe'], 'cocoa': ['kakao'],
    }
    for eng, aliases in _ALIASES.items():
        if eng in clean_name.lower():
            search_terms.extend(aliases)
        for a in aliases:
            if a in clean_name.lower():
                search_terms.append(eng)
    print(f"[MATCH] Final search_terms={search_terms}")
    col_lower = df_col.str.lower()
    mask = pd.Series(False, index=df_col.index)
    for term in search_terms:
        mask = mask | col_lower.str.contains(term, na=False, regex=False)
    print(f"[MATCH] Matched {mask.sum()} of {len(mask)} rows")
    return mask


def fetch_all_instruments_for_ticker(ticker: str, direction: str, ticker_to_ms: dict,
                                      ticker_to_ms_asset: dict = None) -> list:
    """
    Hämta ALLA tillgängliga instrument (mini futures + certifikat) för en ticker och riktning.

    Returnerar en lista av dicts sorterade efter hävstång (lägst först).
    Varje dict har samma format som find_best_minifuture() returnerar.

    Optimized: fetches mini futures and certificates in parallel, batches
    fallback price lookups.
    """
    instruments = []
    _DBG = f"[DERIV] {ticker} {direction}"

    # Slå upp MS-namn och asset-kod
    ms_name = ticker_to_ms.get(ticker)
    if not ms_name:
        base_ticker = ticker.split('.')[0]
        ms_name = ticker_to_ms.get(base_ticker)
    print(f"{_DBG} ms_name={ms_name!r}")

    ms_asset = None
    if ticker_to_ms_asset:
        ms_asset = ticker_to_ms_asset.get(ticker)
        if not ms_asset:
            base_ticker = ticker.split('.')[0]
            ms_asset = ticker_to_ms_asset.get(base_ticker)
    print(f"{_DBG} ms_asset={ms_asset!r} (before roll)")

    if not ms_asset:
        print(f"{_DBG} ABORT: no ms_asset found")
        return instruments

    spot_price = get_spot_price(ticker)
    print(f"{_DBG} spot_price={spot_price}")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # ── Auto-roll stale commodity contracts ──
    if ms_asset.endswith('_Comdty'):
        ms_asset = _roll_commodity_asset(ms_asset, ms_name=ms_name, session=session)
        print(f"{_DBG} ms_asset={ms_asset!r} (after roll)")

    # ── Fetch mini futures AND certificates in parallel ──
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_mf = executor.submit(fetch_minifutures_for_asset, ms_asset, session)
        fut_cert = executor.submit(fetch_certificates_for_asset, ms_asset, session)
        df_mf = fut_mf.result()
        df_cert = fut_cert.result()
    print(f"{_DBG} Fetched: {len(df_mf)} mini futures, {len(df_cert)} certificates")

    # ── Detect "popular products" fallback (invalid asset code) ──
    # MS returns ALL products (~1000) when the asset code is unrecognized.
    # If first row doesn't match our underlying, try alternative exchange suffixes.
    if ms_asset.endswith('_Equity') and len(df_mf) >= 500 and ms_name:
        underlying_col = next((c for c in df_mf.columns if 'underliggande' in c.lower()), None)
        if underlying_col:
            first_underlying = str(df_mf.iloc[0][underlying_col]).lower()
            clean_name = re.sub(r'\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2}$',
                                '', ms_name, flags=re.IGNORECASE).strip().lower()
            if clean_name not in first_underlying and first_underlying not in clean_name:
                # Asset code is invalid — try alternative exchange suffixes
                base_code = ms_asset.rsplit('_', 2)[0]  # e.g. "EQNR"
                _ALT_EXCHANGES = ['NO_Equity', 'SS_Equity', 'DC_Equity', 'US_Equity',
                                  'LN_Equity', 'FP_Equity', 'GY_Equity', 'FH_Equity', 'JT_Equity']
                current_suffix = '_'.join(ms_asset.split('_')[-2:])
                print(f"{_DBG} Detected 'popular products' fallback for {ms_asset}, trying alternatives...")
                for alt in _ALT_EXCHANGES:
                    if alt == current_suffix:
                        continue
                    trial = f"{base_code}_{alt}"
                    df_trial = fetch_minifutures_for_asset(trial, session)
                    if not df_trial.empty and len(df_trial) < 500:
                        # Verify first row matches our underlying
                        trial_first = str(df_trial.iloc[0].get(underlying_col, '')).lower()
                        if clean_name in trial_first or trial_first in clean_name:
                            print(f"{_DBG} Found correct asset: {trial} ({len(df_trial)} products)")
                            ms_asset = trial
                            df_mf = df_trial
                            df_cert = fetch_certificates_for_asset(trial, session)
                            break

    if not df_mf.empty:
        print(f"{_DBG} MF columns: {list(df_mf.columns)}")
        print(f"{_DBG} MF first row: {dict(df_mf.iloc[0])}")

    # ── Process Mini Futures ──
    # Track ISINs needing fallback price lookup
    _mf_need_price = []  # list of (index_in_instruments, isin)

    if not df_mf.empty and ms_name:
        # Verifiera underliggande med fuzzy matching (hanterar commodity-namn)
        underlying_col = None
        for col in df_mf.columns:
            if 'underliggande' in col.lower():
                underlying_col = col
                break
        if underlying_col:
            unique_underlyings = df_mf[underlying_col].unique().tolist()
            print(f"{_DBG} Unique underlyings in data: {unique_underlyings}")
            mask = _match_underlying(df_mf[underlying_col], ms_name)
            n_before = len(df_mf)
            df_mf = df_mf[mask]
            print(f"{_DBG} Underlying filter: {n_before} → {len(df_mf)} (ms_name={ms_name!r})")
        else:
            print(f"{_DBG} WARNING: No 'underliggande' column found!")

        # Filtrera riktning
        riktning_col = None
        for col in df_mf.columns:
            if 'riktning' in col.lower():
                riktning_col = col
                break
        if riktning_col and not df_mf.empty:
            unique_directions = df_mf[riktning_col].unique().tolist()
            print(f"{_DBG} Unique directions: {unique_directions}, filtering for '{direction}'")
            n_before = len(df_mf)
            df_mf = df_mf[df_mf[riktning_col].str.contains(direction, case=False, na=False)]
            print(f"{_DBG} Direction filter: {n_before} → {len(df_mf)}")
        elif not riktning_col:
            print(f"{_DBG} WARNING: No 'riktning' column found!")

        # Pre-find column names once (not per row)
        isin_col = None
        name_col = None
        fin_col = None
        for col in df_mf.columns:
            cl = col.lower()
            if 'isin' in cl and isin_col is None:
                isin_col = col
            if (cl == 'namn' or 'name' in cl) and name_col is None:
                name_col = col
            if ('finansieringsnivånum' in cl or col == 'FinansieringsnivåNum') and fin_col is None:
                fin_col = col
        print(f"{_DBG} Columns: isin={isin_col}, name={name_col}, fin={fin_col}")
        print(f"{_DBG} spot={spot_price}, rows_left={len(df_mf)}, has_fin_col={fin_col is not None}")

        # Beräkna hävstång för varje mini future
        if not df_mf.empty and spot_price is not None and fin_col:
            _dbg_skipped = {'no_fin': 0, 'bad_dir': 0, 'low_lev': 0}
            for _, row in df_mf.iterrows():
                fin_level = row.get(fin_col)
                if fin_level is None or pd.isna(fin_level):
                    _dbg_skipped['no_fin'] += 1
                    continue

                # Filtrera ogiltiga finansieringsnivåer
                if direction.lower() == 'long' and fin_level >= spot_price:
                    _dbg_skipped['bad_dir'] += 1
                    continue
                if direction.lower() == 'short' and fin_level <= spot_price:
                    _dbg_skipped['bad_dir'] += 1
                    continue

                # Beräkna teoretisk hävstång
                if direction.lower() == 'long':
                    leverage = spot_price / (spot_price - fin_level)
                else:
                    leverage = spot_price / (fin_level - spot_price)

                if leverage <= 1:
                    _dbg_skipped['low_lev'] += 1
                    continue

                isin_url = row.get(isin_col, '') if isin_col else ''
                isin = extract_isin_from_url(isin_url)
                avanza_link = create_avanza_link(isin)

                # Use real instrument price from MS Köp/Sälj if available
                mf_instrument_price = row.get('InstrumentPriceSEK', None)
                if mf_instrument_price is not None and pd.notna(mf_instrument_price) and mf_instrument_price > 0:
                    mf_instrument_price = float(mf_instrument_price)
                else:
                    mf_instrument_price = None  # will try batch fallback below

                inst = {
                    'name': row.get(name_col, 'N/A') if name_col else 'N/A',
                    'underlying': row.get(underlying_col, ms_name) if underlying_col else ms_name,
                    'direction': direction,
                    'financing_level': fin_level,
                    'leverage': leverage,
                    'spot_price': spot_price,
                    'instrument_price': mf_instrument_price,
                    'isin': isin or 'N/A',
                    'avanza_link': avanza_link,
                    'ticker': ticker,
                    'product_type': 'Mini Future'
                }
                instruments.append(inst)

                # Queue for batch fallback if price missing
                if mf_instrument_price is None and isin and isin != 'N/A':
                    _mf_need_price.append((len(instruments) - 1, isin))

            mf_count = sum(1 for i in instruments if i.get('product_type') == 'Mini Future')
            print(f"{_DBG} MF leverage loop: {mf_count} valid, skipped={_dbg_skipped}")
        else:
            print(f"{_DBG} MF leverage loop SKIPPED: empty={df_mf.empty}, spot={spot_price}, fin_col={fin_col}")

    # ── Process Certifikat ──
    _cert_need_price = []  # list of (index_in_instruments, isin)

    if not df_cert.empty and ms_name:
        # Verifiera underliggande med fuzzy matching
        if 'Underliggande tillgång' in df_cert.columns:
            mask = _match_underlying(df_cert['Underliggande tillgång'], ms_name)
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

            # Pre-find column names once
            cert_name_col = None
            for col in df_cert.columns:
                if col.lower() == 'namn' or 'name' in col.lower():
                    cert_name_col = col
                    break

            for _, row in df_cert.iterrows():
                leverage_abs = abs(row['DailyLeverage'])

                # Use real instrument price from MS Köp/Sälj if available
                instrument_price = row.get('InstrumentPriceSEK', None)
                if instrument_price is not None and pd.notna(instrument_price) and instrument_price > 0:
                    instrument_price = float(instrument_price)
                else:
                    instrument_price = None  # will try batch fallback below

                cert_isin_url = row.get('ISIN', '')
                cert_isin = extract_isin_from_url(cert_isin_url)

                if instrument_price is None:
                    # Fallback 2: calculate from multiplier
                    financing_level_calc = row.get('FinansieringsnivåNum', None)
                    multiplier_calc = row.get('MultiplikatorNum', None)
                    ratio_calc = row.get('RatioNum', None)
                    if multiplier_calc is None and ratio_calc is not None and ratio_calc > 0:
                        multiplier_calc = 1.0 / ratio_calc
                    if financing_level_calc is not None and spot_price is not None and multiplier_calc is not None:
                        if direction.lower() == 'long':
                            instrument_price = multiplier_calc * (spot_price - financing_level_calc)
                        else:
                            instrument_price = multiplier_calc * (financing_level_calc - spot_price)
                        instrument_price = max(0.01, abs(instrument_price))

                financing_level = row.get('FinansieringsnivåNum', None)
                multiplier = row.get('MultiplikatorNum', None)
                ratio = row.get('RatioNum', None)
                if multiplier is None and ratio is not None and ratio > 0:
                    multiplier = 1.0 / ratio

                avanza_link = create_avanza_link(cert_isin)

                inst = {
                    'name': row.get(cert_name_col, 'N/A') if cert_name_col else row.get('Namn', 'N/A'),
                    'underlying': row.get('Underliggande tillgång', ms_name),
                    'direction': direction,
                    'financing_level': financing_level,
                    'leverage': leverage_abs,
                    'daily_leverage': row['DailyLeverage'],
                    'spot_price': spot_price,
                    'multiplier': multiplier,
                    'ratio': ratio,
                    'instrument_price': instrument_price,
                    'isin': cert_isin or 'N/A',
                    'avanza_link': avanza_link,
                    'ticker': ticker,
                    'product_type': 'Certificate'
                }
                instruments.append(inst)

                # Queue for batch fallback if price still missing
                if instrument_price is None and cert_isin and cert_isin != 'N/A':
                    _cert_need_price.append((len(instruments) - 1, cert_isin))

    # ── Batch fetch missing prices in parallel ──
    all_need_price = _mf_need_price + _cert_need_price
    if all_need_price:
        with ThreadPoolExecutor(max_workers=8) as executor:
            price_futures = {
                executor.submit(_fetch_ms_product_price, isin, session): idx
                for idx, isin in all_need_price
            }
            for fut in as_completed(price_futures):
                idx = price_futures[fut]
                try:
                    price = fut.result()
                    if price is not None:
                        instruments[idx]['instrument_price'] = price
                except Exception:
                    pass

    # Sortera efter hävstång (lägst först)
    instruments.sort(key=lambda x: x['leverage'])

    n_mf = sum(1 for i in instruments if i.get('product_type') == 'Mini Future')
    n_cert = sum(1 for i in instruments if i.get('product_type') == 'Certificate')
    print(f"{_DBG} FINAL: {len(instruments)} instruments ({n_mf} mini, {n_cert} cert)")

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
    
    
    # Auto-roll stale commodity contracts
    if ms_asset and ms_asset.endswith('_Comdty'):
        ms_asset = _roll_commodity_asset(ms_asset, ms_name=ms_name)

    # OPTIMIZATION: Try direct fetch via MS_Asset first (10x faster!)
    df_filtered = None
    if ms_asset:
        # Fetch directly for this specific underlying - only 1 request!
        df_asset = fetch_minifutures_for_asset(ms_asset)

        if not df_asset.empty:
            # VIKTIGT: Verifiera att hämtade minifutures är för rätt underliggande!
            underlying_col = None
            for col in df_asset.columns:
                if 'underliggande' in col.lower():
                    underlying_col = col
                    break

            if underlying_col:
                mask = _match_underlying(df_asset[underlying_col], ms_name)
                df_asset = df_asset[mask]

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

    # Search for best (units_y, units_x) combination.
    # Strategy: collect all candidates with β deviation < 5%, then pick cheapest.
    # If none are under 5%, pick the one with lowest deviation.
    GOOD_ENOUGH_DEV = 0.05  # 5% hedge ratio deviation threshold
    candidates = []

    # Determine search range: from min_units_x up to ~3× that
    max_search_x = max(min_units_x + 10, min_units_x * 3)
    for ux in range(min_units_x, max_search_x + 1):
        # Ideal fractional units_y for perfect hedge: ux * exp_x = β * uy * exp_y
        if beta * exp_per_unit_y > 0:
            uy_ideal = ux * exp_per_unit_x / (beta * exp_per_unit_y)
        else:
            uy_ideal = min_units_y
        # Try floor and ceil of ideal
        for uy in [max(min_units_y, math.floor(uy_ideal)),
                    max(min_units_y, math.ceil(uy_ideal))]:
            actual_beta = (ux * exp_per_unit_x) / (uy * exp_per_unit_y) if uy * exp_per_unit_y > 0 else 0
            deviation = abs(actual_beta - beta) / beta if beta > 0 else 0
            total = uy * price_y + ux * price_x
            candidates.append((deviation, total, uy, ux))

    # Pick: cheapest among "good enough" candidates, or lowest deviation overall
    good = [c for c in candidates if c[0] <= GOOD_ENOUGH_DEV]
    if good:
        best = min(good, key=lambda c: c[1])  # cheapest with dev ≤ 5%
    else:
        best = min(candidates, key=lambda c: (c[0], c[1]))  # lowest dev

    units_y, units_x = best[2], best[3]

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

            # Progress callback for fetch_data: maps batch progress → 10-40%
            def fetch_progress(batch_num, total_batches, msg):
                pct = 10 + int(batch_num / max(1, total_batches) * 30)
                self.progress.emit(pct, msg)

            self.progress.emit(10, f"Fetching data for {len(self.tickers)} tickers...")
            engine.fetch_data(self.tickers, self.lookback, progress_callback=fetch_progress)

            if engine.price_data is None or len(engine.price_data.columns) == 0:
                self.error.emit("No price data loaded")
                return

            loaded = len(engine.price_data.columns)
            self.progress.emit(40, f"Loaded {loaded} tickers, screening pairs...")

            # Progress callback for screen_pairs: maps screening progress → 40-95%
            def screen_progress(phase, current, total, msg):
                if phase == 'correlation':
                    # Correlation phase: 40-55%
                    pct = 40 + int(current / max(1, total) * 15)
                elif phase == 'screening':
                    # Pair testing phase: 55-95%
                    pct = 55 + int(current / max(1, total) * 40)
                else:
                    pct = 40
                self.progress.emit(pct, msg)

            engine.screen_pairs(correlation_prefilter=True, progress_callback=screen_progress)

            self.progress.emit(100, "Complete!")
            self.result.emit(engine)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()



class PriceDataWorker(QObject):
    """Worker som bara hämtar prisdata (utan pairs screening) för Master Scanner."""
    finished = Signal()
    progress = Signal(int, str)
    error = Signal(str)
    result = Signal(object)  # Emittar PairsTradingEngine med price_data

    def __init__(self, tickers: List[str], lookback: str = '2y'):
        super().__init__()
        self.tickers = tickers
        self.lookback = lookback

    @Slot()
    def run(self):
        try:
            print(f"[SCANNER] PriceDataWorker: Fetching {len(self.tickers)} tickers, period={self.lookback}")
            self.progress.emit(5, f"Initializing engine...")
            engine = PairsTradingEngine()

            def fetch_progress(batch_num, total_batches, msg):
                pct = 10 + int(batch_num / max(1, total_batches) * 80)
                self.progress.emit(pct, msg)
                print(f"[SCANNER] Fetch progress: {msg}")

            self.progress.emit(10, f"Downloading {len(self.tickers)} tickers...")
            engine.fetch_data(self.tickers, self.lookback, progress_callback=fetch_progress)

            if engine.price_data is None or len(engine.price_data.columns) == 0:
                self.error.emit("No price data loaded")
                return

            loaded = len(engine.price_data.columns)
            print(f"[SCANNER] PriceDataWorker: Loaded {loaded} tickers successfully")
            self.progress.emit(100, f"Loaded {loaded} tickers")
            self.result.emit(engine)
        except Exception as e:
            print(f"[SCANNER] PriceDataWorker ERROR: {e}")
            import traceback
            traceback.print_exc()
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


class MSInstrumentWorker(QObject):
    """Worker thread for fetching Morgan Stanley instruments (prevents GUI freeze)."""
    finished = Signal()
    result = Signal(list, list, dict)  # all_instruments_y, all_instruments_x, ticker_to_ms
    error = Signal(str)

    def __init__(self, y_ticker, x_ticker, dir_y, dir_x, ticker_to_ms, ticker_to_ms_asset):
        super().__init__()
        self.y_ticker = y_ticker
        self.x_ticker = x_ticker
        self.dir_y = dir_y
        self.dir_x = dir_x
        self.ticker_to_ms = ticker_to_ms
        self.ticker_to_ms_asset = ticker_to_ms_asset

    @Slot()
    def run(self):
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_y = executor.submit(
                    fetch_all_instruments_for_ticker,
                    self.y_ticker, self.dir_y, self.ticker_to_ms, self.ticker_to_ms_asset)
                fut_x = executor.submit(
                    fetch_all_instruments_for_ticker,
                    self.x_ticker, self.dir_x, self.ticker_to_ms, self.ticker_to_ms_asset)
                all_y = fut_y.result()
                all_x = fut_x.result()
            self.result.emit(all_y, all_x, self.ticker_to_ms)
        except Exception as e:
            self.error.emit(str(e))
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


class OptionPriceFetchWorker(QObject):
    """Worker: fetch current bid prices for Avanza options (for straddle P/L tracking)."""
    finished = Signal()
    result = Signal(dict)   # {orderbook_id: {'buy': float, 'sell': float, 'last': float}}
    error = Signal(str)
    status_message = Signal(str)

    def __init__(self, orderbook_ids: list):
        super().__init__()
        self.orderbook_ids = orderbook_ids

    @Slot()
    def run(self):
        try:
            import requests
            self.status_message.emit(f"Fetching {len(self.orderbook_ids)} option prices...")
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
            })
            results = {}
            for ob_id in self.orderbook_ids:
                try:
                    r = session.get(
                        f"https://www.avanza.se/_api/market-guide/option/{ob_id}",
                        timeout=10)
                    if r.status_code == 200:
                        q = r.json().get('quote', {})
                        results[ob_id] = {
                            'buy': q.get('buy'),
                            'sell': q.get('sell'),
                            'last': q.get('last'),
                        }
                except Exception as e:
                    print(f"[OPT_PRICE] Error fetching {ob_id}: {e}")
            self.result.emit(results)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class VolSurfaceWorker(QObject):
    """Worker: fetch IV from Avanza for strikes near ATM per expiry.
    Fokuserad: hamtar bara optioner som ar relevanta (nara ATM, ±5 strikes).
    Ateranvander cachade IDs fran all_straddles for att undvika extra matris-anrop.
    """
    finished = Signal()
    result = Signal(object)   # dict: {surface_df, spot, ticker, garch_ts}
    error = Signal(str)

    def __init__(self, avanza_id: str, ticker: str, spot: float,
                 expiries: list, garch_ts: dict = None, r: float = 0.02,
                 cached_straddles: object = None):
        super().__init__()
        self.avanza_id = avanza_id
        self.ticker = ticker
        self.spot = spot
        self.expiries = expiries
        self.garch_ts = garch_ts or {}
        self.r = r
        self.cached_straddles = cached_straddles  # DataFrame fran options_data

    @Slot()
    def run(self):
        try:
            import requests
            import time as _time
            from datetime import date, datetime as dt
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from requests.adapters import HTTPAdapter

            HEADERS = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            }
            session = requests.Session()
            session.headers.update(HEADERS)
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
            session.mount('https://', adapter)
            today = date.today()
            t0 = _time.time()

            # ── Steg 1: Bestam vilka strikes att hamta IV for ───────────
            # Anvand cachad all_straddles — filtrera till ±5 strikes fran ATM per expiry
            df = self.cached_straddles
            if df is None or not hasattr(df, 'empty') or df.empty:
                self.error.emit("No cached straddle data")
                return

            iv_jobs = []  # (strike, dte, opt_id, is_call)
            n_strikes_per_expiry = 5  # ±5 strikes fran ATM = max 11 per expiry

            for expiry, group in df.groupby('Expiry'):
                expiry_str = str(expiry)[:10]
                try:
                    dte = (dt.strptime(expiry_str, '%Y-%m-%d').date() - today).days
                except (ValueError, TypeError):
                    continue
                if dte < 1:
                    continue

                # Sortera efter naerhet till spot, ta ±N narmaste
                group = group.copy()
                group['_dist'] = (group['Strike'] - self.spot).abs()
                group = group.sort_values('_dist')
                nearby = group.head(n_strikes_per_expiry * 2 + 1)  # Max 11

                for _, row in nearby.iterrows():
                    strike = row.get('Strike', 0)
                    if not strike or strike <= 0:
                        continue
                    c_id = str(row.get('C_Id', ''))
                    p_id = str(row.get('P_Id', ''))
                    if c_id:
                        iv_jobs.append((strike, dte, c_id, True))
                    if p_id:
                        iv_jobs.append((strike, dte, p_id, False))

            n_expiries = df['Expiry'].nunique()
            print(f"[VOLSURF] {self.ticker}: {len(iv_jobs)} IV jobs "
                  f"({n_expiries} expiries x ±{n_strikes_per_expiry} strikes near ATM)")

            if not iv_jobs:
                self.error.emit("No options found for IV")
                return

            # ── Steg 2: Hamta IV parallellt fran Avanza ─────────────────
            iv_results = {}  # (strike, dte, is_call) -> iv_value

            def _fetch_iv(job):
                strike, dte, opt_id, is_call = job
                try:
                    r2 = session.get(
                        f"https://www.avanza.se/_api/market-guide/option/{opt_id}/details",
                        timeout=10)
                    if r2.status_code == 200:
                        ga = r2.json().get('optionAnalytics', {})
                        iv_val = ga.get('implicitMidVolatility')
                        if iv_val is None:
                            iv_b = ga.get('implicitBuyVolatility')
                            iv_a = ga.get('implicitSellVolatility')
                            if iv_b is not None and iv_a is not None:
                                iv_val = (iv_b + iv_a) / 2
                            elif iv_a is not None:
                                iv_val = iv_a
                            elif iv_b is not None:
                                iv_val = iv_b
                        if iv_val is not None and float(iv_val) > 0.5:
                            return (strike, dte, is_call), float(iv_val)
                except Exception:
                    pass
                return (strike, dte, is_call), None

            with ThreadPoolExecutor(max_workers=10) as pool:
                futs = [pool.submit(_fetch_iv, j) for j in iv_jobs]
                for fut in as_completed(futs):
                    key, val = fut.result()
                    if val is not None:
                        iv_results[key] = val

            # ── Steg 3: Bygg surface DataFrame ─────────────────────────
            points = {}  # (strike, dte) -> {'c_iv': float, 'p_iv': float}
            for (strike, dte, is_call), iv in iv_results.items():
                key = (strike, dte)
                if key not in points:
                    points[key] = {'c_iv': None, 'p_iv': None}
                if is_call:
                    points[key]['c_iv'] = iv
                else:
                    points[key]['p_iv'] = iv

            rows = []
            for (strike, dte), ivs in points.items():
                c_iv = ivs['c_iv']
                p_iv = ivs['p_iv']
                if c_iv is not None and p_iv is not None:
                    mid_iv = (c_iv + p_iv) / 2
                elif c_iv is not None:
                    mid_iv = c_iv
                elif p_iv is not None:
                    mid_iv = p_iv
                else:
                    continue
                rows.append({
                    'Strike': strike, 'DTE': dte,
                    'Moneyness': round(strike / self.spot, 3),
                    'C_IV': c_iv, 'P_IV': p_iv, 'MidIV': mid_iv,
                })

            if not rows:
                self.error.emit("No valid IV data from Avanza")
                return

            surface_df = pd.DataFrame(rows).sort_values(['DTE', 'Strike']).reset_index(drop=True)
            elapsed = _time.time() - t0
            print(f"[VOLSURF] {self.ticker}: {len(surface_df)} points "
                  f"({surface_df['DTE'].nunique()} expiries x "
                  f"~{len(surface_df)//max(1,surface_df['DTE'].nunique())} strikes) "
                  f"in {elapsed:.1f}s ({len(iv_results)}/{len(iv_jobs)} IV ok)")

            self.result.emit({
                'surface_df': surface_df,
                'spot': self.spot,
                'ticker': self.ticker,
                'garch_ts': self.garch_ts,
            })
        except Exception as e:
            traceback.print_exc()
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

            # Fetch daily data separately for accurate previous close
            # (15-min bars don't give correct settlement/close for futures)
            prev_close_map = {}
            try:
                daily = yf.download(self.tickers, period='5d', interval='1d',
                                    progress=False, threads=False, ignore_tz=True)
                if daily is not None and not daily.empty:
                    if isinstance(daily.columns, pd.MultiIndex):
                        daily_close = daily['Close']
                    elif 'Close' in daily.columns:
                        daily_close = daily[['Close']]
                        if len(self.tickers) == 1:
                            daily_close.columns = [self.tickers[0]]
                    else:
                        daily_close = daily
                    today = pd.Timestamp.now().normalize()
                    for tk in self.tickers:
                        if tk in daily_close.columns:
                            dc = daily_close[tk].dropna()
                            if len(dc) >= 2:
                                # Om senaste datumet i daglig data ÄR idag → prev_close = näst sista
                                # Om senaste datumet < idag → prev_close = sista (den senaste stängningen)
                                last_date = pd.Timestamp(dc.index[-1]).normalize()
                                if last_date >= today:
                                    prev_close_map[tk] = float(dc.iloc[-2])
                                else:
                                    prev_close_map[tk] = float(dc.iloc[-1])
            except Exception as e:
                print(f"[MarketWatch] Daily download for prev close failed: {e}")

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

                    # Use daily close data for previous close (correct for
                    # futures where 15-min electronic session close ≠ settlement)
                    prev_close = prev_close_map.get(ticker)
                    if prev_close and prev_close > 0 and math.isfinite(prev_close):
                        change = round(last_price - prev_close, 4)
                        change_pct = round((last_price - prev_close) / prev_close * 100, 2)
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

            # MarketWatch: processed all items OK
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
            print(f"[Intraday] Starting OHLC fetch for {len(self.tickers)} tickers...")

            # period='5d' ger data även för stängda marknader (senaste handelsdagar)
            # threads=False to avoid deadlock on Windows (yfinance bug)
            data = yf.download(
                self.tickers, period='5d', interval='15m',
                progress=False, threads=False, ignore_tz=True,
            )
            if data.empty:
                print("[Intraday] No data returned from yfinance")
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

            failed_tickers = [t for t in self.tickers if t not in out]
            print(f"[Intraday] Done: {len(out)}/{len(self.tickers)} tickers OK, {len(failed_tickers)} failed")
            if failed_tickers:
                print(f"[Intraday] Failed tickers: {failed_tickers[:20]}")
                if len(failed_tickers) > 20:
                    print(f"[Intraday] ... and {len(failed_tickers) - 20} more")
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
                        # WS connected OK
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
        except RuntimeError:
            # Expected when loop.stop() is called from stop()
            pass
        except Exception as e:
            if not self._stopped:
                print(f"[WS] AsyncWebSocket error: {e}")
                self.error.emit(str(e))
        finally:
            # Rensa pågående asyncio tasks (t.ex. websockets keepalive) före stängning
            try:
                if self._loop is not None and not self._loop.is_closed():
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.close()
            except Exception:
                pass
            self._loop = None
            pass  # WS stopped

    def stop(self):
        """Stäng AsyncWebSocket-anslutningen."""
        self._stopped = True
        loop = self._loop
        ws = self._ws
        if ws is not None and loop is not None and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(ws.close(), loop)
            except Exception:
                pass
            # Give ws.close() a moment, then force-stop the event loop
            import threading
            def _force_stop():
                import time as _time
                _time.sleep(2)
                try:
                    if loop.is_running():
                        loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass
            threading.Thread(target=_force_stop, daemon=True).start()
        elif loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

    def add_tickers(self, new_tickers: list):
        """Dynamically subscribe additional tickers to the active WebSocket."""
        added = [t for t in new_tickers if t not in self.tickers]
        if not added:
            print(f"[WS] add_tickers: all {len(new_tickers)} tickers already subscribed")
            return
        self.tickers.extend(added)
        print(f"[WS] add_tickers: subscribing {len(added)} new tickers: {', '.join(added[:15])}{'...' if len(added) > 15 else ''}")
        if self._ws is not None and self._loop is not None and self._loop.is_running():
            async def _subscribe():
                try:
                    if self._ws is not None:
                        await self._ws.subscribe(added)
                        print(f"[WS] add_tickers: async subscribe complete for {len(added)} tickers")
                except Exception as e:
                    print(f"[WS] add_tickers error: {e}")
            asyncio.run_coroutine_threadsafe(_subscribe(), self._loop)
        else:
            print(f"[WS] add_tickers: WS not connected, {len(added)} tickers queued for next connect")




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

            print(f"[Volatility] Columns in close: {list(close.columns)}, rows: {len(close)}")

            # Volatility download OK

            # Fallback: ladda ner saknade tickers individuellt
            missing = [t for t in self.tickers if t not in close.columns]
            if missing:
                print(f"[Volatility] Missing from batch download: {missing}, fetching individually...")
                for ticker in missing:
                    try:
                        self.status_message.emit(f"Fetching {ticker} individually...")
                        t = yf.Ticker(ticker)
                        hist = t.history(period=self.period, interval="1d")
                        if hist is not None and not hist.empty and 'Close' in hist.columns:
                            close[ticker] = hist['Close']
                            print(f"[Volatility] {ticker}: got {len(hist)} rows individually")
                        else:
                            print(f"[Volatility] {ticker}: no data from individual fetch (hist={'empty' if hist is None or hist.empty else f'{len(hist)} rows, cols={list(hist.columns)}'})")
                    except Exception as e:
                        print(f"[Volatility] {ticker}: download failed: {e}")

            # Volatility data ready
            self.result.emit(close)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class StaleInstrumentRefreshWorker(QObject):
    """Background worker: fetches current prices for instruments that WS doesn't update.

    Uses yf.download to get latest prices for stale tickers and emits results
    so the main thread can update the treemap cache.
    """
    result = Signal(dict)   # {ticker: {'price': float, 'change': float, 'change_pct': float}}
    finished = Signal()

    def __init__(self, tickers: list, daily_prev_close: dict = None):
        super().__init__()
        self.tickers = list(tickers)
        self.daily_prev_close = daily_prev_close or {}

    @Slot()
    def run(self):
        try:
            import yfinance as yf
            from datetime import date as _date
            data = {}
            if not self.tickers:
                return

            # Use period='2d' with interval='1d' to get the previous session close
            # and the last traded price. This ensures closed markets show yesterday's
            # session change (close vs prior close) instead of 0%.
            daily_df = yf.download(self.tickers, period='5d', interval='1d',
                                   progress=False, threads=False,
                                   ignore_tz=True, multi_level_index=False)
            if daily_df is None or daily_df.empty:
                return
            if len(self.tickers) == 1:
                daily_close = daily_df[['Close']].rename(columns={'Close': self.tickers[0]}) if 'Close' in daily_df.columns else daily_df
            else:
                daily_close = daily_df['Close'] if 'Close' in daily_df.columns else daily_df

            today = _date.today()
            for ticker in self.tickers:
                if ticker not in daily_close.columns:
                    continue
                series = daily_close[ticker].dropna()
                if len(series) < 2:
                    continue
                last_price = float(series.iloc[-1])
                if last_price <= 0:
                    continue

                # Determine prev_close: if last bar is today, use iloc[-2];
                # otherwise last bar IS the latest close, use iloc[-2] as prev
                last_dt = series.index[-1]
                last_d = last_dt.date() if hasattr(last_dt, 'date') else last_dt
                if last_d >= today:
                    # Today's bar exists → price = today's close, prev = yesterday
                    prev_close = float(series.iloc[-2])
                else:
                    # No bar for today → price = last close, prev = day before
                    prev_close = float(series.iloc[-2])

                if prev_close > 0:
                    change = last_price - prev_close
                    change_pct = (change / prev_close) * 100
                else:
                    change = 0
                    change_pct = 0

                data[ticker] = {
                    'price': round(last_price, 4),
                    'change': round(change, 4),
                    'change_pct': round(change_pct, 2),
                }
            if data:
                self.result.emit(data)
        except Exception as e:
            traceback.print_exc()
        finally:
            self.finished.emit()


class FuturesImpliedWorker(QObject):
    """Hämtar futures-pris via yf.download och beräknar implied open (futures daglig %)."""
    result = Signal(dict)   # {futures_ticker: {'price': float, 'change_pct': float}, ...}
    finished = Signal()

    def __init__(self, tickers: list, daily_prev_close: dict = None):
        super().__init__()
        self.tickers = list(tickers)
        self.daily_prev_close = daily_prev_close or {}

    @Slot()
    def run(self):
        try:
            import yfinance as yf
            data = {}
            # Batch-download senaste minutdatan för alla futures
            df = yf.download(self.tickers, period='1d', interval='1m',
                             prepost=True, progress=False, threads=False,
                             ignore_tz=True, multi_level_index=False)
            if df is not None and not df.empty:
                if len(self.tickers) == 1:
                    close = df[['Close']].rename(columns={'Close': self.tickers[0]}) if 'Close' in df.columns else df
                else:
                    close = df['Close'] if 'Close' in df.columns else df

                for ticker in self.tickers:
                    if ticker not in close.columns:
                        continue
                    series = close[ticker].dropna()
                    if series.empty:
                        continue
                    last_price = float(series.iloc[-1])
                    if last_price <= 0:
                        continue

                    # Use futures' own prev close (from batch daily download)
                    prev_close = self.daily_prev_close.get(ticker, 0)
                    if prev_close <= 0:
                        # Fallback: per-ticker fast_info
                        try:
                            t = yf.Ticker(ticker)
                            prev_close = float(t.fast_info.get('previousClose', 0)
                                               or t.fast_info.get('previous_close', 0) or 0)
                        except Exception:
                            prev_close = 0
                    if prev_close > 0:
                        pct = ((last_price - prev_close) / prev_close) * 100
                    elif len(series) >= 2:
                        first_price = float(series.iloc[0])
                        pct = ((last_price - first_price) / first_price) * 100 if first_price > 0 else 0
                    else:
                        continue
                    data[ticker] = {'price': last_price, 'change_pct': round(pct, 2)}
            if data:
                self.result.emit(data)
        except Exception as e:
            print(f"[Futures] Fetch error: {e}")
            traceback.print_exc()
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


class MarkovBatchWorker(QObject):
    """Worker som kör Markov-analys på flera tickers sekventiellt."""
    finished = Signal()
    result = Signal(object)   # dict: {ticker: MarkovResult}
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, tickers: list):
        super().__init__()
        self.tickers = tickers

    @Slot()
    def run(self):
        try:
            results = {}
            total = len(self.tickers)
            analyzer = MarkovChainAnalyzer()
            for i, ticker in enumerate(self.tickers):
                pct = int((i / total) * 100)
                self.progress.emit(pct, f"Markov: {ticker} ({i+1}/{total})")
                print(f"[SCANNER] Markov batch: Analyzing {ticker} ({i+1}/{total})...")
                try:
                    res = analyzer.analyze(ticker)
                    results[ticker] = res
                    print(f"[SCANNER] Markov batch: {ticker} OK — state={res.current_state}, "
                          f"E[r]={res.expected_return:+.2%}")
                except Exception as e:
                    print(f"[SCANNER] Markov batch: {ticker} FAILED — {e}")
            self.progress.emit(100, f"Markov: {len(results)}/{total} complete")
            self.result.emit(results)
        except Exception as e:
            print(f"[SCANNER] Markov batch ERROR: {e}")
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class EPSMeanReversionWorker(QObject):
    """Worker thread for EPS mean reversion analysis (fetches own EPS data)."""
    finished = Signal()
    result = Signal(object)   # (data_dict, screening_df)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, tickers: list, delay: int = 0):
        super().__init__()
        self.tickers = tickers
        self._delay = delay  # sekunder att vänta innan start (undviker yfinance concurrency)

    @Slot()
    def run(self):
        import socket
        import time as _time
        old_timeout = socket.getdefaulttimeout()
        try:
            # Sätt socket timeout för att undvika att yfinance hänger sig
            socket.setdefaulttimeout(30)

            print(f"[EPS-WORKER] Starting EPS analysis for {len(self.tickers)} tickers...")
            print(f"[EPS-WORKER] Socket timeout set to 30s")
            self.progress.emit(10, f"Fetching EPS data for {len(self.tickers)} tickers...")

            t0 = _time.time()
            data = fetch_eps_and_price(self.tickers, period='max')
            t1 = _time.time()
            print(f"[EPS-WORKER] fetch_eps_and_price completed in {t1-t0:.1f}s — {len(data)} tickers OK")

            if not data:
                self.error.emit("No EPS data returned — yfinance may be unavailable")
                return

            self.progress.emit(60, f"Screening {len(data)} tickers...")
            df = eps_screen_tickers(data)
            self.progress.emit(90, "Computing P/E analytics...")

            # Add P/E analysis for each ticker
            pe_analyses = {}
            for ticker, d in data.items():
                pe_result = analyze_pe_mean_reversion(d['pe_ratio'])
                if pe_result:
                    pe_analyses[ticker] = pe_result

            self.progress.emit(100, "EPS analysis complete")
            print(f"[EPS-WORKER] Done — {len(data)} tickers, {len(pe_analyses)} P/E analyses, total {_time.time()-t0:.1f}s")
            self.result.emit((data, df, pe_analyses))
        except Exception as e:
            print(f"[EPS-WORKER] EXCEPTION: {e}")
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            socket.setdefaulttimeout(old_timeout)
            self.finished.emit()


class SqueezeWorker(QObject):
    """Worker thread for TTM Squeeze analysis — downloads OHLCV for OMXS Large Cap."""
    finished = Signal()
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, tickers: list):
        super().__init__()
        self.tickers = tickers

    @Slot()
    def run(self):
        try:
            import yfinance as yf
            import time as _time
            import socket
            socket.setdefaulttimeout(30)

            self.progress.emit(5, f"Downloading OHLCV for {len(self.tickers)} tickers...")

            # Batch-download i grupper om 50 (samma mönster som pairs_engine)
            batch_size = 50
            all_close, all_high, all_low, all_volume = {}, {}, {}, {}
            total_batches = (len(self.tickers) + batch_size - 1) // batch_size

            for batch_num, i in enumerate(range(0, len(self.tickers), batch_size), 1):
                batch = self.tickers[i:i + batch_size]
                pct = int(5 + 50 * batch_num / max(total_batches, 1))
                self.progress.emit(pct, f"Downloading batch {batch_num}/{total_batches}...")
                try:
                    data = yf.download(
                        batch, period='2y', interval='1d',
                        auto_adjust=True, progress=False,
                        threads=False, ignore_tz=True,
                        timeout=30,
                    )
                    if data is None or data.empty:
                        print(f"[SQUEEZE] Batch {batch_num}: empty response")
                        continue

                    # Extrahera OHLCV per ticker
                    for field, target in [('Close', all_close), ('High', all_high),
                                          ('Low', all_low), ('Volume', all_volume)]:
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                if field in data.columns.get_level_values(0):
                                    sub = data[field]
                                else:
                                    continue
                            elif field in data.columns:
                                sub = data[field]
                            else:
                                continue

                            if isinstance(sub, pd.Series):
                                # Enstaka ticker → Series
                                ticker_sym = batch[0]
                                if sub.notna().sum() > 50:
                                    target[ticker_sym] = sub
                            else:
                                for col in sub.columns:
                                    s = sub[col].dropna()
                                    if len(s) > 50:
                                        target[col] = s
                        except Exception as e:
                            print(f"[SQUEEZE] Field {field} extraction error: {e}")

                    n_ok = len(all_close)
                    print(f"[SQUEEZE] Batch {batch_num}/{total_batches}: "
                          f"{n_ok} tickers total so far")

                except Exception as e:
                    print(f"[SQUEEZE] Batch {batch_num} download error: {e}")
                    import traceback as _tb
                    _tb.print_exc()

                if batch_num < total_batches:
                    _time.sleep(1)  # Undvik rate limiting

            print(f"[SQUEEZE] Download complete: {len(all_close)} close, "
                  f"{len(all_high)} high, {len(all_low)} low, {len(all_volume)} volume")

            if not all_close:
                self.error.emit(f"No data downloaded for {len(self.tickers)} tickers")
                return

            price_data = pd.DataFrame(all_close)
            high_data = pd.DataFrame(all_high) if all_high else None
            low_data = pd.DataFrame(all_low) if all_low else None
            volume_data = pd.DataFrame(all_volume) if all_volume else None

            self.progress.emit(60, f"Analyzing {len(price_data.columns)} tickers...")

            analyzer = SqueezeAnalyzer()
            res = analyzer.scan(
                price_data,
                high_data=high_data,
                low_data=low_data,
                volume_data=volume_data,
                progress_callback=self.progress.emit,
            )
            self.result.emit(res)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class OptionsWorker(QObject):
    """Worker thread: hamtar options/straddle-data fran Avanza for squeeze-tickers.
    Parallelliserad: alla tickers hamtas samtidigt med ThreadPoolExecutor.
    """
    finished = Signal()
    result = Signal(object)   # dict: {yf_ticker: straddle_summary_dict, ...}
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, yf_tickers: list, last_prices: dict = None):
        super().__init__()
        self.yf_tickers = yf_tickers
        self.last_prices = last_prices or {}  # yf_ticker -> float (spot)

    @Slot()
    def run(self):
        try:
            if not OPTIONS_AVAILABLE:
                self.error.emit("nasdaq_options.py not available")
                return

            from concurrent.futures import ThreadPoolExecutor, as_completed
            from requests.adapters import HTTPAdapter
            import time as _time

            session = requests.Session()
            # Utoka connection pool for parallella anrop
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
            session.mount('https://', adapter)

            # Steg 1: Mappa tickers till orderBookIDs
            self.progress.emit(10, f"Mapping {len(self.yf_tickers)} tickers...")
            mapping = build_ticker_mapping(self.yf_tickers, session)
            self.progress.emit(30, f"Mapped {len(mapping)}/{len(self.yf_tickers)} tickers")

            # Steg 2: Hamta straddle-data for alla tickers parallellt
            results = {}
            n_mapped = len(mapping)
            completed = [0]  # mutable for closure
            t0 = _time.time()

            def _fetch_ticker(yf_ticker, orderbook_id):
                try:
                    spot = self.last_prices.get(yf_ticker)
                    return yf_ticker, get_straddle_summary(
                        orderbook_id, spot=spot, session=session, max_workers=6)
                except Exception as e:
                    print(f"[OPTIONS] Error for {yf_ticker}: {e}")
                    return yf_ticker, {'error': str(e)}

            # Max 3 tickers parallellt (varje ticker gor ~8 parallella anrop internt)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {
                    pool.submit(_fetch_ticker, yf, obid): yf
                    for yf, obid in mapping.items()
                }
                for fut in as_completed(futures):
                    yf_ticker, summary = fut.result()
                    results[yf_ticker] = summary
                    completed[0] += 1
                    pct = 30 + int(60 * completed[0] / max(n_mapped, 1))
                    self.progress.emit(pct, f"Options {completed[0]}/{n_mapped}: {yf_ticker}")

            elapsed = _time.time() - t0
            print(f"[OPTIONS] All {n_mapped} tickers fetched in {elapsed:.1f}s")
            self.progress.emit(95, f"Options data for {len(results)} tickers")
            self.result.emit(results)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class VolAnalyticsWorker(QObject):
    """Worker: Hamtar 5y OHLC och kor volatilitetsanalys (post-squeeze, YZ, GARCH)."""
    finished = Signal()
    result = Signal(object)   # dict: {yf_ticker: vol_analytics_dict}
    error = Signal(str)
    progress = Signal(int, str)

    def __init__(self, yf_tickers: list, forward_windows: list = None):
        super().__init__()
        self.yf_tickers = yf_tickers
        self.forward_windows = forward_windows or [30, 60, 90, 120, 180]

    @Slot()
    def run(self):
        try:
            import yfinance as yf

            self.progress.emit(10, "Downloading 5y OHLC data...")
            # Batch-download alla tickers
            tickers_str = " ".join(self.yf_tickers)
            df = yf.download(tickers_str, period='5y', progress=False, threads=True)

            if df.empty:
                self.error.emit("No OHLC data downloaded")
                return

            results = {}
            n = len(self.yf_tickers)
            for idx, ticker in enumerate(self.yf_tickers):
                pct = 10 + int(80 * (idx + 1) / max(n, 1))
                self.progress.emit(pct, f"Vol analytics {idx+1}/{n}: {ticker}...")
                try:
                    # Hantera bade single-ticker och multi-ticker format
                    if len(self.yf_tickers) == 1:
                        if isinstance(df.columns, pd.MultiIndex):
                            t_open = df['Open'].iloc[:, 0]
                            t_high = df['High'].iloc[:, 0]
                            t_low = df['Low'].iloc[:, 0]
                            t_close = df['Close'].iloc[:, 0]
                        else:
                            t_open = df['Open']
                            t_high = df['High']
                            t_low = df['Low']
                            t_close = df['Close']
                    else:
                        t_open = df['Open'][ticker].dropna()
                        t_high = df['High'][ticker].dropna()
                        t_low = df['Low'][ticker].dropna()
                        t_close = df['Close'][ticker].dropna()

                    if len(t_close) < 200:
                        continue

                    result = vol_analyze_ticker(
                        t_open, t_high, t_low, t_close,
                        forward_windows=self.forward_windows)
                    results[ticker] = result
                except Exception as e:
                    print(f"[VOL] Error for {ticker}: {e}")

            self.progress.emit(95, f"Vol analytics for {len(results)} tickers")
            self.result.emit(results)
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
            'HONG KONG': ['ASIA'],
            'TOKYO': ['ASIA'],
            'SYDNEY': ['OCEANIA'],
        }
        
        open_regions = set()
        for clock in self._market_clocks:
            is_open = clock.is_open()
            regions = CLOCK_TO_REGIONS.get(clock.city, [])
            if is_open:
                open_regions.update(regions)
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

    def __init__(self, label: str, value: str = "-", tooltip: str = "", parent=None):
        super().__init__(parent)
        if tooltip:
            self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {COLORS['bg_elevated']}, stop:1 {COLORS['bg_card']});
                border: none;
                border-radius: 4px;
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

    def set_title(self, title: str):
        """Update the card title/label."""
        self.label_widget.setText(title)


class VolatilitySparkline(QWidget):
    """Sparkline chart for volatility cards with median line - dynamically scalable."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []
        self.color = QColor(COLORS['accent'])
        self.median_value = None
        self._base_height = 35
        self.setMinimumHeight(35)
        self.setMinimumWidth(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: transparent; border: none;")
    
    def scale_to(self, scale: float):
        """Scale the sparkline height based on window size."""
        height = max(25, int(self._base_height * scale))
        self.setMinimumHeight(height)
    
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
        self.current_value = None
        self.current_percentile = None
        self._history_data = []  # Store historical data for sparkline
        
        self.setMinimumHeight(165)  # Increased to accommodate sparkline
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
        
        # Row 4: Sparkline chart — gets stretch so it fills available space
        self.sparkline = VolatilitySparkline()
        layout.addWidget(self.sparkline, 1)

        # Row 5: Description
        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; font-style: italic; background: transparent; border: none;")
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)
    
    def update_data(self, value: float, change_pct: float, percentile: float, 
                    median: float, mode: float, description: str,
                    history: list = None):
        """Update card with new data.
        
        Args:
            history: Optional list of historical values for sparkline (last ~60 days)
        """
        # Cache values for programmatic access (email summary etc.)
        self.current_value = value
        self.current_percentile = percentile

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
        # NewsCache saved OK
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
    
    def _fetch_ticker_news(self, ticker_symbol: str, cutoff_ts: float, max_per_ticker: int = 3) -> list:
        """Fetch news for a single ticker. Returns max_per_ticker most recent items."""
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

        # Keep only the most recent items per ticker
        news_items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return news_items[:max_per_ticker]
    
    def run(self):
        """Fetch news for all tickers from CSV file using parallel execution."""
        try:
            # Load existing cache
            cached_news = load_news_cache()
            cached_ids = {n.get('id') for n in cached_news if n.get('id')}
            
            # Load tickers from CSV
            tickers = []
            if os.path.exists(self.csv_path):
                try:
                    df = pd.read_csv(self.csv_path, sep=';', encoding='utf-8-sig')
                    if 'Ticker' in df.columns:
                        tickers = df['Ticker'].dropna().tolist()
                except Exception as e:
                    print(f"[NewsFeed] Error reading CSV: {e}")
            
            if not tickers:
                # Fallback to some default tickers
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
                pass  # Using default tickers
            
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
        layout.setSpacing(0)

        # refresh_btn behålls som attribut för bakåtkompatibilitet (visas ej här)
        self.refresh_btn = QPushButton()
        self._base_btn_size = 20

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
# CALENDAR PANEL — Worker, event rows, tab widget, right-panel container
# ============================================================================

class CustomCalendars:
    """Utökad kalender som hämtar data från flera regioner via yfinance.Calendars."""

    # Visningsetikett per region-kod
    _LABELS = {'se': 'SE', 'us': 'US'}

    def __init__(self):
        import yfinance as yf
        self._cal = yf.Calendars()

    def _label(self, region: str) -> str:
        return self._LABELS.get(region.lower(), region.upper())

    def _fmt_start(self, start) -> str:
        """Konverterar start-datum till sträng som CalendarQuery accepterar."""
        if start is None:
            return self._cal._start
        if hasattr(start, 'strftime'):
            return start.strftime('%Y-%m-%d')
        return str(start)

    # ------------------------------------------------------------------
    # Minsta market cap för SE large cap (ca 10 miljarder SEK ≈ 1B USD)
    _SE_LARGE_CAP_USD = 1_000_000_000

    def get_earnings_multi_region(self, regions=("us", "se"), start=None, limit=100) -> pd.DataFrame:
        """Hämtar earnings för samtliga regioner och slår ihop till en DataFrame.

        US: filter_most_active (top-200 mest handlade), alla market caps.
        SE: bara large cap (>1B USD market cap).
        """
        from yfinance.calendars import CalendarQuery

        _start = self._fmt_start(start)
        _end   = self._cal._end

        frames = []
        for region in regions:
            try:
                conditions = [
                    CalendarQuery("eq", ["region", region.lower()]),
                    CalendarQuery("or", [
                        CalendarQuery("eq", ["eventtype", "EAD"]),
                        CalendarQuery("eq", ["eventtype", "ERA"]),
                    ]),
                    CalendarQuery("gte", ["startdatetime", _start]),
                    CalendarQuery("lte", ["startdatetime", _end]),
                    CalendarQuery("gt",  ["intradaymarketcap", 1]),
                ]
                # SE: kräv large cap (>1B USD)
                if region.lower() == 'se':
                    conditions.append(
                        CalendarQuery("gte", ["intradaymarketcap", self._SE_LARGE_CAP_USD])
                    )
                query = CalendarQuery("and", conditions)
                # US: använd filter_most_active via standard-API
                if region.lower() == 'us':
                    try:
                        df = self._cal.get_earnings_calendar(
                            filter_most_active=True, limit=limit)
                    except Exception:
                        df = self._cal._get_data(
                            calendar_type="sp_earnings", query=query, limit=limit)
                else:
                    df = self._cal._get_data(
                        calendar_type="sp_earnings", query=query, limit=limit)
                if df is not None and not df.empty:
                    if 'Marketcap' in df.columns:
                        df = df[df['Marketcap'].notna() & (df['Marketcap'] > 0)]
                    df = df.copy()
                    df['Market'] = self._label(region)
                    frames.append(df)
            except Exception as e:
                print(f"[CustomCalendars] Earnings {region}: {e}")

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        # Dedup: samma bolag kan ha både EAD och ERA event — behåll raden med mest data
        if 'Company' in result.columns:
            # Räkna icke-NaN-kolumner per rad som prioritet
            result['_fill'] = result.notna().sum(axis=1)
            result = result.sort_values('_fill', ascending=False).drop_duplicates(
                subset=['Company', 'Market'], keep='first').drop(columns='_fill')
            result = result.reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    def get_economic_events_multi_region(self, regions=("us", "se"), start=None, limit=100) -> pd.DataFrame:
        """Hämtar ekonomiska händelser och filtrerar på region-koder."""
        from yfinance.calendars import CalendarQuery

        _start = self._fmt_start(start)
        _end   = self._cal._end

        try:
            query = CalendarQuery("and", [
                CalendarQuery("gte", ["startdatetime", _start]),
                CalendarQuery("lte", ["startdatetime", _end]),
            ])
            df = self._cal._get_data(calendar_type="economic_event", query=query, limit=limit)
            if df is not None and not df.empty:
                upper = [r.upper() for r in regions]
                # 'Region' är det omdöpta 'country_code'-fältet
                if 'Region' in df.columns:
                    df = df[df['Region'].str.upper().isin(upper)].copy()
                return df
        except Exception as e:
            print(f"[CustomCalendars] Economic events: {e}")

        return pd.DataFrame()


class CalendarWorker(QObject):
    """Hämtar kalenderdata (earnings/ipos/splits/economic) via yfinance.Calendars i bakgrundstråd."""

    finished = Signal()
    result = Signal(str, list)   # (tab_type, list-of-dicts)
    error = Signal(str, str)     # (tab_type, error_msg)

    def __init__(self, tab_type: str):
        super().__init__()
        self.tab_type = tab_type

    def run(self):
        try:
            import yfinance as yf

            df = None
            if self.tab_type == 'earnings':
                try:
                    df = CustomCalendars().get_earnings_multi_region(
                        regions=("us", "se"))
                except Exception as e:
                    print(f"[CalendarWorker] CustomCalendars misslyckades, använder standard: {e}")
                    df = yf.Calendars().get_earnings_calendar()
            elif self.tab_type == 'economic':
                try:
                    df = CustomCalendars().get_economic_events_multi_region(regions=("us", "se"))
                except Exception as e:
                    print(f"[CalendarWorker] Economic events multi-region misslyckades: {e}")
                    df = yf.Calendars().get_economic_events_calendar()
            elif self.tab_type == 'ipo':
                df = yf.Calendars().get_ipo_info_calendar(limit=100)
            elif self.tab_type == 'splits':
                df = yf.Calendars().get_splits_calendar(limit=100)

            if df is not None and not df.empty:
                # Flytta index till kolumn (datum är ofta index i yfinance-DataFrames)
                if df.index.name or hasattr(df.index, 'name'):
                    df = df.reset_index()
                # Konvertera till list-of-dicts; NaN → None
                records = []
                for rec in df.to_dict('records'):
                    clean = {}
                    for k, v in rec.items():
                        if isinstance(v, float) and (v != v):  # NaN-check
                            clean[k] = None
                        else:
                            clean[k] = v
                    records.append(clean)
                self.result.emit(self.tab_type, records)
            else:
                self.result.emit(self.tab_type, [])
        except Exception as e:
            self.error.emit(self.tab_type, str(e))
        finally:
            self.finished.emit()


class EventsCalendarWorker(QObject):
    """Hämtar veckoöversikt av kalenderdata (earnings/ipo/economic/splits) via yfinance."""

    finished = Signal()
    result = Signal(dict)   # {date_str: {earnings: N, ipo: N, economic: N, splits: N}}
    error = Signal(str)

    def __init__(self, start_date, end_date):
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        try:
            import yfinance as yf
            import pandas as pd
            from collections import defaultdict

            cal = yf.Calendars()
            counts = defaultdict(lambda: {'earnings': 0, 'ipo': 0, 'economic': 0, 'splits': 0})

            # Använd samma datakällor som CalendarWorker för konsistens
            data_frames = {}
            # Earnings — samma som EARNINGS-fliken
            try:
                data_frames['earnings'] = CustomCalendars().get_earnings_multi_region(
                    regions=("us", "se"))
            except Exception:
                try:
                    data_frames['earnings'] = cal.get_earnings_calendar()
                except Exception as e:
                    print(f"[EventsCalendarWorker] earnings: {e}")
            # Economic — samma som EVENTS-fliken
            try:
                data_frames['economic'] = CustomCalendars().get_economic_events_multi_region(
                    regions=("us", "se"))
            except Exception:
                try:
                    data_frames['economic'] = cal.get_economic_events_calendar()
                except Exception as e:
                    print(f"[EventsCalendarWorker] economic: {e}")
            # IPO
            try:
                data_frames['ipo'] = cal.get_ipo_info_calendar(limit=200)
            except Exception as e:
                print(f"[EventsCalendarWorker] ipo: {e}")
            # Splits
            try:
                data_frames['splits'] = cal.get_splits_calendar(limit=200)
            except Exception as e:
                print(f"[EventsCalendarWorker] splits: {e}")

            for cat_key, df in data_frames.items():
                if df is None or df.empty:
                    print(f"[EventsCalendarWorker] {cat_key}: tom DataFrame")
                    continue
                df = df.reset_index()
                # Hitta datumkolumn
                date_col = None
                date_keywords = ['date', 'start', 'time', 'day']
                for c in df.columns:
                    c_lower = str(c).lower()
                    if any(kw in c_lower for kw in date_keywords):
                        date_col = c
                        break
                if date_col is None:
                    for c in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[c]):
                            date_col = c
                            break
                if date_col is None and len(df.columns) > 0:
                    date_col = df.columns[0]
                if date_col is None:
                    continue
                matched = 0
                for val in df[date_col]:
                    try:
                        dt = pd.Timestamp(val).date()
                        if self.start_date <= dt <= self.end_date:
                            counts[dt.isoformat()][cat_key] += 1
                            matched += 1
                    except Exception:
                        continue
                print(f"[EventsCalendarWorker] {cat_key}: {len(df)} rader, {matched} i veckan")

            self.result.emit(dict(counts))
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class EventsCalendarWidget(QWidget):
    """Institutional-grade veckoöversikt med antal händelser per dag."""

    CAT_COLORS = {
        'economic': ('#f59e0b', 'Economic'),
        'ipo':      ('#a855f7', 'IPO'),
        'earnings': ('#d4a574', 'Earnings'),
        'splits':   ('#22d3ee', 'Splits'),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        from datetime import date, timedelta
        today = date.today()
        self._week_start = today - timedelta(days=today.weekday())
        self._worker = None
        self._thread = None
        self._data = {}

        self.setStyleSheet(f"background: {COLORS['bg_dark']};")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 6)
        main_layout.setSpacing(0)

        # ── Header row ──
        header = QHBoxLayout()
        header.setSpacing(8)
        title = QLabel("EVENTS CALENDAR")
        title.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 11px; font-weight: 700;"
            f" letter-spacing: 1.2px; font-family: 'JetBrains Mono', monospace;"
        )
        header.addWidget(title)
        header.addStretch()
        self._date_label = QLabel()
        self._date_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px;"
            f" font-family: 'JetBrains Mono', monospace;"
        )
        header.addWidget(self._date_label)
        nav_btn_style = f"""
            QPushButton {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 3px;
                padding: 2px 7px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                background: {COLORS['bg_hover']};
                color: {COLORS['accent']};
                border-color: {COLORS['accent_dark']};
            }}
        """
        prev_btn = QPushButton("\u25C0")
        prev_btn.setFixedSize(24, 20)
        prev_btn.setStyleSheet(nav_btn_style)
        prev_btn.clicked.connect(self._prev_week)
        next_btn = QPushButton("\u25B6")
        next_btn.setFixedSize(24, 20)
        next_btn.setStyleSheet(nav_btn_style)
        next_btn.clicked.connect(self._next_week)
        header.addWidget(prev_btn)
        header.addWidget(next_btn)
        main_layout.addLayout(header)
        main_layout.addSpacing(6)

        # ── Separator ──
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {COLORS['border_default']};")
        main_layout.addWidget(sep)
        main_layout.addSpacing(6)

        # ── Legend row ──
        legend = QHBoxLayout()
        legend.setSpacing(12)
        for cat_key, (color, display_name) in self.CAT_COLORS.items():
            lbl = QLabel(f"\u25CF {display_name}")
            lbl.setStyleSheet(
                f"color: {color}; font-size: 10px; font-weight: 600;"
                f" font-family: 'JetBrains Mono', monospace;"
            )
            legend.addWidget(lbl)
        legend.addStretch()
        main_layout.addLayout(legend)
        main_layout.addSpacing(6)

        # ── Day rows ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(scroll_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)

        self._day_rows = []
        for i in range(7):
            row = QFrame()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 6, 8, 6)
            rl.setSpacing(0)
            # Dagnamn
            day_label = QLabel()
            day_label.setFixedWidth(90)
            day_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-size: 11px; font-weight: 700;"
                f" font-family: 'JetBrains Mono', monospace;"
            )
            rl.addWidget(day_label)
            # Badges — varje kategori med färgad bakgrund (chip-stil)
            badges_layout = QHBoxLayout()
            badges_layout.setSpacing(4)
            cat_labels = {}
            for cat_key, (color, display_name) in self.CAT_COLORS.items():
                badge = QLabel()
                badge.setAlignment(Qt.AlignCenter)
                badge.setFixedHeight(20)
                badge.setMinimumWidth(32)
                badge.setStyleSheet(f"""
                    QLabel {{
                        background: rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.15);
                        color: {color};
                        font-size: 10px;
                        font-weight: 700;
                        font-family: 'JetBrains Mono', monospace;
                        border-radius: 3px;
                        padding: 1px 6px;
                    }}
                """)
                badge.hide()
                badges_layout.addWidget(badge)
                cat_labels[cat_key] = badge
            badges_layout.addStretch()
            rl.addLayout(badges_layout)
            self._day_rows.append((row, day_label, cat_labels))
            self._rows_layout.addWidget(row)

        self._rows_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 1)

        self._update_header()
        self._load_data()

    def _update_header(self):
        from datetime import timedelta
        end = self._week_start + timedelta(days=6)
        self._date_label.setText(
            f"{self._week_start.strftime('%b %d')} \u2013 {end.strftime('%b %d, %Y')}"
        )

    def _prev_week(self):
        from datetime import timedelta
        self._week_start -= timedelta(days=7)
        self._update_header()
        self._load_data()

    def _next_week(self):
        from datetime import timedelta
        self._week_start += timedelta(days=7)
        self._update_header()
        self._load_data()

    def _load_data(self):
        from datetime import timedelta
        if self._thread and self._thread.isRunning():
            return
        end = self._week_start + timedelta(days=6)
        self._thread = QThread()
        self._worker = EventsCalendarWorker(self._week_start, end)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.result.connect(self._on_data)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()
        self._render_days({})

    def _on_data(self, data):
        self._data = data
        self._render_days(data)

    def _on_error(self, msg):
        print(f"[EventsCalendar] Error: {msg}")
        self._render_days({})

    def _render_days(self, data):
        from datetime import timedelta, date
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        today = date.today()
        # Keep _week_start current if dashboard runs across midnight
        expected_start = today - timedelta(days=today.weekday())
        if self._week_start == expected_start - timedelta(weeks=1):
            self._week_start = expected_start
        for i in range(7):
            dt = self._week_start + timedelta(days=i)
            row, day_label, cat_labels = self._day_rows[i]
            is_today = dt == today
            has_events = dt.isoformat() in data and any(
                data[dt.isoformat()].get(k, 0) > 0 for k in self.CAT_COLORS
            )
            # Styling per dag
            if is_today:
                row.setStyleSheet(f"""
                    QFrame {{
                        background: {COLORS['bg_elevated']};
                        border-left: 3px solid {COLORS['accent']};
                        border-radius: 3px;
                    }}
                """)
                day_label.setText(f"\u25B8 {day_names[i]}, {dt.strftime('%b %d')}")
            else:
                bg = COLORS['bg_card'] if has_events else 'transparent'
                row.setStyleSheet(f"""
                    QFrame {{
                        background: {bg};
                        border-radius: 3px;
                    }}
                    QFrame:hover {{
                        background: {COLORS['bg_elevated']};
                    }}
                """)
                day_label.setText(f"  {day_names[i]}, {dt.strftime('%b %d')}")
            # Badges
            day_data = data.get(dt.isoformat(), {})
            for cat_key, (color, display_name) in self.CAT_COLORS.items():
                badge = cat_labels[cat_key]
                count = day_data.get(cat_key, 0)
                if count > 0:
                    badge.setText(f"{count}")
                    badge.setToolTip(f"{count} {display_name}")
                    badge.show()
                else:
                    badge.hide()


class CalendarEventDetail(QDialog):
    """Frameless popup-dialog som visar alla fält för ett kalenderevenemang."""

    # Nyckelord som indikerar att värdet är ett stort monetärt tal
    _BIG_NUM_KEYS = {'market cap', 'marketcap', 'revenue', 'volume', 'shares', 'outstanding',
                     'enterprise', 'ebitda', 'assets', 'debt', 'float', 'capitalization'}

    @staticmethod
    def _fmt_value(key: str, value) -> str:
        """Formaterar ett värde, med special-hantering av stora tal och datum."""
        if value is None:
            return ''
        # pandas.NaT har strftime men kraschar — kolla pd.isna() först
        try:
            if pd.isna(value):
                return ''
        except (TypeError, ValueError):
            pass
        if hasattr(value, 'strftime'):
            try:
                return value.strftime('%Y-%m-%d %H:%M') if hasattr(value, 'hour') and value.hour else value.strftime('%Y-%m-%d')
            except (ValueError, OSError):
                return ''
        if isinstance(value, float):
            key_lower = key.lower().replace('_', ' ')
            is_big = any(kw in key_lower for kw in CalendarEventDetail._BIG_NUM_KEYS)
            if is_big or abs(value) >= 1_000_000:
                if abs(value) >= 1e12:
                    return f"{value / 1e12:.2f} T"
                elif abs(value) >= 1e9:
                    return f"{value / 1e9:.2f} B"
                elif abs(value) >= 1e6:
                    return f"{value / 1e6:.2f} M"
                elif abs(value) >= 1e3:
                    return f"{value / 1e3:.2f} K"
            return f"{value:,.4f}" if abs(value) < 1_000 else f"{value:,.2f}"
        return str(value)

    def __init__(self, event_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setStyleSheet(f"""
            QDialog {{
                background: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['accent']};
                border-radius: 6px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(0)

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.setColumnMinimumWidth(0, 110)
        grid.setColumnStretch(1, 1)

        row_idx = 0
        for key, value in event_data.items():
            if value is None:
                continue
            val_str = self._fmt_value(str(key), value)
            if not val_str or val_str.lower() in ('none', 'nan', 'nat', ''):
                continue

            k_lbl = QLabel(str(key).replace('_', ' ').title() + ':')
            k_lbl.setAlignment(Qt.AlignRight | Qt.AlignTop)
            k_lbl.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 600; background: transparent;"
            )

            v_lbl = QLabel(val_str)
            v_lbl.setWordWrap(True)
            v_lbl.setMaximumWidth(240)
            v_lbl.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-size: 11px; background: transparent;"
            )

            grid.addWidget(k_lbl, row_idx, 0)
            grid.addWidget(v_lbl, row_idx, 1)
            row_idx += 1

        if row_idx == 0:
            empty = QLabel("Ingen detaljdata tillgänglig")
            empty.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
            outer.addWidget(empty)
        else:
            outer.addLayout(grid)

        self.adjustSize()

        # Placera nära muspekaren, håll innanför skärmen
        cursor_pos = QCursor.pos()
        screen = QApplication.desktop().availableGeometry(cursor_pos)
        x = min(cursor_pos.x() + 12, screen.right() - self.width() - 10)
        y = min(cursor_pos.y() + 12, screen.bottom() - self.height() - 10)
        self.move(x, y)


class CalendarEventRow(QFrame):
    """Klickbar rad i en kalender-tab. Visar namn, badge-värde och datum."""

    clicked_detail = Signal(dict)

    _COLS = {
        'earnings': {
            'primary': ['Company', 'Symbol'],
            'badge':   ['Market'],   
            'date':    ['Event Start Date'],
            'color':   COLORS.get('accent', '#00d4aa'),
        },
        'ipos': {
            'primary': ['Company', 'Symbol'],
            'badge':   ['Market', 'Price', 'Price From', 'Price To'],
            'date':    ['Date', 'Filing Date'],
            'color':   '#3b82f6',
        },
        'splits': {
            'primary': ['Company', 'Symbol'],
            'badge':   ['Market'],
            'date':    ['Payable On'],
            'color':   '#f59e0b',
        },
        'economic': {
            'primary': ['Event'],
            'badge':   ['Region'],
            'date':    ['Event Time', 'For'],
            'color':   '#ec4899',
        },
    }

    def __init__(self, event_data: dict, tab_type: str, parent=None):
        super().__init__(parent)
        self._data = event_data
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(36)

        cfg = self._COLS.get(tab_type, self._COLS['earnings'])
        accent = cfg['color']

        self.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_card']};
                border: none;
                border-left: 2px solid {COLORS['border_subtle']};
                border-top: 2px solid {accent};
                border-radius: 2px;
            }}
            QFrame:hover {{
                background: {COLORS['bg_hover']};
            }}
        """)

        row_layout = QHBoxLayout(self)
        row_layout.setContentsMargins(8, 4, 8, 4)
        row_layout.setSpacing(8)

        # Primär text — försök konfigurerade kolumner, sedan generisk fallback
        primary_val = self._find(event_data, cfg['primary'])
        if primary_val is None:
            primary_val = self._find_any_label(event_data)
        primary_str = str(primary_val) if primary_val is not None else '—'
        self.primary_lbl = QLabel(primary_str)
        self.primary_lbl.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 12px; background: transparent; border: none;"
        )
        self.primary_lbl.setWordWrap(False)
        self.primary_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.primary_lbl.setToolTip(primary_str)
        row_layout.addWidget(self.primary_lbl, stretch=1)

        # Badge — visas bara om listan inte är tom
        badge_keys = cfg.get('badge', [])
        if badge_keys:
            badge_val = self._find(event_data, badge_keys)
            if badge_val is not None:
                badge_str = f"{badge_val:.2f}" if isinstance(badge_val, float) else str(badge_val)
                badge_str = badge_str[:14]
                badge_lbl = QLabel(badge_str)
                badge_lbl.setStyleSheet(f"""
                    QLabel {{
                        color: {accent};
                        border: 1px solid {accent};
                        font-size: 10px;
                        font-weight: 700;
                        padding: 1px 5px;
                        border-radius: 3px;
                        background: transparent;
                    }}
                """)
                badge_lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
                row_layout.addWidget(badge_lbl)

        # Datum — fast bredd till höger
        date_val = self._find(event_data, cfg['date'])
        if date_val is not None:
            date_str = self._fmt_date(date_val)
            if date_str:
                date_lbl = QLabel(date_str)
                date_lbl.setStyleSheet(
                    f"color: {COLORS['text_muted']}; font-size: 10px; background: transparent; border: none;"
                )
                date_lbl.setMinimumWidth(48)
                date_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                date_lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
                row_layout.addWidget(date_lbl)

    @staticmethod
    def _find(data: dict, keys: list):
        """Hitta första icke-None värde bland möjliga kolumnnamn (skiftlägeskänslig + insensitiv)."""
        if not keys:
            return None
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        lower_map = {dk.lower(): dk for dk in data}
        for k in keys:
            real = lower_map.get(k.lower())
            if real and data[real] is not None:
                return data[real]
        return None

    @staticmethod
    def _find_any_label(data: dict):
        """Fallback: returnerar första strängvärde som ser ut som ett namn/evenemang."""
        # Hoppa över korta koder, siffror och "tekniska" kolumner
        _skip = {'country', 'currency', 'impact', 'importance', 'unit', 'type', 'exchange'}
        for k, v in data.items():
            if k.lower() in _skip:
                continue
            if isinstance(v, str) and len(v) > 3 and not v.replace('.', '').replace('-', '').isnumeric():
                return v
        # Sista utvägen: ta vilket strängvärde som helst
        for v in data.values():
            if isinstance(v, str) and len(v) > 1:
                return v
        return None

    @staticmethod
    def _fmt_date(val) -> str:
        if val is None:
            return ''
        # Hantera pandas.NaT
        try:
            if pd.isna(val):
                return ''
        except (TypeError, ValueError):
            pass
        if hasattr(val, 'strftime'):
            try:
                return val.strftime('%d %b')
            except (ValueError, OSError):
                return ''
        s = str(val)
        if s.lower() in ('nat', 'nan', 'none', ''):
            return ''
        try:
            dt = datetime.fromisoformat(s.split('T')[0].split(' ')[0])
            return dt.strftime('%d %b')
        except Exception:
            return s[:10]

    def mousePressEvent(self, event):
        self.clicked_detail.emit(self._data)
        super().mousePressEvent(event)


class CalendarTabWidget(QWidget):
    """Generisk kalender-tab med lazy loading och scroll-lista av händelser."""

    def __init__(self, tab_type: str, parent=None):
        super().__init__(parent)
        self.tab_type = tab_type
        self._loaded = False
        self._loading = False
        self._worker = None
        self._thread = None
        self._rows: list = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Status/placeholder
        self.status_lbl = QLabel("Välj fliken för att ladda data...")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 16px; background: transparent;"
        )

        # Scroll-area med händelserader
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.cnt_layout = QVBoxLayout(self.container)
        self.cnt_layout.setContentsMargins(2, 2, 2, 2)
        self.cnt_layout.setSpacing(3)
        self.cnt_layout.addWidget(self.status_lbl)
        self.cnt_layout.addStretch()

        scroll.setWidget(self.container)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    def load_if_needed(self):
        """Anropas när fliken aktiveras — hämtar data första gången."""
        if not self._loaded and not self._loading:
            self._start_fetch()

    def refresh(self):
        """Tvinga ny hämtning."""
        if self._loading:
            return
        self._loaded = False
        self._start_fetch()

    # ------------------------------------------------------------------
    def _start_fetch(self):
        self._loading = True
        self.status_lbl.setText("Hämtar data...")
        self.status_lbl.show()

        self._thread = QThread()
        self._worker = CalendarWorker(self.tab_type)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_finished)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _on_finished(self):
        self._loading = False
        self._loaded = True
        if self._worker:
            self._worker.deleteLater()
        if self._thread:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    # Datum-kolumner att söka efter vid sortering
    # Exakta kolumnnamn per kalendertyp (verifierade mot yfinance 2026-02-21)
    _DATE_KEYS = [
        'Event Start Date',  # earnings
        'Date',              # ipos
        'Payable On',        # splits
        'Event Time',        # economic
        'For',               # economic (fallback)
        'Filing Date',       # ipos (fallback)
    ]

    @staticmethod
    def _sort_key(record: dict, date_keys: list) -> float:
        """Returnerar unix-tid för sortering; 0.0 om inget datum hittas."""
        for k in date_keys:
            v = record.get(k)
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            if hasattr(v, 'timestamp'):
                try:
                    return v.timestamp()
                except Exception:
                    pass
            if isinstance(v, str) and v.lower() not in ('nat', 'nan', 'none', ''):
                try:
                    # Prova hela strängen först (inkl. tid), sedan bara datumdelen
                    clean = v.replace('T', ' ').strip()
                    try:
                        return datetime.fromisoformat(clean).timestamp()
                    except ValueError:
                        return datetime.fromisoformat(clean.split(' ')[0]).timestamp()
                except Exception:
                    pass
        return 0.0

    def _on_result(self, tab_type: str, records: list):
        self.status_lbl.hide()

        # Rensa gamla rader
        for r in self._rows:
            r.deleteLater()
        self._rows.clear()

        if not records:
            self.status_lbl.setText("Ingen data tillgänglig för perioden.")
            self.status_lbl.show()
            return

        # Sortera kronologiskt fallande (senast datum överst)
        records = sorted(records, key=lambda r: self._sort_key(r, self._DATE_KEYS), reverse=False)

        for record in records:
            row = CalendarEventRow(record, self.tab_type)
            row.clicked_detail.connect(self._show_detail)
            self._rows.append(row)
            # Lägg in före stretch-elementet
            self.cnt_layout.insertWidget(self.cnt_layout.count() - 1, row)

    def _on_error(self, tab_type: str, error: str):
        self.status_lbl.setText(f"Fel vid hämtning: {error}")
        self.status_lbl.show()

    def _show_detail(self, event_data: dict):
        popup = CalendarEventDetail(event_data, self)
        popup.exec_()


# ── Sector Performance (S&P Sector Treemap) ──────────────────────────

SECTOR_KEYS = [
    'basic-materials', 'communication-services', 'consumer-cyclical',
    'consumer-defensive', 'energy', 'financial-services', 'healthcare',
    'industrials', 'real-estate', 'technology', 'utilities',
]

def _ytd_from_download(tickers: list) -> dict:
    """Hämta YTD % för en lista tickers via yf.download. Returnerar {ticker: ytd_pct}."""
    import yfinance as yf
    ytd_map = {}
    if not tickers:
        return ytd_map
    try:
        data = yf.download(tickers, period='ytd', interval='1d',
                           progress=False, threads=True, ignore_tz=True)
        if data.empty:
            return ytd_map
        if isinstance(data.columns, pd.MultiIndex):
            top = data.columns.get_level_values(0).unique()
            col = 'Close' if 'Close' in top else ('Price' if 'Price' in top else top[0])
            close = data[col]
        else:
            close = data
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna()
                if len(s) >= 2:
                    ytd_map[t] = round(float((s.iloc[-1] / s.iloc[0] - 1) * 100), 2)
    except Exception as e:
        print(f"[Sector] YTD download failed for {tickers}: {e}")
    return ytd_map


class SectorDataWorker(QObject):
    """Hämtar S&P-sektordata via yfinance.Sector i bakgrundstråd."""

    finished = Signal()
    result = Signal(list, dict)   # (sector_list, industry_dict)
    error = Signal(str)

    def __init__(self, sector_keys: list = None):
        super().__init__()
        self.sector_keys = sector_keys or SECTOR_KEYS

    @Slot()
    def run(self):
        try:
            import yfinance as yf
            sectors = []
            industries = {}

            # Steg 1: Hämta sektor-objekt och deras .symbol
            sec_objects = {}
            ticker_to_key = {}
            for key in self.sector_keys:
                try:
                    sec = yf.Sector(key)
                    sec_objects[key] = sec
                    sym = getattr(sec, 'symbol', None)
                    if sym:
                        ticker_to_key[sym] = key
                except Exception as e:
                    print(f"[Sector] Failed to create Sector({key}): {e}")

            # Steg 2: Batch-hämta YTD för alla sektor-tickers
            ytd_map = _ytd_from_download(list(ticker_to_key.keys()))
            key_ytd = {ticker_to_key[t]: pct for t, pct in ytd_map.items()}

            # Steg 3: Bygg sektor-lista med overview + YTD
            for key in self.sector_keys:
                try:
                    sec = sec_objects.get(key)
                    if sec is None:
                        raise ValueError(f"No Sector object for {key}")
                    ov = sec.overview or {}
                    name = ov.get('name', key.replace('-', ' ').title())
                    market_weight = ov.get('market_weight', 0)
                    if isinstance(market_weight, str):
                        market_weight = float(market_weight.replace('%', '').strip()) / 100.0

                    ytd_pct = key_ytd.get(key, 0.0)

                    sectors.append({
                        'key': key,
                        'name': name,
                        'market_weight': float(market_weight) if market_weight else 0.01,
                        'ytd_pct': ytd_pct,
                    })

                    # Cache industry data (inkl. industry key för drill-down)
                    try:
                        ind_df = sec.industries
                        if ind_df is not None and not ind_df.empty:
                            # Säkerställ att index (industry key) blir en kolumn
                            if ind_df.index.name:
                                idx_name = ind_df.index.name
                                ind_df = ind_df.reset_index()
                                # Normalisera kolumnnamnet till 'key'
                                if idx_name != 'key' and idx_name in ind_df.columns:
                                    ind_df = ind_df.rename(columns={idx_name: 'key'})
                            elif 'key' not in ind_df.columns:
                                ind_df = ind_df.reset_index()
                                # reset_index skapar 'index'-kolumn
                                if 'index' in ind_df.columns:
                                    ind_df = ind_df.rename(columns={'index': 'key'})
                            records = ind_df.to_dict('records')
                            if records:
                                pass  # Industries loaded
                            industries[key] = records
                        else:
                            industries[key] = []
                    except Exception as e2:
                        print(f"[Sector] Industries for {key} failed: {e2}")
                        industries[key] = []

                    # Sector processed OK

                except Exception as e:
                    print(f"[Sector] Failed to process {key}: {e}")
                    sectors.append({
                        'key': key,
                        'name': key.replace('-', ' ').title(),
                        'market_weight': 0.01,
                        'ytd_pct': key_ytd.get(key, 0.0),
                    })
                    industries[key] = []

            self.result.emit(sectors, industries)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class IndustryYTDWorker(QObject):
    """Hämtar YTD % för industrier via yf.Industry(key).symbol i bakgrundstråd."""

    finished = Signal()
    result = Signal(str, list)   # (sector_key, enriched industry list)
    error = Signal(str)

    def __init__(self, sector_key: str, industry_records: list):
        super().__init__()
        self.sector_key = sector_key
        self.industry_records = industry_records

    @Slot()
    def run(self):
        try:
            # Använd 'symbol' direkt från industry-records (redan cachad från sec.industries)
            ind_tickers = {}  # ticker → index i records
            for i, rec in enumerate(self.industry_records):
                sym = rec.get('symbol', '')
                if sym:
                    ind_tickers[sym] = i

            # Industry tickers loaded

            # Batch-hämta YTD
            ytd_map = _ytd_from_download(list(ind_tickers.keys()))
            # Industry YTD done

            # Berika records med ytd_pct
            enriched = list(self.industry_records)
            for ticker, idx in ind_tickers.items():
                if ticker in ytd_map:
                    enriched[idx]['ytd_pct'] = ytd_map[ticker]

            self.result.emit(self.sector_key, enriched)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class _SectorTreemapPage(QWebEnginePage):
    """Interceptar console.log för sektorklick i treemap."""

    sector_clicked = Signal(str)  # sector key

    def __init__(self, parent=None):
        super().__init__(parent)

    def javaScriptConsoleMessage(self, level, msg, line, src):
        if msg.startswith('SECTOR_CLICK:'):
            key = msg[len('SECTOR_CLICK:'):]
            self.sector_clicked.emit(key)
            return
        if msg.startswith('SECTOR_BACK'):
            self.sector_clicked.emit('__BACK__')
            return
        # Suppress noisy JS output
        if level == 0:
            return


class SectorPerformanceWidget(QWidget):
    """Sektor-treemap med drill-down till industrier."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded = False
        self._loading = False
        self._worker = None
        self._thread = None
        self._ind_worker = None
        self._ind_thread = None
        self._sectors = []
        self._industries = {}
        self._current_view = 'sectors'  # 'sectors' eller sector key

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.status_lbl = QLabel("Välj fliken för att ladda sektordata...")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px; padding: 16px; background: transparent;"
        )
        outer.addWidget(self.status_lbl)

        if WEBENGINE_AVAILABLE and QWebEngineView is not None:
            self.web_view = QWebEngineView()
            self._page = _SectorTreemapPage(self.web_view)
            self._page.sector_clicked.connect(self._on_sector_click)
            self.web_view.setPage(self._page)
            self.web_view.setStyleSheet("background-color: #0a0a0a;")
            self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.web_view.hide()
            outer.addWidget(self.web_view, stretch=1)
        else:
            self.web_view = None

    def load_if_needed(self):
        if not self._loaded and not self._loading:
            self._start_fetch()

    def refresh(self):
        if self._loading:
            return
        self._loaded = False
        self._current_view = 'sectors'
        self._start_fetch()

    def _start_fetch(self):
        self._loading = True
        self.status_lbl.setText("Hämtar sektordata...")
        self.status_lbl.show()
        if self.web_view:
            self.web_view.hide()

        self._thread = QThread()
        self._worker = SectorDataWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._on_finished)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)

        self._thread.start()

    def _on_finished(self):
        self._loading = False
        self._loaded = True
        if self._worker:
            self._worker.deleteLater()
        if self._thread:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None

    def _on_result(self, sectors: list, industries: dict):
        self._sectors = sectors
        self._industries = industries
        self.status_lbl.hide()
        self._render_sector_view()

    def _on_error(self, error: str):
        self.status_lbl.setText(f"Fel vid hämtning: {error}")
        self.status_lbl.show()

    def _on_sector_click(self, key: str):
        if key == '__BACK__':
            self._current_view = 'sectors'
            self._render_sector_view()
        else:
            self._current_view = key
            self._start_industry_fetch(key)

    def _start_industry_fetch(self, sector_key: str):
        """Hämta YTD för industrier i bakgrundstråd, visa sedan treemap."""
        inds = self._industries.get(sector_key, [])
        if not inds:
            sector_name = sector_key.replace('-', ' ').title()
            self.status_lbl.setText(f"Ingen industridata för {sector_name}")
            self.status_lbl.show()
            if self.web_view:
                self.web_view.hide()
            return

        self.status_lbl.setText("Hämtar industridata...")
        self.status_lbl.show()

        # Rensa ev. gammal industri-tråd
        if self._ind_worker:
            self._ind_worker.deleteLater()
        if self._ind_thread:
            self._ind_thread.quit()
            self._ind_thread.deleteLater()

        self._ind_thread = QThread()
        self._ind_worker = IndustryYTDWorker(sector_key, inds)
        self._ind_worker.moveToThread(self._ind_thread)

        self._ind_thread.started.connect(self._ind_worker.run)
        self._ind_worker.finished.connect(self._ind_thread.quit)
        self._ind_worker.finished.connect(self._on_ind_finished)
        self._ind_worker.result.connect(self._on_ind_result)
        self._ind_worker.error.connect(self._on_error)

        self._ind_thread.start()

    def _on_ind_finished(self):
        if self._ind_worker:
            self._ind_worker.deleteLater()
        if self._ind_thread:
            self._ind_thread.deleteLater()
        self._ind_worker = None
        self._ind_thread = None

    def _on_ind_result(self, sector_key: str, enriched_inds: list):
        self._industries[sector_key] = enriched_inds
        self.status_lbl.hide()
        self._render_industry_view(sector_key)

    # ── HTML-generering ──────────────────────────────────────────────

    def _render_sector_view(self):
        if not self.web_view or not self._sectors:
            return

        ids = ['S&P 500']
        labels = ['']
        parents = ['']
        values = [0]
        colors = [0]
        texts = ['']

        # Beräkna min-storlek: minst 4% av största sektor så text alltid syns
        max_w = max((s['market_weight'] for s in self._sectors), default=0.01)
        min_w = max_w * 0.18

        total_w = 0
        for s in self._sectors:
            sid = s['key']
            real_w = s['market_weight']
            w = max(real_w, min_w)
            ytd = s['ytd_pct']
            total_w += w
            ids.append(sid)
            labels.append(f"<b>{s['name']}</b>")
            parents.append('S&P 500')
            values.append(w)
            colors.append(ytd)
            sign = '+' if ytd >= 0 else ''
            texts.append(f"{real_w*100:.1f}%  |  YTD {sign}{ytd:.1f}%")

        values[0] = total_w

        data_json = json.dumps({
            'ids': ids, 'labels': labels, 'parents': parents,
            'values': values, 'colors': colors, 'texts': texts,
        })

        html = self._build_html(data_json, show_back=False, title='S&P 500 Sectors')
        self._load_html(html, 'sector_treemap.html')
        self.web_view.show()

    def _render_industry_view(self, sector_key: str):
        if not self.web_view:
            return

        inds = self._industries.get(sector_key, [])
        sector_name = sector_key.replace('-', ' ').title()
        for s in self._sectors:
            if s['key'] == sector_key:
                sector_name = s['name']
                break

        if not inds:
            self.status_lbl.setText(f"Ingen industridata för {sector_name}")
            self.status_lbl.show()
            self.web_view.hide()
            return

        ids = [sector_name]
        labels = ['']
        parents = ['']
        values = [0]
        colors = [0]
        texts = ['']

        # Extrahera vikter + YTD (ytd_pct satt av IndustryYTDWorker)
        parsed_inds = []
        for ind in inds:
            name = ind.get('name', ind.get('industry key', 'Unknown'))
            w = ind.get('market weight', ind.get('market_weight', 0.01))
            if isinstance(w, str):
                try:
                    w = float(w.replace('%', '').strip()) / 100.0
                except ValueError:
                    w = 0.01
            w = max(float(w), 0.001)

            # Föredra ytd_pct från IndustryYTDWorker (.symbol-baserad)
            ytd = ind.get('ytd_pct', 0.0)
            parsed_inds.append((name, w, float(ytd)))

        max_w = max((w for _, w, _ in parsed_inds), default=0.01)
        min_w = max_w * 0.15

        total_w = 0
        for name, real_w, ytd in parsed_inds:
            w = max(real_w, min_w)
            total_w += w

            ids.append(name)
            labels.append(f"<b>{name}</b>")
            parents.append(sector_name)
            values.append(w)
            colors.append(ytd)
            sign = '+' if ytd >= 0 else ''
            texts.append(f"{real_w*100:.1f}%  |  YTD {sign}{ytd:.1f}%")

        values[0] = total_w

        data_json = json.dumps({
            'ids': ids, 'labels': labels, 'parents': parents,
            'values': values, 'colors': colors, 'texts': texts,
        })

        html = self._build_html(data_json, show_back=True, title=sector_name)
        self._load_html(html, 'sector_treemap.html')
        self.web_view.show()
        self.status_lbl.hide()

    def _build_html(self, data_json: str, show_back: bool = False, title: str = '') -> str:
        back_btn = ''
        if show_back:
            back_btn = f'''
            <div id="backBtn" style="position:absolute;top:6px;left:8px;z-index:100;
                 background:{COLORS['bg_card']};border:1px solid {COLORS['border_default']};
                 border-radius:4px;padding:3px 10px;cursor:pointer;
                 color:{COLORS['text_secondary']};font-size:11px;font-family:monospace;"
                 onclick="console.log('SECTOR_BACK')">← Tillbaka</div>
            '''

        return f'''<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ background:#0a0a0a; width:100%; height:100%; overflow:hidden;
             font-family:'JetBrains Mono',monospace; }}
#map {{ width:100%; height:100%; }}
</style>
</head><body>
{back_btn}
<div id="map"></div>
<script>
(function() {{
    var D = {data_json};

    // Beräkna färger direkt per tile (röd → grön baserat på YTD %)
    function ytdToColor(v) {{
        if (v === 0) return '#1a1a2e';
        var maxV = 15;
        var t = Math.min(Math.abs(v), maxV) / maxV;
        if (v > 0) {{
            var r = Math.round(26 + (30 - 26) * (1 - t));
            var g = Math.round(26 + (120 - 26) * t);
            var b = Math.round(46 + (30 - 46) * t);
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }} else {{
            var r = Math.round(26 + (180 - 26) * t);
            var g = Math.round(26 + (30 - 26) * (1 - t));
            var b = Math.round(46 + (30 - 46) * (1 - t));
            return 'rgb(' + r + ',' + g + ',' + b + ')';
        }}
    }}

    var tileColors = D.colors.map(function(c) {{ return ytdToColor(c); }});

    Plotly.newPlot('map', [{{
        type: 'treemap',
        ids: D.ids,
        labels: D.labels,
        parents: D.parents,
        values: D.values,
        text: D.texts,
        textinfo: 'label+text',
        hoverinfo: 'label+text',
        branchvalues: 'total',
        pathbar: {{ visible: false }},
        tiling: {{ pad: 2 }},
        marker: {{
            colors: tileColors,
            line: {{ width: 1, color: '#111' }},
            showscale: false
        }},
        textfont: {{ color: '#e0e0e0', size: 11 }},
        insidetextorientation: 'horizontal'
    }}], {{
        margin: {{ t: 2, b: 2, l: 2, r: 2 }},
        paper_bgcolor: '#0a0a0a',
        plot_bgcolor: '#0a0a0a',
        font: {{ family: "'JetBrains Mono', monospace", color: '#ccc' }}
    }}, {{
        displayModeBar: false,
        responsive: true
    }});

    {'document.getElementById("map").on("plotly_click", function(evt) { if (!evt || !evt.points || !evt.points[0]) return; var id = evt.points[0].id; if (id && D.parents[D.ids.indexOf(id)] !== "") { console.log("SECTOR_CLICK:" + id); } });' if not show_back else ''}
}})();
</script>
</body></html>'''

    def _load_html(self, html: str, filename: str):
        try:
            treemap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trading')
            os.makedirs(treemap_dir, exist_ok=True)
            path = os.path.join(treemap_dir, filename)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.web_view.load(QUrl.fromLocalFile(path))
        except Exception as e:
            print(f'[Sector] File write error: {e}')
            self.web_view.setHtml(html)


class RightPanelWidget(QWidget):
    """Tabbed höger-panel: News Feed | Earnings | Economic Events | Sectors."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background: transparent;
            }}
            QTabBar::tab {{
                background: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: none;
                border-bottom: 2px solid transparent;
                padding: 5px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.8px;
                min-width: 44px;
            }}
            QTabBar::tab:selected {{
                color: {COLORS['accent']};
                border-bottom: 2px solid {COLORS['accent']};
                background: {COLORS['bg_elevated']};
            }}
            QTabBar::tab:hover:!selected {{
                color: {COLORS['text_primary']};
                background: {COLORS['bg_hover']};
            }}
        """)

        # Tab 0: News Feed
        self._news_tab = NewsFeedWidget()
        self.tab_widget.addTab(self._news_tab, "NEWS")

        # Tab 1: Earnings  Tab 2: Economic Events  Tab 3: IPO  Tab 4: Splits
        self._earnings_tab = CalendarTabWidget('earnings')
        self._economic_tab = CalendarTabWidget('economic')
        self._ipo_tab = CalendarTabWidget('ipo')
        self._splits_tab = CalendarTabWidget('splits')

        self.tab_widget.addTab(self._earnings_tab, "EARNINGS")
        self.tab_widget.addTab(self._economic_tab, "EVENTS")
        self.tab_widget.addTab(self._ipo_tab, "IPO")
        self.tab_widget.addTab(self._splits_tab, "SPLITS")

        # Tab 5: Sector Performance
        self._sector_tab = SectorPerformanceWidget()
        self.tab_widget.addTab(self._sector_tab, "SECTORS")

        # Uppdatera-knapp i tab-barets hörnposition
        self._refresh_corner_btn = QPushButton("↻")
        self._refresh_corner_btn.setFixedSize(22, 22)
        self._refresh_corner_btn.setToolTip("Uppdatera aktuell flik")
        self._refresh_corner_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {COLORS['border_default']};
                border-radius: 4px;
                color: {COLORS['text_secondary']};
                font-size: 13px;
                margin: 2px 4px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['accent']};
                color: {COLORS['accent']};
                background: {COLORS['bg_hover']};
            }}
            QPushButton:pressed {{
                background: {COLORS['bg_card']};
            }}
        """)
        self._refresh_corner_btn.clicked.connect(self._on_corner_refresh)
        self.tab_widget.setCornerWidget(self._refresh_corner_btn, Qt.TopRightCorner)

        self.tab_widget.tabBar().setExpanding(True)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tab_widget, 1)

        # Events Calendar under tabbarna (kompakt veckoöversikt)
        self._events_calendar = EventsCalendarWidget()
        self._events_calendar.setFixedHeight(300)
        layout.addWidget(self._events_calendar)

    def _on_tab_changed(self, index: int):
        """Lazy-load vid första besök på kalender-/sektor-flik."""
        tab_map = {
            1: self._earnings_tab,
            2: self._economic_tab,
            3: self._ipo_tab,
            4: self._splits_tab,
            5: self._sector_tab,
        }
        if index in tab_map:
            tab_map[index].load_if_needed()

    def _on_corner_refresh(self):
        """Uppdaterar aktuell flik — news, kalender eller sektor."""
        idx = self.tab_widget.currentIndex()
        cal_map = {
            1: self._earnings_tab,
            2: self._economic_tab,
            3: self._ipo_tab,
            4: self._splits_tab,
            5: self._sector_tab,
        }
        if idx in cal_map:
            cal_map[idx].refresh()
        # NEWS (idx=0) hanteras av extern koppling via refresh_btn.clicked

    # ------ Bakåtkompatibla egenskaper/metoder (news_feed-gränssnitt) ------

    @property
    def refresh_btn(self):
        """Exponerar hörnknappen — extern koppling triggar news-refresh vid klick."""
        return self._refresh_corner_btn

    def refresh_news(self, csv_path: str = None):
        self._news_tab.refresh_news(csv_path)

    def scale_to(self, scale: float):
        self._news_tab.scale_to(scale)


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
        '^STOXX': ('Europe 600', 'EUROPE'),
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

    # US index → futures mapping for implied open calculation
    US_INDEX_FUTURES = {
        '^GSPC': 'ES=F',   # S&P 500 → E-mini S&P
        '^NDX':  'NQ=F',   # NASDAQ 100 → E-mini NASDAQ
        '^DJI':  'YM=F',   # Dow Jones → E-mini Dow
        '^RUT':  'RTY=F',  # Russell 2000 → E-mini Russell
    }
    FUTURES_TO_SPOT = {v: k for k, v in US_INDEX_FUTURES.items()}

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
        self._daily_prev_close = {}  # {ticker: prev_close} from daily data
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
        self._intraday_max_retries = 20    # Max antal retries (keep trying until all loaded)
        self._ws_last_update_time = {}      # symbol → timestamp of last WS price update
        self._stale_refresh_running = False  # Guard for stale instrument refresh worker

        self._volatility_worker: Optional[VolatilityDataWorker] = None
        self._volatility_thread: Optional[QThread] = None
        self._volatility_running = False  # Säker flagga
        # Cachade historiska serier för live-percentilberäkning (sparas vid yf.download)
        self._vol_hist_cache: Dict[str, np.ndarray] = {}  # ticker → sorterad numpy-array
        self._vol_median_cache: Dict[str, float] = {}     # ticker → median
        self._vol_mode_cache: Dict[str, float] = {}       # ticker → mode
        self._vol_sparkline_cache: Dict[str, list] = {}   # ticker → sparkline-värden
        self._vol_full_history_loaded = False  # True after period='max' fetch or valid disk cache
        self._portfolio_refresh_worker: Optional[PortfolioRefreshWorker] = None
        self._portfolio_refresh_thread: Optional[QThread] = None
        self._portfolio_refresh_running = False  # Säker flagga

        # Morgan Stanley instrument fetch (async)
        self._ms_fetch_thread: Optional[QThread] = None
        self._ms_fetch_worker: Optional[MSInstrumentWorker] = None

        # Markov chain analysis state
        self._markov_thread: Optional[QThread] = None
        self._markov_worker = None
        self._markov_running = False
        self._markov_result = None
        self._markov_batch_results = {}  # {ticker: MarkovResult} från Master Scanner
        self._markov_batch_running = False
        self._markov_batch_thread: Optional[QThread] = None
        self._markov_batch_worker = None

        # Strategy analysis states
        self._eps_mr_running = False
        self._eps_mr_result = None
        self._eps_mr_thread: Optional[QThread] = None
        self._eps_mr_worker = None

        self._squeeze_running = False
        self._squeeze_result = None
        self._options_running = False
        self._options_data = {}  # yf_ticker -> straddle summary dict
        self._vol_analytics_data = {}  # yf_ticker -> vol analytics dict
        self._vol_analytics_running = False
        self._squeeze_thread: Optional[QThread] = None
        self._squeeze_worker = None

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
            for card_name in ['vix_card', 'vvix_card', 'skew_card', 'vvix_vix_card', 'move_card']:
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
            for card_name in ['vix_card', 'vvix_card', 'skew_card', 'vvix_vix_card', 'move_card']:
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
        
        # Timer-check körs varje minut (tyst)
        
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
                print(f"[SCHEDULE] Triggering scheduled scan at {now.strftime('%H:%M')}")
                self._last_scheduled_run = now.strftime('%Y-%m-%d')
                self.statusBar().showMessage("Starting scheduled scan...")
                
                # Set flag for scheduled scan (used by callbacks)
                self._scheduled_snapshot_pending = True
                
                # Refresha earnings-data så att rapportresultat hinner komma in
                if hasattr(self, '_earnings_tab') and self._earnings_tab is not None:
                    self._earnings_tab.refresh()

                # First refresh MF prices asynchronously
                if MF_PRICE_SCRAPING_AVAILABLE and self.portfolio:
                    # Refreshing MF prices before scan
                    self.refresh_mf_prices()
                    # Snapshot will be taken in _on_mf_prices_received when prices arrive
                    # Then we start the scan after a short delay to let prices settle
                    QTimer.singleShot(3000, self._run_scheduled_scan_after_prices)
                else:
                    # No MF prices to fetch, take snapshot and run scan immediately
                    # No MF prices needed, running scan directly
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
            # Scheduled scan starting
            
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
            
            # VIKTIGT: Se till att Arbitrage Scanner-tabben är laddad (lazy loading fix)
            # Förbättrad version med retry-logik
            max_retries = 3
            for attempt in range(max_retries):
                if not self._tabs_loaded.get(1, False):
                    self._load_page_if_needed(1)
                    # Processa events flera gånger för att säkerställa widget-skapande
                    for _ in range(5):
                        QApplication.processEvents()
                        time.sleep(0.05)
                
                # Kontrollera om widgets finns
                has_tickers = hasattr(self, 'tickers_input') and self.tickers_input is not None
                has_btn = hasattr(self, 'run_btn') and self.run_btn is not None
                
                if has_tickers and has_btn:
                    break
                elif attempt < max_retries - 1:
                    time.sleep(0.5)
            
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

            # Scheduled scans use same 2y period

            # Store that this is a scheduled scan so we can send Discord after completion
            self._is_scheduled_scan = True

            # Run analysis (will call on_analysis_complete when done)
            self.run_analysis()
            
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

    def _run_master_scanner_after_scheduled(self):
        """Kör Master Scanner (alla 4 strategier) automatiskt efter schemalagd pairs-scan."""
        print(f"\n[SCHEDULE] Triggering Master Scanner after scheduled pairs scan...")
        try:
            # Ladda Master Scanner-tabben om den inte är laddad
            if not self._tabs_loaded.get(11, False):
                self._load_page_if_needed(11)
                QApplication.processEvents()

            # Kör alla strategier (engine + prisdata finns redan från pairs-scan)
            self._refresh_master_scanner()
        except Exception as e:
            print(f"[SCHEDULE] Master Scanner error: {e}")
            import traceback
            traceback.print_exc()

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
                                'half_life': getattr(row, 'half_life_days', 0),
                                'spread': spread,
                            })
                    except Exception as e:
                        print(f"Error getting z-score for {pair}: {e}")
                        continue
                
                # Filter pairs with |z| >= their optimal z* and sort by |z| descending
                signal_pairs = []
                for p in pairs_with_z:
                    ou_tmp = self.engine.ou_models.get(p['pair'])
                    pair_opt_z = SIGNAL_TAB_THRESHOLD
                    if ou_tmp:
                        try:
                            pair_row = self.engine._pair_index.get(p['pair'])
                            g_p = getattr(pair_row, 'garch_persistence', 0.0) if pair_row else 0.0
                            f_d = getattr(pair_row, 'fractional_d', 0.5) if pair_row else 0.5
                            h_e = getattr(pair_row, 'hurst_exponent', 0.5) if pair_row else 0.5
                            pair_opt_z = ou_tmp.optimal_entry_zscore(
                                garch_persistence=g_p, fractional_d=f_d, hurst=h_e
                            ).get('optimal_z', SIGNAL_TAB_THRESHOLD)
                        except Exception:
                            pass
                    if abs(p['z']) >= pair_opt_z:
                        # Filter out negative expected PnL
                        try:
                            if ou_tmp:
                                exit_z = self.engine.config.get('exit_zscore', 0.0)
                                stop_z_cfg = self.engine.config.get('stop_zscore', 4.0)
                                sp_data = p.get('spread')
                                if sp_data is not None and len(sp_data) > 0:
                                    cur_s = sp_data.iloc[-1]
                                    if p['z'] > 0:
                                        epnl = ou_tmp.expected_pnl(cur_s, ou_tmp.spread_from_z(exit_z), ou_tmp.spread_from_z(stop_z_cfg))['expected_pnl']
                                    else:
                                        epnl = ou_tmp.expected_pnl(cur_s, ou_tmp.spread_from_z(-exit_z), ou_tmp.spread_from_z(-stop_z_cfg))['expected_pnl']
                                    if epnl <= 0:
                                        continue
                        except Exception:
                            pass
                        p['opt_z'] = pair_opt_z
                        signal_pairs.append(p)
                signal_pairs.sort(key=lambda x: abs(x['z']), reverse=True)

                # Take top 5 signals
                top_signals = signal_pairs[:5]

                if top_signals:
                    pairs_text = ""
                    for p in top_signals:
                        z = p['z']
                        pairs_text += f"**{p['pair']}** | Z={z:.2f} (opt={p.get('opt_z', 0):.2f})\n"

                    fields.append({
                        "name": f"🎯 Signals (|Z| ≥ Opt.Z* & E[PnL]>0)",
                        "value": pairs_text,
                        "inline": False
                    })
                else:
                    fields.append({
                        "name": "🎯 Signals",
                        "value": "No pairs with |Z| ≥ Opt.Z*",
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
                parity = pos.get('parity', 1)
                entry_y = pos.get('mf_entry_price_y', 0.0)
                current_y = pos.get('mf_current_price_y', entry_y)
                qty_y = pos.get('mf_qty_y', 0)

                pnl_y = 0.0
                invested_y = 0.0
                if entry_y > 0 and qty_y > 0:
                    pnl_y = (current_y - entry_y) * qty_y * parity
                    invested_y = entry_y * qty_y * parity

                # Calculate P&L for X leg (short/hedge leg)
                entry_x = pos.get('mf_entry_price_x', 0.0)
                current_x = pos.get('mf_current_price_x', entry_x)
                qty_x = pos.get('mf_qty_x', 0)

                pnl_x = 0.0
                invested_x = 0.0
                if entry_x > 0 and qty_x > 0:
                    pnl_x = (current_x - entry_x) * qty_x * parity
                    invested_x = entry_x * qty_x * parity
                
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

    # ── Email: daglig sammanfattning ─────────────────────────────────────────

    def _collect_email_data(self) -> dict:
        """Samla ihop data från alla dashboard-källor för email."""
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "version": APP_VERSION,
        }

        # 1. Alla index från treemap, grupperade per region
        indices_by_region = {}
        for sym, (name, region) in self.MARKET_INSTRUMENTS.items():
            cached = self._market_data_cache.get(sym)
            if cached and cached.get('price'):
                indices_by_region.setdefault(region, []).append({
                    "name": name,
                    "symbol": sym,
                    "price": cached.get('price', 0),
                    "change_pct": cached.get('change_pct', 0),
                })
        data["indices"] = indices_by_region

        # 2. Volatilitetsindex
        vol_items = []
        for card_attr, name in [('vix_card', 'VIX'), ('vvix_card', 'VVIX'),
                                 ('skew_card', 'SKEW'), ('move_card', 'MOVE')]:
            card = getattr(self, card_attr, None)
            if card and hasattr(card, 'current_value') and card.current_value:
                vol_items.append({
                    "name": name,
                    "value": card.current_value or 0,
                    "percentile": getattr(card, 'current_percentile', 0) or 0,
                    "median": self._vol_median_cache.get(f'^{name}', 0),
                })
        data["volatility"] = vol_items

        # 3. Rapporterande bolag (yfinance kolumnnamn: Company, Symbol, Market, etc.)
        earnings = []
        data["_earnings_attempted"] = True
        try:
            cal = CustomCalendars()
            df = cal.get_earnings_multi_region(regions=("us", "se"))
            if df is not None and not df.empty:
                # Normalisera kolumnnamn — yfinance använder CamelCase
                for _, row in df.head(30).iterrows():
                    rec = row.to_dict()
                    # Sök symbol och company i vanliga kolumnnamn
                    symbol = (rec.get("Symbol") or rec.get("symbol")
                              or rec.get("Ticker") or rec.get("ticker") or "")
                    company = (rec.get("Company") or rec.get("company")
                               or rec.get("Company Name") or rec.get("companyshortname") or "")
                    eps_actual = rec.get("EPS Actual") or rec.get("epsactual") or rec.get("Actual")
                    eps_estimate = (rec.get("EPS Consensus") or rec.get("epsconsensus")
                                    or rec.get("EPS Estimate") or rec.get("Estimate"))
                    region = rec.get("Market") or rec.get("region") or rec.get("Region") or ""

                    # Konvertera NaN → None
                    try:
                        if eps_actual is not None and eps_actual != eps_actual:
                            eps_actual = None
                        if eps_estimate is not None and eps_estimate != eps_estimate:
                            eps_estimate = None
                    except (TypeError, ValueError):
                        pass

                    if symbol or company:
                        earnings.append({
                            "symbol": str(symbol),
                            "company": str(company),
                            "eps_actual": float(eps_actual) if eps_actual is not None else None,
                            "eps_estimate": float(eps_estimate) if eps_estimate is not None else None,
                            "region": str(region),
                        })
        except Exception as e:
            print(f"[Email] Could not fetch earnings: {e}")
        # Extra dedup — behåll första (mest data) per company+region
        seen = set()
        unique_earnings = []
        for e in earnings:
            key = (e["company"], e["region"])
            if key not in seen:
                seen.add(key)
                unique_earnings.append(e)
        data["earnings"] = unique_earnings

        # 4. Portföljstatus
        pf_data = {}
        if self.portfolio:
            total_pnl = 0.0
            total_invested = 0.0
            positions = []
            for pos in self.portfolio:
                if pos.get('status', 'OPEN') != 'OPEN':
                    continue
                pair = pos.get('pair', 'Unknown')
                direction = pos.get('direction', 'LONG')

                # Beräkna P&L per position
                parity = pos.get('parity', 1)
                entry_y = pos.get('mf_entry_price_y', 0.0)
                current_y = pos.get('mf_current_price_y', entry_y)
                qty_y = pos.get('mf_qty_y', 0)
                pnl_y = (current_y - entry_y) * qty_y * parity if entry_y > 0 and qty_y > 0 else 0
                inv_y = entry_y * qty_y * parity if entry_y > 0 and qty_y > 0 else 0

                entry_x = pos.get('mf_entry_price_x', 0.0)
                current_x = pos.get('mf_current_price_x', entry_x)
                qty_x = pos.get('mf_qty_x', 0)
                pnl_x = (current_x - entry_x) * qty_x * parity if entry_x > 0 and qty_x > 0 else 0
                inv_x = entry_x * qty_x * parity if entry_x > 0 and qty_x > 0 else 0

                pos_pnl = pnl_y + pnl_x
                pos_inv = inv_y + inv_x
                total_pnl += pos_pnl
                total_invested += pos_inv

                current_z = pos.get('current_z', 0.0)
                if self.engine is not None:
                    try:
                        _, _, live_z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                        current_z = live_z
                    except Exception:
                        pass

                positions.append({
                    "pair": pair,
                    "pnl_pct": (pos_pnl / pos_inv * 100) if pos_inv > 0 else 0,
                    "z_score": current_z,
                    "direction": direction,
                })

            pf_data["total_pnl"] = total_pnl
            pf_data["total_pnl_pct"] = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            pf_data["total_value"] = total_invested + total_pnl
            pf_data["positions"] = positions
        data["portfolio"] = pf_data

        # 5. Scanningsresultat
        scan_data = {}
        if self.engine is not None:
            n_tickers = len(self.engine.price_data.columns) if self.engine.price_data is not None else 0
            n_pairs = len(self.engine.pairs_stats) if self.engine.pairs_stats is not None else 0
            n_viable = len(self.engine.viable_pairs) if self.engine.viable_pairs is not None else 0
            scan_data["n_tickers"] = n_tickers
            scan_data["n_pairs"] = n_pairs
            scan_data["n_viable"] = n_viable

            # Signaler (samma logik som Discord)
            signals = []
            if self.engine.viable_pairs is not None and len(self.engine.viable_pairs) > 0:
                for row in self.engine.viable_pairs.itertuples():
                    pair = row.pair
                    try:
                        ou, spread, z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                        if ou is None:
                            continue
                        pair_opt_z = SIGNAL_TAB_THRESHOLD
                        try:
                            pair_row = self.engine._pair_index.get(pair)
                            g_p = getattr(pair_row, 'garch_persistence', 0.0) if pair_row else 0.0
                            f_d = getattr(pair_row, 'fractional_d', 0.5) if pair_row else 0.5
                            h_e = getattr(pair_row, 'hurst_exponent', 0.5) if pair_row else 0.5
                            pair_opt_z = ou.optimal_entry_zscore(
                                garch_persistence=g_p, fractional_d=f_d, hurst=h_e
                            ).get('optimal_z', SIGNAL_TAB_THRESHOLD)
                        except Exception:
                            pass
                        if abs(z) >= pair_opt_z:
                            # Filter out negative expected PnL
                            try:
                                exit_z = self.engine.config.get('exit_zscore', 0.0)
                                stop_z_cfg = self.engine.config.get('stop_zscore', 4.0)
                                cur_s = spread.iloc[-1]
                                if z > 0:
                                    epnl = ou.expected_pnl(cur_s, ou.spread_from_z(exit_z), ou.spread_from_z(stop_z_cfg))['expected_pnl']
                                else:
                                    epnl = ou.expected_pnl(cur_s, ou.spread_from_z(-exit_z), ou.spread_from_z(-stop_z_cfg))['expected_pnl']
                                if epnl <= 0:
                                    continue
                            except Exception:
                                pass
                            signals.append({
                                "pair": pair,
                                "z": z,
                                "opt_z": pair_opt_z,
                                "half_life": getattr(row, 'half_life_days', 0),
                            })
                    except Exception:
                        continue
                signals.sort(key=lambda x: abs(x['z']), reverse=True)
            scan_data["signals"] = signals[:5]
        data["scan"] = scan_data

        return data

    def send_daily_summary_email(self):
        """Bygg och skicka dagligt sammanfattningsmail (manuell eller automatisk trigger)."""
        config = get_email_config()
        if not config.get("email_address") or not config.get("email_app_password"):
            self.statusBar().showMessage("Email not configured — use File > Configure Email")
            return

        try:
            data = self._collect_email_data()
            html = build_daily_summary_html(data)

            self._email_thread = QThread()
            self._email_worker = EmailWorker(config, html)
            self._email_worker.moveToThread(self._email_thread)

            self._email_thread.started.connect(self._email_worker.run)
            self._email_worker.finished.connect(self._email_thread.quit)
            self._email_worker.finished.connect(self._email_worker.deleteLater)
            self._email_thread.finished.connect(self._email_thread.deleteLater)
            self._email_worker.success.connect(
                lambda msg: self.statusBar().showMessage(msg))
            self._email_worker.error.connect(
                lambda msg: self.statusBar().showMessage(f"Email failed: {msg}"))

            self._email_thread.start()
            self.statusBar().showMessage("Sending daily summary email...")
        except Exception as e:
            self.statusBar().showMessage(f"Email error: {e}")
            print(f"[Email] Error: {e}")
            traceback.print_exc()

    def _show_email_config_dialog(self):
        """Visa konfigurationsdialog för email-inställningar."""
        config = get_email_config()

        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Email")
        dialog.setFixedWidth(420)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {COLORS['bg_dark']}; color: {COLORS['text_primary']}; }}
            QLabel {{ color: {COLORS['text_secondary']}; font-size: 12px; }}
            QLineEdit {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px; padding: 6px 10px; font-size: 13px;
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_darkest']};
                border: none; border-radius: 4px;
                padding: 8px 20px; font-weight: bold; font-size: 13px;
            }}
            QPushButton:hover {{ background-color: {COLORS.get('accent_hover', '#e8a04e')}; }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Gmail SMTP Settings")
        title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 15px; font-weight: bold;")
        layout.addWidget(title)

        hint = QLabel("Use a Gmail App Password (not your regular password).\n"
                       "Generate one at myaccount.google.com > Security > App passwords.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addSpacing(6)

        lbl_from = QLabel("Gmail Address (sender)")
        layout.addWidget(lbl_from)
        edit_from = QLineEdit(config.get("email_address", ""))
        edit_from.setPlaceholderText("your.email@gmail.com")
        layout.addWidget(edit_from)

        lbl_pw = QLabel("App Password")
        layout.addWidget(lbl_pw)
        edit_pw = QLineEdit(config.get("email_app_password", ""))
        edit_pw.setEchoMode(QLineEdit.Password)
        edit_pw.setPlaceholderText("xxxx xxxx xxxx xxxx")
        layout.addWidget(edit_pw)

        lbl_to = QLabel("Recipient Address")
        layout.addWidget(lbl_to)
        edit_to = QLineEdit(config.get("email_recipient", ""))
        edit_to.setPlaceholderText("recipient@example.com")
        layout.addWidget(edit_to)

        layout.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS['bg_card']}; color: {COLORS['text_primary']}; }}
        """)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

        def on_save():
            save_email_config(edit_from.text().strip(), edit_pw.text().strip(), edit_to.text().strip())
            self.statusBar().showMessage("Email configuration saved")
            dialog.accept()

        btn_save.clicked.connect(on_save)
        btn_cancel.clicked.connect(dialog.reject)
        dialog.exec_()

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

        configure_email_action = QAction("Configure Email", self)
        configure_email_action.triggered.connect(self._show_email_config_dialog)
        file_menu.addAction(configure_email_action)

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

        send_email_action = QAction("Send Daily Summary Email", self)
        send_email_action.triggered.connect(self.send_daily_summary_email)
        analysis_menu.addAction(send_email_action)
        
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
        
        # ── Sidebar navigation + stacked content ──
        # OPTIMERING: Lazy loading med container-widgets
        self._tabs_loaded = {
            0: True,   # Market Overview - laddas direkt
            1: False,  # Arbitrage Scanner
            2: False,  # OU Analytics
            3: False,  # Pair Signals
            4: False,  # Portfolio
            5: False,  # Markov Chains
            10: False, # EPS Mean Reversion
        }
        self._tab_containers = {}

        # Sidebar tree
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.setIndentation(16)
        self.nav_tree.setAnimated(True)
        self.nav_tree.setExpandsOnDoubleClick(False)
        self.nav_tree.setRootIsDecorated(False)
        self.nav_tree.setFixedWidth(220)
        self.nav_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {COLORS['bg_darkest']};
                border: none;
                border-right: 1px solid {COLORS['border_subtle']};
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                outline: none;
            }}
            QTreeWidget::item {{
                color: {COLORS['text_muted']};
                padding: 7px 12px;
                border: none;
            }}
            QTreeWidget::item:selected {{
                background-color: rgba(212, 165, 116, 0.12);
                color: {COLORS['accent']};
                border-left: 2px solid {COLORS['accent']};
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {COLORS['bg_hover']};
                color: {COLORS['text_secondary']};
            }}
            QTreeWidget::branch {{
                background: transparent;
            }}
            QTreeWidget::branch:has-children:closed {{
                image: none;
            }}
            QTreeWidget::branch:has-children:open {{
                image: none;
            }}
        """)

        # Stacked content area
        self.content_stack = QStackedWidget()

        # ── Bygg navigation-träd och stacked pages ──
        # Mapping: tree item → stacked index
        self._nav_index_map = {}

        # Topnivå-items (standalone)
        market_item = QTreeWidgetItem(self.nav_tree, ["  ◎  MARKET OVERVIEW"])
        market_item.setFlags(market_item.flags() & ~Qt.ItemIsSelectable | Qt.ItemIsSelectable)
        font_top = QFont("JetBrains Mono", 10, QFont.Bold)
        market_item.setFont(0, font_top)
        self._nav_index_map[id(market_item)] = 0
        self.content_stack.addWidget(self.create_market_overview_tab())  # index 0

        # Master Scanner — standalone
        scanner_item = QTreeWidgetItem(self.nav_tree, ["  ⚡  MASTER SCANNER"])
        scanner_item.setFont(0, font_top)
        self._nav_index_map[id(scanner_item)] = 11
        self._tab_containers[11] = self._create_lazy_container("MASTER SCANNER")
        self.content_stack.addWidget(self._tab_containers[11])
        self._tabs_loaded[11] = False

        # ── Kategori: PAIRS TRADING ──
        cat_pairs = QTreeWidgetItem(self.nav_tree, ["  ◊  PAIRS TRADING"])
        cat_pairs.setFont(0, font_top)
        cat_pairs.setFlags(cat_pairs.flags() & ~Qt.ItemIsSelectable)
        cat_pairs.setForeground(0, QColor(COLORS['text_muted']))
        cat_pairs.setExpanded(True)

        pairs_children = [
            (1, "Arbitrage Scanner"),
            (2, "OU Analytics"),
            (3, "Pair Signals"),
        ]
        for idx, name in pairs_children:
            child = QTreeWidgetItem(cat_pairs, [f"    {name}"])
            self._nav_index_map[id(child)] = idx
            self._tab_containers[idx] = self._create_lazy_container(name.upper())
            self.content_stack.addWidget(self._tab_containers[idx])  # index = idx

        # ── Kategori: KVANTMODELLER ──
        cat_quant = QTreeWidgetItem(self.nav_tree, ["  ∂  QUANT MODELS"])
        cat_quant.setFont(0, font_top)
        cat_quant.setFlags(cat_quant.flags() & ~Qt.ItemIsSelectable)
        cat_quant.setForeground(0, QColor(COLORS['text_muted']))
        cat_quant.setExpanded(True)

        quant_children = [
            (5, "Markov Chains"),
            (10, "EPS Reversion"),
            (12, "TTM Squeeze"),
        ]
        for entry in quant_children:
            idx, name = entry[0], entry[1]
            hidden = entry[2] if len(entry) > 2 else False
            if not hidden:
                child = QTreeWidgetItem(cat_quant, [f"    {name}"])
                self._nav_index_map[id(child)] = idx
            self._tab_containers[idx] = self._create_lazy_container(name.upper())
            self.content_stack.addWidget(self._tab_containers[idx])

        # ── Kategori: PORTFÖLJ ──
        cat_port = QTreeWidgetItem(self.nav_tree, ["  ≡  PORTFOLIO"])
        cat_port.setFont(0, font_top)
        cat_port.setFlags(cat_port.flags() & ~Qt.ItemIsSelectable)
        cat_port.setForeground(0, QColor(COLORS['text_muted']))
        cat_port.setExpanded(True)

        port_child = QTreeWidgetItem(cat_port, ["    Positions & History"])
        self._nav_index_map[id(port_child)] = 4
        self._tab_containers[4] = self._create_lazy_container("PORTFOLIO")
        self.content_stack.addWidget(self._tab_containers[4])

        # Stacked widget index → page index mapping
        # Actual widget addition order: 0=Market, 11=Scanner, 1=Arb, 2=OU, 3=Signals,
        # 5=Markov, 10=EPS, 12=Squeeze, 4=Portfolio
        self._logical_to_stack = {}
        add_order = [0, 11, 1, 2, 3, 5, 10, 12, 4]
        for stack_idx, logical_idx in enumerate(add_order):
            self._logical_to_stack[logical_idx] = stack_idx

        # Connect sidebar click → page switch
        self.nav_tree.currentItemChanged.connect(self._on_nav_item_changed)

        # Select Market Overview by default
        self.nav_tree.setCurrentItem(market_item)

        # Layout: sidebar | content
        nav_splitter = QSplitter(Qt.Horizontal)
        nav_splitter.setHandleWidth(1)
        nav_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border_subtle']};
            }}
        """)
        nav_splitter.addWidget(self.nav_tree)
        nav_splitter.addWidget(self.content_stack)
        nav_splitter.setStretchFactor(0, 0)
        nav_splitter.setStretchFactor(1, 1)
        nav_splitter.setSizes([220, 1200])

        content_layout.addWidget(nav_splitter)
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

    def _toggle_margrabe_section(self):
        visible = not self.margrabe_container.isVisible()
        self.margrabe_container.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.margrabe_header.setText(f"{arrow} SPREAD OPTION (MARGRABE)")

    def _toggle_sizing_section(self):
        visible = not self.sizing_container.isVisible()
        self.sizing_container.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self.sizing_header.setText(f"{arrow} POSITION SIZING")

    def _toggle_robustness_section(self):
        pass  # Robustness section removed (single 2y period)

    def _on_nav_item_changed(self, current, previous):
        """Handle sidebar navigation item click → switch stacked page + lazy load."""
        if current is None:
            return
        item_id = id(current)
        if item_id not in self._nav_index_map:
            return  # Kategori-header, inte klickbar
        logical_index = self._nav_index_map[item_id]
        self.navigate_to_page(logical_index)

    def navigate_to_page(self, logical_index: int):
        """Navigate to a page by logical index. Handles lazy loading + sidebar sync."""
        stack_idx = self._logical_to_stack.get(logical_index, 0)
        self.content_stack.setCurrentIndex(stack_idx)
        self._load_page_if_needed(logical_index)
        # Synka sidebar-markering
        for item_id, idx in self._nav_index_map.items():
            if idx == logical_index:
                # Hitta QTreeWidgetItem via id
                iterator = self._iterate_tree_items()
                for tree_item in iterator:
                    if id(tree_item) == item_id:
                        self.nav_tree.blockSignals(True)
                        self.nav_tree.setCurrentItem(tree_item)
                        self.nav_tree.blockSignals(False)
                        return

    def _iterate_tree_items(self):
        """Iterera alla items i nav-trädet."""
        def _recurse(item):
            yield item
            for i in range(item.childCount()):
                yield from _recurse(item.child(i))
        for i in range(self.nav_tree.topLevelItemCount()):
            yield from _recurse(self.nav_tree.topLevelItem(i))

    def _load_page_if_needed(self, index: int):
        """Lazy-load page content on first visit."""
        if self._tabs_loaded.get(index, False):
            return

        if index not in self._tab_containers:
            return

        self.statusBar().showMessage(f"Loading...")
        QApplication.processEvents()

        tab_creators = {
            1: self.create_arbitrage_scanner_tab,
            2: self.create_ou_analytics_tab,
            3: self.create_signals_tab,
            4: self.create_portfolio_tab,
            5: self.create_markov_chains_tab,
            10: self.create_eps_reversion_tab,
            11: self.create_master_scanner_tab,
            12: self.create_squeeze_tab,
        }

        if index in tab_creators:
            container = self._tab_containers[index]
            layout = container.layout()

            placeholder = container.findChild(QLabel, "placeholder")
            if placeholder:
                placeholder.deleteLater()

            content = tab_creators[index]()
            content_layout = content.layout()
            if content_layout:
                while content_layout.count():
                    item = content_layout.takeAt(0)
                    if item.widget():
                        layout.addWidget(item.widget())
                    elif item.layout():
                        layout.addLayout(item.layout())

            self._tabs_loaded[index] = True

            if self.engine is not None:
                if index == 1:
                    n_tickers = len(self.engine.price_data.columns) if self.engine.price_data is not None else 0
                    n_pairs = len(self.engine.pairs_stats) if self.engine.pairs_stats is not None else 0
                    n_viable = len(self.engine.viable_pairs) if self.engine.viable_pairs is not None else 0
                    self._update_metric_value(self.tickers_metric, str(n_tickers))
                    self._update_metric_value(self.pairs_metric, str(n_pairs))
                    self._update_metric_value(self.viable_metric, str(n_viable))
                    self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
                    self.update_viable_table()
                    self.update_all_pairs_table()
                elif index == 2:
                    self.update_ou_pair_list()
                elif index == 3:
                    self.update_signals_list()

            if index == 4:
                self.update_portfolio_display()

            if index == 10 and self._eps_mr_result is not None:
                self._on_eps_mr_result(self._eps_mr_result)

            self.statusBar().showMessage("Ready")
    
    # ========================================================================
    # MASTER SCANNER
    # ========================================================================

    def create_master_scanner_tab(self) -> QWidget:
        """Aggregerad signalöversikt — samlar signaler från alla strategier."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

        # Header row
        header_row = QHBoxLayout()
        title = QLabel("MASTER SCANNER")
        title.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 2px;
            font-family: 'JetBrains Mono', monospace;
        """)
        header_row.addWidget(title)
        header_row.addStretch()

        # Refresh-knapp
        self.scanner_refresh_btn = QPushButton("⟳  SCAN ALL")
        self.scanner_refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_darkest']};
                border: none;
                border-radius: 4px;
                padding: 8px 20px;
                font-size: 11px;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_bright']};
            }}
        """)
        self.scanner_refresh_btn.clicked.connect(self._refresh_master_scanner)
        header_row.addWidget(self.scanner_refresh_btn)
        main_layout.addLayout(header_row)

        # Summary cards row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)
        self._scanner_total_card = self._create_scanner_summary_card("TOTAL SIGNALS", "0")
        self._scanner_buy_card = self._create_scanner_summary_card("BUY / LONG", "0", COLORS['positive'])
        self._scanner_sell_card = self._create_scanner_summary_card("SELL / SHORT", "0", COLORS['negative'])
        self._scanner_neutral_card = self._create_scanner_summary_card("NEUTRAL", "0", COLORS['text_muted'])
        cards_row.addWidget(self._scanner_total_card)
        cards_row.addWidget(self._scanner_buy_card)
        cards_row.addWidget(self._scanner_sell_card)
        cards_row.addWidget(self._scanner_neutral_card)
        main_layout.addLayout(cards_row)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)

        filter_label = QLabel("FILTER:")
        filter_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; font-weight: 600;")
        filter_row.addWidget(filter_label)

        self._scanner_strategy_filter = QComboBox()
        self._scanner_strategy_filter.addItems([
            "All Strategies", "Pairs", "EPS Reversion", "Markov", "TTM Squeeze"
        ])
        self._scanner_strategy_filter.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
                font-family: 'JetBrains Mono', monospace;
                min-width: 130px;
            }}
        """)
        self._scanner_strategy_filter.currentTextChanged.connect(self._apply_scanner_filter)
        filter_row.addWidget(self._scanner_strategy_filter)

        self._scanner_direction_filter = QComboBox()
        self._scanner_direction_filter.addItems(["All Directions", "LONG", "SHORT", "NEUTRAL"])
        self._scanner_direction_filter.setStyleSheet(self._scanner_strategy_filter.styleSheet())
        self._scanner_direction_filter.currentTextChanged.connect(self._apply_scanner_filter)
        filter_row.addWidget(self._scanner_direction_filter)

        self._scanner_min_score = QDoubleSpinBox()
        self._scanner_min_score.setRange(0.0, 1.0)
        self._scanner_min_score.setSingleStep(0.1)
        self._scanner_min_score.setValue(0.0)
        self._scanner_min_score.setPrefix("Min Score: ")
        self._scanner_min_score.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
                font-family: 'JetBrains Mono', monospace;
                min-width: 120px;
            }}
        """)
        self._scanner_min_score.valueChanged.connect(self._apply_scanner_filter)
        filter_row.addWidget(self._scanner_min_score)
        filter_row.addStretch()
        main_layout.addLayout(filter_row)

        # Signal table
        self.scanner_table = QTableWidget()
        self.scanner_table.setColumnCount(7)
        self.scanner_table.setHorizontalHeaderLabels([
            "Ticker/Pair", "Strategy", "Signal", "Score",
            "Direction", "Details", "Time"
        ])
        self.scanner_table.horizontalHeader().setStyleSheet(f"""
            QHeaderView::section {{
                background-color: {COLORS['bg_darkest']};
                color: {COLORS['accent']};
                border: none;
                border-bottom: 1px solid {COLORS['border_subtle']};
                padding: 8px 6px;
                font-size: 10px;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                letter-spacing: 1px;
                text-transform: uppercase;
            }}
        """)
        self.scanner_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_dark']};
                gridline-color: {COLORS['border_subtle']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                color: {COLORS['text_primary']};
            }}
            QTableWidget::item {{
                padding: 6px 8px;
                border-bottom: 1px solid {COLORS['bg_elevated']};
            }}
            QTableWidget::item:selected {{
                background-color: rgba(212, 165, 116, 0.15);
            }}
            QTableWidget::item:hover {{
                background-color: {COLORS['bg_hover']};
            }}
        """)
        self.scanner_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.scanner_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.scanner_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scanner_table.verticalHeader().setVisible(False)
        self.scanner_table.horizontalHeader().setStretchLastSection(True)
        self.scanner_table.setSortingEnabled(True)

        # Kolumn-bredder
        header = self.scanner_table.horizontalHeader()
        header.resizeSection(0, 130)   # Ticker/Pair
        header.resizeSection(1, 120)   # Strategy
        header.resizeSection(2, 140)   # Signal
        header.resizeSection(3, 70)    # Score
        header.resizeSection(4, 90)    # Direction
        header.resizeSection(5, 250)   # Details
        # Time stretches

        # Dubbelklick → navigera till strategi
        self.scanner_table.doubleClicked.connect(self._on_scanner_row_activated)

        main_layout.addWidget(self.scanner_table, 1)

        # Status label
        self.scanner_status = QLabel("Click SCAN ALL to collect signals from all strategies")
        self.scanner_status.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            padding: 4px;
        """)
        main_layout.addWidget(self.scanner_status)

        # Intern signal-cache
        self._scanner_signals = []

        # Ladda cachade resultat från disk
        QTimer.singleShot(100, self._load_scanner_cache)

        return tab

    def _create_scanner_summary_card(self, label: str, value: str, color: str = None) -> QFrame:
        """Liten summary-card för Master Scanner."""
        card = QFrame()
        color = color or COLORS['accent']
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 10px 16px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 1.2px;
            font-family: 'JetBrains Mono', monospace;
        """)
        layout.addWidget(lbl)

        val = QLabel(value)
        val.setObjectName("value")
        val.setStyleSheet(f"""
            color: {color};
            font-size: 22px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        """)
        layout.addWidget(val)

        return card

    def _refresh_master_scanner(self):
        """Kör alla tillgängliga strategier, samla signaler när klara.

        Om prisdata saknas → ladda ner automatiskt från CSV-tickerlistan.
        """
        print(f"\n{'='*70}")
        print(f"[SCANNER] === MASTER SCANNER: SCAN ALL started ===")
        print(f"[SCANNER] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        self.scanner_refresh_btn.setEnabled(False)
        self.scanner_status.setText("Initializing scan...")
        QApplication.processEvents()

        has_data = self.engine is not None and self.engine.price_data is not None
        print(f"[SCANNER] Engine exists: {self.engine is not None}")
        if has_data:
            print(f"[SCANNER] Price data: {len(self.engine.price_data.columns)} tickers loaded")
        else:
            print(f"[SCANNER] Price data: NONE — will auto-download")

        if has_data:
            # Prisdata finns redan → kör strategier direkt
            self._launch_all_strategies()
        else:
            # Prisdata saknas → ladda ner först
            self._download_price_data_for_scanner()

    def _download_price_data_for_scanner(self):
        """Ladda ner prisdata från CSV-tickerlistan innan strategier körs."""
        print(f"[SCANNER] Loading ticker list from: {SCHEDULED_CSV_PATH}")

        if not os.path.exists(SCHEDULED_CSV_PATH):
            msg = f"Ticker CSV not found: {SCHEDULED_CSV_PATH}"
            print(f"[SCANNER] ERROR: {msg}")
            self.scanner_status.setText(msg)
            self.scanner_refresh_btn.setEnabled(True)
            return

        tickers = load_tickers_from_csv(SCHEDULED_CSV_PATH)
        if not tickers:
            msg = "No tickers found in CSV"
            print(f"[SCANNER] ERROR: {msg}")
            self.scanner_status.setText(msg)
            self.scanner_refresh_btn.setEnabled(True)
            return

        print(f"[SCANNER] Loaded {len(tickers)} tickers from CSV")
        self.scanner_status.setText(f"Downloading price data for {len(tickers)} tickers...")

        # Kör nedladdning i bakgrundstråd
        self._scanner_fetch_thread = QThread()
        self._scanner_fetch_worker = PriceDataWorker(tickers, '2y')
        self._scanner_fetch_worker.moveToThread(self._scanner_fetch_thread)

        self._scanner_fetch_thread.started.connect(self._scanner_fetch_worker.run)
        self._scanner_fetch_worker.progress.connect(self._on_scanner_fetch_progress)
        self._scanner_fetch_worker.result.connect(self._on_scanner_fetch_result)
        self._scanner_fetch_worker.error.connect(self._on_scanner_fetch_error)
        self._scanner_fetch_worker.finished.connect(self._scanner_fetch_thread.quit)

        self._scanner_fetch_thread.start()

    def _on_scanner_fetch_progress(self, pct, msg):
        self.scanner_status.setText(f"[{pct}%] {msg}")

    def _on_scanner_fetch_error(self, msg):
        print(f"[SCANNER] Price data fetch ERROR: {msg}")
        self.scanner_status.setText(f"Error loading data: {msg}")
        self.scanner_refresh_btn.setEnabled(True)

    def _on_scanner_fetch_result(self, engine):
        """Prisdata nedladdad — spara engine och kör strategier."""
        n_tickers = len(engine.price_data.columns) if engine.price_data is not None else 0
        print(f"[SCANNER] Price data loaded: {n_tickers} tickers")

        # Spara engine om det inte redan finns en (bevarar existerande pairs-analys)
        if self.engine is None:
            self.engine = engine
            print(f"[SCANNER] New engine created and assigned")
        elif self.engine.price_data is None:
            self.engine.price_data = engine.price_data
            self.engine.raw_price_data = engine.price_data.copy()
            print(f"[SCANNER] Price data assigned to existing engine")

        # Navigera tillbaka till Master Scanner
        self.navigate_to_page(11)
        self._launch_all_strategies()

    def _launch_all_strategies(self):
        """Starta alla strategi-analyser."""
        print(f"\n[SCANNER] --- Launching all strategies ---")
        launched = []

        # Spara nuvarande sida (Master Scanner) för att återställa efter lazy-loading
        scanner_stack_idx = self.content_stack.currentIndex()

        # Strategier som behöver engine.price_data
        strategies_needing_data = {
            12: ('TTM Squeeze', 'run_squeeze_analysis', '_squeeze_running'),
        }
        # VRP och EPS startas EFTER Markov (undviker concurrent yfinance-downloads)

        all_strategies = {**strategies_needing_data}

        for idx, (name, method_name, running_attr) in all_strategies.items():
            # Hoppa över om redan har cachade resultat
            result_attrs = {10: '_eps_mr_result', 12: '_squeeze_result'}
            if getattr(self, result_attrs.get(idx, ''), None) is not None:
                print(f"[SCANNER]   {name}: SKIPPED (cached result exists)")
                launched.append(f"{name} (cached)")
                continue

            if getattr(self, running_attr, False):
                print(f"[SCANNER]   {name}: SKIPPED (already running)")
                launched.append(f"{name} (running)")
                continue

            # Lazy-load tabben om den inte är laddad (widgets krävs)
            if not self._tabs_loaded.get(idx, False):
                print(f"[SCANNER]   {name}: Lazy-loading tab {idx}...")
                self._load_page_if_needed(idx)
                QApplication.processEvents()

            method = getattr(self, method_name, None)
            if method:
                try:
                    print(f"[SCANNER]   {name}: Starting analysis...")
                    method()
                    launched.append(name)
                    print(f"[SCANNER]   {name}: LAUNCHED OK")
                except Exception as e:
                    print(f"[SCANNER]   {name}: ERROR — {e}")
                    import traceback
                    traceback.print_exc()
                    launched.append(f"{name} (error)")

        # ── Markov batch: kör på ett urval av tickers ──
        if not self._markov_batch_running:
            markov_tickers = self._get_markov_scan_tickers()
            if markov_tickers and MARKOV_AVAILABLE:
                print(f"[SCANNER]   Markov: Starting batch analysis on {len(markov_tickers)} tickers...")
                self._markov_batch_running = True
                self._markov_batch_thread = QThread()
                self._markov_batch_worker = MarkovBatchWorker(markov_tickers)
                self._markov_batch_worker.moveToThread(self._markov_batch_thread)
                self._markov_batch_thread.started.connect(self._markov_batch_worker.run)
                self._markov_batch_worker.progress.connect(self._on_scanner_fetch_progress)
                self._markov_batch_worker.result.connect(self._on_markov_batch_result)
                self._markov_batch_worker.error.connect(self._on_markov_batch_error)
                self._markov_batch_worker.finished.connect(self._markov_batch_thread.quit)
                self._markov_batch_thread.start()
                launched.append('Markov')
                markov_will_callback = True
            else:
                print(f"[SCANNER]   Markov: SKIPPED ({'already running' if self._markov_batch_running else 'not available'})")
                markov_will_callback = False
        else:
            launched.append('Markov (running)')
            markov_will_callback = True  # already running, will callback when done

        # ── EPS: startas EFTER Markov (undviker concurrent yfinance-downloads) ──
        self._yf_queue = []  # (name, method_name, tab_idx)

        # EPS
        eps_idx = 10
        if getattr(self, '_eps_mr_result', None) is not None:
            print(f"[SCANNER]   EPS Reversion: SKIPPED (cached result exists)")
            launched.append('EPS Reversion (cached)')
        elif self._eps_mr_running:
            print(f"[SCANNER]   EPS Reversion: SKIPPED (already running)")
            launched.append('EPS Reversion (running)')
        else:
            if not self._tabs_loaded.get(eps_idx, False):
                print(f"[SCANNER]   EPS Reversion: Lazy-loading tab {eps_idx}...")
                self._load_page_if_needed(eps_idx)
                QApplication.processEvents()
            self._yf_queue.append(('EPS Reversion', 'run_eps_mr_analysis', eps_idx))
            launched.append('EPS Reversion (queued)')

        self._eps_pending_scanner_launch = len(self._yf_queue) > 0
        if self._yf_queue:
            queued_names = [q[0] for q in self._yf_queue]
            if markov_will_callback:
                print(f"[SCANNER]   yfinance queue: {queued_names} — will start after Markov completes")
            else:
                print(f"[SCANNER]   yfinance queue: {queued_names} — starting immediately (no Markov)")
                self._launch_next_yf_strategy()

        # Återställ synlig sida till Master Scanner
        self.content_stack.setCurrentIndex(scanner_stack_idx)

        print(f"[SCANNER] Launched: {launched}")
        self.scanner_status.setText(f"Running: {', '.join(launched)}...")

        # Starta poll-timer med timeout (max 5 min)
        self._scanner_pending = True
        self._scanner_start_time = time.time()
        if hasattr(self, '_scanner_check_timer') and self._scanner_check_timer.isActive():
            self._scanner_check_timer.stop()
        self._scanner_check_timer = QTimer(self)
        self._scanner_check_timer.timeout.connect(self._check_scanner_complete)
        self._scanner_check_timer.start(1500)

    def _get_markov_scan_tickers(self) -> list:
        """Välj ett urval tickers för Markov batch-scan.

        Använder index + mest likvida aktier från CSV.
        """
        # Populära index och storbolag
        core_tickers = [
            '^GSPC', '^OMX', '^IXIC', '^DJI',
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'VOLV-B.ST', 'ERIC-B.ST', 'SEB-A.ST', 'SHB-A.ST', 'ABB.ST',
            'SAND.ST', 'HM-B.ST', 'ATLAS-A.ST', 'INVE-B.ST',
        ]
        # Lägg till tickers som har engine price_data (dvs redan nedladdade)
        if self.engine is not None and self.engine.price_data is not None:
            available = list(self.engine.price_data.columns)
            # Filtrera core_tickers till de som finns i data
            valid_core = [t for t in core_tickers if t in available]
            # Begränsa till max 25 tickers totalt för rimlig tid
            return valid_core[:25]
        return core_tickers[:15]

    def _on_markov_batch_result(self, results):
        """Markov batch klar — spara resultat och starta yfinance-kön."""
        self._markov_batch_results = results
        self._markov_batch_running = False
        print(f"[SCANNER] Markov batch complete: {len(results)} tickers analyzed")
        if getattr(self, '_eps_pending_scanner_launch', False):
            self._launch_next_yf_strategy()

    def _on_markov_batch_error(self, msg):
        """Markov batch error."""
        self._markov_batch_running = False
        print(f"[SCANNER] Markov batch ERROR: {msg}")
        if getattr(self, '_eps_pending_scanner_launch', False):
            self._launch_next_yf_strategy()

    def _launch_next_yf_strategy(self):
        """Startar nästa strategi i yfinance-kön (sekventiell för att undvika concurrent downloads)."""
        if not hasattr(self, '_yf_queue') or not self._yf_queue:
            self._eps_pending_scanner_launch = False
            print(f"[SCANNER]   yfinance queue: Empty — all launched")
            return

        name, method_name, tab_idx = self._yf_queue.pop(0)
        print(f"[SCANNER]   {name}: Starting analysis (from yfinance queue)...")
        try:
            method = getattr(self, method_name, None)
            if method:
                method()
                print(f"[SCANNER]   {name}: LAUNCHED OK")
            else:
                print(f"[SCANNER]   {name}: Method {method_name} not found!")
        except Exception as e:
            print(f"[SCANNER]   {name}: LAUNCH ERROR — {e}")
            import traceback
            traceback.print_exc()

    def _check_scanner_complete(self):
        """Poll om alla strategi-analyser är klara. Timeout efter 5 min."""
        elapsed = time.time() - self._scanner_start_time
        still_running = []
        for name, attr in [
            ('EPS MR', '_eps_mr_running'),
            ('TTM Squeeze', '_squeeze_running'),
            ('Markov', '_markov_batch_running'),
        ]:
            if getattr(self, attr, False):
                still_running.append(name)

        if still_running and elapsed < 300:  # 5 min timeout (EPS startar efter Markov)
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self.scanner_status.setText(f"Waiting for: {', '.join(still_running)}... ({mins}:{secs:02d})")
        else:
            if still_running:
                print(f"[SCANNER] TIMEOUT after {elapsed:.0f}s — still running: {still_running}")
                # Force-reset stuck flags
                for name, attr in [
                    ('EPS MR', '_eps_mr_running'),
                    ('TTM Squeeze', '_squeeze_running'),
                    ('Markov', '_markov_batch_running'),
                ]:
                    if getattr(self, attr, False):
                        print(f"[SCANNER]   Force-resetting {name}")
                        setattr(self, attr, False)

            self._scanner_check_timer.stop()
            self._scanner_pending = False
            self.scanner_refresh_btn.setEnabled(True)
            print(f"[SCANNER] All strategies complete ({elapsed:.0f}s) — collecting signals...")
            self._collect_scanner_signals()

    def _collect_scanner_signals(self):
        """Samla signaler från alla cachade strategi-resultat."""
        print(f"\n[SCANNER] --- Collecting signals ---")
        self.scanner_status.setText("Collecting signals...")
        QApplication.processEvents()

        signals = []
        now_str = datetime.now().strftime("%H:%M")

        # ── 1. Pairs signals ──
        print(f"[SCANNER]   Pairs: engine={self.engine is not None}, "
              f"viable_pairs={self.engine.viable_pairs is not None if self.engine else 'N/A'}")
        if self.engine is not None and self.engine.viable_pairs is not None:
            vp = self.engine.viable_pairs
            n_pairs_checked = 0
            n_pairs_signal = 0
            for _, row in vp.iterrows():
                pair = row.get('pair', '')
                z = row.get('z_score', 0)
                opt_z = row.get('optimal_z', 2.0)
                hl = row.get('half_life_days', 0)
                hurst = row.get('hurst_exponent', 0.5)
                n_pairs_checked += 1

                # Strikt: enbart par med |z| >= optimal_z
                if abs(z) >= opt_z:
                    direction = "SHORT" if z > 0 else "LONG"
                    score = min(abs(z) / (opt_z * 2), 1.0)
                    signals.append({
                        'ticker': pair,
                        'strategy': 'Pairs',
                        'signal': f'ENTRY Z = {z:+.2f} (opt: {opt_z:.1f})',
                        'score': round(score, 2),
                        'direction': direction,
                        'details': f'HL {hl:.0f}d | Hurst {hurst:.2f}',
                        'time': now_str,
                        'nav_index': 1,
                    })
                    n_pairs_signal += 1
            print(f"[SCANNER]   Pairs: {n_pairs_checked} viable, {n_pairs_signal} with entry signal")

        # ── 2. EPS Mean Reversion signals ──
        print(f"[SCANNER]   EPS MR: result={self._eps_mr_result is not None}")
        if self._eps_mr_result is not None:
            try:
                data, df, pe_analyses = self._eps_mr_result
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        sig = str(row.get('Signal', 'Neutral'))
                        if sig == 'Neutral' or not sig:
                            continue
                        ticker = str(row.get('Ticker', '?'))
                        try:
                            spr_z = float(row.get('Spr Z', 0) or 0)
                        except (ValueError, TypeError):
                            spr_z = 0.0
                        try:
                            adf_p = float(row.get('ADF p', 1.0) or 1.0)
                        except (ValueError, TypeError):
                            adf_p = 1.0
                        try:
                            hl = float(row.get('HL(d)', 0) or 0)
                        except (ValueError, TypeError):
                            hl = 0.0
                        passes = str(row.get('Pass', '0/4'))
                        direction = "LONG" if 'ndervärderad' in sig else "SHORT"
                        score = min(abs(spr_z) / 3.0, 1.0)
                        print(f"[SCANNER]   EPS MR signal: {ticker} {sig} Z={spr_z:+.2f} dir={direction} score={score:.2f}")
                        signals.append({
                            'ticker': ticker,
                            'strategy': 'EPS Reversion',
                            'signal': sig,
                            'score': round(score, 2),
                            'direction': direction,
                            'details': f'Z {spr_z:+.2f} | ADF {adf_p:.3f} | HL {hl:.0f}d | {passes}',
                            'time': now_str,
                            'nav_index': 10,
                        })
            except Exception as e:
                print(f"[SCANNER]   EPS MR error: {e}")

        # ── 7. Markov signals (batch) ──
        markov_sources = {}
        # Inkludera enskilt resultat (om manuell analys körts)
        if self._markov_result is not None:
            t = getattr(self._markov_result, 'ticker', '?')
            markov_sources[t] = self._markov_result
        # Inkludera batch-resultat
        if self._markov_batch_results:
            markov_sources.update(self._markov_batch_results)
        print(f"[SCANNER]   Markov: {len(markov_sources)} tickers analyzed")

        n_markov_sig = 0
        for ticker, res in markov_sources.items():
            state = getattr(res, 'current_state', -1)
            forecast = getattr(res, 'forecast_probs', None)
            exp_ret = getattr(res, 'expected_return', 0) or 0

            if forecast is not None and len(forecast) >= 2:
                max_prob = max(forecast)
                # Strikt: bara visa starka signaler (P > 0.65)
                if max_prob > 0.65:
                    direction = "LONG" if state == 1 else "SHORT"
                    state_name = "POS" if state == 1 else "NEG"
                    score = round(max_prob, 2)
                    signals.append({
                        'ticker': ticker,
                        'strategy': 'Markov',
                        'signal': f'State: {state_name}',
                        'score': score,
                        'direction': direction,
                        'details': f'E[r] {exp_ret:+.2%} | P(next) {forecast[0]:.0%}/{forecast[1]:.0%}',
                        'time': now_str,
                        'nav_index': 5,
                    })
                    n_markov_sig += 1
        print(f"[SCANNER]   Markov: {n_markov_sig} signals (P > 0.65)")

        # ── 8. TTM Squeeze signals ──
        print(f"[SCANNER]   TTM Squeeze: result={self._squeeze_result is not None}")
        if self._squeeze_result is not None:
            n_sqz_sig = 0
            for ticker, tr in self._squeeze_result.ticker_results.items():
                if tr.squeeze_on or tr.squeeze_firing:
                    sig_text = f"SQUEEZE ({tr.squeeze_days}d)"
                    score = round(tr.score / 100.0, 2)

                    # Lägg till straddle-info om tillgänglig
                    details = f"Sqz {tr.squeeze_days}d"
                    opts = self._options_data.get(ticker, {})
                    if opts and 'error' not in opts and opts.get('nearest_atm') is not None:
                        atm = opts['nearest_atm']
                        straddle = atm['Straddle'] if 'Straddle' in atm.index else None
                        cost_pct = atm['Cost_pct'] if 'Cost_pct' in atm.index else None
                        if straddle and not pd.isna(straddle):
                            details += f" | Straddle {straddle:.1f} ({cost_pct:.1f}%)"

                    signals.append({
                        'ticker': ticker,
                        'strategy': 'TTM Squeeze',
                        'signal': sig_text,
                        'score': score,
                        'direction': 'NEUTRAL',
                        'details': details,
                        'time': now_str,
                        'nav_index': 12,
                    })
                    n_sqz_sig += 1
            print(f"[SCANNER]   TTM Squeeze: {n_sqz_sig} signals (squeeze on/firing)")

        # Sortera efter score
        signals.sort(key=lambda s: s['score'], reverse=True)
        self._scanner_signals = signals
        self._populate_scanner_table(signals)

        # Uppdatera summary cards
        n_total = len(signals)
        n_buy = sum(1 for s in signals if s['direction'] == 'LONG')
        n_sell = sum(1 for s in signals if s['direction'] == 'SHORT')
        n_neutral = n_total - n_buy - n_sell

        print(f"\n[SCANNER] === RESULTS ===")
        print(f"[SCANNER] Total: {n_total} | Buy: {n_buy} | Sell: {n_sell} | Neutral: {n_neutral}")
        for s in signals[:20]:
            print(f"[SCANNER]   {s['direction']:6s} | {s['score']:.2f} | {s['strategy']:15s} | {s['ticker']:15s} | {s['signal']}")
        if len(signals) > 20:
            print(f"[SCANNER]   ... and {len(signals) - 20} more")
        print(f"[SCANNER] {'='*50}")

        self._scanner_total_card.findChild(QLabel, "value").setText(str(n_total))
        self._scanner_buy_card.findChild(QLabel, "value").setText(str(n_buy))
        self._scanner_sell_card.findChild(QLabel, "value").setText(str(n_sell))
        self._scanner_neutral_card.findChild(QLabel, "value").setText(str(n_neutral))

        strategies_with_data = set(s['strategy'] for s in signals)
        all_strategies = {'Pairs', 'EPS Reversion', 'Markov', 'TTM Squeeze'}
        missing = all_strategies - strategies_with_data
        if missing:
            status_msg = f"{n_total} signals found. No signals from: {', '.join(sorted(missing))}"
            print(f"[SCANNER] {status_msg}")
            self.scanner_status.setText(status_msg)
        else:
            self.scanner_status.setText(f"{n_total} signals found across all strategies")

        # Spara resultat till disk-cache
        self._save_scanner_cache(signals)

    # ── Scanner disk-cache ─────────────────────────────────────────────

    _SCANNER_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'scan_reports', 'scanner_cache.json')

    def _save_scanner_cache(self, signals: list):
        """Spara scanner-signaler till disk."""
        try:
            cache = {
                'timestamp': datetime.now().isoformat(),
                'signals': signals,
            }
            os.makedirs(os.path.dirname(self._SCANNER_CACHE_FILE), exist_ok=True)
            with open(self._SCANNER_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"[SCANNER] Cache saved: {len(signals)} signals → {self._SCANNER_CACHE_FILE}")
        except Exception as e:
            print(f"[SCANNER] Cache save error: {e}")

    def _load_scanner_cache(self):
        """Ladda cachade scanner-signaler från disk och visa i tabellen."""
        try:
            if not os.path.exists(self._SCANNER_CACHE_FILE):
                return
            with open(self._SCANNER_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)

            signals = cache.get('signals', [])
            timestamp = cache.get('timestamp', '')
            if not signals:
                return

            # Parsea timestamp (cache gäller tills nästa schemalagda scan)
            try:
                ts = datetime.fromisoformat(timestamp)
                age = datetime.now() - ts
                if age.days >= 1:
                    age_str = f"{ts.strftime('%Y-%m-%d %H:%M')}"
                else:
                    age_str = ts.strftime('%H:%M')
            except Exception:
                age_str = "?"

            self._scanner_signals = signals
            self._populate_scanner_table(signals)

            # Uppdatera summary cards
            n_total = len(signals)
            n_buy = sum(1 for s in signals if s['direction'] == 'LONG')
            n_sell = sum(1 for s in signals if s['direction'] == 'SHORT')
            n_neutral = n_total - n_buy - n_sell

            self._scanner_total_card.findChild(QLabel, "value").setText(str(n_total))
            self._scanner_buy_card.findChild(QLabel, "value").setText(str(n_buy))
            self._scanner_sell_card.findChild(QLabel, "value").setText(str(n_sell))
            self._scanner_neutral_card.findChild(QLabel, "value").setText(str(n_neutral))

            self.scanner_status.setText(f"Cached: {n_total} signals from {age_str}")
            print(f"[SCANNER] Cache loaded: {n_total} signals from {timestamp}")
        except Exception as e:
            print(f"[SCANNER] Cache load error: {e}")

    def _populate_scanner_table(self, signals: list):
        """Fyll scanner-tabellen med signaler."""
        self.scanner_table.setSortingEnabled(False)
        self.scanner_table.setRowCount(len(signals))

        for row, sig in enumerate(signals):
            # Ticker
            ticker_item = QTableWidgetItem(sig['ticker'])
            ticker_item.setForeground(QColor(COLORS['text_primary']))
            self.scanner_table.setItem(row, 0, ticker_item)

            # Strategy — färg per strategi
            strategy_colors = {
                'Pairs': '#d4a574',
                'EPS Reversion': '#f59e0b',
                'Markov': '#ec4899',
                'TTM Squeeze': '#f97316',
            }
            strat_item = QTableWidgetItem(sig['strategy'])
            strat_color = strategy_colors.get(sig['strategy'], COLORS['text_secondary'])
            strat_item.setForeground(QColor(strat_color))
            self.scanner_table.setItem(row, 1, strat_item)

            # Signal
            signal_item = QTableWidgetItem(sig['signal'])
            self.scanner_table.setItem(row, 2, signal_item)

            # Score — numerisk sortering (explicit float() för numpy-kompatibilitet)
            score_val = float(sig['score']) if sig['score'] is not None else 0.0
            score_item = QTableWidgetItem(f"{score_val:.2f}")
            score_item.setData(Qt.UserRole, score_val)
            if score_val >= 0.7:
                score_item.setForeground(QColor(COLORS['positive']))
            elif score_val >= 0.4:
                score_item.setForeground(QColor(COLORS['warning']))
            else:
                score_item.setForeground(QColor(COLORS['text_muted']))
            self.scanner_table.setItem(row, 3, score_item)

            # Direction
            dir_item = QTableWidgetItem(sig['direction'])
            if sig['direction'] == 'LONG':
                dir_item.setForeground(QColor(COLORS['positive']))
            elif sig['direction'] == 'SHORT':
                dir_item.setForeground(QColor(COLORS['negative']))
            else:
                dir_item.setForeground(QColor(COLORS['text_muted']))
            self.scanner_table.setItem(row, 4, dir_item)

            # Details
            detail_item = QTableWidgetItem(sig['details'])
            detail_item.setForeground(QColor(COLORS['text_secondary']))
            self.scanner_table.setItem(row, 5, detail_item)

            # Time
            time_item = QTableWidgetItem(sig['time'])
            time_item.setForeground(QColor(COLORS['text_muted']))
            self.scanner_table.setItem(row, 6, time_item)

            # Spara nav_index som data på första kolumnen
            ticker_item.setData(Qt.UserRole, sig['nav_index'])

        self.scanner_table.setSortingEnabled(True)

    def _apply_scanner_filter(self):
        """Filtrera scanner-tabellen baserat på valda filter."""
        if not hasattr(self, '_scanner_signals'):
            return
        strategy_filter = self._scanner_strategy_filter.currentText()
        direction_filter = self._scanner_direction_filter.currentText()
        min_score = self._scanner_min_score.value()

        filtered = []
        strategy_map = {
            'All Strategies': None,
            'Pairs': 'Pairs',
            'EPS Reversion': 'EPS Reversion',
            'Markov': 'Markov',
            'TTM Squeeze': 'TTM Squeeze',
        }
        direction_map = {
            'All Directions': None,
            'LONG': 'LONG',
            'SHORT': 'SHORT',
            'NEUTRAL': 'NEUTRAL',
        }

        strat_val = strategy_map.get(strategy_filter)
        dir_val = direction_map.get(direction_filter)

        for sig in self._scanner_signals:
            if strat_val and sig['strategy'] != strat_val:
                continue
            if dir_val and sig['direction'] != dir_val:
                continue
            if sig['score'] < min_score:
                continue
            filtered.append(sig)

        self._populate_scanner_table(filtered)

    def _on_scanner_row_activated(self, index):
        """Dubbelklick på scanner-rad → navigera till rätt strategi."""
        row = index.row()
        ticker_item = self.scanner_table.item(row, 0)
        if ticker_item is None:
            return
        nav_index = ticker_item.data(Qt.UserRole)
        if nav_index is not None:
            self.navigate_to_page(nav_index)

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
                    pass  # Treemap JS console message (suppressed)

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
        vol_cards_row.addWidget(self.vix_card, 1)

        self.vvix_card = VolatilityCard("^VVIX", "VVIX", "Vol of Vol")
        vol_cards_row.addWidget(self.vvix_card, 1)

        self.skew_card = VolatilityCard("^SKEW", "SKEW", "Tail Risk")
        vol_cards_row.addWidget(self.skew_card, 1)

        self.vvix_vix_card = VolatilityCard("VVIX/VIX", "VVIX/VIX", "Vol Ratio")
        vol_cards_row.addWidget(self.vvix_vix_card, 1)

        self.move_card = VolatilityCard("^MOVE", "MOVE", "Bond Vol")
        vol_cards_row.addWidget(self.move_card, 1)

        vol_section_layout.addLayout(vol_cards_row)
        center_layout.addWidget(vol_section)

        center_panel.setMinimumWidth(600)
        main_layout.addWidget(center_panel, stretch=7)

        # RIGHT: Tabbed panel — News | Earnings | IPOs | Splits | Economic Events
        self.news_feed = RightPanelWidget()
        self.news_feed.setMinimumWidth(300)
        self.news_feed.setMaximumWidth(420)
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
            # News skipped — yfinance busy
            return
        
        # Also check threads physically alive
        if (self._market_watch_thread is not None and self._market_watch_thread.isRunning()):
            # News skipped — market watch thread alive
            return
        if (self._volatility_thread is not None and self._volatility_thread.isRunning()):
            # News skipped — volatility thread alive
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
        
        # Period label
        period_group = QVBoxLayout()
        period_label = QLabel("PERIOD:")
        period_label.setStyleSheet("color: #d4a574; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 6px; border: none;")
        period_group.addWidget(period_label)
        period_value = QLabel("2 years")
        period_value.setStyleSheet("color: #e0e0e0; font-size: 13px; padding: 6px; background: #1a1a1a; border: 1px solid #333; border-radius: 4px;")
        period_group.addWidget(period_value)
        config_layout.addLayout(period_group)
        
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
            "• Half-life: 1-60 days",
            "• Engle-Granger p-value: \u2264 0.05",
            "• Johansen trace \u2265 critical value",
            "• Hurst exponent: \u2264 0.50",
            "• Correlation: \u2265 0.70",
            "\u2500\u2500 Kalman Validation \u2500\u2500",
            "• Param stability: > 0.40",
            "• Innovation ratio: [0.4, 2.5]",
            "• CUSUM regime: < 15.0",
            "• HMM P(MR): soft metric",
            "\u2500\u2500 Period \u2500\u2500",
            "• 2 years historical data",
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

        self.viable_table.setColumnCount(16)
        self.viable_table.setHorizontalHeaderLabels([
            "Pair", "Z-Score", "Half-life (days)", "EG p-value", "Johansen trace",
            "Hurst", "Frac.d", "Correlation",
            "Kalman Stab.", "Innov. Ratio", "Regime Score", "P(MR)", "HMM State",
            "\u03b8 Sig.", "Tail Dep", "Opt.Z*"
        ])
        # Tooltips for scanner columns
        _scanner_tips = [
            "Stock pair Y/X. Spread = Y - \u03b2\u00b7X - \u03b1",
            "Composite quality score [0-1].\nWeighted: IR(30%), WinProb(20%), Hurst(15%),\nKalman(15%), Robustness(10%), HL(10%).",
            "Current Z-score of the spread.\n|Z| > 2.0 = signal (highlighted).\nComputed from shortest passing window.",
            "Days for spread to move halfway to equilibrium.\nln(2)/\u03b8 \u00d7 252. Ideal: 5-60 days.",
            "Engle-Granger cointegration p-value.\np < 0.05 = significant mean reversion.",
            "Johansen trace statistic.\nHigher = stronger cointegration evidence.",
            "Hurst exponent. H < 0.5 = mean-reverting (good).\nH \u2248 0.5 = random walk. H > 0.5 = trending.",
            "Fractional integration parameter d.\nd < 0 = strong MR, 0-0.5 = weak MR,\n0.5-1 = borderline, > 1 = non-stationary.",
            "Pearson correlation between Y and X.\nHigher = more stable hedge. > 0.8 desirable.",
            "Windows passed / windows tested.\nHigher = more robust across time periods.",
            "Kalman parameter stability [0-1].\n1 = perfectly stable \u03b8. > 0.5 required.",
            "Normalized innovation ratio.\nShould be \u22481.0 (valid range [0.5, 2.0]).",
            "CUSUM regime change score.\n< 15.0 = no structural break detected.",
            "HMM probability of mean-reverting regime.\n\u2265 0.7 = strong MR, \u2265 0.4 = moderate, < 0.3 = fail.",
            "HMM 3-state regime: MR = Mean-Reverting,\nTR = Trending, CR = Crisis.",
            "Is \u03b8 (mean-reversion speed)\nstatistically significant at 95% CI?",
            "Lower tail dependence (Student-t copula).\nHigher = more co-crashes. Symmetric for t-copula.",
            "Optimal entry z-score that maximizes\nexpected profit per trading day.",
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

        # (Window breakdown removed — single 2y period)

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
        self._ou_pair_debounce = QTimer()
        self._ou_pair_debounce.setSingleShot(True)
        self._ou_pair_debounce.setInterval(400)  # ms
        self._ou_pair_debounce.timeout.connect(self._on_ou_pair_debounced)
        self.ou_pair_combo.currentTextChanged.connect(self._on_ou_pair_typing)
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
        # LEFT SIDE: Compact Metric Cards in grid layout (like Pair Signals)
        # =====================================================================
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
        left_layout.setSpacing(10)

        def _ou_divider():
            d = QFrame()
            d.setFrameShape(QFrame.HLine)
            d.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
            d.setMaximumHeight(1)
            return d

        def _ou_header(text):
            h = QLabel(text)
            h.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 700; letter-spacing: 1px; border: none;")
            return h

        # === OU DETAILS ===
        left_layout.addWidget(_ou_header("OU DETAILS"))

        ou_grid = QGridLayout()
        ou_grid.setSpacing(8)

        self.ou_theta_card = CompactMetricCard("MEAN REVERSION (θ)", "-",
            tooltip="θ — Speed of mean reversion (annualized).\nHigher = faster reversion. Typical: 5-100.")
        self.ou_mu_card = CompactMetricCard("MEAN SPREAD (μ)", "-",
            tooltip="μ — Long-term equilibrium spread level.\nDerived from Kalman state: a/(1-b).")
        self.ou_halflife_card = CompactMetricCard("HALF-LIFE", "-",
            tooltip="Trading days to move halfway to equilibrium.\nln(2)/θ × 252. Ideal for trading: 5-60 days.")
        self.ou_zscore_card = CompactMetricCard("CURRENT Z-SCORE", "-",
            tooltip="Std deviations from equilibrium: (S-μ)/σ_eq.\n|z|>2 = entry signal. Positive = above mean.")
        self.ou_hedge_card = CompactMetricCard("BETA (β)", "-",
            tooltip="Hedge ratio β from EG regression: Y = α+β·X+ε.\nFor 1 share Y, short β shares X.")
        self.ou_status_card = CompactMetricCard("STATUS", "-",
            tooltip="VIABLE = passes all tests:\n• EG p<0.05 • Hurst<0.5 • HL 1-252d")

        ou_grid.addWidget(self.ou_theta_card, 0, 0)
        ou_grid.addWidget(self.ou_mu_card, 0, 1)
        ou_grid.addWidget(self.ou_halflife_card, 0, 2)
        ou_grid.addWidget(self.ou_zscore_card, 1, 0)
        ou_grid.addWidget(self.ou_hedge_card, 1, 1)
        ou_grid.addWidget(self.ou_status_card, 1, 2)
        left_layout.addLayout(ou_grid)

        # === KALMAN FILTER ===
        left_layout.addWidget(_ou_divider())
        left_layout.addWidget(_ou_header("KALMAN FILTER"))

        kalman_grid = QGridLayout()
        kalman_grid.setSpacing(8)

        self.kalman_stability_card = CompactMetricCard("PARAM STABILITY", "-",
            tooltip="θ stability over last 60 days (1-CV).\n>0.7 good, 0.4-0.7 caution, <0.4 unstable.")
        self.kalman_ess_card = CompactMetricCard("EFFECTIVE N", "-",
            tooltip="Effective sample size adjusted for autocorrelation.\nLower than N = redundant observations.")
        self.kalman_theta_ci_card = CompactMetricCard("θ (95% CI)", "-",
            tooltip="95% CI for half-life (days) from Kalman covariance.\n'inf' = cannot exclude random walk (θ≈0).")
        self.kalman_mu_ci_card = CompactMetricCard("μ (95% CI)", "-",
            tooltip="95% CI for equilibrium spread level.\nWide CI = uncertain reversion target.")
        self.kalman_innovation_card = CompactMetricCard("INNOV. RATIO", "-",
            tooltip="Actual/expected innovation variance. Should ≈ 1.0.\n>1.5 = model underestimates uncertainty.")
        self.kalman_regime_card = CompactMetricCard("REGIME CHANGE", "-",
            tooltip="CUSUM on innovations. Threshold=4.0.\nHigh = structural break, parameters unreliable.")

        kalman_grid.addWidget(self.kalman_stability_card, 0, 0)
        kalman_grid.addWidget(self.kalman_ess_card, 0, 1)
        kalman_grid.addWidget(self.kalman_theta_ci_card, 0, 2)
        kalman_grid.addWidget(self.kalman_mu_ci_card, 1, 0)
        kalman_grid.addWidget(self.kalman_innovation_card, 1, 1)
        kalman_grid.addWidget(self.kalman_regime_card, 1, 2)
        left_layout.addLayout(kalman_grid)

        # === EXPECTED MOVE ===
        left_layout.addWidget(_ou_divider())
        left_layout.addWidget(_ou_header("EXPECTED MOVE"))

        exp_grid = QGridLayout()
        exp_grid.setSpacing(8)

        self.exp_spread_change_card = CompactMetricCard("Δ SPREAD", "-",
            tooltip="Expected spread change over 1 half-life.\nBased on OU: E[S_t] = μ + (S₀-μ)·e^(-θt).")
        self.exp_y_only_card = CompactMetricCard("Y (100%)", "-",
            tooltip="Y price move if Y absorbs 100% of convergence.")
        self.exp_x_only_card = CompactMetricCard("X (100%)", "-",
            tooltip="X price move if X absorbs 100% of convergence.")

        exp_grid.addWidget(self.exp_spread_change_card, 0, 0)
        exp_grid.addWidget(self.exp_y_only_card, 0, 1)
        exp_grid.addWidget(self.exp_x_only_card, 0, 2)
        left_layout.addLayout(exp_grid)

        # === GARCH + TAIL + FRACTIONAL + HEDGE (compact combined row) ===
        left_layout.addWidget(_ou_divider())
        left_layout.addWidget(_ou_header("GARCH VOLATILITY"))

        garch_grid = QGridLayout()
        garch_grid.setSpacing(8)

        self.garch_alpha_card = CompactMetricCard("ARCH (α)", "-",
            tooltip="News impact coefficient. How much a single surprise\naffects next-period variance. Typical: 0.03-0.15.")
        self.garch_beta_card = CompactMetricCard("GARCH (β)", "-",
            tooltip="Persistence coefficient. How much past variance\npersists. α+β close to 1 = high persistence.")
        self.garch_persist_card = CompactMetricCard("PERSISTENCE", "-",
            tooltip="α+β. < 0.90 = low persistence (vol shocks die quickly).\n> 0.98 = near-IGARCH (almost unit root in variance).")
        self.garch_cvol_card = CompactMetricCard("CURRENT VOL", "-",
            tooltip="Current conditional vol / long-run vol.\n1.0 = normal. >1.5 = elevated. <0.7 = calm.")

        garch_grid.addWidget(self.garch_alpha_card, 0, 0)
        garch_grid.addWidget(self.garch_beta_card, 0, 1)
        garch_grid.addWidget(self.garch_persist_card, 0, 2)
        garch_grid.addWidget(self.garch_cvol_card, 0, 3)
        left_layout.addLayout(garch_grid)

        # === TAIL DEPENDENCE ===
        left_layout.addWidget(_ou_divider())
        left_layout.addWidget(_ou_header("TAIL DEPENDENCE"))

        tail_grid = QGridLayout()
        tail_grid.setSpacing(8)

        self.tail_lower_card = CompactMetricCard("λ LOWER", "-",
            tooltip="Lower tail dependence (Student-t copula).\nP(both crash together). > 0.3 = high co-crash risk.")
        self.tail_upper_card = CompactMetricCard("λ UPPER", "-",
            tooltip="Upper tail dependence. Symmetric for t-copula.\n> 0.3 = high co-rally probability.")
        self.tail_asym_card = CompactMetricCard("ASYMMETRY", "-",
            tooltip="λ_upper - λ_lower. Non-zero = asymmetric tail risk.\nPositive = more co-rallies than co-crashes.")

        tail_grid.addWidget(self.tail_lower_card, 0, 0)
        tail_grid.addWidget(self.tail_upper_card, 0, 1)
        tail_grid.addWidget(self.tail_asym_card, 0, 2)
        left_layout.addLayout(tail_grid)

        # === FRACTIONAL INTEGRATION + DYNAMIC HEDGE RATIO ===
        left_layout.addWidget(_ou_divider())
        left_layout.addWidget(_ou_header("FRACTIONAL INTEGRATION / DYNAMIC HEDGE"))

        frac_grid = QGridLayout()
        frac_grid.setSpacing(8)

        self.frac_d_card = CompactMetricCard("FRAC. d", "-",
            tooltip="Fractional integration order (GPH estimator).\nd < 0 = strong MR, 0-0.5 = weak MR,\n0.5-1 = borderline, > 1 = non-stationary.")
        self.frac_class_card = CompactMetricCard("CLASSIFICATION", "-",
            tooltip="Mean-reversion strength classification\nbased on fractional d parameter.")
        self.kalman_beta_card = CompactMetricCard("KALMAN β", "-",
            tooltip="Current Kalman-filtered hedge ratio.\nTime-varying β from state-space model.")
        self.kalman_beta_stab_card = CompactMetricCard("β STABILITY", "-",
            tooltip="Stability of dynamic hedge ratio [0-1].\n1 = perfectly stable β over last 60 days.")

        frac_grid.addWidget(self.frac_d_card, 0, 0)
        frac_grid.addWidget(self.frac_class_card, 0, 1)
        frac_grid.addWidget(self.kalman_beta_card, 0, 2)
        frac_grid.addWidget(self.kalman_beta_stab_card, 0, 3)
        left_layout.addLayout(frac_grid)

        # === REGIME STATE (HMM) ===
        left_layout.addWidget(_ou_divider())
        left_layout.addWidget(_ou_header("REGIME STATE (HMM)"))

        hmm_grid = QGridLayout()
        hmm_grid.setSpacing(8)

        self.ou_hmm_regime_card = CompactMetricCard("CURRENT REGIME", "-",
            tooltip="HMM 3-state regime: MR = Mean-Reverting,\nTR = Trending, CR = Crisis.")
        self.ou_hmm_pmr_card = CompactMetricCard("P(MEAN-REVERT)", "-",
            tooltip="HMM probability of mean-reverting regime.\n≥ 0.7 = strong MR, ≥ 0.4 = moderate, < 0.3 = fail.")
        self.ou_hmm_ptr_card = CompactMetricCard("P(TRENDING)", "-",
            tooltip="HMM probability of trending regime.\n> 0.3 = caution, pair may be directional.")
        self.ou_hmm_pcr_card = CompactMetricCard("P(CRISIS)", "-",
            tooltip="HMM probability of crisis regime.\n> 0.2 = elevated risk of structural break.")
        self.ou_hmm_stab_card = CompactMetricCard("REGIME STABILITY", "-",
            tooltip="Stability of current regime assignment [0-1].\nHigher = more confident regime classification.")
        self.ou_cusum_card = CompactMetricCard("CUSUM SCORE", "-",
            tooltip="CUSUM on OU residuals. Threshold=15.0.\nHigh = structural break, parameters unreliable.")

        hmm_grid.addWidget(self.ou_hmm_regime_card, 0, 0)
        hmm_grid.addWidget(self.ou_hmm_pmr_card, 0, 1)
        hmm_grid.addWidget(self.ou_hmm_ptr_card, 0, 2)
        hmm_grid.addWidget(self.ou_hmm_pcr_card, 0, 3)
        hmm_grid.addWidget(self.ou_hmm_stab_card, 1, 0)
        hmm_grid.addWidget(self.ou_cusum_card, 1, 1)
        left_layout.addLayout(hmm_grid)

        # Push content to top
        left_layout.addStretch()

        # Wrap in scroll area
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
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
            zscore_col.addWidget(SectionHeader("Z-SCORE (Y − β·X − α)"))
            
            self.ou_zscore_date_axis = DateAxisItem(orientation='bottom')
            self.ou_zscore_plot = pg.PlotWidget(axisItems={'bottom': self.ou_zscore_date_axis})
            self.ou_zscore_plot.setLabel('left', 'Z-Score')
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
        
        # Set ratio 2:3 (left:right)
        splitter.setSizes([400, 600])
        
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
        self.signal_count_label = QLabel("⚡ 0 viable pairs with |Z| ≥ Opt.Z*")
        self.signal_count_label.setStyleSheet(f"color: {COLORS['positive']}; font-size: 13px; font-weight: 500;")
        top_layout.addWidget(self.signal_count_label)
        
        top_layout.addStretch()
        
        # Signal selector
        selector_label = QLabel("Select Signal:")
        selector_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; border: none;")
        top_layout.addWidget(selector_label)
        
        self.signal_combo = QComboBox()
        self.signal_combo.setMinimumWidth(280)
        self.signal_combo.setMaximumWidth(400)
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

        self.signal_opt_z_card = self._create_compact_metric_card("OPTIMAL Z*", "-")
        state_grid.addWidget(self.signal_opt_z_card, 1, 2)

        self.signal_cvar_card = self._create_compact_metric_card("CVaR (95%)", "-")
        state_grid.addWidget(self.signal_cvar_card, 0, 3)

        left_layout.addLayout(state_grid)

        # ===== TRADE METRICS section =====
        trade_metrics_divider = QFrame()
        trade_metrics_divider.setFrameShape(QFrame.HLine)
        trade_metrics_divider.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        trade_metrics_divider.setMaximumHeight(1)
        left_layout.addWidget(trade_metrics_divider)

        trade_metrics_header = QLabel("TRADE METRICS")
        trade_metrics_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 700; letter-spacing: 1px; border: none;")
        left_layout.addWidget(trade_metrics_header)

        tm_grid = QGridLayout()
        tm_grid.setSpacing(8)

        self.signal_winprob_card = self._create_compact_metric_card("WIN PROB", "-")
        tm_grid.addWidget(self.signal_winprob_card, 0, 0)

        self.signal_epnl_card = self._create_compact_metric_card("EXPECTED PnL", "-")
        tm_grid.addWidget(self.signal_epnl_card, 0, 1)

        self.signal_kelly_card = self._create_compact_metric_card("KELLY f (¼)", "-")
        tm_grid.addWidget(self.signal_kelly_card, 0, 2)

        self.signal_rr_card = self._create_compact_metric_card("RISK : REWARD", "-")
        tm_grid.addWidget(self.signal_rr_card, 0, 3)

        self.signal_confidence_card = self._create_compact_metric_card("CONFIDENCE", "-")
        tm_grid.addWidget(self.signal_confidence_card, 1, 0)

        self.signal_avghold_card = self._create_compact_metric_card("AVG HOLD", "-")
        tm_grid.addWidget(self.signal_avghold_card, 1, 1)

        self.signal_windays_card = self._create_compact_metric_card("AVG WIN DAYS", "-")
        tm_grid.addWidget(self.signal_windays_card, 1, 2)

        self.signal_lossdays_card = self._create_compact_metric_card("AVG LOSS DAYS", "-")
        tm_grid.addWidget(self.signal_lossdays_card, 1, 3)

        left_layout.addLayout(tm_grid)

        # Spread Option section
        spread_opt_divider = QFrame()
        spread_opt_divider.setFrameShape(QFrame.HLine)
        spread_opt_divider.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        spread_opt_divider.setMaximumHeight(1)
        left_layout.addWidget(spread_opt_divider)

        self.margrabe_header = QPushButton("▶ SPREAD OPTION (MARGRABE)")
        self.margrabe_header.setStyleSheet(f"""
            color: {COLORS['accent']}; font-size: 13px; font-weight: 700;
            letter-spacing: 1px; border: none; text-align: left;
            padding: 2px 0px; background: transparent;
        """)
        self.margrabe_header.setCursor(Qt.PointingHandCursor)
        self.margrabe_header.clicked.connect(self._toggle_margrabe_section)
        left_layout.addWidget(self.margrabe_header)

        self.margrabe_container = QWidget()
        margrabe_grid = QGridLayout(self.margrabe_container)
        margrabe_grid.setSpacing(8)
        margrabe_grid.setContentsMargins(0, 0, 0, 0)

        self.margrabe_fv_card = self._create_compact_metric_card("FAIR VALUE", "-")
        margrabe_grid.addWidget(self.margrabe_fv_card, 0, 0)

        self.margrabe_iv_card = self._create_compact_metric_card("IMPLIED VOL", "-")
        margrabe_grid.addWidget(self.margrabe_iv_card, 0, 1)

        self.margrabe_delta_y_card = self._create_compact_metric_card("Δ Y", "-")
        margrabe_grid.addWidget(self.margrabe_delta_y_card, 0, 2)

        self.margrabe_delta_x_card = self._create_compact_metric_card("Δ X", "-")
        margrabe_grid.addWidget(self.margrabe_delta_x_card, 1, 0)

        self.margrabe_gamma_card = self._create_compact_metric_card("GAMMA", "-")
        margrabe_grid.addWidget(self.margrabe_gamma_card, 1, 1)

        self.margrabe_vega_card = self._create_compact_metric_card("VEGA", "-")
        margrabe_grid.addWidget(self.margrabe_vega_card, 1, 2)

        self.margrabe_theta_card = self._create_compact_metric_card("THETA (/day)", "-")
        margrabe_grid.addWidget(self.margrabe_theta_card, 2, 0)

        self.margrabe_container.setVisible(False)
        left_layout.addWidget(self.margrabe_container)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"background-color: {COLORS['border_subtle']};")
        divider.setMaximumHeight(1)
        left_layout.addWidget(divider)
        
        # Position Sizing header (kollapsbar)
        self.sizing_header = QPushButton("▶ POSITION SIZING")
        self.sizing_header.setStyleSheet(f"""
            color: {COLORS['accent']}; font-size: 13px; font-weight: 700;
            letter-spacing: 1px; border: none; text-align: left;
            padding: 2px 0px; background: transparent;
        """)
        self.sizing_header.setCursor(Qt.PointingHandCursor)
        self.sizing_header.clicked.connect(self._toggle_sizing_section)
        left_layout.addWidget(self.sizing_header)

        self.sizing_container = QWidget()
        sizing_container_layout = QVBoxLayout(self.sizing_container)
        sizing_container_layout.setContentsMargins(0, 0, 0, 0)
        sizing_container_layout.setSpacing(8)

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
        sizing_container_layout.addLayout(sizing_layout)

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

        sizing_container_layout.addLayout(pos_layout)
        self.sizing_container.setVisible(False)
        left_layout.addWidget(self.sizing_container)

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
            zscore_header = QLabel("Z-SCORE (Y − β·X − α)")
            zscore_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
            right_layout.addWidget(zscore_header)
            
            # Create date axis for spread plot
            self.signal_zscore_date_axis = DateAxisItem(orientation='bottom')
            self.signal_zscore_plot = pg.PlotWidget(axisItems={'bottom': self.signal_zscore_date_axis})
            self.signal_zscore_plot.setLabel('left', 'Z-Score')
            self.signal_zscore_plot.showGrid(x=False, y=False, alpha=0.3)
            self.signal_zscore_plot.setMinimumHeight(120)
            self.signal_zscore_plot.setMouseEnabled(x=True, y=True)
            self.signal_zscore_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            right_layout.addWidget(self.signal_zscore_plot, stretch=1)

            # Monte Carlo fan chart
            mc_header = QLabel("MONTE CARLO SIMULATION (5000 paths)")
            mc_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
            right_layout.addWidget(mc_header)

            self.signal_mc_plot = pg.PlotWidget()
            self.signal_mc_plot.setLabel('left', 'Spread')
            self.signal_mc_plot.setLabel('bottom', 'Days')
            self.signal_mc_plot.showGrid(x=False, y=False, alpha=0.3)
            self.signal_mc_plot.setMinimumHeight(120)
            self.signal_mc_plot.setMouseEnabled(x=True, y=True)
            self.signal_mc_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            right_layout.addWidget(self.signal_mc_plot, stretch=1)

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
        
        # Positions table with dynamic columns
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(13)
        self.positions_table.setHorizontalHeaderLabels([
            "POSITION", "TYPE", "SIGNAL", "STATUS",
            "LEG 1 (Y/CALL)", "ENTRY PRICE", "QUANTITY", "P/L",
            "LEG 2 (X/PUT)", "ENTRY PRICE", "QUANTITY", "P/L",
            "CLOSE"
        ])

        # Row height
        self.positions_table.verticalHeader().setDefaultSectionSize(55)
        self.positions_table.verticalHeader().setVisible(False)

        # Dynamic column widths
        header = self.positions_table.horizontalHeader()

        self.positions_table.setColumnWidth(0, 200)   # PAIR
        self.positions_table.setColumnWidth(1, 100)    # DIR
        self.positions_table.setColumnWidth(2, 90)     # Z
        self.positions_table.setColumnWidth(3, 90)     # STATUS
        self.positions_table.setColumnWidth(4, 240)   # MINI Y
        self.positions_table.setColumnWidth(5, 110)   # ENTRY Y
        self.positions_table.setColumnWidth(6, 90)     # QTY Y
        self.positions_table.setColumnWidth(7, 120)   # P/L Y
        self.positions_table.setColumnWidth(8, 240)   # MINI X
        self.positions_table.setColumnWidth(9, 110)   # ENTRY X
        self.positions_table.setColumnWidth(10, 90)    # QTY X
        self.positions_table.setColumnWidth(11, 120)   # P/L X
        self.positions_table.setColumnWidth(12, 70)    # CLOSE

        header.setSectionResizeMode(QHeaderView.Interactive)
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

        # Concentration Risk section
        conc_frame = QFrame()
        conc_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        conc_layout = QHBoxLayout(conc_frame)
        conc_layout.setContentsMargins(10, 6, 10, 6)
        conc_layout.setSpacing(20)

        conc_title = QLabel("CONCENTRATION RISK")
        conc_title.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
        conc_layout.addWidget(conc_title)

        self.eff_bets_label = QLabel("Effective Bets: <span style='color:#e8e8e8; font-weight:600;'>-</span>")
        self.eff_bets_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        conc_layout.addWidget(self.eff_bets_label)

        self.conc_score_label = QLabel("Concentration: <span style='color:#888;'>-</span>")
        self.conc_score_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        conc_layout.addWidget(self.conc_score_label)

        self.max_corr_label = QLabel("Max Corr Pair: <span style='color:#888;'>-</span>")
        self.max_corr_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        conc_layout.addWidget(self.max_corr_label)

        conc_layout.addStretch()
        current_layout.addWidget(conc_frame)

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
        
        # Auto-load trade history when switching to that tab
        self.portfolio_subtabs.currentChanged.connect(self._on_portfolio_subtab_changed)

        layout.addWidget(self.portfolio_subtabs)

        return tab

    def _on_portfolio_subtab_changed(self, index: int):
        """Handle portfolio sub-tab change."""
        if index == 1:
            self._update_trade_history_display()
    
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
        n_states = N_MARKOV_STATES if MARKOV_AVAILABLE else 2
        state_shorts = [MARKOV_STATES[i]['short'] for i in range(n_states)] if MARKOV_AVAILABLE else ['NEG', 'POS']
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
            for j in range(n_states):
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
            for s_id, name in enumerate(['NEG', 'POS']):
                card = CompactMetricCard(f"P({name})", "-")
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
        n_states = N_MARKOV_STATES if MARKOV_AVAILABLE else 2
        state_shorts_full = [MARKOV_STATES[i]['short'] for i in range(n_states)] if MARKOV_AVAILABLE else ['NEG', 'POS']
        for row in range(n_states):
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
        n_st = T.shape[0]
        for i in range(n_st):
            for j in range(n_st):
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
        for s_id in range(r.n_states):
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
        for s_id in range(r.n_states):
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
            parts = [f"{MARKOV_STATES[i]['short']}:{r.stationary_dist[i]:.0%}" for i in range(r.n_states)]
        else:
            parts = [f"S{i}:{r.stationary_dist[i]:.0%}" for i in range(r.n_states)]
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

        for s_id in range(r.n_states):
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
        n_st = T.shape[0]
        for i in range(n_st):
            for j in range(n_st):
                txt = pg.TextItem(f"{T[i, j]:.0%}", color='#ffffff', anchor=(0.5, 0.5))
                txt.setPos(j + 0.5, i + 0.5)
                txt.setFont(QFont('JetBrains Mono', 9))
                plot.addItem(txt)
                self.markov_heatmap_texts.append(txt)

        # Axis labels
        state_shorts = [MARKOV_STATES[i]['short'] for i in range(n_st)] if MARKOV_AVAILABLE else [f'S{i}' for i in range(n_st)]
        x_axis = plot.getAxis('bottom')
        x_axis.setTicks([[(i + 0.5, s) for i, s in enumerate(state_shorts)]])
        y_axis = plot.getAxis('left')
        y_axis.setTicks([[(i + 0.5, s) for i, s in enumerate(state_shorts)]])
        plot.setXRange(0, n_st)
        plot.setYRange(0, n_st)

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

        n_st = r.n_states
        for i in range(n_st):
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

        # Baseline at uniform probability
        plot.addLine(y=100.0 / n_st, pen=pg.mkPen('#666666', width=1, style=Qt.DashLine))

        state_shorts = [MARKOV_STATES[i]['short'] for i in range(n_st)] if MARKOV_AVAILABLE else [f'S{i}' for i in range(n_st)]
        x_axis = plot.getAxis('bottom')
        x_axis.setTicks([[(i, s) for i, s in enumerate(state_shorts)]])
        plot.setXRange(-0.5, n_st - 0.5)
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

        n_st = r.n_states
        for h_idx, (label, probs) in enumerate(horizons):
            x = np.arange(n_st) + offsets[h_idx]
            bar = pg.BarGraphItem(
                x=x, height=probs * 100, width=bar_width,
                brush=pg.mkBrush(horizon_colors[h_idx]),
                pen=pg.mkPen(None),
                name=label
            )
            plot.addItem(bar)

        state_shorts = [MARKOV_STATES[i]['short'] for i in range(n_st)] if MARKOV_AVAILABLE else [f'S{i}' for i in range(n_st)]
        x_axis = plot.getAxis('bottom')
        x_axis.setTicks([[(i, s) for i, s in enumerate(state_shorts)]])
        plot.setXRange(-0.5, n_st - 0.5)
        max_val = max(r.forecast_probs.max(), r.forecast_2w.max(), r.forecast_4w.max()) * 100
        plot.setYRange(0, max_val * 1.2 + 5)
        plot.addLegend(offset=(10, 10))

    # ========================================================================
    # TAB 11: EPS MEAN REVERSION
    # ========================================================================

    def create_eps_reversion_tab(self) -> QWidget:
        """Create EPS Mean Reversion tab with ticker input, screening table, and P/E charts."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 10, 15, 10)

        # ── Top bar ──────────────────────────────────────────────────────
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

        lbl = QLabel("TICKERS")
        lbl.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;text-transform:uppercase;letter-spacing:1px;background:transparent;border:none;")
        top_layout.addWidget(lbl)

        self.eps_ticker_input = QLineEdit()
        self.eps_ticker_input.setPlaceholderText("AAPL, MSFT, GOOG, JNJ, JPM, PG, XOM, V")
        self.eps_ticker_input.setMinimumWidth(400)
        self.eps_ticker_input.setStyleSheet(f"""
            QLineEdit {{
                background: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 4px;
                padding: 5px 10px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 12px;
            }}
        """)
        top_layout.addWidget(self.eps_ticker_input)

        self.eps_run_btn = QPushButton("ANALYZE EPS")
        self.eps_run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent']};
                color: {COLORS['bg_darkest']};
                border: none; border-radius: 4px;
                padding: 6px 18px; font-weight: 600; font-size: 12px; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {COLORS['accent_bright']}; }}
            QPushButton:disabled {{ background: {COLORS['text_disabled']}; color: {COLORS['text_muted']}; }}
        """)
        self.eps_run_btn.clicked.connect(self.run_eps_mr_analysis)
        top_layout.addWidget(self.eps_run_btn)

        self.eps_status_label = QLabel("")
        self.eps_status_label.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;background:transparent;border:none;")
        top_layout.addWidget(self.eps_status_label)
        top_layout.addStretch()

        self.eps_card_tickers = CompactMetricCard("TICKERS", "-")
        self.eps_card_signals = CompactMetricCard("SIGNALS", "-")
        self.eps_card_underval = CompactMetricCard("UNDERVALUED", "-")
        self.eps_card_overval = CompactMetricCard("OVERVALUED", "-")
        for card in [self.eps_card_tickers, self.eps_card_signals,
                     self.eps_card_underval, self.eps_card_overval]:
            card.setFixedWidth(140)
            top_layout.addWidget(card)

        main_layout.addWidget(top_bar)

        # ── Splitter ─────────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['border_subtle']}; }}")

        # ── LEFT: Screening table ────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 10, 5)
        left_layout.setSpacing(8)

        left_layout.addWidget(SectionHeader("EPS SCREENING"))
        self.eps_screening_table = QTableWidget()
        self.eps_screening_table.setColumnCount(9)
        self.eps_screening_table.setHorizontalHeaderLabels(
            ["Ticker", "Spr Z", "ADF p", "HL(d)", "R²", "Hurst", "β", "Pass", "Signal"])
        self.eps_screening_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.eps_screening_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.eps_screening_table.setAlternatingRowColors(True)
        self.eps_screening_table.setStyleSheet(f"""
            QTableWidget {{
                background: {COLORS['bg_card']};
                alternate-background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                gridline-color: {COLORS['border_subtle']};
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
            }}
            QTableWidget::item {{ padding: 3px 6px; }}
            QHeaderView::section {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['accent']};
                font-weight: 600; font-size: 10px;
                border: none; border-bottom: 1px solid {COLORS['border_subtle']};
                padding: 4px;
            }}
        """)
        self.eps_screening_table.currentCellChanged.connect(self._on_eps_ticker_selected)
        left_layout.addWidget(self.eps_screening_table)
        splitter.addWidget(left_widget)

        # ── RIGHT: Charts ────────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(8)

        pg = get_pyqtgraph()
        if pg is not None:
            DateAxisItem = get_date_axis_item_class()

            right_layout.addWidget(SectionHeader("PRICE & TTM EPS"))
            self.eps_pe_date_axis = DateAxisItem(orientation='bottom')
            self.eps_pe_plot = pg.PlotWidget(
                axisItems={'bottom': self.eps_pe_date_axis})
            self.eps_pe_plot.setMinimumHeight(200)
            self.eps_pe_plot.showGrid(x=True, y=True, alpha=0.15)
            # Sekundär y-axel för EPS
            self.eps_pe_plot.setLabel('left', 'Price', color=COLORS['accent'])
            self._eps_vb2 = pg.ViewBox()
            self.eps_pe_plot.scene().addItem(self._eps_vb2)
            self.eps_pe_plot.getAxis('right').linkToView(self._eps_vb2)
            self._eps_vb2.setXLink(self.eps_pe_plot)
            self.eps_pe_plot.getAxis('right').setLabel('TTM EPS', color='#22c55e')
            self.eps_pe_plot.showAxis('right')
            # Synka viewboxar vid resize
            self.eps_pe_plot.getViewBox().sigResized.connect(
                lambda: self._eps_vb2.setGeometry(self.eps_pe_plot.getViewBox().sceneBoundingRect()))
            right_layout.addWidget(self.eps_pe_plot)

            right_layout.addWidget(SectionHeader("PRICE vs EPS SPREAD (Z-SCORE)"))
            self.eps_spread_date_axis = DateAxisItem(orientation='bottom')
            self.eps_spread_plot = pg.PlotWidget(
                axisItems={'bottom': self.eps_spread_date_axis})
            self.eps_spread_plot.setMinimumHeight(200)
            self.eps_spread_plot.showGrid(x=True, y=True, alpha=0.15)
            right_layout.addWidget(self.eps_spread_plot)
        else:
            right_layout.addWidget(QLabel("Install pyqtgraph for charts"))

        splitter.addWidget(right_widget)
        splitter.setSizes([450, 550])
        main_layout.addWidget(splitter)

        return tab

    # ── EPS worker methods ───────────────────────────────────────────────

    def run_eps_mr_analysis(self):
        if not EPS_MR_AVAILABLE:
            self.eps_status_label.setText("eps_mean_reversion.py not found")
            return
        if self._eps_mr_running:
            self.eps_status_label.setText("Already running...")
            return

        # Get tickers from input or use defaults
        raw = self.eps_ticker_input.text().strip()
        if raw:
            tickers = [t.strip().upper() for t in raw.replace(';', ',').split(',') if t.strip()]
        else:
            tickers = ["AAPL", "MSFT", "GOOG", "JNJ", "JPM", "PG", "XOM", "V"]
            self.eps_ticker_input.setText(", ".join(tickers))

        if not tickers:
            self.eps_status_label.setText("Enter at least one ticker")
            return

        self._eps_mr_running = True
        self.eps_run_btn.setEnabled(False)
        self.eps_status_label.setText(f"Analyzing {len(tickers)} tickers...")

        self._eps_mr_thread = QThread()
        self._eps_mr_worker = EPSMeanReversionWorker(tickers)
        self._eps_mr_worker.moveToThread(self._eps_mr_thread)

        self._eps_mr_thread.started.connect(self._eps_mr_worker.run)
        self._eps_mr_worker.progress.connect(self._on_eps_mr_progress)
        self._eps_mr_worker.result.connect(self._on_eps_mr_result)
        self._eps_mr_worker.error.connect(self._on_eps_mr_error)
        self._eps_mr_worker.finished.connect(self._on_eps_mr_finished)
        self._eps_mr_worker.finished.connect(self._eps_mr_thread.quit)

        self._eps_mr_thread.start()

    def _on_eps_mr_progress(self, pct, msg):
        if hasattr(self, 'eps_status_label'):
            self.eps_status_label.setText(f"[{pct}%] {msg}")

    def _on_eps_mr_error(self, msg):
        print(f"[SCANNER] EPS MR error: {msg}")
        self._eps_mr_running = False
        if hasattr(self, 'eps_status_label'):
            self.eps_status_label.setText(f"Error: {msg}")
            self.eps_status_label.setStyleSheet(f"color:{COLORS['negative']};font-size:11px;background:transparent;border:none;")

    def _on_eps_mr_finished(self):
        self._eps_mr_running = False
        if hasattr(self, 'eps_run_btn'):
            self.eps_run_btn.setEnabled(True)

    def _on_eps_mr_result(self, result):
        self._eps_mr_result = result

        if not hasattr(self, 'eps_card_tickers'):
            return

        data, df, pe_analyses = result
        self._eps_mr_data = data
        self._eps_mr_pe_analyses = pe_analyses

        n_tickers = len(data)
        n_signals = len(df[df['Signal'] != 'Neutral']) if len(df) > 0 else 0
        n_under = len(df[df['Signal'].str.contains('Undervärderad', na=False)]) if len(df) > 0 else 0
        n_over = len(df[df['Signal'].str.contains('Övervärderad', na=False)]) if len(df) > 0 else 0

        self.eps_status_label.setText(f"Complete — {n_tickers} tickers analyzed")
        self.eps_status_label.setStyleSheet(f"color:{COLORS['positive']};font-size:11px;background:transparent;border:none;")
        self.eps_card_tickers.set_value(str(n_tickers))
        self.eps_card_signals.set_value(str(n_signals), COLORS['accent'] if n_signals > 0 else COLORS['text_muted'])
        self.eps_card_underval.set_value(str(n_under), COLORS['positive'] if n_under > 0 else COLORS['text_muted'])
        self.eps_card_overval.set_value(str(n_over), COLORS['negative'] if n_over > 0 else COLORS['text_muted'])

        # Populate screening table
        self._update_eps_screening_table(df)

    def _update_eps_screening_table(self, df):
        """Populate EPS screening table."""
        self.eps_screening_table.setRowCount(len(df))
        for row in range(len(df)):
            r = df.iloc[row]
            for col, key in enumerate(["Ticker", "Spr Z", "ADF p", "HL(d)", "R²", "Hurst", "β", "Pass", "Signal"]):
                val = r.get(key, "")
                item = QTableWidgetItem(str(val) if val is not None else "-")

                # Color coding
                if key == "Signal":
                    if "Undervärderad" in str(val):
                        item.setForeground(QColor(COLORS['positive']))
                    elif "Övervärderad" in str(val):
                        item.setForeground(QColor(COLORS['negative']))
                elif key == "Spr Z":
                    try:
                        z = float(val)
                        item.setForeground(QColor(COLORS['positive'] if z < -1 else COLORS['negative'] if z > 1 else COLORS['text_primary']))
                    except (ValueError, TypeError):
                        pass
                elif key == "ADF p":
                    try:
                        p = float(val)
                        item.setForeground(QColor(COLORS['positive'] if p <= 0.10 else COLORS['text_muted']))
                    except (ValueError, TypeError):
                        pass
                elif key == "Pass":
                    if "4/4" in str(val):
                        item.setForeground(QColor(COLORS['positive']))
                    elif "3/4" in str(val):
                        item.setForeground(QColor(COLORS['accent']))

                self.eps_screening_table.setItem(row, col, item)

    def _on_eps_ticker_selected(self, row, col, prev_row, prev_col):
        """Handle ticker selection in EPS table — update P/E and spread charts."""
        if self._eps_mr_result is None or row < 0:
            return

        data, df, pe_analyses = self._eps_mr_result
        if row >= len(df):
            return

        ticker = df.iloc[row]['Ticker']
        if ticker not in data:
            return

        self._render_eps_charts(ticker, data[ticker], pe_analyses.get(ticker))

    def _render_eps_charts(self, ticker, ticker_data, pe_analysis):
        """Render Price+EPS (dual axis) and spread Z-score charts for selected ticker."""
        pg = get_pyqtgraph()
        if pg is None:
            return

        # Chart 1: Price (vänster y-axel) + TTM EPS (höger y-axel)
        if hasattr(self, 'eps_pe_plot'):
            plot = self.eps_pe_plot
            plot.clear()
            self._eps_vb2.clear()

            price = ticker_data.get('price')
            ttm_eps = ticker_data.get('ttm_eps')

            if price is not None and len(price) > 0:
                # Gemensamt index (price har daglig data, ttm_eps interpolerad)
                if ttm_eps is not None and len(ttm_eps) > 0:
                    common = price.index.intersection(ttm_eps.index)
                    price_plot = price.loc[common]
                    eps_plot = ttm_eps.loc[common]
                else:
                    price_plot = price
                    eps_plot = None

                x = np.arange(len(price_plot))
                self.eps_pe_date_axis.set_dates(price_plot.index)

                # Price på primär y-axel
                price_y = price_plot.values.flatten().astype(float)
                plot.plot(x, price_y, pen=pg.mkPen(COLORS['accent'], width=2), name=f'{ticker} Price')

                # TTM EPS på sekundär y-axel
                if eps_plot is not None and len(eps_plot) > 0:
                    eps_y = eps_plot.values.flatten().astype(float)
                    eps_curve = pg.PlotCurveItem(x, eps_y,
                                                  pen=pg.mkPen('#22c55e', width=2), name='TTM EPS')
                    self._eps_vb2.addItem(eps_curve)
                    # Synka geometry
                    self._eps_vb2.setGeometry(plot.getViewBox().sceneBoundingRect())
                    # Autoscale EPS-axeln
                    self._eps_vb2.enableAutoRange(axis=pg.ViewBox.YAxis)

                # Aktuella värden
                curr_price = float(price_plot.iloc[-1])
                label_parts = [f"Price: {curr_price:,.1f}"]
                if eps_plot is not None and len(eps_plot) > 0:
                    curr_eps = float(eps_plot.iloc[-1])
                    label_parts.append(f"EPS: {curr_eps:.2f}")
                    if curr_eps > 0:
                        label_parts.append(f"P/E: {curr_price/curr_eps:.1f}")
                txt = pg.TextItem('\n'.join(label_parts), color='#e8e8e8', anchor=(1, 0))
                txt.setPos(len(price_plot) - 1, price_plot.iloc[-1])
                txt.setFont(QFont('JetBrains Mono', 9))
                plot.addItem(txt)

        # Chart 2: Price-EPS spread Z-score
        if hasattr(self, 'eps_spread_plot'):
            plot = self.eps_spread_plot
            plot.clear()

            spread_result = analyze_spread_mean_reversion(ticker_data['price'], ticker_data['ttm_eps'])
            if spread_result and 'spread' in spread_result:
                spread = spread_result['spread']
                s_mean = spread_result['spread_mean']
                s_std = spread_result['spread_std']

                # Normalize to z-score
                z_series = (spread - s_mean) / s_std if s_std > 0 else spread * 0
                x = np.arange(len(z_series))
                self.eps_spread_date_axis.set_dates(z_series.index)
                plot.plot(x, z_series.values, pen=pg.mkPen(COLORS['accent'], width=2), name='Spread Z')

                # Horisontella linjer vid z=0, ±1, ±2
                plot.addLine(y=0, pen=pg.mkPen('#888888', width=1))
                plot.addLine(y=1, pen=pg.mkPen(COLORS['negative'], width=1, style=Qt.DashLine))
                plot.addLine(y=-1, pen=pg.mkPen(COLORS['positive'], width=1, style=Qt.DashLine))
                plot.addLine(y=2, pen=pg.mkPen(COLORS['negative'], width=1, style=Qt.DotLine))
                plot.addLine(y=-2, pen=pg.mkPen(COLORS['positive'], width=1, style=Qt.DotLine))

                # Aktuellt värde
                txt = pg.TextItem(f"Z: {spread_result['z_score']:.2f}\nHL: {spread_result['half_life']:.0f}d",
                                  color='#e8e8e8', anchor=(1, 0))
                txt.setPos(len(z_series) - 1, z_series.iloc[-1])
                txt.setFont(QFont('JetBrains Mono', 9))
                plot.addItem(txt)

    # ========================================================================
    # TTM SQUEEZE
    # ========================================================================

    def create_squeeze_tab(self) -> QWidget:
        """Create TTM Squeeze tab with scanner table and charts."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(10, 6, 10, 6)

        # ── Top bar (compact) ─────────────────────────────────────────
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)

        self.squeeze_run_btn = QPushButton("ANALYZE SQUEEZE")
        self.squeeze_run_btn.setFixedHeight(28)
        self.squeeze_run_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['accent']};
                color: {COLORS['bg_darkest']};
                border: none; border-radius: 4px;
                padding: 4px 16px; font-weight: 600; font-size: 11px; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {COLORS['accent_bright']}; }}
            QPushButton:disabled {{ background: {COLORS['text_disabled']}; color: {COLORS['text_muted']}; }}
        """)
        self.squeeze_run_btn.clicked.connect(self.run_squeeze_analysis)
        top_row.addWidget(self.squeeze_run_btn)

        self.squeeze_open_pos_btn = QPushButton("OPEN STRADDLE")
        self.squeeze_open_pos_btn.setFixedHeight(28)
        self.squeeze_open_pos_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['positive']};
                color: {COLORS['bg_darkest']};
                border: none; border-radius: 4px;
                padding: 4px 16px; font-weight: 600; font-size: 11px; letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: #4ade80; }}
            QPushButton:disabled {{ background: {COLORS['text_disabled']}; color: {COLORS['text_muted']}; }}
        """)
        self.squeeze_open_pos_btn.clicked.connect(self._open_straddle_dialog)
        self.squeeze_open_pos_btn.setEnabled(False)
        top_row.addWidget(self.squeeze_open_pos_btn)

        self.squeeze_status_label = QLabel("")
        self.squeeze_status_label.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;")
        top_row.addWidget(self.squeeze_status_label)
        top_row.addStretch()

        self.squeeze_card_total = CompactMetricCard("TICKERS", "-")
        self.squeeze_card_squeezing = CompactMetricCard("SQUEEZING", "-")
        for card in [self.squeeze_card_total, self.squeeze_card_squeezing]:
            card.setFixedWidth(100)
            card.setFixedHeight(36)
            top_row.addWidget(card)

        main_layout.addLayout(top_row)

        # ── Shared scroll CSS ─────────────────────────────────────────────
        scroll_css = f"""
            QScrollArea {{
                background: {COLORS['bg_card']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                background: {COLORS['bg_dark']}; width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['text_disabled']}; border-radius: 4px;
            }}
            QScrollBar:horizontal {{
                background: {COLORS['bg_dark']}; height: 8px;
            }}
            QScrollBar::handle:horizontal {{
                background: {COLORS['text_disabled']}; border-radius: 4px;
            }}
        """
        table_css = f"""
            QTableWidget {{
                background: {COLORS['bg_card']};
                alternate-background-color: {COLORS['bg_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                gridline-color: {COLORS['border_subtle']};
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
            }}
            QTableWidget::item {{ padding: 3px 6px; }}
            QHeaderView::section {{
                background: {COLORS['bg_elevated']};
                color: {COLORS['accent']};
                font-weight: 600; font-size: 10px;
                border: none; border-bottom: 1px solid {COLORS['border_subtle']};
                padding: 4px;
            }}
        """

        # ── MAIN VERTICAL SPLITTER (top row / bottom row) ───────────────
        v_splitter = QSplitter(Qt.Vertical)
        v_splitter.setHandleWidth(3)
        v_splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['border_subtle']}; }}")

        # ════════════════════════════════════════════════════════════════
        # TOP ROW: Scanner (left) | Price Chart (right)
        # ════════════════════════════════════════════════════════════════
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setHandleWidth(3)
        top_splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['border_subtle']}; }}")

        # ── TOP-LEFT: Squeeze Scanner table ──
        tl_widget = QWidget()
        tl_layout = QVBoxLayout(tl_widget)
        tl_layout.setContentsMargins(4, 4, 4, 4)
        tl_layout.setSpacing(4)
        tl_layout.addWidget(SectionHeader("SQUEEZE SCANNER"))

        self.squeeze_table = QTableWidget()
        self.squeeze_table.setColumnCount(7)
        self.squeeze_table.setHorizontalHeaderLabels(
            ["Ticker", "Signal", "Sqz Days", "Strike", "Straddle", "Cost%", "IV%"])
        self.squeeze_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.squeeze_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.squeeze_table.setAlternatingRowColors(True)
        self.squeeze_table.setStyleSheet(table_css)
        self.squeeze_table.currentCellChanged.connect(self._on_squeeze_ticker_selected)
        tl_layout.addWidget(self.squeeze_table, 1)
        top_splitter.addWidget(tl_widget)

        # ── TOP-RIGHT: Tabs (Price Chart | Vol Surface) ──
        tr_widget = QWidget()
        tr_layout = QVBoxLayout(tr_widget)
        tr_layout.setContentsMargins(4, 4, 4, 4)
        tr_layout.setSpacing(0)

        self.squeeze_chart_tabs = QTabWidget()
        self.squeeze_chart_tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{
                background: {COLORS['bg_elevated']}; color: {COLORS['text_muted']};
                padding: 6px 16px; border: none; font-size: 11px; font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['bg_card']}; color: {COLORS['accent']};
                border-bottom: 2px solid {COLORS['accent']};
            }}
        """)

        # Tab 1: Price + BB/KC chart
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(0, 4, 0, 0)
        chart_layout.setSpacing(0)

        pg = get_pyqtgraph()
        if pg is not None:
            DateAxisItem = get_date_axis_item_class()
            self.squeeze_price_date_axis = DateAxisItem(orientation='bottom')
            self.squeeze_price_plot = pg.PlotWidget(
                axisItems={'bottom': self.squeeze_price_date_axis})
            self.squeeze_price_plot.showGrid(x=True, y=True, alpha=0.15)
            self.squeeze_price_plot.addLegend(offset=(10, 10))
            chart_layout.addWidget(self.squeeze_price_plot, 1)
        else:
            chart_layout.addWidget(QLabel("Install pyqtgraph for charts"))
        self.squeeze_chart_tabs.addTab(chart_tab, "PRICE + BB/KC")

        # Tab 2: Volatility Surface
        surface_tab = QWidget()
        surface_layout = QVBoxLayout(surface_tab)
        surface_layout.setContentsMargins(0, 4, 0, 0)
        surface_layout.setSpacing(0)

        if WEBENGINE_AVAILABLE:
            self.vol_surface_view = QWebEngineView()
            self.vol_surface_view.setStyleSheet(f"background: {COLORS['bg_card']};")
            self.vol_surface_view.setHtml(
                f"<body style='background:{COLORS['bg_card']};color:{COLORS['text_muted']};"
                f"font-family:monospace;display:flex;align-items:center;justify-content:center;"
                f"height:100vh;margin:0'><p>Select a ticker to load vol surface</p></body>")
            surface_layout.addWidget(self.vol_surface_view, 1)
        else:
            surface_layout.addWidget(QLabel("Install QtWebEngine for vol surface"))
        self.squeeze_chart_tabs.addTab(surface_tab, "VOL SURFACE")

        tr_layout.addWidget(self.squeeze_chart_tabs, 1)
        top_splitter.addWidget(tr_widget)

        top_splitter.setSizes([280, 720])
        v_splitter.addWidget(top_splitter)

        # ════════════════════════════════════════════════════════════════
        # BOTTOM ROW: Straddle Pricing (left) | Key Insights (right)
        # ════════════════════════════════════════════════════════════════
        bot_splitter = QSplitter(Qt.Horizontal)
        bot_splitter.setHandleWidth(3)
        bot_splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['border_subtle']}; }}")

        # ── BOTTOM-LEFT: Straddle Pricing ──
        bl_widget = QWidget()
        bl_layout = QVBoxLayout(bl_widget)
        bl_layout.setContentsMargins(2, 2, 2, 2)
        bl_layout.setSpacing(2)
        bl_layout.addWidget(SectionHeader("STRADDLE PRICING"))

        self.squeeze_options_label = QLabel("Run squeeze scan to load options data")
        self.squeeze_options_label.setTextFormat(Qt.RichText)
        self.squeeze_options_label.setOpenExternalLinks(True)
        self.squeeze_options_label.setWordWrap(False)
        self.squeeze_options_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.squeeze_options_label.setStyleSheet(f"""
            color: {COLORS['text_muted']}; font-size: 13px;
            font-family: 'JetBrains Mono', monospace;
            background: {COLORS['bg_card']};
            padding: 2px 4px;
        """)
        options_scroll = QScrollArea()
        options_scroll.setWidget(self.squeeze_options_label)
        options_scroll.setWidgetResizable(True)
        options_scroll.setStyleSheet(scroll_css)
        bl_layout.addWidget(options_scroll, 1)
        bot_splitter.addWidget(bl_widget)

        # ── BOTTOM-RIGHT: Key Insights ──
        br_widget = QWidget()
        br_layout = QVBoxLayout(br_widget)
        br_layout.setContentsMargins(4, 4, 4, 4)
        br_layout.setSpacing(4)
        br_layout.addWidget(SectionHeader("KEY INSIGHTS"))

        self.squeeze_insights_label = QLabel("Select a ticker to view insights")
        self.squeeze_insights_label.setTextFormat(Qt.RichText)
        self.squeeze_insights_label.setWordWrap(True)
        self.squeeze_insights_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.squeeze_insights_label.setStyleSheet(f"""
            color: {COLORS['text_primary']}; font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            background: {COLORS['bg_card']};
            padding: 10px;
        """)
        insights_scroll = QScrollArea()
        insights_scroll.setWidget(self.squeeze_insights_label)
        insights_scroll.setWidgetResizable(True)
        insights_scroll.setStyleSheet(scroll_css)
        br_layout.addWidget(insights_scroll, 1)
        bot_splitter.addWidget(br_widget)

        bot_splitter.setSizes([680, 320])
        v_splitter.addWidget(bot_splitter)

        v_splitter.setSizes([450, 450])
        main_layout.addWidget(v_splitter)

        return tab

    def run_squeeze_analysis(self):
        """Launch TTM Squeeze analysis on OMXS Large Cap tickers."""
        if not SQUEEZE_AVAILABLE:
            if hasattr(self, 'squeeze_status_label'):
                self.squeeze_status_label.setText("squeeze.py not found")
            return
        if self._squeeze_running:
            return

        self._squeeze_running = True
        if hasattr(self, 'squeeze_run_btn'):
            self.squeeze_run_btn.setEnabled(False)
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText("Downloading OMXS Large Cap data...")

        from app_config import OMXS_LARGE_CAP
        self._squeeze_thread = QThread()
        self._squeeze_worker = SqueezeWorker(OMXS_LARGE_CAP)
        self._squeeze_worker.moveToThread(self._squeeze_thread)

        self._squeeze_thread.started.connect(self._squeeze_worker.run)
        self._squeeze_worker.progress.connect(self._on_squeeze_progress)
        self._squeeze_worker.result.connect(self._on_squeeze_result)
        self._squeeze_worker.error.connect(self._on_squeeze_error)
        self._squeeze_worker.finished.connect(self._on_squeeze_finished)
        self._squeeze_worker.finished.connect(self._squeeze_thread.quit)

        self._squeeze_thread.start()

    def _on_squeeze_progress(self, pct, msg):
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText(f"[{pct}%] {msg}")

    def _on_squeeze_error(self, msg):
        print(f"[SCANNER] TTM Squeeze error: {msg}")
        self._squeeze_running = False
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText(f"Error: {msg}")
            self.squeeze_status_label.setStyleSheet(f"color:{COLORS['negative']};font-size:11px;background:transparent;border:none;")

    def _on_squeeze_finished(self):
        self._squeeze_running = False
        if hasattr(self, 'squeeze_run_btn'):
            self.squeeze_run_btn.setEnabled(True)

    def _on_squeeze_result(self, result):
        self._squeeze_result = result
        n_squeeze = result.n_squeeze_on + result.n_squeeze_firing
        print(f"[SQUEEZE] Complete: {result.n_total} tickers, {n_squeeze} in squeeze")

        if not hasattr(self, 'squeeze_card_total'):
            return

        self.squeeze_status_label.setText(f"Complete — {n_squeeze} in squeeze")
        self.squeeze_status_label.setStyleSheet(
            f"color:{COLORS['positive']};font-size:11px;")

        self.squeeze_card_total.set_value(str(result.n_total))
        sqz_color = COLORS['warning'] if n_squeeze > 0 else COLORS['text_muted']
        self.squeeze_card_squeezing.set_value(str(n_squeeze), sqz_color)

        # Populate table
        self._populate_squeeze_table(result)

        # Auto-fetch options data
        if OPTIONS_AVAILABLE:
            self._run_options_fetch(result)

    def _populate_squeeze_table(self, result):
        """Fill squeeze scanner table — only tickers in squeeze, filtered to .ST."""
        df = result.summary
        if df is None or df.empty:
            self.squeeze_table.setRowCount(0)
            return

        # Bara tickers i aktiv squeeze (SQUEEZE_ON eller SQUEEZE_FIRE_*), filtrera bort NO_SQUEEZE + icke-.ST
        squeeze_df = df[
            (df['Signal'] != 'NO_SQUEEZE') &
            (df['Ticker'].str.endswith('.ST', na=False))
        ].copy()

        self.squeeze_table.setSortingEnabled(False)
        self.squeeze_table.setRowCount(len(squeeze_df))

        for row_idx, (_, row) in enumerate(squeeze_df.iterrows()):
            # Ticker
            ticker_item = QTableWidgetItem(str(row.get('Ticker', '')))
            ticker_item.setForeground(QColor(COLORS['text_primary']))
            self.squeeze_table.setItem(row_idx, 0, ticker_item)

            # Signal
            signal_item = QTableWidgetItem('SQUEEZE')
            signal_item.setForeground(QColor(COLORS['warning']))
            self.squeeze_table.setItem(row_idx, 1, signal_item)

            # Sqz Days
            sqz_days = int(row.get('Sqz Days', 0) or 0)
            days_item = QTableWidgetItem(str(sqz_days) if sqz_days > 0 else '-')
            days_item.setData(Qt.UserRole, sqz_days)
            days_item.setForeground(QColor(COLORS['text_primary']))
            self.squeeze_table.setItem(row_idx, 2, days_item)

            # Options-kolumner fylls i senare av _update_squeeze_table_options
            for col in (3, 4, 5, 6):
                item = QTableWidgetItem('-')
                item.setForeground(QColor(COLORS['text_disabled']))
                self.squeeze_table.setItem(row_idx, col, item)

        self.squeeze_table.setSortingEnabled(True)

    def _on_squeeze_ticker_selected(self, row, col, prev_row, prev_col):
        """Render charts and options panel for selected ticker."""
        if row < 0 or self._squeeze_result is None:
            return
        ticker_item = self.squeeze_table.item(row, 0)
        if ticker_item is None:
            return
        ticker = ticker_item.text()
        if ticker in self._squeeze_result.ticker_results:
            self._render_squeeze_charts(self._squeeze_result.ticker_results[ticker])
        # Update options panel + insights
        if self._options_data:
            self._update_options_panel(ticker)
            self._load_vol_surface(ticker)
        self._update_squeeze_insights(ticker)
        # Enable open straddle button
        if hasattr(self, 'squeeze_open_pos_btn'):
            has_opts = bool(self._options_data.get(ticker, {}).get('atm_straddles') is not None)
            self.squeeze_open_pos_btn.setEnabled(has_opts)

    def _render_squeeze_charts(self, tr):
        """Render price+bands and momentum charts for a ticker."""
        pg = get_pyqtgraph()
        if pg is None:
            return

        # ── Chart 1: Price + BB + KC ────────────────────────────────────
        if hasattr(self, 'squeeze_price_plot') and len(tr.price) > 0:
            plot = self.squeeze_price_plot
            plot.clear()
            # Visa senaste 252 dagar
            n_show = min(252, len(tr.price))
            price = tr.price.iloc[-n_show:]
            bb_u = tr.bb_upper.iloc[-n_show:]
            bb_l = tr.bb_lower.iloc[-n_show:]
            bb_m = tr.bb_mid.iloc[-n_show:]
            kc_u = tr.kc_upper.iloc[-n_show:]
            kc_l = tr.kc_lower.iloc[-n_show:]
            sqz = tr.squeeze_hist.iloc[-n_show:]

            x = np.arange(len(price))
            self.squeeze_price_date_axis.set_dates(price.index)

            # Price
            plot.plot(x, price.values.flatten().astype(float),
                      pen=pg.mkPen(COLORS['accent'], width=2), name=f'{tr.ticker}')
            # BB
            plot.plot(x, bb_u.values.flatten().astype(float),
                      pen=pg.mkPen('#e74c3c', width=1, style=Qt.DashLine), name='BB Upper')
            plot.plot(x, bb_l.values.flatten().astype(float),
                      pen=pg.mkPen('#e74c3c', width=1, style=Qt.DashLine), name='BB Lower')
            # KC
            plot.plot(x, kc_u.values.flatten().astype(float),
                      pen=pg.mkPen('#3b82f6', width=1, style=Qt.DotLine), name='KC Upper')
            plot.plot(x, kc_l.values.flatten().astype(float),
                      pen=pg.mkPen('#3b82f6', width=1, style=Qt.DotLine), name='KC Lower')
            # SMA midline
            plot.plot(x, bb_m.values.flatten().astype(float),
                      pen=pg.mkPen('#888888', width=1))

            # Squeeze-bakgrund: markera perioder där squeeze är on (höjd = BB-banden)
            sqz_vals = sqz.values.flatten().astype(float)
            bb_u_vals = bb_u.values.flatten().astype(float)
            bb_l_vals = bb_l.values.flatten().astype(float)
            for i in range(len(sqz_vals)):
                if sqz_vals[i] > 0.5:
                    bar_bottom = bb_l_vals[i] if not np.isnan(bb_l_vals[i]) else 0
                    bar_top = bb_u_vals[i] if not np.isnan(bb_u_vals[i]) else 0
                    bar_h = bar_top - bar_bottom
                    if bar_h > 0:
                        bar = pg.BarGraphItem(x=[i], height=[bar_h],
                                              width=1, y0=bar_bottom,
                                              brush=pg.mkBrush(255, 165, 0, 25))
                        plot.addItem(bar)

            # Info text
            sqz_label = 'SQUEEZE' if (tr.squeeze_on or tr.squeeze_firing) else 'OFF'
            sqz_txt = f"{sqz_label} ({tr.squeeze_days}d)" if tr.squeeze_days > 0 else sqz_label
            txt = pg.TextItem(sqz_txt, color='#e8e8e8', anchor=(1, 0))
            txt.setPos(len(price) - 1, float(price.iloc[-1]))
            txt.setFont(QFont('JetBrains Mono', 9))
            plot.addItem(txt)

    # ========================================================================
    # OPTIONS / STRADDLE DATA
    # ========================================================================

    def _run_options_fetch(self, squeeze_result):
        """Launch options worker to fetch straddle data for squeeze tickers only."""
        if self._options_running:
            return

        # Bara .ST tickers som är i squeeze
        squeeze_tickers = []
        last_prices = {}
        for ticker, tr in squeeze_result.ticker_results.items():
            if not ticker.endswith('.ST'):
                continue
            if tr.squeeze_on or tr.squeeze_firing:
                squeeze_tickers.append(ticker)
            if hasattr(tr, 'price') and len(tr.price) > 0:
                last_prices[ticker] = float(tr.price.iloc[-1])

        if not squeeze_tickers:
            return

        self._options_running = True
        is_silent = getattr(self, '_options_refresh_silent', False)

        if not is_silent:
            self.squeeze_status_label.setText(f"Loading options for {len(squeeze_tickers)} squeeze tickers...")
            self.squeeze_status_label.setStyleSheet(
                f"color:{COLORS['text_muted']};font-size:11px;")
            if hasattr(self, 'squeeze_options_label'):
                self.squeeze_options_label.setText(
                    f"Fetching straddle data for {len(squeeze_tickers)} tickers...")

        self._options_thread = QThread()
        self._options_worker = OptionsWorker(squeeze_tickers, last_prices)
        self._options_worker.moveToThread(self._options_thread)

        self._options_thread.started.connect(self._options_worker.run)
        self._options_worker.progress.connect(self._on_options_progress)
        self._options_worker.result.connect(self._on_options_result)
        self._options_worker.error.connect(self._on_options_error)
        self._options_worker.finished.connect(self._on_options_finished)
        self._options_worker.finished.connect(self._options_thread.quit)

        self._options_thread.start()

    def _on_options_progress(self, pct, msg):
        if getattr(self, '_options_refresh_silent', False):
            return
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText(f"[Options {pct}%] {msg}")

    def _on_options_error(self, msg):
        print(f"[OPTIONS] Error: {msg}")
        self._options_running = False
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText(f"Options error: {msg}")

    def _on_options_finished(self):
        self._options_running = False
        self._options_refresh_silent = False

    def _on_options_result(self, results):
        """Handle options data — update table and store data."""
        self._options_data = results
        n_with_options = sum(1 for v in results.values() if 'error' not in v)
        print(f"[OPTIONS] Complete: {n_with_options}/{len(results)} tickers with options")

        if hasattr(self, 'squeeze_status_label'):
            n_squeeze = 0
            if self._squeeze_result:
                n_squeeze = self._squeeze_result.n_squeeze_on + self._squeeze_result.n_squeeze_firing
            self.squeeze_status_label.setText(
                f"Complete — {n_squeeze} in squeeze, {n_with_options} with options")
            self.squeeze_status_label.setStyleSheet(
                f"color:{COLORS['positive']};font-size:11px;")

        # Uppdatera tabellen med options-kolumner
        self._update_squeeze_table_options()

        # Starta vol analytics (efter options så vi har DTE-matchade fönster)
        if self._squeeze_result and not self._vol_analytics_running:
            self._run_vol_analytics(self._squeeze_result)

        # Starta auto-refresh timer (var 60:e sekund under marknadstid)
        if not hasattr(self, '_options_refresh_timer'):
            self._options_refresh_timer = QTimer(self)
            self._options_refresh_timer.timeout.connect(self._refresh_options_data)
        self._options_refresh_timer.start(60_000)  # 60 sekunder

    def _refresh_options_data(self):
        """Periodisk uppdatering av options-data (bara under marknadstid)."""
        if self._options_running or not self._squeeze_result:
            return

        # Kolla om det är marknadstid (Stockholm 09:00-17:30)
        from datetime import datetime
        now = datetime.now()
        if now.weekday() >= 5:  # Lördag/söndag
            return
        hour_min = now.hour * 100 + now.minute
        if hour_min < 900 or hour_min > 1730:
            return

        print("[OPTIONS] Auto-refreshing options data...")
        self._options_refresh_silent = True
        self._run_options_fetch(self._squeeze_result)

    def _update_squeeze_table_options(self):
        """Fill in Strike/Straddle/Cost%/IV% columns from options data."""
        table = self.squeeze_table
        for row_idx in range(table.rowCount()):
            ticker_item = table.item(row_idx, 0)
            if not ticker_item:
                continue
            ticker = ticker_item.text()
            opts = self._options_data.get(ticker, {})

            if not opts or 'error' in opts or opts.get('nearest_atm') is None:
                for col in (3, 4, 5, 6):
                    item = QTableWidgetItem('-')
                    item.setForeground(QColor(COLORS['text_disabled']))
                    table.setItem(row_idx, col, item)
                continue

            atm = opts['nearest_atm']
            strike = atm['Strike'] if 'Strike' in atm.index else 0
            straddle = atm['Straddle'] if 'Straddle' in atm.index else 0
            cost_pct = atm['Cost_pct'] if 'Cost_pct' in atm.index else 0
            iv = atm['IV'] if 'IV' in atm.index else None

            # Strike
            strike_item = QTableWidgetItem(f"{strike:.0f}" if strike else '-')
            strike_item.setForeground(QColor(COLORS['text_primary']))
            strike_item.setData(Qt.UserRole, float(strike or 0))
            table.setItem(row_idx, 3, strike_item)

            # Straddle price
            if straddle and not pd.isna(straddle):
                straddle_item = QTableWidgetItem(f"{straddle:.1f}")
                straddle_item.setForeground(QColor(COLORS['accent']))
                straddle_item.setData(Qt.UserRole, float(straddle))
            else:
                straddle_item = QTableWidgetItem('-')
                straddle_item.setForeground(QColor(COLORS['text_disabled']))
            table.setItem(row_idx, 4, straddle_item)

            # Cost %
            if cost_pct and not pd.isna(cost_pct):
                cost_item = QTableWidgetItem(f"{cost_pct:.1f}%")
                if cost_pct < 4:
                    cost_color = COLORS['positive']
                elif cost_pct < 7:
                    cost_color = COLORS['warning']
                else:
                    cost_color = COLORS['negative']
                cost_item.setForeground(QColor(cost_color))
                cost_item.setData(Qt.UserRole, float(cost_pct))
            else:
                cost_item = QTableWidgetItem('-')
                cost_item.setForeground(QColor(COLORS['text_disabled']))
            table.setItem(row_idx, 5, cost_item)

            # IV %
            if iv and not pd.isna(iv):
                iv_item = QTableWidgetItem(f"{iv:.0f}%")
                if iv < 20:
                    iv_color = COLORS['positive']
                elif iv < 35:
                    iv_color = COLORS['warning']
                else:
                    iv_color = COLORS['negative']
                iv_item.setForeground(QColor(iv_color))
                iv_item.setData(Qt.UserRole, float(iv))
            else:
                iv_item = QTableWidgetItem('-')
                iv_item.setForeground(QColor(COLORS['text_disabled']))
            table.setItem(row_idx, 6, iv_item)

    # ========================================================================
    # KEY INSIGHTS (rule-based analysis per squeeze ticker)
    # ========================================================================

    def _update_squeeze_insights(self, ticker: str):
        """Generate and display rule-based key insights for a squeeze ticker."""
        if not hasattr(self, 'squeeze_insights_label'):
            return

        accent = COLORS['accent']
        muted = COLORS['text_muted']
        pos = COLORS['positive']
        warn = COLORS['warning']
        neg = COLORS['negative']
        text = COLORS['text_primary']

        insights = []

        def _add(icon, title, body, color=text):
            insights.append(
                f"<div style='margin-bottom:10px'>"
                f"<span style='color:{accent};font-weight:700;font-size:13px'>{icon} {title}</span><br>"
                f"<span style='color:{color};font-size:12px;line-height:1.5'>{body}</span>"
                f"</div>")

        # ── Squeeze data ──
        tr = None
        if self._squeeze_result and ticker in self._squeeze_result.ticker_results:
            tr = self._squeeze_result.ticker_results[ticker]

        opts = self._options_data.get(ticker, {})
        vol_data = self._vol_analytics_data.get(ticker, {})
        atm_df = opts.get('atm_straddles')
        spot = opts.get('spot', 0)

        # HV from price series
        hv_20d = hv_60d = hv_252d = None
        hv_pct = None
        if tr and hasattr(tr, 'price') and len(tr.price) > 20:
            returns = tr.price.pct_change().dropna()
            sqrt252 = 252 ** 0.5
            hv_20d = float(returns.iloc[-20:].std() * sqrt252 * 100)
            if len(returns) > 60:
                hv_60d = float(returns.iloc[-60:].std() * sqrt252 * 100)
            if len(returns) > 252:
                hv_252d = float(returns.iloc[-252:].std() * sqrt252 * 100)
            hv_pct = tr.hv_percentile

        # ── 1. SQUEEZE CONTEXT ──
        if tr:
            sqz_days = tr.squeeze_days
            if sqz_days >= 30:
                _add("1.", "Squeeze — Lång kompression",
                     f"{ticker} har varit i squeeze i <b>{sqz_days} dagar</b>. "
                     f"Långvariga squeezes tenderar att resultera i starkare breakouts. "
                     f"Hög sannolikhet för signifikant volatilitetsexpansion.", warn)
            elif sqz_days >= 10:
                _add("1.", "Squeeze — Mogen",
                     f"Squeeze pågår sedan <b>{sqz_days} dagar</b>. "
                     f"Bollinger Bands har komprimerat inuti Keltner Channels, "
                     f"vilket signalerar ackumulerad energi för ett kommande utbrott.")
            else:
                _add("1.", "Squeeze — Tidig fas",
                     f"Squeeze startade för <b>{sqz_days} dagar</b> sedan. "
                     f"Volatiliteten har precis börjat komprimera. "
                     f"Avvakta bekräftelse eller ta position i förväg för bättre priser.")

        # ── 2. VOLATILITY ANALYSIS ──
        garch_data = vol_data.get('garch', {})
        garch_current = garch_data.get('current_vol')
        garch_lr = garch_data.get('long_run_vol')
        yz_20d = vol_data.get('yz_20d')

        if hv_20d and hv_252d:
            compression = (1 - hv_20d / hv_252d) * 100
            if compression > 30:
                vol_verdict = f"Kraftig volkompress: HV(20d) {hv_20d:.0f}% vs HV(1y) {hv_252d:.0f}% " \
                              f"(<b>{compression:.0f}% under normalen</b>)."
                vol_color = warn
            elif compression > 10:
                vol_verdict = f"Måttlig kompression: HV(20d) {hv_20d:.0f}% vs HV(1y) {hv_252d:.0f}% " \
                              f"({compression:.0f}% under normalen)."
                vol_color = text
            else:
                vol_verdict = f"Begränsad kompression: HV(20d) {hv_20d:.0f}% nära HV(1y) {hv_252d:.0f}%."
                vol_color = muted

            extras = []
            if yz_20d:
                extras.append(f"Yang-Zhang(20d): {yz_20d:.1f}%")
            if garch_current:
                extras.append(f"GARCH nuvarande: {garch_current:.1f}%")
            if garch_lr:
                extras.append(f"GARCH lång sikt: {garch_lr:.1f}%")
            if extras:
                vol_verdict += " " + " · ".join(extras) + "."

            if garch_current and garch_lr and garch_lr > garch_current * 1.15:
                vol_verdict += f" <b>GARCH-modellen förväntar sig mean reversion uppåt</b> " \
                               f"({garch_current:.0f}% → {garch_lr:.0f}%)."

            _add("2.", "Volatilitet", vol_verdict, vol_color)
        elif hv_20d:
            _add("2.", "Volatilitet", f"HV(20d): {hv_20d:.0f}%"
                 + (f", HV Rank {hv_pct:.0f}%ile" if hv_pct else ""))

        # ── 3. POST-SQUEEZE EXPANSION ──
        post_squeeze = vol_data.get('post_squeeze', {})
        if post_squeeze:
            # Hitta bäst populerad bucket
            best_ps = None
            for w in [90, 60, 120, 30, 180]:
                ps = post_squeeze.get(w, {})
                if ps.get('n_samples', 0) >= 3:
                    best_ps = (w, ps)
                    break

            if best_ps:
                w, ps = best_ps
                exp_mean = ps.get('expansion_mean', 0)
                post_rv = ps.get('post_rv_mean', 0)
                n = ps.get('n_samples', 0)
                during_rv = ps.get('during_rv_mean', 0)

                if exp_mean and exp_mean > 1.5:
                    ps_verdict = (
                        f"Stark historisk expansion: efter {n} tidigare squeezes "
                        f"ökade realiserad vol i snitt <b>{exp_mean:.2f}x</b> "
                        f"(från {during_rv:.0f}% → {post_rv:.0f}%) inom {w} dagar. "
                        f"Talar starkt för straddle-strategi.")
                    ps_color = pos
                elif exp_mean and exp_mean > 1.2:
                    ps_verdict = (
                        f"Måttlig expansion: {exp_mean:.2f}x i snitt "
                        f"({during_rv:.0f}% → {post_rv:.0f}%) efter {n} squeezes ({w}d fönster). "
                        f"Straddle kan vara lönsam om priset är rimligt.")
                    ps_color = text
                else:
                    ps_verdict = (
                        f"Begränsad expansion: {exp_mean:.2f}x i snitt efter {n} squeezes. "
                        f"Historiken stöder inte en aggressiv straddle-position.")
                    ps_color = muted

                _add("3.", "Post-Squeeze Historik (5y)", ps_verdict, ps_color)
            else:
                _add("3.", "Post-Squeeze Historik", "Otillräckligt med data (< 3 squeezes).", muted)

        # ── 4. OPTIONS PRICING ──
        if atm_df is not None and not atm_df.empty:
            from datetime import date as _date, datetime as _dt
            _today = _date.today()
            hv_ref = hv_252d or hv_60d or hv_20d

            # Analysera alla rader
            best_row = None
            best_score = -999

            for _, row in atm_df.iterrows():
                exp = str(row.get('Expiry', ''))[:10]
                try:
                    dte = (_dt.strptime(exp, '%Y-%m-%d').date() - _today).days
                except (ValueError, TypeError):
                    continue

                iv = row.get('IV', float('nan'))
                cost = row.get('Cost_pct', float('nan'))

                # Scoring: prefer 60-180d, low IV/HV, low cost
                score = 0
                if 60 <= dte <= 180:
                    score += 20
                elif 30 <= dte <= 270:
                    score += 10

                if pd.notna(iv) and hv_ref and hv_ref > 0:
                    ratio = iv / hv_ref
                    if ratio < 0.7:
                        score += 15
                    elif ratio < 1.0:
                        score += 10
                    elif ratio > 1.3:
                        score -= 10

                # GARCH spread bonus
                garch_ts = garch_data.get('term_structure', {})
                if garch_ts and pd.notna(iv):
                    keys = sorted(garch_ts.keys())
                    if keys:
                        g_key = min(keys, key=lambda k: abs(k - dte))
                        g_val = garch_ts[g_key].get('garch_vol', 0)
                        if g_val and iv < g_val:
                            score += 10  # IV under GARCH = billigt

                if pd.notna(cost) and cost < 12:
                    score += 5

                if score > best_score:
                    best_score = score
                    best_row = row
                    best_dte = dte

            if best_row is not None:
                b_exp = str(best_row.get('Expiry', ''))[:10]
                b_cost = best_row.get('Cost_pct', float('nan'))
                b_iv = best_row.get('IV', float('nan'))
                b_straddle = best_row.get('Straddle', float('nan'))
                b_strike = best_row.get('Strike', 0)

                rec_parts = [f"<b>Rekommenderad expiry: {b_exp} ({best_dte}d)</b>"]
                if pd.notna(b_straddle):
                    rec_parts.append(f"Straddle: {b_straddle:.1f} SEK (strike {b_strike:.0f})")
                if pd.notna(b_cost):
                    rec_parts.append(f"Kostnad: {b_cost:.1f}% av strike")
                if pd.notna(b_iv) and hv_ref and hv_ref > 0:
                    ratio = b_iv / hv_ref
                    if ratio < 0.8:
                        rec_parts.append(f"IV/HV = {ratio:.2f}x — <span style='color:{pos}'>IV billig vs historisk vol</span>")
                    elif ratio < 1.2:
                        rec_parts.append(f"IV/HV = {ratio:.2f}x — rimlig prissättning")
                    else:
                        rec_parts.append(f"IV/HV = {ratio:.2f}x — <span style='color:{neg}'>IV dyr</span>")

                # GARCH-based assessment
                garch_ts = garch_data.get('term_structure', {})
                if garch_ts and pd.notna(b_iv):
                    keys = sorted(garch_ts.keys())
                    if keys:
                        g_key = min(keys, key=lambda k: abs(k - best_dte))
                        g_vol = garch_ts[g_key].get('garch_vol', 0)
                        if g_vol:
                            spread = b_iv - g_vol
                            if spread < -3:
                                rec_parts.append(
                                    f"GARCH forecasar {g_vol:.0f}% men IV är {b_iv:.0f}% — "
                                    f"<span style='color:{pos}'>{abs(spread):.0f}pp underprissatt</span>")
                            elif spread > 3:
                                rec_parts.append(
                                    f"GARCH forecasar {g_vol:.0f}% men IV är {b_iv:.0f}% — "
                                    f"<span style='color:{neg}'>{spread:.0f}pp överprissatt</span>")

                _add("4.", "Optimal Straddle", "<br>".join(rec_parts), accent)

            # IV skew analysis
            c_ivs = atm_df['C_IV'].dropna()
            p_ivs = atm_df['P_IV'].dropna()
            if len(c_ivs) > 0 and len(p_ivs) > 0:
                avg_c_iv = c_ivs.mean()
                avg_p_iv = p_ivs.mean()
                skew = avg_p_iv - avg_c_iv
                if skew > 5:
                    _add("5.", "Put-Skew",
                         f"Kraftig put-skew: Put IV {avg_p_iv:.0f}% vs Call IV {avg_c_iv:.0f}% "
                         f"(diff {skew:.0f}pp). Marknaden prisar in nedsiderisk. "
                         f"Breakeven lägre på nedsidan, potentiellt fördelaktigt vid volexpansion nedåt.", warn)
                elif skew > 2:
                    _add("5.", "Put-Skew",
                         f"Normal put-skew: Put IV {avg_p_iv:.0f}% vs Call IV {avg_c_iv:.0f}% "
                         f"({skew:.0f}pp skillnad).", muted)

        elif not opts or 'error' in opts:
            _add("4.", "Options", "Ingen optionsdata tillgänglig för denna ticker.", muted)

        # ── 5. OVERALL VERDICT ──
        if tr and opts and not ('error' in opts):
            verdict_parts = []
            verdict_color = text

            # Count bullish/bearish signals
            bullish = 0
            bearish = 0

            if hv_20d and hv_252d and (hv_20d / hv_252d) < 0.7:
                bullish += 1  # Significant vol compression
            if garch_current and garch_lr and garch_lr > garch_current * 1.2:
                bullish += 1  # GARCH mean reversion up
            if post_squeeze:
                for w in [60, 90, 120]:
                    ps = post_squeeze.get(w, {})
                    if ps.get('expansion_mean', 0) and ps['expansion_mean'] > 1.3:
                        bullish += 1
                        break

            # IV pricing check
            if atm_df is not None and not atm_df.empty and hv_ref:
                med_iv = atm_df['IV'].dropna().median()
                if pd.notna(med_iv):
                    if med_iv / hv_ref < 0.7:
                        bullish += 1  # Cheap IV
                    elif med_iv / hv_ref > 1.3:
                        bearish += 1  # Expensive IV

            if bullish >= 3:
                verdict = "STARK SIGNAL"
                verdict_parts.append(
                    f"<span style='color:{pos};font-weight:700;font-size:14px'>{verdict}</span><br>"
                    f"Flera faktorer talar för straddle: volkompress, historisk expansion, "
                    f"och/eller rimlig IV-prissättning.")
                verdict_color = pos
            elif bullish >= 2:
                verdict = "POSITIV SIGNAL"
                verdict_parts.append(
                    f"<span style='color:{pos}'>{verdict}</span><br>"
                    f"Goda förutsättningar för straddle-strategi, "
                    f"men kontrollera specifik expiry och kostnad.")
                verdict_color = pos
            elif bearish >= 2:
                verdict = "SVAG SIGNAL"
                verdict_parts.append(
                    f"<span style='color:{neg}'>{verdict}</span><br>"
                    f"IV relativt dyr och/eller begränsad historisk expansion. "
                    f"Överväg att avvakta bättre priser.")
                verdict_color = neg
            else:
                verdict = "NEUTRAL"
                verdict_parts.append(
                    f"<span style='color:{warn}'>{verdict}</span><br>"
                    f"Blandade signaler. Squeeze identifierad men "
                    f"ingen tydlig edge i prissättningen.")
                verdict_color = warn

            _add("&#9733;", "Sammanfattning", "".join(verdict_parts), verdict_color)

        # Build HTML
        if not insights:
            _add("-", ticker, "Kör squeeze-scan och invänta options/vol-data.", muted)

        html = (
            f"<div style='font-family:JetBrains Mono,monospace;padding:2px;"
            f"color:{COLORS['text_primary']}'>"
            + "".join(insights) +
            "</div>")
        self.squeeze_insights_label.setText(html)

    # ========================================================================
    # VOL ANALYTICS (Post-Squeeze Expansion, Yang-Zhang, GARCH)
    # ========================================================================

    def _run_vol_analytics(self, squeeze_result):
        """Launch VolAnalyticsWorker for squeeze tickers with DTE-matched windows."""
        if self._vol_analytics_running:
            return

        if not VOL_ANALYTICS_AVAILABLE:
            return

        # Samla .ST tickers i squeeze
        squeeze_tickers = []
        for ticker, tr in squeeze_result.ticker_results.items():
            if not ticker.endswith('.ST'):
                continue
            if tr.squeeze_on or tr.squeeze_firing:
                squeeze_tickers.append(ticker)

        if not squeeze_tickers:
            return

        # Bestäm forward_windows baserat på options-DTEs om tillgängliga
        forward_windows = set()
        for ticker in squeeze_tickers:
            opts = self._options_data.get(ticker, {})
            atm_df = opts.get('atm_straddles')
            if atm_df is not None and not atm_df.empty:
                from datetime import date as _date, datetime as _dt
                _today = _date.today()
                for _, row in atm_df.iterrows():
                    exp = str(row.get('Expiry', ''))[:10]
                    try:
                        dte = (_dt.strptime(exp, '%Y-%m-%d').date() - _today).days
                        if dte > 5:
                            forward_windows.add(dte)
                    except (ValueError, TypeError):
                        pass

        # Fallback: standardfönster
        if not forward_windows:
            forward_windows = {30, 60, 90, 120, 180}
        # Lägg alltid till standard-referenspunkter
        forward_windows.update({30, 60, 90, 120, 180})
        fw_list = sorted(forward_windows)

        self._vol_analytics_running = True
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText(
                f"Running vol analytics for {len(squeeze_tickers)} tickers...")
            self.squeeze_status_label.setStyleSheet(
                f"color:{COLORS['text_muted']};font-size:11px;")

        self._vol_thread = QThread()
        self._vol_worker = VolAnalyticsWorker(squeeze_tickers, fw_list)
        self._vol_worker.moveToThread(self._vol_thread)

        self._vol_thread.started.connect(self._vol_worker.run)
        self._vol_worker.progress.connect(self._on_vol_analytics_progress)
        self._vol_worker.result.connect(self._on_vol_analytics_result)
        self._vol_worker.error.connect(self._on_vol_analytics_error)
        self._vol_worker.finished.connect(self._on_vol_analytics_finished)
        self._vol_worker.finished.connect(self._vol_thread.quit)

        self._vol_thread.start()

    def _on_vol_analytics_progress(self, pct, msg):
        if hasattr(self, 'squeeze_status_label'):
            self.squeeze_status_label.setText(f"[Vol {pct}%] {msg}")

    def _on_vol_analytics_error(self, msg):
        print(f"[VOL_ANALYTICS] Error: {msg}")
        self._vol_analytics_running = False

    def _on_vol_analytics_finished(self):
        self._vol_analytics_running = False

    def _on_vol_analytics_result(self, results):
        """Store vol analytics results and refresh panel if a ticker is selected."""
        self._vol_analytics_data = results
        n = len(results)
        print(f"[VOL_ANALYTICS] Complete: {n} tickers analyzed")

        if hasattr(self, 'squeeze_status_label'):
            n_squeeze = 0
            if self._squeeze_result:
                n_squeeze = self._squeeze_result.n_squeeze_on + self._squeeze_result.n_squeeze_firing
            n_opts = sum(1 for v in self._options_data.values() if 'error' not in v)
            self.squeeze_status_label.setText(
                f"Complete — {n_squeeze} in squeeze, {n_opts} with options, {n} vol analytics")
            self.squeeze_status_label.setStyleSheet(
                f"color:{COLORS['positive']};font-size:11px;")

        # Refresh panel + insights om en ticker är vald
        if hasattr(self, 'squeeze_table'):
            current_row = self.squeeze_table.currentRow()
            if current_row >= 0:
                ticker_item = self.squeeze_table.item(current_row, 0)
                if ticker_item:
                    self._update_options_panel(ticker_item.text())
                    self._update_squeeze_insights(ticker_item.text())

    # ========================================================================
    # VOLATILITY SURFACE (3D Plotly)
    # ========================================================================

    def _load_vol_surface(self, ticker: str):
        """Launch worker to fetch full chain and build vol surface."""
        if not WEBENGINE_AVAILABLE or not hasattr(self, 'vol_surface_view'):
            return
        if not OPTIONS_AVAILABLE:
            return

        opts = self._options_data.get(ticker, {})
        if not opts or 'error' in opts:
            return

        spot = opts.get('spot', 0)
        expiries = opts.get('expiries', [])
        if not expiries or spot <= 0:
            return

        # Avanza orderbookId for underlying
        avanza_id = opts.get('orderbook_id', '')
        if not avanza_id:
            return

        # GARCH term structure
        vol_data = self._vol_analytics_data.get(ticker, {})
        garch_ts = vol_data.get('garch', {}).get('term_structure', {})

        # Risk-free rate from options greeks (if available)
        r = 0.02
        atm_df = opts.get('atm_straddles')
        if atm_df is not None and not atm_df.empty:
            # Try to get rate from first row that has greeks
            for _, row in atm_df.iterrows():
                rfr = row.get('risk_free_rate')
                if rfr is not None and rfr > 0:
                    r = rfr / 100 if rfr > 1 else rfr
                    break

        # Stoppa eventuell pagaende vol surface-worker
        if hasattr(self, '_vol_surface_thread') and self._vol_surface_thread is not None:
            try:
                if self._vol_surface_thread.isRunning():
                    self._vol_surface_thread.quit()
                    self._vol_surface_thread.wait(2000)
            except RuntimeError:
                pass
            self._vol_surface_thread = None
            self._vol_surface_worker = None

        # Visa laddar-indikator
        n_exp = len([e for e in expiries if e >= datetime.now().strftime('%Y-%m-%d')])
        self.vol_surface_view.setHtml(
            f"<html><body style='background:{COLORS['bg_card']};color:{COLORS['text_muted']};"
            f"font-family:JetBrains Mono,monospace;display:flex;align-items:center;"
            f"justify-content:center;height:100vh;font-size:13px;'>"
            f"<div>Hamtar IV fran Avanza for {ticker} ({n_exp} expiries)...</div>"
            f"</body></html>")

        # Ateranvand cachad straddle-data (alla strikes, inte bara ATM)
        cached_straddles = opts.get('all_straddles')

        self._vol_surface_thread = QThread()
        self._vol_surface_worker = VolSurfaceWorker(
            avanza_id, ticker, spot, expiries, garch_ts, r,
            cached_straddles=cached_straddles)
        self._vol_surface_worker.moveToThread(self._vol_surface_thread)

        self._vol_surface_thread.started.connect(self._vol_surface_worker.run)
        self._vol_surface_worker.result.connect(self._render_vol_surface)
        self._vol_surface_worker.error.connect(
            lambda e: print(f"[VOLSURF] Error: {e}"))
        self._vol_surface_worker.finished.connect(self._vol_surface_thread.quit)

        self._vol_surface_thread.start()

    def _render_vol_surface(self, data: dict):
        """Render 3D volatility surface with GARCH reference plane."""
        try:
            if not hasattr(self, 'vol_surface_view'):
                return

            surface_df = data.get('surface_df')
            if surface_df is None or surface_df.empty:
                return

            # Ignorera resultat om anvandaren redan bytt ticker
            ticker = data['ticker']
            current_row = self.squeeze_table.currentRow() if hasattr(self, 'squeeze_table') else -1
            if current_row >= 0:
                current_item = self.squeeze_table.item(current_row, 0)
                if current_item and current_item.text() != ticker:
                    print(f"[VOLSURF] Skipping stale result for {ticker}")
                    return

            spot = data['spot']
            garch_ts = data.get('garch_ts', {})

            # Pivot: DTE × Strike → MidIV
            pivot = surface_df.pivot_table(values='MidIV', index='DTE', columns='Strike')
            pivot = pivot.dropna(how='all').dropna(axis=1, how='all')

            if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
                print(f"[VOLSURF] {ticker}: Not enough data for surface "
                      f"({pivot.shape[0]} expiries x {pivot.shape[1]} strikes)")
                return

            strikes = pivot.columns.values.astype(float)
            dtes = pivot.index.values.astype(float)
            iv_grid = pivot.values  # shape (n_dte, n_strike)

            # Check we have enough valid data points
            valid_mask = ~np.isnan(iv_grid)
            if valid_mask.sum() < 4:
                return

            # Build GARCH reference plane (flat across strikes, varies by DTE)
            has_garch = False
            garch_grid = np.full_like(iv_grid, np.nan)
            if garch_ts:
                # Keys might be int or str — normalize
                norm_ts = {}
                for k, v in garch_ts.items():
                    try:
                        norm_ts[int(k)] = v
                    except (ValueError, TypeError):
                        pass
                if norm_ts:
                    garch_keys = sorted(norm_ts.keys())
                    for i, dte in enumerate(dtes):
                        closest = min(garch_keys, key=lambda k: abs(k - dte))
                        gv = norm_ts[closest]
                        garch_vol = gv.get('garch_vol') if isinstance(gv, dict) else gv
                        if garch_vol is not None:
                            garch_grid[i, :] = float(garch_vol)
                            has_garch = True

            # Compute edge: IV - GARCH (negative = IV cheap = edge)
            # Om GARCH saknas, farg baseras pa absolutniva istallet
            if has_garch:
                edge_grid = iv_grid - garch_grid
                colorbar_title = 'IV − GARCH (pp)'
            else:
                # Farg baseras pa IV relativt median
                median_iv = np.nanmedian(iv_grid)
                edge_grid = iv_grid - median_iv
                colorbar_title = 'IV − Median (pp)'

            # Build Plotly HTML
            bg = COLORS['bg_card']
            text_col = COLORS['text_primary']
            accent = COLORS['accent']

            import json as _json

            # Konvertera NaN → null for JSON
            def _grid_to_json(grid):
                return _json.dumps([
                    [None if (v is None or np.isnan(v)) else round(float(v), 2) for v in row]
                    for row in grid
                ])

            # Y-axel: sekventiella index (0,1,2,...) med DTE som tick-labels
            # Sa att varje expiry far lika spacing istallet for interpolerade mellanrum
            dte_labels = [f"{int(d)}d" for d in dtes]
            y_indices = list(range(len(dtes)))

            # Bygg custom hover med riktiga DTE-varden
            # customdata[i][j] = faktisk DTE for rad i
            dte_ints = [int(d) for d in dtes]

            html = f"""<!DOCTYPE html>
<html><head>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:{bg}; overflow:hidden; }}
  #chart {{ width:100%; height:100vh; }}
</style>
</head><body>
<div id="chart"></div>
<script>
var strikes = {_json.dumps([round(float(s), 1) for s in strikes])};
var yIdx = {_json.dumps(y_indices)};
var dteLabels = {_json.dumps(dte_labels)};
var dteValues = {_json.dumps(dte_ints)};
var iv_z = {_grid_to_json(iv_grid)};
var garch_z = {_grid_to_json(garch_grid)};
var edge_z = {_grid_to_json(edge_grid)};
var spot = {spot};

// Bygg customdata: varje cell far [dte, edge]
var customdata = [];
for (var i = 0; i < dteValues.length; i++) {{
  var row = [];
  for (var j = 0; j < strikes.length; j++) {{
    row.push([dteValues[i], edge_z[i] ? edge_z[i][j] : null]);
  }}
  customdata.push(row);
}}

// IV Surface colored by edge
var ivSurface = {{
  x: strikes, y: yIdx, z: iv_z,
  type: 'surface',
  name: 'Implied Vol (Avanza)',
  surfacecolor: edge_z,
  customdata: customdata,
  colorscale: [
    [0, '#22c55e'],
    [0.35, '#166534'],
    [0.5, '#374151'],
    [0.65, '#7f1d1d'],
    [1, '#ef4444']
  ],
  cmin: -10, cmid: 0, cmax: 10,
  colorbar: {{
    title: '{colorbar_title}',
    titleside: 'right',
    titlefont: {{color: '{text_col}', size: 11}},
    tickfont: {{color: '{text_col}', size: 10}},
    ticksuffix: 'pp',
    len: 0.6, y: 0.5,
  }},
  opacity: 0.92,
  hovertemplate: 'Strike: %{{x:.0f}}<br>DTE: %{{customdata[0]}}d<br>IV: %{{z:.1f}}%<br>Edge: %{{customdata[1]:.1f}}pp<extra></extra>',
  contours: {{
    z: {{ show: true, usecolormap: false, color: 'rgba(255,255,255,0.15)', width: 1 }}
  }},
}};

var traces = [ivSurface];

// GARCH reference plane (only if data exists)
var hasGarch = {_json.dumps(has_garch)};
if (hasGarch) {{
  var garchPlane = {{
    x: strikes, y: yIdx, z: garch_z,
    type: 'surface',
    name: 'GARCH Forecast',
    colorscale: [[0, '{accent}'], [1, '{accent}']],
    showscale: false,
    opacity: 0.25,
    hovertemplate: 'GARCH: %{{z:.1f}}%<extra>GARCH</extra>',
  }};
  traces.push(garchPlane);
}}

var layout = {{
  scene: {{
    xaxis: {{
      title: 'Strike',
      color: '{text_col}',
      gridcolor: 'rgba(255,255,255,0.08)',
      zerolinecolor: 'rgba(255,255,255,0.15)',
    }},
    yaxis: {{
      title: 'Expiry (DTE)',
      color: '{text_col}',
      gridcolor: 'rgba(255,255,255,0.08)',
      tickvals: yIdx,
      ticktext: dteLabels,
    }},
    zaxis: {{
      title: 'IV %',
      color: '{text_col}',
      gridcolor: 'rgba(255,255,255,0.08)',
    }},
    bgcolor: '{bg}',
    camera: {{
      eye: {{x: 1.5, y: -1.8, z: 0.8}},
    }},
  }},
  paper_bgcolor: '{bg}',
  plot_bgcolor: '{bg}',
  font: {{ color: '{text_col}', family: 'JetBrains Mono, monospace', size: 11 }},
  title: {{
    text: '{ticker} Volatility Surface (green = edge)',
    font: {{ size: 14, color: '{accent}' }},
  }},
  margin: {{ l: 0, r: 0, t: 40, b: 0 }},
  showlegend: true,
  legend: {{
    x: 0.01, y: 0.98,
    bgcolor: 'rgba(0,0,0,0.5)',
    font: {{ color: '{text_col}', size: 10 }},
  }},
}};

Plotly.newPlot('chart', traces, layout, {{
  responsive: true,
  displayModeBar: true,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
}});
</script></body></html>"""

            self.vol_surface_view.setHtml(html)
            print(f"[VOLSURF] {ticker}: Surface rendered "
                  f"({len(dtes)} expiries x {len(strikes)} strikes, GARCH={'yes' if has_garch else 'no'})")

        except Exception as e:
            print(f"[VOLSURF] Render error: {e}")
            traceback.print_exc()

    def _update_options_panel(self, ticker: str):
        """Update the straddle pricing panel for selected ticker."""
        if not hasattr(self, 'squeeze_options_label'):
            return

        opts = self._options_data.get(ticker, {})
        if not opts or 'error' in opts:
            self.squeeze_options_label.setText(
                f"<span style='color:{COLORS['text_muted']}'>{ticker}: No options data</span>")
            return

        spot = opts.get('spot', 0)
        n_opts = opts.get('n_options', 0)
        expiries = opts.get('expiries', [])
        atm_df = opts.get('atm_straddles')

        # Hämta HV från squeeze-resultatet
        hv_pct = None
        hv_20d = hv_60d = hv_252d = None
        if self._squeeze_result and ticker in self._squeeze_result.ticker_results:
            tr = self._squeeze_result.ticker_results[ticker]
            hv_pct = tr.hv_percentile  # 0-100 percentil

            if hasattr(tr, 'price') and len(tr.price) > 20:
                returns = tr.price.pct_change().dropna()
                sqrt252 = 252 ** 0.5
                hv_20d = float(returns.iloc[-20:].std() * sqrt252 * 100)
                if len(returns) > 60:
                    hv_60d = float(returns.iloc[-60:].std() * sqrt252 * 100)
                if len(returns) > 252:
                    hv_252d = float(returns.iloc[-252:].std() * sqrt252 * 100)

        # CSS
        accent = COLORS['accent']
        muted = COLORS['text_muted']
        pos = COLORS['positive']
        warn = COLORS['warning']
        neg = COLORS['negative']

        # Vol analytics data for this ticker
        vol_data = self._vol_analytics_data.get(ticker, {})
        yz_20d = vol_data.get('yz_20d')
        yz_60d = vol_data.get('yz_60d')
        garch_data = vol_data.get('garch', {})
        garch_current = garch_data.get('current_vol')
        garch_lr = garch_data.get('long_run_vol')
        post_squeeze = vol_data.get('post_squeeze', {})
        has_vol_analytics = bool(vol_data)

        html = [f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;color:{COLORS['text_primary']};margin:0;padding:0'>"]

        # Header: ticker + spot + HV info
        header_parts = [f"<span style='color:{accent};font-size:13px;font-weight:700'>{ticker}</span>"]
        header_parts.append(f"<span style='color:{muted}'>&nbsp;Spot {spot:.1f}")
        if hv_20d is not None:
            header_parts.append(f" &middot; HV(20d) {hv_20d:.0f}%")
        if hv_60d is not None:
            header_parts.append(f" &middot; HV(60d) {hv_60d:.0f}%")
        if hv_252d is not None:
            header_parts.append(f" &middot; HV(1y) {hv_252d:.0f}%")
        if hv_pct is not None:
            header_parts.append(f" &middot; HV Rank {hv_pct:.0f}%ile")
        header_parts.append(f" &middot; {len(expiries)} expiries</span>")
        html.append("".join(header_parts))

        # Vol analytics header line
        if has_vol_analytics:
            vol_parts = [f"<br><span style='color:{muted};font-size:12px'>"]
            if yz_20d is not None:
                vol_parts.append(f"YZ(20d) {yz_20d:.1f}%")
            if yz_60d is not None:
                vol_parts.append(f" &middot; YZ(60d) {yz_60d:.1f}%")
            if garch_current is not None:
                vol_parts.append(f" &middot; GARCH now {garch_current:.1f}%")
            if garch_lr is not None:
                vol_parts.append(f" &middot; GARCH LR {garch_lr:.1f}%")
            # Best post-squeeze expansion summary
            if post_squeeze:
                # Visa det mest relevanta fönstret (90d)
                ps_90 = post_squeeze.get(90, {})
                if ps_90.get('expansion_mean') is not None:
                    vol_parts.append(
                        f" &middot; Post-squeeze exp {ps_90['expansion_mean']:.2f}x"
                        f" (n={ps_90.get('n_samples', 0)})")
            vol_parts.append("</span>")
            html.append("".join(vol_parts))

        avanza_base = "https://www.avanza.se/borshandlade-produkter/optioner-terminer/om-optionen.html"
        link_css = f"color:{accent};text-decoration:none"

        if atm_df is not None and not atm_df.empty:
            # Sub-header CSS
            sub_hdr = f"color:{COLORS['text_muted']};font-size:11px;font-weight:400;padding:1px 3px;text-align:center;border-bottom:1px solid {COLORS['border_subtle']}"
            grp_hdr = f"color:{accent};font-weight:700;font-size:11px;padding:2px 3px;text-align:center;border-bottom:1px solid {COLORS['border_subtle']}"
            cell_sm = f"padding:2px 3px;text-align:right;font-size:12px;white-space:nowrap"

            bdr_subtle = COLORS['border_subtle']
            bdr_l = f"border-left:1px solid {bdr_subtle}"

            # HV for ratio: prefer 1y, fallback 60d, then 20d
            hv_ref = hv_252d or hv_60d or hv_20d
            hv_label = "1y" if hv_252d else ("60d" if hv_60d else "20d")

            vol_cols = 3 if has_vol_analytics else 0

            html.append(f"<table style='width:100%;border-collapse:collapse;margin-top:4px'>")
            # Group header row
            vol_hdr = ""
            if has_vol_analytics:
                vol_hdr = f"<td style='{grp_hdr};{bdr_l}' colspan='3'>VOL ANALYTICS</td>"
            html.append(
                f"<tr>"
                f"<td style='{grp_hdr};text-align:left' rowspan='2'>Expiry</td>"
                f"<td style='{grp_hdr}' rowspan='2'>DTE</td>"
                f"<td style='{grp_hdr}' rowspan='2'>Strike</td>"
                f"<td style='{grp_hdr};{bdr_l}' colspan='5'>CALL</td>"
                f"<td style='{grp_hdr};{bdr_l}' colspan='5'>PUT</td>"
                f"<td style='{grp_hdr};{bdr_l}' colspan='3'>STRADDLE</td>"
                f"{vol_hdr}"
                f"</tr>")
            # Sub-header row
            vol_sub = ""
            if has_vol_analytics:
                vol_sub = (
                    f"<td style='{sub_hdr};{bdr_l}'>PostSqz</td>"
                    f"<td style='{sub_hdr}'>GARCH</td>"
                    f"<td style='{sub_hdr}'>IV&minus;G</td>"
                )
            html.append(
                f"<tr>"
                f"<td style='{sub_hdr};{bdr_l}'>Ask</td>"
                f"<td style='{sub_hdr}'>IV%</td>"
                f"<td style='{sub_hdr}'>&Delta;</td>"
                f"<td style='{sub_hdr}'>&Theta;</td>"
                f"<td style='{sub_hdr}'>Vega</td>"
                f"<td style='{sub_hdr};{bdr_l}'>Ask</td>"
                f"<td style='{sub_hdr}'>IV%</td>"
                f"<td style='{sub_hdr}'>&Delta;</td>"
                f"<td style='{sub_hdr}'>&Theta;</td>"
                f"<td style='{sub_hdr}'>Vega</td>"
                f"<td style='{sub_hdr};{bdr_l}'>Price</td>"
                f"<td style='{sub_hdr}'>Cost%</td>"
                f"<td style='{sub_hdr}'>IV/HV({hv_label})</td>"
                f"{vol_sub}"
                f"</tr>")

            from datetime import date as _date, datetime as _dt
            _today = _date.today()

            def _iv_fmt(iv_val):
                if pd.notna(iv_val):
                    color = pos if iv_val < 20 else (warn if iv_val < 35 else neg)
                    return f"<span style='color:{color};font-weight:600'>{iv_val:.1f}</span>"
                return f"<span style='color:{muted}'>-</span>"

            def _greek_fmt(val, fmt='.3f'):
                return f"{val:{fmt}}" if pd.notna(val) else "-"

            row_bdr = f"border-bottom:1px solid {COLORS['bg_elevated']}"

            for _, row in atm_df.iterrows():
                exp = str(row.get('Expiry', ''))[:10]
                try:
                    dte = (_dt.strptime(exp, '%Y-%m-%d').date() - _today).days
                except (ValueError, TypeError):
                    dte = 0
                strike = row.get('Strike', 0)
                c_ask = row.get('C_Ask', float('nan'))
                p_ask = row.get('P_Ask', float('nan'))
                c_name = row.get('C_Name', '')
                c_id = str(row.get('C_Id', ''))
                p_name = row.get('P_Name', '')
                p_id = str(row.get('P_Id', ''))
                straddle = row.get('Straddle', float('nan'))
                cost = row.get('Cost_pct', float('nan'))
                # Separate greeks per leg
                c_iv = row.get('C_IV', float('nan'))
                p_iv = row.get('P_IV', float('nan'))
                c_delta = row.get('C_Delta', float('nan'))
                p_delta = row.get('P_Delta', float('nan'))
                c_theta = row.get('C_Theta', float('nan'))
                p_theta = row.get('P_Theta', float('nan'))
                c_vega = row.get('C_Vega', float('nan'))
                p_vega = row.get('P_Vega', float('nan'))

                # Call ask with link
                if pd.notna(c_ask) and c_name:
                    c_url = f"{avanza_base}/{c_id}/{c_name.lower()}"
                    c_str = f"<a href='{c_url}' style='{link_css}' title='{c_name}'>{c_ask:.2f}</a>"
                elif pd.notna(c_ask):
                    c_str = f"{c_ask:.2f}"
                else:
                    c_str = "-"

                # Put ask with link
                if pd.notna(p_ask) and p_name:
                    p_url = f"{avanza_base}/{p_id}/{p_name.lower()}"
                    p_str = f"<a href='{p_url}' style='{link_css}' title='{p_name}'>{p_ask:.2f}</a>"
                elif pd.notna(p_ask):
                    p_str = f"{p_ask:.2f}"
                else:
                    p_str = "-"

                # Straddle
                s_str = f"{straddle:.1f}" if pd.notna(straddle) else "-"
                cost_str = f"{cost:.1f}" if pd.notna(cost) else "-"
                s_color = accent if pd.notna(straddle) else muted

                # IV/HV ratio vs 1y HV (not 20d)
                avg_iv = row.get('IV', float('nan'))
                if pd.notna(avg_iv) and hv_ref and hv_ref > 0:
                    ratio = avg_iv / hv_ref
                    ratio_str = f"{ratio:.2f}x"
                    ratio_color = pos if ratio < 0.8 else (warn if ratio < 1.2 else neg)
                else:
                    ratio_str = "-"
                    ratio_color = muted

                # Highlight 60-180d range (ideal for straddles)
                dte_color = accent if 60 <= dte <= 180 else muted

                # Vol analytics columns: find closest DTE match
                vol_cells = ""
                if has_vol_analytics:
                    # Post-squeeze RV for this DTE
                    ps_rv_str = "-"
                    ps_rv_color = muted
                    garch_str = "-"
                    garch_color = muted
                    iv_garch_str = "-"
                    iv_garch_color = muted

                    # Hitta närmaste fönster i post_squeeze och garch term structure
                    garch_ts = garch_data.get('term_structure', {})

                    # Exakt DTE-match eller närmaste
                    def _find_nearest(data_dict, target_dte):
                        if not data_dict:
                            return None, None
                        keys = sorted(data_dict.keys())
                        best_k = min(keys, key=lambda k: abs(k - target_dte))
                        if abs(best_k - target_dte) <= max(15, target_dte * 0.3):
                            return best_k, data_dict[best_k]
                        return None, None

                    if dte > 5:
                        # Post-squeeze expansion
                        ps_key, ps_val = _find_nearest(post_squeeze, dte)
                        if ps_val and ps_val.get('post_rv_mean') is not None:
                            ps_rv = ps_val['post_rv_mean']
                            ps_n = ps_val.get('n_samples', 0)
                            ps_rv_str = f"{ps_rv:.0f}%"
                            ps_rv_color = warn if ps_rv > (hv_20d or 20) else pos
                            if ps_val.get('expansion_mean') is not None:
                                ps_rv_str += f" ({ps_val['expansion_mean']:.1f}x)"

                        # GARCH forecast
                        g_key, g_val = _find_nearest(garch_ts, dte)
                        if g_val and g_val.get('garch_vol') is not None:
                            garch_vol = g_val['garch_vol']
                            garch_str = f"{garch_vol:.0f}%"
                            garch_color = warn if garch_vol > (hv_20d or 20) else pos

                            # IV minus GARCH spread
                            avg_iv_val = row.get('IV', float('nan'))
                            if pd.notna(avg_iv_val):
                                spread = avg_iv_val - garch_vol
                                iv_garch_str = f"{spread:+.1f}"
                                # Positiv = IV dyrt vs GARCH, negativ = billigt
                                iv_garch_color = neg if spread > 5 else (pos if spread < -2 else warn)

                    vol_cells = (
                        f"<td style='{cell_sm};{bdr_l};color:{ps_rv_color}'>{ps_rv_str}</td>"
                        f"<td style='{cell_sm};color:{garch_color}'>{garch_str}</td>"
                        f"<td style='{cell_sm};color:{iv_garch_color};font-weight:600'>{iv_garch_str}</td>"
                    )

                html.append(
                    f"<tr style='{row_bdr}'>"
                    f"<td style='{cell_sm};text-align:left;color:{muted}'>{exp}</td>"
                    f"<td style='{cell_sm};color:{dte_color}'>{dte}d</td>"
                    f"<td style='{cell_sm}'>{strike:.0f}</td>"
                    # Call columns
                    f"<td style='{cell_sm};{bdr_l}'>{c_str}</td>"
                    f"<td style='{cell_sm}'>{_iv_fmt(c_iv)}</td>"
                    f"<td style='{cell_sm};color:{muted}'>{_greek_fmt(c_delta, '+.2f')}</td>"
                    f"<td style='{cell_sm};color:{muted}'>{_greek_fmt(c_theta)}</td>"
                    f"<td style='{cell_sm};color:{muted}'>{_greek_fmt(c_vega)}</td>"
                    # Put columns
                    f"<td style='{cell_sm};{bdr_l}'>{p_str}</td>"
                    f"<td style='{cell_sm}'>{_iv_fmt(p_iv)}</td>"
                    f"<td style='{cell_sm};color:{muted}'>{_greek_fmt(p_delta, '+.2f')}</td>"
                    f"<td style='{cell_sm};color:{muted}'>{_greek_fmt(p_theta)}</td>"
                    f"<td style='{cell_sm};color:{muted}'>{_greek_fmt(p_vega)}</td>"
                    # Straddle columns
                    f"<td style='{cell_sm};{bdr_l};color:{s_color};font-weight:700'>{s_str}</td>"
                    f"<td style='{cell_sm}'>{cost_str}%</td>"
                    f"<td style='{cell_sm};color:{ratio_color};font-weight:600'>{ratio_str}</td>"
                    # Vol analytics columns
                    f"{vol_cells}"
                    f"</tr>")

            html.append("</table>")
        else:
            html.append(f"<p style='color:{muted}'>No priced ATM straddles found</p>")

        html.append("</div>")
        self.squeeze_options_label.setText("".join(html))

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
            '^FTSE': 3, '^FCHI': 3.5, '^STOXX': 10, '^N100': 5,
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

        /* Compute text color for a treemap tile based on its change_pct.
           Interpolates the colorscale → luminance → white or black text. */
        function tileTextColor(pct) {{
            // Colorscale breakpoints: [-3→#ff3333, -0.9→#8b0000, 0→#262626, 0.9→#006400, 3→#33ff33]
            var stops = [
                {{p: -3,   r: 255, g: 51,  b: 51 }},
                {{p: -0.9, r: 139, g: 0,   b: 0  }},
                {{p:  0,   r: 38,  g: 38,  b: 38 }},
                {{p:  0.9, r: 0,   g: 100, b: 0  }},
                {{p:  3,   r: 51,  g: 255, b: 51 }}
            ];
            var v = Math.max(-3, Math.min(3, pct || 0));
            var r, g, b;
            if (v <= stops[0].p) {{ r = stops[0].r; g = stops[0].g; b = stops[0].b; }}
            else if (v >= stops[4].p) {{ r = stops[4].r; g = stops[4].g; b = stops[4].b; }}
            else {{
                for (var i = 0; i < stops.length - 1; i++) {{
                    if (v >= stops[i].p && v <= stops[i+1].p) {{
                        var t = (v - stops[i].p) / (stops[i+1].p - stops[i].p);
                        r = Math.round(stops[i].r + t * (stops[i+1].r - stops[i].r));
                        g = Math.round(stops[i].g + t * (stops[i+1].g - stops[i].g));
                        b = Math.round(stops[i].b + t * (stops[i+1].b - stops[i].b));
                        break;
                    }}
                }}
            }}
            var lum = 0.299 * r + 0.587 * g + 0.114 * b;
            return lum > 140 ? '#000000' : '#e8e8e8';
        }}

        /* Build font_colors array from current d.colors (change_pct per tile) */
        d.font_colors = d.colors.map(function(c) {{ return tileTextColor(c); }});

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
                insidetextfont: {{ color: d.font_colors, size: d.font_sizes }},
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
            var nKeys = Object.keys(updates).length;
            console.log('[JS-DBG] updatePrices called: ' + nKeys + ' symbols, ids.length=' + d.ids.length);

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
                if (upd.implied) {{
                    var iSign = upd.implied_pct >= 0 ? '+' : '';
                    d.text[idx] += '<br><span style="color:#8899aa;font-size:0.85em">Impl. open '
                                 + iSign + upd.implied_pct.toFixed(2) + '%</span>';
                    d.colors[idx] = upd.implied_pct;
                }} else {{
                    d.colors[idx] = newPct;
                }}

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

            if (changedTickers.length === 0) {{
                console.log('[JS-DBG] updatePrices: 0 matched ids! First 5 update keys: '
                    + Object.keys(updates).slice(0,5).join(',')
                    + ' | First 5 ids: ' + d.ids.slice(0,5).join(','));
                return;
            }}
            console.log('[JS-DBG] updatePrices: ' + changedTickers.length + ' tiles matched and updated');

            /* Recompute font colors for changed tiles */
            d.font_colors = d.colors.map(function(c) {{ return tileTextColor(c); }});

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
                insidetextfont: {{ color: d.font_colors, size: d.font_sizes }},
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

                    /* Find this tile's correct resting text color */
                    var restColor = '#e8e8e8';
                    var idx = d.ids.indexOf('item:' + Object.keys(updates).find(function(s) {{
                        return s.replace(/^\\^/, '').replace(/=X$/, '').replace(/=F$/, '') === label;
                    }} || ''));
                    if (idx >= 0 && d.font_colors[idx]) restColor = d.font_colors[idx];

                    if (_flashTimers[label]) clearTimeout(_flashTimers[label]);
                    target.setAttribute('fill', flashColor);
                    _flashTimers[label] = setTimeout(function() {{
                        target.setAttribute('fill', restColor);
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
            # Treemap HTML skrivet OK
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
            # Migration: trade history fields updated

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
            return  # Already ran
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
                    'min_half_life': 0,
                    'max_half_life': 60,
                    'max_adf_pvalue': 0.05,
                    'min_hurst': 0.0,
                    'max_hurst': 0.5,
                    'min_correlation': 0.6,
                }
            
            # Skapa engine med config
            engine = PairsTradingEngine(config=config)
            
            # Sätt cached data direkt
            engine.price_data = price_data
            engine.raw_price_data = raw_price_data if raw_price_data is not None else price_data.copy()
            engine.viable_pairs = viable_pairs
            engine.pairs_stats = pairs_stats if pairs_stats is not None else []
            engine.ou_models = ou_models
            # (window_details removed — single 2y period)

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
            
            # Engine cache applied OK

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
                # Portfolio file changed externally, reloading
                
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
                # Engine cache changed externally, reloading
                
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
        # Manual refresh — restart WS
        self._ws_treemap_rendered = False
        self._start_ws_market_feed(trigger_volatility=False)
    
    def _start_volatility_refresh_safe(self):
        """Safely start volatility refresh after a delay."""
        # Volatility refresh starting
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
                # Only trust cache as full history if it has substantial data (>200 points)
                vix_hist = self._vol_hist_cache.get('^VIX')
                if vix_hist is not None and len(vix_hist) > 200:
                    self._vol_full_history_loaded = True
                else:
                    self._vol_full_history_loaded = False

                saved_str = cache.get('saved_at', '')
                saved_dt = datetime.fromisoformat(saved_str)
                age_hours = (datetime.now() - saved_dt).total_seconds() / 3600
                if age_hours < 24 and self._vol_full_history_loaded:
                    # Cache har full historik — visa cachad data som omedelbar fallback,
                    # men gör alltid en kort 5d-refresh för aktuella priser
                    self._apply_cached_volatility_cards()
                    self.statusBar().showMessage("Volatility cache loaded, refreshing current prices...")
                    # Kort refresh för att uppdatera aktuella priser mot cachad historik
                else:
                    # Visa cachad data som fallback, sedan hämta full historik
                    self._apply_cached_volatility_cards()
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
            'VVIX/VIX': [
                (10, "VIX complacency, spike risk high"), (25, "Low vol-of-vol relative to VIX"),
                (50, "Below median ratio"), (75, "Normal vol regime"),
                (90, "Elevated uncertainty about VIX"), (95, "High vol uncertainty vs fear"),
                (100, "Extreme vol dislocation"),
            ],
        }
        card_map = {'^VIX': 'vix_card', '^VVIX': 'vvix_card',
                    '^SKEW': 'skew_card', '^MOVE': 'move_card',
                    'VVIX/VIX': 'vvix_vix_card'}

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
            'VVIX/VIX': [
                (10, "VIX complacency, spike risk high"), (25, "Low vol-of-vol relative to VIX"),
                (50, "Below median ratio"), (75, "Normal vol regime"),
                (90, "Elevated uncertainty about VIX"), (95, "High vol uncertainty vs fear"),
                (100, "Extreme vol dislocation"),
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
        else:
            card.value_label.setText("N/A")
            card.desc_label.setText(f"No {ticker.replace('^', '')} data available")

    def _is_us_market_open(self) -> bool:
        """Check if US market is currently open via HeaderBar clock."""
        for clock in self.header_bar._market_clocks:
            if clock.city == 'NEW YORK':
                return clock.is_open()
        return False

    def _fetch_futures_implied(self):
        """Hämta futures-priser i bakgrundstråd för implied open."""
        us_open = self._is_us_market_open()
        if not hasattr(self, '_futures_dbg_count'):
            self._futures_dbg_count = 0
        self._futures_dbg_count += 1
        # Log every 4th call (~2 min) or first call
        if self._futures_dbg_count <= 2 or self._futures_dbg_count % 4 == 0:
            print(f"[FUTURES-DBG] _fetch_futures_implied called #{self._futures_dbg_count}, "
                  f"us_market_open={us_open}, "
                  f"already_fetching={getattr(self, '_futures_fetching', False)}")
        if us_open:
            # Rensa implied-flaggor när marknaden är öppen
            for spot in self.US_INDEX_FUTURES:
                if spot in self._market_data_cache:
                    if self._market_data_cache[spot].get('implied_open'):
                        self._market_data_cache[spot]['implied_open'] = False
                        self._ws_cache_dirty = True
                        self._ws_changed_symbols.add(spot)
            return
        if getattr(self, '_futures_fetching', False):
            return
        self._futures_fetching = True
        thread = QThread()
        worker = FuturesImpliedWorker(
            list(self.US_INDEX_FUTURES.values()),
            daily_prev_close=dict(self._daily_prev_close),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._on_futures_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, '_futures_fetching', False))
        worker.finished.connect(worker.deleteLater)
        self._futures_thread_ref = thread  # prevent GC
        self._futures_worker_ref = worker
        thread.start()

    def _on_futures_result(self, data: dict):
        """Implied open från futures' egna change_pct via yf.download."""
        for futures_sym, fdata in data.items():
            spot_sym = self.FUTURES_TO_SPOT.get(futures_sym)
            if not spot_sym or spot_sym not in self._market_data_cache:
                print(f"[FUTURES-DBG] {futures_sym} → spot={spot_sym}, "
                      f"in_cache={spot_sym in self._market_data_cache if spot_sym else 'N/A'}, SKIPPED")
                continue
            self._market_data_cache[spot_sym]['implied_open'] = True
            self._market_data_cache[spot_sym]['implied_pct'] = fdata['change_pct']
            self._ws_cache_dirty = True
            self._ws_changed_symbols.add(spot_sym)
        if data:
            parts = [f"{self.FUTURES_TO_SPOT.get(k,'?')}:{v['change_pct']:+.2f}%" for k, v in data.items()]
            print(f"[FUTURES-DBG] Implied open: {', '.join(parts)}")

    def _refresh_stale_instruments(self):
        """Identify instruments that WS hasn't updated recently and fetch via yf.download."""
        if self._stale_refresh_running:
            return
        now = time.time()
        stale_threshold = 45  # seconds without WS update → considered stale

        stale_tickers = []
        for symbol in self.MARKET_INSTRUMENTS:
            last_update = self._ws_last_update_time.get(symbol, 0)
            if now - last_update > stale_threshold:
                stale_tickers.append(symbol)

        if not stale_tickers:
            return

        self._stale_refresh_running = True
        thread = QThread()
        worker = StaleInstrumentRefreshWorker(
            stale_tickers,
            daily_prev_close=dict(self._daily_prev_close),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(self._on_stale_refresh_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, '_stale_refresh_running', False))
        worker.finished.connect(worker.deleteLater)
        self._stale_refresh_thread_ref = thread  # prevent GC
        self._stale_refresh_worker_ref = worker
        thread.start()

    def _on_stale_refresh_result(self, data: dict):
        """Apply stale instrument prices to the treemap cache."""
        updated = 0
        for symbol, info in data.items():
            if symbol not in self.MARKET_INSTRUMENTS:
                continue
            name, market = self.MARKET_INSTRUMENTS[symbol]
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {
                    'symbol': symbol, 'name': name, 'market': market,
                }
            cached = self._market_data_cache[symbol]
            cached['price'] = info['price']
            cached['change'] = info['change']
            cached['change_pct'] = info['change_pct']
            self._ws_changed_symbols.add(symbol)
            updated += 1

        if updated:
            self._ws_cache_dirty = True

    def _check_daily_prev_close_refresh(self, tickers):
        """Refresh daily prev close if date has changed (dashboard left running overnight)."""
        from datetime import date
        today = date.today()
        if not hasattr(self, '_prev_close_date') or self._prev_close_date != today:
            self._prev_close_date = today
            print(f"[MarketWatch] New day detected ({today}), refreshing daily prev close...")
            self._fetch_daily_prev_close(tickers)
            # Force treemap full re-render with fresh data
            self._ws_treemap_rendered = False
            QTimer.singleShot(3000, self._render_initial_treemap_from_ws)

    def _fetch_daily_prev_close(self, tickers):
        """Fetch daily close data to get accurate previous close for change% calc.

        WS change_percent is unreliable for futures (contract rollovers cause
        huge spurious changes like -22% on CL=F). Daily bars give the correct
        settlement/close price.
        """
        try:
            import yfinance as yf
            import socket
            socket.setdefaulttimeout(30)
            daily = yf.download(tickers, period='5d', interval='1d',
                                progress=False, threads=False, ignore_tz=True,
                                multi_level_index=False)
            if daily is None or daily.empty:
                return
            if len(tickers) == 1:
                # Single ticker: flat columns → rename Close to ticker name
                if 'Close' in daily.columns:
                    daily_close = daily[['Close']].rename(columns={'Close': tickers[0]})
                else:
                    daily_close = daily
            else:
                # Multiple tickers: Close is already a DataFrame with ticker columns
                daily_close = daily['Close'] if 'Close' in daily.columns else daily
            from datetime import date as _date
            today = _date.today()
            for tk in tickers:
                if tk in daily_close.columns:
                    dc = daily_close[tk].dropna()
                    if len(dc) >= 2:
                        # If today's bar exists: prev_close = iloc[-2] (yesterday)
                        # If no bar today: prev_close = iloc[-1] (last close = yesterday)
                        last_dt = dc.index[-1]
                        last_d = last_dt.date() if hasattr(last_dt, 'date') else last_dt
                        if last_d >= today:
                            self._daily_prev_close[tk] = float(dc.iloc[-2])
                        else:
                            self._daily_prev_close[tk] = float(dc.iloc[-1])
            # For index futures: override with fast_info previousClose (settlement price)
            # yf.download daily Close ≠ settlement, causing wrong implied open calculation.
            # Commodity futures don't need this — treemap uses WS change directly.
            _INDEX_FUTURES = {'ES=F', 'NQ=F', 'YM=F', 'RTY=F'}
            futures_in_tickers = [t for t in tickers if t in _INDEX_FUTURES]
            for ftk in futures_in_tickers:
                try:
                    t = yf.Ticker(ftk)
                    fi = t.fast_info
                    pc = float(fi.get('previousClose', 0) or fi.get('previous_close', 0) or 0)
                    if pc > 0:
                        self._daily_prev_close[ftk] = pc
                except Exception:
                    pass
        except Exception as e:
            print(f"[MarketWatch] Daily prev close fetch FAILED: {e}")
            import traceback
            traceback.print_exc()

    def _start_ws_market_feed(self, trigger_volatility=True):
        """Start WebSocket for all market instruments (replaces yf.download for treemap)."""
        tickers = list(self.MARKET_INSTRUMENTS.keys())
        # Add VIX/VVIX for live volatility card updates (SKEW/MOVE via yf.download only)
        for extra in ['^VIX', '^VVIX']:
            if extra not in tickers:
                tickers.append(extra)
        # Add futures tickers so implied open can use their daily prev close
        for ftk in self.US_INDEX_FUTURES.values():
            if ftk not in tickers:
                tickers.append(ftk)

        # Fetch daily previous close for accurate change% calculation
        # (WS change_percent is unreliable for futures due to contract rollovers)
        self._fetch_daily_prev_close(tickers)

        # Pre-populate _market_data_cache for ALL instruments so implied open and
        # stale refresh can find spot symbols immediately (before any WS data arrives)
        for symbol, (name, market) in self.MARKET_INSTRUMENTS.items():
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {
                    'symbol': symbol, 'name': name, 'market': market,
                    'price': 0, 'change': 0, 'change_pct': 0,
                }

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
        # WS feed started

        # Volatilitetspercentiler: ladda från disk-cache om <24h gammal, annars yf.download
        if trigger_volatility:
            QTimer.singleShot(2000, self._load_or_fetch_volatility)

        # Fetch intraday OHLC (1d/5m) to seed overlay candlestick charts
        QTimer.singleShot(5000, self._fetch_intraday_ohlc)

        # Implied open: hämta futures-priser periodiskt (var 30:e sekund)
        if not hasattr(self, '_futures_timer') or self._futures_timer is None:
            self._futures_timer = QTimer(self)
            self._futures_timer.timeout.connect(self._fetch_futures_implied)
            self._futures_timer.start(30000)  # var 30:e sekund
        # Första fetch — vänta 12s så att initial treemap hunnit rendera (populates cache)
        QTimer.singleShot(12000, self._fetch_futures_implied)

        # Stale instrument refresh: fetch prices for instruments that WS doesn't update
        # First refresh after 15s (after initial treemap + first futures), then every 60s
        QTimer.singleShot(15000, self._refresh_stale_instruments)
        if not hasattr(self, '_stale_refresh_timer') or self._stale_refresh_timer is None:
            self._stale_refresh_timer = QTimer(self)
            self._stale_refresh_timer.timeout.connect(self._refresh_stale_instruments)
            self._stale_refresh_timer.start(60000)  # var 60:e sekund

        # Refresh daily prev close periodically (handles overnight dashboard sessions)
        from datetime import date
        self._prev_close_date = date.today()
        if not hasattr(self, '_daily_refresh_timer') or self._daily_refresh_timer is None:
            self._daily_refresh_timer = QTimer(self)
            self._daily_refresh_timer.timeout.connect(lambda: self._check_daily_prev_close_refresh(tickers))
            self._daily_refresh_timer.start(600_000)  # check every 10 min

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
            # Treemap rendered
            self.statusBar().showMessage(f"Market data: {len(items)} instruments ({ws_count} live)")
        else:
            print("[WS] No items for treemap")

    def _fetch_intraday_ohlc(self, tickers_subset=None):
        """Fetch today's intraday OHLC for instruments (background).

        Args:
            tickers_subset: Optional list of specific tickers to fetch.
                           If None, fetches all MARKET_INSTRUMENTS.
        """
        # Clean up old intraday thread if still running
        if self._intraday_thread is not None and self._intraday_thread.isRunning():
            self._intraday_thread.quit()
            self._intraday_thread.wait(3000)
        if self._intraday_worker is not None:
            self._intraday_worker.deleteLater()
            self._intraday_worker = None
        if self._intraday_thread is not None:
            self._intraday_thread.deleteLater()
            self._intraday_thread = None

        tickers = tickers_subset if tickers_subset else list(self.MARKET_INSTRUMENTS.keys())
        print(f"[Intraday] Fetching {len(tickers)} tickers" + (" (retry subset)" if tickers_subset else ""))
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
                    # Commodity futures: use daily_prev_close (full session close)
                    # instead of intraday previous_close (regular session close only).
                    _COMMODITY_FUTURES = {'GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'HG=F'}
                    daily_prev = self._daily_prev_close.get(symbol)
                    if symbol in _COMMODITY_FUTURES and daily_prev and daily_prev > 0:
                        prev_close = daily_prev
                        change_pct = round((close_price - daily_prev) / daily_prev * 100, 2)
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

        # Intraday OHLC seeded

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

        # Determine which tickers are still missing OHLC data
        total_tickers = len(self.MARKET_INSTRUMENTS)
        all_tickers = set(self.MARKET_INSTRUMENTS.keys())
        loaded_tickers = set(self._intraday_ohlc_seed.keys())
        missing_tickers = sorted(all_tickers - loaded_tickers)
        loaded_count = len(loaded_tickers)

        print(f"[Intraday] OHLC status: {loaded_count}/{total_tickers} tickers loaded")
        if missing_tickers:
            print(f"[Intraday] Missing tickers ({len(missing_tickers)}): {missing_tickers[:15]}")
            if len(missing_tickers) > 15:
                print(f"[Intraday] ... and {len(missing_tickers) - 15} more")

        # Retry missing tickers until ALL are loaded
        if missing_tickers and self._intraday_retry_count < self._intraday_max_retries:
            self._intraday_retry_count += 1
            # Backoff: 15s, 30s, 45s, 60s, then 60s forever
            retry_delays = [15000, 30000, 45000, 60000]
            delay = retry_delays[min(self._intraday_retry_count - 1, len(retry_delays) - 1)]
            print(f"[Intraday] {loaded_count}/{total_tickers} OK, {len(missing_tickers)} missing — retry #{self._intraday_retry_count} in {delay//1000}s")
            self.statusBar().showMessage(f"Intraday: {loaded_count}/{total_tickers} tickers, retry #{self._intraday_retry_count}...", 10000)
            QTimer.singleShot(delay, lambda mt=missing_tickers: self._fetch_intraday_ohlc(mt))
        elif not missing_tickers:
            self._intraday_retry_count = 0  # Reset for next refresh cycle
            self.statusBar().showMessage(f"Intraday data loaded: {loaded_count}/{total_tickers} tickers", 5000)
            print(f"[Intraday] All {total_tickers} tickers loaded successfully")
        else:
            self.statusBar().showMessage(f"Intraday: {loaded_count}/{total_tickers} after {self._intraday_max_retries} retries", 10000)
            print(f"[Intraday] Final: {len(missing_tickers)} tickers could not be loaded after {self._intraday_max_retries} retries: {missing_tickers}")

    def _on_intraday_ohlc_error(self, error_msg: str):
        """Handle intraday OHLC error — retry med backoff."""
        print(f"[Intraday] Error: {error_msg}")
        if self._intraday_retry_count < self._intraday_max_retries:
            self._intraday_retry_count += 1
            retry_delays = [15000, 30000, 45000, 60000]
            delay = retry_delays[min(self._intraday_retry_count - 1, len(retry_delays) - 1)]
            # On error, retry only missing tickers
            all_tickers = set(self.MARKET_INSTRUMENTS.keys())
            loaded_tickers = set(self._intraday_ohlc_seed.keys())
            missing_tickers = sorted(all_tickers - loaded_tickers)
            print(f"[Intraday] Retry #{self._intraday_retry_count} in {delay//1000}s after error ({len(missing_tickers)} tickers to fetch)")
            self.statusBar().showMessage(f"Intraday error, retry #{self._intraday_retry_count} in {delay//1000}s...", 10000)
            QTimer.singleShot(delay, lambda mt=missing_tickers: self._fetch_intraday_ohlc(mt if mt else None))
        else:
            self.statusBar().showMessage(f"Intraday fetch failed after {self._intraday_max_retries} retries", 10000)

    def _on_treemap_click(self, ticker: str):
        """Handle treemap tile click — now handled by JS overlay in treemap."""
        pass  # Treemap click handled by JS

    def _on_ws_price_update(self, update: dict):
        """Handle a single live price update from WebSocket."""
        try:
            self._on_ws_price_update_inner(update)
        except Exception as e:
            print(f"[WS] Price update error for {update.get('symbol', '?')}: {e}")

    def _on_ws_price_update_inner(self, update: dict):
        """Inner handler — all WS price update logic."""
        symbol = update['symbol']

        # Track last WS update time per symbol (for stale detection)
        self._ws_last_update_time[symbol] = time.time()

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

            # Commodity futures: WS change_percent is unreliable due to contract
            # rollovers (e.g. -22% on CL=F). Use daily_prev_close instead.
            # All other instruments: WS change values are correct (indices, FX, crypto).
            # Stale/closed instruments are handled by StaleInstrumentRefreshWorker.
            _COMMODITY_FUTURES = {'GC=F', 'SI=F', 'CL=F', 'BZ=F', 'NG=F', 'HG=F'}
            daily_prev = self._daily_prev_close.get(symbol)
            if symbol in _COMMODITY_FUTURES and daily_prev and daily_prev > 0:
                cached['change'] = round(update['price'] - daily_prev, 4)
                cached['change_pct'] = round((update['price'] - daily_prev) / daily_prev * 100, 2)
            else:
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
                    'implied': c.get('implied_open', False),
                    'implied_pct': round(c.get('implied_pct', 0), 2),
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
            try:
                self._ws_worker.price_update.disconnect()
                self._ws_worker.connected.disconnect()
                self._ws_worker.status_message.disconnect()
                self._ws_worker.error.disconnect()
            except (RuntimeError, TypeError):
                pass
        if self._ws_thread is not None and self._ws_thread.isRunning():
            self._ws_thread.quit()
            self._ws_thread.wait(8000)
        if self._ws_worker is not None:
            self._ws_worker.deleteLater()
        if self._ws_thread is not None:
            self._ws_thread.deleteLater()
        self._ws_worker = None
        self._ws_thread = None

    def refresh_market_data(self):
        """Refresh volatility data (VIX, VVIX, SKEW, MOVE) asynchronously.

        OPTIMERING: Flyttar tung yfinance.download till bakgrundstrad.
        First load uses period='max' for full percentile history. Subsequent
        refreshes use period='5d' and merge with cached history.
        Cooldown of 5 minutes prevents excessive fetching.
        """
        # Don't start if already running - använd säker flagga
        if self._volatility_running:
            return

        # Also check if old thread is still alive
        if self._volatility_thread is not None and self._volatility_thread.isRunning():
            return

        # Cooldown: skip if last fetch was less than 5 minutes ago
        now = time.time()
        last_vol = getattr(self, '_volatility_last_start', 0)
        if last_vol > 0 and (now - last_vol) < 300:
            return

        tickers = ['^VIX', '^VVIX', '^SKEW', '^MOVE']

        # First fetch needs period='max' for percentile history.
        # Subsequent refreshes only need period='5d' — merge with cached history.
        # Also force full fetch if VVIX/VIX ratio cache is missing (new card).
        ratio_cache_missing = 'VVIX/VIX' not in self._vol_hist_cache
        if self._vol_full_history_loaded and not ratio_cache_missing:
            vol_period = '5d'
        else:
            vol_period = 'max'
        self._vol_requested_period = vol_period
        vix_n = len(self._vol_hist_cache.get('^VIX', []))
        print(f"[Volatility] period={vol_period}, full_history={self._vol_full_history_loaded}, VIX cache={vix_n} pts")

        # Sätt flagga INNAN vi skapar tråden
        self._volatility_running = True
        self._volatility_last_start = now
        
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
            
            self._volatility_thread.start()
            self.statusBar().showMessage("Fetching volatility data in background...")
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
        
        # Volatility thread finished
        
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
        """Handle received volatility data - runs on GUI thread (safe).

        Supports short-period refreshes (5d): uses cached history for
        percentile/median/mode, only updates current value and sparkline.
        """
        try:
            if len(close) == 0:
                print("No volatility data returned")
                return

            # Short refresh = we requested '5d' and already have full cached history
            requested = getattr(self, '_vol_requested_period', 'max')
            is_short_refresh = self._vol_full_history_loaded and requested != 'max'
            print(f"[Volatility] Received {len(close)} rows, period={requested}, is_short_refresh={is_short_refresh}, VIX cache={len(self._vol_hist_cache.get('^VIX', []))} pts")

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

                        # Use cached history for percentile if short refresh
                        if is_short_refresh and '^VIX' in self._vol_hist_cache:
                            hist_sorted = self._vol_hist_cache['^VIX']
                            vix_pct = np.searchsorted(hist_sorted, vix_val) / len(hist_sorted) * 100
                            median = self._vol_median_cache.get('^VIX', vix_series.median())
                            mode = self._vol_mode_cache.get('^VIX', median)
                        else:
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
                        
                        # Sparkline: use cached sparkline + new values for short refresh
                        if is_short_refresh and '^VIX' in self._vol_sparkline_cache:
                            old_spark = self._vol_sparkline_cache['^VIX']
                            new_vals = vix_series.tolist()
                            history = (old_spark + new_vals)[-SPARKLINE_DAYS:]
                        else:
                            history = vix_series.tail(SPARKLINE_DAYS).tolist()

                        self.vix_card.update_data(vix_val, vix_chg, vix_pct, median, mode, desc, history=history)
                        # Only update full cache on full history fetch
                        if not is_short_refresh:
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

                        if is_short_refresh and '^VVIX' in self._vol_hist_cache:
                            hist_sorted = self._vol_hist_cache['^VVIX']
                            vvix_pct = np.searchsorted(hist_sorted, vvix_val) / len(hist_sorted) * 100
                            median = self._vol_median_cache.get('^VVIX', vvix_series.median())
                            mode = self._vol_mode_cache.get('^VVIX', median)
                        else:
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
                        
                        if is_short_refresh and '^VVIX' in self._vol_sparkline_cache:
                            old_spark = self._vol_sparkline_cache['^VVIX']
                            new_vals = vvix_series.tolist()
                            history = (old_spark + new_vals)[-SPARKLINE_DAYS:]
                        else:
                            history = vvix_series.tail(SPARKLINE_DAYS).tolist()

                        self.vvix_card.update_data(vvix_val, vvix_chg, vvix_pct, median, mode, desc, history=history)
                        if not is_short_refresh:
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

                        if is_short_refresh and '^SKEW' in self._vol_hist_cache:
                            hist_sorted = self._vol_hist_cache['^SKEW']
                            skew_pct = np.searchsorted(hist_sorted, skew_val) / len(hist_sorted) * 100
                            median = self._vol_median_cache.get('^SKEW', skew_series.median())
                            mode = self._vol_mode_cache.get('^SKEW', median)
                        else:
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

                        if is_short_refresh and '^SKEW' in self._vol_sparkline_cache:
                            old_spark = self._vol_sparkline_cache['^SKEW']
                            new_vals = skew_series.tolist()
                            history = (old_spark + new_vals)[-SPARKLINE_DAYS:]
                        else:
                            history = skew_series.tail(SPARKLINE_DAYS).tolist()

                        self.skew_card.update_data(skew_val, skew_chg, skew_pct, median, mode, desc, history=history)
                        if not is_short_refresh:
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
            
            # VVIX/VIX Ratio
            try:
                vvix_vix_ok = False
                cache_key = 'VVIX/VIX'
                has_vvix = '^VVIX' in close.columns
                has_vix = '^VIX' in close.columns
                print(f"[VVIX/VIX] has_vvix={has_vvix}, has_vix={has_vix}, close.columns={list(close.columns)}")
                if has_vvix and has_vix:
                    vvix_s = close['^VVIX'].dropna()
                    vix_s = close['^VIX'].dropna()
                    # Align on common dates
                    common_idx = vvix_s.index.intersection(vix_s.index)
                    if len(common_idx) >= 1:
                        vvix_aligned = vvix_s.loc[common_idx]
                        vix_aligned = vix_s.loc[common_idx]
                        ratio_series = vvix_aligned / vix_aligned
                        ratio_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()

                        if len(ratio_series) > 0:
                            ratio_val = ratio_series.iloc[-1]
                            ratio_prev = ratio_series.iloc[-2] if len(ratio_series) > 1 else ratio_val
                            ratio_chg = ((ratio_val / ratio_prev) - 1) * 100

                            # Build full ratio history from cached VIX/VVIX if ratio cache missing
                            if cache_key not in self._vol_hist_cache:
                                vvix_hist = self._vol_hist_cache.get('^VVIX')
                                vix_hist = self._vol_hist_cache.get('^VIX')
                                if vvix_hist is not None and vix_hist is not None:
                                    # Both are sorted arrays; build ratio from sparkline caches instead
                                    vvix_spark = self._vol_sparkline_cache.get('^VVIX', [])
                                    vix_spark = self._vol_sparkline_cache.get('^VIX', [])
                                    if vvix_spark and vix_spark:
                                        min_len = min(len(vvix_spark), len(vix_spark))
                                        ratio_spark = [vv / vi for vv, vi in zip(vvix_spark[-min_len:], vix_spark[-min_len:]) if vi > 0]
                                        if ratio_spark:
                                            self._vol_hist_cache[cache_key] = np.sort(ratio_spark)
                                            self._vol_median_cache[cache_key] = float(np.median(ratio_spark))
                                            mode_val = self._vol_median_cache[cache_key]
                                            self._vol_mode_cache[cache_key] = mode_val
                                            self._vol_sparkline_cache[cache_key] = ratio_spark

                            if is_short_refresh and cache_key in self._vol_hist_cache:
                                hist_sorted = self._vol_hist_cache[cache_key]
                                ratio_pct = np.searchsorted(hist_sorted, ratio_val) / len(hist_sorted) * 100
                                median = self._vol_median_cache.get(cache_key, ratio_series.median())
                                mode = self._vol_mode_cache.get(cache_key, median)
                            else:
                                if len(ratio_series) >= 20:
                                    ratio_pct = (ratio_series < ratio_val).sum() / len(ratio_series) * 100
                                    median = ratio_series.median()
                                    mode_series = ratio_series.mode()
                                    mode = mode_series.iloc[0] if len(mode_series) > 0 else median
                                elif cache_key in self._vol_hist_cache:
                                    hist_sorted = self._vol_hist_cache[cache_key]
                                    ratio_pct = np.searchsorted(hist_sorted, ratio_val) / len(hist_sorted) * 100
                                    median = self._vol_median_cache.get(cache_key, ratio_val)
                                    mode = self._vol_mode_cache.get(cache_key, median)
                                else:
                                    ratio_pct = 50.0
                                    median = ratio_val
                                    mode = ratio_val

                            if ratio_pct < 10:
                                desc = "VIX complacency, spike risk high"
                            elif ratio_pct < 25:
                                desc = "Low vol-of-vol relative to VIX"
                            elif ratio_pct < 50:
                                desc = "Below median ratio"
                            elif ratio_pct < 75:
                                desc = "Normal vol regime"
                            elif ratio_pct < 90:
                                desc = "Elevated uncertainty about VIX"
                            elif ratio_pct < 95:
                                desc = "High vol uncertainty vs fear"
                            else:
                                desc = "Extreme vol dislocation"

                            if is_short_refresh and cache_key in self._vol_sparkline_cache:
                                old_spark = self._vol_sparkline_cache[cache_key]
                                new_vals = ratio_series.tolist()
                                history = (old_spark + new_vals)[-SPARKLINE_DAYS:]
                            else:
                                history = ratio_series.tail(SPARKLINE_DAYS).tolist()
                                # If only a few points, extend from existing sparkline
                                if len(history) < 5 and cache_key in self._vol_sparkline_cache:
                                    old_spark = self._vol_sparkline_cache[cache_key]
                                    history = (old_spark + history)[-SPARKLINE_DAYS:]

                            self.vvix_vix_card.update_data(ratio_val, ratio_chg, ratio_pct, median, mode, desc, history=history)
                            if not is_short_refresh:
                                self._vol_hist_cache[cache_key] = np.sort(ratio_series.values)
                                self._vol_median_cache[cache_key] = median
                                self._vol_mode_cache[cache_key] = mode
                            self._vol_sparkline_cache[cache_key] = history
                            vvix_vix_ok = True
                if not vvix_vix_ok:
                    self._apply_vol_card_fallback(cache_key, self.vvix_vix_card)
            except Exception as e:
                print(f"VVIX/VIX ratio error: {e}")
                self._apply_vol_card_fallback('VVIX/VIX', self.vvix_vix_card)

            # MOVE (Bond Market Volatility)
            try:
                move_ok = False
                if '^MOVE' in close.columns:
                    move_series = close['^MOVE'].dropna()
                    if len(move_series) > 0:
                        move_val = move_series.iloc[-1]
                        move_prev = move_series.iloc[-2] if len(move_series) > 1 else move_val
                        move_chg = ((move_val / move_prev) - 1) * 100

                        if is_short_refresh and '^MOVE' in self._vol_hist_cache:
                            hist_sorted = self._vol_hist_cache['^MOVE']
                            move_pct = np.searchsorted(hist_sorted, move_val) / len(hist_sorted) * 100
                            median = self._vol_median_cache.get('^MOVE', move_series.median())
                            mode = self._vol_mode_cache.get('^MOVE', median)
                        else:
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

                        if is_short_refresh and '^MOVE' in self._vol_sparkline_cache:
                            old_spark = self._vol_sparkline_cache['^MOVE']
                            new_vals = move_series.tolist()
                            history = (old_spark + new_vals)[-SPARKLINE_DAYS:]
                        else:
                            history = move_series.tail(SPARKLINE_DAYS).tolist()

                        self.move_card.update_data(move_val, move_chg, move_pct, median, mode, desc, history=history)
                        if not is_short_refresh:
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
            # Bara spara efter full history fetch, inte efter 5d refresh
            if self._vol_hist_cache and not is_short_refresh:
                save_volatility_cache(
                    self._vol_hist_cache, self._vol_median_cache,
                    self._vol_mode_cache, self._vol_sparkline_cache)
                self._vol_full_history_loaded = True

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
        
        config = {
            'lookback_period': '2y',
            'min_half_life': 0,
            'max_half_life': 60,
            'max_adf_pvalue': 0.05,
            'min_correlation': 0.60,
        }

        self.worker_thread = QThread()
        self.worker = AnalysisWorker(tickers, '2y', config)
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

        # Auto-run strategy analyses after scan (regardless of tab state)

        # If this was a scheduled scan, send Discord and refresh volatility cache
        if self._is_scheduled_scan:
            self._is_scheduled_scan = False
            self._scheduled_scan_running = False
            self.send_scan_results_to_discord()
            # Skicka dagligt sammanfattningsmail (fördröjt så Discord hinner först)
            QTimer.singleShot(5000, self.send_daily_summary_email)
            # Uppdatera volatilitets-percentilcache (daglig refresh)
            QTimer.singleShot(8000, self._start_volatility_refresh_safe)
            # Kör Master Scanner (alla 4 strategier) efter pairs-scan
            # Fördröjd start så att Discord/email hinner skickas
            QTimer.singleShot(12000, self._run_master_scanner_after_scheduled)
            self.statusBar().showMessage("Scheduled scan complete - results sent to Discord & Email")
    
    # ------------------------------------------------------------------
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
        # Sort by half-life ascending
        if 'half_life_days' in df.columns:
            df = df.sort_values('half_life_days')
        if hasattr(self, 'viable_count_label'):
            self.viable_count_label.setText(f"({len(df)})")

        # Fix #1: Disable updates during batch operations
        self.viable_table.setUpdatesEnabled(False)
        try:
            self.viable_table.setRowCount(len(df))
            for i, row in enumerate(df.itertuples()):
                self.viable_table.setItem(i, 0, QTableWidgetItem(row.pair))
                # Z-Score (column 1)
                try:
                    ou_tmp, _, z_val = self.engine.get_pair_ou_params(row.pair, use_raw_data=True)
                    z_item = QTableWidgetItem(f"{z_val:+.2f}")
                    try:
                        g_p = getattr(row, 'garch_persistence', 0.0)
                        f_d = getattr(row, 'fractional_d', 0.5)
                        h_e = getattr(row, 'hurst_exponent', 0.5)
                        pair_opt_z = ou_tmp.optimal_entry_zscore(
                            garch_persistence=g_p, fractional_d=f_d, hurst=h_e
                        ).get('optimal_z', SIGNAL_TAB_THRESHOLD)
                    except Exception:
                        pair_opt_z = SIGNAL_TAB_THRESHOLD
                    if abs(z_val) >= pair_opt_z:
                        z_item.setForeground(QColor(COLORS['positive'] if z_val < 0 else COLORS['negative']))
                        z_item.setBackground(QColor(COLORS['positive_bg'] if z_val < 0 else COLORS['negative_bg']))
                    self.viable_table.setItem(i, 1, z_item)
                except Exception:
                    self.viable_table.setItem(i, 1, QTableWidgetItem("-"))
                self.viable_table.setItem(i, 2, QTableWidgetItem(f"{row.half_life_days:.2f}"))
                self.viable_table.setItem(i, 3, QTableWidgetItem(f"{row.eg_pvalue:.4f}"))
                self.viable_table.setItem(i, 4, QTableWidgetItem(f"{row.johansen_trace:.2f}"))
                self.viable_table.setItem(i, 5, QTableWidgetItem(f"{row.hurst_exponent:.2f}"))
                # Fractional d (column 6)
                fd = getattr(row, 'fractional_d', None)
                fd_class = getattr(row, 'fractional_d_class', 'unknown')
                if fd is not None and not (isinstance(fd, float) and np.isnan(fd)):
                    fd_item = QTableWidgetItem(f"{fd:.2f}")
                    if fd_class == 'strong_MR':
                        fd_item.setForeground(QColor(COLORS['positive']))
                    elif fd_class == 'weak_MR':
                        fd_item.setForeground(QColor("#66BB6A"))
                    elif fd_class == 'borderline':
                        fd_item.setForeground(QColor(COLORS['warning']))
                    else:
                        fd_item.setForeground(QColor(COLORS['negative']))
                    fd_item.setToolTip(f"Classification: {fd_class}")
                    self.viable_table.setItem(i, 6, fd_item)
                else:
                    self.viable_table.setItem(i, 6, QTableWidgetItem("-"))
                self.viable_table.setItem(i, 7, QTableWidgetItem(f"{row.correlation:.2f}"))
                # Kalman diagnostics (columns 8-11)
                ks = getattr(row, 'kalman_stability', None)
                self.viable_table.setItem(i, 8, QTableWidgetItem(
                    f"{ks:.2f}" if ks is not None else "N/A"))
                ir = getattr(row, 'kalman_innovation_ratio', None)
                self.viable_table.setItem(i, 9, QTableWidgetItem(
                    f"{ir:.2f}" if ir is not None else "N/A"))
                rs = getattr(row, 'kalman_regime_score', None)
                self.viable_table.setItem(i, 10, QTableWidgetItem(
                    f"{rs:.2f}" if rs is not None else "N/A"))
                # P(MR) column 11 — color-coded
                pmr = getattr(row, 'hmm_p_mean_reverting', None)
                if pmr is not None and not (isinstance(pmr, float) and np.isnan(pmr)):
                    pmr_item = QTableWidgetItem(f"{pmr:.3f}")
                    if pmr >= 0.7:
                        pmr_item.setForeground(QColor(COLORS['positive']))
                    elif pmr >= 0.4:
                        pmr_item.setForeground(QColor(COLORS['warning']))
                    else:
                        pmr_item.setForeground(QColor(COLORS['negative']))
                    self.viable_table.setItem(i, 11, pmr_item)
                else:
                    self.viable_table.setItem(i, 11, QTableWidgetItem("-"))
                # HMM State column 12 — color-coded
                hmm_state = getattr(row, 'hmm_current_state', None)
                if hmm_state is not None:
                    _hmm_labels = {0: "MR", 1: "TR", 2: "CR"}
                    _hmm_colors = {0: COLORS['positive'], 1: COLORS['warning'], 2: COLORS['negative']}
                    hmm_text = _hmm_labels.get(int(hmm_state), "?")
                    hmm_item = QTableWidgetItem(hmm_text)
                    hmm_item.setForeground(QColor(_hmm_colors.get(int(hmm_state), COLORS['text_muted'])))
                    self.viable_table.setItem(i, 12, hmm_item)
                else:
                    self.viable_table.setItem(i, 12, QTableWidgetItem("-"))
                # θ Sig. (column 13)
                ts_raw = getattr(row, 'kalman_theta_significant', None)
                ts_text = "N/A" if ts_raw is None else ("Yes" if bool(ts_raw) else "No")
                self.viable_table.setItem(i, 13, QTableWidgetItem(ts_text))
                # Tail Dependence (column 14)
                td = getattr(row, 'tail_dep_lower', None)
                if td is not None and not (isinstance(td, float) and np.isnan(td)):
                    td_val = float(td)
                    td_item = QTableWidgetItem(f"{td_val:.3f}")
                    if td_val > 0.3:
                        td_item.setForeground(QColor(COLORS['negative']))
                        td_item.setToolTip("High tail dependence — co-crash risk!")
                    elif td_val > 0.1:
                        td_item.setForeground(QColor(COLORS['warning']))
                    else:
                        td_item.setForeground(QColor(COLORS['positive']))
                    self.viable_table.setItem(i, 14, td_item)
                else:
                    self.viable_table.setItem(i, 14, QTableWidgetItem("-"))
                # Optimal Z* (column 15) — beräkna om den saknas
                _oz = getattr(row, 'optimal_z', None)
                if _oz is None or (isinstance(_oz, float) and np.isnan(_oz)):
                    # Kolumnen saknas i cache — beräkna on-the-fly
                    try:
                        _ou_tmp = OUProcess(row.theta, row.mu, row.eq_std)
                        _oz = _ou_tmp.optimal_entry_zscore(
                            garch_persistence=getattr(row, 'garch_persistence', 0.0),
                            fractional_d=getattr(row, 'fractional_d', 0.5),
                            hurst=getattr(row, 'hurst_exponent', 0.5),
                        ).get('optimal_z', 2.0)
                    except Exception:
                        _oz = None
                if _oz is not None and _oz != 0:
                    oz_item = QTableWidgetItem(f"{float(_oz):.2f}")
                    oz_item.setForeground(QColor(COLORS['accent']))
                    self.viable_table.setItem(i, 15, oz_item)
                else:
                    self.viable_table.setItem(i, 15, QTableWidgetItem("-"))
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
        """Single-click: select pair."""
        selected = self.viable_table.selectedItems()
        if not selected:
            return
        pair_item = self.viable_table.item(selected[0].row(), 0)
        if pair_item is None:
            return
        self.selected_pair = pair_item.text()

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
        self.navigate_to_page(2)
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
                min_hl = self.engine.config.get('min_half_life', 0)
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
    
    def _on_ou_pair_typing(self, text: str):
        """Debounce: restart timer on each keystroke in OU pair combo."""
        self._ou_pair_debounce.start()

    def _on_ou_pair_debounced(self):
        """Called after user stops typing in OU pair combo."""
        pair = self.ou_pair_combo.currentText()
        self.on_ou_pair_changed(pair)

    def on_ou_pair_changed(self, pair: str):
        """Handle OU pair selection change."""
        if not pair or self.engine is None:
            return
        # Only update if pair is a valid known pair (not partial typing)
        if '/' not in pair:
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
                min_hl = self.engine.config.get('min_half_life', 0)
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
            
            # Prefer Kalman β (current, dynamic) over static OLS β
            kb = pair_stats.get('kalman_beta', pair_stats['hedge_ratio']) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_beta', pair_stats['hedge_ratio'])
            self.ou_hedge_card.set_value(f"{kb:.4f}")
            
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
                hl_lo_str = f"{hl_lo:.0f}" if np.isfinite(hl_lo) else "∞"
                hl_hi_str = f"{hl_hi:.0f}" if np.isfinite(hl_hi) else "∞"
                ci_color = "#ffc107" if not np.isfinite(hl_hi) else None
                self.kalman_theta_ci_card.set_value(f"{hl_lo_str}–{hl_hi_str}d", ci_color)
                
                # μ confidence interval
                mu_lo = kalman.mu - 1.96 * kalman.mu_std
                mu_hi = kalman.mu + 1.96 * kalman.mu_std
                self.kalman_mu_ci_card.set_value(f"[{mu_lo:.2f}, {mu_hi:.2f}]")
                
                # Innovation ratio (should be ~1.0 if filter is well-calibrated)
                ir = kalman.innovation_ratio
                ir_color = "#00c853" if 0.5 < ir < 2.0 else "#ff1744"
                self.kalman_innovation_card.set_value(f"{ir:.2f}", ir_color)
                
                # Regime change CUSUM score — prefer stored scanner value for consistency
                scanner_rc = pair_stats.get('kalman_regime_score', None) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_regime_score', None)
                rc = scanner_rc if scanner_rc is not None and scanner_rc < 90 else kalman.regime_change_score
                rc_color = "#ff1744" if rc > 4.0 else ("#ffc107" if rc > 2.0 else "#00c853")
                rc_text = f"{rc:.1f}" + (" ⚠" if rc > 4.0 else "")
                self.kalman_regime_card.set_value(rc_text, rc_color)
            else:
                # Kalman not available (fallback method used)
                for card in [self.kalman_stability_card, self.kalman_ess_card,
                             self.kalman_theta_ci_card, self.kalman_mu_ci_card,
                             self.kalman_innovation_card, self.kalman_regime_card]:
                    card.set_value("N/A", "#666666")

            # Update GARCH Volatility cards
            garch_alpha = pair_stats.get('garch_alpha', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'garch_alpha', 0)
            garch_beta = pair_stats.get('garch_beta', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'garch_beta', 0)
            garch_persist = pair_stats.get('garch_persistence', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'garch_persistence', 0)
            garch_cvol = pair_stats.get('garch_current_vol', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'garch_current_vol', 0)

            # Show GARCH values — even when persistence=0 (no vol clustering)
            garch_computed = (garch_alpha > 0 or garch_beta > 0 or garch_persist > 0
                              or garch_cvol > 0)
            if garch_computed:
                self.garch_alpha_card.set_value(f"{garch_alpha:.4f}")
                self.garch_beta_card.set_value(f"{garch_beta:.4f}")
                if garch_persist == 0:
                    self.garch_persist_card.set_value("0.0000 (flat)", "#00c853")
                else:
                    persist_color = "#ff1744" if garch_persist > 0.98 else ("#ffc107" if garch_persist > 0.95 else "#00c853")
                    self.garch_persist_card.set_value(f"{garch_persist:.4f}", persist_color)
                cvol_color = "#ff1744" if garch_cvol > 1.5 else ("#ffc107" if garch_cvol > 1.2 else "#00c853")
                self.garch_cvol_card.set_value(f"{garch_cvol:.2f}×", cvol_color)
            else:
                from pairs_engine import ARCH_AVAILABLE
                na_text = "N/A (install arch)" if not ARCH_AVAILABLE else "N/A"
                for card in [self.garch_alpha_card, self.garch_beta_card,
                             self.garch_persist_card, self.garch_cvol_card]:
                    card.set_value(na_text, "#666666")

            # Update Tail Dependence cards
            td_lower = pair_stats.get('tail_dep_lower', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'tail_dep_lower', 0)
            td_upper = pair_stats.get('tail_dep_upper', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'tail_dep_upper', 0)

            # Always show tail dependence values (0.0 = no tail dependence, valid result)
            td_color_l = "#ff1744" if td_lower > 0.3 else ("#ffc107" if td_lower > 0.1 else "#00c853")
            td_color_u = "#ff1744" if td_upper > 0.3 else ("#ffc107" if td_upper > 0.1 else "#00c853")
            self.tail_lower_card.set_value(f"{td_lower:.4f}", td_color_l)
            self.tail_upper_card.set_value(f"{td_upper:.4f}", td_color_u)
            asym = td_upper - td_lower
            self.tail_asym_card.set_value(f"{asym:+.4f}")

            # Update Fractional Integration cards
            frac_d = pair_stats.get('fractional_d', 0.5) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'fractional_d', 0.5)
            frac_class = pair_stats.get('fractional_d_class', 'unknown') if hasattr(pair_stats, 'get') else getattr(pair_stats, 'fractional_d_class', 'unknown')

            d_color = "#00c853" if frac_d < 0 else ("#66BB6A" if frac_d < 0.5 else ("#ffc107" if frac_d < 1.0 else "#ff1744"))
            self.frac_d_card.set_value(f"{frac_d:.3f}", d_color)

            class_labels = {'strong_MR': 'Strong MR', 'weak_MR': 'Weak MR',
                           'borderline': 'Borderline', 'non_stationary': 'Non-Stationary',
                           'unknown': 'N/A'}
            class_colors = {'strong_MR': '#00c853', 'weak_MR': '#66BB6A',
                           'borderline': '#ffc107', 'non_stationary': '#ff1744',
                           'unknown': '#666666'}
            self.frac_class_card.set_value(class_labels.get(frac_class, frac_class),
                                           class_colors.get(frac_class, '#ffffff'))

            # Update Dynamic Hedge Ratio cards
            kb = pair_stats.get('kalman_beta', 1.0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_beta', 1.0)
            kb_stab = pair_stats.get('kalman_beta_stability', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_beta_stability', 0)

            self.kalman_beta_card.set_value(f"{kb:.4f}")
            stab_color = "#00c853" if kb_stab > 0.7 else ("#ffc107" if kb_stab > 0.4 else "#ff1744")
            self.kalman_beta_stab_card.set_value(f"{kb_stab:.1%}", stab_color)

            # Update HMM Regime State cards
            if hasattr(self, 'ou_hmm_regime_card'):
                _get = pair_stats.get if hasattr(pair_stats, 'get') else lambda k, d=None: getattr(pair_stats, k, d)
                hmm_pmr = _get('hmm_p_mean_reverting', 1.0) or 1.0
                hmm_ptr = _get('hmm_p_trending', 0.0) or 0.0
                hmm_pcr = _get('hmm_p_crisis', 0.0) or 0.0
                hmm_state = int(_get('hmm_current_state', 0) or 0)
                hmm_stab = _get('hmm_regime_stability', 1.0) or 1.0
                cusum = _get('kalman_regime_score', 0.0) or 0.0

                _labels = {0: "Mean-Reverting", 1: "Trending", 2: "Crisis"}
                _colors = {0: COLORS['positive'], 1: COLORS['warning'], 2: COLORS['negative']}
                self.ou_hmm_regime_card.set_value(_labels.get(hmm_state, "N/A"), _colors.get(hmm_state, COLORS['text_muted']))

                pmr_color = COLORS['positive'] if hmm_pmr >= 0.7 else (COLORS['warning'] if hmm_pmr >= 0.4 else COLORS['negative'])
                self.ou_hmm_pmr_card.set_value(f"{hmm_pmr:.3f}", pmr_color)
                self.ou_hmm_ptr_card.set_value(f"{hmm_ptr:.3f}", COLORS['warning'] if hmm_ptr > 0.3 else COLORS['text_muted'])
                self.ou_hmm_pcr_card.set_value(f"{hmm_pcr:.3f}", COLORS['negative'] if hmm_pcr > 0.2 else COLORS['text_muted'])
                self.ou_hmm_stab_card.set_value(f"{hmm_stab:.2f}")
                self.ou_cusum_card.set_value(f"{cusum:.2f}")

            # Calculate Expected Move to Mean
            # Spread: S = Y - β*X - α
            # Z = (S - μ) / σ_eq
            # To reach Z=0: ΔS = -Z * σ_eq
            hedge_ratio = pair_stats.get('kalman_beta', pair_stats['hedge_ratio']) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_beta', pair_stats['hedge_ratio'])
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
           
            # Update charts
            if ensure_pyqtgraph():
                self.update_ou_charts(pair, ou, spread, z, pair_stats)

        except Exception as e:
            print(f"OU display error: {e}")

    def _update_window_robustness(self, pair: str, pair_stats):
        """No-op — robustness section removed (single 2y period)."""
        pass

    def update_ou_charts(self, pair: str, ou, spread: pd.Series, z: float, pair_stats):
        """Update OU analytics charts with dates, crosshairs, and synchronized zoom."""
        tickers = pair.split('/')
        if len(tickers) != 2:
            return

        y_ticker, x_ticker = tickers
        hedge_ratio = pair_stats.get('kalman_beta', pair_stats['hedge_ratio']) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_beta', pair_stats['hedge_ratio'])
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

        # ===== SPREAD with μ and ±Opt.Z*σ BANDS =====
        self.ou_zscore_plot.clear()

        spread_values = spread.values

        # Compute Opt. Z* for band multiplier
        try:
            g_p = pair_stats.get('garch_persistence', 0.0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'garch_persistence', 0.0)
            f_d = pair_stats.get('fractional_d', 0.5) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'fractional_d', 0.5)
            h_e = pair_stats.get('hurst_exponent', 0.5) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'hurst_exponent', 0.5)
            band_z = ou.optimal_entry_zscore(
                garch_persistence=g_p, fractional_d=f_d, hurst=h_e
            ).get('optimal_z', 2.0)
        except Exception:
            band_z = 2.0
        band_color = '#ffaa00'  # guldgul för entry-zoner

        # Normalize spread to z-score: z = (spread - μ) / σ
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

            with np.errstate(divide='ignore', invalid='ignore'):
                z_values = (spread_values - mu_vals) / std_vals
                z_values = np.where(np.isfinite(z_values), z_values, 0.0)
        else:
            # Fallback: static μ and σ from final OU params
            mu_vals = np.full(len(spread_values), ou.mu)
            std_vals = np.full(len(spread_values), ou.eq_std)
            z_values = (spread_values - ou.mu) / ou.eq_std if ou.eq_std > 0 else np.zeros(len(spread_values))

        # Plot z-score bands: ±Opt.Z* as fill, μ=0 line
        upper_z = np.full(len(x_axis), band_z)
        lower_z = np.full(len(x_axis), -band_z)
        try:
            upper_curve = self.ou_zscore_plot.plot(x_axis, upper_z, pen=pg.mkPen(band_color, width=1, style=Qt.DashLine))
            lower_curve = self.ou_zscore_plot.plot(x_axis, lower_z, pen=pg.mkPen(band_color, width=1, style=Qt.DashLine))
            fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(255, 170, 0, 20))
            self.ou_zscore_plot.addItem(fill)
        except Exception:
            pass
        self.ou_zscore_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))

        # Plot z-score as bar chart
        # Color bars: red above upper band, green below lower band, orange within bands
        bar_colors = []
        for z in z_values:
            if z > band_z:
                bar_colors.append(pg.mkBrush(255, 50, 50, 160))    # red — above upper band
            elif z < -band_z:
                bar_colors.append(pg.mkBrush(50, 255, 50, 160))    # green — below lower band
            else:
                bar_colors.append(pg.mkBrush(255, 170, 0, 160))    # orange — within bands
        # Clamp z-score for display to ±3 (keeps plot readable)
        z_clamped = np.clip(z_values, -3.0, 3.0)
        bar_width = (x_axis[-1] - x_axis[0]) / len(x_axis) * 0.8 if len(x_axis) > 1 else 1.0
        bar_item = pg.BarGraphItem(x=x_axis, height=z_clamped,
                                   y0=0, width=bar_width, brushes=bar_colors,
                                   pen=pg.mkPen(None))
        self.ou_zscore_plot.addItem(bar_item)

        # Keep unclamped z-score values for crosshair tooltip
        zscore_values = z_values

        # ===== AUTO-RANGE: Reset view to show full data after plotting =====
        self._ou_syncing_plots = True
        self.ou_price_plot.enableAutoRange()
        self.ou_zscore_plot.enableAutoRange(axis='x')
        self.ou_zscore_plot.setYRange(-3.3, 3.3, padding=0)
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
                data_series={'Z-Score': zscore_values},
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

        # Adaptive x-axis: choose unit and range based on half-life
        if hl_days < 0.5:
            # Very fast mean-reversion → show in trading hours
            hours_per_day = 6.5
            x_end = max(5 * hl_days * hours_per_day, 2.0)  # at least 2 hours
            x_vals = np.linspace(0, x_end, 200)
            taus = x_vals / (252 * hours_per_day)  # hours → trading years
            hl_x = hl_days * hours_per_day
            x_label = 'Trading Hours'
        elif hl_days < 5:
            x_end = max(5 * hl_days, 3.0)
            x_vals = np.linspace(0, x_end, 200)
            taus = x_vals / 252
            hl_x = hl_days
            x_label = 'Days'
        else:
            x_end = min(5 * hl_days, 120)
            x_vals = np.linspace(0, x_end, 200)
            taus = x_vals / 252
            hl_x = hl_days
            x_label = 'Days'

        expected_path = np.array([ou.conditional_mean(current_spread, t) for t in taus])
        cond_std = np.array([ou.conditional_std(t) for t in taus])

        # ±2σ confidence band (light fill)
        upper_2s = expected_path + 2 * cond_std
        lower_2s = expected_path - 2 * cond_std
        u2_curve = self.ou_path_plot.plot(x_vals, upper_2s, pen=pg.mkPen(None))
        l2_curve = self.ou_path_plot.plot(x_vals, lower_2s, pen=pg.mkPen(None))
        fill_2s = pg.FillBetweenItem(u2_curve, l2_curve, brush=pg.mkBrush(212, 165, 116, 25))
        self.ou_path_plot.addItem(fill_2s)

        # ±1σ confidence band (darker fill)
        upper_1s = expected_path + cond_std
        lower_1s = expected_path - cond_std
        u1_curve = self.ou_path_plot.plot(x_vals, upper_1s, pen=pg.mkPen(None))
        l1_curve = self.ou_path_plot.plot(x_vals, lower_1s, pen=pg.mkPen(None))
        fill_1s = pg.FillBetweenItem(u1_curve, l1_curve, brush=pg.mkBrush(212, 165, 116, 50))
        self.ou_path_plot.addItem(fill_1s)

        # OU expected path (orange)
        self.ou_path_plot.plot(x_vals, expected_path, pen=pg.mkPen('#d4a574', width=2))

        self.ou_path_plot.addLine(y=ou.mu, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
        self.ou_path_plot.addLine(x=hl_x, pen=pg.mkPen('#ffaa00', width=1, style=Qt.DashLine))
        self.ou_path_plot.setLabel('bottom', x_label)
        
    def _clear_signal_display(self):
        """Reset all signal cards, charts and labels to default state."""
        # State cards
        for card in [self.signal_pair_card, self.signal_z_card, self.signal_dir_card,
                     self.signal_hedge_card, self.signal_hl_card, self.signal_opt_z_card,
                     self.signal_cvar_card]:
            card.set_value("-")
        # Trade metric cards
        for card in [self.signal_winprob_card, self.signal_epnl_card,
                     self.signal_kelly_card, self.signal_rr_card,
                     self.signal_confidence_card, self.signal_avghold_card,
                     self.signal_windays_card, self.signal_lossdays_card]:
            card.set_value("-")
        # Margrabe cards
        if hasattr(self, 'margrabe_fv_card'):
            for card in [self.margrabe_fv_card, self.margrabe_iv_card,
                         self.margrabe_delta_y_card, self.margrabe_delta_x_card,
                         self.margrabe_gamma_card, self.margrabe_vega_card,
                         self.margrabe_theta_card]:
                card.set_value("-")
        # Charts
        if hasattr(self, 'signal_price_plot'):
            self.signal_price_plot.clear()
        if hasattr(self, 'signal_zscore_plot'):
            self.signal_zscore_plot.clear()
        if hasattr(self, 'signal_mc_plot'):
            self.signal_mc_plot.clear()
        # Disable open position button
        if hasattr(self, 'open_position_btn'):
            self.open_position_btn.setEnabled(False)

    def update_signals_list(self):
        """Update signals dropdown."""
        self.signal_combo.clear()
        self._clear_signal_display()

        if self.engine is None or self.engine.viable_pairs is None:
            self.signal_count_label.setText("⚡ 0 viable pairs with |Z| ≥ Opt.Z*")
            return

        signals = []
        min_hl = self.engine.config.get('min_half_life', 0)
        max_hl = self.engine.config.get('max_half_life', 60)

        for row in self.engine.viable_pairs.itertuples():
            try:
                ou, spread, z = self.engine.get_pair_ou_params(row.pair, use_raw_data=True)

                # Re-check viability with CURRENT Kalman half-life
                current_hl = ou.half_life_days()
                if not (min_hl <= current_hl <= max_hl):
                    continue

                # Use per-pair optimal z* as entry threshold
                try:
                    g_p = getattr(row, 'garch_persistence', 0.0)
                    f_d = getattr(row, 'fractional_d', 0.5)
                    h_e = getattr(row, 'hurst_exponent', 0.5)
                    opt_result = ou.optimal_entry_zscore(
                        garch_persistence=g_p, fractional_d=f_d, hurst=h_e)
                    opt_z = opt_result.get('optimal_z', SIGNAL_TAB_THRESHOLD)
                except Exception:
                    opt_z = SIGNAL_TAB_THRESHOLD

                if abs(z) >= opt_z:
                    # Filter out pairs with negative expected PnL
                    try:
                        exit_z = self.engine.config.get('exit_zscore', 0.0)
                        stop_z_cfg = self.engine.config.get('stop_zscore', 4.0)
                        cur_s = spread.iloc[-1]
                        if z > 0:
                            tp_s = ou.spread_from_z(exit_z)
                            sl_s = ou.spread_from_z(stop_z_cfg)
                        else:
                            tp_s = ou.spread_from_z(-exit_z)
                            sl_s = ou.spread_from_z(-stop_z_cfg)
                        epnl = ou.expected_pnl(cur_s, tp_s, sl_s)['expected_pnl']
                        if epnl <= 0:
                            continue
                    except Exception:
                        pass  # If calculation fails, keep the pair
                    signals.append((row.pair, z, opt_z))
            except (ValueError, KeyError, Exception):
                pass

        self.signal_count_label.setText(
            f"⚡ {len(signals)} viable pairs with |Z| ≥ Opt.Z* & E[PnL]>0")
        
        for pair, z, opt_z in signals:
            direction = "LONG" if z < 0 else "SHORT"
            self.signal_combo.addItem(f"{pair}  Z: {z:.2f} (opt: {opt_z:.2f})", pair)

    def on_signal_changed(self, text: str):
        """Handle signal selection change."""
        # Extract pair name from combo data or text
        pair = (self.signal_combo.currentData() or text.split("  ")[0].strip()) if text else ""
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

            # Optimal Z* card
            if hasattr(self, 'signal_opt_z_card'):
                try:
                    g_p = pair_stats.get('garch_persistence', 0.0)
                    f_d = pair_stats.get('fractional_d', 0.5)
                    h_e = pair_stats.get('hurst_exponent', 0.5)
                    opt_result = ou.optimal_entry_zscore(
                        garch_persistence=g_p,
                        fractional_d=f_d,
                        hurst=h_e,
                    )
                    opt_z = opt_result['optimal_z']
                    rt_days = opt_result['roundtrip_days']
                    self._signal_opt_z = opt_z
                    self.signal_opt_z_card.set_value(f"{opt_z:.2f}")
                    self.signal_opt_z_card.setToolTip(
                        f"Optimal entry z-score: {opt_z:.2f}\n"
                        f"Expected roundtrip: {rt_days:.1f} days")
                except Exception:
                    self._signal_opt_z = None
                    self.signal_opt_z_card.set_value("-")

            # CVaR (95%) card
            if hasattr(self, 'signal_cvar_card'):
                garch_cvar = pair_stats.get('garch_cvar_95', 0) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'garch_cvar_95', 0)
                if garch_cvar > 0:
                    self.signal_cvar_card.set_value(f"{garch_cvar:.4f}", COLORS['warning'])
                else:
                    self.signal_cvar_card.set_value("N/A", "#666666")

            # Margrabe spread option valuation
            if hasattr(self, 'margrabe_fv_card'):
                try:
                    marg = self.engine.margrabe_valuation(pair)
                    if marg:
                        self.margrabe_fv_card.set_value(f"{marg['fair_value']:.2f}")
                        self.margrabe_iv_card.set_value(f"{marg['implied_vol']:.1%}")
                        self.margrabe_delta_y_card.set_value(f"{marg['delta_y']:.4f}")
                        self.margrabe_delta_x_card.set_value(f"{marg['delta_x']:.4f}")
                        self.margrabe_gamma_card.set_value(f"{marg['gamma']:.6f}")
                        self.margrabe_vega_card.set_value(f"{marg['vega']:.2f}")
                        self.margrabe_theta_card.set_value(f"{marg['theta']:.4f}")
                    else:
                        for card in [self.margrabe_fv_card, self.margrabe_iv_card,
                                     self.margrabe_delta_y_card, self.margrabe_delta_x_card,
                                     self.margrabe_gamma_card, self.margrabe_vega_card,
                                     self.margrabe_theta_card]:
                            card.set_value("N/A", "#666666")
                except Exception:
                    for card in [self.margrabe_fv_card, self.margrabe_iv_card,
                                 self.margrabe_delta_y_card, self.margrabe_delta_x_card,
                                 self.margrabe_gamma_card, self.margrabe_vega_card,
                                 self.margrabe_theta_card]:
                        card.set_value("N/A", "#666666")

            # ===== TRADE METRICS cards =====
            if hasattr(self, 'signal_winprob_card'):
                try:
                    current_spread = spread.iloc[-1]
                    exit_z = self.engine.config['exit_zscore']
                    stop_z = self.engine.config['stop_zscore']

                    # Compute TP/SL levels based on direction
                    if z > 0:
                        tp_spread = ou.spread_from_z(exit_z)
                        sl_spread = ou.spread_from_z(stop_z)
                    else:
                        tp_spread = ou.spread_from_z(-exit_z)
                        sl_spread = ou.spread_from_z(-stop_z)

                    metrics = ou.expected_pnl(current_spread, tp_spread, sl_spread)

                    # Win probability
                    wp = metrics['win_prob']
                    wp_color = "#00c853" if wp >= 0.65 else (COLORS['warning'] if wp >= 0.50 else "#ff1744")
                    self.signal_winprob_card.set_value(f"{wp:.1%}", wp_color)

                    # Expected PnL
                    epnl = metrics['expected_pnl']
                    epnl_color = "#00c853" if epnl > 0 else "#ff1744"
                    self.signal_epnl_card.set_value(f"{epnl:+.4f}", epnl_color)

                    # Kelly fraction (display as 1/4 Kelly)
                    kf_raw = metrics['kelly_fraction']
                    kf = kf_raw * 0.25  # 1/4 Kelly
                    kf_color = "#00c853" if kf > 0.10 else (COLORS['warning'] if kf > 0 else "#ff1744")
                    self.signal_kelly_card.set_value(f"{kf:.1%}", kf_color)

                    # Risk : Reward
                    rr = metrics['risk_reward']
                    self.signal_rr_card.set_value(f"1 : {rr:.1f}")

                    # Confidence (use raw Kelly for threshold comparison)
                    if wp >= 0.65 and kf_raw >= 0.15:
                        conf, conf_color = "HIGH", "#00c853"
                    elif wp >= 0.50:
                        conf, conf_color = "MEDIUM", COLORS['warning']
                    else:
                        conf, conf_color = "LOW", "#ff1744"
                    self.signal_confidence_card.set_value(conf, conf_color)

                    # Average holding days
                    avg_hold = (metrics['avg_win_days'] * metrics['win_prob'] +
                                metrics['avg_loss_days'] * metrics['loss_prob'])
                    if np.isnan(avg_hold):
                        avg_hold = ou.half_life_days() * 2
                    self.signal_avghold_card.set_value(f"{avg_hold:.1f}d")

                    # Win / loss days
                    win_d = metrics['avg_win_days']
                    loss_d = metrics['avg_loss_days']
                    self.signal_windays_card.set_value(
                        f"{win_d:.1f}d" if not np.isnan(win_d) else "N/A")
                    self.signal_lossdays_card.set_value(
                        f"{loss_d:.1f}d" if not np.isnan(loss_d) else "N/A")

                    # MC fan chart
                    if hasattr(self, 'signal_mc_plot'):
                        try:
                            mc_data = ou.mc_fan_data(current_spread, z,
                                                     z_exit=exit_z, z_stop=stop_z)
                            self._update_mc_plot(mc_data, ou)
                        except Exception as mc_err:
                            print(f"MC chart error: {mc_err}")

                except Exception as tm_err:
                    print(f"Trade metrics error: {tm_err}")
                    for card in [self.signal_winprob_card, self.signal_epnl_card,
                                 self.signal_kelly_card, self.signal_rr_card,
                                 self.signal_confidence_card, self.signal_avghold_card,
                                 self.signal_windays_card, self.signal_lossdays_card]:
                        card.set_value("N/A", "#666666")

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
    
    def _update_mc_plot(self, mc_data, ou):
        """Update Monte Carlo fan chart with simulation data."""
        pg = get_pyqtgraph()
        if pg is None or not hasattr(self, 'signal_mc_plot'):
            return

        plot = self.signal_mc_plot
        plot.clear()

        time_days = mc_data['time_days']
        fan = mc_data['fan']

        # 10-90 percentile band (very faint)
        p10_curve = plot.plot(time_days, fan['p10'], pen=pg.mkPen(None))
        p90_curve = plot.plot(time_days, fan['p90'], pen=pg.mkPen(None))
        fill_outer = pg.FillBetweenItem(p10_curve, p90_curve,
                                         brush=pg.mkBrush(0, 229, 255, 20))
        plot.addItem(fill_outer)

        # 25-75 percentile band (semi-transparent)
        p25_curve = plot.plot(time_days, fan['p25'], pen=pg.mkPen(None))
        p75_curve = plot.plot(time_days, fan['p75'], pen=pg.mkPen(None))
        fill_inner = pg.FillBetweenItem(p25_curve, p75_curve,
                                         brush=pg.mkBrush(0, 229, 255, 50))
        plot.addItem(fill_inner)

        # Median line (bright cyan)
        plot.plot(time_days, fan['p50'],
                  pen=pg.mkPen('#00e5ff', width=2))

        # Sample paths (thin, semi-transparent)
        path_colors = ['#d4a574', '#2196f3', '#ab47bc', '#26a69a',
                        '#ef5350', '#66bb6a', '#ffa726', '#42a5f5',
                        '#ec407a', '#78909c']
        paths = mc_data['paths']
        n_show = min(20, len(paths))
        for i in range(n_show):
            color = path_colors[i % len(path_colors)]
            plot.plot(time_days[:paths.shape[1]], paths[i],
                      pen=pg.mkPen(color, width=1, style=Qt.SolidLine),
                      connect='finite')

        # TP line (green dashed)
        tp = mc_data['take_profit_level']
        plot.addLine(y=tp, pen=pg.mkPen('#00c853', width=1, style=Qt.DashLine))

        # SL line (red dashed)
        sl = mc_data['stop_loss_level']
        plot.addLine(y=sl, pen=pg.mkPen('#ff1744', width=1, style=Qt.DashLine))

        # Mean line (white dotted)
        plot.addLine(y=mc_data['mu_level'],
                     pen=pg.mkPen('#ffffff', width=1, style=Qt.DotLine))

        # Entry dot at t=0
        if len(fan['p50']) > 0:
            entry_val = fan['p50'][0]
            scatter = pg.ScatterPlotItem(
                [0], [entry_val], size=8,
                brush=pg.mkBrush('#00e5ff'), pen=pg.mkPen('#ffffff', width=1))
            plot.addItem(scatter)

        plot.enableAutoRange()

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
            hedge_ratio = pair_stats.get('kalman_beta', pair_stats['hedge_ratio']) if hasattr(pair_stats, 'get') else getattr(pair_stats, 'kalman_beta', pair_stats['hedge_ratio'])
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
            # Band multiplier: use Opt. Z* if available, else fallback to 2
            band_z = getattr(self, '_signal_opt_z', None) or 2.0
            band_color = '#ffaa00'  # guldgul för entry-zoner

            # Normalize spread to z-score: z = (spread - μ) / σ
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

                with np.errstate(divide='ignore', invalid='ignore'):
                    z_values = (spread_values - mu_vals) / std_vals
                    z_values = np.where(np.isfinite(z_values), z_values, 0.0)
            else:
                mu_vals = np.full(len(spread_values), ou.mu)
                std_vals = np.full(len(spread_values), ou.eq_std)
                z_values = (spread_values - ou.mu) / ou.eq_std if ou.eq_std > 0 else np.zeros(len(spread_values))

            # Plot z-score bands: ±Opt.Z* as fill, μ=0 line
            upper_z = np.full(len(x_axis), band_z)
            lower_z = np.full(len(x_axis), -band_z)
            try:
                upper_curve = self.signal_zscore_plot.plot(x_axis, upper_z, pen=pg.mkPen(band_color, width=1, style=Qt.DashLine))
                lower_curve = self.signal_zscore_plot.plot(x_axis, lower_z, pen=pg.mkPen(band_color, width=1, style=Qt.DashLine))
                fill = pg.FillBetweenItem(upper_curve, lower_curve, brush=pg.mkBrush(255, 170, 0, 20))
                self.signal_zscore_plot.addItem(fill)
            except Exception:
                pass
            self.signal_zscore_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))

            # Plot z-score as bar chart
            # Color bars: red above upper band, green below lower band, orange within bands
            bar_colors = []
            for z in z_values:
                if z > band_z:
                    bar_colors.append(pg.mkBrush(255, 50, 50, 160))    # red — above upper band
                elif z < -band_z:
                    bar_colors.append(pg.mkBrush(50, 255, 50, 160))    # green — below lower band
                else:
                    bar_colors.append(pg.mkBrush(255, 170, 0, 160))    # orange — within bands
            # Clamp z-score for display to ±3 (keeps plot readable)
            z_clamped = np.clip(z_values, -3.0, 3.0)
            bar_width = (x_axis[-1] - x_axis[0]) / len(x_axis) * 0.8 if len(x_axis) > 1 else 1.0
            bar_item = pg.BarGraphItem(x=x_axis, height=z_clamped,
                                       y0=0, width=bar_width, brushes=bar_colors,
                                       pen=pg.mkPen(None))
            self.signal_zscore_plot.addItem(bar_item)
            zscore_values = z_values

            # ===== AUTO-RANGE: Reset view to show full data after plotting =====
            self._signal_syncing_plots = True
            self.signal_price_plot.enableAutoRange()
            self.signal_zscore_plot.enableAutoRange(axis='x')
            self.signal_zscore_plot.setYRange(-3.3, 3.3, padding=0)
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
                    data_series={'Z-Score': zscore_values},
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
        Runs in a background QThread to keep the GUI responsive.
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
        self._ms_pair = pair  # spara för callback

        # VIKTIGT: Rensa BÅDA korten FÖRST innan vi hämtar nya produkter
        self._clear_mini_future_card('y', y_ticker, dir_y)
        self._clear_mini_future_card('x', x_ticker, dir_x)
        self._clear_mf_position_sizing()
        self.current_mini_futures = {'y': None, 'x': None}

        # Load ticker mapping (fast, file-based — OK on main thread)
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

        # Stop any previous MS fetch thread
        if self._ms_fetch_thread is not None:
            try:
                if self._ms_fetch_thread.isRunning():
                    self._ms_fetch_thread.quit()
                    self._ms_fetch_thread.wait(3000)
            except RuntimeError:
                pass  # Already deleted
            # Keep old thread ref alive until it finishes (prevent GC crash)
            if not hasattr(self, '_ms_old_threads'):
                self._ms_old_threads = []
            self._ms_old_threads.append(self._ms_fetch_thread)
            self._ms_fetch_thread = None

        # Launch async worker
        self._ms_fetch_worker = MSInstrumentWorker(
            y_ticker, x_ticker, dir_y, dir_x, ticker_to_ms, ticker_to_ms_asset)
        thread = QThread()
        self._ms_fetch_thread = thread
        self._ms_fetch_worker.moveToThread(thread)

        thread.started.connect(self._ms_fetch_worker.run)
        self._ms_fetch_worker.result.connect(self._on_ms_instruments_loaded)
        self._ms_fetch_worker.error.connect(lambda e: self.statusBar().showMessage(f"MS fetch error: {e}"))
        self._ms_fetch_worker.finished.connect(thread.quit)
        self._ms_fetch_worker.finished.connect(self._ms_fetch_worker.deleteLater)
        # Only clear ref if this thread is still the active one (prevents clobbering new thread)
        thread.finished.connect(lambda t=thread: setattr(self, '_ms_fetch_thread', None) if self._ms_fetch_thread is t else None)
        thread.finished.connect(thread.deleteLater)

        thread.start()

    def _on_ms_instruments_loaded(self, all_instruments_y: list, all_instruments_x: list, ticker_to_ms: dict):
        """Callback when MS instrument fetch completes — updates UI on main thread."""
        # Guard: ignore stale results if user switched pair during fetch
        pair = getattr(self, '_ms_pair', '')
        signal_pair = getattr(self, 'signal_selected_pair', '')
        if pair != signal_pair:
            print(f"[MS] Ignoring stale result for {pair} (current pair: {signal_pair})")
            return

        # Spara ofiltrerade listor + mapping för produkttyp-toggle
        self._all_instruments_y = all_instruments_y
        self._all_instruments_x = all_instruments_x
        self._last_ticker_to_ms = ticker_to_ms

        tickers = pair.split('/')
        if len(tickers) != 2:
            return
        y_ticker, x_ticker = tickers
        dir_y = getattr(self, '_current_dir_y', 'Long')
        dir_x = getattr(self, '_current_dir_x', 'Short')

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

            # Set notional to minimum capital for this pair
            # (reset per pair selection to avoid carry-over from previous pairs)
            min_capital = math.ceil(min_result['total_capital'])
            self.notional_spin.blockSignals(True)
            self.notional_spin.setValue(min_capital)
            self.notional_spin.blockSignals(False)

            notional = self.notional_spin.value()

            # Skala: hitta största X-enheter som ryms inom notional
            units_y = min_result['units_y']
            units_x = min_result['units_x']
            if notional > min_result['total_capital']:
                ux = units_x
                while True:
                    next_x = ux + 1
                    ratio = next_x * exp_per_unit_x / (beta * exp_per_unit_y) if (beta > 0 and exp_per_unit_y > 0) else 1.0
                    next_y = max(1, math.ceil(ratio)) if math.isfinite(ratio) else 1
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
        """Filtrera instrument for ett ben baserat pa dess aktiva produkttyp."""
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

        # Anvand alltid originaltickern fran all_insts (forsvinner inte vid tom filtered)
        if all_insts:
            ticker = all_insts[0]['ticker']
            direction = all_insts[0]['direction']
        else:
            ticker = ''
            direction = 'Long'

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
            if mf_data.get('instrument_price'):
                info_text += f"\nInstrument Price: {mf_data['instrument_price']:.2f} SEK"

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
    
    # ════════════════════════════════════════════════════════════════
    # STRADDLE POSITION (from TTM Squeeze tab)
    # ════════════════════════════════════════════════════════════════

    def _open_straddle_dialog(self):
        """Show dialog to open a straddle position from squeeze tab."""
        if not hasattr(self, 'squeeze_table'):
            return
        row = self.squeeze_table.currentRow()
        if row < 0:
            return
        ticker_item = self.squeeze_table.item(row, 0)
        if not ticker_item:
            return
        ticker = ticker_item.text()
        opts = self._options_data.get(ticker, {})
        atm_df = opts.get('atm_straddles')
        if atm_df is None or atm_df.empty:
            QMessageBox.warning(self, "No Data", f"No options data for {ticker}")
            return

        spot = opts.get('spot', 0)

        # Build expiry choices
        from datetime import date as _date, datetime as _dt
        _today = _date.today()
        expiry_choices = []
        for _, r in atm_df.iterrows():
            exp = str(r.get('Expiry', ''))[:10]
            try:
                dte = (_dt.strptime(exp, '%Y-%m-%d').date() - _today).days
            except (ValueError, TypeError):
                dte = 0
            strike = r.get('Strike', 0)
            straddle = r.get('Straddle', float('nan'))
            cost = r.get('Cost_pct', float('nan'))
            c_name = r.get('C_Name', '')
            c_id = str(r.get('C_Id', ''))
            p_name = r.get('P_Name', '')
            p_id = str(r.get('P_Id', ''))
            c_ask = r.get('C_Ask', float('nan'))
            p_ask = r.get('P_Ask', float('nan'))
            c_iv = r.get('C_IV', float('nan'))
            p_iv = r.get('P_IV', float('nan'))
            label = f"{exp} ({dte}d) — Strike {strike:.0f}"
            if pd.notna(straddle):
                label += f" — Straddle {straddle:.1f}"
            if pd.notna(cost):
                label += f" ({cost:.1f}%)"
            expiry_choices.append({
                'label': label, 'expiry': exp, 'dte': dte, 'strike': strike,
                'c_name': c_name, 'c_id': c_id, 'p_name': p_name, 'p_id': p_id,
                'c_ask': c_ask, 'p_ask': p_ask, 'c_iv': c_iv, 'p_iv': p_iv,
                'straddle': straddle, 'cost_pct': cost,
            })

        # ── Dialog ──
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Open Straddle — {ticker}")
        dlg.setMinimumWidth(500)
        dlg.setStyleSheet(f"""
            QDialog {{ background: {COLORS['bg_dark']}; color: {COLORS['text_primary']}; }}
            QLabel {{ color: {COLORS['text_primary']}; font-size: 12px; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {COLORS['bg_card']}; color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']}; border-radius: 4px;
                padding: 6px; font-size: 12px;
            }}
        """)

        form = QVBoxLayout(dlg)
        form.setSpacing(12)

        # Ticker + spot
        form.addWidget(QLabel(f"<span style='color:{COLORS['accent']};font-size:14px;font-weight:700'>"
                              f"{ticker}</span> &nbsp; Spot: {spot:.1f}"))

        # Expiry combo
        form.addWidget(QLabel("Expiry / Strike:"))
        expiry_combo = QComboBox()
        for ec in expiry_choices:
            expiry_combo.addItem(ec['label'])
        # Pre-select 60-180d if available
        for idx, ec in enumerate(expiry_choices):
            if 60 <= ec['dte'] <= 180:
                expiry_combo.setCurrentIndex(idx)
                break
        form.addWidget(expiry_combo)

        # Call entry
        call_frame = QFrame()
        call_layout = QHBoxLayout(call_frame)
        call_layout.setContentsMargins(0, 0, 0, 0)
        call_layout.addWidget(QLabel("Call entry price:"))
        call_price_spin = QDoubleSpinBox()
        call_price_spin.setRange(0, 100000)
        call_price_spin.setDecimals(2)
        call_price_spin.setSingleStep(0.1)
        call_layout.addWidget(call_price_spin)
        call_layout.addWidget(QLabel("Qty:"))
        call_qty_spin = QSpinBox()
        call_qty_spin.setRange(1, 10000)
        call_qty_spin.setValue(10)
        call_layout.addWidget(call_qty_spin)
        form.addWidget(call_frame)

        # Put entry
        put_frame = QFrame()
        put_layout = QHBoxLayout(put_frame)
        put_layout.setContentsMargins(0, 0, 0, 0)
        put_layout.addWidget(QLabel("Put entry price:"))
        put_price_spin = QDoubleSpinBox()
        put_price_spin.setRange(0, 100000)
        put_price_spin.setDecimals(2)
        put_price_spin.setSingleStep(0.1)
        put_layout.addWidget(put_price_spin)
        put_layout.addWidget(QLabel("Qty:"))
        put_qty_spin = QSpinBox()
        put_qty_spin.setRange(1, 10000)
        put_qty_spin.setValue(10)
        put_layout.addWidget(put_qty_spin)
        form.addWidget(put_frame)

        # Parity (multiplikator)
        parity_frame = QFrame()
        parity_layout = QHBoxLayout(parity_frame)
        parity_layout.setContentsMargins(0, 0, 0, 0)
        parity_layout.addWidget(QLabel("Paritet (multiplikator):"))
        parity_spin = QSpinBox()
        parity_spin.setRange(1, 10000)
        parity_spin.setValue(100)  # Default för svenska aktieoptioner
        parity_spin.setToolTip("Antal underliggande per kontrakt (vanligtvis 100 för aktieoptioner)")
        parity_layout.addWidget(parity_spin)
        parity_layout.addStretch()
        form.addWidget(parity_frame)

        # Info label (updates with selected expiry)
        info_label = QLabel("")
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color:{COLORS['text_muted']};font-size:11px;padding:4px;")
        form.addWidget(info_label)

        def _on_expiry_changed(idx):
            if 0 <= idx < len(expiry_choices):
                ec = expiry_choices[idx]
                if pd.notna(ec['c_ask']):
                    call_price_spin.setValue(ec['c_ask'])
                if pd.notna(ec['p_ask']):
                    put_price_spin.setValue(ec['p_ask'])
                parts = [f"Call: {ec['c_name']}" if ec['c_name'] else "",
                         f"Put: {ec['p_name']}" if ec['p_name'] else ""]
                if pd.notna(ec['c_iv']):
                    parts.append(f"Call IV: {ec['c_iv']:.1f}%")
                if pd.notna(ec['p_iv']):
                    parts.append(f"Put IV: {ec['p_iv']:.1f}%")
                info_label.setText(" · ".join(p for p in parts if p))

        expiry_combo.currentIndexChanged.connect(_on_expiry_changed)
        _on_expiry_changed(expiry_combo.currentIndex())

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(f"background:{COLORS['bg_elevated']};color:{COLORS['text_primary']};"
                                 f"border:1px solid {COLORS['border_subtle']};border-radius:4px;padding:8px 20px;")
        cancel_btn.clicked.connect(dlg.reject)
        btn_layout.addWidget(cancel_btn)

        open_btn = QPushButton("Open Position")
        open_btn.setStyleSheet(f"background:{COLORS['positive']};color:{COLORS['bg_darkest']};"
                               f"border:none;border-radius:4px;padding:8px 20px;font-weight:700;")
        open_btn.clicked.connect(dlg.accept)
        btn_layout.addWidget(open_btn)
        form.addLayout(btn_layout)

        if dlg.exec() == QDialog.Accepted:
            idx = expiry_combo.currentIndex()
            ec = expiry_choices[idx]
            self._create_straddle_position(
                ticker=ticker, spot=spot, expiry=ec['expiry'], dte=ec['dte'],
                strike=ec['strike'],
                c_name=ec['c_name'], c_id=ec['c_id'],
                p_name=ec['p_name'], p_id=ec['p_id'],
                c_entry=call_price_spin.value(), c_qty=call_qty_spin.value(),
                p_entry=put_price_spin.value(), p_qty=put_qty_spin.value(),
                c_iv=ec['c_iv'], p_iv=ec['p_iv'],
                cost_pct=ec['cost_pct'],
                parity=parity_spin.value(),
            )

    def _create_straddle_position(self, ticker, spot, expiry, dte, strike,
                                   c_name, c_id, p_name, p_id,
                                   c_entry, c_qty, p_entry, p_qty,
                                   c_iv, p_iv, cost_pct, parity=100):
        """Create a straddle position and add to portfolio."""
        # HV snapshot
        hv_20d = None
        if self._squeeze_result and ticker in self._squeeze_result.ticker_results:
            tr = self._squeeze_result.ticker_results[ticker]
            if hasattr(tr, 'price') and len(tr.price) > 20:
                returns = tr.price.pct_change().dropna()
                hv_20d = float(returns.iloc[-20:].std() * (252 ** 0.5) * 100)

        avg_iv = None
        if pd.notna(c_iv) and pd.notna(p_iv):
            avg_iv = (c_iv + p_iv) / 2

        position = {
            'position_type': 'straddle',
            'pair': ticker,
            'direction': 'STRADDLE',
            'entry_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'status': 'OPEN',
            'expiry': expiry,
            'dte_at_entry': dte,
            'strike': strike,
            'entry_spot': spot,
            'entry_iv': avg_iv,
            'entry_hv': hv_20d,
            'entry_cost_pct': cost_pct if pd.notna(cost_pct) else None,
            # Call leg → Y columns
            'mini_y_name': c_name,
            'mini_y_isin': c_id,
            'mf_entry_price_y': c_entry,
            'mf_qty_y': c_qty,
            'mf_current_price_y': c_entry,  # Initial = entry
            # Put leg → X columns
            'mini_x_name': p_name,
            'mini_x_isin': p_id,
            'mf_entry_price_x': p_entry,
            'mf_qty_x': p_qty,
            'mf_current_price_x': p_entry,  # Initial = entry
            # Parity (multiplikator) — antal underliggande per kontrakt
            'parity': parity,
            # Pairs compat fields
            'entry_z': 0, 'current_z': 0, 'hedge_ratio': 1.0,
            'notional': round((c_entry * c_qty + p_entry * p_qty) * parity, 2),
        }

        self.portfolio.append(position)
        self.update_portfolio_display()
        self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
        self._save_and_sync_portfolio()

        self.statusBar().showMessage(
            f"Opened straddle: {ticker} {expiry} strike {strike} | "
            f"Call {c_entry:.2f} x{c_qty} + Put {p_entry:.2f} x{p_qty}")

    # ════════════════════════════════════════════════════════════════
    # PAIRS POSITION (from Signals tab)
    # ════════════════════════════════════════════════════════════════

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
                'window_size': None,  # Multi-window removed (single 2y period)
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
            is_straddle = pos.get('position_type') == 'straddle'

            # POSITION (col 0)
            pos_text = pos['pair']
            if is_straddle:
                expiry = pos.get('expiry', '')
                strike = pos.get('strike', 0)
                pos_text = f"{pos['pair']} {expiry} K{strike:.0f}"
            pair_item = QTableWidgetItem(pos_text)
            pair_item.setForeground(_QCOLOR_TEXT)
            pair_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 0, pair_item)

            # TYPE (col 1)
            dir_text = pos['direction']
            dir_item = QTableWidgetItem(dir_text)
            dir_color = QColor(COLORS['accent']) if is_straddle else _QCOLOR_TEXT
            dir_item.setForeground(dir_color)
            dir_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 1, dir_item)

            # SIGNAL (col 2) — Z-score for pairs, DTE for straddles
            if is_straddle:
                from datetime import date as _date_p, datetime as _dt_p
                try:
                    exp_date = _dt_p.strptime(pos.get('expiry', ''), '%Y-%m-%d').date()
                    dte_now = (exp_date - _date_p.today()).days
                    dte_text = f"{dte_now}d"
                    dte_color = QColor(COLORS['warning']) if dte_now < 30 else _QCOLOR_TEXT
                except (ValueError, TypeError):
                    dte_text = "-"
                    dte_color = _QCOLOR_MUTED
                z_item = QTableWidgetItem(dte_text)
                z_item.setForeground(dte_color)
            else:
                current_z = pos.get('current_z', pos.get('entry_z', 0))
                z_item = QTableWidgetItem(f"{current_z:.2f}")
                z_item.setForeground(_QCOLOR_TEXT)
            z_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 2, z_item)

            # STATUS (col 3)
            direction = pos['direction']
            if is_straddle:
                # Straddle status: check P/L and DTE
                c_entry = pos.get('mf_entry_price_y', 0) or 0
                p_entry = pos.get('mf_entry_price_x', 0) or 0
                c_cur = pos.get('mf_current_price_y', 0) or 0
                p_cur = pos.get('mf_current_price_x', 0) or 0
                c_qty = pos.get('mf_qty_y', 0) or 0
                p_qty = pos.get('mf_qty_x', 0) or 0
                total_entry = c_entry * c_qty + p_entry * p_qty
                total_cur = c_cur * c_qty + p_cur * p_qty
                if total_entry > 0 and total_cur > 0:
                    pnl_pct = (total_cur / total_entry - 1) * 100
                    if pnl_pct > 20:
                        status_text = 'TAKE PROFIT'
                        status_qcolor = QColor(COLORS['positive'])
                    elif pnl_pct < -30:
                        status_text = 'STOP LOSS'
                        status_qcolor = _QCOLOR_NEGATIVE
                    else:
                        status_text = 'OPEN'
                        status_qcolor = _QCOLOR_TEXT
                else:
                    status_text = 'OPEN'
                    status_qcolor = _QCOLOR_TEXT
            else:
                current_z = pos.get('current_z', pos.get('entry_z', 0))
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

            # LEG 1 / Y / CALL (col 4)
            mini_y_isin = pos.get('mini_y_isin', '')
            mini_y_name = pos.get('mini_y_name', '')
            leg1_prefix = "CALL: " if is_straddle else ""
            if mini_y_name:
                mini_y_item = QTableWidgetItem(f"{leg1_prefix}{mini_y_name}")
                tip = f"{'Call option' if is_straddle else mini_y_name}\nID: {mini_y_isin}"
                mini_y_item.setToolTip(tip)
                mini_y_item.setForeground(_QCOLOR_TEXT)
            elif mini_y_isin:
                mini_y_item = QTableWidgetItem(f"{leg1_prefix}{mini_y_isin[:12]}")
                mini_y_item.setToolTip(f"ID: {mini_y_isin}")
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
            parity = pos.get('parity', 1)  # 1 for pairs, 100 for options

            if entry_price_y > 0 and qty_y > 0:
                pnl_y = (current_price_y - entry_price_y) * qty_y * parity
                pnl_y_pct = ((current_price_y / entry_price_y) - 1) * 100 if entry_price_y > 0 else 0
                pnl_y_text = f"{pnl_y:+,.0f} ({pnl_y_pct:+.1f}%)"
                pnl_y_qcolor = _QCOLOR_POSITIVE if pnl_y >= 0 else _QCOLOR_NEGATIVE
                total_pnl += pnl_y
                total_invested += entry_price_y * qty_y * parity
            else:
                pnl_y_text = "-"
                pnl_y_qcolor = _QCOLOR_MUTED

            pnl_y_item = QTableWidgetItem(pnl_y_text)
            pnl_y_item.setForeground(pnl_y_qcolor)
            pnl_y_item.setTextAlignment(Qt.AlignCenter)
            self.positions_table.setItem(i, 7, pnl_y_item)

            # LEG 2 / X / PUT (col 8)
            mini_x_isin = pos.get('mini_x_isin', '')
            mini_x_name = pos.get('mini_x_name', '')
            leg2_prefix = "PUT: " if is_straddle else ""
            if mini_x_name:
                mini_x_item = QTableWidgetItem(f"{leg2_prefix}{mini_x_name}")
                tip = f"{'Put option' if is_straddle else mini_x_name}\nID: {mini_x_isin}"
                mini_x_item.setToolTip(tip)
                mini_x_item.setForeground(_QCOLOR_TEXT)
            elif mini_x_isin:
                mini_x_item = QTableWidgetItem(f"{leg2_prefix}{mini_x_isin[:12]}")
                mini_x_item.setToolTip(f"ID: {mini_x_isin}")
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
                pnl_x = (current_price_x - entry_price_x) * qty_x * parity
                pnl_x_pct = ((current_price_x / entry_price_x) - 1) * 100 if entry_price_x > 0 else 0
                pnl_x_text = f"{pnl_x:+,.0f} ({pnl_x_pct:+.1f}%)"
                pnl_x_qcolor = _QCOLOR_POSITIVE if pnl_x >= 0 else _QCOLOR_NEGATIVE
                total_pnl += pnl_x
                total_invested += entry_price_x * qty_x * parity
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

        # Update concentration risk metrics
        self._update_concentration_risk()

    def _update_concentration_risk(self):
        """Update concentration risk metrics in the portfolio tab."""
        try:
            if not hasattr(self, 'eff_bets_label'):
                return

            if self.engine is None or len(self.portfolio) < 2:
                self.eff_bets_label.setText(
                    f"Effective Bets: <span style='color:#e8e8e8; font-weight:600;'>"
                    f"{len(self.portfolio)}</span>")
                self.conc_score_label.setText(
                    "Concentration: <span style='color:#888;'>N/A (need ≥2 positions)</span>")
                self.max_corr_label.setText(
                    "Max Corr Pair: <span style='color:#888;'>-</span>")
                return

            # Temporarily assign positions to the engine for calculation
            open_pairs = {pos['pair']: pos for pos in self.portfolio}
            old_positions = self.engine.positions
            self.engine.positions = open_pairs

            conc = self.engine.calculate_spread_correlations()

            self.engine.positions = old_positions

            eff_bets = conc.get('effective_bets', len(self.portfolio))
            conc_score = conc.get('concentration_score', 0)
            max_pair = conc.get('max_corr_pair', ('', '', 0))

            # Color coding
            conc_color = "#ff1744" if conc_score > 0.5 else ("#ffc107" if conc_score > 0.3 else "#22c55e")

            self.eff_bets_label.setText(
                f"Effective Bets: <span style='color:#e8e8e8; font-weight:600;'>"
                f"{eff_bets:.1f}/{len(self.portfolio)}</span>")
            self.conc_score_label.setText(
                f"Concentration: <span style='color:{conc_color};'>"
                f"{conc_score:.1%}</span>")

            if max_pair[0] and max_pair[1]:
                self.max_corr_label.setText(
                    f"Max Corr: <span style='color:#888;'>"
                    f"{max_pair[0]} / {max_pair[1]} ({max_pair[2]:.2f})</span>")
            else:
                self.max_corr_label.setText(
                    "Max Corr Pair: <span style='color:#888;'>-</span>")
        except Exception as e:
            print(f"Concentration risk update error: {e}")

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
        parity = pos.get('parity', 1)

        if entry_price_y > 0 and qty_y > 0 and current_price_y > 0:
            pnl_y = (current_price_y - entry_price_y) * qty_y * parity
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
            pnl_x = (current_price_x - entry_price_x) * qty_x * parity
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
            parity = pos.get('parity', 1)
            entry_y = pos.get('mf_entry_price_y', 0.0)
            current_y = pos.get('mf_current_price_y', 0.0)
            qty_y = pos.get('mf_qty_y', 0)

            if entry_y > 0 and qty_y > 0 and current_y > 0:
                total_pnl += (current_y - entry_y) * qty_y * parity
                total_invested += entry_y * qty_y * parity

            entry_x = pos.get('mf_entry_price_x', 0.0)
            current_x = pos.get('mf_current_price_x', 0.0)
            qty_x = pos.get('mf_qty_x', 0)

            if entry_x > 0 and qty_x > 0 and current_x > 0:
                total_pnl += (current_x - entry_x) * qty_x * parity
                total_invested += entry_x * qty_x * parity
        
        if total_invested > 0:
            total_pnl_pct = (total_pnl / total_invested) * 100
            pnl_color = "#22c55e" if total_pnl >= 0 else "#ff1744"
            self.unrealized_pnl_label.setText(f"Unrealized: <span style='color:{pnl_color};'>{total_pnl:+,.0f} SEK ({total_pnl_pct:+.2f}%)</span>")
        else:
            self.unrealized_pnl_label.setText("Unrealized: <span style='color:#888;'>+0.00%</span>")
    
    def refresh_mf_prices(self):
        """Refresh current prices for all portfolio instruments (MF + options)."""
        if not self.portfolio:
            self.statusBar().showMessage("No positions to update")
            return

        # Separate straddle vs pairs positions
        mf_isins = []
        opt_ids = []
        self._mf_isin_to_positions = {}
        self._opt_id_to_positions = {}

        for idx, pos in enumerate(self.portfolio):
            is_straddle = pos.get('position_type') == 'straddle'
            mini_y_id = pos.get('mini_y_isin')
            mini_x_id = pos.get('mini_x_isin')

            if is_straddle:
                # Straddle: fetch from Avanza (orderbookId)
                if mini_y_id:
                    if mini_y_id not in self._opt_id_to_positions:
                        self._opt_id_to_positions[mini_y_id] = []
                        opt_ids.append(mini_y_id)
                    self._opt_id_to_positions[mini_y_id].append((idx, 'y'))
                if mini_x_id:
                    if mini_x_id not in self._opt_id_to_positions:
                        self._opt_id_to_positions[mini_x_id] = []
                        opt_ids.append(mini_x_id)
                    self._opt_id_to_positions[mini_x_id].append((idx, 'x'))
            else:
                # Pairs: fetch from Morgan Stanley (ISIN)
                if mini_y_id:
                    if mini_y_id not in self._mf_isin_to_positions:
                        self._mf_isin_to_positions[mini_y_id] = []
                        mf_isins.append(mini_y_id)
                    self._mf_isin_to_positions[mini_y_id].append((idx, 'y'))
                if mini_x_id:
                    if mini_x_id not in self._mf_isin_to_positions:
                        self._mf_isin_to_positions[mini_x_id] = []
                        mf_isins.append(mini_x_id)
                    self._mf_isin_to_positions[mini_x_id].append((idx, 'x'))

        launched = 0

        # Launch MF price fetch (pairs positions)
        if mf_isins and MF_PRICE_SCRAPING_AVAILABLE:
            self.statusBar().showMessage(f"Fetching {len(mf_isins)} MF + {len(opt_ids)} option prices...")
            self._price_thread = QThread()
            self._price_worker = PriceFetchWorker(mf_isins)
            self._price_worker.moveToThread(self._price_thread)
            self._price_thread.started.connect(self._price_worker.run)
            self._price_worker.finished.connect(self._price_thread.quit)
            self._price_worker.finished.connect(self._price_worker.deleteLater)
            self._price_thread.finished.connect(self._price_thread.deleteLater)
            self._price_worker.result.connect(self._on_mf_prices_received)
            self._price_worker.error.connect(self._on_mf_prices_error)
            self._price_worker.status_message.connect(self.statusBar().showMessage)
            self._price_thread.start()
            launched += 1

        # Launch option price fetch (straddle positions)
        if opt_ids:
            self.statusBar().showMessage(f"Fetching {len(opt_ids)} option prices...")
            self._opt_price_thread = QThread()
            self._opt_price_worker = OptionPriceFetchWorker(opt_ids)
            self._opt_price_worker.moveToThread(self._opt_price_thread)
            self._opt_price_thread.started.connect(self._opt_price_worker.run)
            self._opt_price_worker.finished.connect(self._opt_price_thread.quit)
            self._opt_price_worker.finished.connect(self._opt_price_worker.deleteLater)
            self._opt_price_thread.finished.connect(self._opt_price_thread.deleteLater)
            self._opt_price_worker.result.connect(self._on_option_prices_received)
            self._opt_price_worker.error.connect(self._on_mf_prices_error)
            self._opt_price_worker.status_message.connect(self.statusBar().showMessage)
            self._opt_price_thread.start()
            launched += 1

        if launched == 0:
            self.statusBar().showMessage("No instruments to update")
    
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
    
    def _on_option_prices_received(self, quotes: dict):
        """Handle fetched option prices from Avanza — update straddle positions."""
        try:
            updated_count = 0
            for ob_id, price_data in quotes.items():
                if ob_id in self._opt_id_to_positions:
                    # Använd bid (köpkurs) = vad vi faktiskt får vid försäljning.
                    # Avanza visar "last" men bid ger sannare exit-värdering.
                    price = price_data.get('buy') or price_data.get('last')
                    if price is not None:
                        for pos_idx, leg in self._opt_id_to_positions[ob_id]:
                            if pos_idx < len(self.portfolio):
                                self.portfolio[pos_idx][f'mf_current_price_{leg}'] = price
                                updated_count += 1

            self._save_and_sync_portfolio()
            self.update_portfolio_display()
            self.statusBar().showMessage(f"Updated {updated_count} option prices from {len(quotes)} instruments")
        except Exception as e:
            print(f"Error processing option prices: {e}")
            self.statusBar().showMessage(f"Option price error: {e}")

    def _on_mf_prices_error(self, error: str):
        """Handle price fetch error."""
        print(f"Price fetch error: {error}")
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
            parity = pos.get('parity', 1)

            entry_y = pos.get('mf_entry_price_y')
            current_y = pos.get('mf_current_price_y')
            qty_y = pos.get('mf_qty_y', 0)
            if entry_y and current_y and qty_y:
                pnl_y = (current_y - entry_y) * qty_y * parity

            entry_x = pos.get('mf_entry_price_x')
            current_x = pos.get('mf_current_price_x')
            qty_x = pos.get('mf_qty_x', 0)
            if entry_x and current_x and qty_x:
                pnl_x = (current_x - entry_x) * qty_x * parity

            realized_pnl_sek = pnl_y + pnl_x

            # Capital = actual invested amount (entry_price * qty * parity per leg)
            total_capital = 0
            if entry_y and qty_y:
                total_capital += entry_y * qty_y * parity
            if entry_x and qty_x:
                total_capital += entry_x * qty_x * parity
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
            
            is_straddle = pos.get('position_type') == 'straddle'
            status = pos.get('status', 'MANUAL CLOSE')

            # Confirmation dialog
            if is_straddle:
                msg = f"Close straddle for {pos['pair']}?\n\n"
                msg += f"Expiry: {pos.get('expiry', '?')}\n"
                msg += f"Strike: {pos.get('strike', 0):.0f}\n"
                msg += f"Call: {pos.get('mini_y_name', '')} entry {entry_y or 0:.2f} → {current_y or 0:.2f}\n"
                msg += f"Put: {pos.get('mini_x_name', '')} entry {entry_x or 0:.2f} → {current_x or 0:.2f}\n"
                msg += f"Realized P/L: {realized_pnl_sek:+.0f} SEK ({realized_pnl_pct:+.1f}%)\n"
            else:
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
                    'position_type': pos.get('position_type', 'pairs'),
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
                    # Preserve leg details
                    'mini_y_name': pos.get('mini_y_name', ''),
                    'mini_x_name': pos.get('mini_x_name', ''),
                    'mf_entry_price_y': entry_y,
                    'mf_entry_price_x': entry_x,
                    'mf_close_price_y': current_y,
                    'mf_close_price_x': current_x,
                    'mf_qty_y': qty_y,
                    'mf_qty_x': qty_x,
                }
                # Straddle-specific fields
                if is_straddle:
                    trade_record['expiry'] = pos.get('expiry', '')
                    trade_record['strike'] = pos.get('strike', 0)
                    trade_record['entry_iv'] = pos.get('entry_iv')
                    trade_record['entry_hv'] = pos.get('entry_hv')
                self.trade_history.append(trade_record)
                
                self.update_portfolio_display()
                self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
                self._save_and_sync_portfolio()
                
                if is_straddle:
                    self.statusBar().showMessage(
                        f"Closed straddle {pos['pair']} {pos.get('expiry', '')} - {result} "
                        f"(P/L: {realized_pnl_sek:+.0f} SEK, {realized_pnl_pct:+.1f}%)")
                else:
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
    
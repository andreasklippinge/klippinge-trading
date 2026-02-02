"""
Att göra:

"""

"""
Pairs Trading Terminal - PyQt5 Version (Streamlit Layout)
==========================================================

Matches the Streamlit dashboard layout exactly.

ASYNC FIX (2025-01-19):
    Fixed GUI freeze during scheduled scans by moving blocking HTTP operations
    to worker threads:
    
    1. DiscordWorker - Async Discord webhook notifications
    2. PriceFetchWorker - Async mini futures price fetching
    
    Modified methods:
    - send_discord_notification() - Now runs in background thread
    - refresh_mf_prices() - Now runs in background thread  
    - _take_daily_portfolio_snapshot() - No longer blocks on price fetch
    - check_hmm_schedule() - Better async coordination
    - send_scan_results_to_discord() - Now uses LIVE portfolio data (not stale snapshots)
    
    New methods:
    - _on_mf_prices_received() - Callback for async price fetch
    - _on_mf_prices_error() - Error handler for price fetch
    - _run_scheduled_scan_after_prices() - Scheduled scan coordinator
    - _calculate_live_portfolio_for_discord() - Live P&L calculation for Discord
    
    Bug fixes:
    - Removed duplicate portfolio fields in Discord messages
    - Discord now shows LIVE P&L matching dashboard (not snapshot data)

Requirements:
    pip install pyqtgraph numpy pandas yfinance PyQtWebEngine beautifulsoup4 requests

Run with:
    python main_window.py
"""

import sys
import os
import re
import socket
import time
from datetime import datetime
from typing import Optional, Dict, List
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QLineEdit, QTextEdit, QSplitter, QTabWidget,
    QHeaderView, QAbstractItemView, QSpinBox, QDoubleSpinBox,
    QProgressBar, QStatusBar, QMenuBar, QMenu,
    QFrame, QScrollArea, QGridLayout, QSizePolicy, QMessageBox,
    QFileDialog, QCheckBox, QAction, QStackedWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal, QThread, QObject, pyqtSlot as Slot, QSize, QUrl, QPointF
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

# Import trading engine
from pairs_engine import PairsTradingEngine, OUProcess, load_tickers_from_csv
from regime_hmm import (
    RegimeDetector, AdvancedRegimeDetector, REGIMES, PYMC_AVAILABLE,
    SimpleGaussianHMM, HiddenSemiMarkovModel, ParticleFilterHMM,
    TimeVaryingTransitionHMM, UncertaintyQuantifier, BayesianHMM
)

# ── Application configuration (portable paths for distribution) ──
from app_config import (
    Paths, APP_VERSION, APP_NAME,
    get_discord_webhook_url, save_discord_webhook_url,
    initialize_user_data, resource_path, get_user_data_dir,
    find_ticker_csv, find_matched_tickers_csv, setup_logging,
    print_config, _is_frozen
)

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
        from PyQt5.QtWebEngineWidgets import QWebEngineView
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
                except:
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
                        except:
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
                self.plot.removeItem(self.vLine)
                self.plot.removeItem(self.hLine)
                self.plot.removeItem(self.label)
            except:
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

# ============================================================================
# TYPOGRAPHY - Consistent Font Sizes Throughout Dashboard
# ============================================================================

TYPOGRAPHY = {
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
    background-color: {COLORS['bg_elevated']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['accent_dark']};
    padding: 8px;
    border-radius: 4px;
    font-size: 11px;
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

def save_portfolio(portfolio: list, filepath: str = PORTFOLIO_FILE) -> bool:
    """
    Save portfolio positions to JSON file with file locking.
    Safe for Google Drive sync between multiple computers.
    
    Args:
        portfolio: List of position dictionaries
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    if not _acquire_lock(filepath):
        print("[Portfolio] Could not save - file is locked")
        return False
    
    try:
        data = {
            "positions": portfolio,
            "last_updated": datetime.now().isoformat(),
            "last_saved_by": socket.gethostname(),
            "version": "1.1"
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
        except:
            return []
        
    except Exception as e:
        print(f"[Portfolio] Error loading: {e}")
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
                except:
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
        
        print(f"[Engine Cache] Saved {n_tickers} tickers, {n_pairs} viable pairs ({file_size:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"[Engine Cache] Error saving: {e}")
        import traceback
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
        import traceback
        traceback.print_exc()
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
        import time
        if time.time() - self._timestamps.get(key, 0) > self._ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value):
        """Set value with current timestamp."""
        import time
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
        paths_to_try = [
            csv_path,
            # Check underliggande_matchade_tickers.csv FIRST (has MS_Asset)
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
        import traceback
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
                link = td.find("a")
                if link:
                    if link.has_attr("href"):
                        tds.append(link["href"])
                    elif link.has_attr("onclick"):
                        m = re.search(r"'(.*?)'", link["onclick"])
                        if m:
                            tds.append("https://etp.morganstanley.com" + m.group(1))
                        else:
                            tds.append(link.get_text(strip=True))
                    else:
                        tds.append(link.get_text(strip=True))
                else:
                    tds.append(td.get_text(strip=True))
            
            if len(tds) == len(headers):
                rows.append(dict(zip(headers, tds)))
        
        return rows
    except Exception as e:
        return []


def fetch_minifutures_for_asset(ms_asset: str, session=None) -> pd.DataFrame:
    """
    Fetch mini futures directly for a specific underlying asset using MS_Asset code.
    
    This is 10x faster than scraping all pages because it queries only the specific
    underlying instead of all ~3000 products.
    
    Args:
        ms_asset: MS asset code like 'ATT_SS_Equity', 'AAPL_US_Equity', etc.
        session: Optional requests session (will create one if not provided)
        
    Returns:
        DataFrame with mini futures for this specific underlying
    """
    if not SCRAPING_AVAILABLE or not ms_asset:
        return pd.DataFrame()
    
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    BASE_URL = "https://etp.morganstanley.com/se/sv/produkter"
    params = {
        "f_pc": "LeverageProducts",
        "f_pt": "MiniFuture",
        "f_asset": ms_asset,  # Direct asset filter!
        "p_s": 100,
        "p_n": 1
    }
    
    try:
        r = session.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()
        
        trs = table.find_all("tr")
        if len(trs) < 2:
            return pd.DataFrame()
            
        headers = [th.get_text(strip=True) for th in trs[0].find_all("th")]
        
        rows = []
        for tr in trs[1:]:
            tds = []
            for td in tr.find_all("td"):
                link = td.find("a")
                if link:
                    if link.has_attr("href"):
                        tds.append(link["href"])
                    elif link.has_attr("onclick"):
                        m = re.search(r"'(.*?)'", link["onclick"])
                        if m:
                            tds.append("https://etp.morganstanley.com" + m.group(1))
                        else:
                            tds.append(link.get_text(strip=True))
                    else:
                        tds.append(link.get_text(strip=True))
                else:
                    tds.append(td.get_text(strip=True))
            
            if len(tds) == len(headers):
                rows.append(dict(zip(headers, tds)))
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
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
        
        return df
        
    except Exception as e:
        print(f"Error fetching minifutures for {ms_asset}: {e}")
        return pd.DataFrame()


def fetch_certificates_for_asset(ms_asset: str, session=None) -> pd.DataFrame:
    """
    Fetch Bull/Bear certificates (ConstantLeverage) for a specific underlying asset.
    
    Similar to fetch_minifutures_for_asset but for constant leverage products.
    URL: f_pt=ConstantLeverage instead of f_pt=MiniFuture
    
    Args:
        ms_asset: MS asset code like 'ASML_US_Equity', 'ATT_SS_Equity', etc.
        session: Optional requests session (will create one if not provided)
        
    Returns:
        DataFrame with Bull/Bear certificates for this specific underlying
    """
    if not SCRAPING_AVAILABLE or not ms_asset:
        return pd.DataFrame()
    
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    BASE_URL = "https://etp.morganstanley.com/se/sv/produkter"
    params = {
        "f_pc": "LeverageProducts",
        "f_pt": "ConstantLeverage",  # Bull/Bear certificates!
        "f_asset": ms_asset,
        "p_s": 100,
        "p_n": 1
    }
    
    try:
        r = session.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if table is None:
            return pd.DataFrame()
        
        trs = table.find_all("tr")
        if len(trs) < 2:
            return pd.DataFrame()
            
        headers = [th.get_text(strip=True) for th in trs[0].find_all("th")]
        
        rows = []
        for tr in trs[1:]:
            tds = []
            for td in tr.find_all("td"):
                link = td.find("a")
                if link:
                    if link.has_attr("href"):
                        tds.append(link["href"])
                    elif link.has_attr("onclick"):
                        m = re.search(r"'(.*?)'", link["onclick"])
                        if m:
                            tds.append("https://etp.morganstanley.com" + m.group(1))
                        else:
                            tds.append(link.get_text(strip=True))
                    else:
                        tds.append(link.get_text(strip=True))
                else:
                    tds.append(td.get_text(strip=True))
            
            if len(tds) == len(headers):
                rows.append(dict(zip(headers, tds)))
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Parse daily leverage ("Daglig hävstång")
        def parse_leverage(x):
            """Parse leverage like '2x', '3x', '-2x', '-3x' etc."""
            if not isinstance(x, str):
                return None
            # Remove 'x' and parse
            clean = x.replace('x', '').replace('X', '').strip()
            try:
                return float(clean)
            except ValueError:
                return None
        
        # Try different column names for leverage
        leverage_cols = ['Daglig hävstång', 'Hävstång', 'Leverage']
        for col in leverage_cols:
            if col in df.columns:
                df['DailyLeverage'] = df[col].apply(parse_leverage)
                break
        
        return df
        
    except Exception as e:
        print(f"Error fetching certificates for {ms_asset}: {e}")
        return pd.DataFrame()


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
    
    return {
        'name': best.get('Namn', 'N/A'),
        'underlying': best.get('Underliggande tillgång', ms_name),
        'direction': direction,
        'financing_level': None,  # Certificates don't have financing level
        'leverage': best['LeverageAbs'],  # Use absolute value for display
        'daily_leverage': best['DailyLeverage'],  # Keep original with sign
        'spot_price': spot_price,
        'isin': isin or 'N/A',
        'avanza_link': avanza_link,
        'ticker': ticker,
        'product_type': 'Certificate'  # Mark as certificate, not mini future
    }


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


def get_spot_price(ticker: str) -> float:
    """Get current spot price for a ticker."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except:
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
                    minifutures_df[underlying_col].str.contains(ms_name, case=False, na=False) &
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
    hmm_loaded = Signal(object)
    status_message = Signal(str)
    
    def __init__(self, portfolio_file: str, engine_cache_file: str, hmm_cache_path: str):
        super().__init__()
        self.portfolio_file = portfolio_file
        self.engine_cache_file = engine_cache_file
        self.hmm_cache_path = hmm_cache_path
    
    @Slot()
    def run(self):
        try:
            # 1. Load portfolio (fast - small JSON)
            self.status_message.emit("Loading portfolio...")
            if os.path.exists(self.portfolio_file):
                positions = load_portfolio(self.portfolio_file)
                if positions:
                    self.portfolio_loaded.emit(positions)
            
            # 2. Load HMM cache (medium - pickle)
            self.status_message.emit("Loading regime model...")
            if os.path.exists(self.hmm_cache_path):
                try:
                    detector = RegimeDetector(model_type='hsmm', lookback_years=30)
                    if detector.load_model(self.hmm_cache_path):
                        self.hmm_loaded.emit(detector)
                except Exception as e:
                    pass
            
            # 3. Load engine cache (slowest - large pickle with price data)
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
            engine.fetch_data(self.tickers, self.lookback)
            
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


class HMMWorker(QObject):
    """Worker thread for fitting Advanced HMM model."""
    finished = Signal()
    progress = Signal(int, str)
    error = Signal(str)
    result = Signal(object)
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]  # Exponential backoff in seconds
    
    @Slot()
    def run(self):
        import time
        last_error = None
        
        # Try fitting with retries
        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    delay = self.RETRY_DELAYS[min(attempt - 1, len(self.RETRY_DELAYS) - 1)]
                    self.progress.emit(5, f"Retry {attempt}/{self.MAX_RETRIES} after {delay}s wait...")
                    time.sleep(delay)
                
                self.progress.emit(5, "Initializing HSMM regime detector...")
                
                # Create detector with HSMM and all enhancements
                detector = RegimeDetector(
                    model_type='hsmm',            # Hidden Semi-Markov Model
                    lookback_years=30,
                    use_particle_filter=True,     # Online updates
                    use_tv_transitions=True       # Time-varying transitions
                )
                
                def progress_callback(pct, msg):
                    self.progress.emit(pct, msg)
                
                # Fit the model (handles all components internally)
                detector.fit(progress_callback=progress_callback)
                
                # Model auto-saves to cache, but we also save explicitly
                self.progress.emit(95, "Saving model cache...")
                detector.save_model()
                
                self.result.emit(detector)
                return  # Success - exit early
                
            except Exception as e:
                import traceback
                last_error = f"{str(e)}\n{traceback.format_exc()}"
                
                # Check if it's a data fetch error (worth retrying)
                if "Only got 0 tickers" in str(e) or "Need at least" in str(e):
                    continue  # Retry on data fetch failures
                else:
                    break  # Don't retry on other errors
        
        # All retries failed - try loading cached model as fallback
        self.progress.emit(90, "Data fetch failed, loading cached model...")
        
        try:
            detector = RegimeDetector(model_type='hsmm', lookback_years=30)
            if detector.load_model():
                self.progress.emit(95, "Using cached model (data fetch failed)")
                self.result.emit(detector)
                return
        except Exception as cache_error:
            pass
        
        # Both fit and cache failed - emit error
        self.error.emit(f"Data fetch failed after {self.MAX_RETRIES} retries, no cached model available.\n{last_error}")
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
    """Worker thread for fetching market watch data (async yfinance to prevent GUI freeze).
    
    OPTIMERING: Flyttar tung yfinance.download från main thread till bakgrund.
    """
    finished = Signal()
    result = Signal(object, dict)  # (DataFrame, all_instruments dict)
    error = Signal(str)
    status_message = Signal(str)
    
    def __init__(self, all_instruments: dict):
        super().__init__()
        # VIKTIGT: Kopiera dict för thread safety
        self.all_instruments = dict(all_instruments)
        self.tickers = list(self.all_instruments.keys())
    
    @Slot()
    def run(self):
        try:
            import yfinance as yf
            self.status_message.emit(f"Fetching market data for {len(self.tickers)} instruments...")
            
            # Download 5 days of 30-min data
            # FIX: threads=False förhindrar "Cannot join tz-naive with tz-aware DatetimeIndex" fel
            # som uppstår när yfinance multi-threading mixar timezone-data för många tickers
            try:
                data = yf.download(self.tickers, period='5d', interval="30m", progress=False, threads=False, ignore_tz=True)
            except (TypeError, RuntimeError) as e:
                # Fallback: försök igen med lägre batch-storlek om det fortfarande misslyckas
                print(f"[MarketDataWorker] First attempt failed ({e}), retrying with smaller batches...")
                # Dela upp i batches om 50 tickers
                all_data = []
                batch_size = 50
                for i in range(0, len(self.tickers), batch_size):
                    batch = self.tickers[i:i+batch_size]
                    try:
                        batch_data = yf.download(batch, period='5d', interval="30m", progress=False, threads=False, ignore_tz=True)
                        if not batch_data.empty:
                            all_data.append(batch_data)
                    except Exception as batch_e:
                        print(f"[MarketDataWorker] Batch {i//batch_size + 1} failed: {batch_e}")
                        continue
                
                if all_data:
                    data = pd.concat(all_data, axis=1)
                else:
                    data = pd.DataFrame()
            
            
            if data.empty:
                self.error.emit("No market data returned")
                return
            
            # Handle both single and multi-column results
            if 'Close' in data.columns:
                close = data['Close']
            elif isinstance(data.columns, pd.MultiIndex):
                close = data['Close']
            else:
                close = data
            
            # Kopiera dict igen för säker signalering
            self.result.emit(close, dict(self.all_instruments))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class VolatilityDataWorker(QObject):
    """Worker thread for fetching volatility/market data (async yfinance to prevent GUI freeze).
    
    OPTIMERING: Flyttar tung yfinance.download från main thread till bakgrund.
    """
    finished = Signal()
    result = Signal(object)  # DataFrame with close prices
    error = Signal(str)
    status_message = Signal(str)
    
    def __init__(self, tickers: list):
        super().__init__()
        self.tickers = list(tickers)  # Kopiera för säkerhet
    
    @Slot()
    def run(self):
        try:
            import yfinance as yf
            self.status_message.emit(f"Fetching volatility data for {len(self.tickers)} tickers...")
            
            # Use shorter period to speed up
            data = yf.download(self.tickers, period='max', progress=False, threads=False, ignore_tz=True)
            
            if data.empty:
                self.error.emit("No volatility data returned")
                return
            
            # Handle both single and multi-column results
            if 'Close' in data.columns:
                close = data['Close']
            elif isinstance(data.columns, pd.MultiIndex):
                close = data['Close']
            else:
                close = data
            
            self.result.emit(close)
            
        except Exception as e:
            import traceback
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
        import copy
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
                    if pos.get('status', 'OPEN') != 'OPEN':
                        continue
                    try:
                        pair = pos['pair']
                        ou, spread, current_z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                        pos['previous_z'] = pos.get('current_z', pos['entry_z'])
                        pos['current_z'] = current_z
                        
                        # Check TP/SL conditions
                        tp_z = pos.get('exit_z', 0.0)
                        sl_z = pos.get('stop_z', 3.0)
                        
                        if pos['direction'] == 'LONG':
                            if current_z >= tp_z:
                                pos['status'] = 'TP HIT'
                            elif current_z <= -abs(sl_z):
                                pos['status'] = 'SL HIT'
                        else:
                            if current_z <= tp_z:
                                pos['status'] = 'TP HIT'
                            elif current_z >= abs(sl_z):
                                pos['status'] = 'SL HIT'
                        
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
    """Individual market clock widget (DST-aware)."""

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

        self.setFixedSize(95, 52)

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
    """Professional header bar with logo, clocks, and market status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Fix #4: Cache clock references instead of using findChildren
        self._market_clocks = []

        self.setFixedHeight(62)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 6, 20, 6)
        layout.setSpacing(20)

        # === LEFT: LOGO + TITLE ===
        left_section = QHBoxLayout()
        left_section.setSpacing(12)

        logo = QLabel("◈")
        logo.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:32px; font-weight:bold;"
        )
        left_section.addWidget(logo)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)

        title = QLabel("KLIPPINGE INVESTMENT")
        title.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:20px; font-weight:700; letter-spacing:1px;"
        )
        title_layout.addWidget(title)

        subtitle = QLabel("TRADING TERMINAL")
        subtitle.setStyleSheet(
            f"color:{COLORS['accent']}; font-size:15px; font-weight:500; letter-spacing:2px;"
        )
        title_layout.addWidget(subtitle)

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
    """Metric display card with gradient background."""
    
    def __init__(self, label: str, value: str = "-", parent=None):
        super().__init__(parent)
        # self.setMinimumHeight(90)
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
    
    def set_value(self, value: str, color: str = None):
        if color is None:
            color = COLORS['text_primary']
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


class VolatilityCard(QFrame):
    """Volatility indicator card with gradient background."""
    
    def __init__(self, ticker: str, name: str, description: str = "", parent=None):
        super().__init__(parent)
        self.ticker = ticker
        self.name = name
        self.setMinimumHeight(125)
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(5)
        
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
        
        # Row 4: Description
        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; font-style: italic; background: transparent; border: none;")
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)
        
        layout.addStretch()
    
    def update_data(self, value: float, change_pct: float, percentile: float, 
                    median: float, mode: float, description: str):
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

class SparklineWidget(QWidget):
    """Mini sparkline chart widget for market watch."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.prices = []
        self.color = QColor(COLORS['positive'])
        self.setFixedSize(50, 18)
    
    def set_data(self, prices: list, color: str = None):
        """Set price data and color for sparkline."""
        if color is None:
            color = COLORS['positive']
        self.prices = prices if prices else []
        self.color = QColor(color)
        self.update()
    
    def paintEvent(self, event):
        """Draw the sparkline."""
        if len(self.prices) < 2:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate bounds
        min_p = min(self.prices)
        max_p = max(self.prices)
        price_range = max_p - min_p if max_p != min_p else 1
        
        w = self.width()
        h = self.height()
        padding = 2
        
        # Create points
        points = []
        for i, price in enumerate(self.prices):
            x = padding + (i / (len(self.prices) - 1)) * (w - 2 * padding)
            y = h - padding - ((price - min_p) / price_range) * (h - 2 * padding)
            points.append((x, y))
        
        # Draw filled area (gradient)
        fill_color = QColor(self.color)
        fill_color.setAlpha(50)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)
        
        polygon = QPolygonF()
        polygon.append(QPointF(points[0][0], h))
        for x, y in points:
            polygon.append(QPointF(x, y))
        polygon.append(QPointF(points[-1][0], h))
        painter.drawPolygon(polygon)
        
        # Draw line
        pen = QPen(self.color, 1.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        
        for i in range(len(points) - 1):
            painter.drawLine(
                QPointF(points[i][0], points[i][1]),
                QPointF(points[i+1][0], points[i+1][1])
            )


class SectionHeader(QLabel):
    """Section header label with amber accent."""
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
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


class CompactHMMCard(QFrame):
    """Compact HMM Regime card matching VolatilityCard style."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(125)
        self.setMinimumWidth(180)
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
        
        # Row 1: Header
        header = QLabel("HIDDEN MARKOV MODEL REGIME")
        header.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; font-weight: 600; background: transparent; border: none;")
        layout.addWidget(header)
        
        # Row 2: Regime name with icon
        regime_row = QHBoxLayout()
        regime_row.setSpacing(6)
        self.regime_icon = QLabel("◉")
        self.regime_icon.setStyleSheet(f"color: {COLORS['warning']}; font-size: 18px; background: transparent; border: none;")
        regime_row.addWidget(self.regime_icon)
        
        self.regime_name = QLabel("NEUTRAL")
        self.regime_name.setStyleSheet(f"color: {COLORS['warning']}; font-size: 16px; font-weight: 700; background: transparent; border: none;")
        regime_row.addWidget(self.regime_name)
        regime_row.addStretch()
        layout.addLayout(regime_row)
        
        # Row 3: Confidence
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        layout.addWidget(self.confidence_label)
        
        # Row 4: Duration
        self.duration_label = QLabel("Duration: --")
        self.duration_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        layout.addWidget(self.duration_label)
        
        # Row 5: Next regime forecast
        self.forecast_label = QLabel("Next: --")
        self.forecast_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        layout.addWidget(self.forecast_label)
        
        layout.addStretch()
    
    def update_regime(self, regime_name: str, confidence: float, duration: str, 
                      next_regime: str, regime_color: str):
        """Update the regime display."""
        self.regime_icon.setStyleSheet(f"color: {regime_color}; font-size: 18px; background: transparent; border: none;")
        self.regime_name.setText(regime_name)
        self.regime_name.setStyleSheet(f"color: {regime_color}; font-size: 16px; font-weight: 700; background: transparent; border: none;")
        self.confidence_label.setText(f"Confidence: {confidence:.0f}%")
        self.duration_label.setText(f"Duration: {duration}")
        
        # Next regime with color
        next_color = COLORS['positive'] if 'RISK-ON' in next_regime else (
            COLORS['negative'] if 'RISK-OFF' in next_regime else COLORS['warning'])
        self.forecast_label.setText(f"Next: ")
        self.forecast_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px; background: transparent; border: none;")
        # Update to show colored next regime
        self.forecast_label.setText(f"Next: {next_regime}")


class NewsItem(QFrame):
    """Compact news item widget - shows only title, ticker and time."""
    
    clicked = Signal(str)  # Emits URL when clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.url = ""
        self.setCursor(Qt.PointingHandCursor)
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
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        
        # Title (takes most space)
        self.title_label = QLabel("")
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px; background: transparent; border: none;")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label, stretch=1)
        
        # Ticker (small badge)
        self.ticker_label = QLabel("")
        self.ticker_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['accent']};
                border: 1px solid {COLORS['accent']};
                font-size: 11px;
                font-weight: 600;
                padding: 2px 5px;
                border-radius: 3px;
            }}
        """)
        self.ticker_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.ticker_label.setFixedSize(50,25)
        self.ticker_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        
        layout.addWidget(self.ticker_label)
                
        # Time (right side)
        self.time_label = QLabel("")
        self.time_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; background: transparent; border: none; text-align: center;")
        self.time_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.time_label.setFixedSize(50,25)
        layout.addWidget(self.time_label)
    
    def set_news(self, title: str, time_str: str, url: str, ticker: str):
        """Set the news item data."""
    
        self.title_label.setText(title)
        self.time_label.setText(time_str)
    
        self.url = url
        self.ticker = ticker or ""
    
        # Tooltip shows full ticker
        self.ticker_label.setToolTip(self.ticker)
    
        # Show max 5 chars in badge
        self.ticker_label.setText(self.ticker[:5])
    
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
            from datetime import datetime, timedelta
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
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=24)
        cutoff_ts = cutoff.timestamp()
        
        valid_news = [n for n in news_items if n.get('timestamp', 0) > cutoff_ts]
        
        with open(NEWS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(valid_news, f, ensure_ascii=False, indent=2)
        print(f"[NewsCache] Saved {len(valid_news)} news items to cache")
    except Exception as e:
        print(f"[NewsCache] Error saving cache: {e}")


class NewsFeedWorker(QObject):
    """Worker thread for fetching news from yfinance for all tickers."""
    
    finished = Signal()
    result = Signal(list)  # List of news items
    error = Signal(str)
    status_message = Signal(str)
    
    def __init__(self, csv_path: str = None):
        super().__init__()
        self.csv_path = csv_path or SCHEDULED_CSV_PATH
    
    def run(self):
        """Fetch news for all tickers from CSV file."""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
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
            
            self.status_message.emit(f"Fetching news for {len(tickers)} tickers...")
            
            # 24 hour cutoff
            cutoff = datetime.now() - timedelta(hours=24)
            cutoff_ts = cutoff.timestamp()
            
            all_news = list(cached_news)  # Start with cached
            new_count = 0
            
            # Fetch news for each ticker
            for i, ticker_symbol in enumerate(tickers):
                if i % 20 == 0:
                    self.status_message.emit(f"Fetching news... {i}/{len(tickers)} tickers")
                
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    news_list = ticker.news
                    
                    if not news_list:
                        continue
                    
                    for item in news_list:
                        # Handle both old and new yfinance formats
                        # New format has 'content' dict, old format has fields directly
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
                                    # Parse ISO format: "2026-01-28T00:50:04Z"
                                    dt = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                                    timestamp = dt.timestamp()
                                except:
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
                        
                        # Skip duplicates
                        if news_id in cached_ids:
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
                        
                        news_item = {
                            'id': news_id,
                            'title': title,
                            'source': source,
                            'time': time_str,
                            'timestamp': timestamp,
                            'ticker': ticker_symbol,
                            'url': url
                        }
                        
                        all_news.append(news_item)
                        cached_ids.add(news_id)
                        new_count += 1
                        
                except Exception as e:
                    # Skip problematic tickers silently
                    continue
            
            # Sort by timestamp (newest first)
            all_news.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            
            # Save to cache
            save_news_cache(all_news)
            
            self.status_message.emit(f"Loaded {len(all_news)} news ({new_count} new)")
            self.result.emit(all_news)
            
        except Exception as e:
            import traceback
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
        self.refresh_btn.setFixedSize(25, 25)
        
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_elevated']};
                border: 1px solid {COLORS['border_default']};
                border-radius: 4px;
                color: {COLORS['text_secondary']};
                font-size: 10px;
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
        self.detector: Optional[RegimeDetector] = None
        self.selected_pair: Optional[str] = None  # For analytics tab
        self.signal_selected_pair: Optional[str] = None  # For signals tab (separate to avoid conflicts)
        self.portfolio: List[Dict] = []
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
        self._startup_complete = False  # Förhindra market watch innan startup är klar
        
        # OPTIMERING: Cache för marknadsdata (visa stängda marknader med senaste data)
        self._market_data_cache: Optional[pd.DataFrame] = None  # Cached close prices
        self._all_instruments_full: Dict = {}  # Full instrument dict (alla marknader)
        
        self._volatility_worker: Optional[VolatilityDataWorker] = None
        self._volatility_thread: Optional[QThread] = None
        self._volatility_running = False  # Säker flagga
        self._portfolio_refresh_worker: Optional[PortfolioRefreshWorker] = None
        self._portfolio_refresh_thread: Optional[QThread] = None
        self._portfolio_refresh_running = False  # Säker flagga
        
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
        
        # Auto-refresh timer (15 minutes = 900000 ms)
        # OPTIMERING: Ändrat från 5 min till 15 min för att minska frysningar
        self.auto_refresh_timer = QTimer(self)
        self.auto_refresh_timer.timeout.connect(self.auto_refresh_data)
        self.auto_refresh_timer.start(300000)  # 15 minutes
        
        # Portfolio & engine cache sync timer (90 seconds) - for Google Drive sync
        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self.sync_from_drive)
        self.sync_timer.start(90000)  # 90 seconds
        
        # Check for scheduled HMM update every minute
        self.hmm_schedule_timer = QTimer(self)
        self.hmm_schedule_timer.timeout.connect(self.check_hmm_schedule)
        self.hmm_schedule_timer.start(60000)  # 1 minute
        
        # Track last HMM update date
        self.last_hmm_update_date = None
    
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
    
    def auto_refresh_data(self):
        """Auto-refresh market data, Z-scores and MF prices.
        
        OPTIMERING: Serialiserade anrop för att undvika yfinance race conditions.
        Ordning: market_watch → (fördröjd) volatility → portfolio
        """
        # Vänta tills startup är klar
        if not self._startup_complete:
            return
        
        self.statusBar().showMessage("Auto-refreshing market data...")
        
        try:
            # Starta market watch (smart filtering - endast öppna marknader)
            self.refresh_market_watch()
            
            # Starta volatility refresh med fördröjning (så market watch hinner köra klart)
            QTimer.singleShot(5000, self._start_volatility_refresh_safe)
            
            # Portfolio refresh körs separat (ingen yfinance)
            self._auto_refresh_portfolio()
            
            self.last_updated_label.setText(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            self.statusBar().showMessage(f"Auto-refresh error: {str(e)[:50]}")
            print(f"Auto-refresh error: {e}")
    
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
        """Handle refreshed portfolio data - runs on GUI thread (safe)."""
        try:
            # Update local portfolio with refreshed data
            self.portfolio = updated_portfolio
            
            # Save and update display
            self._save_and_sync_portfolio()
            # OPTIMERING: Uppdatera endast om Portfolio-tabben är laddad
            if self._tabs_loaded.get(4, False):
                self.update_portfolio_display()
            
        except Exception as e:
            pass
    
    def _on_portfolio_refresh_error(self, error: str):
        """Handle portfolio refresh error."""
        pass
    
    def check_hmm_schedule(self):
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
        """Run the full scheduled scan: load CSV, analyze pairs, fit HMM, send to Discord."""
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
            print(f"[SCHEDULED SCAN] Tickers set in input field, starting analysis...")
            
            # Store that this is a scheduled scan so we can send Discord after completion
            self._is_scheduled_scan = True
            
            # Run analysis (will call on_analysis_complete when done)
            self.run_analysis()
            print(f"[SCHEDULED SCAN] Analysis started successfully")
            
        except Exception as e:
            import traceback
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
        """Send pair scan and HMM results to Discord."""
        if self.engine is None:
            return
        
        try:
            # Build the message
            n_tickers = len(self.engine.price_data.columns)
            n_pairs = len(self.engine.pairs_stats) if self.engine.pairs_stats is not None else 0
            n_viable = len(self.engine.viable_pairs) if self.engine.viable_pairs is not None else 0
            
            # Get regime info (supports both old and advanced detector)
            regime_text = "Not fitted"
            regime_color = 0x888888
            change_forecast_text = ""
            
            if self.detector is not None:
                try:
                    # Try advanced detector first
                    regime_data = self.detector.get_current_regime()
                    primary = regime_data['primary']
                    current_state = primary['state_id']
                    regime_info = REGIMES[current_state]
                    prob = primary['probability']
                    regime_text = f"{regime_info['name']} ({prob:.0%})"
                    regime_color = int(regime_info['color'].replace('#', ''), 16)
                    
                    # Add change forecast
                    forecast = regime_data.get('change_forecast', {})
                    if forecast:
                        change_forecast_text = (
                            f"Most likely next: {forecast.get('most_likely_next', 'N/A')}"
                        )
                except Exception as e:
                    # Fallback to old detector format
                    if hasattr(self.detector, 'states') and self.detector.states is not None and len(self.detector.states) > 0:
                        current_state = self.detector.states.iloc[-1]
                        regime_info = REGIMES[current_state]
                        prob = self.detector.regime_probs[-1, current_state]
                        regime_text = f"**{regime_info['name']}** ~{prob:.0%} Confidence"
                        regime_color = int(regime_info['color'].replace('#', ''), 16)
            
            # Build fields for embed
            fields = [
                {"name": "Tickers Analyzed", "value": str(n_tickers), "inline": True},
                {"name": "Pairs Tested", "value": str(n_pairs), "inline": True},
                {"name": "Viable Pairs", "value": str(n_viable), "inline": True},
                {"name": "📊 Current Market Regime", "value": regime_text, "inline": False},
                {"name": "🔮 Regime Change Forecast", "value": change_forecast_text, "inline": False},
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
                color=regime_color,
                fields=fields,
                footer="Klippinge Investment Trading Terminal"
            )
            
        except Exception as e:
            print(f"Error sending scan results to Discord: {e}")
            import traceback
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
                current_z = pos.get('current_z', 0.0)
                
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
        
        fit_hmm_action = QAction("Fit HMM Model", self)
        fit_hmm_action.triggered.connect(self.fit_hmm_model)
        analysis_menu.addAction(fit_hmm_action)
        
        load_hmm_action = QAction("Reload HMM Model", self)
        load_hmm_action.triggered.connect(self.load_cached_hmm)
        analysis_menu.addAction(load_hmm_action)
        
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
        
        refresh_market_action = QAction("Refresh Market Watch (Smart)", self)
        refresh_market_action.setShortcut("F5")
        refresh_market_action.triggered.connect(self.refresh_market_watch)
        data_menu.addAction(refresh_market_action)
        
        refresh_market_full_action = QAction("Refresh Market Watch (All 69 Indices)", self)
        refresh_market_full_action.setShortcut("Shift+F5")
        refresh_market_full_action.triggered.connect(lambda: self.refresh_market_watch(force_full=True))
        data_menu.addAction(refresh_market_full_action)
        
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
        }
        
        # Spara container-widgets för lazy loading
        self._tab_containers = {}
        
        # Tab 0: Market Overview - laddas direkt (starttab)
        self.tabs.addTab(self.create_market_overview_tab(), "|◎| MARKET OVERVIEW")
        
        # Tab 1-4: Containers med placeholders som fylls on-demand
        self._tab_containers[1] = self._create_lazy_container("ARBITRAGE SCANNER")
        self.tabs.addTab(self._tab_containers[1], "|◊| ARBITRAGE SCANNER")
        
        self._tab_containers[2] = self._create_lazy_container("ORNSTEIN-UHLENBECK ANALYTICS")
        self.tabs.addTab(self._tab_containers[2], "|∂x| ORNSTEIN-UHLENBECK ANALYTICS")
        
        self._tab_containers[3] = self._create_lazy_container("PAIR SIGNALS")
        self.tabs.addTab(self._tab_containers[3], "|⧗| PAIR SIGNALS")
        
        self._tab_containers[4] = self._create_lazy_container("PORTFOLIO")
        self.tabs.addTab(self._tab_containers[4], "|≡| PORTFOLIO")
        
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
        """Create Market Overview tab with new layout."""
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # LEFT: Market Watch (scrollable list)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_layout.addWidget(SectionHeader("GLOBAL MARKETS"),
                              alignment=Qt.AlignCenter)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.market_list = QWidget()
        self.market_list_layout = QVBoxLayout(self.market_list)
        self.market_list_layout.setContentsMargins(0, 0, 0, 0)
        self.market_list_layout.setSpacing(4)  # Ökat från 2 för mer luft
        
        self.market_items = {}
        
        scroll.setWidget(self.market_list)
        left_layout.addWidget(scroll)
        
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(380)
        main_layout.addWidget(left_panel, stretch=2)
        
        # CENTER: Map + Volatility cards below
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)
        
        # Map container
        map_container = QFrame()
        map_container.setStyleSheet("background-color: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 4px;")
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(0)
        
        # Try to use QWebEngineView for Plotly map
        if WEBENGINE_AVAILABLE and QWebEngineView is not None:
            self.map_widget = QWebEngineView()
            self.map_widget.setStyleSheet("background-color: #0a0a0a;")
            self.map_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.update_plotly_map([])
            map_layout.addWidget(self.map_widget, stretch=1)
        else:
            if ensure_pyqtgraph():
                self.map_widget = pg.PlotWidget()
                self.map_widget.setBackground('#0d0d0d')
                self.map_widget.hideAxis('left')
                self.map_widget.hideAxis('bottom')
                self.map_widget.setMouseEnabled(x=False, y=False)
                self.market_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen(None))
                self.map_widget.addItem(self.market_scatter)
                map_layout.addWidget(self.map_widget, stretch=1)
            else:
                map_placeholder = QLabel("🌍 World Map\n(Install PyQtWebEngine)")
                map_placeholder.setAlignment(Qt.AlignCenter)
                map_placeholder.setStyleSheet("background-color: #111; color: #444;")
                map_layout.addWidget(map_placeholder, stretch=1)
        
        # Last updated label
        self.last_updated_label = QLabel(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.last_updated_label.setStyleSheet("color: #444444; font-size: 10px; border: none; background: transparent; padding: 2px 5px;")
        self.last_updated_label.setAlignment(Qt.AlignRight)
        map_layout.addWidget(self.last_updated_label)
        
        center_layout.addWidget(map_container, stretch=1)
        
        # Volatility Cards (4 horizontal cards below map)
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
        
        # Compact HMM Regime card (next to MOVE)
        self.hmm_compact_card = CompactHMMCard()
        vol_cards_row.addWidget(self.hmm_compact_card)
        
        vol_section_layout.addLayout(vol_cards_row)
        center_layout.addWidget(vol_section)
        
        # Center panel now takes more space since HMM moved
        center_panel.setMinimumWidth(600)
        main_layout.addWidget(center_panel, stretch=6)
        
        # RIGHT: News Feed (replaces old HMM panel)
        self.news_feed = NewsFeedWidget()
        self.news_feed.setMinimumWidth(300)
        self.news_feed.setMaximumWidth(380)
        self.news_feed.refresh_btn.clicked.connect(self._refresh_news_feed)
        main_layout.addWidget(self.news_feed, stretch=2)
        
        # Create hidden regime widgets for compatibility with update methods
        # These are not displayed but needed for existing update_regime_display() logic
        self._hidden_regime_container = QWidget()
        _hidden_layout = QVBoxLayout(self._hidden_regime_container)
        
        self.regime_frame = QFrame()
        self.regime_icon_label = QLabel("◉")
        self.regime_name_label = QLabel("Loading...")
        self.regime_prob_label = QLabel("Confidence: --")
        self.regime_desc_label = QLabel("")
        
        # Forecast widgets
        self.next_regime_label = QLabel("--")
        self.expected_change_label = QLabel("--")
        self.expected_remaining_label = QLabel("--")
        
        # Stats widgets - MÅSTE matcha namnen i _update_regime_statistics_advanced()
        self.regime_duration_label = QLabel("--")
        self.regime_avg_duration_label = QLabel("--")
        self.regime_transitions_label = QLabel("--")
        self.regime_stability_label = QLabel("--")
        self.regime_lookback_label = QLabel("--")
        self.regime_window_label = QLabel("--")
        
        # Distribution widgets dict
        self.regime_dist_labels = {
            0: QLabel("--"),  # RISK-ON
            1: QLabel("--"),  # NEUTRAL
            2: QLabel("--"),  # RISK-OFF
        }
        
        # Probability bars dict - used by _update_regime_statistics_advanced()
        self.regime_prob_bars = {
            0: QProgressBar(),  # RISK-ON
            1: QProgressBar(),  # NEUTRAL
            2: QProgressBar(),  # RISK-OFF
        }
        
        # Alternative prob bars/labels used by some code paths
        self.prob_bars = {
            "RISK-ON": QProgressBar(),
            "NEUTRAL": QProgressBar(),
            "RISK-OFF": QProgressBar(),
        }
        self.prob_labels = {
            "RISK-ON": QLabel("--"),
            "NEUTRAL": QLabel("--"),
            "RISK-OFF": QLabel("--"),
        }
        
        # Change probability labels
        self.change_prob_1d_label = QLabel("--")
        self.change_prob_5d_label = QLabel("--")
        self.change_prob_20d_label = QLabel("--")
        
        # Transition matrix labels dict (tuple keys)
        self.trans_matrix_labels = {}
        for i in range(3):
            for j in range(3):
                self.trans_matrix_labels[(i, j)] = QLabel("--")
        
        # Also keep old trans_labels for compatibility
        self.trans_labels = [[QLabel("--") for _ in range(3)] for _ in range(3)]
        self.expected_duration_legend = QLabel("")
        
        # Model status label
        self.model_fitted_label = QLabel("Model: Not fitted")
        
        return tab
    
    def _refresh_news_feed(self):
        """Refresh the news feed using tickers from CSV file."""
        if hasattr(self, 'news_feed') and self.news_feed:
            # Use SCHEDULED_CSV_PATH which contains all tickers
            self.news_feed.refresh_news(SCHEDULED_CSV_PATH)
    
    def _refresh_news_feed_safe(self):
        """Safely refresh news feed - waits for other yfinance operations to complete."""
        # Don't run if market watch is still running
        if self._market_watch_running or self._volatility_running:
            # Retry in 5 seconds
            QTimer.singleShot(5000, self._refresh_news_feed_safe)
            return
        
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
    
    def _normalize_probabilities(self, raw_probs: np.ndarray, min_prob: float = 0.01, max_prob: float = 0.90) -> np.ndarray:
        """
        Normalize probabilities with minimum and maximum constraints.
        
        Args:
            raw_probs: Raw probability array
            min_prob: Minimum probability for any state (default 1%)
            max_prob: Maximum probability for any state (default 90%)
        
        Returns:
            Normalized probabilities that sum to 1.0
        """
        n_states = len(raw_probs)
        
        # Step 1: Ensure minimum for all states
        adjusted = np.maximum(raw_probs, min_prob)
        
        # Step 2: Cap maximum
        adjusted = np.minimum(adjusted, max_prob)
        
        # Step 3: Normalize to sum to 1.0
        total = adjusted.sum()
        if total > 0:
            normalized = adjusted / total
        else:
            normalized = np.ones(n_states) / n_states
        
        # Step 4: Iterative adjustment if max exceeded after normalization
        for _ in range(10):  # Max iterations
            if normalized.max() <= max_prob + 0.001:
                break
            excess = normalized.max() - max_prob
            max_idx = normalized.argmax()
            normalized[max_idx] = max_prob
            # Redistribute excess proportionally to others
            other_mask = np.arange(n_states) != max_idx
            other_sum = normalized[other_mask].sum()
            if other_sum > 0:
                normalized[other_mask] += excess * (normalized[other_mask] / other_sum)
            normalized = normalized / normalized.sum()
        
        return normalized
    
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
        
        # Lookback
        lookback_group = QVBoxLayout()
        lookback_label = QLabel("LOOKBACK:")
        lookback_label.setStyleSheet("color: #d4a574; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 6px; border: none;")
        lookback_group.addWidget(lookback_label)
        self.lookback_combo = QComboBox()
        self.lookback_combo.addItems(["1y", "2y", "5y"])
        self.lookback_combo.setCurrentText("2y")
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
            "• Engle-Granger p-value: ≤ 0.05",
            "• Johansen trace ≥ 15.4943",
            "• Hurst exponent: ≤ 0.50",
            "• Correlation: ≥ 0.70"
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

        self.viable_table.setColumnCount(6)
        self.viable_table.setHorizontalHeaderLabels([
            "Pair", "Half-life (days)", "Engle-Granger p-value", "Johansen trace", "Hurst exponent", "Correlation"
        ])
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
        self.viable_table.itemSelectionChanged.connect(self.on_viable_pair_selected)
        results_layout.addWidget(self.viable_table)
        
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
        self.ou_pair_combo.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333; padding: 6px;")
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
        left_layout.addWidget(SectionHeader("ORNSTEIN-UHLENBECK DETAILS"))
        
        self.ou_theta_card = MetricCard("MEAN REVERSION", "-")
        self.ou_mu_card = MetricCard("MEAN SPREAD", "-")
        self.ou_halflife_card = MetricCard("HALF-LIFE", "-")
        self.ou_zscore_card = MetricCard("CURRENT Z-SCORE", "-")
        self.ou_hedge_card = MetricCard("BETA", "-")
        self.ou_status_card = MetricCard("STATUS", "-")
        
        left_layout.addWidget(self.ou_theta_card)
        left_layout.addWidget(self.ou_mu_card)
        left_layout.addWidget(self.ou_halflife_card)
        left_layout.addWidget(self.ou_zscore_card)
        left_layout.addWidget(self.ou_hedge_card)
        left_layout.addWidget(self.ou_status_card)
        
        
        # =========================
        # Expected Move
        # =========================
        left_layout.addWidget(SectionHeader("EXPECTED MOVE"))
        
        self.exp_spread_change_card = MetricCard("Δ SPREAD", "-")
        self.exp_y_only_card = MetricCard("Y (100%)", "-")
        self.exp_x_only_card = MetricCard("X (100%)", "-")
        self.exp_y_half_card = MetricCard("Y (50%)", "-")
        self.exp_x_half_card = MetricCard("X (50%)", "-")
        
        left_layout.addWidget(self.exp_spread_change_card)
        left_layout.addWidget(self.exp_y_only_card)
        left_layout.addWidget(self.exp_x_only_card)
        left_layout.addWidget(self.exp_y_half_card)
        left_layout.addWidget(self.exp_x_half_card)
        
        
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
            
            # Row 1: Price comparison (full width) with DateAxisItem
            right_layout.addWidget(SectionHeader("BETA ADJUSTED PRICE COMPARISON"))
            
            # Create custom date axis for price plot
            self.ou_price_date_axis = DateAxisItem(orientation='bottom')
            self.ou_price_plot = pg.PlotWidget(axisItems={'bottom': self.ou_price_date_axis})
            self.ou_price_plot.setLabel('left', 'Price')
            self.ou_price_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_price_plot.addLegend()
            self.ou_price_plot.setMinimumHeight(200)
            # Enable mouse interaction
            self.ou_price_plot.setMouseEnabled(x=True, y=True)
            self.ou_price_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            right_layout.addWidget(self.ou_price_plot)
            
            # Row 2: Z-score and Expected Path side by side
            row2 = QHBoxLayout()
            row2.setSpacing(10)
            
            zscore_col = QVBoxLayout()
            zscore_col.addWidget(SectionHeader("SPREAD Z-SCORE"))
            
            # Create custom date axis for zscore plot
            self.ou_zscore_date_axis = DateAxisItem(orientation='bottom')
            self.ou_zscore_plot = pg.PlotWidget(axisItems={'bottom': self.ou_zscore_date_axis})
            self.ou_zscore_plot.setLabel('left', 'Z')
            self.ou_zscore_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_zscore_plot.addLine(y=2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.ou_zscore_plot.addLine(y=-2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.ou_zscore_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))
            self.ou_zscore_plot.setMinimumHeight(100)
            # Enable mouse interaction
            self.ou_zscore_plot.setMouseEnabled(x=True, y=True)
            self.ou_zscore_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            self.ou_zscore_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            
            # SYNC: Manual bidirectional sync via signals (better than setXLink for autoRange)
            self._syncing_plots = False  # Prevent infinite loop
            
            def sync_price_to_zscore(viewbox, range):
                if not self._syncing_plots:
                    self._syncing_plots = True
                    self.ou_zscore_plot.setXRange(range[0][0], range[0][1], padding=0)
                    self._syncing_plots = False
            
            def sync_zscore_to_price(viewbox, range):
                if not self._syncing_plots:
                    self._syncing_plots = True
                    self.ou_price_plot.setXRange(range[0][0], range[0][1], padding=0)
                    self._syncing_plots = False
            
            self.ou_price_plot.sigRangeChanged.connect(sync_price_to_zscore)
            self.ou_zscore_plot.sigRangeChanged.connect(sync_zscore_to_price)
            
            zscore_col.addWidget(self.ou_zscore_plot)
            row2.addLayout(zscore_col)
            
            path_col = QVBoxLayout()
            path_col.addWidget(SectionHeader("EXPECTED PATH"))
            self.ou_path_plot = pg.PlotWidget()
            self.ou_path_plot.setLabel('left', 'Spread')
            self.ou_path_plot.setLabel('bottom', 'Days')
            self.ou_path_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_path_plot.setMinimumHeight(100)
            path_col.addWidget(self.ou_path_plot)
            row2.addLayout(path_col)
            
            right_layout.addLayout(row2)
            
            # Row 3: ACF and Conditional Distribution side by side
            row3 = QHBoxLayout()
            row3.setSpacing(10)
            
            acf_col = QVBoxLayout()
            acf_col.addWidget(SectionHeader("AUTOCORRELATION (ACF)"))
            self.ou_acf_plot = pg.PlotWidget()
            self.ou_acf_plot.setLabel('left', 'ACF')
            self.ou_acf_plot.setLabel('bottom', 'Lag (days)')
            self.ou_acf_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_acf_plot.addLine(y=0, pen=pg.mkPen('#666666', width=1))
            self.ou_acf_plot.setMinimumHeight(100)
            acf_col.addWidget(self.ou_acf_plot)
            row3.addLayout(acf_col)
            
            dist_col = QVBoxLayout()
            dist_col.addWidget(SectionHeader("CONDITIONAL DISTRIBUTION"))
            self.ou_dist_plot = pg.PlotWidget()
            self.ou_dist_plot.setLabel('left', 'Density (×0.001)')
            self.ou_dist_plot.setLabel('bottom', 'Spread')
            self.ou_dist_plot.showGrid(x=False, y=False, alpha=0.3)
            self.ou_dist_plot.setMinimumHeight(100)
            dist_col.addWidget(self.ou_dist_plot)
            row3.addLayout(dist_col)
            
            right_layout.addLayout(row3)
            
            # Initialize crosshair managers (will be populated with data when plots update)
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
        deriv_header = QLabel("AVAILABLE DERIVATIVES FROM MORGAN STANLEY")
        deriv_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 13px; font-weight: 600; letter-spacing: 1px; border: none;")
        left_layout.addWidget(deriv_header)
        
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
        
        self.mini_y_name = QLabel("No mini future found")
        self.mini_y_name.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; font-weight: 500; background: transparent; border: none;")
        mini_y_layout.addWidget(self.mini_y_name)
        
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
        
        self.mini_x_name = QLabel("No mini future found")
        self.mini_x_name.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; font-weight: 500; background: transparent; border: none;")
        mini_x_layout.addWidget(self.mini_x_name)
        
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
        mf_header = QLabel("MINI FUTURES POSITION SIZING")
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
            
            # Z-score plot
            zscore_header = QLabel("SPREAD Z-SCORE")
            zscore_header.setStyleSheet(f"color: {COLORS['accent']}; font-size: 12px; font-weight: 700; letter-spacing: 1px;")
            right_layout.addWidget(zscore_header)
            
            # Create date axis for zscore plot
            self.signal_zscore_date_axis = DateAxisItem(orientation='bottom')
            self.signal_zscore_plot = pg.PlotWidget(axisItems={'bottom': self.signal_zscore_date_axis})
            self.signal_zscore_plot.setLabel('left', 'Z')
            self.signal_zscore_plot.showGrid(x=False, y=False, alpha=0.3)
            self.signal_zscore_plot.addLine(y=2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.signal_zscore_plot.addLine(y=-2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.signal_zscore_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))
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
        
        # Positions table with dynamic columns
        self.positions_table = QTableWidget()
        self.positions_table.setColumnCount(14)
        self.positions_table.setHorizontalHeaderLabels([
            "PAIR", "DIRECTION", "Z-SCORE", "STOP-LOSS", "STATUS",
            "Y LEG", "ENTRY PRICE", "QUANTITY", "P/L",
            "X LEG", "ENTRY PRICE", "QUANTITY", "P/L",
            "CLOSE POSITION"
        ])
        
        # Set row height
        self.positions_table.verticalHeader().setDefaultSectionSize(55)  # Ökat från 50
        self.positions_table.verticalHeader().setVisible(False)
        
        # Dynamic column widths - larger defaults
        header = self.positions_table.horizontalHeader()
        header.setStretchLastSection(False)
        
        # Set fixed widths for specific columns
        self.positions_table.setColumnWidth(0, 110)   # PAIR
        self.positions_table.setColumnWidth(1, 110)    # DIR
        self.positions_table.setColumnWidth(2, 110)    # Z
        self.positions_table.setColumnWidth(3, 110)    # SL
        self.positions_table.setColumnWidth(4, 110)    # STATUS
        self.positions_table.setColumnWidth(5, 250)   # MINI Y (namn)
        self.positions_table.setColumnWidth(6, 110)   # ENTRY Y
        self.positions_table.setColumnWidth(7, 110)    # QTY Y
        self.positions_table.setColumnWidth(8, 130)   # P/L Y
        self.positions_table.setColumnWidth(9, 250)   # MINI X (namn)
        self.positions_table.setColumnWidth(10, 110)  # ENTRY X
        self.positions_table.setColumnWidth(11, 110)   # QTY X
        self.positions_table.setColumnWidth(12, 130)  # P/L X
        self.positions_table.setColumnWidth(13, 150)   # CLOSE
        
        # Allow horizontal scrolling if needed
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
        
        # Clear all button (smaller)
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
        
        # Benchmark Analysis sub-tab
        benchmark_tab = self._create_benchmark_analysis_subtab()
        self.portfolio_subtabs.addTab(benchmark_tab, "📈 BENCHMARK ANALYSIS")
        
        # Auto-load benchmark when switching to that tab
        self.portfolio_subtabs.currentChanged.connect(self._on_portfolio_subtab_changed)
        self._benchmark_loaded = False  # Track if benchmark data has been loaded
        
        layout.addWidget(self.portfolio_subtabs)
        
        return tab
    
    def _on_portfolio_subtab_changed(self, index: int):
        """Handle portfolio sub-tab change - auto-load benchmark data when switching to benchmark tab."""
        # Index 1 = Benchmark Analysis tab
        if index == 1:
            # Always load/update benchmark data when switching to this tab
            self._load_or_update_benchmark()
    
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
        
        # Benchmark selector
        bench_label = QLabel("Benchmark:")
        bench_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        header_layout.addWidget(bench_label)
        
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItems([
            "SPY (S&P 500)", 
            "QQQ (Nasdaq 100)", 
            "IWM (Russell 2000)", 
            "^OMX (OMX Stockholm)",
            "^OMXS30 (OMXS30)"
        ])
        self.benchmark_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLORS['bg_hover']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 150px;
                font-size: 11px;
            }}
        """)
        self.benchmark_combo.currentIndexChanged.connect(self._on_benchmark_changed)
        header_layout.addWidget(self.benchmark_combo)
        
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
        
        # Set column widths (10 columns)
        header = self.benchmark_stats_table.horizontalHeader()
        self.benchmark_stats_table.setColumnWidth(0, 165)
        for i in range(1, 10):
            self.benchmark_stats_table.setColumnWidth(i, 100)
        header.setStretchLastSection(True)
        
        self.benchmark_stats_table.verticalHeader().setVisible(False)
        self.benchmark_stats_table.setAlternatingRowColors(True)
        self.benchmark_stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Initialize with metric names
        for i, metric in enumerate(metrics):
            item = QTableWidgetItem(metric)
            item.setForeground(QColor(COLORS['text_primary']))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
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
            chart_layout.addWidget(self.benchmark_chart)
            
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
                    except:
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
                    
                    # Sharpe (annualized)
                    if pc.std() > 0:
                        sharpe = (pc.mean() * np.sqrt(252)) / pc.std()
                        self._set_benchmark_cell(5, col, sharpe, is_ratio=True)
                    
                    if br.std() > 0:
                        bench_sharpe = (br.mean() * np.sqrt(252)) / br.std()
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
                            sortino = (pc.mean() * np.sqrt(252)) / downside_std
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


    def _on_benchmark_changed(self, index: int):
        """Handle benchmark selection change."""
        pass  # Don't auto-update, wait for button click
    
    def _get_benchmark_ticker(self) -> str:
        """Get the yfinance ticker for the selected benchmark."""
        text = self.benchmark_combo.currentText()
        if "SPY" in text:
            return "SPY"
        elif "QQQ" in text:
            return "QQQ"
        elif "IWM" in text:
            return "IWM"
        elif "OMXS30" in text:
            return "^OMXS30"
        elif "OMX" in text:
            return "^OMX"
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
            except:
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
                    sortino = (port_ret.mean() * trading_days) / downside_std
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
            import traceback
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
                    except:
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
                            except:
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
    # WORLD MAP
    # ========================================================================
    
    def update_plotly_map(self, market_data: List[Dict]):
        """Update the Plotly world map with region-based clustering and zoom-aware display."""
        # Check if we have a QWebEngineView-based map widget
        if not WEBENGINE_AVAILABLE or not hasattr(self, 'map_widget'):
            # Fallback for pyqtgraph scatter plot
            if ensure_pyqtgraph() and hasattr(self, 'market_scatter'):
                self.update_market_map_spots(market_data)
            return
        
        # Build map data JSON
        map_json = json.dumps(market_data)
        
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
        }}
        #map {{ width: 100%; height: 100%; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        const mapData = {map_json};
        
        // Region definitions with center coordinates
        const REGIONS = {{
            'NORTH_AMERICA': {{
                name: 'North America',
                tickers: ['^GSPC', '^DJI', '^NDX', '^RUT', '^GSPTSE'],
                center: {{ lat: 42, lon: -95 }}
            }},
            'SOUTH_AMERICA': {{
                name: 'South America', 
                tickers: ['^MXX', '^BVSP', '^MERV', '^IPSA', '^COLO-IV'],
                center: {{ lat: -15, lon: -60 }}
            }},
            'EUROPE': {{
                name: 'Europe',
                tickers: ['^FTSE', '^FCHI', '^STOXX', '^AEX', '^GDAXI', '^SSMI', '^ATX', 
                         '^IBEX', 'FTSEMIB.MI', 'PSI20.LS', '^BFX', '^ISEQ', 'WIG20.WA', 'XU100.IS'],
                center: {{ lat: 48, lon: 8 }}
            }},
            'NORDICS': {{
                name: 'Nordics',
                tickers: ['^OMX', 'OBX.OL', '^OMXC25', '^OMXH25'],
                center: {{ lat: 60, lon: 18 }}
            }},
            'MIDDLE_EAST': {{
                name: 'Middle East',
                tickers: ['^TA125.TA', '^TASI.SR', 'DFMGI.AE', 'FADGI.FGI', '^GNRI.QA'],
                center: {{ lat: 26, lon: 45 }}
            }},
            'AFRICA': {{
                name: 'Africa',
                tickers: ['^JN0U.JO', '^CASE30'],
                center: {{ lat: 0, lon: 25 }}
            }},
            'ASIA': {{
                name: 'Asia',
                tickers: ['^N225', '^HSI', '000001.SS', '^KS11', '^TWII', '^NSEI', '^BSESN',
                         '^STI', '^JKSE', '^KLSE', '^SET50.BK', '^VNINDEX.VN', 'PSEI.PS', 
                         '000300.SS', '399106.SZ'],
                center: {{ lat: 25, lon: 105 }}
            }},
            'OCEANIA': {{
                name: 'Oceania',
                tickers: ['^AXJO', '^NZ50'],
                center: {{ lat: -30, lon: 150 }}
            }}
        }};
        
        // Zoom threshold - below this longitude range, show individual points
        const ZOOM_THRESHOLD = 120; // degrees of longitude visible
        
        let currentZoomLevel = 350; // Start zoomed out (full world view)
        
        function getRegionForTicker(ticker) {{
            for (const [regionKey, region] of Object.entries(REGIONS)) {{
                if (region.tickers.includes(ticker)) {{
                    return regionKey;
                }}
            }}
            return null;
        }}
        
        function clusterByRegion(data) {{
            const regionData = {{}};
            const unassigned = [];
            
            // Group data by region
            for (const item of data) {{
                const region = getRegionForTicker(item.ticker);
                if (region) {{
                    if (!regionData[region]) {{
                        regionData[region] = [];
                    }}
                    regionData[region].push(item);
                }} else {{
                    unassigned.push(item);
                }}
            }}
            
            // Create clusters for each region
            const clusters = [];
            for (const [regionKey, items] of Object.entries(regionData)) {{
                if (items.length === 0) continue;
                
                const region = REGIONS[regionKey];
                const avgChange = items.reduce((s, p) => s + p.change, 0) / items.length;
                
                // Color based on average change
                let color;
                if (avgChange >= 1.0) color = '#22c55e';
                else if (avgChange >= 0.25) color = '#4ade80';
                else if (avgChange >= 0) color = '#86efac';
                else if (avgChange >= -0.25) color = '#fca5a5';
                else if (avgChange >= -1.0) color = '#f87171';
                else color = '#ef4444';
                
                clusters.push({{
                    regionKey: regionKey,
                    name: region.name,
                    lat: region.center.lat,
                    lon: region.center.lon,
                    count: items.length,
                    avgChange: avgChange,
                    color: color,
                    items: items
                }});
            }}
            
            return {{ clusters, singles: unassigned }};
        }}
        
        function buildTraces(showClusters) {{
            const traces = [];
            
            if (mapData.length === 0) {{
                traces.push({{
                    type: 'scattergeo',
                    lon: [],
                    lat: [],
                    mode: 'markers',
                    marker: {{ size: 10 }},
                    showlegend: false
                }});
                return traces;
            }}
            
            if (showClusters) {{
                // Show region clusters
                const {{ clusters, singles }} = clusterByRegion(mapData);
                
                if (clusters.length > 0) {{
                    // Glow effect layer
                    traces.push({{
                        type: 'scattergeo',
                        lon: clusters.map(c => c.lon),
                        lat: clusters.map(c => c.lat),
                        mode: 'markers',
                        marker: {{
                            size: clusters.map(c => Math.min(25 + c.count * 5, 55)),
                            color: clusters.map(c => c.color),
                            opacity: 0.2,
                            line: {{ width: 0 }}
                        }},
                        hoverinfo: 'skip',
                        showlegend: false
                    }});
                    
                    // Main cluster markers
                    const clusterHoverTexts = clusters.map(c => {{
                        const items = c.items.map(item => 
                            `<b>${{item.name}}</b>: ${{item.change > 0 ? '+' : ''}}${{item.change.toFixed(2)}}%`
                        ).join('<br>');
                        return `<b>${{c.name}}</b><br>${{c.count}} indices (avg: ${{c.avgChange > 0 ? '+' : ''}}${{c.avgChange.toFixed(2)}}%)<br><br>${{items}}`;
                    }});
                    
                    traces.push({{
                        type: 'scattergeo',
                        lon: clusters.map(c => c.lon),
                        lat: clusters.map(c => c.lat),
                        mode: 'markers',
                        marker: {{
                            size: clusters.map(c => Math.min(15 + c.count * 3, 35)),
                            color: clusters.map(c => c.color),
                            opacity: 0.85,
                            line: {{ width: 2, color: '#d4a574' }},
                            symbol: 'circle'
                        }},
                        text: clusters.map(c => c.name),
                        customdata: clusterHoverTexts,
                        hovertemplate: '%{{customdata}}<extra></extra>',
                        showlegend: false
                    }});
                }}
                
                // Show unassigned as singles
                if (singles.length > 0) {{
                    const singleHoverTexts = singles.map(d => 
                        `<b>${{d.name}}</b><br>${{d.price}}<br>${{d.change > 0 ? '+' : ''}}${{d.change.toFixed(2)}}%`
                    );
                    
                    traces.push({{
                        type: 'scattergeo',
                        lon: singles.map(d => d.lon),
                        lat: singles.map(d => d.lat),
                        mode: 'markers',
                        marker: {{
                            size: 10,
                            color: singles.map(d => d.color),
                            line: {{ width: 1.5, color: '#0a0a0a' }},
                            opacity: 0.95
                        }},
                        text: singles.map(d => d.name),
                        customdata: singleHoverTexts,
                        hovertemplate: '%{{customdata}}<extra></extra>',
                        showlegend: false
                    }});
                }}
            }} else {{
                // Show all individual points
                const hoverTexts = mapData.map(d => 
                    `<b>${{d.name}}</b><br>${{d.price}}<br>${{d.change > 0 ? '+' : ''}}${{d.change.toFixed(2)}}%`
                );
                
                traces.push({{
                    type: 'scattergeo',
                    lon: mapData.map(d => d.lon),
                    lat: mapData.map(d => d.lat),
                    mode: 'markers',
                    marker: {{
                        size: 10,
                        color: mapData.map(d => d.color),
                        line: {{ width: 1.5, color: '#0a0a0a' }},
                        opacity: 0.95
                    }},
                    text: mapData.map(d => d.name),
                    customdata: hoverTexts,
                    hovertemplate: '%{{customdata}}<extra></extra>',
                    showlegend: false
                }});
            }}
            
            return traces;
        }}
        
        function getLayout() {{
            return {{
                geo: {{
                    projection: {{ 
                        type: 'natural earth',
                        scale: 1.1
                    }},
                    showland: true,
                    landcolor: '#1a1a1a',
                    oceancolor: '#0a0a0a',
                    coastlinecolor: '#d4a574',
                    coastlinewidth: 0.5,
                    showcountries: true,
                    countrycolor: '#333',
                    countrywidth: 0.3,
                    showframe: false,
                    bgcolor: '#0a0a0a',
                    showocean: true,
                    fitbounds: false,
                    resolution: 110,
                    lataxis: {{ range: [-55, 75] }},
                    lonaxis: {{ range: [-170, 180] }},
                    center: {{ lat: 20, lon: 10 }}
                }},
                margin: {{ l: 0, r: 0, t: 0, b: 0 }},
                paper_bgcolor: '#0a0a0a',
                plot_bgcolor: '#0a0a0a',
                autosize: true,
                hoverlabel: {{
                    bgcolor: '#1a1a1a',
                    font: {{ size: 11, family: 'monospace', color: '#ffaa00' }},
                    bordercolor: '#d4a574',
                    align: 'left'
                }},
                dragmode: 'pan'
            }};
        }}
        
        function buildMap() {{
            const showClusters = currentZoomLevel > ZOOM_THRESHOLD;
            const traces = buildTraces(showClusters);
            const layout = getLayout();
            
            Plotly.newPlot('map', traces, layout, {{ 
                displayModeBar: false, 
                responsive: true,
                scrollZoom: true
            }});
            
            // Listen for zoom/pan events
            const mapDiv = document.getElementById('map');
            mapDiv.on('plotly_relayout', function(eventData) {{
                let newZoomLevel = currentZoomLevel;
                
                // Calculate visible longitude range
                if (eventData['geo.lonaxis.range']) {{
                    const lonRange = eventData['geo.lonaxis.range'];
                    newZoomLevel = Math.abs(lonRange[1] - lonRange[0]);
                }} else if (eventData['geo.projection.scale']) {{
                    // Estimate from scale
                    newZoomLevel = 350 / eventData['geo.projection.scale'];
                }}
                
                // Check if we need to switch between clusters and individual points
                const wasShowingClusters = currentZoomLevel > ZOOM_THRESHOLD;
                const shouldShowClusters = newZoomLevel > ZOOM_THRESHOLD;
                
                if (wasShowingClusters !== shouldShowClusters) {{
                    currentZoomLevel = newZoomLevel;
                    
                    // Rebuild with new display mode, preserving current view
                    const newTraces = buildTraces(shouldShowClusters);
                    
                    // Get current geo settings
                    const currentGeo = mapDiv._fullLayout.geo;
                    const newLayout = getLayout();
                    
                    // Preserve current view bounds if available
                    if (currentGeo) {{
                        if (currentGeo.lonaxis && currentGeo.lonaxis.range) {{
                            newLayout.geo.lonaxis.range = currentGeo.lonaxis.range;
                        }}
                        if (currentGeo.lataxis && currentGeo.lataxis.range) {{
                            newLayout.geo.lataxis.range = currentGeo.lataxis.range;
                        }}
                        if (currentGeo.projection && currentGeo.projection.scale) {{
                            newLayout.geo.projection.scale = currentGeo.projection.scale;
                        }}
                        if (currentGeo.center) {{
                            newLayout.geo.center = currentGeo.center;
                        }}
                    }}
                    
                    Plotly.react('map', newTraces, newLayout);
                }} else {{
                    currentZoomLevel = newZoomLevel;
                }}
            }});
            
            // Resize on window changes
            window.addEventListener('resize', function() {{
                Plotly.Plots.resize('map');
            }});
        }}
        
        buildMap();
    </script>
</body>
</html>
'''
        self.map_widget.setHtml(html)
    
    def build_map_data(self, spots_data: List[Dict]) -> List[Dict]:
        """Convert spot data to map format with coordinates."""

        NYC_LON = -74.01096
        NYC_LAT = 40.70694
        r = 0.5

        # Market locations (longitude, latitude)
        locations = {
        
            # ======================
            # America
            # ======================
            '^GSPC': (NYC_LON + 0.0000, NYC_LAT + r),       # norr
            '^DJI':  (NYC_LON + r,       NYC_LAT + 0.0000), # öst
            '^NDX':  (NYC_LON + 0.0000, NYC_LAT - r),       # syd
            '^RUT':  (NYC_LON - r,       NYC_LAT + 0.0000), # väst
        
            '^GSPTSE': (-79.3832, 43.6532),      # TSX Composite – Toronto
        
            '^MXX': (-99.1332, 19.4326),         # Mexico – Mexico City
            '^BVSP': (-46.6333, -23.5505),       # Brazil – São Paulo
            '^MERV': (-58.3816, -34.6037),       # Argentina – Buenos Aires
            '^IPSA': (-70.6693, -33.4489),       # Chile – Santiago
            '^COLO-IV': (-74.0721, 4.7110),      # Colombia – Bogotá
        
            # ======================
            # Europe
            # ======================
            '^FTSE': (-0.1278, 51.5074),         # UK – London
            '^FCHI': (2.3522, 48.8566),          # France – Paris
            '^STOXX': (4.0000, 50.0000),         # STOXX Europe 600 (central)
            '^AEX': (4.9041, 52.3676),           # Netherlands – Amsterdam
            '^GDAXI': (8.6821, 50.1109),         # Germany – Frankfurt
            '^SSMI': (8.5417, 47.3769),          # Switzerland – Zurich
            '^ATX': (16.3738, 48.2082),          # Austria – Vienna
            '^IBEX': (-3.7038, 40.4168),         # Spain – Madrid
            'FTSEMIB.MI': (9.1900, 45.4642),     # Italy – Milan
            'PSI20.LS': (-9.1393, 38.7223),      # Portugal – Lisbon
            '^BFX': (4.3517, 50.8503),           # Belgium – Brussels
            '^ISEQ': (-6.2603, 53.3498),         # Ireland – Dublin
            '^OMX': (18.0686, 59.3293),          # Sweden – Stockholm
            'OBX.OL': (10.7522, 59.9139),        # Norway – Oslo
            '^OMXC25': (12.5683, 55.6761),       # Denmark – Copenhagen
            '^OMXH25': (24.9384, 60.1699),       # Finland – Helsinki
            'WIG20.WA': (21.0122, 52.2297),      # Poland – Warsaw
            'XU100.IS': (28.9784, 41.0082),      # Turkey – Istanbul
        
            # ======================
            # Middle East
            # ======================
            '^TA125.TA': (34.7818, 32.0853),     # Israel – Tel Aviv
            '^TASI.SR': (46.6753, 24.7136),      # Saudi Arabia – Riyadh
            'DFMGI.AE': (55.2708, 25.2048),      # UAE – Dubai
            'FADGI.FGI': (54.3773, 24.4539),     # UAE – Abu Dhabi
            '^GNRI.QA': (51.5310, 25.2854),      # Qatar – Doha
        
            # ======================
            # Africa
            # ======================
            '^JN0U.JO': (28.0473, -26.2041),     # South Africa – Johannesburg
            '^CASE30': (31.2357, 30.0444),       # Egypt – Cairo
        
            # ======================
            # Asia
            # ======================
            '^N225': (139.6917, 35.6895),        # Japan – Tokyo
            '^HSI': (114.1694, 22.3193),         # Hong Kong
            '000001.SS': (121.4737, 31.2304),    # China – Shanghai
            '^KS11': (126.9780, 37.5665),        # South Korea – Seoul
            '^TWII': (121.5654, 25.0330),        # Taiwan – Taipei
            '^NSEI': (72.8777, 19.0760),         # India – Mumbai
            '^BSESN': (72.8777, 19.0760),        # India – Mumbai
            '^STI': (103.8198, 1.3521),          # Singapore
            '^JKSE': (106.8456, -6.2088),        # Indonesia – Jakarta
            '^KLSE': (101.6869, 3.1390),         # Malaysia – Kuala Lumpur
            '^SET50.BK': (100.5018, 13.7563),    # Thailand – Bangkok
            '^VNINDEX.VN': (106.6297, 10.8231),  # Vietnam – Ho Chi Minh City
            'PSEI.PS': (120.9842, 14.5995),      # Philippines – Manila
            '000300.SS': (121.4737, 31.2304),    # China – Shanghai (CSI 300)
            '399106.SZ': (114.0579, 22.5431),    # China – Shenzhen
        
            # ======================
            # Oceania
            # ======================
            '^AXJO': (151.2093, -33.8688),       # Australia – Sydney
            '^NZ50': (174.7762, -41.2865),       # New Zealand – Wellington
        }
        
        map_data = []
        for item in spots_data:
            ticker = item['ticker']
            if ticker in locations:
                lon, lat = locations[ticker]
                map_data.append({
                    'ticker': ticker,
                    'name': item['name'],
                    'lon': lon,
                    'lat': lat,
                    'change': item['change'],
                    'price': item.get('price', ''),
                    'color': item.get('color', '#ffc107')
                })
        
        return map_data

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
        
        # Get HMM cache path
        hmm_cache_path = Paths.regime_cache_file()
        
        # Create and start startup worker
        self._startup_thread = QThread()
        self._startup_worker = StartupWorker(PORTFOLIO_FILE, ENGINE_CACHE_FILE, hmm_cache_path)
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
        self._startup_worker.hmm_loaded.connect(self._on_startup_hmm_loaded)
        self._startup_worker.status_message.connect(self.statusBar().showMessage)
        
        # Start loading
        self.statusBar().showMessage("Loading cached data...")
        self._startup_thread.start()
    
    def _on_startup_portfolio_loaded(self, positions: list):
        """Handle portfolio loaded from startup worker."""
        self.portfolio = positions
        if self._tabs_loaded.get(4, False):
            self.update_portfolio_display()
        self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
        if os.path.exists(PORTFOLIO_FILE):
            self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
    
    def _on_startup_engine_loaded(self, cache_data: dict):
        """Handle engine cache loaded from startup worker."""
        self._apply_engine_cache(cache_data)
        if os.path.exists(ENGINE_CACHE_FILE):
            self._engine_cache_mtime = os.path.getmtime(ENGINE_CACHE_FILE)
    
    def _on_startup_hmm_loaded(self, detector):
        """Handle HMM detector loaded from startup worker."""
        self.detector = detector
        self.update_hmm_display()
    
    def _on_startup_finished(self):
        """Handle startup worker finished - start market data fetching."""
        # Markera att startup är klar
        self._startup_complete = True
        
        # OPTIMERING: Kör FULL market watch vid startup för att fylla cachen med alla marknader
        # Efterföljande refreshes använder smart filtering (endast öppna marknader)
        QTimer.singleShot(500, lambda: self.refresh_market_watch(force_full=True))
        
        # Starta volatility refresh EFTER market watch (5 sekunder fördröjning)
        QTimer.singleShot(5000, self._start_volatility_refresh_safe)
        
        # Ladda nyhetsflödet efter en längre fördröjning
        QTimer.singleShot(8000, self._refresh_news_feed_safe)
    
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
            import traceback
            traceback.print_exc()
    
    def _apply_engine_cache(self, cache_data: dict):
        """Apply loaded engine cache to restore full functionality."""
        import pandas as pd
        
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
                    'min_half_life': 5,
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
            import traceback
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
        """Handle portfolio sync from background thread."""
        self.portfolio = new_positions
        if os.path.exists(PORTFOLIO_FILE):
            self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
        # OPTIMERING: Uppdatera endast om Portfolio-tabben är laddad
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
                    self.portfolio = new_positions
                    self._portfolio_file_mtime = current_mtime
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
        """Save portfolio and update mtime to prevent unnecessary reloads."""
        if save_portfolio(self.portfolio):
            # Uppdatera mtime så sync-timern inte laddar om direkt
            if os.path.exists(PORTFOLIO_FILE):
                self._portfolio_file_mtime = os.path.getmtime(PORTFOLIO_FILE)
    
    def load_cached_hmm(self):
        """Load cached HMM model and update all UI elements.
        
        Prioritizes AdvancedRegimeDetector cache, falls back to basic RegimeDetector.
        """
        # Regime icons (3-state model)
        regime_icons = {
            0: "▲",   # RISK-ON - up arrow
            1: "●",   # NEUTRAL - circle
            2: "▼",   # RISK-OFF - down arrow
        }
        
        # First try to load AdvancedRegimeDetector cache
        advanced_cache_path = Paths.regime_cache_file()
        
        if os.path.exists(advanced_cache_path):
            try:
                import pickle
                with open(advanced_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Get detector directly from cache
                if 'detector' in cache_data:
                    detector = cache_data['detector']
                    cache_time = cache_data.get('timestamp', None)
                else:
                    # Old format - the cache is the detector itself
                    detector = cache_data
                    cache_time = None
                
                if detector.fitted:
                    self.detector = detector
                    
                    # Get current regime using advanced method
                    regime_data = detector.get_current_regime()
                    
                    if regime_data and 'primary' in regime_data:
                        primary = regime_data['primary']
                        current_state = primary.get('state_id', 0)
                        regime_info = REGIMES.get(current_state, REGIMES[0])
                        prob = primary.get('probability', 0.5)
                        
                        # Set main regime display
                        icon = regime_icons.get(current_state, "●")
                        self.regime_icon_label.setText(icon)
                        self.regime_icon_label.setStyleSheet(f"color: {regime_info['color']}; font-size: 28px; background: transparent; border: none;")
                        
                        self.regime_name_label.setText(f"{regime_info['name']}")
                        self.regime_name_label.setStyleSheet(f"color: {regime_info['color']}; font-size: 18px; font-weight: 700; background: transparent; border: none;")
                        self.regime_prob_label.setText(f"Confidence: {prob:.0%}")
                        self.regime_desc_label.setText(regime_info['description'])
                        
                        # Update advanced statistics
                        self._update_regime_statistics_advanced(detector, regime_data)
                    
            except (ImportError, ModuleNotFoundError) as e:
                # Cache was created with old module structure - delete it
                print(f"Cache incompatible with new module structure, deleting: {e}")
                try:
                    os.remove(advanced_cache_path)
                    print(f"Deleted old cache file: {advanced_cache_path}")
                except:
                    pass
            except Exception as e:
                print(f"Advanced HMM cache load error: {e}")
                import traceback
                traceback.print_exc()
        
        # Fall back to basic RegimeDetector with new API
        try:
            detector = RegimeDetector(model_type='hsmm', lookback_years=30)
            if detector.load_model():
                self.detector = detector
                
                # Get current regime
                regime_data = detector.get_current_regime()
                if regime_data and 'primary' in regime_data:
                    primary = regime_data['primary']
                    current_state = primary.get('state_id', 0)
                    regime_info = REGIMES.get(current_state, REGIMES[0])
                    prob = primary.get('probability', 0.5)
                    
                    # Set main regime display
                    icon = regime_icons.get(current_state, "●")
                    self.regime_icon_label.setText(icon)
                    self.regime_icon_label.setStyleSheet(f"color: {regime_info['color']}; font-size: 28px; background: transparent; border: none;")
                    
                    self.regime_name_label.setText(f"{regime_info['name']}")
                    self.regime_name_label.setStyleSheet(f"color: {regime_info['color']}; font-size: 18px; font-weight: 600; background: transparent; border: none;")
                    self.regime_prob_label.setText(f"Confidence: {prob:.0%}")
                    self.regime_desc_label.setText(regime_info['description'])
                    
                    # Update statistics
                    self._update_regime_statistics_advanced(detector, regime_data)
                    
                    return
            
            # No cache available
            self._set_no_regime_state()
        except Exception as e:
            self._set_no_regime_state()
            print(f"HMM load error: {e}")
    
    def _update_regime_statistics(self, detector, current_state):
        """Update the expanded regime statistics UI."""
        try:
            states = detector.states
            probs = detector.regime_probs
            
            # Calculate current regime duration
            duration = 1
            for i in range(len(states) - 2, -1, -1):
                if states.iloc[i] == current_state:
                    duration += 1
                else:
                    break
            self.regime_duration_label.setText(f"{duration} months")
            
            # Calculate average duration for current regime
            regime_durations = []
            current_duration = 0
            for state in states:
                if state == current_state:
                    current_duration += 1
                elif current_duration > 0:
                    regime_durations.append(current_duration)
                    current_duration = 0
            if current_duration > 0:
                regime_durations.append(current_duration)
            
            avg_duration = np.mean(regime_durations) if regime_durations else 0
            self.regime_avg_duration_label.setText(f"{avg_duration:.1f} months")
            
            # Count transitions
            transitions = sum(1 for i in range(1, len(states)) if states.iloc[i] != states.iloc[i-1])
            self.regime_transitions_label.setText(f"{transitions}")
            
            # Calculate stability (% of time in current regime)
            stability = (states == current_state).sum() / len(states) * 100
            self.regime_stability_label.setText(f"{stability:.1f}%")
            
            # Update window label
            if hasattr(self, 'regime_window_label'):
                self.regime_window_label.setText(f"{len(states)} months")
            
            # Update regime distribution labels
            if hasattr(self, 'regime_dist_labels'):
                for state_id, pct_lbl in self.regime_dist_labels.items():
                    pct = (states == state_id).sum() / len(states) * 100
                    pct_lbl.setText(f"{pct:.1f}%")
            
            # Update probability bars (new format: tuple of (bar, label))
            if probs is not None and len(probs) > 0:
                current_probs = probs[-1]
                for state, bar_data in self.regime_prob_bars.items():
                    if isinstance(bar_data, tuple):
                        prob_bar, pct_lbl = bar_data
                        if state < len(current_probs):
                            val = int(current_probs[state] * 100)
                            prob_bar.setValue(val)
                            pct_lbl.setText(f"{val}%")
                        else:
                            prob_bar.setValue(0)
                            pct_lbl.setText("0%")
                    else:
                        # Old format (just bar)
                        if state < len(current_probs):
                            bar_data.setValue(int(current_probs[state] * 100))
                        else:
                            bar_data.setValue(0)
        except Exception as e:
            print(f"Error updating regime statistics: {e}")
    
    def _set_no_regime_state(self):
        """Set UI state when no regime data available."""
        if hasattr(self, 'regime_icon_label') and self.regime_icon_label:
            self.regime_icon_label.setText("○")
            self.regime_icon_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 28px; background: transparent; border: none;")
        if hasattr(self, 'regime_name_label') and self.regime_name_label:
            self.regime_name_label.setText("No regime data")
            self.regime_name_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 18px; font-weight: 600; background: transparent; border: none;")
        if hasattr(self, 'regime_prob_label') and self.regime_prob_label:
            self.regime_prob_label.setText("Run HMM fit from menu")
        if hasattr(self, 'regime_desc_label') and self.regime_desc_label:
            self.regime_desc_label.setText("")
        
        if hasattr(self, 'regime_duration_label') and self.regime_duration_label:
            self.regime_duration_label.setText("-- months")
        if hasattr(self, 'regime_avg_duration_label') and self.regime_avg_duration_label:
            self.regime_avg_duration_label.setText("-- months")
        if hasattr(self, 'regime_transitions_label') and self.regime_transitions_label:
            self.regime_transitions_label.setText("--")
        if hasattr(self, 'regime_stability_label') and self.regime_stability_label:
            self.regime_stability_label.setText("--%")
        
        if hasattr(self, 'regime_window_label') and self.regime_window_label:
            self.regime_window_label.setText("--")
        
        if hasattr(self, 'regime_dist_labels') and self.regime_dist_labels:
            for pct_lbl in self.regime_dist_labels.values():
                if pct_lbl:
                    pct_lbl.setText("--%")
        
        if hasattr(self, 'regime_prob_bars') and self.regime_prob_bars:
            for bar_data in self.regime_prob_bars.values():
                if isinstance(bar_data, tuple):
                    bar_data[0].setValue(0)
                    bar_data[1].setText("0%")
                elif bar_data:
                    bar_data.setValue(0)
        
        # Reset change forecast labels
        if hasattr(self, 'change_prob_1d_label') and self.change_prob_1d_label:
            self.change_prob_1d_label.setText("--")
        if hasattr(self, 'change_prob_5d_label') and self.change_prob_5d_label:
            self.change_prob_5d_label.setText("--")
        if hasattr(self, 'change_prob_20d_label') and self.change_prob_20d_label:
            self.change_prob_20d_label.setText("--")
        if hasattr(self, 'expected_change_label') and self.expected_change_label:
            self.expected_change_label.setText("-- months")
        if hasattr(self, 'next_regime_label') and self.next_regime_label:
            self.next_regime_label.setText("--")
        if hasattr(self, 'expected_remaining_label') and self.expected_remaining_label:
            self.expected_remaining_label.setText("-- months")
        
        if hasattr(self, 'model_fitted_label') and self.model_fitted_label:
            self.model_fitted_label.setText("Model: Not fitted")
            self.model_fitted_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; background: transparent;")
        
        # Sync compact HMM card
        self._sync_compact_hmm_card("NEUTRAL", 0, "--", "--", COLORS['warning'])
    
    def _sync_compact_hmm_card(self, regime_name: str, confidence: float, duration: str, 
                                next_regime: str, regime_color: str):
        """Sync the compact HMM card with current regime data."""
        if hasattr(self, 'hmm_compact_card') and self.hmm_compact_card is not None:
            self.hmm_compact_card.update_regime(
                regime_name=regime_name,
                confidence=confidence,
                duration=duration,
                next_regime=next_regime,
                regime_color=regime_color
            )
    
    def _set_error_regime_state(self, error: str):
        """Set UI state when error loading regime."""
        self.regime_icon_label.setText("✕")
        self.regime_icon_label.setStyleSheet(f"color: {COLORS['negative']}; font-size: 28px; background: transparent; border: none;")
        self.regime_name_label.setText("HMM not loaded")
        self.regime_name_label.setStyleSheet(f"color: {COLORS['negative']}; font-size: 18px; font-weight: 600; background: transparent; border: none;")
        self.regime_prob_label.setText(f"Error: {error[:40]}")
        self.regime_desc_label.setText("")
        
        self.regime_duration_label.setText("-- months")
        self.regime_avg_duration_label.setText("-- months")
        self.regime_transitions_label.setText("--")
        self.regime_stability_label.setText("--%")
        
        if hasattr(self, 'regime_window_label'):
            self.regime_window_label.setText("--")
        
        if hasattr(self, 'regime_dist_labels'):
            for pct_lbl in self.regime_dist_labels.values():
                pct_lbl.setText("--%")
        
        for bar_data in self.regime_prob_bars.values():
            if isinstance(bar_data, tuple):
                bar_data[0].setValue(0)
                bar_data[1].setText("0%")
            else:
                bar_data.setValue(0)
        
        self.model_fitted_label.setText("Model: Error")
        self.model_fitted_label.setStyleSheet("color: #ff1744; font-size: 11px; background: transparent;")
        
        # Sync compact HMM card
        self._sync_compact_hmm_card("ERROR", 0, "--", "--", COLORS['negative'])
    
    def fit_hmm_model(self):
        """Fit HMM model in background thread."""
        self.regime_name_label.setText("Fitting model...")
        self.regime_prob_label.setText("Please wait...")
        self.regime_desc_label.setText("")
        self.statusBar().showMessage("Fitting HMM model...")
        
        # Run in background thread
        self.hmm_thread = QThread()
        self.hmm_worker = HMMWorker()
        self.hmm_worker.moveToThread(self.hmm_thread)
        
        self.hmm_thread.started.connect(self.hmm_worker.run)
        self.hmm_worker.finished.connect(self.hmm_thread.quit)
        self.hmm_worker.progress.connect(self.on_hmm_progress)
        self.hmm_worker.result.connect(self.on_hmm_complete)
        self.hmm_worker.error.connect(self.on_hmm_error)
        
        self.hmm_thread.start()
    
    def on_hmm_progress(self, progress: int, message: str):
        """Handle HMM progress update."""
        self.regime_prob_label.setText(f"{message} ({progress}%)")
        self.statusBar().showMessage(message)
    
    def on_hmm_complete(self, detector):
        """Handle HMM fit completion with advanced detector."""
        self.detector = detector
        
        # Regime icons (3-state model)
        regime_icons = {
            0: "▲",   # RISK-ON
            1: "●",   # NEUTRAL
            2: "▼",   # RISK-OFF
        }
        
        try:
            # Get comprehensive regime info from advanced detector
            regime_data = detector.get_current_regime()
            primary = regime_data['primary']
            
            current_state = primary['state_id']
            regime_info = REGIMES[current_state]
            prob = primary['probability']
            
            # Set icon
            icon = regime_icons.get(current_state, "●")
            self.regime_icon_label.setText(icon)
            self.regime_icon_label.setStyleSheet(f"color: {regime_info['color']}; font-size: 28px; background: transparent; border: none;")
            
            self.regime_name_label.setText(f"{regime_info['name']}")
            self.regime_name_label.setStyleSheet(f"color: {regime_info['color']}; font-size: 18px; font-weight: 600; background: transparent; border: none;")
            self.regime_prob_label.setText(f"Confidence: {prob:.0%}")
            self.regime_desc_label.setText(regime_info['description'])
            
            # Update expanded statistics with advanced metrics
            self._update_regime_statistics_advanced(detector, regime_data)
            
        except Exception as e:
            print(f"Error in on_hmm_complete: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic display
            self.regime_name_label.setText("Model Fitted")
            self.regime_prob_label.setText("See details below")
        
        self.statusBar().showMessage("Advanced HMM model fitted successfully")
        
        # If this was part of a scheduled scan, send Discord notification
        if hasattr(self, '_is_scheduled_scan') and self._is_scheduled_scan:
            self._is_scheduled_scan = False  # Reset flag
            self._scheduled_scan_running = False  # Reset guard flag
            self.send_scan_results_to_discord()
            self.statusBar().showMessage("Scheduled scan complete - results sent to Discord")
    
    def _update_regime_statistics_advanced(self, detector, regime_data):
        """Update regime statistics UI with advanced metrics."""
        try:
            primary = regime_data['primary']
            change_forecast = regime_data['change_forecast']
            stability = regime_data['stability']
            
            current_state = primary['state_id']
            
            # Current duration (from HSMM if available)
            if 'current_duration' in primary:
                duration = primary['current_duration']
            else:
                duration = stability.get('current_duration', 0)
            
            # Defensive: check if widgets exist before updating
            if hasattr(self, 'regime_duration_label') and self.regime_duration_label:
                self.regime_duration_label.setText(f"{duration} months")
            
            # Average duration
            avg_duration = stability.get('avg_regime_duration', 0)
            if hasattr(self, 'regime_avg_duration_label') and self.regime_avg_duration_label:
                self.regime_avg_duration_label.setText(f"{avg_duration:.1f} months")
            
            # Transitions
            transitions = stability.get('total_transitions', 0)
            if hasattr(self, 'regime_transitions_label') and self.regime_transitions_label:
                self.regime_transitions_label.setText(f"{transitions}")
            
            # Stability score
            regime_dist = stability.get('regime_distribution', {})
            current_regime_name = REGIMES[current_state]['name']
            stability_pct = regime_dist.get(current_regime_name, 0) * 100
            if hasattr(self, 'regime_stability_label') and self.regime_stability_label:
                self.regime_stability_label.setText(f"{stability_pct:.1f}%")
            
            # Update window label
            if hasattr(self, 'regime_window_label'):
                n_obs = len(detector.states) if hasattr(detector, 'states') else 0
                self.regime_window_label.setText(f"{n_obs} months")
            
            # Update regime distribution labels
            if hasattr(self, 'regime_dist_labels'):
                for state_id, pct_lbl in self.regime_dist_labels.items():
                    regime_name = REGIMES[state_id]['name']
                    pct = regime_dist.get(regime_name, 0) * 100
                    pct_lbl.setText(f"{pct:.1f}%")
            
            # Update probability bars with proper normalization:
            # 1. Ensure minimum 1% for all regimes
            # 2. Cap maximum at 90%
            # 3. Normalize so sum = 100%
            if hasattr(detector, 'regime_probs') and detector.regime_probs is not None:
                raw_probs = detector.regime_probs[-1].copy()
                n_states = len(raw_probs)
                
                # Store normalized probs for transition matrix too
                self._normalized_probs = self._normalize_probabilities(raw_probs)
                
                for state, bar_data in self.regime_prob_bars.items():
                    if isinstance(bar_data, tuple):
                        prob_bar, pct_lbl = bar_data
                        if state < len(self._normalized_probs):
                            val = max(1, int(round(self._normalized_probs[state] * 100)))
                            prob_bar.setValue(val)
                            pct_lbl.setText(f"{val}%")
                    else:
                        if state < len(self._normalized_probs):
                            bar_data.setValue(max(1, int(round(self._normalized_probs[state] * 100))))
            
            # Update advanced metrics (change probabilities - now monthly)
            if hasattr(self, 'change_prob_1d_label'):
                self.change_prob_1d_label.setText(f"{change_forecast.get('1m', 0.1):.1%}")
            if hasattr(self, 'change_prob_5d_label'):
                self.change_prob_5d_label.setText(f"{change_forecast.get('3m', 0.3):.1%}")
            if hasattr(self, 'change_prob_20d_label'):
                self.change_prob_20d_label.setText(f"{change_forecast.get('12m', 0.7):.1%}")
            if hasattr(self, 'expected_change_label'):
                self.expected_change_label.setText(f"{change_forecast.get('expected_months', 6):.0f} months")
            if hasattr(self, 'next_regime_label'):
                next_regime_name = change_forecast['most_likely_next']
                self.next_regime_label.setText(next_regime_name)
                # Use color from forecast if available, otherwise lookup by name
                if 'next_state_color' in change_forecast:
                    next_color = change_forecast['next_state_color']
                else:
                    # Fallback colors matching REGIMES in regime_hmm.py (3-state model)
                    regime_colors = {
                        'RISK-ON': '#22c55e',      # Green
                        'NEUTRAL': '#f59e0b',      # Orange/Amber
                        'RISK-OFF': '#ef4444'      # Red
                    }
                    next_color = regime_colors.get(next_regime_name, COLORS['text_secondary'])
                self.next_regime_label.setStyleSheet(f"color: {next_color}; font-size: 12px; font-weight: 600; background: transparent;")
            
            # HSMM-specific: Expected remaining duration (in months)
            if hasattr(self, 'expected_remaining_label') and 'expected_remaining' in primary:
                self.expected_remaining_label.setText(f"{primary['expected_remaining']:.0f} months")
            
            # Update transition matrix display with RAW values (no normalization - show actual HMM parameters)
            if hasattr(self, 'trans_matrix_labels') and hasattr(detector, 'get_transition_matrix'):
                try:
                    raw_trans = detector.get_transition_matrix()
                    n_states = raw_trans.shape[0]  # Get actual number of states
                    
                    # Display raw transition probabilities
                    for from_state in range(n_states):
                        for to_state in range(n_states):
                            if (from_state, to_state) in self.trans_matrix_labels:
                                prob_pct = raw_trans[from_state, to_state] * 100
                                cell_lbl = self.trans_matrix_labels[(from_state, to_state)]
                                if prob_pct >= 10:
                                    cell_lbl.setText(f"{prob_pct:.0f}%")
                                elif prob_pct >= 1:
                                    cell_lbl.setText(f"{prob_pct:.1f}%")
                                else:
                                    cell_lbl.setText(f"{prob_pct:.2f}%")
                    
                    # Update expected duration legend based on current regime (monthly)
                    if hasattr(self, 'expected_duration_legend'):
                        stay_prob = raw_trans[current_state, current_state]
                        if stay_prob < 1:
                            expected_months = 1 / (1 - stay_prob)
                            self.expected_duration_legend.setText(f"P(stay)={stay_prob:.1%} → Expected ~{expected_months:.0f} months")
                        else:
                            self.expected_duration_legend.setText("P(stay)≈100% → Very stable regime")
                            
                except Exception as te:
                    print(f"Error updating transition matrix: {te}")
            
            # Sync compact HMM card with current data
            current_regime_name = REGIMES[current_state]['name']
            current_color = REGIMES[current_state]['color']
            duration_str = f"{duration} mo" if isinstance(duration, (int, float)) else str(duration)
            next_regime = change_forecast.get('most_likely_next', '--')
            prob_pct = primary.get('probability', 0.5) * 100
            
            self._sync_compact_hmm_card(
                regime_name=current_regime_name,
                confidence=prob_pct,
                duration=duration_str,
                next_regime=next_regime,
                regime_color=current_color
            )
            
        except Exception as e:
            print(f"Error updating advanced regime statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def on_hmm_error(self, error: str):
        """Handle HMM fit error."""
        self.regime_icon_label.setText("✕")
        self.regime_icon_label.setStyleSheet("color: #ff1744; font-size: 24px; background: transparent; border: none;")
        self.regime_name_label.setText("Error fitting HMM")
        self.regime_name_label.setStyleSheet("color: #ff1744; font-size: 11px; font-weight: 600; background: transparent; border: none;")
        self.regime_prob_label.setText(error[:40])
        self.statusBar().showMessage(f"HMM error: {error}")
        
        # Sync compact card
        self._sync_compact_hmm_card("ERROR", 0, "--", "--", COLORS['negative'])
        
        # If this was part of a scheduled scan, still send Discord notification with error
        if hasattr(self, '_is_scheduled_scan') and self._is_scheduled_scan:
            self._is_scheduled_scan = False  # Reset flag
            self._scheduled_scan_running = False  # Reset guard flag
            self.send_discord_notification(
                title="⚠️ Scheduled Scan - HMM Error",
                description=f"Pair scan completed but HMM fitting failed:\n{error}",
                color=0xff0000  # Red
            )
            self.statusBar().showMessage("Scheduled scan: HMM failed - error sent to Discord")
    
    def refresh_market_watch(self, force_full: bool = False):
        """Refresh market watch data asynchronously.
        
        OPTIMERING: Flyttar tung yfinance.download till bakgrundstråd.
        
        Args:
            force_full: If True, fetch ALL indices regardless of market hours.
                       If False (default), only fetch indices for open markets.
        """
        # Vänta tills startup är klar (förhindra tidiga anrop)
        if not self._startup_complete:
            print(f"[MarketWatch] Skipped - startup not complete yet")
            return
        
        # Don't start if already running - använd säker flagga
        if self._market_watch_running:
            self.statusBar().showMessage("Market watch already updating...")
            return
        
        # Clear existing items
        while self.market_list_layout.count():
            child = self.market_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Show loading indicator
        loading_label = QLabel("⏳ Loading market data...")
        loading_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; padding: 20px;")
        loading_label.setAlignment(Qt.AlignCenter)
        self.market_list_layout.addWidget(loading_label)
        
        # Complete instrument definitions
        indices = {
            # ======================
            # America
            # ======================
            '^GSPC': ('S&P 500', 'AMERICA'),
            '^NDX': ('NASDAQ 100', 'AMERICA'),
            '^DJI': ('Dow Jones', 'AMERICA'),
            '^RUT': ('Russell 2000', 'AMERICA'),
            '^GSPTSE': ('Toronto', 'AMERICA'),
            '^MXX': ('Mexico City', 'AMERICA'),
            '^BVSP': ('São Paulo', 'AMERICA'),
            '^MERV': ('Buenos Aires', 'AMERICA'),
            '^IPSA': ('Santiago', 'AMERICA'),
            '^COLO-IV': ('Bogotá', 'AMERICA'),
            # ======================
            # Europe
            # ======================
            '^FTSE': ('London', 'EUROPE'),
            '^FCHI': ('Paris', 'EUROPE'),
            '^STOXX': ('Europe 600', 'EUROPE'),
            '^AEX': ('Amsterdam', 'EUROPE'),
            '^GDAXI': ('Frankfurt', 'EUROPE'),
            '^SSMI': ('Zürich', 'EUROPE'),
            '^ATX': ('Vienna', 'EUROPE'),
            '^IBEX': ('Madrid', 'EUROPE'),
            'FTSEMIB.MI': ('Milano', 'EUROPE'),
            'PSI20.LS': ('Lisbon', 'EUROPE'),
            '^BFX': ('Brussels', 'EUROPE'),
            '^ISEQ': ('Dublin', 'EUROPE'),
            'XU100.IS': ('Istanbul', 'EUROPE'),            
            '^OMX': ('Stockholm', 'EUROPE'),
            'OBX.OL': ('Oslo', 'EUROPE'),
            '^OMXC25': ('Copenhagen', 'EUROPE'),
            '^OMXH25': ('Helsinki', 'EUROPE'),
            'WIG20.WA': ('Warsaw', 'EUROPE'),
            # ======================
            # Middle East
            # ======================
            '^TA125.TA': ('Tel Aviv', 'MIDDLE EAST'),
            '^TASI.SR': ('Riyadh', 'MIDDLE EAST'),
            'DFMGI.AE': ('Dubai', 'MIDDLE EAST'),
            'FADGI.FGI': ('Abu Dhabi', 'MIDDLE EAST'),
            '^GNRI.QA': ('Qatar', 'MIDDLE EAST'),
            # ======================
            # Africa
            # ======================
            '^JN0U.JO': ('Johannesburg', 'AFRICA'),
            '^CASE30': ('Cairo', 'AFRICA'),
            # ======================
            # Asia
            # ======================
            '^N225': ('Tokyo', 'ASIA'),
            '^HSI': ('Hong Kong', 'ASIA'),
            '000001.SS': ('Shanghai', 'ASIA'),
            '^KS11': ('Seoul', 'ASIA'),
            '^TWII': ('Taipei', 'ASIA'),
            '^NSEI': ('Nifty 50', 'ASIA'),
            '^BSESN': ('Mumbai', 'ASIA'),
            '^STI': ('Singapore', 'ASIA'),
            '^JKSE': ('Jakarta', 'ASIA'),
            '^KLSE': ('Kuala Lumpur', 'ASIA'),
            '^SET50.BK': ('Bangkok', 'ASIA'),
            '^VNINDEX.VN': ('Ho Chi Minh City', 'ASIA'),
            'PSEI.PS': ('Manila', 'ASIA'),
            '000300.SS': ('Shanghai', 'ASIA'),
            '399106.SZ': ('Shenzen', 'ASIA'),
            # ======================
            # Oceania
            # ======================
            '^AXJO': ('Sydney', 'OCEANIA'),
            '^NZ50': ('Wellington', 'OCEANIA'),
        }
        
        macro = {
            # Currencies
            'EURUSD=X': ('EUR/USD', 'CURRENCIES'),
            'EURSEK=X': ('EUR/SEK', 'CURRENCIES'),
            'GBPUSD=X': ('GBP/USD', 'CURRENCIES'),
            'USDJPY=X': ('USD/JPY', 'CURRENCIES'),
            'USDCHF=X': ('USD/CHF', 'CURRENCIES'),
            'AUDUSD=X': ('AUD/USD', 'CURRENCIES'),
            'USDCAD=X': ('USD/CAD', 'CURRENCIES'),
            'USDSEK=X': ('USD/SEK', 'CURRENCIES'),
            # Commodities
            'GC=F': ('Gold', 'COMMODITIES'),
            'SI=F': ('Silver', 'COMMODITIES'),
            'CL=F': ('Crude Oil', 'COMMODITIES'),
            'NG=F': ('Natural Gas', 'COMMODITIES'),
            'HG=F': ('Copper', 'COMMODITIES'),
            # Yields
            '^TNX': ('10Y Yield', 'YIELDS'),
            '^FVX': ('5Y Yield', 'YIELDS'),
            '^TYX': ('30Y Yield', 'YIELDS'),
            '^IRX': ('3M Yield', 'YIELDS'),
        }
        
        # =====================================================================
        # SMART MARKET FILTERING - Only fetch data for open markets
        # Macro data (currencies, commodities, yields) fetches ALWAYS (24h markets)
        # =====================================================================
        if force_full:
            # Force full refresh - fetch all indices
            filtered_indices = indices
            print(f"[MarketWatch] FULL REFRESH: Fetching all {len(indices)} indices")
            self.statusBar().showMessage(f"Full refresh: Fetching all {len(indices)} indices + {len(macro)} macro...")
        else:
            # Get currently open market regions from header clocks
            open_regions = self.header_bar.get_open_markets()
            
            # Regions without dedicated clocks - always include if any related market is open
            # Africa trades roughly aligned with Europe/Middle East
            if 'EUROPE' in open_regions or 'MIDDLE EAST' in open_regions:
                open_regions.add('AFRICA')
            
            # Filter indices to only include open markets
            if open_regions:
                # Some markets are open - filter to just those regions
                filtered_indices = {
                    ticker: (name, region) 
                    for ticker, (name, region) in indices.items()
                    if region in open_regions
                }
                closed_count = len(indices) - len(filtered_indices)
                
                # Status message
                open_list = ', '.join(sorted(open_regions))
                print(f"[MarketWatch] Open markets: {open_list}")
                print(f"[MarketWatch] Fetching {len(filtered_indices)} indices (skipping {closed_count} from closed markets)")
                self.statusBar().showMessage(f"Fetching {len(filtered_indices)} indices + {len(macro)} macro ({closed_count} closed markets skipped)...")
            else:
                # No markets open (weekend/off-hours) - fetch minimal set for display
                # Just fetch major indices for reference (they'll show last close)
                MAJOR_INDICES = {'^GSPC', '^NDX', '^DJI', '^FTSE', '^GDAXI', '^N225', '^HSI'}
                filtered_indices = {
                    ticker: (name, region)
                    for ticker, (name, region) in indices.items()
                    if ticker in MAJOR_INDICES
                }
                print(f"[MarketWatch] All markets closed - fetching {len(filtered_indices)} major indices only")
                self.statusBar().showMessage(f"Markets closed - fetching {len(filtered_indices)} major indices + {len(macro)} macro...")
        
        # Combine filtered indices with macro (macro ALWAYS fetched - 24h markets)
        all_instruments = {**filtered_indices, **macro}
        
        # Store FULL instruments dict for display (all markets, not just open ones)
        self._all_instruments_full = {**indices, **macro}
        
        # Store full indices dict for display purposes (to show closed markets greyed out)
        self._all_indices_cache = indices
        
        # Sätt flagga INNAN vi skapar tråden
        self._market_watch_running = True
        
        # Create and start worker
        self._market_watch_thread = QThread()
        self._market_watch_worker = MarketWatchWorker(all_instruments)
        self._market_watch_worker.moveToThread(self._market_watch_thread)
        
        self._market_watch_thread.started.connect(self._market_watch_worker.run)
        self._market_watch_worker.finished.connect(self._market_watch_thread.quit)
        # VIKTIGT: Återställ flagga och rensa referenser när tråden är klar
        self._market_watch_thread.finished.connect(self._on_market_watch_thread_finished)
        self._market_watch_worker.result.connect(self._on_market_watch_received)
        self._market_watch_worker.error.connect(self._on_market_watch_error)
        self._market_watch_worker.status_message.connect(self.statusBar().showMessage)
        
        self._market_watch_thread.start()
        self.statusBar().showMessage("Fetching market data in background...")
    
    def _on_market_watch_thread_finished(self):
        """Handle market watch thread completion - clear references and flag."""
        self._market_watch_running = False
        
        # VIKTIGT: Använd deleteLater() istället för att direkt sätta till None
        # Detta låter Qt städa upp ordentligt
        if self._market_watch_worker is not None:
            self._market_watch_worker.deleteLater()
        if self._market_watch_thread is not None:
            self._market_watch_thread.deleteLater()
        
        # Rensa referenserna efter deleteLater() är scheduled
        self._market_watch_worker = None
        self._market_watch_thread = None
        
        # NOTERA: Volatility triggas INTE här längre - körs separat från _on_startup_finished
        # Detta förhindrar multipla volatility-körningar
    
    def _start_volatility_refresh_safe(self):
        """Safely start volatility refresh after a delay."""
        print("[Volatility] Starting safe volatility refresh...")
        try:
            self.refresh_market_data()
        except Exception as e:
            print(f"[Volatility] Error starting refresh: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_market_watch_received(self, close, all_instruments: dict):
        """Handle received market watch data - runs on GUI thread (safe).
        
        OPTIMERING: UI-uppdatering sker efter att data hämtats i bakgrunden.
        Kombinerar ny data med cachad data för att visa alla marknader.
        """
        try:
            # DEBUG: Visa vad som faktiskt hämtades
            fetched_tickers = list(close.columns) if hasattr(close, 'columns') else []
            print(f"[MarketWatch] Received {len(fetched_tickers)} tickers from yfinance")
            print(f"[MarketWatch] Requested {len(all_instruments)} instruments")
            if len(fetched_tickers) < 20:
                print(f"[MarketWatch] Fetched tickers: {fetched_tickers}")
            
            # =====================================================================
            # SMART CACHING - Combine new data with cached data for full display
            # =====================================================================
            # Update cache with newly fetched data
            if self._market_data_cache is None:
                self._market_data_cache = close.copy()
                print(f"[MarketWatch] Created new cache with {len(close.columns)} columns")
            else:
                # Update only the columns we just fetched (preserves cached data for closed markets)
                before_cols = len(self._market_data_cache.columns)
                for col in close.columns:
                    self._market_data_cache[col] = close[col]
                after_cols = len(self._market_data_cache.columns)
                print(f"[MarketWatch] Updated cache: {before_cols} -> {after_cols} columns")
            
            # Use full instruments dict (all markets) for display, not just what was fetched
            if self._all_instruments_full:
                instruments = dict(self._all_instruments_full)
                print(f"[MarketWatch] Using _all_instruments_full with {len(instruments)} instruments")
            else:
                instruments = dict(all_instruments)
                print(f"[MarketWatch] WARNING: _all_instruments_full is empty, using all_instruments with {len(instruments)} instruments")
            
            # Use combined data (new + cached) for display
            display_close = self._market_data_cache
            print(f"[MarketWatch] display_close has {len(display_close.columns)} columns")
            
            # Track which tickers were just updated vs showing cached
            freshly_updated_tickers = set(close.columns)
            print(f"[MarketWatch] freshly_updated_tickers: {len(freshly_updated_tickers)}")
            
            # Clear loading indicator
            while self.market_list_layout.count():
                child = self.market_list_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            
            # Get unique trading dates
            if hasattr(display_close.index, 'date'):
                dates = pd.Series(display_close.index.date)
                unique_dates = sorted(dates.unique())
                
                if len(unique_dates) >= 2:
                    most_recent_day = unique_dates[-1]
                    previous_day = unique_dates[-2]
                elif len(unique_dates) == 1:
                    most_recent_day = unique_dates[0]
                    previous_day = None
                else:
                    most_recent_day = None
                    previous_day = None
            else:
                most_recent_day = None
                previous_day = None
                       
            # Group by category
            categories = {}
            for ticker, (name, category) in instruments.items():
                if category not in categories:
                    categories[category] = []
                categories[category].append((ticker, name))
            
            
            # Order of categories
            category_order = ['AMERICA', 'EUROPE', 'MIDDLE EAST', 'AFRICA', 'ASIA', 'OCEANIA', 
                            'CURRENCIES', 'COMMODITIES', 'YIELDS']
            
            # Category colors
            category_colors = {
                'AMERICA': COLORS['accent'],
                'EUROPE': COLORS['accent'],
                'MIDDLE EAST': COLORS['accent'],
                'AFRICA': COLORS['accent'],
                'ASIA': COLORS['accent'],
                'OCEANIA': COLORS['accent'],
                'CURRENCIES': COLORS['info'],
                'COMMODITIES': COLORS['warning'],
                'YIELDS': '#9c27b0',
            }
            
            # Map coordinates for scatter plot
            scatter_spots = []
            widgets_added = 0
            fresh_count = 0
            cached_count = 0
            
            for category in category_order:
                if category not in categories:
                    continue
                    
                # Region header
                cat_color = category_colors.get(category, COLORS['accent'])
                region_label = QLabel(category)
                region_label.setStyleSheet(f"color: {cat_color}; font-size: 13px; font-weight: 700; padding: 12px 0 4px 0; letter-spacing: 1.5px; border-bottom: 2px solid {COLORS['accent']};")
                self.market_list_layout.addWidget(region_label)
                widgets_added += 1
                
                for ticker, name in categories[category]:
                    # Check if we have data for this ticker (either fresh or cached)
                    if ticker not in display_close.columns:
                        continue
                    
                    # Track if this is cached data (market closed)
                    is_cached = ticker not in freshly_updated_tickers
                    
                    try:
                        ticker_series = display_close[ticker].dropna()
                        if len(ticker_series) < 1:
                            continue
                        
                        latest = ticker_series.iloc[-1]
                        
                        # Calculate change and sparkline data
                        if most_recent_day is not None and hasattr(ticker_series.index, 'date'):
                            ticker_dates = pd.Series(ticker_series.index.date)
                            unique_ticker_dates = sorted(ticker_dates.unique())
                            
                            if len(unique_ticker_dates) >= 2:
                                ticker_latest_day = unique_ticker_dates[-1]
                                ticker_prev_day = unique_ticker_dates[-2]
                                
                                latest_mask = ticker_dates == ticker_latest_day
                                latest_day_data = ticker_series[latest_mask.values]
                                
                                prev_mask = ticker_dates == ticker_prev_day
                                prev_day_data = ticker_series[prev_mask.values]
                                
                                sparkline_data = latest_day_data.tolist() if len(latest_day_data) > 0 else [latest]
                                
                                if len(prev_day_data) > 0:
                                    prev_close = prev_day_data.iloc[-1]
                                    current_close = latest_day_data.iloc[-1] if len(latest_day_data) > 0 else latest
                                    if pd.notna(prev_close) and prev_close != 0:
                                        change = ((current_close / prev_close) - 1) * 100
                                    else:
                                        change = 0.0
                                else:
                                    change = 0.0
                            elif len(unique_ticker_dates) == 1:
                                ticker_day = unique_ticker_dates[0]
                                day_mask = ticker_dates == ticker_day
                                day_data = ticker_series[day_mask.values]
                                sparkline_data = day_data.tolist() if len(day_data) > 0 else [latest]
                                change = 0.0
                            else:
                                sparkline_data = [latest]
                                change = 0.0
                        else:
                            sparkline_data = ticker_series.tolist()[-20:]
                            prev_close = ticker_series.iloc[0] if len(ticker_series) > 1 else latest
                            if pd.notna(prev_close) and prev_close != 0:
                                change = ((latest / prev_close) - 1) * 100
                            else:
                                change = 0.0
                        
                        # Create market item frame - dimmed style for cached/closed markets
                        item_frame = QFrame()
                        if is_cached:
                            # Dimmed style for closed markets (cached data)
                            item_frame.setStyleSheet(f"""
                                QFrame {{
                                    background: {COLORS['bg_card']};
                                    border: 1px solid {COLORS['border_subtle']};
                                    border-radius: 3px;
                                    opacity: 0.7;
                                }}
                            """)
                        else:
                            item_frame.setStyleSheet(f"""
                                QFrame {{
                                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 {COLORS['bg_elevated']}, 
                                        stop:1 {COLORS['bg_card']});
                                    border: 1px solid {COLORS['border_subtle']};
                                    border-radius: 3px;
                                }}
                                QFrame:hover {{
                                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 {COLORS['bg_hover']}, 
                                        stop:1 {COLORS['bg_elevated']});
                                    border-color: {COLORS['accent_dark']};
                                }}
                            """)
                        item_layout = QHBoxLayout(item_frame)
                        item_layout.setContentsMargins(8, 5, 8, 5)
                        item_layout.setSpacing(8)
                        
                        # Name label - dimmed for closed markets
                        display_name = name.upper()
                        if is_cached:
                            name_color = COLORS['text_muted']
                        else:
                            name_color = COLORS['text_primary']
                        
                        name_label = QLabel(display_name)
                        name_label.setStyleSheet(f"color: {name_color}; font-size: 13px; font-weight: 500; background: transparent; border: none;")
                        name_label.setMinimumWidth(85)
                        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                        item_layout.addWidget(name_label, stretch=2)
                        
                        # Format price
                        if category == 'CURRENCIES':
                            price_text = f"{latest:.4f}"
                        elif category == 'YIELDS':
                            price_text = f"{latest:.2f}%"
                        else:
                            price_text = f"{latest:,.2f}"
                        
                        # Price label
                        price_label = QLabel(price_text)
                        price_label.setStyleSheet(f"color: {COLORS['accent_bright']}; font-size: 13px; font-family: 'JetBrains Mono', monospace; font-weight: 600; background: transparent; border: none;")
                        price_label.setMinimumWidth(70)
                        price_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        item_layout.addWidget(price_label, stretch=1)
                        
                        # Change label
                        if change > 0:
                            change_color = COLORS['positive']
                        elif change < 0:
                            change_color = COLORS['negative']
                        else:
                            change_color = COLORS['neutral']
                        
                        change_label = QLabel(f"{change:+.2f}%")
                        change_label.setStyleSheet(f"color: {change_color}; font-size: 13px; font-family: 'JetBrains Mono', monospace; font-weight: 600; background: transparent; border: none;")
                        change_label.setMinimumWidth(55)
                        change_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        item_layout.addWidget(change_label, stretch=1)
                        
                        # Sparkline
                        sparkline = SparklineWidget()
                        sparkline.set_data(sparkline_data, change_color)
                        sparkline.setMinimumWidth(40)
                        sparkline.setMaximumWidth(60)
                        item_layout.addWidget(sparkline, stretch=0)
                        self.market_list_layout.addWidget(item_frame)
                        widgets_added += 1
                        
                        # Track fresh vs cached
                        if is_cached:
                            cached_count += 1
                        else:
                            fresh_count += 1
                        
                        # Add to scatter map data
                        if category in ['AMERICA', 'EUROPE', 'ASIA', 'OCEANIA', 'MIDDLE EAST', 'AFRICA']:
                            scatter_spots.append({
                                'ticker': ticker,
                                'change': change,
                                'name': name,
                                'price': price_text,
                                'color': change_color
                            })
                        
                    except Exception as e:
                        pass
            
            self.market_list_layout.addStretch()
            
            # Update map if available
            if hasattr(self, 'map_widget'):
                map_data = self.build_map_data(scatter_spots)
                self.update_plotly_map(map_data)
            
            # Force layout update
            self.market_list.update()
            self.market_list.repaint()
            
            self.statusBar().showMessage(f"Market data updated: {fresh_count} live, {cached_count} cached (closed markets)")
                    
        except Exception as e:
            import traceback
            print(f"Market watch UI error: {e}")
            traceback.print_exc()
            self.statusBar().showMessage(f"Market data error: {e}")
    
    def _on_market_watch_error(self, error: str):
        """Handle market watch fetch error."""
        # Clear loading indicator
        while self.market_list_layout.count():
            child = self.market_list_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        error_label = QLabel(f"Error: {error[:50]}")
        error_label.setStyleSheet("color: #ff1744; font-size: 11px;")
        self.market_list_layout.addWidget(error_label)
        self.statusBar().showMessage(f"Market data error: {error}")
    
    def update_market_map_spots(self, spots_data):
        """Update the world map with market data."""
        if not PYQTGRAPH_AVAILABLE or not hasattr(self, 'market_scatter'):
            return
        
        NYC_LON = -74.01096
        NYC_LAT = 40.70694
        r = 0.5

        # Market locations (longitude, latitude)
        locations = {
        
            # ======================
            # America
            # ======================
            '^GSPC': (NYC_LON + 0.0000, NYC_LAT + r),       # norr
            '^DJI':  (NYC_LON + r,       NYC_LAT + 0.0000), # öst
            '^NDX':  (NYC_LON + 0.0000, NYC_LAT - r),       # syd
            '^RUT':  (NYC_LON - r,       NYC_LAT + 0.0000), # väst
        
            '^GSPTSE': (-79.3832, 43.6532),      # TSX Composite – Toronto
        
            '^MXX': (-99.1332, 19.4326),         # Mexico – Mexico City
            '^BVSP': (-46.6333, -23.5505),       # Brazil – São Paulo
            '^MERV': (-58.3816, -34.6037),       # Argentina – Buenos Aires
            '^IPSA': (-70.6693, -33.4489),       # Chile – Santiago
            '^COLO-IV': (-74.0721, 4.7110),      # Colombia – Bogotá
        
            # ======================
            # Europe
            # ======================
            '^FTSE': (-0.1278, 51.5074),         # UK – London
            '^FCHI': (2.3522, 48.8566),          # France – Paris
            '^STOXX': (4.0000, 50.0000),         # STOXX Europe 600 (central)
            '^AEX': (4.9041, 52.3676),           # Netherlands – Amsterdam
            '^GDAXI': (8.6821, 50.1109),         # Germany – Frankfurt
            '^SSMI': (8.5417, 47.3769),          # Switzerland – Zurich
            '^ATX': (16.3738, 48.2082),          # Austria – Vienna
            '^IBEX': (-3.7038, 40.4168),         # Spain – Madrid
            'FTSEMIB.MI': (9.1900, 45.4642),     # Italy – Milan
            'PSI20.LS': (-9.1393, 38.7223),      # Portugal – Lisbon
            '^BFX': (4.3517, 50.8503),           # Belgium – Brussels
            '^ISEQ': (-6.2603, 53.3498),         # Ireland – Dublin
            '^OMX': (18.0686, 59.3293),          # Sweden – Stockholm
            'OBX.OL': (10.7522, 59.9139),        # Norway – Oslo
            '^OMXC25': (12.5683, 55.6761),       # Denmark – Copenhagen
            '^OMXH25': (24.9384, 60.1699),       # Finland – Helsinki
            'WIG20.WA': (21.0122, 52.2297),      # Poland – Warsaw
            'XU100.IS': (28.9784, 41.0082),      # Turkey – Istanbul
        
            # ======================
            # Middle East
            # ======================
            '^TA125.TA': (34.7818, 32.0853),     # Israel – Tel Aviv
            '^TASI.SR': (46.6753, 24.7136),      # Saudi Arabia – Riyadh
            'DFMGI.AE': (55.2708, 25.2048),      # UAE – Dubai
            'FADGI.FGI': (54.3773, 24.4539),     # UAE – Abu Dhabi
            '^GNRI.QA': (51.5310, 25.2854),      # Qatar – Doha
        
            # ======================
            # Africa
            # ======================
            '^JN0U.JO': (28.0473, -26.2041),     # South Africa – Johannesburg
            '^CASE30': (31.2357, 30.0444),       # Egypt – Cairo
        
            # ======================
            # Asia
            # ======================
            '^N225': (139.6917, 35.6895),        # Japan – Tokyo
            '^HSI': (114.1694, 22.3193),         # Hong Kong
            '000001.SS': (121.4737, 31.2304),    # China – Shanghai
            '^KS11': (126.9780, 37.5665),        # South Korea – Seoul
            '^TWII': (121.5654, 25.0330),        # Taiwan – Taipei
            '^NSEI': (72.8777, 19.0760),         # India – Mumbai
            '^BSESN': (72.8777, 19.0760),        # India – Mumbai
            '^STI': (103.8198, 1.3521),          # Singapore
            '^JKSE': (106.8456, -6.2088),        # Indonesia – Jakarta
            '^KLSE': (101.6869, 3.1390),         # Malaysia – Kuala Lumpur
            '^SET50.BK': (100.5018, 13.7563),    # Thailand – Bangkok
            '^VNINDEX.VN': (106.6297, 10.8231),  # Vietnam – Ho Chi Minh City
            'PSEI.PS': (120.9842, 14.5995),      # Philippines – Manila
            '000300.SS': (121.4737, 31.2304),    # China – Shanghai (CSI 300)
            '399106.SZ': (114.0579, 22.5431),    # China – Shenzhen
        
            # ======================
            # Oceania
            # ======================
            '^AXJO': (151.2093, -33.8688),       # Australia – Sydney
            '^NZ50': (174.7762, -41.2865),       # New Zealand – Wellington
        }
        
        spots = []
        for item in spots_data:
            ticker = item['ticker']
            change = item['change']
            
            if ticker in locations:
                lon, lat = locations[ticker]
                
                # Color based on change
                if change > 0.5:
                    color = '#00c853'  # Green
                elif change < -0.5:
                    color = '#ff1744'  # Red
                else:
                    color = '#ffc107'  # Yellow
                
                spots.append({
                    'pos': (lon, lat),
                    'brush': pg.mkBrush(color),
                    'size': 8 + min(abs(change) * 2, 8)
                })
        
        if spots:
            self.market_scatter.setData(spots)
    
    def refresh_market_data(self):
        """Refresh volatility data (VIX, VVIX, SKEW, MOVE) asynchronously.
        
        OPTIMERING: Flyttar tung yfinance.download till bakgrundstrad.
        """
        print("[Volatility] refresh_market_data called")
        
        # Don't start if already running - använd säker flagga
        if self._volatility_running:
            self.statusBar().showMessage("Volatility data already updating...")
            print("[Volatility] Already running, skipping")
            return
        
        tickers = ['^VIX', '^VVIX', '^SKEW', '^MOVE']
        
        # Sätt flagga INNAN vi skapar tråden
        self._volatility_running = True
        print(f"[Volatility] Creating thread for {tickers}")
        
        try:
            # Create and start worker
            self._volatility_thread = QThread()
            self._volatility_worker = VolatilityDataWorker(tickers)
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
            import traceback
            traceback.print_exc()
            self._volatility_running = False
    
    def _on_volatility_thread_finished(self):
        """Handle volatility thread completion - clear references and flag."""
        print("[Volatility] Thread finished")
        self._volatility_running = False
        
        # Använd deleteLater() för säker cleanup
        if self._volatility_worker is not None:
            self._volatility_worker.deleteLater()
        if self._volatility_thread is not None:
            self._volatility_thread.deleteLater()
        
        self._volatility_worker = None
        self._volatility_thread = None
        print("[Volatility] References cleared")
    
    def _on_volatility_data_received(self, close):
        """Handle received volatility data - runs on GUI thread (safe)."""
        try:
            if len(close) == 0:
                print("No volatility data returned")
                return
            
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
                        
                        self.vix_card.update_data(vix_val, vix_chg, vix_pct, median, mode, desc)
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
         
                        self.vvix_card.update_data(vvix_val, vvix_chg, vvix_pct, median, mode, desc)
                    else:
                        self.vvix_card.value_label.setText("N/A")
                        self.vvix_card.desc_label.setText("No VVIX data available")
            except Exception as e:
                print(f"VVIX error: {e}")
                self.vvix_card.value_label.setText("Error")
            
            # SKEW
            try:
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

                        self.skew_card.update_data(skew_val, skew_chg, skew_pct, median, mode, desc)
                    else:
                        self.skew_card.value_label.setText("N/A")
                        self.skew_card.desc_label.setText("No SKEW data available")
            except Exception as e:
                print(f"SKEW error: {e}")
                self.skew_card.value_label.setText("Error")
            
            # MOVE (Bond Market Volatility)
            try:
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

                        self.move_card.update_data(move_val, move_chg, move_pct, median, mode, desc)
                    else:
                        self.move_card.value_label.setText("N/A")
                        self.move_card.desc_label.setText("No MOVE data available")
            except Exception as e:
                print(f"MOVE error: {e}")
                self.move_card.value_label.setText("Error")
            
            self.last_updated_label.setText(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            self.statusBar().showMessage("Volatility data updated")
                
        except Exception as e:
            print(f"Volatility data UI error: {e}")
            import traceback
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
            'lookback_period': self.lookback_combo.currentText(),
            'min_half_life': 5,
            'max_half_life': 60,
            'max_adf_pvalue': 0.05,
            'min_correlation': 0.70,
        }
        
        self.worker_thread = QThread()
        self.worker = AnalysisWorker(tickers, config['lookback_period'], config)
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
        
        # If this was a scheduled scan, continue with HMM fitting
        if self._is_scheduled_scan:
            self.statusBar().showMessage("Scheduled scan: Fitting HMM model...")
            # Store flag for HMM completion
            self._scheduled_hmm_pending = True
            self.fit_hmm_model()
    
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
        if hasattr(self, 'viable_count_label'):
            self.viable_count_label.setText(f"({len(df)})")
        
        # Fix #1: Disable updates during batch operations
        self.viable_table.setUpdatesEnabled(False)
        try:
            self.viable_table.setRowCount(len(df))
            # Fix #5: Use itertuples instead of iterrows (10-100x faster)
            for i, row in enumerate(df.itertuples()):
                self.viable_table.setItem(i, 0, QTableWidgetItem(row.pair))
                self.viable_table.setItem(i, 1, QTableWidgetItem(f"{row.half_life_days:.2f}"))
                self.viable_table.setItem(i, 2, QTableWidgetItem(f"{row.eg_pvalue:.4f}"))
                self.viable_table.setItem(i, 3, QTableWidgetItem(f"{row.johansen_trace:.2f}"))
                self.viable_table.setItem(i, 4, QTableWidgetItem(f"{row.hurst_exponent:.2f}"))
                self.viable_table.setItem(i, 5, QTableWidgetItem(f"{row.correlation:.2f}"))
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
        """Handle viable pair selection."""
        selected = self.viable_table.selectedItems()
        if selected:
            pair = self.viable_table.item(selected[0].row(), 0).text()
            self.selected_pair = pair
            
            # Switch to OU tab and select this pair
            self.ou_pair_combo.setCurrentText(pair)
            self.tabs.setCurrentIndex(2)
    
    def update_ou_pair_list(self):
        """Update OU analytics pair dropdown."""
        self.ou_pair_combo.clear()
        
        if self.engine is None:
            return
        
        if self.viable_only_check.isChecked():
            if self.engine.viable_pairs is not None:
                pairs = self.engine.viable_pairs['pair'].tolist()
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
                is_viable = True
            else:
                pair_stats = self.engine.pairs_stats[self.engine.pairs_stats['pair'] == pair].iloc[0]
                is_viable = pair_stats.get('is_viable', False)
            
            # Update metric cards
            self.ou_theta_card.set_value(f"{pair_stats['theta']:.2f}")
            self.ou_mu_card.set_value(f"{ou.mu:.2f}")
            self.ou_halflife_card.set_value(f"{ou.half_life_days():.2f} days")
            
            z_color = "#ff1744" if z > 2 else ("#00c853" if z < -2 else "#ffffff")
            self.ou_zscore_card.set_value(f"{z:.2f}", z_color)
            
            self.ou_hedge_card.set_value(f"{pair_stats['hedge_ratio']:.4f}")
            
            status_text = "✅ VIABLE" if is_viable else "⚠️ NON-VIABLE"
            status_color = "#00c853" if is_viable else "#ffc107"
            self.ou_status_card.set_value(status_text, status_color)
            
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
            self.exp_y_only_card.set_title(f"{y_ticker} (100% of the move)")
            self.exp_y_only_card.set_value(f"{delta_y_full:+.2f} ({delta_y_full_pct:+.2f}%)", y_color)
            
            # X only (100%)
            x_color = "#00c853" if delta_x_full > 0 else "#ff1744"
            self.exp_x_only_card.set_title(f"{x_ticker} (100% of the move)")
            self.exp_x_only_card.set_value(f"{delta_x_full:+.2f} ({delta_x_full_pct:+.2f}%)", x_color)
            
            # Y half (50%)
            y_half_color = "#00c853" if delta_y_half > 0 else "#ff1744"
            self.exp_y_half_card.set_title(f"{y_ticker} (50% of the move)")
            self.exp_y_half_card.set_value(f"{delta_y_half:+.2f} ({delta_y_half_pct:+.2f}%)", y_half_color)
            
            # X half (50%)
            x_half_color = "#00c853" if delta_x_half > 0 else "#ff1744"
            self.exp_x_half_card.set_title(f"{x_ticker} (50% of the move)")
            self.exp_x_half_card.set_value(f"{delta_x_half:+.2f} ({delta_x_half_pct:+.2f}%)", x_half_color)
            
            # Update charts
            if ensure_pyqtgraph():
                self.update_ou_charts(pair, ou, spread, z, pair_stats)
                
        except Exception as e:
            print(f"OU display error: {e}")
    
    def update_ou_charts(self, pair: str, ou, spread: pd.Series, z: float, pair_stats):
        """Update OU analytics charts with dates, crosshairs, and synchronized zoom."""
        tickers = pair.split('/')
        if len(tickers) != 2:
            return
        
        y_ticker, x_ticker = tickers
        hedge_ratio = pair_stats['hedge_ratio']
        intercept = pair_stats.get('intercept', 0)
        
        prices = self.engine.price_data[[y_ticker, x_ticker]].dropna()
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
        self.ou_price_plot.plot(x_axis, y_series, pen=pg.mkPen('#d4a574', width=2), name=f"{y_ticker} (Y)")
        self.ou_price_plot.plot(x_axis, x_adjusted, pen=pg.mkPen('#2196f3', width=2), name=f"β·{x_ticker} + α")
        
        # ===== Z-SCORE =====
        self.ou_zscore_plot.clear()
        zscore = (spread - ou.mu) / ou.eq_std
        self.ou_zscore_plot.plot(x_axis, zscore.values, pen=pg.mkPen('#9c27b0', width=2))
        self.ou_zscore_plot.addLine(y=2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
        self.ou_zscore_plot.addLine(y=-2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
        self.ou_zscore_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))
        
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
                data_series={y_ticker: y_series, f"β·{x_ticker}+α": x_adjusted},
                label_format="{:.2f}"
            )
            
            self.zscore_crosshair = CrosshairManager(
                self.ou_zscore_plot,
                dates=dates,
                data_series={'Z-Score': zscore.values},
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
        days_range = np.arange(0, 70)
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
        
        # ===== ACF PLOT =====
        # The dashed lines near 0 are 95% SIGNIFICANCE BANDS (±1.96/√n).
        # They show the threshold for statistically significant autocorrelation.
        # ACF values WITHIN these bands are NOT significantly different from zero.
        # The orange line is the theoretical OU decay curve: ρ(τ) = exp(-θτ)
        if hasattr(self, 'ou_acf_plot'):
            self.ou_acf_plot.clear()
            max_lag = min(60, len(spread) // 4)
            
            # Empirical ACF
            spread_centered = spread - spread.mean()
            empirical_acf = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    empirical_acf.append(1.0)
                else:
                    acf_val = spread_centered.autocorr(lag)
                    empirical_acf.append(acf_val if not np.isnan(acf_val) else 0)
            
            # Theoretical ACF for OU process: ρ(τ) = exp(-θτ)
            lags = np.arange(max_lag + 1)
            theoretical_acf_ou = np.exp(-ou.theta * lags / 252)  # θ is annualized
            
            # Plot empirical as bars
            bar_width = 0.4
            for i, acf_val in enumerate(empirical_acf):
                color = '#3b82f6' if acf_val >= 0 else '#ef4444'
                bar = pg.BarGraphItem(x=[i], height=[acf_val], width=bar_width, brush=color)
                self.ou_acf_plot.addItem(bar)
            
            # Plot OU theoretical decay as line (orange)
            self.ou_acf_plot.plot(lags, theoretical_acf_ou, pen=pg.mkPen('#d4a574', width=2), name='OU Theoretical')
            
            # 95% Significance bands: values WITHIN these lines are NOT significant
            # Formula: ±1.96/√n (Bartlett's approximation for white noise)
            n = len(spread)
            sig_level = 1.96 / np.sqrt(n)
            self.ou_acf_plot.addLine(y=sig_level, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.ou_acf_plot.addLine(y=-sig_level, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.ou_acf_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))
    
    def update_signals_list(self):
        """Update signals dropdown."""
        self.signal_combo.clear()
        
        if self.engine is None or self.engine.viable_pairs is None:
            self.signal_count_label.setText(f"⚡ 0 viable pairs with |Z| ≥ {SIGNAL_TAB_THRESHOLD}")
            return
        
        signals = []
        # Fix #5: Use itertuples instead of iterrows
        for row in self.engine.viable_pairs.itertuples():
            try:
                ou, spread, z = self.engine.get_pair_ou_params(row.pair, use_raw_data=True)
                if abs(z) >= SIGNAL_TAB_THRESHOLD:
                    signals.append((row.pair, z))
            except:
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
                    self.lev_y_label.setText(f"Leverage {y_ticker}")
                if hasattr(self, 'lev_x_label'):
                    self.lev_x_label.setText(f"Leverage {x_ticker}")
            
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
            
            prices = self.engine.price_data[[y_ticker, x_ticker]].dropna()
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
            self.signal_price_plot.plot(x_axis, y_series, pen=pg.mkPen('#d4a574', width=2), name=f"{y_ticker} (Y)")
            self.signal_price_plot.plot(x_axis, x_adjusted, pen=pg.mkPen('#2196f3', width=2), name=f"β·{x_ticker} + α")
            
            # ===== Z-SCORE =====
            self.signal_zscore_plot.clear()
            zscore = (spread - ou.mu) / ou.eq_std
            self.signal_zscore_plot.plot(x_axis, zscore.values, pen=pg.mkPen('#9c27b0', width=2))
            self.signal_zscore_plot.addLine(y=2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.signal_zscore_plot.addLine(y=-2, pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            self.signal_zscore_plot.addLine(y=0, pen=pg.mkPen('#ffffff', width=1))
            
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
                    data_series={y_ticker: y_series, f"β·{x_ticker}+α": x_adjusted},
                    label_format="{:.2f}"
                )
                
                self.signal_zscore_crosshair = CrosshairManager(
                    self.signal_zscore_plot,
                    dates=dates,
                    data_series={'Z-Score': zscore.values},
                    label_format="{:.2f}"
                )
                
                # Link crosshairs for synchronization
                self.signal_price_crosshair.add_synced_manager(self.signal_zscore_crosshair)
                self.signal_zscore_crosshair.add_synced_manager(self.signal_price_crosshair)
            
        except Exception as e:
            print(f"Signal plot update error: {e}")
    
    def update_mini_futures(self, pair: str, z: float):
        """Update mini-futures suggestions by scraping Morgan Stanley."""
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
        
        # Load ticker mapping
        ticker_to_ms, ms_to_ticker, ticker_to_ms_asset = load_ticker_mapping(force_reload=True)
        
        if not ticker_to_ms:
            self.mini_y_ticker.setText(f"{y_ticker} - MAPPING UNAVAILABLE")
            self.mini_y_name.setText("Ticker mapping file not found")
            self.mini_y_name.setStyleSheet("color: #888888; font-size: 11px; font-weight: 500; background: transparent;")
            self.mini_y_info.setText("")
            
            self.mini_x_ticker.setText(f"{x_ticker} - MAPPING UNAVAILABLE")
            self.mini_x_name.setText("Ticker mapping file not found")
            self.mini_x_name.setStyleSheet("color: #888888; font-size: 11px; font-weight: 500; background: transparent;")
            self.mini_x_info.setText("")
            self._clear_mf_position_sizing()
            return
        
        # Fetch mini futures using DIRECT URL method (10x faster!)
        # If MS_Asset mapping is available, we fetch directly for each ticker
        # instead of scraping all ~3000 products
        self.statusBar().showMessage("Fetching mini futures data from Morgan Stanley...")
        QApplication.processEvents()
        
        # Try direct fetch first (fast), fallback to full scrape if needed
        minifutures_df = None
        if not ticker_to_ms_asset:
            # No MS_Asset mapping - fall back to old slow method
            minifutures_df = scrape_ms_minifutures()
            if minifutures_df.empty:
                self.statusBar().showMessage("Could not fetch mini futures data")
                self._update_mini_future_card_not_found(y_ticker, dir_y, ticker_to_ms, 'y')
                self._update_mini_future_card_not_found(x_ticker, dir_x, ticker_to_ms, 'x')
                self._clear_mf_position_sizing()
                return
        
        # Find best mini futures for each leg
        # Uses direct fetch via MS_Asset if available (fast!) or falls back to minifutures_df
        mf_y = find_best_minifuture(y_ticker, dir_y, ticker_to_ms, minifutures_df, ticker_to_ms_asset)
        mf_x = find_best_minifuture(x_ticker, dir_x, ticker_to_ms, minifutures_df, ticker_to_ms_asset)
        
        # Store for position sizing
        self.current_mini_futures = {'y': mf_y, 'x': mf_x}
        
        # Update Y ticker card
        self._update_mini_future_card(y_ticker, dir_y, mf_y, ticker_to_ms, 'y')
        
        # Update X ticker card
        self._update_mini_future_card(x_ticker, dir_x, mf_x, ticker_to_ms, 'x')
        
        # Calculate and display position sizing
        self._update_mf_position_sizing(pair, y_ticker, x_ticker, dir_y, dir_x, mf_y, mf_x, direction)
        
        self.statusBar().showMessage("Mini futures data loaded")
    
    def _update_mf_position_sizing(self, pair: str, y_ticker: str, x_ticker: str, 
                                    dir_y: str, dir_x: str, mf_y: dict, mf_x: dict, direction: str):
        """Calculate and display mini futures position sizing."""
        try:
            # Get hedge ratio
            pair_stats = self.engine.viable_pairs[self.engine.viable_pairs['pair'] == pair].iloc[0]
            hedge_ratio = pair_stats['hedge_ratio']
            
            # Get notional from spin box
            notional = self.notional_spin.value()
            
            # Calculate position sizing
            mf_positions = calculate_minifuture_position(notional, hedge_ratio, mf_y, mf_x, direction)
            
            # Store for later use
            self.current_mf_positions = mf_positions
            
            # Update Y leg display
            action_y = "LONG POSITION:" if dir_y == "Long" else "SHORT POSITION:"
            color_y = "#00c853" if dir_y == "Long" else "#ff1744"
            
            if mf_y:
                self.mf_pos_y_action.setText(f"{action_y} {y_ticker}")
                self.mf_pos_y_action.setStyleSheet(f"color: {color_y}; font-size: 11px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_y_capital.setText(f"{mf_positions['capital_y']:,.0f} SEK")
                self.mf_pos_y_capital.setStyleSheet(f"color: {color_y}; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_y_info.setText("capital allocation")
                self.mf_pos_y_exposure.setText(f"→ {mf_positions['exposure_y']:,.0f} SEK exposure ({mf_y['leverage']:.2f}x)")
            else:
                self.mf_pos_y_action.setText(f"{y_ticker}")
                self.mf_pos_y_action.setStyleSheet("color: #888888; font-size: 11px; background: transparent; border: none;")
                self.mf_pos_y_capital.setText("N/A")
                self.mf_pos_y_capital.setStyleSheet("color: #555555; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_y_info.setText("No mini future available")
                self.mf_pos_y_exposure.setText("Use regular shares")
            
            # Update X leg display
            action_x = "LONG POSITION:" if dir_x == "Long" else "SHORT POSITION:"
            color_x = "#00c853" if dir_x == "Long" else "#ff1744"
            
            if mf_x:
                self.mf_pos_x_action.setText(f"{action_x} {x_ticker}")
                self.mf_pos_x_action.setStyleSheet(f"color: {color_x}; font-size: 11px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_x_capital.setText(f"{mf_positions['capital_x']:,.0f} SEK")
                self.mf_pos_x_capital.setStyleSheet(f"color: {color_x}; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_x_info.setText("capital allocation")
                self.mf_pos_x_exposure.setText(f"→ {mf_positions['exposure_x']:,.0f} SEK exposure ({mf_x['leverage']:.2f}x)")
            else:
                self.mf_pos_x_action.setText(f"{x_ticker}")
                self.mf_pos_x_action.setStyleSheet("color: #888888; font-size: 11px; background: transparent; border: none;")
                self.mf_pos_x_capital.setText("N/A")
                self.mf_pos_x_capital.setStyleSheet("color: #555555; font-size: 20px; font-weight: 600; background: transparent; border: none;")
                self.mf_pos_x_info.setText("No mini future available")
                self.mf_pos_x_exposure.setText("Use regular shares")
            
            # Update total display
            self.mf_pos_total_capital.setText(f"{mf_positions['total_capital']:,.0f} SEK")
            notional_ratio = mf_positions.get('notional_ratio', hedge_ratio)
            self.mf_pos_total_beta.setText(f"β = {hedge_ratio:.4f} | Notional ratio = {notional_ratio:.2f}")
            self.mf_pos_total_exposure.setText(f"Total exposure: {mf_positions['total_exposure']:,.0f} SEK")
            self.mf_pos_eff_leverage.setText(f"Effective leverage: {mf_positions['effective_leverage']:.2f}x")
            
            # Update minimum capital info
            if mf_y or mf_x:
                if mf_positions['is_at_minimum']:
                    info_text = f"ℹ️Minimum capital required: {mf_positions['min_total_capital']:,.0f} SEK "
                    info_text += f"(binding constraint: {mf_positions['binding_leg']} leg). Position is at minimum size."
                else:
                    info_text = f"ℹ️Minimum capital required: {mf_positions['min_total_capital']:,.0f} SEK "
                    info_text += f"(binding constraint: {mf_positions['binding_leg']} leg). Scaled up to match {notional:,.0f} SEK notional."
                self.mf_min_cap_label.setText(info_text)
            else:
                self.mf_min_cap_label.setText("ℹ️ No mini futures available for position sizing calculation")
                
        except Exception as e:
            print(f"Error calculating MF position sizing: {e}")
            import traceback
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
    
    def _update_mini_future_card(self, ticker: str, direction: str, mf_data: dict, ticker_to_ms: dict, leg: str):
        """Update a mini future/certificate card with data."""
        action = "BUY" if direction == "Long" else "SELL"
        color = "#00c853" if direction == "Long" else "#ff1744"
        
        if leg == 'y':
            ticker_label = self.mini_y_ticker
            name_label = self.mini_y_name
            info_label = self.mini_y_info
            frame = self.mini_y_frame
        else:
            ticker_label = self.mini_x_ticker
            name_label = self.mini_x_name
            info_label = self.mini_x_info
            frame = self.mini_x_frame
        
        if mf_data:
            # Determine product type
            product_type = mf_data.get('product_type', 'Mini Future')
            is_certificate = product_type == 'Certificate'
            
            # Update ticker label with product type
            if is_certificate:
                ticker_label.setText(f"{action} {ticker} BULL/BEAR {direction.upper()}")
            else:
                ticker_label.setText(f"{action} {ticker} MINI {direction.upper()}")
            ticker_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600; background: transparent; border: none;")
            
            name_label.setText(mf_data['name'])
            name_label.setStyleSheet("color: #e8e8e8; font-size: 12px; font-weight: 500; background: transparent; border: none;")
            
            # Build info text based on product type
            info_text = f"Underlying: {mf_data['underlying']}\n"
            
            if is_certificate:
                # Certificates don't have financing level
                info_text += f"Product Type: Bull/Bear Certificate\n"
                if mf_data.get('spot_price'):
                    info_text += f"Spot Price: {mf_data['spot_price']:,.2f}\n"
                daily_lev = mf_data.get('daily_leverage', mf_data.get('leverage', 0))
                info_text += f"Daily Leverage: {daily_lev:.0f}x"
            else:
                # Mini futures have financing level
                if mf_data.get('financing_level'):
                    info_text += f"Financing Level: {mf_data['financing_level']:,.2f}\n"
                if mf_data.get('spot_price'):
                    info_text += f"Spot Price: {mf_data['spot_price']:,.2f}\n"
                info_text += f"Leverage: {mf_data['leverage']:.2f}x"
            
            info_label.setText(info_text)
            
            # Make frame clickable if Avanza link exists
            if mf_data.get('avanza_link'):
                frame.setCursor(Qt.PointingHandCursor)
                frame.setProperty('avanza_link', mf_data['avanza_link'])
                frame.mousePressEvent = lambda e, url=mf_data['avanza_link']: QDesktopServices.openUrl(QUrl(url))
                
                # Add visual hint
                name_label.setText(f"{mf_data['name']} ↗")
            
            frame.setStyleSheet(f"""
                QFrame#metricCard {{
                    background-color: #141414;
                    border: 1px solid {color};
                    border-radius: 4px;
                }}
                QFrame#metricCard:hover {{
                    background-color: #1a1a1a;
                    border-color: #ffaa00;
                }}
            """)
        else:
            self._update_mini_future_card_not_found(ticker, direction, ticker_to_ms, leg)
    
    def _update_mini_future_card_not_found(self, ticker: str, direction: str, ticker_to_ms: dict, leg: str):
        """Update mini future card when not found."""
        ms_name = ticker_to_ms.get(ticker, ticker)
        
        if leg == 'y':
            self.mini_y_ticker.setText(f"{ticker} - NO PRODUCT FOUND")
            self.mini_y_ticker.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {TYPOGRAPHY['body_small']}px; background: transparent;")
            self.mini_y_name.setText(f"No mini future or certificate for {ticker}")
            self.mini_y_name.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {TYPOGRAPHY['body_small']}px; font-weight: 500; background: transparent;")
            self.mini_y_info.setText(f"Searched for: {ms_name} ({direction})")
            self.mini_y_frame.setStyleSheet(f"""
                QFrame#metricCard {{
                    background-color: {COLORS['bg_elevated']};
                    border: none;
                    border-radius: 4px;
                }}
            """)
        else:
            self.mini_x_ticker.setText(f"{ticker} - NO PRODUCT FOUND")
            self.mini_x_ticker.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {TYPOGRAPHY['body_small']}px; background: transparent;")
            self.mini_x_name.setText(f"No mini future or certificate for {ticker}")
            self.mini_x_name.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: {TYPOGRAPHY['body_small']}px; font-weight: 500; background: transparent;")
            self.mini_x_info.setText(f"Searched for: {ms_name} ({direction})")
            self.mini_x_frame.setStyleSheet(f"""
                QFrame#metricCard {{
                    background-color: {COLORS['bg_elevated']};
                    border: none;
                    border-radius: 4px;
                }}
            """)
    
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
                self.buy_action_label.setText(f"BUY {y_ticker}")
                self.sell_action_label.setText(f"SELL {x_ticker}")
                dir_y, dir_x = 'Long', 'Short'
                direction = 'LONG'
            else:  # Short spread: sell Y, buy X
                self.sell_action_label.setText(f"SELL {y_ticker}")
                self.buy_action_label.setText(f"BUY {x_ticker}")
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
            }
            
            self.portfolio.append(position)
            self.update_portfolio_display()
            self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
            
            # Auto-save portfolio
            self._save_and_sync_portfolio()
            
            self.statusBar().showMessage(f"Opened position for {pair} | SL: {strategy.get('stop_z', 3.0):.1f}σ | Win: {strategy.get('win_prob', 0)*100:.0f}%")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open position:\n{e}")
            import traceback
            traceback.print_exc()
    
    def update_portfolio_display(self):
        """Update portfolio positions table."""
        # OPTIMERING: Guard för lazy loading - hoppa över om Portfolio-tab inte är laddad
        if not hasattr(self, 'positions_table') or self.positions_table is None:
            return
        
        self.positions_table.setRowCount(len(self.portfolio))
        
        # Update summary labels
        open_count = len([p for p in self.portfolio if p.get('status', 'OPEN') == 'OPEN'])
        closed_count = len([p for p in self.portfolio if p.get('status', 'OPEN') != 'OPEN'])
        
        self.open_pos_label.setText(f"Open: <span style='color:#22c55e; font-weight:600;'>{open_count}</span>")
        self.closed_pos_label.setText(f"Closed: <span style='color:#888;'>{closed_count}</span>")
        
        # Calculate total P/L
        total_pnl = 0.0
        total_invested = 0.0
        
        for i, pos in enumerate(self.portfolio):
            # PAIR (col 0)
            pair_item = QTableWidgetItem(pos['pair'])
            pair_item.setForeground(QColor("#e8e8e8"))
            self.positions_table.setItem(i, 0, pair_item)
            
            # DIRECTION (col 1)
            dir_item = QTableWidgetItem(pos['direction'])
            dir_item.setForeground(QColor("#e8e8e8"))
            self.positions_table.setItem(i, 1, dir_item)
            
            # CURRENT Z (col 2)
            current_z = pos.get('current_z', pos['entry_z'])
            z_item = QTableWidgetItem(f"{current_z:.2f}")
            z_item.setForeground(QColor("#e8e8e8"))
            self.positions_table.setItem(i, 2, z_item)
            
            # SL (col 3) - Stop Loss Z-score
            sl_z = pos.get('stop_z', 3.0)
            sl_item = QTableWidgetItem(f"{sl_z:.2f}")
            sl_item.setForeground(QColor("#e8e8e8"))
            self.positions_table.setItem(i, 3, sl_item)
            
            # STATUS (col 4)
            status = pos.get('status', 'OPEN')
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor("#e8e8e8"))
            self.positions_table.setItem(i, 4, status_item)
            
            # MINI Y (col 5) - Show NAME (ISIN in tooltip)
            mini_y_isin = pos.get('mini_y_isin', '')
            mini_y_name = pos.get('mini_y_name', '')
            if mini_y_name:
                # Korta ner namnet om det är för långt
                display_name = mini_y_name
                mini_y_item = QTableWidgetItem(display_name)
                mini_y_item.setToolTip(f"{mini_y_name}\nISIN: {mini_y_isin}")
                mini_y_item.setForeground(QColor("#e8e8e8"))  # Vit text
            elif mini_y_isin:
                mini_y_item = QTableWidgetItem(mini_y_isin[:12])
                mini_y_item.setToolTip(f"ISIN: {mini_y_isin}")
                mini_y_item.setForeground(QColor("#e8e8e8"))
            else:
                mini_y_item = QTableWidgetItem("-")
                mini_y_item.setForeground(QColor("#666666"))
            self.positions_table.setItem(i, 5, mini_y_item)
            
            # ENTRY Y (col 6) - Editable entry price
            entry_y_widget = QDoubleSpinBox()
            entry_y_widget.setRange(0, 1000000)
            entry_y_widget.setDecimals(2)
            entry_y_widget.setSingleStep(1.0)
            entry_y_widget.setValue(pos.get('mf_entry_price_y', 0.0))
            entry_y_widget.setFixedHeight(36)
            entry_y_widget.setStyleSheet("""
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
            """)
            entry_y_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_entry_price(idx, 'y', val))
            self.positions_table.setCellWidget(i, 6, entry_y_widget)
            
            # QTY Y (col 7) - Editable quantity
            qty_y_widget = QSpinBox()
            qty_y_widget.setRange(0, 100000)
            qty_y_widget.setValue(pos.get('mf_qty_y', 0))
            qty_y_widget.setFixedHeight(36)
            qty_y_widget.setStyleSheet("""
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
            """)
            qty_y_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_qty(idx, 'y', val))
            self.positions_table.setCellWidget(i, 7, qty_y_widget)
            
            # P/L Y (col 8) - Calculated P/L for Y leg
            entry_price_y = pos.get('mf_entry_price_y', 0.0)
            current_price_y = pos.get('mf_current_price_y', 0.0)
            qty_y = pos.get('mf_qty_y', 0)
            
            if entry_price_y > 0 and qty_y > 0:
                pnl_y = (current_price_y - entry_price_y) * qty_y
                pnl_y_pct = ((current_price_y / entry_price_y) - 1) * 100 if entry_price_y > 0 else 0
                pnl_y_text = f"{pnl_y:+,.0f} ({pnl_y_pct:+.1f}%)"
                pnl_y_color = "#22c55e" if pnl_y >= 0 else "#ff1744"
                total_pnl += pnl_y
                total_invested += entry_price_y * qty_y
            else:
                pnl_y_text = "-"
                pnl_y_color = "#666666"
            
            pnl_y_item = QTableWidgetItem(pnl_y_text)
            pnl_y_item.setForeground(QColor(pnl_y_color))
            self.positions_table.setItem(i, 8, pnl_y_item)
            
            # MINI X (col 9) - Show NAME (ISIN in tooltip)
            mini_x_isin = pos.get('mini_x_isin', '')
            mini_x_name = pos.get('mini_x_name', '')
            if mini_x_name:
                # Korta ner namnet om det är för långt
                display_name = mini_x_name
                mini_x_item = QTableWidgetItem(display_name)
                mini_x_item.setToolTip(f"{mini_x_name}\nISIN: {mini_x_isin}")
                mini_x_item.setForeground(QColor("#e8e8e8"))  # Vit text
            elif mini_x_isin:
                mini_x_item = QTableWidgetItem(mini_x_isin[:12])
                mini_x_item.setToolTip(f"ISIN: {mini_x_isin}")
                mini_x_item.setForeground(QColor("#e8e8e8"))
            else:
                mini_x_item = QTableWidgetItem("-")
                mini_x_item.setForeground(QColor("#666666"))
            self.positions_table.setItem(i, 9, mini_x_item)
            
            # ENTRY X (col 10) - Editable entry price
            entry_x_widget = QDoubleSpinBox()
            entry_x_widget.setRange(0, 1000000)
            entry_x_widget.setDecimals(2)
            entry_x_widget.setSingleStep(1.0)
            entry_x_widget.setValue(pos.get('mf_entry_price_x', 0.0))
            entry_x_widget.setFixedHeight(36)
            entry_x_widget.setStyleSheet("""
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
            """)
            entry_x_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_entry_price(idx, 'x', val))
            self.positions_table.setCellWidget(i, 10, entry_x_widget)
            
            # QTY X (col 11) - Editable quantity
            qty_x_widget = QSpinBox()
            qty_x_widget.setRange(0, 100000)
            qty_x_widget.setValue(pos.get('mf_qty_x', 0))
            qty_x_widget.setFixedHeight(36)
            qty_x_widget.setStyleSheet("""
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
            """)
            qty_x_widget.valueChanged.connect(lambda val, idx=i: self._update_mf_qty(idx, 'x', val))
            self.positions_table.setCellWidget(i, 11, qty_x_widget)
            
            # P/L X (col 12) - Calculated P/L for X leg
            entry_price_x = pos.get('mf_entry_price_x', 0.0)
            current_price_x = pos.get('mf_current_price_x', 0.0)
            qty_x = pos.get('mf_qty_x', 0)
            
            if entry_price_x > 0 and qty_x > 0:
                pnl_x = (current_price_x - entry_price_x) * qty_x
                pnl_x_pct = ((current_price_x / entry_price_x) - 1) * 100 if entry_price_x > 0 else 0
                pnl_x_text = f"{pnl_x:+,.0f} ({pnl_x_pct:+.1f}%)"
                pnl_x_color = "#22c55e" if pnl_x >= 0 else "#ff1744"
                total_pnl += pnl_x
                total_invested += entry_price_x * qty_x
            else:
                pnl_x_text = "-"
                pnl_x_color = "#666666"
            
            pnl_x_item = QTableWidgetItem(pnl_x_text)
            pnl_x_item.setForeground(QColor(pnl_x_color))
            self.positions_table.setItem(i, 12, pnl_x_item)
            
            # CLOSE button (col 13)
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
            self.positions_table.setCellWidget(i, 13, close_btn)
        
        # Update total P/L summary
        if total_invested > 0:
            total_pnl_pct = (total_pnl / total_invested) * 100
            pnl_color = "#22c55e" if total_pnl >= 0 else "#ff1744"
            self.unrealized_pnl_label.setText(f"Unrealized: <span style='color:{pnl_color};'>{total_pnl:+,.0f} SEK ({total_pnl_pct:+.2f}%)</span>")
        else:
            self.unrealized_pnl_label.setText("Unrealized: <span style='color:#888;'>+0.00%</span>")
    
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
            pnl_y_color = "#22c55e" if pnl_y >= 0 else "#ff1744"
        else:
            pnl_y_text = "-"
            pnl_y_color = "#666666"
        
        pnl_y_item = QTableWidgetItem(pnl_y_text)
        pnl_y_item.setForeground(QColor(pnl_y_color))
        self.positions_table.setItem(idx, 8, pnl_y_item)
        
        # P/L X
        entry_price_x = pos.get('mf_entry_price_x', 0.0)
        current_price_x = pos.get('mf_current_price_x', 0.0)
        qty_x = pos.get('mf_qty_x', 0)
        
        if entry_price_x > 0 and qty_x > 0 and current_price_x > 0:
            pnl_x = (current_price_x - entry_price_x) * qty_x
            pnl_x_pct = ((current_price_x / entry_price_x) - 1) * 100
            pnl_x_text = f"{pnl_x:+,.0f} ({pnl_x_pct:+.1f}%)"
            pnl_x_color = "#22c55e" if pnl_x >= 0 else "#ff1744"
        else:
            pnl_x_text = "-"
            pnl_x_color = "#666666"
        
        pnl_x_item = QTableWidgetItem(pnl_x_text)
        pnl_x_item.setForeground(QColor(pnl_x_color))
        self.positions_table.setItem(idx, 12, pnl_x_item)
        
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
            if pos.get('status', 'OPEN') != 'OPEN':
                continue
                
            try:
                pair = pos['pair']
                ou, spread, current_z = self.engine.get_pair_ou_params(pair, use_raw_data=True)
                
                # Store previous z for comparison
                pos['previous_z'] = pos.get('current_z', pos['entry_z'])
                pos['current_z'] = current_z
                
                # Check TP/SL conditions
                tp_z = pos.get('exit_z', 0.0)
                sl_z = pos.get('stop_z', 3.0)
                entry_z = pos['entry_z']
                
                if pos['direction'] == 'LONG':
                    # Long spread: entered at negative Z, want Z to rise toward 0
                    # TP when Z crosses above exit_z (closer to 0)
                    # SL when Z goes more negative (below -stop_z)
                    if current_z >= tp_z:
                        pos['status'] = 'TP HIT'
                    elif current_z <= -abs(sl_z):
                        pos['status'] = 'SL HIT'
                else:
                    # Short spread: entered at positive Z, want Z to fall toward 0
                    # TP when Z crosses below exit_z (closer to 0)
                    # SL when Z goes more positive (above stop_z)
                    if current_z <= tp_z:
                        pos['status'] = 'TP HIT'
                    elif current_z >= abs(sl_z):
                        pos['status'] = 'SL HIT'
                
                updated_count += 1
                
            except Exception as e:
                print(f"Error updating Z for {pos['pair']}: {e}")
        
        self.update_portfolio_display()
        
        # Auto-save portfolio with updated Z-scores and status
        self._save_and_sync_portfolio()
        
        self.statusBar().showMessage(f"Updated Z-scores for {updated_count} positions")
    
    def close_position(self, index: int):
        """Close a position with P&L calculation."""
        if 0 <= index < len(self.portfolio):
            pos = self.portfolio[index]
            
            # Calculate Z change
            entry_z = pos['entry_z']
            current_z = pos.get('current_z', entry_z)
            z_change = current_z - entry_z
            
            # For LONG: profit when Z increases (becomes less negative)
            # For SHORT: profit when Z decreases
            if pos['direction'] == 'LONG':
                profit = z_change > 0
            else:
                profit = z_change < 0
            
            status = pos.get('status', 'MANUAL CLOSE')
            
            # Ask for confirmation with details
            msg = f"Close position for {pos['pair']}?\n\n"
            msg += f"Direction: {pos['direction']}\n"
            msg += f"Entry Z: {entry_z:.2f}\n"
            msg += f"Current Z: {current_z:.2f}\n"
            msg += f"Z Change: {z_change:+.2f}\n"
            msg += f"Status: {status}"
            
            reply = QMessageBox.question(self, "Close Position", msg,
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                pos = self.portfolio.pop(index)
                self.update_portfolio_display()
                self._update_metric_value(self.positions_metric, str(len(self.portfolio)))
                
                # Auto-save portfolio
                self._save_and_sync_portfolio()
                
                result = "PROFIT" if profit else "LOSS"
                self.statusBar().showMessage(f"Closed {pos['pair']} - {result} (Z: {entry_z:.2f} → {current_z:.2f})")
    
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

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
        
    window = PairsTradingTerminal()
    window.show()

    # Check for updates in background (non-blocking)
    try:
        from auto_updater import AutoUpdater
        window._updater = AutoUpdater(window)
        window._updater.check_for_updates(silent=True)
    except Exception as e:
        print(f"Auto-updater disabled: {e}")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
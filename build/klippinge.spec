# -*- mode: python ; coding: utf-8 -*-
"""
Klippinge Investment Trading Terminal - PyInstaller Spec File
=============================================================

Build with:
    pyinstaller build/klippinge.spec

Or use the build script:
    python build.py
"""

import os
import sys

# ── Path Resolution ──────────────────────────────────────────────────────────
# SPECPATH is set by PyInstaller to the directory containing this .spec file.
# Since the spec is in build/, we go one level up to reach the project root.
PROJECT_ROOT = os.path.normpath(os.path.join(SPECPATH, '..'))

def proj(relative_path):
    """Resolve a path relative to the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)

# ── Configuration ────────────────────────────────────────────────────────────

APP_NAME = "KlippingeTrading"
MAIN_SCRIPT = proj("dashboard_PyQt5.py")
ICON_FILE = proj("logo.ico")
VERSION = "1.0.0"

# ── Analysis ─────────────────────────────────────────────────────────────────

block_cipher = None

# Data files to bundle (source, destination_folder)
datas = [
    # CSV ticker data
    (proj('index_tickers.csv'), '.'),
    (proj('underliggande_matchade_tickers.csv'), '.'),
    
    # Configuration files
    (proj('notification_config.json'), '.'),
    (proj('ib_ticker_mapping.json'), '.'),
    
    # Icons & assets
    (proj('logo.ico'), '.'),
    (proj('assets/styles.css'), 'assets'),
    
    # Trading data defaults (for first-run initialization)
    (proj('Trading/index_tickers.csv'), 'Trading'),
    (proj('Trading/portfolio_positions.json'), 'Trading'),
    (proj('Trading/benchmark_cache.json'), 'Trading'),
]

# Only include data files that exist
datas = [(src, dst) for src, dst in datas if os.path.exists(src)]

# Hidden imports that PyInstaller may miss
hidden_imports = [
    # Core engine modules
    'pairs_engine',
    'regime_hmm',
    'portfolio_history',
    'scrape_prices_MS',
    'app_config',
    'auto_updater',
    
    # PyQt5
    'PyQt5',
    'PyQt5.QtWidgets',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.sip',
    'PyQt5.QtWebEngineWidgets',
    
    # pyqtgraph (lazy loaded in dashboard)
    'pyqtgraph',
    'pyqtgraph.graphicsItems',
    'pyqtgraph.graphicsItems.ScatterPlotItem',
    'pyqtgraph.graphicsItems.PlotDataItem',
    
    # Scientific computing
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    'pandas',
    'scipy',
    'scipy.stats',
    'scipy.special',
    'scipy.optimize',
    'scipy.integrate',
    'statsmodels',
    'statsmodels.api',
    'statsmodels.tsa.stattools',
    'statsmodels.tsa.vector_ar.vecm',
    
    # Data fetching
    'yfinance',
    'requests',
    'bs4',
    'urllib3',
    
    # Standard library that may be missed
    'zoneinfo',
    'pickle',
    'json',
    'csv',
    'socket',
    'concurrent.futures',
    'multiprocessing',
    'dataclasses',
    
    # Optional but commonly used
    'openpyxl',
]

# Packages to exclude (reduce size)
excludes = [
    # Heavy optional dependencies not needed
    'tkinter',
    'matplotlib',
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'pytest',
    'setuptools',
    
    # Selenium (only for ticker scraping, not runtime)
    'selenium',
    
    # PyMC (very heavy, optional)
    'pymc',
    'theano',
    'aesara',
    'pytensor',
    
    # Other large packages
    'torch',
    'tensorflow',
    'sklearn',
]

a = Analysis(
    [MAIN_SCRIPT],
    pathex=[PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


# ── Remove unnecessary files to reduce size ──────────────────────────────────

# Filter out test files and docs
a.datas = [d for d in a.datas if not any(x in d[0].lower() for x in [
    'test', 'tests', '__pycache__', '.pyc', 'example', 'doc',
])]


# ── Bundle ───────────────────────────────────────────────────────────────────

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],                  # NOT one-file (use COLLECT for directory mode)
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON_FILE if os.path.exists(ICON_FILE) else None,
    
    # Windows version info
    version_info=None,   # Set via version_info.txt if needed
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

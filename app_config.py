"""
Klippinge Investment Trading Terminal - Application Configuration
=================================================================

Centralized configuration for portable deployment.
Replaces all hardcoded Google Drive paths with platform-aware locations.

Data locations:
    - App data (bundled):  Inside the install/exe directory
    - User data (mutable): %APPDATA%/KlippingeTrading/ (Windows)
                           ~/.klippinge-trading/ (Linux/Mac)
"""

import os
import sys
import json
import shutil
import platform
from pathlib import Path

# ── Version ──────────────────────────────────────────────────────────────────
APP_NAME = "Klippinge Investment Trading Terminal"
APP_VERSION = "1.1.0"
APP_AUTHOR = "Klippinge Investment"
GITHUB_REPO = "andreasklippinge/klippinge-trading"  # ← Ändra detta!


# ── Path Resolution ──────────────────────────────────────────────────────────

def _is_frozen() -> bool:
    """Check if running as a PyInstaller bundle."""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')


def get_app_dir() -> Path:
    """
    Get the application directory (where bundled resources live).
    
    - Frozen (PyInstaller): sys._MEIPASS (temp extraction dir)
    - Development: Directory containing this file
    """
    if _is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent.resolve()


def get_install_dir() -> Path:
    """
    Get the actual install directory (where the .exe lives).
    
    - Frozen: Directory containing the .exe
    - Development: Same as app_dir
    """
    if _is_frozen():
        return Path(sys.executable).parent.resolve()
    return get_app_dir()


def get_user_data_dir() -> Path:
    """
    Get the user data directory for mutable files (portfolio, cache, configs).
    
    Windows: %APPDATA%/KlippingeTrading/
    Linux:   ~/.klippinge-trading/
    macOS:   ~/Library/Application Support/KlippingeTrading/
    """
    system = platform.system()
    
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        data_dir = base / "KlippingeTrading"
    elif system == "Darwin":
        data_dir = Path.home() / "Library" / "Application Support" / "KlippingeTrading"
    else:
        data_dir = Path.home() / ".klippinge-trading"
    
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_trading_data_dir() -> Path:
    """Subdirectory for trading-specific data (portfolio, engine cache)."""
    d = get_user_data_dir() / "Trading"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    """Subdirectory for log files."""
    d = get_user_data_dir() / "Logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Resource Paths (bundled with app, read-only) ────────────────────────────

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to a bundled resource file.
    Works both in development and when frozen with PyInstaller.
    """
    return str(get_app_dir() / relative_path)


# ── Concrete File Paths ─────────────────────────────────────────────────────

class Paths:
    """All application file paths in one place."""
    
    # ── Bundled resources (read-only) ──
    @staticmethod
    def logo_icon() -> str:
        return resource_path("logo.ico")
    
    @staticmethod
    def default_tickers_csv() -> str:
        """Default index_tickers.csv bundled with the app."""
        return resource_path("index_tickers.csv")
    
    @staticmethod
    def default_matched_tickers_csv() -> str:
        """Default underliggande_matchade_tickers.csv bundled with the app."""
        return resource_path("underliggande_matchade_tickers.csv")
    
    @staticmethod
    def borsdata_xlsx() -> str:
        """Borsdata reference spreadsheet."""
        return resource_path("Borsdata_2025-05-30.xlsx")
    
    @staticmethod
    def styles_css() -> str:
        return resource_path("assets/styles.css")
    
    # ── User data (read/write) ──
    @staticmethod
    def portfolio_file() -> str:
        return str(get_trading_data_dir() / "portfolio_positions.json")
    
    @staticmethod
    def engine_cache_file() -> str:
        return str(get_trading_data_dir() / "engine_cache.pkl")
    
    @staticmethod
    def portfolio_history_file() -> str:
        return str(get_trading_data_dir() / "portfolio_history.json")
    
    @staticmethod
    def benchmark_cache_file() -> str:
        return str(get_trading_data_dir() / "benchmark_cache.json")
    
    @staticmethod
    def news_cache_file() -> str:
        return str(get_user_data_dir() / "news_cache.json")
    
    @staticmethod
    def regime_cache_file() -> str:
        return str(get_user_data_dir() / "regime_cache_weekly.pkl")
    
    @staticmethod
    def notification_config_file() -> str:
        return str(get_user_data_dir() / "notification_config.json")
    
    @staticmethod
    def ib_ticker_mapping_file() -> str:
        return str(get_user_data_dir() / "ib_ticker_mapping.json")
    
    @staticmethod
    def user_tickers_csv() -> str:
        """User's custom tickers CSV (if they've loaded one)."""
        return str(get_user_data_dir() / "user_tickers.csv")
    
    @staticmethod
    def scheduled_csv_path() -> str:
        """
        CSV used for scheduled scans.
        Checks user data first, falls back to bundled default.
        """
        user_csv = get_user_data_dir() / "user_tickers.csv"
        if user_csv.exists():
            return str(user_csv)
        
        trading_csv = get_trading_data_dir() / "index_tickers.csv"
        if trading_csv.exists():
            return str(trading_csv)
        
        return Paths.default_tickers_csv()


# ── Discord Configuration ────────────────────────────────────────────────────

def get_discord_webhook_url() -> str:
    """
    Load Discord webhook URL from config file instead of hardcoding.
    Users configure this in Settings or via notification_config.json.
    
    SECURITY: Never hardcode webhook URLs in distributed code!
    """
    config_path = Paths.notification_config_file()
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("discord_webhook_url", "")
    except (json.JSONDecodeError, IOError):
        pass
    return ""


def save_discord_webhook_url(url: str) -> None:
    """Save Discord webhook URL to config."""
    config_path = Paths.notification_config_file()
    config = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    
    config["discord_webhook_url"] = url
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


# ── First-Run Setup ─────────────────────────────────────────────────────────

def initialize_user_data():
    """
    Copy default data files to user data directory on first run.
    Only copies files that don't already exist (preserves user modifications).
    """
    app_dir = get_app_dir()
    user_dir = get_user_data_dir()
    trading_dir = get_trading_data_dir()
    
    # Files to copy from app bundle → user Trading data dir
    trading_defaults = [
        "index_tickers.csv",
        "portfolio_positions.json",
        "engine_cache.pkl",
        "portfolio_history.json",
        "benchmark_cache.json",
    ]
    
    for filename in trading_defaults:
        src = app_dir / "Trading" / filename
        if not src.exists():
            src = app_dir / filename
        dst = trading_dir / filename
        
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Initialized: {dst}")
    
    # Files to copy from app bundle → user data root
    root_defaults = [
        "notification_config.json",
        "ib_ticker_mapping.json",
        "news_cache.json",
    ]
    
    for filename in root_defaults:
        src = app_dir / filename
        dst = user_dir / filename
        
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Initialized: {dst}")
    
    print(f"User data directory: {user_dir}")


# ── Ticker File Resolution ──────────────────────────────────────────────────

def find_ticker_csv() -> str:
    """
    Find the best available ticker CSV file.
    Priority:
        1. User's custom file in user data dir
        2. Trading subdirectory copy
        3. Bundled default
    """
    candidates = [
        get_user_data_dir() / "user_tickers.csv",
        get_trading_data_dir() / "index_tickers.csv",
        get_app_dir() / "underliggande_matchade_tickers.csv",
        get_app_dir() / "index_tickers.csv",
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    return Paths.default_tickers_csv()


def find_matched_tickers_csv() -> str:
    """
    Find the underliggande_matchade_tickers.csv file.
    Priority:
        1. User data directory
        2. Trading subdirectory
        3. Bundled default
    """
    candidates = [
        get_user_data_dir() / "underliggande_matchade_tickers.csv",
        get_trading_data_dir() / "underliggande_matchade_tickers.csv",
    ]
    for path in candidates:
        if path.exists():
            return str(path)

    return Paths.default_matched_tickers_csv()


# ── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging():
    """Configure file-based logging for the distributed app."""
    import logging
    from datetime import datetime
    
    log_file = get_logs_dir() / f"terminal_{datetime.now():%Y-%m-%d}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    return logging.getLogger("klippinge")


# ── Print Configuration (for debugging) ─────────────────────────────────────

def print_config():
    """Print current configuration for debugging."""
    print(f"\n{'='*60}")
    print(f"  {APP_NAME} v{APP_VERSION}")
    print(f"{'='*60}")
    print(f"  Frozen:       {_is_frozen()}")
    print(f"  App dir:      {get_app_dir()}")
    print(f"  Install dir:  {get_install_dir()}")
    print(f"  User data:    {get_user_data_dir()}")
    print(f"  Trading data: {get_trading_data_dir()}")
    print(f"  Logs:         {get_logs_dir()}")
    print(f"  Platform:     {platform.system()} {platform.release()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_config()
    initialize_user_data()

"""
Klippinge Trading Terminal - Build Script
==========================================

Builds the distributable .exe package using PyInstaller.

Usage:
    python build.py              # Full build
    python build.py --clean      # Clean previous builds first
    python build.py --onefile    # Single .exe (slower startup, larger file)
    python build.py --installer  # Also build Inno Setup installer (requires ISCC)

Requirements:
    pip install pyinstaller
"""

import os
import sys
import shutil
import subprocess
import argparse
import time
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────

APP_NAME = "KlippingeTrading"
VERSION = "1.0.0"
SPEC_FILE = "build/klippinge.spec"
INNO_SCRIPT = "build/installer.iss"

# Files required in the source directory before building
REQUIRED_FILES = [
    "dashboard_PyQt5.py",
    "pairs_engine.py",
    "regime_hmm.py",
    "portfolio_history.py",
    "scrape_prices_MS.py",
    "app_config.py",
    "auto_updater.py",
    "logo.ico",
    "index_tickers.csv",
]


def print_header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_step(step: int, msg: str):
    print(f"  [{step}] {msg}")


def check_prerequisites():
    """Verify all build prerequisites are met."""
    print_header("Checking Prerequisites")
    
    errors = []
    
    # Check Python version
    print_step(1, f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ required")
    
    # Check PyInstaller
    try:
        import PyInstaller
        print_step(2, f"PyInstaller {PyInstaller.__version__}")
    except ImportError:
        errors.append("PyInstaller not installed. Run: pip install pyinstaller")
    
    # Check required source files
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        errors.append(f"Missing source files: {', '.join(missing)}")
        print_step(3, f"Source files: {len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)} FAIL")
        for f in missing:
            print(f"       Missing: {f}")
    else:
        print_step(3, f"Source files: {len(REQUIRED_FILES)}/{len(REQUIRED_FILES)} OK")
    
    # Check key dependencies are installed
    deps_ok = True
    for dep in ['PyQt5', 'numpy', 'pandas', 'scipy', 'statsmodels', 'yfinance', 'pyqtgraph']:
        try:
            __import__(dep)
        except ImportError:
            errors.append(f"Missing dependency: {dep}")
            deps_ok = False
    
    print_step(4, f"Dependencies: {'OK' if deps_ok else 'FAIL'}")
    
    if errors:
        print(f"\nFAIL Prerequisites check failed:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)
    
    print(f"\n  OK All prerequisites met!")


def clean_build():
    """Remove previous build artifacts."""
    print_header("Cleaning Previous Builds")
    
    dirs_to_clean = ['build/temp', 'dist', '__pycache__']
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Removed: {d}/")
    
    # Clean .pyc files
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.pyc'):
                os.remove(os.path.join(root, f))


def build_exe(one_file: bool = False):
    """Run PyInstaller to create the executable."""
    print_header(f"Building {'Single File' if one_file else 'Directory'} Distribution")
    
    start = time.time()
    
    if one_file:
        # One-file mode: everything packed into a single .exe
        # Slower startup but easier to distribute
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--noconfirm',
            '--clean',
            '--name', APP_NAME,
            '--windowed',           # No console
            '--onefile',
            '--icon', 'logo.ico',
            
            # Data files
            '--add-data', 'index_tickers.csv;.',
            '--add-data', 'underliggande_matchade_tickers.csv;.',
            '--add-data', 'notification_config.json;.',
            '--add-data', 'logo.ico;.',
            '--add-data', 'assets/styles.css;assets',
            '--add-data', 'Trading/index_tickers.csv;Trading',
            '--add-data', 'Trading/portfolio_positions.json;Trading',
            '--add-data', 'Trading/benchmark_cache.json;Trading',
            
            # Hidden imports
            '--hidden-import', 'pairs_engine',
            '--hidden-import', 'regime_hmm',
            '--hidden-import', 'portfolio_history',
            '--hidden-import', 'scrape_prices_MS',
            '--hidden-import', 'app_config',
            '--hidden-import', 'auto_updater',
            '--hidden-import', 'pyqtgraph',
            '--hidden-import', 'scipy.special',
            '--hidden-import', 'statsmodels.tsa.stattools',
            '--hidden-import', 'statsmodels.tsa.vector_ar.vecm',
            
            # Excludes
            '--exclude-module', 'tkinter',
            '--exclude-module', 'matplotlib',
            '--exclude-module', 'selenium',
            '--exclude-module', 'pymc',
            '--exclude-module', 'IPython',
            
            'dashboard_PyQt5.py',
        ]
    else:
        # Directory mode: use spec file (recommended)
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--noconfirm',
            '--clean',
            SPEC_FILE,
        ]
    
    print(f"  Running: {' '.join(cmd[:6])}...")
    print(f"  (this may take 3-10 minutes)\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n  FAIL Build failed after {elapsed:.0f}s")
        sys.exit(1)
    
    print(f"\n  OK Build completed in {elapsed:.0f}s")
    
    # Report output
    dist_dir = Path('dist') / APP_NAME
    if one_file:
        exe_path = Path('dist') / f'{APP_NAME}.exe'
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print(f"  Output: {exe_path} ({size_mb:.1f} MB)")
    elif dist_dir.exists():
        total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"  Output: {dist_dir}/ ({size_mb:.1f} MB)")


def build_installer():
    """Build Windows installer using Inno Setup (if available)."""
    print_header("Building Windows Installer")
    
    # Check for Inno Setup
    iscc_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        shutil.which("ISCC"),
    ]
    
    iscc = None
    for p in iscc_paths:
        if p and os.path.exists(p):
            iscc = p
            break
    
    if not iscc:
        print("  [WARN] Inno Setup not found. Skipping installer build.")
        print("  Install from: https://jrsoftware.org/isinfo.php")
        return
    
    if not os.path.exists(INNO_SCRIPT):
        print(f"  [WARN] Inno Setup script not found: {INNO_SCRIPT}")
        return
    
    cmd = [iscc, INNO_SCRIPT]
    print(f"  Running: ISCC {INNO_SCRIPT}")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n  OK Installer created in dist/")
    else:
        print(f"\n  FAIL Installer build failed")


def create_portable_zip():
    """Create a portable .zip distribution (no installer needed)."""
    print_header("Creating Portable ZIP")
    
    dist_dir = Path('dist') / APP_NAME
    if not dist_dir.exists():
        print("  FAIL Build directory not found. Run build first.")
        return
    
    zip_name = f"{APP_NAME}-v{VERSION}-portable-win64"
    zip_path = Path('dist') / zip_name
    
    print(f"  Creating: {zip_path}.zip")
    shutil.make_archive(str(zip_path), 'zip', 'dist', APP_NAME)
    
    size_mb = (Path(str(zip_path) + '.zip')).stat().st_size / (1024 * 1024)
    print(f"  OK Created: {zip_path}.zip ({size_mb:.1f} MB)")


def post_build_report():
    """Print summary of build outputs."""
    print_header("Build Summary")
    
    dist = Path('dist')
    if not dist.exists():
        return
    
    print("  Output files:")
    for item in sorted(dist.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"    [ZIP] {item.name}  ({size_mb:.1f} MB)")
        elif item.is_dir():
            total = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            size_mb = total / (1024 * 1024)
            count = sum(1 for f in item.rglob('*') if f.is_file())
            print(f"    [DIR] {item.name}/  ({count} files, {size_mb:.1f} MB)")
    
    print(f"\n  Next steps:")
    print(f"    1. Test the .exe: dist/{APP_NAME}/{APP_NAME}.exe")
    print(f"    2. Upload to GitHub Releases for auto-update")
    print(f"    3. Share the ZIP or installer with users")


def main():
    parser = argparse.ArgumentParser(description=f"Build {APP_NAME}")
    parser.add_argument('--clean', action='store_true', help='Clean previous builds')
    parser.add_argument('--onefile', action='store_true', help='Build single .exe (slower startup)')
    parser.add_argument('--installer', action='store_true', help='Also build Inno Setup installer')
    parser.add_argument('--zip', action='store_true', help='Create portable ZIP')
    parser.add_argument('--skip-checks', action='store_true', help='Skip prerequisite checks')
    args = parser.parse_args()
    
    print_header(f"Building {APP_NAME} v{VERSION}")
    
    if args.clean:
        clean_build()
    
    if not args.skip_checks:
        check_prerequisites()
    
    build_exe(one_file=args.onefile)
    
    if args.zip:
        create_portable_zip()
    
    if args.installer:
        build_installer()
    
    post_build_report()


if __name__ == '__main__':
    main()

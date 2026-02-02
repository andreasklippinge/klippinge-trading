# Klippinge Trading Terminal

A professional pairs trading and portfolio management terminal built with PyQt5.

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![License](https://img.shields.io/badge/License-Proprietary-red)

## Features

- **Pairs Trading Analysis** — Cointegration testing, Ornstein-Uhlenbeck modeling, z-score signals
- **HMM Regime Detection** — Bayesian Hidden Markov Models for market regime identification
- **Portfolio Management** — Position tracking, daily snapshots, benchmark analysis vs S&P 500
- **Automated Scanning** — Scheduled pair screening across 590+ Swedish stocks
- **Morgan Stanley Mini Futures** — Live pricing and leveraged position sizing
- **Interactive Brokers Integration** — Live streaming data for Swedish equities and OMX futures
- **Discord Notifications** — Automated alerts for trade signals and scan results
- **World Market Overview** — Real-time global market dashboard with volatility metrics

## Installation

### Option 1: Installer (recommended)
Download the latest **Setup.exe** from [Releases](https://github.com/andreasklippinge/klippinge-trading/releases) and run it.

### Option 2: Portable
Download the **portable ZIP** from [Releases](https://github.com/andreasklippinge/klippinge-trading/releases), extract and run `KlippingeTrading.exe`.

### Option 3: From source
```bash
git clone https://github.com/andreasklippinge/klippinge-trading.git
cd klippinge-trading
pip install -r requirements.txt
python dashboard_PyQt5.py
```

## Project Structure

```
klippinge-trading/
├── dashboard_PyQt5.py         # Main application (PyQt5 GUI)
├── pairs_engine.py            # Statistical arbitrage engine (OU, cointegration)
├── regime_hmm.py              # Bayesian HMM regime detection
├── portfolio_history.py       # Portfolio snapshot tracking
├── scrape_prices_MS.py        # Morgan Stanley live pricing
├── app_config.py              # Portable paths and configuration
├── auto_updater.py            # GitHub release auto-updater
├── build.py                   # PyInstaller build script
├── Trading/                   # Data files (tickers, mappings)
├── assets/                    # CSS styles
├── build/                     # Build configuration (spec, installer)
└── .github/workflows/         # CI/CD (auto-build on tag)
```

## Building

```bash
# Build .exe
python build.py

# Build .exe + portable ZIP
python build.py --zip
```

Requires Python 3.12+ and [Inno Setup 6](https://jrsoftware.org/isdl.php) for the installer.

## Configuration

User data is stored in `%APPDATA%\KlippingeTrading\`:
- `portfolio_positions.json` — Active positions
- `portfolio_history.json` — Historical snapshots
- `notification_config.json` — Discord webhook settings

## Release Workflow

```bash
git add .
git commit -m "Description of changes"
git tag v1.1.0
git push origin main --tags
# GitHub Actions builds and publishes automatically
```

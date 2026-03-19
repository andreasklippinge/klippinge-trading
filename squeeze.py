"""
TTM Squeeze Scanner
===================
Identifierar volatilitetskompression via TTM Squeeze (Bollinger Bands inuti
Keltner Channels) med kompletterande volatilitetsindikatorer.

Signallogik:
    - SQUEEZE ON:  BB inuti KC → volatiliteten komprimerad
    - SQUEEZE FIRE: BB bryter ur KC efter squeeze → volatilitetsexpansion
    - NO SQUEEZE:  BB utanför KC → normal volatilitet

Kompletterande indikatorer:
    - HV Percentile:  nuvarande historisk vol vs 252d historik
    - BBW Percentile:  Bollinger Band Width vs 252d historik
    - ATR Compression: ATR(5) / ATR(50) — under 0.75 = komprimerad
    - Momentum:        linjär regression av (close - midline) → riktning
    - Volume Ratio:    senaste 5d volym / 50d medel
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class TickerSqueezeResult:
    """Per-ticker squeeze analysis."""
    ticker: str
    squeeze_on: bool                    # BB inuti KC just nu
    squeeze_firing: bool                # just brutit ur squeeze
    squeeze_days: int                   # konsekutiva dagar i squeeze (0 om ej squeeze)
    momentum: float                     # momentum-värde (linjär regression)
    momentum_direction: str             # 'BULL' / 'BEAR' / 'NEUTRAL'
    momentum_increasing: bool           # momentum accelererar (abs ökar)
    hv_percentile: float                # HV percentil (0-100)
    bbw_percentile: float               # BBW percentil (0-100)
    atr_compression: float              # ATR(5)/ATR(50) ratio
    volume_ratio: float                 # 5d vol / 50d vol
    signal: str                         # 'SQUEEZE_FIRE_BULL' / 'SQUEEZE_FIRE_BEAR' /
                                        # 'SQUEEZE_ON' / 'NO_SQUEEZE'
    score: float                        # komposit-score (högre = starkare squeeze-signal)
    # Tidsserier för charts
    price: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bb_upper: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bb_lower: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bb_mid: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    kc_upper: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    kc_lower: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    kc_mid: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    momentum_hist: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    squeeze_hist: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class SqueezeResult:
    """Container for full squeeze scan results."""
    ticker_results: Dict[str, TickerSqueezeResult]
    summary: pd.DataFrame               # screening-tabell
    n_squeeze_on: int
    n_squeeze_firing: int
    n_total: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _linear_regression_value(series: pd.Series, period: int) -> pd.Series:
    """Rullande linjär regressionsvärde (senaste punkt på regressionslinjen)."""
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(period - 1, len(values)):
        y = values[i - period + 1:i + 1]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(period, dtype=float)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
        result.iloc[i] = y_mean + slope * (period - 1 - x_mean)
    return result


# ── Analyzer ─────────────────────────────────────────────────────────────────

class SqueezeAnalyzer:
    """TTM Squeeze analyzer med volatilitetsindikatorer."""

    def __init__(self,
                 bb_length: int = 20,
                 bb_mult: float = 2.0,
                 kc_length: int = 20,
                 kc_mult: float = 1.5,
                 mom_length: int = 12,
                 atr_fast: int = 5,
                 atr_slow: int = 50,
                 hv_window: int = 20,
                 hv_lookback: int = 252):
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.mom_length = mom_length
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        self.hv_window = hv_window
        self.hv_lookback = hv_lookback

    def analyze_ticker(self, close: pd.Series, high: pd.Series = None,
                       low: pd.Series = None, volume: pd.Series = None,
                       ticker: str = "") -> Optional[TickerSqueezeResult]:
        """Analysera en enskild ticker för TTM Squeeze."""
        min_required = max(self.bb_length, self.kc_length, self.atr_slow) + 20
        if len(close) < min_required:
            print(f"[SQUEEZE] {ticker}: SKIP — only {len(close)} bars (need {min_required})")
            return None

        close = close.astype(float)

        # Om high/low saknas, approximera från close
        if high is None:
            high = close * 1.005  # Enkel approximation
        else:
            high = high.astype(float)
        if low is None:
            low = close * 0.995
        else:
            low = low.astype(float)

        # ── Bollinger Bands ─────────────────────────────────────────────
        bb_mid = close.rolling(self.bb_length).mean()
        bb_std = close.rolling(self.bb_length).std()
        bb_upper = bb_mid + self.bb_mult * bb_std
        bb_lower = bb_mid - self.bb_mult * bb_std

        # ── Keltner Channels ────────────────────────────────────────────
        kc_mid = _ema(close, self.kc_length)
        atr = _atr(high, low, close, self.kc_length)
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr

        # ── Squeeze detection ───────────────────────────────────────────
        # Squeeze ON = BB helt inuti KC
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

        # Räkna konsekutiva squeeze-dagar (bakifrån)
        squeeze_days = 0
        if squeeze_on.iloc[-1]:
            for i in range(len(squeeze_on) - 1, -1, -1):
                if squeeze_on.iloc[i]:
                    squeeze_days += 1
                else:
                    break

        # Squeeze FIRE = squeeze var ON igår men OFF idag
        squeeze_firing = False
        if len(squeeze_on) >= 2:
            squeeze_firing = bool(squeeze_on.iloc[-2]) and not bool(squeeze_on.iloc[-1])

        # ── Momentum (TTM Squeeze-stil) ─────────────────────────────────
        # Midline = genomsnitt av donchian midline och SMA
        highest = high.rolling(self.bb_length).max()
        lowest = low.rolling(self.bb_length).min()
        donchian_mid = (highest + lowest) / 2.0
        midline = (donchian_mid + bb_mid) / 2.0

        # Momentum = linjär regression av (close - midline)
        delta = close - midline
        momentum_hist = _linear_regression_value(delta, self.mom_length)

        current_mom = float(momentum_hist.iloc[-1]) if not np.isnan(momentum_hist.iloc[-1]) else 0.0
        prev_mom = float(momentum_hist.iloc[-2]) if len(momentum_hist) >= 2 and not np.isnan(momentum_hist.iloc[-2]) else 0.0

        if current_mom > 0:
            mom_direction = 'BULL'
        elif current_mom < 0:
            mom_direction = 'BEAR'
        else:
            mom_direction = 'NEUTRAL'

        mom_increasing = abs(current_mom) > abs(prev_mom)

        # ── HV Percentile ───────────────────────────────────────────────
        log_ret = np.log(close / close.shift(1)).dropna()
        hv = log_ret.rolling(self.hv_window).std() * np.sqrt(252)
        hv_current = float(hv.iloc[-1]) if not np.isnan(hv.iloc[-1]) else 0.0
        hv_history = hv.iloc[-self.hv_lookback:]
        hv_percentile = float((hv_history < hv_current).sum() / len(hv_history) * 100) if len(hv_history) > 10 else 50.0

        # ── BBW Percentile ──────────────────────────────────────────────
        bbw = (bb_upper - bb_lower) / bb_mid
        bbw_current = float(bbw.iloc[-1]) if not np.isnan(bbw.iloc[-1]) else 0.0
        bbw_history = bbw.iloc[-self.hv_lookback:]
        bbw_percentile = float((bbw_history < bbw_current).sum() / len(bbw_history) * 100) if len(bbw_history) > 10 else 50.0

        # ── ATR Compression ─────────────────────────────────────────────
        atr_fast = _atr(high, low, close, self.atr_fast)
        atr_slow = _atr(high, low, close, self.atr_slow)
        atr_ratio = float(atr_fast.iloc[-1] / atr_slow.iloc[-1]) if atr_slow.iloc[-1] > 0 else 1.0

        # ── Volume Ratio ────────────────────────────────────────────────
        if volume is not None and len(volume) > 50:
            vol_5d = volume.iloc[-5:].mean()
            vol_50d = volume.iloc[-50:].mean()
            vol_ratio = float(vol_5d / vol_50d) if vol_50d > 0 else 1.0
        else:
            vol_ratio = 1.0

        # ── Signal ──────────────────────────────────────────────────────
        if squeeze_firing:
            signal = f"SQUEEZE_FIRE_{mom_direction}"
        elif squeeze_on.iloc[-1]:
            signal = "SQUEEZE_ON"
        else:
            signal = "NO_SQUEEZE"

        # ── Komposit-score ──────────────────────────────────────────────
        # Högre = starkare squeeze-signal för straddle-köp
        score = 0.0
        if squeeze_on.iloc[-1] or squeeze_firing:
            # Längre squeeze = starkare potential
            score += min(squeeze_days / 20.0, 1.0) * 30  # max 30p för squeeze-längd

            # Lägre HV percentile = mer komprimerad vol
            score += max(0, (50 - hv_percentile) / 50.0) * 25  # max 25p

            # Lägre BBW percentile = smalare bands
            score += max(0, (50 - bbw_percentile) / 50.0) * 20  # max 20p

            # ATR compression
            score += max(0, (1.0 - atr_ratio)) * 15  # max 15p

            # Volume dry-up
            if vol_ratio < 0.7:
                score += 10  # bonus för låg volym

        if squeeze_firing:
            score += 20  # Bonus — squeeze just fired

        return TickerSqueezeResult(
            ticker=ticker,
            squeeze_on=bool(squeeze_on.iloc[-1]),
            squeeze_firing=squeeze_firing,
            squeeze_days=squeeze_days,
            momentum=current_mom,
            momentum_direction=mom_direction,
            momentum_increasing=mom_increasing,
            hv_percentile=hv_percentile,
            bbw_percentile=bbw_percentile,
            atr_compression=atr_ratio,
            volume_ratio=vol_ratio,
            signal=signal,
            score=round(score, 1),
            price=close,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_mid=bb_mid,
            kc_upper=kc_upper,
            kc_lower=kc_lower,
            kc_mid=kc_mid,
            momentum_hist=momentum_hist,
            squeeze_hist=squeeze_on.astype(float),
        )

    def scan(self, price_data: pd.DataFrame,
             high_data: pd.DataFrame = None,
             low_data: pd.DataFrame = None,
             volume_data: pd.DataFrame = None,
             progress_callback: Callable = None) -> SqueezeResult:
        """
        Scanna alla tickers för TTM Squeeze.

        Parameters
        ----------
        price_data : DataFrame med close-priser, kolumner = tickers
        high_data, low_data, volume_data : valfritt, samma format
        progress_callback : (pct: int, msg: str)

        Returns
        -------
        SqueezeResult
        """
        tickers = list(price_data.columns)
        results = {}
        n_total = len(tickers)
        n_skipped = 0

        print(f"[SQUEEZE] scan(): {n_total} tickers, price_data shape={price_data.shape}")

        for i, ticker in enumerate(tickers):
            if progress_callback and i % 5 == 0:
                pct = int(10 + 80 * i / max(n_total, 1))
                progress_callback(pct, f"Analyzing {ticker} ({i+1}/{n_total})...")

            close = price_data[ticker].dropna()
            high = high_data[ticker].dropna() if high_data is not None and ticker in high_data.columns else None
            low = low_data[ticker].dropna() if low_data is not None and ticker in low_data.columns else None
            volume = volume_data[ticker].dropna() if volume_data is not None and ticker in volume_data.columns else None

            result = self.analyze_ticker(close, high, low, volume, ticker)
            if result is not None:
                results[ticker] = result
            else:
                n_skipped += 1

        print(f"[SQUEEZE] scan(): {len(results)} OK, {n_skipped} skipped")

        # Bygg screening-tabell
        rows = []
        for ticker, r in results.items():
            rows.append({
                "Ticker": ticker,
                "Signal": r.signal,
                "Sqz Days": r.squeeze_days,
                "Mom": round(r.momentum, 3),
                "Dir": r.momentum_direction,
                "HV %ile": round(r.hv_percentile, 0),
                "BBW %ile": round(r.bbw_percentile, 0),
                "ATR Comp": round(r.atr_compression, 2),
                "Vol Ratio": round(r.volume_ratio, 2),
                "Score": r.score,
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values("Score", ascending=False).reset_index(drop=True)

        n_squeeze_on = sum(1 for r in results.values() if r.squeeze_on)
        n_firing = sum(1 for r in results.values() if r.squeeze_firing)

        if progress_callback:
            progress_callback(100, f"Squeeze scan complete — {n_squeeze_on} squeezing, {n_firing} firing")

        return SqueezeResult(
            ticker_results=results,
            summary=df,
            n_squeeze_on=n_squeeze_on,
            n_squeeze_firing=n_firing,
            n_total=len(results),
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    import sys

    tickers = sys.argv[1:] if len(sys.argv) > 1 else [
        "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM"
    ]

    print(f"\n{'='*70}")
    print(f"  TTM Squeeze Scanner — {len(tickers)} tickers")
    print(f"{'='*70}\n")

    print("Downloading price data...")
    data = yf.download(tickers, period="2y", group_by="ticker", progress=False)

    close_data = pd.DataFrame()
    high_data = pd.DataFrame()
    low_data = pd.DataFrame()
    vol_data = pd.DataFrame()

    for t in tickers:
        try:
            close_data[t] = data[t]["Close"]
            high_data[t] = data[t]["High"]
            low_data[t] = data[t]["Low"]
            vol_data[t] = data[t]["Volume"]
        except (KeyError, TypeError):
            print(f"  {t}: No data")

    analyzer = SqueezeAnalyzer()
    result = analyzer.scan(close_data, high_data, low_data, vol_data)

    print(f"\n{'─'*70}")
    print(result.summary.to_string(index=False))
    print(f"{'─'*70}")
    print(f"\nSqueezing: {result.n_squeeze_on}  |  Firing: {result.n_squeeze_firing}  |  Total: {result.n_total}")

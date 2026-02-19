"""
Markov Chain Analysis Engine
============================

Discrete Markov chain model for weekly return forecasting.
Uses 5 percentile-based states computed from each ticker's own return distribution.

States:
    0 = Strong Down  (0-10th percentile)
    1 = Down          (10-35th percentile)
    2 = Flat          (35-65th percentile)
    3 = Up            (65-90th percentile)
    4 = Strong Up     (90-100th percentile)

Pipeline:
    1. Fetch daily prices → resample to weekly (Friday close)
    2. Classify weekly returns into 5 states via percentile thresholds
    3. Build 5×5 transition matrix with Jeffreys prior smoothing
    4. Compute stationary distribution, spectral gap, mixing time
    5. Forecast next-week probabilities from current state row
    6. Track intraweek return vs last Friday's forecast
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

# ── State definitions ──────────────────────────────────────────────────────────

N_MARKOV_STATES = 5

MARKOV_STATES = {
    0: {'name': 'Strong Down', 'short': 'SD', 'color': '#ef4444'},
    1: {'name': 'Down',        'short': 'D',  'color': '#f59e0b'},
    2: {'name': 'Flat',        'short': 'F',  'color': '#a0a0a0'},
    3: {'name': 'Up',          'short': 'U',  'color': '#22c55e'},
    4: {'name': 'Strong Up',   'short': 'SU', 'color': '#3b82f6'},
}

PERCENTILE_BOUNDARIES = [10, 35, 65, 90]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MarkovResult:
    """Container for all Markov chain analysis outputs."""
    ticker: str
    weekly_returns: np.ndarray
    weekly_dates: np.ndarray
    state_sequence: np.ndarray
    thresholds: np.ndarray           # 4 percentile boundaries
    transition_matrix: np.ndarray    # 5×5
    stationary_dist: np.ndarray      # length 5
    n_states: int = N_MARKOV_STATES

    current_state: int = 0
    forecast_probs: np.ndarray = field(default_factory=lambda: np.zeros(N_MARKOV_STATES))
    forecast_2w: np.ndarray = field(default_factory=lambda: np.zeros(N_MARKOV_STATES))
    forecast_4w: np.ndarray = field(default_factory=lambda: np.zeros(N_MARKOV_STATES))

    last_friday_close: float = 0.0
    current_price: float = 0.0
    current_intraweek_return: float = 0.0
    current_intraweek_state: int = 2  # default Flat

    state_stats: Dict = field(default_factory=dict)
    eigenvalue_gap: float = 0.0
    mixing_time: float = 0.0
    n_observations: int = 0
    data_start: str = ""
    data_end: str = ""
    expected_return: float = 0.0


# ── Analyzer ──────────────────────────────────────────────────────────────────

class MarkovChainAnalyzer:
    """Discrete Markov chain analyzer for weekly ticker returns."""

    def __init__(self, lookback_years: int = 7, min_weeks: int = 100):
        self.lookback_years = lookback_years
        self.min_weeks = min_weeks

    # ── Data fetching ─────────────────────────────────────────────────────

    def fetch_weekly_data(self, ticker: str) -> Tuple[pd.Series, pd.DatetimeIndex]:
        """Download daily data, resample to weekly Friday closes, return pct_change."""
        import yfinance as yf

        end = datetime.now()
        start = end - timedelta(days=self.lookback_years * 365)

        df = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                         end=end.strftime('%Y-%m-%d'),
                         progress=False, auto_adjust=True, ignore_tz=True)

        if df.empty or len(df) < 5:
            raise ValueError(f"No data returned for {ticker}")

        # Handle multi-level columns from yfinance
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        # Resample to weekly (Friday close)
        weekly_close = close.resample('W-FRI').last().dropna()
        weekly_returns = weekly_close.pct_change().dropna()

        if len(weekly_returns) < self.min_weeks:
            raise ValueError(
                f"Only {len(weekly_returns)} weeks of data for {ticker} "
                f"(need {self.min_weeks})"
            )

        return weekly_returns, weekly_close.index

    # ── State classification ──────────────────────────────────────────────

    def classify_returns(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classify returns into 5 states using percentile boundaries.

        Returns (state_sequence, thresholds).
        """
        thresholds = np.percentile(returns, PERCENTILE_BOUNDARIES)
        states = np.digitize(returns, thresholds)  # 0..4
        return states.astype(int), thresholds

    # ── Transition matrix ─────────────────────────────────────────────────

    def build_transition_matrix(self, state_sequence: np.ndarray,
                                alpha: float = 0.5) -> np.ndarray:
        """Build 5×5 transition matrix with Jeffreys prior smoothing.

        alpha=0.5 is the Jeffreys prior (non-informative).
        """
        n = N_MARKOV_STATES
        counts = np.full((n, n), alpha)

        for i in range(len(state_sequence) - 1):
            s_from = state_sequence[i]
            s_to = state_sequence[i + 1]
            if 0 <= s_from < n and 0 <= s_to < n:
                counts[s_from, s_to] += 1

        # Row-normalize
        row_sums = counts.sum(axis=1, keepdims=True)
        T = counts / row_sums
        return T

    # ── Stationary distribution ───────────────────────────────────────────

    def compute_stationary_distribution(self, T: np.ndarray) -> np.ndarray:
        """Compute stationary distribution as left eigenvector for eigenvalue 1."""
        eigenvalues, eigenvectors = np.linalg.eig(T.T)

        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stat = np.real(eigenvectors[:, idx])
        stat = np.abs(stat)
        stat /= stat.sum()
        return stat

    # ── Spectral gap & mixing time ────────────────────────────────────────

    def compute_eigenvalue_gap(self, T: np.ndarray) -> Tuple[float, float]:
        """Compute spectral gap (1 - |lambda_2|) and approximate mixing time."""
        eigenvalues = np.linalg.eigvals(T)
        mods = np.abs(eigenvalues)
        mods_sorted = np.sort(mods)[::-1]  # descending

        if len(mods_sorted) < 2:
            return 1.0, 1.0

        lambda2 = mods_sorted[1]
        gap = 1.0 - lambda2
        gap = max(gap, 1e-6)  # avoid division by zero
        mixing_time = 1.0 / gap
        return float(gap), float(mixing_time)

    # ── Forecasting ───────────────────────────────────────────────────────

    def get_forecast(self, T: np.ndarray, current_state: int) -> np.ndarray:
        """Next-week forecast: row of T for current state."""
        return T[current_state, :]

    def get_multi_step_forecast(self, T: np.ndarray, current_state: int,
                                steps: int) -> np.ndarray:
        """Multi-step forecast: T^steps row for current state."""
        T_power = np.linalg.matrix_power(T, steps)
        return T_power[current_state, :]

    # ── Intraweek tracking ────────────────────────────────────────────────

    def compute_intraweek_tracking(self, ticker: str, thresholds: np.ndarray,
                                   last_friday_close: float) -> Tuple[float, int, float]:
        """Fetch current price and classify partial-week return.

        Returns (intraweek_return_pct, intraweek_state, current_price).
        """
        import yfinance as yf

        df = yf.download(ticker, period='5d', progress=False,
                         auto_adjust=True, ignore_tz=True)

        if df.empty:
            return 0.0, 2, last_friday_close  # Flat if no data

        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        current_price = float(close.iloc[-1])
        if last_friday_close == 0:
            return 0.0, 2, current_price

        intraweek_return = (current_price / last_friday_close) - 1.0
        state = int(np.digitize([intraweek_return], thresholds)[0])
        state = min(state, N_MARKOV_STATES - 1)
        return intraweek_return, state, current_price

    # ── State statistics ──────────────────────────────────────────────────

    def compute_state_statistics(self, returns: np.ndarray,
                                 state_sequence: np.ndarray) -> Dict:
        """Compute per-state statistics: avg return, volatility, count, avg duration."""
        stats = {}
        for s in range(N_MARKOV_STATES):
            mask = state_sequence == s
            state_returns = returns[mask]
            count = int(mask.sum())

            avg_ret = float(np.mean(state_returns)) if count > 0 else 0.0
            vol = float(np.std(state_returns)) if count > 1 else 0.0

            # Average consecutive duration
            durations = []
            run = 0
            for val in state_sequence:
                if val == s:
                    run += 1
                else:
                    if run > 0:
                        durations.append(run)
                    run = 0
            if run > 0:
                durations.append(run)
            avg_dur = float(np.mean(durations)) if durations else 0.0

            stats[s] = {
                'avg_return': avg_ret,
                'volatility': vol,
                'count': count,
                'frequency': count / len(state_sequence) if len(state_sequence) > 0 else 0.0,
                'avg_duration': avg_dur,
            }
        return stats

    # ── Full pipeline ─────────────────────────────────────────────────────

    def analyze(self, ticker: str,
                progress_callback: Optional[Callable] = None) -> MarkovResult:
        """Run the full Markov chain analysis pipeline for a ticker."""

        def _progress(pct: int, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        _progress(5, f"Fetching weekly data for {ticker}...")
        weekly_returns, weekly_dates = self.fetch_weekly_data(ticker)

        returns_arr = weekly_returns.values.astype(float)
        dates_arr = weekly_returns.index.to_numpy()

        _progress(20, "Classifying returns into states...")
        state_seq, thresholds = self.classify_returns(returns_arr)

        _progress(35, "Building transition matrix...")
        T = self.build_transition_matrix(state_seq)

        _progress(50, "Computing stationary distribution...")
        stat_dist = self.compute_stationary_distribution(T)

        _progress(60, "Computing spectral diagnostics...")
        gap, mix_time = self.compute_eigenvalue_gap(T)

        current_state = int(state_seq[-1])

        _progress(70, "Generating forecasts...")
        forecast_1w = self.get_forecast(T, current_state)
        forecast_2w = self.get_multi_step_forecast(T, current_state, 2)
        forecast_4w = self.get_multi_step_forecast(T, current_state, 4)

        _progress(80, "Computing state statistics...")
        state_stats = self.compute_state_statistics(returns_arr, state_seq)

        # Expected return: probability-weighted average
        expected_ret = sum(
            forecast_1w[s] * state_stats[s]['avg_return']
            for s in range(N_MARKOV_STATES)
        )

        # Last Friday close for intraweek tracking
        last_friday_close = 0.0
        weekly_close_dates = weekly_dates
        if len(weekly_close_dates) > 0:
            # Re-fetch to get the actual close value
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=30)
            recent = yf.download(ticker, start=start.strftime('%Y-%m-%d'),
                                 end=end.strftime('%Y-%m-%d'),
                                 progress=False, auto_adjust=True, ignore_tz=True)
            if not recent.empty:
                close_col = recent['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                weekly_recent = close_col.resample('W-FRI').last().dropna()
                if len(weekly_recent) > 0:
                    last_friday_close = float(weekly_recent.iloc[-1])

        _progress(90, "Tracking intraweek performance...")
        intraweek_ret = 0.0
        intraweek_state = 2
        current_price = last_friday_close
        if last_friday_close > 0:
            intraweek_ret, intraweek_state, current_price = \
                self.compute_intraweek_tracking(ticker, thresholds, last_friday_close)

        _progress(95, "Assembling results...")

        data_start = str(dates_arr[0])[:10] if len(dates_arr) > 0 else ""
        data_end = str(dates_arr[-1])[:10] if len(dates_arr) > 0 else ""

        result = MarkovResult(
            ticker=ticker,
            weekly_returns=returns_arr,
            weekly_dates=dates_arr,
            state_sequence=state_seq,
            thresholds=thresholds,
            transition_matrix=T,
            stationary_dist=stat_dist,
            current_state=current_state,
            forecast_probs=forecast_1w,
            forecast_2w=forecast_2w,
            forecast_4w=forecast_4w,
            last_friday_close=last_friday_close,
            current_price=current_price,
            current_intraweek_return=intraweek_ret,
            current_intraweek_state=intraweek_state,
            state_stats=state_stats,
            eigenvalue_gap=gap,
            mixing_time=mix_time,
            n_observations=len(returns_arr),
            data_start=data_start,
            data_end=data_end,
            expected_return=expected_ret,
        )

        _progress(100, "Analysis complete.")
        return result

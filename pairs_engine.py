"""
Pairs Trading Engine - Institutional-Grade Statistical Arbitrage System
========================================================================

Core engine for pairs trading using Ornstein-Uhlenbeck process modeling.
Includes universe screening, pair selection, signal generation, position sizing,
and risk management.

Author: Andreas
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats, integrate
from scipy.special import erf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

# Number of workers for parallel processing
N_WORKERS = min(8, multiprocessing.cpu_count())


# ============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# ============================================================================

@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters."""
    theta: float          # Mean reversion speed (annualized)
    mu: float             # Long-term mean
    sigma: float          # Volatility (annualized)
    eq_std: float         # Equilibrium standard deviation
    half_life_days: float # Half-life in trading days
    ar1_coef: float       # AR(1) coefficient (b)
    valid: bool = True
    reason: str = ""


@dataclass
class PairStatistics:
    """Statistical test results for a pair."""
    ticker_y: str
    ticker_x: str
    adf_pvalue: float
    adf_statistic: float
    eg_pvalue: float
    johansen_trace: float
    johansen_crit: float
    hurst_exponent: float
    correlation: float
    is_cointegrated: bool


@dataclass
class TradeSignal:
    """Generated trade signal with full context."""
    pair: str
    signal_type: str      # 'LONG_SPREAD', 'SHORT_SPREAD', 'NO_TRADE', 'EXIT'
    current_zscore: float
    entry_spread: float
    take_profit_spread: float
    stop_loss_spread: float
    win_probability: float
    expected_pnl: float
    kelly_fraction: float
    risk_reward: float
    avg_holding_days: float
    confidence: str       # 'HIGH', 'MEDIUM', 'LOW'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionRisk:
    """Risk metrics for a position."""
    pair: str
    current_pnl: float
    current_pnl_pct: float
    days_held: int
    current_zscore: float
    distance_to_stop: float
    distance_to_target: float
    max_drawdown: float
    risk_status: str      # 'OK', 'WARNING', 'CRITICAL'


# ============================================================================
# ORNSTEIN-UHLENBECK PROCESS CLASS
# ============================================================================

class OUProcess:
    """
    Ornstein-Uhlenbeck process analytics for pairs trading.
    
    dS = θ(μ - S)dt + σdW
    
    Provides closed-form solutions for conditional distributions,
    first passage times, and win probabilities.
    """
    
    def __init__(self, theta: float, mu: float, sigma: float):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.eq_var = sigma**2 / (2 * theta)
        self.eq_std = np.sqrt(self.eq_var)
    
    def zscore(self, S: float) -> float:
        """Convert spread level to z-score."""
        return (S - self.mu) / self.eq_std
    
    def spread_from_z(self, z: float) -> float:
        """Convert z-score to spread level."""
        return self.mu + z * self.eq_std
    
    def half_life_days(self) -> float:
        """Half-life in trading days."""
        return np.log(2) / self.theta * 252
    
    def conditional_mean(self, S0: float, tau: float) -> float:
        """E[S_τ | S_0] = μ + (S_0 - μ)e^{-θτ}"""
        return self.mu + (S0 - self.mu) * np.exp(-self.theta * tau)
    
    def conditional_var(self, tau: float) -> float:
        """Var[S_τ | S_0] = (σ²/2θ)(1 - e^{-2θτ})"""
        return self.eq_var * (1 - np.exp(-2 * self.theta * tau))
    
    def conditional_std(self, tau: float) -> float:
        return np.sqrt(self.conditional_var(tau))
    
    def prob_above_level(self, S0: float, target: float, tau: float) -> float:
        """P(S_τ > target | S_0)"""
        mean = self.conditional_mean(S0, tau)
        std = self.conditional_std(tau)
        return 1 - stats.norm.cdf(target, loc=mean, scale=std)
    
    def confidence_interval(self, S0: float, tau: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Confidence interval for S_τ given S_0."""
        mean = self.conditional_mean(S0, tau)
        std = self.conditional_std(tau)
        alpha = 1 - confidence
        lower = stats.norm.ppf(alpha/2, loc=mean, scale=std)
        upper = stats.norm.ppf(1 - alpha/2, loc=mean, scale=std)
        return lower, upper
    
    def expected_hitting_time(self, S0: float, target: float = None) -> float:
        """Expected time to hit target level from S0 (in years)."""
        if target is None:
            target = self.mu
        
        x0 = abs(S0 - self.mu) / self.eq_std
        x_target = abs(target - self.mu) / self.eq_std
        
        if x0 <= x_target:
            return 0.0
        
        def integrand(z):
            if z < 1e-10:
                return 0
            return (1 - np.exp(-z**2)) / z
        
        result, _ = integrate.quad(integrand, x_target, x0)
        return result / self.theta
    
    def win_probability(self, S0: float, take_profit: float, stop_loss: float,
                       n_sims: int = 5000, max_time_years: float = 0.5) -> Dict:
        """
        P(hit take_profit before stop_loss | S_0) via Monte Carlo.
        """
        dt = 1/252
        n_steps = int(max_time_years / dt)

        rng = np.random.RandomState(42)
        S = np.full(n_sims, S0)

        hit_tp = np.zeros(n_sims, dtype=bool)
        hit_sl = np.zeros(n_sims, dtype=bool)
        hit_time = np.full(n_sims, np.nan)

        going_down = S0 > self.mu

        for step in range(n_steps):
            if going_down:
                new_tp = (S <= take_profit) & ~hit_tp & ~hit_sl
                new_sl = (S >= stop_loss) & ~hit_tp & ~hit_sl
            else:
                new_tp = (S >= take_profit) & ~hit_tp & ~hit_sl
                new_sl = (S <= stop_loss) & ~hit_tp & ~hit_sl

            hit_tp |= new_tp
            hit_sl |= new_sl
            hit_time[new_tp | new_sl] = step * dt

            if (hit_tp | hit_sl).all():
                break

            # Early exit: >99% resolved → marginal gain from continuing
            if step > 10 and (hit_tp | hit_sl).sum() / n_sims > 0.99:
                break

            dW = rng.randn(n_sims) * np.sqrt(dt)
            S = S + self.theta * (self.mu - S) * dt + self.sigma * dW
        
        n_wins = hit_tp.sum()
        n_losses = hit_sl.sum()
        n_timeout = n_sims - n_wins - n_losses
        
        return {
            'win_prob': n_wins / n_sims,
            'loss_prob': n_losses / n_sims,
            'timeout_prob': n_timeout / n_sims,
            'avg_win_time_days': np.nanmean(hit_time[hit_tp]) * 252 if n_wins > 0 else np.nan,
            'avg_loss_time_days': np.nanmean(hit_time[hit_sl]) * 252 if n_losses > 0 else np.nan,
        }
    
    def expected_pnl(self, S0: float, take_profit: float, stop_loss: float) -> Dict:
        """Calculate expected P&L and Kelly fraction for a trade."""
        outcome = self.win_probability(S0, take_profit, stop_loss)
        
        p = outcome['win_prob']
        win_pnl = abs(take_profit - S0)
        loss_pnl = abs(stop_loss - S0)
        
        expected = p * win_pnl - (1 - p) * loss_pnl
        risk_reward = win_pnl / loss_pnl if loss_pnl > 0 else np.inf
        
        b = win_pnl / loss_pnl if loss_pnl > 0 else 0
        q = 1 - p
        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, kelly)
        
        return {
            'win_prob': p,
            'loss_prob': outcome['loss_prob'],
            'timeout_prob': outcome['timeout_prob'],
            'win_pnl': win_pnl,
            'loss_pnl': loss_pnl,
            'expected_pnl': expected,
            'risk_reward': risk_reward,
            'kelly_fraction': kelly,
            'avg_win_days': outcome['avg_win_time_days'],
            'avg_loss_days': outcome['avg_loss_time_days']
        }
    
    def calculate_expected_moves(self, current_spread: float, y_price: float, x_price: float,
                                  hedge_ratio: float, y_volatility: float = None, 
                                  x_volatility: float = None) -> Dict:
        """
        Calculate expected price moves for mean reversion.
        
        The spread is defined as: S = Y - β*X - α
        Z-score: Z = (S - μ) / σ_eq
        
        To reach Z=0, spread must change by: ΔS = -Z * σ_eq
        
        This can happen through:
        1. Only Y moving: ΔY = ΔS
        2. Only X moving: ΔX = -ΔS / β
        3. Both moving proportionally (volatility-weighted)
        
        Args:
            current_spread: Current spread value (S)
            y_price: Current price of Y
            x_price: Current price of X
            hedge_ratio: Beta coefficient (β)
            y_volatility: Optional - annualized volatility of Y for proportional calc
            x_volatility: Optional - annualized volatility of X for proportional calc
            
        Returns:
            Dict with expected moves in absolute terms and percentages
        """
        z = self.zscore(current_spread)
        delta_spread = -z * self.eq_std  # Spread change needed to reach mean
        
        # Scenario 1: Only Y moves
        delta_y_only = delta_spread
        delta_y_only_pct = (delta_y_only / y_price) * 100 if y_price != 0 else 0
        
        # Scenario 2: Only X moves
        delta_x_only = -delta_spread / hedge_ratio if hedge_ratio != 0 else 0
        delta_x_only_pct = (delta_x_only / x_price) * 100 if x_price != 0 else 0
        
        # Scenario 3: Proportional move (volatility-weighted if available)
        if y_volatility is not None and x_volatility is not None and y_volatility > 0 and x_volatility > 0:
            # Weight by inverse volatility (less volatile asset moves more)
            total_vol = y_volatility + abs(hedge_ratio) * x_volatility
            y_weight = 1 - (y_volatility / total_vol)
            x_weight = 1 - y_weight
            
            delta_y_prop = y_weight * delta_spread
            delta_x_prop = -x_weight * delta_spread / hedge_ratio if hedge_ratio != 0 else 0
        else:
            # Equal contribution assumption
            delta_y_prop = delta_spread / 2
            delta_x_prop = -delta_spread / (2 * hedge_ratio) if hedge_ratio != 0 else 0
        
        delta_y_prop_pct = (delta_y_prop / y_price) * 100 if y_price != 0 else 0
        delta_x_prop_pct = (delta_x_prop / x_price) * 100 if x_price != 0 else 0
        
        return {
            'z_score': z,
            'delta_spread_required': delta_spread,
            # Scenario 1: Only Y
            'delta_y_only': delta_y_only,
            'delta_y_only_pct': delta_y_only_pct,
            # Scenario 2: Only X
            'delta_x_only': delta_x_only,
            'delta_x_only_pct': delta_x_only_pct,
            # Scenario 3: Proportional
            'delta_y_proportional': delta_y_prop,
            'delta_y_proportional_pct': delta_y_prop_pct,
            'delta_x_proportional': delta_x_prop,
            'delta_x_proportional_pct': delta_x_prop_pct,
            # Direction indicators
            'y_direction': 'UP' if delta_y_only > 0 else 'DOWN',
            'x_direction': 'UP' if delta_x_only > 0 else 'DOWN',
        }


# ============================================================================
# KALMAN FILTER OU ESTIMATOR - Institutional-Grade Adaptive Parameter Tracking
# ============================================================================
# 
# References:
#   - Mehra (1970): Adaptive Kalman filtering with unknown noise covariances
#   - Rauch, Tung, Striebel (1965): Maximum likelihood estimates of linear
#     dynamic systems (RTS smoother)
#   - Elliott, van der Hoek, Malcolm (2005): Pairs trading with Kalman filter
#   - Triantafyllopoulos (2011): Time-varying parameter estimation via the 
#     Kalman filter
# ============================================================================

@dataclass
class KalmanOUResult:
    """Complete result from Kalman filter OU estimation."""
    # Current OU parameters (filtered estimates)
    theta: float              # Mean reversion speed (annualized)
    mu: float                 # Long-term mean
    sigma: float              # Volatility (annualized)
    eq_std: float             # Equilibrium standard deviation
    half_life_days: float     # Half-life in trading days
    ar1_coef: float           # AR(1) coefficient (b)
    valid: bool = True
    reason: str = ""
    
    # Kalman-specific diagnostics
    theta_std: float = 0.0    # Std of theta estimate (from P matrix)
    mu_std: float = 0.0       # Std of mu estimate
    param_stability: float = 0.0     # Rolling stability metric [0-1]
    regime_change_score: float = 0.0 # CUSUM score for structural breaks
    innovation_ratio: float = 0.0    # Normalized innovation (should be ~1)
    effective_sample_size: float = 0.0  # How many obs effectively used
    
    # Time series of tracked parameters (for plotting)
    theta_history: Optional[np.ndarray] = None
    mu_history: Optional[np.ndarray] = None
    sigma_history: Optional[np.ndarray] = None
    eq_std_history: Optional[np.ndarray] = None   # Time-varying equilibrium std
    zscore_history: Optional[np.ndarray] = None    # Time-varying z-scores (using params at each t)
    theta_upper: Optional[np.ndarray] = None   # 95% CI upper
    theta_lower: Optional[np.ndarray] = None   # 95% CI lower
    mu_upper: Optional[np.ndarray] = None
    mu_lower: Optional[np.ndarray] = None


class KalmanOUEstimator:
    """
    Kalman filter for adaptive Ornstein-Uhlenbeck parameter estimation.
    
    Models the discrete OU process as a linear state-space system:
    
        State:       x_t = [a_t, b_t]'   (intercept and AR(1) coefficient)
        Transition:  x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
        Observation: S_t = H_t · x_t + v_t,  v_t ~ N(0, R)
                     where H_t = [1, S_{t-1}]
    
    Features:
        - Adaptive Q estimation via innovation monitoring (Mehra 1970)
        - RTS smoother for improved historical estimates
        - Joseph form covariance update for numerical stability  
        - CUSUM test on normalized innovations for regime change detection
        - Delta method for confidence intervals on transformed parameters
        - Automatic observation noise (R) estimation
    
    Args:
        q_scale: Base process noise scaling factor (default 1e-5).
                 Controls how fast parameters can drift.
                 Higher = more responsive but noisier.
        adaptive_q: If True, automatically adjust Q based on innovations.
        smoother: If True, run RTS backward pass for better historical estimates.
        cusum_threshold: Threshold for regime change detection (default 4.0).
        min_warmup: Minimum observations before parameters are considered valid.
    """
    
    def __init__(self, 
                 q_scale: float = 1e-5,
                 adaptive_q: bool = True,
                 smoother: bool = True,
                 cusum_threshold: float = 4.0,
                 min_warmup: int = 30):
        self.q_scale = q_scale
        self.adaptive_q = adaptive_q
        self.smoother = smoother
        self.cusum_threshold = cusum_threshold
        self.min_warmup = min_warmup
        
        # Filter state (initialized on first call)
        self._x = None        # State vector [a, b]
        self._P = None        # State covariance
        self._R = None        # Observation noise variance (scalar)
        self._Q = None        # Process noise covariance (2x2)
        
        # Innovation tracking for adaptive Q and diagnostics
        self._innovation_window = 50
        self._innovations = []       # Raw innovations y_t
        self._norm_innovations = []  # Standardized innovations y_t / sqrt(S_t)
    
    def fit(self, spread: pd.Series, dt: float = 1/252) -> KalmanOUResult:
        """
        Run Kalman filter (+ optional RTS smoother) on spread series.
        
        This is the main entry point. Processes the entire series and returns
        the final filtered (or smoothed) OU parameter estimates with full
        diagnostics.
        
        Args:
            spread: Spread time series (e.g., Engle-Granger residuals)
            dt: Time step in years (1/252 for daily data)
            
        Returns:
            KalmanOUResult with current parameters, confidence intervals,
            stability metrics, and parameter history arrays.
        """
        spread = spread.dropna()
        n = len(spread)
        S = spread.values
        
        if n < self.min_warmup:
            return KalmanOUResult(
                theta=0, mu=0, sigma=0, eq_std=0,
                half_life_days=np.inf, ar1_coef=0,
                valid=False, reason=f'insufficient_data_{n}<{self.min_warmup}'
            )
        
        # ── INITIALIZATION ──
        # Use first min_warmup observations for robust initialization via OLS
        init_chunk = S[:self.min_warmup]
        S_lag = init_chunk[:-1]
        S_cur = init_chunk[1:]
        
        X_init = np.column_stack([np.ones_like(S_lag), S_lag])
        try:
            beta_init = np.linalg.lstsq(X_init, S_cur, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta_init = np.array([0.0, 0.95])
        
        residuals_init = S_cur - X_init @ beta_init
        R_init = float(np.var(residuals_init))
        
        if R_init < 1e-20:
            R_init = 1e-10
        
        # Initialize state
        x = beta_init.copy()              # [a, b]
        P = np.eye(2) * 0.01             # Initial uncertainty
        R = R_init                         # Observation noise
        Q = np.eye(2) * self.q_scale       # Process noise
        
        # ── STORAGE for history and smoother ──
        x_hist = np.zeros((n - 1, 2))    # Filtered state at each step
        P_hist = np.zeros((n - 1, 2, 2)) # Filtered covariance at each step
        x_pred_hist = np.zeros((n - 1, 2))    # Predicted state (for smoother)
        P_pred_hist = np.zeros((n - 1, 2, 2)) # Predicted covariance (for smoother)
        
        innovations = np.zeros(n - 1)
        norm_innovations = np.zeros(n - 1)
        innovation_vars = np.zeros(n - 1)
        
        # Adaptive Q tracking
        adaptive_window = min(self._innovation_window, n // 4)
        q_history = np.zeros(n - 1)
        
        # ── FORWARD PASS (Kalman Filter) ──
        for t in range(1, n):
            idx = t - 1  # Storage index
            
            # === PREDICT ===
            x_pred = x                          # F = I → x_pred = x (no copy needed)
            P_pred = P + Q                     # P_pred = F·P·Fᵀ + Q = P + Q
            
            x_pred_hist[idx] = x_pred
            P_pred_hist[idx] = P_pred
            
            # === OBSERVATION ===
            H = np.array([1.0, S[t - 1]])     # H_t = [1, S_{t-1}]
            y_pred = H @ x_pred                 # Predicted observation
            y = S[t] - y_pred                   # Innovation
            
            # Innovation covariance: S = H·P_pred·Hᵀ + R (scalar)
            S_inn = H @ P_pred @ H + R
            S_inn = max(S_inn, 1e-20)           # Numerical guard
            
            innovations[idx] = y
            innovation_vars[idx] = S_inn
            norm_innovations[idx] = y / np.sqrt(S_inn)
            
            # === KALMAN GAIN ===
            K = P_pred @ H / S_inn              # K = P_pred·Hᵀ·S⁻¹ (2x1 vector)
            
            # === UPDATE (Joseph form for numerical stability) ===
            # P = (I - K·H)·P_pred·(I - K·H)ᵀ + K·R·Kᵀ
            # This is always symmetric positive semi-definite
            IKH = np.eye(2) - np.outer(K, H)
            P = IKH @ P_pred @ IKH.T + R * np.outer(K, K)
            
            # State update
            x = x_pred + K * y
            
            # Enforce stationarity: b must be in (0, 1)
            x[1] = np.clip(x[1], 0.001, 0.999)
            
            # Store filtered results
            x_hist[idx] = x
            P_hist[idx] = P
            
            # === ADAPTIVE Q (Mehra 1970) ===
            if self.adaptive_q and idx >= adaptive_window:
                # Estimate actual innovation covariance from recent data
                recent_inn = innovations[idx - adaptive_window + 1:idx + 1]
                C_hat = np.mean(recent_inn ** 2)
                
                # Expected innovation variance if Q were zero: H·P_base·Hᵀ + R
                # Excess variance indicates parameter drift → increase Q
                expected_S = R  # Minimum expected innovation variance
                excess = max(0, C_hat - expected_S)
                
                # Scale Q proportionally to excess innovation variance
                # More excess → parameters are drifting faster → increase Q
                q_adapt = self.q_scale + excess * 0.01
                q_adapt = min(q_adapt, 0.01)  # Cap to prevent instability
                Q = np.eye(2) * q_adapt
                q_history[idx] = q_adapt
            else:
                q_history[idx] = self.q_scale
            
            # === ADAPTIVE R (observation noise) ===
            # Update R using exponential moving average of squared innovations
            if idx >= 10:
                alpha_R = 0.02  # Slow adaptation
                R = (1 - alpha_R) * R + alpha_R * y ** 2
                R = max(R, 1e-20)
        
        # ── RTS SMOOTHER (backward pass) ──
        if self.smoother and n > self.min_warmup + 10:
            x_smooth = np.zeros_like(x_hist)
            P_smooth = np.zeros_like(P_hist)
            
            # Initialize smoother from last filtered estimate
            x_smooth[-1] = x_hist[-1]
            P_smooth[-1] = P_hist[-1]
            
            for t in range(n - 3, -1, -1):
                # Smoother gain: L = P_t · Fᵀ · P_{t+1|t}⁻¹
                # Since F = I: L = P_t · P_{t+1|t}⁻¹
                try:
                    P_pred_inv = np.linalg.inv(P_pred_hist[t + 1])
                    L = P_hist[t] @ P_pred_inv
                except np.linalg.LinAlgError:
                    L = np.eye(2) * 0.5
                
                # Smoothed estimates
                x_smooth[t] = x_hist[t] + L @ (x_smooth[t + 1] - x_pred_hist[t + 1])
                P_smooth[t] = P_hist[t] + L @ (P_smooth[t + 1] - P_pred_hist[t + 1]) @ L.T
                
                # Enforce stationarity on smoothed estimates too
                x_smooth[t, 1] = np.clip(x_smooth[t, 1], 0.001, 0.999)
            
            # Use smoothed estimates for history (but filtered for current)
            x_final_hist = x_smooth
            P_final_hist = P_smooth
        else:
            x_final_hist = x_hist
            P_final_hist = P_hist
        
        # ── EXTRACT OU PARAMETERS ──
        # Use last filtered estimate for current parameters
        a_final = x[0]
        b_final = x[1]
        
        if b_final <= 0 or b_final >= 1:
            return KalmanOUResult(
                theta=0, mu=0, sigma=0, eq_std=0,
                half_life_days=np.inf, ar1_coef=float(b_final),
                valid=False, reason='explosive_or_unit_root'
            )
        
        theta = -np.log(b_final) / dt
        mu = a_final / (1 - b_final)
        
        # Estimate sigma from innovation sequence (last 100 points)
        recent_n = min(100, len(innovations))
        residual_std = np.std(innovations[-recent_n:])
        sigma = residual_std * np.sqrt(2 * theta / (1 - b_final ** 2))
        eq_std = sigma / np.sqrt(2 * theta) if theta > 0 else 0
        half_life = np.log(2) / theta * 252 if theta > 0 else np.inf
        
        # ── CONFIDENCE INTERVALS via Delta Method ──
        # Var(theta) ≈ (∂theta/∂b)² · Var(b), where ∂theta/∂b = -1/(b·dt)
        var_b = P[1, 1]
        var_a = P[0, 0]
        cov_ab = P[0, 1]
        
        dtheta_db = 1.0 / (b_final * dt) if b_final > 0 else 0
        theta_std = abs(dtheta_db) * np.sqrt(var_b)
        
        # Var(mu) ≈ (∂mu/∂a)²·Var(a) + (∂mu/∂b)²·Var(b) + 2·(∂mu/∂a)(∂mu/∂b)·Cov(a,b)
        dmu_da = 1.0 / (1 - b_final)
        dmu_db = a_final / (1 - b_final) ** 2
        mu_var = (dmu_da ** 2 * var_a + dmu_db ** 2 * var_b + 
                  2 * dmu_da * dmu_db * cov_ab)
        mu_std = np.sqrt(max(0, mu_var))
        
        # ── PARAMETER HISTORY (convert a,b to theta,mu,sigma) ──
        b_hist = x_final_hist[:, 1]
        a_hist = x_final_hist[:, 0]
        
        # Clamp for numerical safety
        b_hist_safe = np.clip(b_hist, 0.001, 0.999)
        
        theta_hist = -np.log(b_hist_safe) / dt
        mu_hist = a_hist / (1 - b_hist_safe)
        
        # Rolling sigma from innovations
        sigma_hist = np.full(len(x_final_hist), np.nan)
        roll_win = min(30, len(innovations) // 3)
        if roll_win >= 5:
            for i in range(roll_win, len(innovations)):
                local_std = np.std(innovations[i - roll_win:i])
                local_b = b_hist_safe[i]
                local_theta = theta_hist[i]
                if local_theta > 0 and (1 - local_b ** 2) > 0:
                    sigma_hist[i] = local_std * np.sqrt(2 * local_theta / (1 - local_b ** 2))
        
        # ── TIME-VARYING Z-SCORES ──
        # Use expanding window z-scores: at each time t, z = (S_t - mean_t) / std_t
        # where mean_t and std_t are computed from ALL spread data up to t.
        # This matches how practitioners actually think about z-scores:
        # "how extreme is the spread right now relative to its history so far?"
        #
        # Warmup: require at least 60 observations before producing z-scores.
        # This avoids the numerical instability of tiny denominators.
        
        zscore_warmup = max(60, self.min_warmup * 2)
        eq_std_hist = np.full(len(x_final_hist), np.nan)
        zscore_hist = np.full(len(x_final_hist), np.nan)
        
        spread_vals = S[1:]  # S[1:] aligns with x_hist indices
        
        for i in range(zscore_warmup, len(x_final_hist)):
            # Use Kalman mu_t as the center, but compute dispersion from
            # historical spread residuals around the evolving mu
            local_mu = mu_hist[i]
            
            # Expanding window std: all data up to point i, centered on local mu
            historical = spread_vals[:i + 1]
            local_std = np.std(historical - local_mu)
            
            if local_std > 1e-10:
                eq_std_hist[i] = local_std
                z = (spread_vals[i] - local_mu) / local_std
                # Clamp to ±3: anything beyond is not actionable
                zscore_hist[i] = np.clip(z, -3, 3)
        
        # Confidence intervals on theta and mu
        P_b_hist = P_final_hist[:, 1, 1]
        P_a_hist = P_final_hist[:, 0, 0]
        P_ab_hist = P_final_hist[:, 0, 1]
        
        theta_std_hist = np.abs(1.0 / (b_hist_safe * dt)) * np.sqrt(np.maximum(P_b_hist, 0))
        theta_upper = theta_hist + 1.96 * theta_std_hist
        theta_lower = np.maximum(0, theta_hist - 1.96 * theta_std_hist)
        
        dmu_da_hist = 1.0 / (1 - b_hist_safe)
        dmu_db_hist = a_hist / (1 - b_hist_safe) ** 2
        mu_var_hist = (dmu_da_hist ** 2 * P_a_hist + 
                       dmu_db_hist ** 2 * P_b_hist + 
                       2 * dmu_da_hist * dmu_db_hist * P_ab_hist)
        mu_std_hist = np.sqrt(np.maximum(0, mu_var_hist))
        mu_upper = mu_hist + 1.96 * mu_std_hist
        mu_lower = mu_hist - 1.96 * mu_std_hist
        
        # ── STABILITY METRICS ──
        # Parameter stability: coefficient of variation of theta over last 60 obs
        stability_window = min(60, len(theta_hist))
        recent_theta = theta_hist[-stability_window:]
        if np.mean(recent_theta) > 0:
            theta_cv = np.std(recent_theta) / np.mean(recent_theta)
            param_stability = max(0, 1 - theta_cv)  # 1 = perfectly stable
        else:
            param_stability = 0.0
        
        # ── REGIME CHANGE DETECTION (CUSUM on normalized innovations) ──
        if len(norm_innovations) > self.min_warmup:
            # Two-sided CUSUM on squared normalized innovations
            # Under H0 (no change): E[v²] = 1
            v2 = norm_innovations[self.min_warmup:] ** 2
            cusum_plus = 0.0
            cusum_minus = 0.0
            max_cusum = 0.0
            k = 0.5  # Allowance parameter (slack)
            
            for vi in v2:
                cusum_plus = max(0, cusum_plus + vi - 1 - k)
                cusum_minus = max(0, cusum_minus - vi + 1 - k)
                max_cusum = max(max_cusum, cusum_plus, cusum_minus)
            
            # Normalize by √N so threshold is comparable across series lengths
            regime_change_score = max_cusum / np.sqrt(len(v2))
        else:
            regime_change_score = 0.0
        
        # ── INNOVATION DIAGNOSTICS ──
        recent_norm = norm_innovations[-min(100, len(norm_innovations)):]
        innovation_ratio = np.mean(recent_norm ** 2) if len(recent_norm) > 0 else 0
        
        # Effective sample size (based on autocorrelation of innovations)
        if len(innovations) > 20:
            rho1 = np.corrcoef(innovations[:-1], innovations[1:])[0, 1]
            rho1 = np.clip(rho1, -0.99, 0.99)
            ess = len(innovations) * (1 - rho1) / (1 + rho1)
            ess = max(1, min(ess, len(innovations)))
        else:
            ess = float(len(innovations))
        
        return KalmanOUResult(
            theta=float(theta),
            mu=float(mu),
            sigma=float(sigma),
            eq_std=float(eq_std),
            half_life_days=float(half_life),
            ar1_coef=float(b_final),
            valid=True,
            theta_std=float(theta_std),
            mu_std=float(mu_std),
            param_stability=float(param_stability),
            regime_change_score=float(regime_change_score),
            innovation_ratio=float(innovation_ratio),
            effective_sample_size=float(ess),
            theta_history=theta_hist,
            mu_history=mu_hist,
            sigma_history=sigma_hist,
            eq_std_history=eq_std_hist,
            zscore_history=zscore_hist,
            theta_upper=theta_upper,
            theta_lower=theta_lower,
            mu_upper=mu_upper,
            mu_lower=mu_lower,
        )
    
    def fit_to_ou_parameters(self, spread: pd.Series, dt: float = 1/252) -> OUParameters:
        """
        Convenience method: run Kalman filter and return result as standard
        OUParameters dataclass for drop-in compatibility with existing code.
        """
        result = self.fit(spread, dt)
        
        return OUParameters(
            theta=result.theta,
            mu=result.mu,
            sigma=result.sigma,
            eq_std=result.eq_std,
            half_life_days=result.half_life_days,
            ar1_coef=result.ar1_coef,
            valid=result.valid,
            reason=result.reason
        )


# ============================================================================
# AR(2) PROCESS CLASS - Extended Mean Reversion Dynamics
# ============================================================================

@dataclass
class AR2Parameters:
    """AR(2) process fitted parameters and diagnostics."""
    phi1: float                 # First lag coefficient
    phi2: float                 # Second lag coefficient  
    c: float                    # Constant term
    mu: float                   # Long-term mean
    sigma: float                # Innovation standard deviation
    
    # Dynamics classification
    discriminant: float         # φ₁² + 4φ₂
    dynamics_type: str          # 'overdamped', 'critically_damped', 'underdamped'
    is_stationary: bool         # Whether process is stationary
    
    # For underdamped (oscillating) case
    damping_factor: float       # r = √(-φ₂), controls decay rate
    oscillation_freq: float     # ω, radians per period
    cycle_period: float         # T = 2π/ω in days
    damping_ratio: float        # How quickly oscillations decay
    
    # Comparison with OU
    effective_half_life: float  # Comparable to OU half-life
    ou_equivalent_theta: float  # What θ would give similar short-term behavior
    
    valid: bool = True
    reason: str = ""


class AR2Process:
    """
    AR(2) process analytics for pairs trading with oscillating spreads.
    
    S_t = c + φ₁·S_{t-1} + φ₂·S_{t-2} + ε_t
    
    Captures mean reversion WITH momentum/overshooting behavior that
    simple OU/AR(1) models miss. Essential when ACF shows oscillation
    or negative values at longer lags.
    
    Key insight: When φ₂ < 0 and φ₁² + 4φ₂ < 0, the spread oscillates
    around its mean rather than decaying monotonically.
    """
    
    def __init__(self, phi1: float, phi2: float, c: float, sigma: float):
        """
        Initialize AR(2) process.
        
        Args:
            phi1: First lag coefficient
            phi2: Second lag coefficient (negative = overshooting tendency)
            c: Constant term
            sigma: Innovation standard deviation
        """
        self.phi1 = phi1
        self.phi2 = phi2
        self.c = c
        self.sigma = sigma
        
        # Calculate long-term mean
        denom = 1 - phi1 - phi2
        self.mu = c / denom if abs(denom) > 1e-10 else 0
        
        # Stationarity check: roots of characteristic equation inside unit circle
        # Equivalent conditions: |φ₂| < 1, φ₂ + φ₁ < 1, φ₂ - φ₁ < 1
        self.is_stationary = (
            abs(phi2) < 1 and 
            phi2 + phi1 < 1 and 
            phi2 - phi1 < 1
        )
        
        # Characteristic equation: λ² - φ₁λ - φ₂ = 0
        # Discriminant determines dynamics type
        self.discriminant = phi1**2 + 4*phi2
        
        if self.discriminant > 0:
            # Real roots - overdamped (no oscillation)
            self.dynamics_type = 'overdamped'
            sqrt_disc = np.sqrt(self.discriminant)
            self.lambda1 = (phi1 + sqrt_disc) / 2
            self.lambda2 = (phi1 - sqrt_disc) / 2
            self.damping_factor = max(abs(self.lambda1), abs(self.lambda2))
            self.oscillation_freq = 0
            self.cycle_period = np.inf
            
        elif self.discriminant == 0:
            # Repeated real root - critically damped
            self.dynamics_type = 'critically_damped'
            self.lambda1 = self.lambda2 = phi1 / 2
            self.damping_factor = abs(self.lambda1)
            self.oscillation_freq = 0
            self.cycle_period = np.inf
            
        else:
            # Complex roots - underdamped (oscillating!)
            self.dynamics_type = 'underdamped'
            # λ = r·e^(±iω) where r = √(-φ₂), ω = arccos(φ₁/(2√(-φ₂)))
            self.damping_factor = np.sqrt(-phi2)  # r
            
            # Oscillation frequency
            cos_omega = phi1 / (2 * self.damping_factor)
            cos_omega = np.clip(cos_omega, -1, 1)  # Numerical safety
            self.oscillation_freq = np.arccos(cos_omega)  # ω in radians
            
            # Cycle period in days
            if self.oscillation_freq > 0:
                self.cycle_period = 2 * np.pi / self.oscillation_freq
            else:
                self.cycle_period = np.inf
        
        # Damping ratio: how quickly oscillations decay
        # 0 = no damping (pure oscillation), 1 = critically damped
        if self.damping_factor > 0 and self.damping_factor < 1:
            self.damping_ratio = -np.log(self.damping_factor)
        else:
            self.damping_ratio = 0
        
        # Effective half-life (comparable to OU)
        if self.damping_factor > 0 and self.damping_factor < 1:
            self.effective_half_life = np.log(2) / (-np.log(self.damping_factor))
        else:
            self.effective_half_life = np.inf
        
        # OU-equivalent theta (for short-term comparison)
        # Match first-lag autocorrelation: ρ(1)_OU = e^(-θ/252) ≈ ρ(1)_AR2
        ar2_acf1 = self.acf(1)
        if ar2_acf1 > 0:
            self.ou_equivalent_theta = -np.log(ar2_acf1) * 252
        else:
            self.ou_equivalent_theta = np.inf
    
    def acf(self, k: int) -> float:
        """
        Theoretical autocorrelation at lag k.
        
        For AR(2), ACF satisfies the Yule-Walker equations:
            ρ(k) = φ₁·ρ(k-1) + φ₂·ρ(k-2)  for k ≥ 2
            ρ(0) = 1
            ρ(1) = φ₁ / (1 - φ₂)
        """
        if k == 0:
            return 1.0
        elif k == 1:
            if abs(1 - self.phi2) > 1e-10:
                return self.phi1 / (1 - self.phi2)
            return 0
        else:
            # Recursive calculation
            rho = [1.0, self.phi1 / (1 - self.phi2) if abs(1 - self.phi2) > 1e-10 else 0]
            for i in range(2, k + 1):
                rho_k = self.phi1 * rho[-1] + self.phi2 * rho[-2]
                rho.append(rho_k)
            return rho[k]
    
    def acf_array(self, max_lag: int) -> np.ndarray:
        """Calculate ACF for lags 0 to max_lag."""
        return np.array([self.acf(k) for k in range(max_lag + 1)])
    
    def impulse_response(self, n_periods: int) -> np.ndarray:
        """
        Impulse response function - how a shock propagates over time.
        
        Shows the path of mean reversion after a 1-unit shock.
        """
        psi = np.zeros(n_periods)
        psi[0] = 1
        if n_periods > 1:
            psi[1] = self.phi1
        for t in range(2, n_periods):
            psi[t] = self.phi1 * psi[t-1] + self.phi2 * psi[t-2]
        return psi
    
    def forecast(self, S_t: float, S_t_minus_1: float, n_steps: int) -> np.ndarray:
        """
        Point forecast n steps ahead.
        
        Args:
            S_t: Current spread value
            S_t_minus_1: Previous spread value
            n_steps: Forecast horizon
            
        Returns:
            Array of forecasted spread values
        """
        forecast = np.zeros(n_steps)
        s_prev = S_t
        s_prev2 = S_t_minus_1
        
        for t in range(n_steps):
            s_next = self.c + self.phi1 * s_prev + self.phi2 * s_prev2
            forecast[t] = s_next
            s_prev2 = s_prev
            s_prev = s_next
        
        return forecast
    
    def forecast_with_bands(self, S_t: float, S_t_minus_1: float, n_steps: int, 
                           confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast with confidence bands via Monte Carlo.
        
        Returns:
            Tuple of (mean_forecast, lower_band, upper_band)
        """
        n_sims = 2000
        np.random.seed(42)
        
        paths = np.zeros((n_sims, n_steps))
        
        for sim in range(n_sims):
            s_prev = S_t
            s_prev2 = S_t_minus_1
            
            for t in range(n_steps):
                eps = np.random.randn() * self.sigma
                s_next = self.c + self.phi1 * s_prev + self.phi2 * s_prev2 + eps
                paths[sim, t] = s_next
                s_prev2 = s_prev
                s_prev = s_next
        
        mean_forecast = paths.mean(axis=0)
        alpha = (1 - confidence) / 2
        lower_band = np.percentile(paths, alpha * 100, axis=0)
        upper_band = np.percentile(paths, (1 - alpha) * 100, axis=0)
        
        return mean_forecast, lower_band, upper_band
    
    def optimal_exit_timing(self, current_z: float, target_z: float = 0.0) -> Dict:
        """
        Estimate optimal exit timing based on AR(2) dynamics.
        
        For underdamped systems, the spread will overshoot - this calculates
        when to exit BEFORE the overshoot.
        """
        if self.dynamics_type == 'underdamped' and self.cycle_period < 200:
            # First crossing time (approximate)
            # For damped oscillation starting at z, first zero crossing ≈ T/4
            quarter_cycle = self.cycle_period / 4
            
            # But we want to exit slightly before (when momentum is still favorable)
            optimal_exit = quarter_cycle * 0.8
            
            # Expected overshoot magnitude (dampens with r^k)
            overshoot_factor = self.damping_factor ** (self.cycle_period / 2)
            expected_overshoot = abs(current_z) * overshoot_factor
            
            return {
                'dynamics': 'oscillating',
                'optimal_exit_days': optimal_exit,
                'first_zero_crossing': quarter_cycle,
                'expected_overshoot_z': expected_overshoot,
                'cycle_period': self.cycle_period,
                'recommendation': f"Consider exiting around day {optimal_exit:.0f} before overshoot"
            }
        else:
            return {
                'dynamics': 'monotonic',
                'optimal_exit_days': self.effective_half_life,
                'recommendation': "Standard OU-style exit at mean"
            }
    
    def trading_recommendation(self, current_z: float) -> Dict:
        """
        Generate trading recommendation based on AR(2) dynamics.
        """
        rec = {
            'dynamics_type': self.dynamics_type,
            'cycle_period': self.cycle_period if self.dynamics_type == 'underdamped' else None,
            'effective_half_life': self.effective_half_life,
            'damping_factor': self.damping_factor,
        }
        
        if self.dynamics_type == 'underdamped':
            rec['exit_strategy'] = 'EARLY'
            rec['target_z'] = 0.5 * np.sign(current_z) * -1  # Exit before mean
            rec['reason'] = f"Oscillating dynamics with {self.cycle_period:.1f}-day cycle. Exit early to avoid overshoot."
            rec['risk'] = 'MEDIUM - spread will likely overshoot mean'
        elif self.dynamics_type == 'overdamped':
            rec['exit_strategy'] = 'STANDARD'
            rec['target_z'] = 0.0
            rec['reason'] = "Monotonic decay to mean. Standard OU-style exit."
            rec['risk'] = 'LOW - predictable mean reversion'
        else:
            rec['exit_strategy'] = 'STANDARD'
            rec['target_z'] = 0.0
            rec['reason'] = "Critically damped - fastest decay without oscillation."
            rec['risk'] = 'LOW'
        
        return rec
    
    def get_parameters(self) -> AR2Parameters:
        """Return structured parameter object."""
        return AR2Parameters(
            phi1=self.phi1,
            phi2=self.phi2,
            c=self.c,
            mu=self.mu,
            sigma=self.sigma,
            discriminant=self.discriminant,
            dynamics_type=self.dynamics_type,
            is_stationary=self.is_stationary,
            damping_factor=self.damping_factor,
            oscillation_freq=self.oscillation_freq,
            cycle_period=self.cycle_period,
            damping_ratio=self.damping_ratio,
            effective_half_life=self.effective_half_life,
            ou_equivalent_theta=self.ou_equivalent_theta,
            valid=self.is_stationary,
            reason="" if self.is_stationary else "Non-stationary AR(2) process"
        )
    
    @classmethod
    def fit(cls, spread: pd.Series, method: str = 'yule_walker') -> 'AR2Process':
        """
        Fit AR(2) model to spread series.
        
        Args:
            spread: Time series of spread values
            method: 'yule_walker' (default, matches ACF) or 'ols' (minimizes residuals)
            
        Returns:
            Fitted AR2Process instance
        """
        if len(spread) < 10:
            raise ValueError("Need at least 10 observations to fit AR(2)")
        
        S = spread.values
        n = len(S)
        
        if method == 'yule_walker':
            # =====================================================
            # YULE-WALKER ESTIMATION
            # =====================================================
            # Fits AR parameters directly from autocorrelations.
            # This ensures the theoretical ACF matches empirical ACF.
            #
            # Yule-Walker equations for AR(2):
            #   ρ(1) = φ₁ + φ₂·ρ(1)  →  ρ(1) = φ₁/(1-φ₂)
            #   ρ(2) = φ₁·ρ(1) + φ₂
            #
            # Solving for φ₁, φ₂:
            #   φ₂ = (ρ(2) - ρ(1)²) / (1 - ρ(1)²)
            #   φ₁ = ρ(1)·(1 - φ₂)
            # =====================================================
            
            # Calculate sample autocorrelations
            spread_centered = S - np.mean(S)
            gamma0 = np.sum(spread_centered**2) / n  # Variance (γ₀)
            
            # γ(1) and γ(2) - autocovariances
            gamma1 = np.sum(spread_centered[1:] * spread_centered[:-1]) / n
            gamma2 = np.sum(spread_centered[2:] * spread_centered[:-2]) / n
            
            # ρ(1) and ρ(2) - autocorrelations
            rho1 = gamma1 / gamma0 if gamma0 > 0 else 0
            rho2 = gamma2 / gamma0 if gamma0 > 0 else 0
            
            # Solve Yule-Walker equations
            denom = 1 - rho1**2
            if abs(denom) > 1e-10:
                phi2 = (rho2 - rho1**2) / denom
                phi1 = rho1 * (1 - phi2)
            else:
                # Degenerate case - fall back to simple AR(1)
                phi1 = rho1
                phi2 = 0
            
            # Ensure stationarity: constrain parameters if needed
            # Stationarity requires: |φ₂| < 1, φ₁ + φ₂ < 1, φ₂ - φ₁ < 1
            if abs(phi2) >= 1:
                phi2 = np.sign(phi2) * 0.95
            if phi1 + phi2 >= 1:
                scale = 0.95 / (phi1 + phi2)
                phi1 *= scale
                phi2 *= scale
            if phi2 - phi1 >= 1:
                phi2 = phi1 + 0.95
            
            # Estimate constant from mean
            mu = np.mean(S)
            c = mu * (1 - phi1 - phi2)
            
            # Estimate innovation variance
            # σ²_ε = γ₀·(1 - φ₁·ρ(1) - φ₂·ρ(2))
            sigma_sq = gamma0 * (1 - phi1 * rho1 - phi2 * rho2)
            sigma = np.sqrt(max(sigma_sq, 1e-10))
            
        else:
            # =====================================================
            # OLS ESTIMATION (original method)
            # =====================================================
            S_t = S[2:]           # S_t
            S_lag1 = S[1:-1]      # S_{t-1}
            S_lag2 = S[:-2]       # S_{t-2}
            
            X = np.column_stack([np.ones(len(S_t)), S_lag1, S_lag2])
            
            try:
                coeffs = np.linalg.lstsq(X, S_t, rcond=None)[0]
            except np.linalg.LinAlgError:
                coeffs = np.linalg.pinv(X) @ S_t
            
            c, phi1, phi2 = coeffs
            
            residuals = S_t - (c + phi1 * S_lag1 + phi2 * S_lag2)
            sigma = np.std(residuals, ddof=3)
        
        return cls(phi1=phi1, phi2=phi2, c=c, sigma=sigma)


# ============================================================================
# PAIRS TRADING ENGINE
# ============================================================================

class PairsTradingEngine:
    """
    Complete pairs trading system with universe screening, pair selection,
    signal generation, and risk management.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize engine with configuration.
        
        Config options:
            lookback_period: Data lookback (default '2y')
            min_half_life: Minimum half-life in days (default 5)
            max_half_life: Maximum half-life in days (default 30)
            max_adf_pvalue: Maximum ADF p-value for cointegration (default 0.10)
            entry_zscore: Z-score threshold for entry (default 2.0)
            exit_zscore: Z-score for take-profit (default 0.0)
            stop_zscore: Z-score for stop-loss (default 3.5)
            min_win_prob: Minimum win probability (default 0.55)
            min_correlation: Minimum correlation for pair (default 0.5)
            max_hurst: Maximum Hurst exponent (default 0.5)
        """
        self.config = config or {}
        self._set_defaults()
        
        self.price_data = None
        self.raw_price_data = None  # Stores data before global alignment
        self.pairs_stats = []
        self._pair_index = {}  # O(1) pair lookup
        self.viable_pairs = []
        self.ou_models = {}
        self.signals = {}
        self.positions = {}
        self._ou_params_cache = {}  # pair -> {ou, spread, z, data_len}
        self._window_details = {}  # pair -> list of per-window result dicts
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            'lookback_period': 'max',
            'min_half_life': 1,
            'max_half_life': 60,
            'max_adf_pvalue': 0.05,
            'entry_zscore': 2.0,
            'exit_zscore': 0.0,
            'stop_zscore': 3.5,
            'min_win_prob': 0.55,
            'min_correlation': 0.8,
            'max_hurst': 0.5,
            'fractional_kelly': 0.25,
            'max_position_pct': 0.10,
            'max_sector_exposure': 0.30,
            'drawdown_limit': 0.15,
            'robustness_windows': [500, 750, 1000, 1250, 1500, 1750, 2000],
            'min_windows_passed': 4,
            'max_data_points': 2500,
        }
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    # ------------------------------------------------------------------------
    # LAYER 1: DATA ACQUISITION
    # ------------------------------------------------------------------------
    
    def fetch_data(self, tickers: List[str], period: str = None, progress_callback=None) -> pd.DataFrame:
        """
        Fetch adjusted close prices for tickers with proper batch handling.
        
        Args:
            tickers: List of ticker symbols
            period: Data period (1y, 2y, 5y, 10y, max)
            progress_callback: Optional function(batch_num, total_batches, message)
        """
        period = period or self.config['lookback_period']
        
        # Map period to approximate trading days for truncation
        # yfinance batch downloads return union of all dates across tickers,
        # so we need to truncate to the requested period
        period_to_days = {
            '1y': 252,
            '2y': 504,
            '5y': 1260,
            '10y': 2520,
            'max': None  # No truncation
        }
        max_days = period_to_days.get(period, None)
        
        # Clean tickers
        tickers = [str(t).strip() for t in tickers if t and str(t).strip()]
        
        if not tickers:
            self.price_data = pd.DataFrame()
            return self.price_data
        
        # Download in smaller batches for reliability
        batch_size = 100
        all_data = {}
        failed_count = 0
        
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(tickers), batch_size):
            batch = tickers[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            if progress_callback:
                progress_callback(batch_num, total_batches, f"Batch {batch_num}/{total_batches}: {len(batch)} tickers")
            
            try:
                # Download with group_by='ticker' for cleaner output
                raw_data = yf.download(
                    batch, 
                    period=period, 
                    interval='1d', 
                    auto_adjust=True, 
                    progress=False,
                    threads=False,
                    group_by='ticker',
                    ignore_tz=True
                )
                
                if raw_data.empty:
                    failed_count += len(batch)
                    continue
                
                # Parse the data based on number of tickers
                if len(batch) == 1:
                    # Single ticker - simple DataFrame with OHLCV columns
                    ticker = batch[0]
                    if 'Close' in raw_data.columns:
                        all_data[ticker] = raw_data['Close'].copy()
                else:
                    # Multiple tickers with group_by='ticker'
                    # Format: (Ticker, OHLCV) MultiIndex columns
                    for ticker in batch:
                        try:
                            if ticker in raw_data.columns.get_level_values(0):
                                ticker_df = raw_data[ticker]
                                if 'Close' in ticker_df.columns:
                                    series = ticker_df['Close'].copy()
                                    if series.notna().sum() > 50:  # At least 50 valid points
                                        all_data[ticker] = series
                        except Exception:
                            failed_count += 1
                            continue
                            
            except Exception as e:
                print(f"Batch {batch_num} failed: {e}")
                failed_count += len(batch)
                continue
        
        if not all_data:
            print(f"No data loaded. All {len(tickers)} tickers failed.")
            self.price_data = pd.DataFrame()
            self.raw_price_data = pd.DataFrame()
            return self.price_data
        
        # Combine into DataFrame
        data = pd.DataFrame(all_data)
        
        # IMPORTANT: Truncate to requested period
        # yfinance batch downloads return union of all dates across all tickers,
        # so if any ticker has 3y of data, the whole DataFrame gets 3y of dates
        if max_days is not None and len(data) > max_days:
            data = data.iloc[-max_days:]
        
        # Safety cap: truncate to max_data_points
        max_pts = self.config.get('max_data_points')
        if max_pts and len(data) > max_pts:
            data = data.iloc[-max_pts:]

        # Forward fill gaps first
        data = data.ffill()

        # Store raw data for pair-specific calculations (read-only reference, no copy needed)
        self.raw_price_data = data

        # Keep NaN - correlation uses min_periods, pair testing uses pairwise dropna
        self.price_data = data

        if progress_callback:
            progress_callback(total_batches, total_batches, f"Loaded {len(data.columns)} tickers ({len(data)} days)")
        
        return data
    
    # ------------------------------------------------------------------------
    # LAYER 2: PAIR SELECTION & SCREENING
    # ------------------------------------------------------------------------
    
    def calculate_hurst_exponent(self, series: pd.Series, min_window: int = 10, 
                                   max_window: int = None, order: int = 1) -> float:
        """
        Calculate Hurst exponent using Detrended Fluctuation Analysis (DFA).
        
        DFA is superior to R/S analysis for financial time series because it:
        - Removes local polynomial trends before measuring fluctuations
        - More robust to non-stationarities  
        - Better validated for financial data
        
        Args:
            series: Time series (spread)
            min_window: Minimum window size for fluctuation calculation
            max_window: Maximum window size (default: n//4)
            order: Polynomial order for detrending (1=linear, 2=quadratic)
        
        Returns:
            H < 0.5: Mean-reverting (anti-persistent)
            H = 0.5: Random walk (Brownian motion)
            H > 0.5: Trending (persistent)
        """
        series = series.dropna().values
        n = len(series)
        
        if n < 50:
            return 0.5  # Not enough data
        
        # Integrate the series (cumulative sum of deviations from mean)
        x = series - np.mean(series)  # Just center, no integration → proper H valuesx = series - np.mean(series)  # Just center, no integration → proper H values
        
        # Set max window
        if max_window is None:
            max_window = n // 4
        max_window = min(max_window, n // 4)
        
        if max_window <= min_window:
            return 0.5
        
        # Generate window sizes (logarithmically spaced)
        window_sizes = np.unique(np.logspace(
            np.log10(min_window), 
            np.log10(max_window), 
            num=20
        ).astype(int))
        
        # Filter valid window sizes
        window_sizes = window_sizes[(window_sizes >= min_window) & (window_sizes <= max_window)]
        
        if len(window_sizes) < 4:
            return 0.5
        
        fluctuations = []

        for w in window_sizes:
            n_windows = n // w
            if n_windows < 2:
                continue

            # Vectoriserad: reshapa alla segment till en matris och batch-processa
            t = np.arange(w, dtype=np.float64)
            segments = x[:n_windows * w].reshape(n_windows, w)

            try:
                # Batch polyfit: beräkna linjär trend för alla segment samtidigt
                # y = a*t + b → Vandermonde-matris
                V = np.vstack([t, np.ones(w)]).T  # (w, 2)
                # Least squares: coeffs = (V^T V)^-1 V^T segments^T
                VtV_inv_Vt = np.linalg.lstsq(V, segments.T, rcond=None)[0]  # (2, n_windows)
                trends = (V @ VtV_inv_Vt).T  # (n_windows, w)
                detrended = segments - trends
                rms_vals = np.sqrt(np.mean(detrended ** 2, axis=1))

                # Filtrera bort ogiltiga värden
                valid = (rms_vals > 0) & np.isfinite(rms_vals)
                if valid.any():
                    fluctuations.append((w, np.mean(rms_vals[valid])))
            except (ValueError, np.linalg.LinAlgError):
                continue

        if len(fluctuations) < 4:
            return 0.5

        # Hurst exponent = slope of log(F) vs log(n)
        log_n = np.log([f[0] for f in fluctuations])
        log_f = np.log([f[1] for f in fluctuations])

        try:
            hurst, _ = np.polyfit(log_n, log_f, 1)

            # Sanity bounds
            if not np.isfinite(hurst):
                return 0.5
            return np.clip(hurst, 0.0, 1.0)
        except (ValueError, np.linalg.LinAlgError):
            return 0.5
    
    def test_cointegration(self, y: pd.Series, x: pd.Series) -> Dict:
        """
        Run Engle-Granger and Johansen cointegration tests on a pair.
        
        Returns:
            Dict with hedge_ratio, intercept, spread, eg_pvalue, johansen results
        """
        # Ensure aligned indices
        combined = pd.concat([y, x], axis=1).dropna()
        if len(combined) < 50:
            return {
                'hedge_ratio': 1.0,
                'intercept': 0.0,
                'spread': pd.Series(),
                'adf_statistic': 0,
                'adf_pvalue': 1.0,
                'johansen_trace': 0,
                'johansen_crit': 999,
                'is_cointegrated': False
            }
        
        y_aligned = combined.iloc[:, 0]
        x_aligned = combined.iloc[:, 1]
        
        # === Engle-Granger Test ===
        # Step 1: OLS regression to find hedge ratio
        try:
            x_const = sm.add_constant(x_aligned)
            model = sm.OLS(y_aligned, x_const).fit()
            hedge_ratio = float(model.params.iloc[1])
            intercept = float(model.params.iloc[0])
            
            # Step 2: ADF test on the residual spread
            spread = y_aligned - hedge_ratio * x_aligned - intercept
            adf_result = adfuller(spread.dropna(), autolag='AIC')
            eg_stat = float(adf_result[0])
            eg_pvalue = float(adf_result[1])
        except Exception:
            hedge_ratio = 1.0
            intercept = 0.0
            spread = pd.Series()
            eg_stat = 0.0
            eg_pvalue = 1.0
        
        # === Johansen Test ===
        try:
            # Reset index to ensure clean numeric index for Johansen
            data_pair = combined.reset_index(drop=True)
            johansen_result = coint_johansen(data_pair.values, 0, 1)
            trace_stat = float(johansen_result.lr1[0])
            trace_crit = float(johansen_result.cvt[0, 1])  # 95% critical value
        except Exception:
            trace_stat = 0.0
            trace_crit = 999.0
        
        return {
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'spread': spread,
            'adf_statistic': eg_stat,
            'adf_pvalue': eg_pvalue,
            'johansen_trace': trace_stat,
            'johansen_crit': trace_crit,
            'is_cointegrated': (eg_pvalue < self.config['max_adf_pvalue'] and trace_stat > trace_crit)
        }
    
    def fit_ou_parameters(self, spread: pd.Series, dt: float = 1/252) -> OUParameters:
        """Fit OU parameters to spread using AR(1) regression."""
        spread = spread.dropna()
        
        S_lag = spread.shift(1).dropna()
        S_current = spread.iloc[1:]
        
        X = sm.add_constant(S_lag)
        model = sm.OLS(S_current, X).fit()
        
        a = model.params.iloc[0]
        b = model.params.iloc[1]
        residual_std = model.resid.std()
        
        if b <= 0 or b >= 1:
            return OUParameters(
                theta=0, mu=0, sigma=0, eq_std=0, 
                half_life_days=np.inf, ar1_coef=b,
                valid=False, reason='explosive_or_unit_root'
            )
        
        theta = -np.log(b) / dt
        mu = a / (1 - b)
        sigma = residual_std * np.sqrt(2 * theta / (1 - b**2))
        eq_std = sigma / np.sqrt(2 * theta)
        half_life_days = np.log(2) / theta * 252
        
        return OUParameters(
            theta=theta, mu=mu, sigma=sigma, eq_std=eq_std,
            half_life_days=half_life_days, ar1_coef=b, valid=True
        )
    
    def _test_pair_single_window(self, t1: str, t2: str, y: pd.Series,
                                    x: pd.Series, window_size: int) -> Dict:
        """
        Test a pair on a single data window. Cheapest-first with early exit.

        Order: Correlation → OLS+Hurst → Engle-Granger → Johansen → Kalman+half-life

        Returns:
            Dict with 'passed' bool and diagnostics/OU params if passed.
        """
        fail = {
            'passed': False, 'window_size': window_size, 'failed_at': None,
            'correlation': None, 'hurst_exponent': None,
            'eg_pvalue': None, 'eg_statistic': None,
            'johansen_trace': None, 'johansen_crit': None,
            'half_life_days': None, 'kalman_stability': None,
            'kalman_innovation_ratio': None, 'kalman_regime_score': None,
            'kalman_theta_significant': None,
        }

        # --- 1. Correlation (~0.1ms) ---
        corr = y.pct_change().corr(x.pct_change())
        if np.isnan(corr) or corr < self.config['min_correlation']:
            fail['correlation'] = corr if not np.isnan(corr) else 0.0
            fail['failed_at'] = 'correlation'
            return fail
        fail['correlation'] = corr

        # --- 2. OLS hedge ratio + spread → Hurst DFA (~2ms) ---
        try:
            x_const = sm.add_constant(x)
            model = sm.OLS(y, x_const).fit()
            hedge_ratio = float(model.params.iloc[1])
            intercept = float(model.params.iloc[0])
            spread = y - hedge_ratio * x - intercept
        except Exception:
            fail['failed_at'] = 'ols'
            return fail

        if len(spread) < 50:
            fail['failed_at'] = 'spread_length'
            return fail

        try:
            hurst = self.calculate_hurst_exponent(spread)
        except Exception:
            hurst = 0.5
        fail['hurst_exponent'] = hurst
        if hurst > self.config.get('max_hurst', 0.5):
            fail['failed_at'] = 'hurst'
            return fail

        # --- 3. Engle-Granger ADF on spread (~5ms) ---
        try:
            adf_result = adfuller(spread.dropna(), autolag='AIC')
            eg_pvalue = float(adf_result[1])
            eg_stat = float(adf_result[0])
        except Exception:
            fail['failed_at'] = 'eg_cointegration'
            return fail
        fail['eg_pvalue'] = eg_pvalue
        fail['eg_statistic'] = eg_stat
        if eg_pvalue >= self.config['max_adf_pvalue']:
            fail['failed_at'] = 'eg_cointegration'
            return fail

        # --- 4. Johansen trace test (~10ms) ---
        try:
            combined = pd.concat([y, x], axis=1).reset_index(drop=True)
            johansen_result = coint_johansen(combined.values, 0, 1)
            johansen_trace = float(johansen_result.lr1[0])
            johansen_crit = float(johansen_result.cvt[0, 1])
        except Exception:
            fail['failed_at'] = 'johansen'
            return fail
        fail['johansen_trace'] = johansen_trace
        fail['johansen_crit'] = johansen_crit
        if johansen_trace <= johansen_crit:
            fail['failed_at'] = 'johansen'
            return fail

        # --- 5. Kalman OU + half-life (~50ms) ---
        ou_params = None
        try:
            kalman = KalmanOUEstimator(
                q_scale=1e-5, adaptive_q=True,
                smoother=True, cusum_threshold=4.0
            )
            kalman_result = kalman.fit(spread)
            if kalman_result.valid:
                ou_params = OUParameters(
                    theta=kalman_result.theta,
                    mu=kalman_result.mu,
                    sigma=kalman_result.sigma,
                    eq_std=kalman_result.eq_std,
                    half_life_days=kalman_result.half_life_days,
                    ar1_coef=kalman_result.ar1_coef,
                    valid=True
                )
                ou_params._kalman = kalman_result
        except Exception:
            pass

        if ou_params is None:
            try:
                ou_params = self.fit_ou_parameters(spread)
            except Exception:
                fail['failed_at'] = 'half_life'
                return fail

        if ou_params is None or not ou_params.valid:
            fail['failed_at'] = 'half_life'
            return fail

        half_life = ou_params.half_life_days
        fail['half_life_days'] = half_life
        if not (self.config['min_half_life'] <= half_life <= self.config['max_half_life']):
            fail['failed_at'] = 'half_life'
            return fail

        # Kalman diagnostics — at least 3 of 4 must pass
        kalman_res = getattr(ou_params, '_kalman', None)
        if kalman_res is not None:
            k_stability = kalman_res.param_stability
            k_inn_ratio = kalman_res.innovation_ratio
            k_regime = kalman_res.regime_change_score
            k_theta_sig = (kalman_res.theta > 1.96 * kalman_res.theta_std
                           if kalman_res.theta_std > 0 else False)
            kalman_checks = [
                k_stability > 0.4,
                0.4 <= k_inn_ratio <= 2.5,
                k_regime < 5.0,
                k_theta_sig,
            ]
            passes_kalman = sum(kalman_checks) >= 3
        else:
            passes_kalman = False
            k_stability = 0.0
            k_inn_ratio = 0.0
            k_regime = 99.0
            k_theta_sig = False

        fail['kalman_stability'] = round(k_stability, 4)
        fail['kalman_innovation_ratio'] = round(k_inn_ratio, 4)
        fail['kalman_regime_score'] = round(k_regime, 4)
        fail['kalman_theta_significant'] = bool(k_theta_sig)
        if not passes_kalman:
            fail['failed_at'] = 'kalman'
            return fail

        # --- All tests passed ---
        return {
            'passed': True,
            'window_size': window_size,
            'correlation': corr,
            'hurst_exponent': hurst,
            'eg_pvalue': eg_pvalue,
            'eg_statistic': eg_stat,
            'johansen_trace': johansen_trace,
            'johansen_crit': johansen_crit,
            'half_life_days': half_life,
            'theta': ou_params.theta,
            'mu': ou_params.mu,
            'sigma': ou_params.sigma,
            'eq_std': ou_params.eq_std,
            'ar1_coef': ou_params.ar1_coef,
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'ou_valid': True,
            'ou_params': ou_params,
            'kalman_stability': round(k_stability, 4),
            'kalman_innovation_ratio': round(k_inn_ratio, 4),
            'kalman_regime_score': round(k_regime, 4),
            'kalman_theta_significant': bool(k_theta_sig),
        }

    def _test_pair_multi_window(self, pair_info: Tuple, data: pd.DataFrame,
                                 raw_data: pd.DataFrame) -> Dict:
        """
        Test a pair across multiple lookback windows for robustness.

        OU parameters are taken from the shortest passing window (most current).
        """
        t1, t2, group_name, precomputed_corr = pair_info

        # Defaults for failure return
        fail_result = {
            'pair': f"{t1}/{t2}", 'ticker_y': t1, 'ticker_x': t2,
            'group': group_name,
            'correlation': precomputed_corr if precomputed_corr else 0,
            'hurst_exponent': 0.5, 'eg_pvalue': 1.0, 'eg_statistic': 0,
            'johansen_trace': 0, 'johansen_crit': 999,
            'half_life_days': np.inf, 'theta': 0, 'mu': 0, 'sigma': 0,
            'eq_std': 0, 'ar1_coef': 0, 'hedge_ratio': 1.0, 'intercept': 0.0,
            'ou_valid': False,
            'passes_eg': False, 'passes_johansen': False, 'passes_coint': False,
            'passes_halflife': False, 'passes_hurst': False, 'passes_kalman': False,
            'kalman_stability': 0.0, 'kalman_innovation_ratio': 0.0,
            'kalman_regime_score': 99.0, 'kalman_theta_significant': False,
            'data_length': 0, 'is_viable': False,
            'robustness_score': 0.0, 'windows_passed': 0,
            'windows_tested': 0, 'passing_windows': [],
        }

        try:
            # Align pair data
            if t1 in raw_data.columns and t2 in raw_data.columns:
                pair_data = raw_data[[t1, t2]].dropna()
            else:
                pair_data = data[[t1, t2]].dropna()

            data_length = len(pair_data)
            if data_length < 50:
                fail_result['data_length'] = data_length
                return fail_result

            windows = sorted(self.config.get('robustness_windows',
                                             [500, 750, 1000, 1250, 1500, 1750, 2000]))
            min_windows_req = self.config.get('min_windows_passed', 4)

            windows_passed = 0
            windows_tested = 0
            passing_windows = []
            best_result = None  # From shortest passing window
            all_window_results = []

            for window_size in windows:
                if data_length < window_size:
                    continue
                windows_tested += 1

                y = pair_data.iloc[-window_size:][t1]
                x = pair_data.iloc[-window_size:][t2]

                result = self._test_pair_single_window(t1, t2, y, x, window_size)
                # Store per-window result (strip heavy ou_params object)
                window_record = {k: v for k, v in result.items() if k != 'ou_params'}
                all_window_results.append(window_record)
                if result['passed']:
                    windows_passed += 1
                    passing_windows.append(window_size)
                    if best_result is None:  # Shortest = most current
                        best_result = result

            if windows_tested == 0:
                fail_result['data_length'] = data_length
                fail_result['window_details'] = all_window_results
                return fail_result

            robustness_score = windows_passed / windows_tested if windows_tested > 0 else 0.0
            is_viable = (windows_passed >= min_windows_req) and (best_result is not None)

            if best_result is not None:
                return {
                    'pair': f"{t1}/{t2}", 'ticker_y': t1, 'ticker_x': t2,
                    'group': group_name,
                    'correlation': best_result['correlation'],
                    'hurst_exponent': best_result['hurst_exponent'],
                    'eg_pvalue': best_result['eg_pvalue'],
                    'eg_statistic': best_result['eg_statistic'],
                    'johansen_trace': best_result['johansen_trace'],
                    'johansen_crit': best_result['johansen_crit'],
                    'half_life_days': best_result['half_life_days'],
                    'theta': best_result['theta'],
                    'mu': best_result['mu'],
                    'sigma': best_result['sigma'],
                    'eq_std': best_result['eq_std'],
                    'ar1_coef': best_result['ar1_coef'],
                    'hedge_ratio': best_result['hedge_ratio'],
                    'intercept': best_result['intercept'],
                    'ou_valid': True,
                    'passes_eg': True, 'passes_johansen': True,
                    'passes_coint': True, 'passes_halflife': True,
                    'passes_hurst': True, 'passes_kalman': True,
                    'kalman_stability': best_result['kalman_stability'],
                    'kalman_innovation_ratio': best_result['kalman_innovation_ratio'],
                    'kalman_regime_score': best_result['kalman_regime_score'],
                    'kalman_theta_significant': best_result['kalman_theta_significant'],
                    'data_length': data_length,
                    'is_viable': is_viable,
                    'robustness_score': robustness_score,
                    'windows_passed': windows_passed,
                    'windows_tested': windows_tested,
                    'passing_windows': passing_windows,
                    'window_details': all_window_results,
                }
            else:
                fail_result['data_length'] = data_length
                fail_result['windows_tested'] = windows_tested
                fail_result['robustness_score'] = 0.0
                fail_result['window_details'] = all_window_results
                return fail_result

        except Exception as e:
            fail_result['error'] = str(e)[:100]
            return fail_result

    def _test_single_pair(self, pair_info: Tuple, data: pd.DataFrame, raw_data: pd.DataFrame) -> Dict:
        """
        Test a single pair for cointegration using Option B:
        Engle-Granger + Johansen confirmation.
        
        Viability requires:
        - Engle-Granger p-value < 0.05
        - Johansen trace statistic > critical value (95%)
        - Half-life within acceptable range
        - Hurst exponent <= 0.5
        
        Args:
            pair_info: Tuple of (t1, t2, group_name, precomputed_corr)
            data: Price data DataFrame
            raw_data: Raw price data DataFrame
            
        Returns:
            Dict with pair test results
        """
        t1, t2, group_name, precomputed_corr = pair_info
        
        try:
            # Use raw_price_data for pair-specific alignment
            if t1 in raw_data.columns and t2 in raw_data.columns:
                pair_data = raw_data[[t1, t2]].dropna()
                y = pair_data[t1]
                x = pair_data[t2]
            else:
                y = data[t1]
                x = data[t2]
            
            data_length = len(y)
            
            # Return correlation
            if precomputed_corr is not None:
                corr = precomputed_corr
            else:
                corr = y.pct_change().corr(x.pct_change())
            
            # Initialize defaults
            hurst = 0.5
            ou_params = None
            eg_pvalue = 1.0
            eg_stat = 0.0
            johansen_trace = 0.0
            johansen_crit = 999.0
            hedge_ratio = 1.0
            intercept = 0.0
            spread = pd.Series()
            
            # Only run tests if we have enough data
            if data_length >= 50:
                # === Run full cointegration tests (Engle-Granger + Johansen) ===
                try:
                    coint_result = self.test_cointegration(y, x)
                    eg_pvalue = coint_result['adf_pvalue']
                    eg_stat = coint_result['adf_statistic']
                    johansen_trace = coint_result['johansen_trace']
                    johansen_crit = coint_result['johansen_crit']
                    hedge_ratio = coint_result['hedge_ratio']
                    intercept = coint_result['intercept']
                    spread = coint_result.get('spread', pd.Series())
                except Exception:
                    pass
                
                # === Hurst exponent on the EG spread ===
                if len(spread) > 50:
                    try:
                        hurst = self.calculate_hurst_exponent(spread)
                    except Exception:
                        hurst = 0.5
                
                # === OU parameters on the Engle-Granger spread ===
                # Primary: Kalman filter (consistent with OU Analytics tab)
                # Fallback: AR(1) OLS
                if len(spread) > 50:
                    try:
                        kalman = KalmanOUEstimator(
                            q_scale=1e-5, adaptive_q=True,
                            smoother=True, cusum_threshold=4.0
                        )
                        kalman_result = kalman.fit(spread)
                        if kalman_result.valid:
                            ou_params = OUParameters(
                                theta=kalman_result.theta,
                                mu=kalman_result.mu,
                                sigma=kalman_result.sigma,
                                eq_std=kalman_result.eq_std,
                                half_life_days=kalman_result.half_life_days,
                                ar1_coef=kalman_result.ar1_coef,
                                valid=True
                            )
                            ou_params._kalman = kalman_result
                    except Exception:
                        pass
                    # Fallback to OLS if Kalman failed
                    if ou_params is None:
                        try:
                            ou_params = self.fit_ou_parameters(spread)
                        except Exception:
                            ou_params = None

            # === Determine what passed/failed ===
            
            # Engle-Granger test: p-value < threshold
            passes_eg = eg_pvalue < self.config['max_adf_pvalue']
            
            # Johansen test: trace statistic > critical value
            passes_johansen = johansen_trace > johansen_crit
            
            # Combined cointegration: BOTH tests must pass (Option B)
            passes_coint = passes_eg and passes_johansen
            
            # Hurst exponent: must indicate mean reversion (< 0.5)
            passes_hurst = hurst <= self.config.get('max_hurst', 0.5)
            
            # OU parameters and half-life
            if ou_params is not None and ou_params.valid:
                passes_halflife = (self.config['min_half_life'] <= ou_params.half_life_days <= 
                                  self.config['max_half_life'])
                half_life = ou_params.half_life_days
                theta = ou_params.theta
                mu = ou_params.mu
                sigma = ou_params.sigma
                eq_std = ou_params.eq_std
                ar1_coef = ou_params.ar1_coef
                ou_valid = True
            else:
                passes_halflife = False
                half_life = np.inf
                theta = 0
                mu = 0
                sigma = 0
                eq_std = 0
                ar1_coef = 0
                ou_valid = False
            
            # === Kalman diagnostics (integrated — no separate post-screening step) ===
            kalman_result = getattr(ou_params, '_kalman', None) if ou_params else None
            if kalman_result is not None:
                k_stability = kalman_result.param_stability
                k_inn_ratio = kalman_result.innovation_ratio
                k_regime = kalman_result.regime_change_score
                k_theta_sig = (kalman_result.theta > 1.96 * kalman_result.theta_std
                               if kalman_result.theta_std > 0 else False)
                kalman_checks = [
                    k_stability > 0.4,
                    0.4 <= k_inn_ratio <= 2.5,
                    k_regime < 5.0,
                    k_theta_sig,
                ]
                passes_kalman = sum(kalman_checks) >= 3
            else:
                k_stability = 0.0
                k_inn_ratio = 0.0
                k_regime = 99.0
                k_theta_sig = False
                passes_kalman = False

            # === Final viability ===
            # EG + Johansen + half-life + Hurst + Kalman validation
            is_viable = (passes_coint and passes_halflife and ou_valid
                         and passes_hurst and passes_kalman)

            return {
                'pair': f"{t1}/{t2}",
                'ticker_y': t1,
                'ticker_x': t2,
                'group': group_name,
                'correlation': corr,
                'hurst_exponent': hurst,
                'eg_pvalue': eg_pvalue,
                'eg_statistic': eg_stat,
                'johansen_trace': johansen_trace,
                'johansen_crit': johansen_crit,
                'half_life_days': half_life,
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'eq_std': eq_std,
                'ar1_coef': ar1_coef,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'ou_valid': ou_valid,
                'passes_eg': passes_eg,
                'passes_johansen': passes_johansen,
                'passes_coint': passes_coint,
                'passes_halflife': passes_halflife,
                'passes_hurst': passes_hurst,
                'passes_kalman': passes_kalman,
                'kalman_stability': round(k_stability, 4),
                'kalman_innovation_ratio': round(k_inn_ratio, 4),
                'kalman_regime_score': round(k_regime, 4),
                'kalman_theta_significant': bool(k_theta_sig),
                'data_length': data_length,
                'is_viable': is_viable
            }
        except Exception as e:
            return {
                'pair': f"{t1}/{t2}",
                'ticker_y': t1,
                'ticker_x': t2,
                'group': group_name,
                'correlation': precomputed_corr if precomputed_corr else 0,
                'hurst_exponent': 0.5,
                'eg_pvalue': 1.0,
                'eg_statistic': 0,
                'johansen_trace': 0,
                'johansen_crit': 999,
                'half_life_days': np.inf,
                'theta': 0,
                'mu': 0,
                'sigma': 0,
                'eq_std': 0,
                'ar1_coef': 0,
                'hedge_ratio': 1.0,
                'intercept': 0.0,
                'ou_valid': False,
                'passes_eg': False,
                'passes_johansen': False,
                'passes_coint': False,
                'passes_halflife': False,
                'passes_hurst': False,
                'passes_kalman': False,
                'kalman_stability': 0.0,
                'kalman_innovation_ratio': 0.0,
                'kalman_regime_score': 99.0,
                'kalman_theta_significant': False,
                'data_length': 0,
                'is_viable': False,
                'error': str(e)[:100]
            }
    
    def _kalman_validate_pair(self, row, data: pd.DataFrame,
                             raw_data: pd.DataFrame) -> dict:
        """Run Kalman filter validation on a viable pair.

        Returns dict with Kalman diagnostic columns.
        """
        defaults = {
            'kalman_stability': 0.0,
            'kalman_innovation_ratio': 0.0,
            'kalman_regime_score': 99.0,
            'kalman_theta_significant': False,
            'passes_kalman': False,
        }

        try:
            t1 = row.ticker_y
            t2 = row.ticker_x

            if t1 in raw_data.columns and t2 in raw_data.columns:
                pair_data = raw_data[[t1, t2]].dropna()
            else:
                pair_data = data[[t1, t2]].dropna()

            if len(pair_data) < 60:
                return defaults

            y = pair_data[t1]
            x = pair_data[t2]

            # Compute EG spread using hedge ratio from screening
            hedge = getattr(row, 'hedge_ratio', 1.0)
            intercept = getattr(row, 'intercept', 0.0)
            spread = y - hedge * x - intercept

            # Run Kalman filter
            kalman = KalmanOUEstimator(
                q_scale=1e-5, adaptive_q=True,
                smoother=True, cusum_threshold=4.0
            )
            result = kalman.fit(spread)

            if not result.valid:
                return defaults

            stability = result.param_stability
            inn_ratio = result.innovation_ratio
            regime = result.regime_change_score

            # Theta significance: 95% CI excludes 0
            theta_sig = (result.theta > 1.96 * result.theta_std
                         if result.theta_std > 0 else False)

            # Relaxed criteria — at least 3 of 4 must pass
            checks = [
                stability > 0.4,
                0.4 <= inn_ratio <= 2.5,
                regime < 5.0,
                theta_sig,
            ]
            passes = sum(checks) >= 3

            pair_name = getattr(row, 'pair', '?')
            print(f"  [Kalman] {pair_name}: stab={stability:.2f} ir={inn_ratio:.2f} "
                  f"regime={regime:.1f} theta_sig={theta_sig} → {'PASS' if passes else 'FAIL'}")

            return {
                'kalman_stability': round(stability, 4),
                'kalman_innovation_ratio': round(inn_ratio, 4),
                'kalman_regime_score': round(regime, 4),
                'kalman_theta_significant': bool(theta_sig),
                'passes_kalman': bool(passes),
            }
        except Exception as e:
            print(f"Kalman validation failed for {getattr(row, 'pair', '?')}: {e}")
            return defaults

    def screen_pairs(self, tickers: List[str] = None,
                     sector_groups: Dict[str, List[str]] = None,
                     use_exchange_groups: bool = True,
                     correlation_prefilter: bool = True,
                     min_group_size: int = 2,
                     progress_callback = None,
                     use_parallel: bool = True) -> pd.DataFrame:
        """
        Screen all pairs for cointegration and OU viability.
        
        Args:
            use_parallel: If True, use parallel processing for pair testing (default: True)
        """
        if self.price_data is None or self.price_data.empty:
            if tickers is None:
                raise ValueError("No price data loaded. Call fetch_data first or provide tickers.")
            self.fetch_data(tickers)
        
        data = self.price_data
        available_tickers = data.columns.tolist()
        available_tickers_set = set(available_tickers)

        if progress_callback:
            progress_callback('grouping', 0, 0, f"Starting with {len(available_tickers)} tickers")

        groups = {'ALL': available_tickers}

        # Filter to groups with enough tickers (O(1) set lookup)
        groups = {k: [t for t in v if t in available_tickers_set]
                  for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}
        
        if progress_callback:
            total_tickers = sum(len(v) for v in groups.values())
            progress_callback('grouping', 0, 0, f"Found {len(groups)} groups with {total_tickers} tickers")
        
        # Generate candidate pairs
        candidate_pairs = []
        
        if correlation_prefilter and len(available_tickers) > 1:
            if progress_callback:
                progress_callback('correlation', 0, 1, "Computing correlation matrix...")
            
            # Compute returns for correlation - DON'T drop NaN globally
            # pandas .corr() handles pairwise complete observations automatically
            returns = data.pct_change()
            
            # Fixed min_periods for deterministic results across scan sizes
            # 100 ≈ ~5 months of daily data — enough for reliable correlation
            min_obs_for_corr = 100
            
            # Check we have enough data
            valid_return_counts = returns.count()
            
            for group_idx, (group_name, group_tickers) in enumerate(groups.items()):
                if len(group_tickers) < 2:
                    continue
                
                # Get returns for this group - only tickers with enough data
                valid_tickers = [t for t in group_tickers 
                               if t in returns.columns and valid_return_counts.get(t, 0) >= min_obs_for_corr]
                if len(valid_tickers) < 2:
                    continue
                
                group_returns = returns[valid_tickers]
                # Use pairwise complete observations with scaled min_periods
                corr_matrix = group_returns.corr(min_periods=min_obs_for_corr)
                
                if progress_callback:
                    progress_callback('correlation', group_idx + 1, len(groups), 
                                    f"Group: {group_name} ({len(valid_tickers)} tickers, min_periods={min_obs_for_corr})")
                
                # Find pairs above correlation threshold (.values för snabb numpy-access)
                corr_values = corr_matrix.values
                corr_columns = corr_matrix.columns
                min_corr = self.config['min_correlation']
                for i in range(len(corr_columns)):
                    for j in range(i + 1, len(corr_columns)):
                        corr = corr_values[i, j]
                        if not np.isnan(corr) and corr >= min_corr:
                            ticker_a = corr_columns[i]
                            ticker_b = corr_columns[j]
                            # Always order alphabetically for consistency
                            t1, t2 = (ticker_a, ticker_b) if ticker_a < ticker_b else (ticker_b, ticker_a)
                            candidate_pairs.append((t1, t2, group_name, corr))
        else:
            # No correlation prefilter - generate all pairs within groups
            for group_name, group_tickers in groups.items():
                valid_tickers = [t for t in group_tickers if t in available_tickers]
                for i in range(len(valid_tickers)):
                    for j in range(i + 1, len(valid_tickers)):
                        ticker_a, ticker_b = valid_tickers[i], valid_tickers[j]
                        # Always order alphabetically for consistency
                        t1, t2 = (ticker_a, ticker_b) if ticker_a < ticker_b else (ticker_b, ticker_a)
                        candidate_pairs.append((t1, t2, group_name, None))
        
        if progress_callback:
            progress_callback('screening', 0, max(1, len(candidate_pairs)), 
                            f"Testing {len(candidate_pairs)} candidate pairs...")
        
        results = []
        raw_data = getattr(self, 'raw_price_data', data)
        
        # Use parallel processing for large number of pairs
        if use_parallel and len(candidate_pairs) > 10:
            # Parallel execution using ThreadPoolExecutor
            completed = 0
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = {
                    executor.submit(self._test_pair_multi_window, pair_info, data, raw_data): pair_info 
                    for pair_info in candidate_pairs
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if progress_callback and (completed % 10 == 0 or completed == len(candidate_pairs)):
                        progress_callback('screening', completed, len(candidate_pairs), 
                                        f"Tested {completed}/{len(candidate_pairs)} pairs")
        else:
            # Sequential execution for small number of pairs
            for pair_idx, pair_info in enumerate(candidate_pairs):
                t1, t2, group_name, precomputed_corr = pair_info
                
                if progress_callback and (pair_idx % 5 == 0 or pair_idx == len(candidate_pairs) - 1):
                    progress_callback('screening', pair_idx + 1, len(candidate_pairs), 
                                    f"Testing {t1}/{t2}")
                
                result = self._test_pair_multi_window(pair_info, data, raw_data)
                results.append(result)

        results_df = pd.DataFrame(results)

        # Extract window_details into separate dict before sorting
        self._window_details = {}
        if len(results_df) > 0 and 'window_details' in results_df.columns:
            for _, row in results_df.iterrows():
                if row.get('pair') and isinstance(row.get('window_details'), list):
                    self._window_details[row['pair']] = row['window_details']
            results_df = results_df.drop(columns=['window_details'])

        if len(results_df) > 0:
            # Sort by robustness_score descending (most robust first)
            if 'robustness_score' in results_df.columns:
                results_df = results_df.sort_values('robustness_score', ascending=False)
            else:
                results_df = results_df.sort_values('half_life_days')

        self.pairs_stats = results_df
        # O(1) lookup index: pair string → row dict
        self._pair_index = {}
        if len(results_df) > 0:
            for row in results_df.itertuples():
                self._pair_index[row.pair] = row
        self.viable_pairs = results_df[results_df['is_viable']].copy() if len(results_df) > 0 else pd.DataFrame()
        self._ou_params_cache = {}  # Invalidate OU cache on new scan

        print(f"[Scan] {len(self.viable_pairs)} viable pairs (Kalman-integrated)")

        # Build OU models for viable pairs
        for row in self.viable_pairs.itertuples():
            self.ou_models[row.pair] = OUProcess(
                theta=row.theta,
                mu=row.mu,
                sigma=row.sigma
            )

        if progress_callback:
            progress_callback('complete', len(results_df), len(results_df),
                            f"Found {len(self.viable_pairs)} viable pairs from {len(results_df)} tested ({len(candidate_pairs)} candidates)")
        
        return results_df
    
    # ------------------------------------------------------------------------
    # LAYER 3: SIGNAL GENERATION
    # ------------------------------------------------------------------------
    
    def get_current_zscore(self, ticker_y: str, ticker_x: str, 
                           ou_params: OUParameters = None) -> Tuple[float, float]:
        """Get current z-score for a pair."""
        pair_data = self.price_data[[ticker_y, ticker_x]].dropna()
        if len(pair_data) < 10:
            return 0.0, 0.0
        log_spread = np.log(pair_data[ticker_y] / pair_data[ticker_x])
        current_spread = log_spread.iloc[-1]
        
        if ou_params is None:
            ou_params = self.fit_ou_parameters(log_spread)
        
        zscore = (current_spread - ou_params.mu) / ou_params.eq_std
        return zscore, current_spread
    
    def generate_signal(self, pair: str, ou_model: OUProcess = None) -> TradeSignal:
        """Generate trade signal for a pair with OU probability calculations."""
        t1, t2 = pair.split('/')
        
        # Use pairwise aligned data
        pair_data = self.price_data[[t1, t2]].dropna()
        if len(pair_data) < 50:
            return TradeSignal(
                pair=pair, signal_type='NO_TRADE', current_zscore=0,
                entry_spread=0, take_profit_spread=0, stop_loss_spread=0,
                win_probability=0, expected_pnl=0, kelly_fraction=0,
                risk_reward=0, avg_holding_days=0, confidence='LOW'
            )
        
        if ou_model is None:
            ou_model = self.ou_models.get(pair)

        # Beräkna log_spread en gång och återanvänd
        log_spread = np.log(pair_data[t1] / pair_data[t2])

        if ou_model is None:
            # Fit on the fly
            ou_params = self.fit_ou_parameters(log_spread)
            if not ou_params.valid:
                return TradeSignal(
                    pair=pair, signal_type='NO_TRADE', current_zscore=0,
                    entry_spread=0, take_profit_spread=0, stop_loss_spread=0,
                    win_probability=0, expected_pnl=0, kelly_fraction=0,
                    risk_reward=0, avg_holding_days=0, confidence='LOW'
                )
            ou_model = OUProcess(ou_params.theta, ou_params.mu, ou_params.sigma)

        current_spread = log_spread.iloc[-1]
        current_z = ou_model.zscore(current_spread)
        
        entry_z = self.config['entry_zscore']
        exit_z = self.config['exit_zscore']
        stop_z = self.config['stop_zscore']
        
        # No signal if not at entry threshold
        if abs(current_z) < entry_z:
            return TradeSignal(
                pair=pair, signal_type='NO_TRADE', current_zscore=current_z,
                entry_spread=current_spread, take_profit_spread=0, stop_loss_spread=0,
                win_probability=0, expected_pnl=0, kelly_fraction=0,
                risk_reward=0, avg_holding_days=0, confidence='LOW'
            )
        
        # Determine direction and levels
        if current_z > 0:
            signal_type = 'SHORT_SPREAD'
            take_profit = ou_model.spread_from_z(exit_z)
            stop_loss = ou_model.spread_from_z(stop_z)
        else:
            signal_type = 'LONG_SPREAD'
            take_profit = ou_model.spread_from_z(-exit_z)
            stop_loss = ou_model.spread_from_z(-stop_z)
        
        # Calculate expected outcome
        metrics = ou_model.expected_pnl(current_spread, take_profit, stop_loss)
        
        # Determine confidence
        if metrics['win_prob'] >= 0.65 and metrics['kelly_fraction'] >= 0.15:
            confidence = 'HIGH'
        elif metrics['win_prob'] >= self.config['min_win_prob']:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
            signal_type = 'NO_TRADE'
        
        avg_holding = (metrics['avg_win_days'] * metrics['win_prob'] + 
                      metrics['avg_loss_days'] * metrics['loss_prob'])
        if np.isnan(avg_holding):
            avg_holding = ou_model.half_life_days() * 2
        
        return TradeSignal(
            pair=pair,
            signal_type=signal_type,
            current_zscore=current_z,
            entry_spread=current_spread,
            take_profit_spread=take_profit,
            stop_loss_spread=stop_loss,
            win_probability=metrics['win_prob'],
            expected_pnl=metrics['expected_pnl'],
            kelly_fraction=metrics['kelly_fraction'],
            risk_reward=metrics['risk_reward'],
            avg_holding_days=avg_holding,
            confidence=confidence
        )
    
    def generate_all_signals(self) -> Dict[str, TradeSignal]:
        """Generate signals for all viable pairs."""
        self.signals = {}
        
        for pair in self.ou_models.keys():
            signal = self.generate_signal(pair)
            self.signals[pair] = signal
        
        return self.signals
    
    def get_active_signals(self) -> List[TradeSignal]:
        """Get all pairs with active trade signals."""
        return [s for s in self.signals.values() 
                if s.signal_type in ['LONG_SPREAD', 'SHORT_SPREAD']]
    
    # ------------------------------------------------------------------------
    # LAYER 4: POSITION SIZING
    # ------------------------------------------------------------------------
    
    def calculate_position_size(self, signal: TradeSignal, 
                                portfolio_value: float,
                                current_positions: int = 0) -> Dict:
        """
        Calculate position size using fractional Kelly criterion
        with volatility targeting.
        """
        if signal.signal_type == 'NO_TRADE':
            return {'shares_y': 0, 'shares_x': 0, 'dollar_size': 0}
        
        # Fractional Kelly
        kelly = signal.kelly_fraction * self.config['fractional_kelly']
        
        # Cap at max position size
        kelly = min(kelly, self.config['max_position_pct'])
        
        # Dollar allocation
        dollar_size = portfolio_value * kelly
        
        # Get current prices (last valid price)
        t1, t2 = signal.pair.split('/')
        price_y = self.price_data[t1].dropna().iloc[-1] if t1 in self.price_data.columns else 0
        price_x = self.price_data[t2].dropna().iloc[-1] if t2 in self.price_data.columns else 0
        
        if price_y <= 0 or price_x <= 0:
            return {'shares_y': 0, 'shares_x': 0, 'dollar_size': 0}
        
        # For log spread: equal dollar exposure on each leg
        shares_y = int(dollar_size / 2 / price_y)
        shares_x = int(dollar_size / 2 / price_x)
        
        # Adjust direction
        if signal.signal_type == 'SHORT_SPREAD':
            shares_y = -shares_y  # Short Y
            shares_x = abs(shares_x)  # Long X
        else:
            shares_y = abs(shares_y)  # Long Y
            shares_x = -shares_x  # Short X
        
        return {
            'shares_y': shares_y,
            'shares_x': shares_x,
            'dollar_size': dollar_size,
            'kelly_fraction': kelly,
            'price_y': price_y,
            'price_x': price_x
        }
    
    # ------------------------------------------------------------------------
    # LAYER 5: RISK MANAGEMENT
    # ------------------------------------------------------------------------
    
    def calculate_portfolio_risk(self, positions: Dict) -> Dict:
        """Calculate portfolio-level risk metrics."""
        if not positions:
            return {
                'gross_exposure': 0,
                'net_exposure': 0,
                'num_positions': 0,
                'sector_exposures': {},
                'total_pnl': 0,
                'max_drawdown': 0
            }
        
        gross = sum(abs(p.get('dollar_size', 0)) for p in positions.values())
        
        sector_exp = {}
        for pair, pos in positions.items():
            sector = pos.get('sector', 'Unknown')
            sector_exp[sector] = sector_exp.get(sector, 0) + abs(pos.get('dollar_size', 0))
        
        return {
            'gross_exposure': gross,
            'num_positions': len(positions),
            'sector_exposures': sector_exp,
        }
    
    def check_position_risk(self, pair: str, entry_info: Dict, 
                            current_spread: float) -> PositionRisk:
        """Monitor risk for an open position."""
        ou_model = self.ou_models.get(pair)
        if ou_model is None:
            return None
        
        current_z = ou_model.zscore(current_spread)
        entry_z = ou_model.zscore(entry_info['entry_spread'])
        
        # P&L calculation (simplified for log spread)
        if entry_info['signal_type'] == 'SHORT_SPREAD':
            pnl = entry_info['entry_spread'] - current_spread
        else:
            pnl = current_spread - entry_info['entry_spread']
        
        pnl_pct = pnl / abs(entry_info['entry_spread']) * 100
        
        # Distance to targets
        dist_to_target = abs(current_z - self.config['exit_zscore'])
        dist_to_stop = abs(self.config['stop_zscore'] - abs(current_z))
        
        # Risk status
        if abs(current_z) >= self.config['stop_zscore']:
            status = 'CRITICAL'
        elif abs(current_z) >= self.config['stop_zscore'] - 0.5:
            status = 'WARNING'
        else:
            status = 'OK'
        
        days_held = (datetime.now() - entry_info.get('entry_time', datetime.now())).days
        
        return PositionRisk(
            pair=pair,
            current_pnl=pnl,
            current_pnl_pct=pnl_pct,
            days_held=days_held,
            current_zscore=current_z,
            distance_to_stop=dist_to_stop,
            distance_to_target=dist_to_target,
            max_drawdown=0,  # Would need position history to calculate
            risk_status=status
        )
    
    # ------------------------------------------------------------------------
    # LAYER 6: ANALYSIS & REPORTING
    # ------------------------------------------------------------------------
    
    def get_window_details(self, pair: str) -> list:
        """Return per-window test results for a pair."""
        return self._window_details.get(pair, [])

    def get_pair_ou_params(self, pair: str, use_raw_data: bool = True,
                           method: str = 'kalman',
                           window_size: int = None) -> Tuple[OUProcess, pd.Series, float]:
        """
        Get OU parameters for a specific pair using Engle-Granger spread.

        Supports multiple estimation methods:
        - 'kalman': Kalman filter with adaptive Q and RTS smoother (default, most sophisticated)
        - 'adaptive_window': Two-pass half-life-based window selection
        - 'ols': Simple AR(1) OLS on full data (fastest, least adaptive)

        IMPORTANT: This uses the EG spread (Y - β*X - α) with the hedge ratio
        from screening, NOT the simple log spread. This ensures consistency
        between scanner results and analytics display.

        Args:
            pair: Pair string like "TICKER1/TICKER2"
            use_raw_data: If True, use raw_price_data (before global alignment)
            method: Estimation method ('kalman', 'adaptive_window', 'ols')
            window_size: If set, use exactly this many data points (overrides
                         automatic shortest-window selection)

        Returns:
            Tuple of (OUProcess, eg_spread_series, current_zscore)
        """
        t1, t2 = pair.split('/')

        # Use raw data if available and requested, otherwise fall back to price_data
        if use_raw_data and self.raw_price_data is not None and len(self.raw_price_data) > 0:
            data_source = self.raw_price_data
        else:
            data_source = self.price_data

        if t1 not in data_source.columns or t2 not in data_source.columns:
            raise ValueError(f"Tickers {t1} or {t2} not found in data")

        # Get pair-specific aligned data
        pair_data = data_source[[t1, t2]].dropna()

        if window_size is not None:
            # Explicit window size override
            if len(pair_data) > window_size:
                pair_data = pair_data.iloc[-window_size:]
        else:
            # Truncate to shortest passing robustness window for consistency
            # with scanner results (scanner OU params come from shortest window)
            pair_row = self._pair_index.get(pair)
            if pair_row is not None:
                pw = getattr(pair_row, 'passing_windows', None)
                if pw and len(pw) > 0:
                    shortest_window = min(pw)
                    if len(pair_data) > shortest_window:
                        pair_data = pair_data.iloc[-shortest_window:]

        # Check OU params cache (same data length = same result)
        cache_key = (pair, method, use_raw_data, window_size)
        if cache_key in self._ou_params_cache:
            cached = self._ou_params_cache[cache_key]
            if cached['data_len'] == len(pair_data):
                return cached['ou'], cached['spread'], cached['z']
        
        if len(pair_data) < 50:
            raise ValueError(f"Insufficient data for pair {pair}: only {len(pair_data)} points")
        
        y = pair_data[t1]
        x = pair_data[t2]
        
        # Get hedge ratio and intercept from stored screening results (O(1) lookup)
        hedge_ratio = 1.0
        intercept = 0.0

        if pair_row is None:
            pair_row = self._pair_index.get(pair)
        if pair_row is not None:
            hedge_ratio = float(pair_row.hedge_ratio)
            intercept = float(pair_row.intercept)

        # If no stored hedge ratio, calculate fresh using OLS
        if hedge_ratio == 1.0 and intercept == 0.0:
            try:
                x_const = sm.add_constant(x)
                model = sm.OLS(y, x_const).fit()
                hedge_ratio = float(model.params.iloc[1])
                intercept = float(model.params.iloc[0])
            except Exception:
                pass  # Keep defaults
        
        # Calculate Engle-Granger spread: Y - β*X - α
        eg_spread = y - hedge_ratio * x - intercept
        
        # ===== OU PARAMETER ESTIMATION =====
        ou_params = None
        kalman_result = None
        
        if method == 'kalman':
            # PRIMARY: Kalman filter with adaptive Q and RTS smoother
            try:
                kalman = KalmanOUEstimator(
                    q_scale=1e-5,
                    adaptive_q=True,
                    smoother=True,
                    min_warmup=30
                )
                kalman_result = kalman.fit(eg_spread)
                
                if kalman_result.valid:
                    ou_params = OUParameters(
                        theta=kalman_result.theta,
                        mu=kalman_result.mu,
                        sigma=kalman_result.sigma,
                        eq_std=kalman_result.eq_std,
                        half_life_days=kalman_result.half_life_days,
                        ar1_coef=kalman_result.ar1_coef,
                        valid=True
                    )
                    
                    # Store Kalman diagnostics on the OUParameters for access upstream
                    ou_params._kalman = kalman_result
                    
            except Exception as e:
                print(f"Kalman filter failed for {pair}: {e}, falling back to adaptive window")
                method = 'adaptive_window'
        
        if method == 'adaptive_window' or (method == 'kalman' and ou_params is None):
            # FALLBACK: Two-pass adaptive window estimation
            full_ou = self.fit_ou_parameters(eg_spread)
            
            if full_ou.valid and 2 < full_ou.half_life_days < 120:
                optimal_window = int(4 * full_ou.half_life_days)
                optimal_window = max(30, min(optimal_window, len(eg_spread)))
                windowed_spread = eg_spread.iloc[-optimal_window:]
                ou_params = self.fit_ou_parameters(windowed_spread)
                
                if not ou_params.valid:
                    ou_params = full_ou
            else:
                ou_params = full_ou
        
        if method == 'ols' and ou_params is None:
            # SIMPLEST: Standard OLS on full data
            ou_params = self.fit_ou_parameters(eg_spread)
        
        if ou_params is None or not ou_params.valid:
            raise ValueError(f"Could not fit valid OU model for {pair} (method={method})")
        
        # Create OUProcess
        ou = OUProcess(ou_params.theta, ou_params.mu, ou_params.sigma)
        
        # Store Kalman result on OUProcess for dashboard access
        if kalman_result is not None:
            ou._kalman = kalman_result
        
        # Calculate current Z-score
        # Prefer Kalman time-varying z-score (consistent with what's plotted)
        # Fall back to static z-score if Kalman history not available
        if (kalman_result is not None 
            and kalman_result.zscore_history is not None 
            and len(kalman_result.zscore_history) > 0
            and not np.isnan(kalman_result.zscore_history[-1])):
            current_z = float(kalman_result.zscore_history[-1])
        else:
            current_spread = eg_spread.iloc[-1]
            current_z = ou.zscore(current_spread)
        
        # Cache result for subsequent calls with same data
        self._ou_params_cache[cache_key] = {
            'ou': ou, 'spread': eg_spread, 'z': current_z,
            'data_len': len(pair_data)
        }

        return ou, eg_spread, current_z

    def get_spread_history(self, pair: str, use_raw_data: bool = True) -> pd.DataFrame:
        """Get historical spread data with z-scores and bands using Engle-Granger spread."""
        t1, t2 = pair.split('/')
        
        # Use raw data if available and requested
        if use_raw_data and self.raw_price_data is not None and len(self.raw_price_data) > 0:
            data_source = self.raw_price_data
        else:
            data_source = self.price_data
        
        # Get pair-specific aligned data
        pair_data = data_source[[t1, t2]].dropna()
        y = pair_data[t1]
        x = pair_data[t2]
        
        # Get hedge ratio and intercept from stored results (O(1) lookup)
        hedge_ratio = 1.0
        intercept = 0.0

        pair_row = self._pair_index.get(pair)
        if pair_row is not None:
            hedge_ratio = float(pair_row.hedge_ratio)
            intercept = float(pair_row.intercept)

        # Calculate EG spread
        eg_spread = y - hedge_ratio * x - intercept
        
        # Get or create OU model using pair-specific data
        try:
            ou_model, _, _ = self.get_pair_ou_params(pair, use_raw_data=use_raw_data)
        except (ValueError, KeyError, Exception):
            ou_params = self.fit_ou_parameters(eg_spread)
            if not ou_params.valid:
                return pd.DataFrame()
            ou_model = OUProcess(ou_params.theta, ou_params.mu, ou_params.sigma)
        
        df = pd.DataFrame({
            'spread': eg_spread,
            'zscore': (eg_spread - ou_model.mu) / ou_model.eq_std,
            'mu': ou_model.mu,
            'upper_1': ou_model.mu + ou_model.eq_std,
            'lower_1': ou_model.mu - ou_model.eq_std,
            'upper_2': ou_model.mu + 2 * ou_model.eq_std,
            'lower_2': ou_model.mu - 2 * ou_model.eq_std,
            'upper_3': ou_model.mu + 3 * ou_model.eq_std,
            'lower_3': ou_model.mu - 3 * ou_model.eq_std,
        })
        
        return df
    
    def estimate_optimal_window(self, spread: pd.Series, 
                                min_window: int = 20,
                                max_window: int = 252,
                                method: str = 'half_life') -> Dict:
        """
        Estimate optimal rolling window size for a spread using quantitative methods.
        
        Methods:
        - 'half_life': Use 3-5x the estimated half-life (most common for OU)
        - 'variance_ratio': Find window where variance ratio stabilizes
        - 'cross_validation': Use predictive accuracy to select window
        - 'aic': Use Akaike Information Criterion
        
        Args:
            spread: The spread series to analyze
            min_window: Minimum window to consider (default: 20 days)
            max_window: Maximum window to consider (default: 252 days = 1 year)
            method: Selection method ('half_life', 'variance_ratio', 'cross_validation', 'aic')
            
        Returns:
            Dict with optimal_window, method_used, confidence, diagnostics
        """
        spread = spread.dropna()
        n = len(spread)
        
        if n < min_window * 2:
            return {
                'optimal_window': min_window,
                'method_used': 'minimum_default',
                'confidence': 'low',
                'reason': 'insufficient_data',
                'diagnostics': {}
            }
        
        max_window = min(max_window, n // 2)  # Can't use more than half the data
        
        if method == 'half_life':
            # Method 1: Base window on estimated half-life
            # Rule of thumb: Use 3-5 half-lives for stable parameter estimation
            ou = self.fit_ou_parameters(spread)
            
            if ou.valid and ou.half_life_days > 0 and ou.half_life_days < np.inf:
                half_life = ou.half_life_days
                
                # Optimal window = 4 * half_life (captures ~94% of mean reversion)
                # But also consider statistical stability
                optimal = int(4 * half_life)
                
                # Ensure minimum statistical power (at least 20 obs)
                # and maximum relevance (not too old data)
                optimal = max(min_window, min(optimal, max_window))
                
                # Confidence based on half-life estimation quality
                if 5 <= half_life <= 60:
                    confidence = 'high'
                elif 3 <= half_life <= 90:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                return {
                    'optimal_window': optimal,
                    'method_used': 'half_life',
                    'confidence': confidence,
                    'estimated_half_life': half_life,
                    'half_lives_in_window': optimal / half_life,
                    'diagnostics': {
                        'theta': ou.theta,
                        'ar1_coef': ou.ar1_coef,
                        'eq_std': ou.eq_std
                    }
                }
            else:
                # Fall back to variance ratio if OU fit fails
                method = 'variance_ratio'
        
        if method == 'variance_ratio':
            # Method 2: Find window where variance ratio test is most significant
            # Variance ratio = Var(k-period returns) / (k * Var(1-period returns))
            # For mean-reverting series, VR < 1
            
            window_candidates = np.linspace(min_window, max_window, 20).astype(int)
            vr_scores = []
            
            returns = spread.diff().dropna()
            var_1 = returns.var()
            
            for w in window_candidates:
                if w >= len(spread):
                    continue
                k_returns = spread.diff(w).dropna()
                var_k = k_returns.var()
                vr = var_k / (w * var_1) if var_1 > 0 else 1
                # Score: how much VR deviates from 1 (lower = more mean-reverting)
                vr_scores.append((w, vr, abs(1 - vr)))
            
            if vr_scores:
                # Find window with strongest mean reversion signal
                best = max(vr_scores, key=lambda x: x[2])
                optimal = best[0]
                vr_value = best[1]
                
                confidence = 'high' if vr_value < 0.5 else ('medium' if vr_value < 0.8 else 'low')
                
                return {
                    'optimal_window': optimal,
                    'method_used': 'variance_ratio',
                    'confidence': confidence,
                    'variance_ratio': vr_value,
                    'diagnostics': {
                        'all_vr_scores': vr_scores
                    }
                }
        
        if method == 'cross_validation':
            # Method 3: Use walk-forward cross-validation
            # Select window that minimizes out-of-sample prediction error
            
            window_candidates = np.linspace(min_window, max_window, 10).astype(int)
            cv_scores = []
            
            test_size = max(20, n // 10)  # Use last 10% or 20 obs for testing
            
            for w in window_candidates:
                if w + test_size >= n:
                    continue
                    
                errors = []
                for t in range(n - test_size, n):
                    if t - w < 0:
                        continue
                    train_spread = spread.iloc[t-w:t]
                    ou = self.fit_ou_parameters(train_spread)
                    
                    if ou.valid:
                        # Predict next value
                        current = spread.iloc[t-1]
                        predicted = ou.expected_value(current, 1/252)
                        actual = spread.iloc[t]
                        errors.append((predicted - actual) ** 2)
                
                if errors:
                    mse = np.mean(errors)
                    cv_scores.append((w, mse))
            
            if cv_scores:
                best = min(cv_scores, key=lambda x: x[1])
                optimal = best[0]
                mse = best[1]
                
                return {
                    'optimal_window': optimal,
                    'method_used': 'cross_validation',
                    'confidence': 'high',
                    'mse': mse,
                    'diagnostics': {
                        'all_cv_scores': cv_scores
                    }
                }
        
        if method == 'aic':
            # Method 4: Use AIC to balance fit and complexity
            # Penalize both underfitting (small window) and overfitting (large window)
            
            window_candidates = np.linspace(min_window, max_window, 15).astype(int)
            aic_scores = []
            
            for w in window_candidates:
                chunk = spread.iloc[-w:]
                ou = self.fit_ou_parameters(chunk)
                
                if ou.valid:
                    # Simplified AIC: -2*log(L) + 2*k
                    # For OU, k=3 (theta, mu, sigma)
                    residuals = []
                    for i in range(1, len(chunk)):
                        pred = ou.expected_value(chunk.iloc[i-1], 1/252)
                        residuals.append(chunk.iloc[i] - pred)
                    
                    if residuals:
                        rss = sum(r**2 for r in residuals)
                        # AIC approximation
                        aic = w * np.log(rss / w) + 2 * 3
                        aic_scores.append((w, aic))
            
            if aic_scores:
                best = min(aic_scores, key=lambda x: x[1])
                optimal = best[0]
                
                return {
                    'optimal_window': optimal,
                    'method_used': 'aic',
                    'confidence': 'medium',
                    'aic': best[1],
                    'diagnostics': {
                        'all_aic_scores': aic_scores
                    }
                }
        
        # Default fallback
        return {
            'optimal_window': 60,
            'method_used': 'default',
            'confidence': 'low',
            'reason': 'no_method_succeeded',
            'diagnostics': {}
        }
    
    def get_rolling_statistics(self, pair: str, window: int = None, 
                               adaptive: bool = True) -> pd.DataFrame:
        """
        Calculate rolling statistics for parameter stability monitoring.
        
        Args:
            pair: Pair string (e.g., 'AAPL/MSFT')
            window: Rolling window size. If None and adaptive=True, optimal window is estimated
            adaptive: If True and window is None, estimate optimal window from data
            
        Returns:
            DataFrame with rolling_mean, rolling_std, rolling_half_life, and metadata
        """
        t1, t2 = pair.split('/')
        pair_data = self.price_data[[t1, t2]].dropna()
        
        if len(pair_data) < 30:
            return pd.DataFrame()
        
        log_spread = np.log(pair_data[t1] / pair_data[t2])
        
        # Determine window size
        if window is None and adaptive:
            window_info = self.estimate_optimal_window(log_spread, method='half_life')
            window = window_info['optimal_window']
            window_method = window_info['method_used']
            window_confidence = window_info['confidence']
        else:
            window = window or 60  # Default fallback
            window_method = 'user_specified' if window else 'default'
            window_confidence = 'n/a'
        
        if len(pair_data) < window:
            return pd.DataFrame()
        
        rolling_mean = log_spread.rolling(window).mean()
        rolling_std = log_spread.rolling(window).std()
        
        # Rolling half-life estimation
        half_lives = []
        for i in range(window, len(log_spread)):
            chunk = log_spread.iloc[i-window:i]
            ou = self.fit_ou_parameters(chunk)
            half_lives.append(ou.half_life_days if ou.valid else np.nan)
        
        half_life_series = pd.Series([np.nan] * window + half_lives, index=log_spread.index)
        
        result = pd.DataFrame({
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_half_life': half_life_series
        })
        
        # Store metadata as attributes
        result.attrs['window'] = window
        result.attrs['window_method'] = window_method
        result.attrs['window_confidence'] = window_confidence
        
        return result
    
    def summary_report(self) -> Dict:
        """Generate summary report of screening results."""
        total_pairs = len(self.pairs_stats) if len(self.pairs_stats) > 0 else 0
        viable = len(self.viable_pairs) if len(self.viable_pairs) > 0 else 0
        
        active_signals = self.get_active_signals()
        
        return {
            'total_pairs_tested': total_pairs,
            'viable_pairs': viable,
            'viability_rate': viable / total_pairs * 100 if total_pairs > 0 else 0,
            'active_signals': len(active_signals),
            'long_signals': sum(1 for s in active_signals if s.signal_type == 'LONG_SPREAD'),
            'short_signals': sum(1 for s in active_signals if s.signal_type == 'SHORT_SPREAD'),
            'high_confidence': sum(1 for s in active_signals if s.confidence == 'HIGH'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_tickers_from_csv(filepath: str, column: str = 'ticker', separator: str = None) -> List[str]:
    """Load ticker list from CSV file."""
    # Try different encodings
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            # Auto-detect separator
            if separator is None:
                with open(filepath, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                    if ';' in first_line:
                        sep = ';'
                    elif '\t' in first_line:
                        sep = '\t'
                    else:
                        sep = ','
            else:
                sep = separator
            
            df = pd.read_csv(filepath, sep=sep, encoding=encoding)
            
            # Normalize column names
            df.columns = [str(c).strip().lower() for c in df.columns]
            
            if 'ticker' in df.columns:
                tickers = df['ticker'].dropna().astype(str).str.strip().tolist()
            elif 'symbol' in df.columns:
                tickers = df['symbol'].dropna().astype(str).str.strip().tolist()
            else:
                # Try first column
                tickers = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            
            # Filter out empty strings and headers that might have slipped through
            tickers = [t for t in tickers if t and t.lower() not in ['ticker', 'symbol', '']]
            
            return tickers
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # If it's not an encoding error, raise it
            if 'codec' not in str(e).lower():
                raise
            continue
    
    raise ValueError(f"Could not read CSV file with any supported encoding: {filepath}")
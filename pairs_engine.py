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
                       n_sims: int = 20000, max_time_years: float = 0.5) -> Dict:
        """
        P(hit take_profit before stop_loss | S_0) via Monte Carlo.
        """
        dt = 1/252
        n_steps = int(max_time_years / dt)
        
        np.random.seed(42)
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
            
            dW = np.random.randn(n_sims) * np.sqrt(dt)
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
        self.viable_pairs = []
        self.ou_models = {}
        self.signals = {}
        self.positions = {}
    
    def _set_defaults(self):
        """Set default configuration values."""
        defaults = {
            'lookback_period': '2y',
            'min_half_life': 5,
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
        batch_size = 200
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
                    threads=True,
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
        
        # Forward fill gaps first
        data = data.ffill()
        
        # Drop tickers with more than 30% missing data
        missing_pct = data.isna().sum() / len(data)
        valid_tickers = missing_pct[missing_pct < 0.30].index.tolist()
        data = data[valid_tickers]
        
        # Store raw data for pair-specific calculations
        self.raw_price_data = data.copy()
        
        # Keep NaN - correlation uses min_periods, pair testing uses pairwise dropna
        self.price_data = data
        
        if progress_callback:
            progress_callback(total_batches, total_batches, f"Loaded {len(data.columns)}/{len(tickers)} tickers ({len(data)} days)")
        
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
            
            rms_list = []
            for i in range(n_windows):
                # Extract segment
                start_idx = i * w
                end_idx = start_idx + w
                segment = x[start_idx:end_idx]
                
                # Fit polynomial trend to segment
                t = np.arange(w)
                try:
                    coeffs = np.polyfit(t, segment, order)
                    trend = np.polyval(coeffs, t)
                    
                    # Calculate RMS of detrended segment
                    detrended = segment - trend
                    rms = np.sqrt(np.mean(detrended ** 2))
                    
                    if rms > 0 and np.isfinite(rms):
                        rms_list.append(rms)
                except:
                    continue
            
            if rms_list:
                fluctuations.append((w, np.mean(rms_list)))
        
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
        except:
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
                if len(spread) > 50:
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
            
            # === Final viability: Option B ===
            # EG passes AND Johansen passes AND half-life valid AND Hurst valid
            is_viable = passes_coint and passes_halflife and ou_valid and passes_hurst
            
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
                'data_length': 0,
                'is_viable': False,
                'error': str(e)[:100]
            }
    
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
        
        if progress_callback:
            progress_callback('grouping', 0, 0, f"Starting with {len(available_tickers)} tickers")

        groups = {'ALL': available_tickers}
        
        # Filter to groups with enough tickers
        groups = {k: [t for t in v if t in available_tickers] 
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
            
            # Scale min_periods with data length for reliable correlations
            # Use at least 20% of available data, minimum 50, capped at 252 (1 year)
            data_length = len(returns)
            min_obs_for_corr = min(252, max(50, int(data_length * 0.2)))
            
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
                
                # Find pairs above correlation threshold
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        ticker_a = corr_matrix.columns[i]
                        ticker_b = corr_matrix.columns[j]
                        # Always order alphabetically for consistency
                        t1, t2 = (ticker_a, ticker_b) if ticker_a < ticker_b else (ticker_b, ticker_a)
                        corr = corr_matrix.iloc[i, j]
                        
                        if not np.isnan(corr) and corr >= self.config['min_correlation']:
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
                    executor.submit(self._test_single_pair, pair_info, data, raw_data): pair_info 
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
                
                result = self._test_single_pair(pair_info, data, raw_data)
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            results_df = results_df.sort_values('half_life_days')
        
        self.pairs_stats = results_df
        self.viable_pairs = results_df[results_df['is_viable']].copy() if len(results_df) > 0 else pd.DataFrame()
        
        # Build OU models for viable pairs
        for _, row in self.viable_pairs.iterrows():
            pair = row['pair']
            self.ou_models[pair] = OUProcess(
                theta=row['theta'],
                mu=row['mu'],
                sigma=row['sigma']
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
        
        if ou_model is None:
            # Fit on the fly
            log_spread = np.log(pair_data[t1] / pair_data[t2])
            ou_params = self.fit_ou_parameters(log_spread)
            if not ou_params.valid:
                return TradeSignal(
                    pair=pair, signal_type='NO_TRADE', current_zscore=0,
                    entry_spread=0, take_profit_spread=0, stop_loss_spread=0,
                    win_probability=0, expected_pnl=0, kelly_fraction=0,
                    risk_reward=0, avg_holding_days=0, confidence='LOW'
                )
            ou_model = OUProcess(ou_params.theta, ou_params.mu, ou_params.sigma)
        
        log_spread = np.log(pair_data[t1] / pair_data[t2])
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
    
    def get_pair_ou_params(self, pair: str, use_raw_data: bool = True) -> Tuple[OUProcess, pd.Series, float]:
        """
        Get OU parameters for a specific pair using Engle-Granger spread.
        
        IMPORTANT: This uses the EG spread (Y - β*X - α) with the hedge ratio
        from screening, NOT the simple log spread. This ensures consistency
        between scanner results and analytics display.
        
        Args:
            pair: Pair string like "TICKER1/TICKER2"
            use_raw_data: If True, use raw_price_data (before global alignment)
        
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
        
        if len(pair_data) < 50:
            raise ValueError(f"Insufficient data for pair {pair}: only {len(pair_data)} points")
        
        y = pair_data[t1]
        x = pair_data[t2]
        
        # Get hedge ratio and intercept from stored screening results
        hedge_ratio = 1.0
        intercept = 0.0
        
        if self.pairs_stats is not None and len(self.pairs_stats) > 0:
            pair_row = self.pairs_stats[self.pairs_stats['pair'] == pair]
            if len(pair_row) > 0:
                hedge_ratio = float(pair_row['hedge_ratio'].iloc[0])
                intercept = float(pair_row['intercept'].iloc[0])
        
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
        
        # Fit OU parameters on the EG spread
        ou_params = self.fit_ou_parameters(eg_spread)
        
        if not ou_params.valid:
            raise ValueError(f"Could not fit valid OU model for {pair}")
        
        # Create OUProcess
        ou = OUProcess(ou_params.theta, ou_params.mu, ou_params.sigma)
        
        # Calculate current Z-score
        current_spread = eg_spread.iloc[-1]
        current_z = ou.zscore(current_spread)
        
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
        
        # Get hedge ratio and intercept from stored results
        hedge_ratio = 1.0
        intercept = 0.0
        
        if self.pairs_stats is not None and len(self.pairs_stats) > 0:
            pair_row = self.pairs_stats[self.pairs_stats['pair'] == pair]
            if len(pair_row) > 0:
                hedge_ratio = float(pair_row['hedge_ratio'].iloc[0])
                intercept = float(pair_row['intercept'].iloc[0])
        
        # Calculate EG spread
        eg_spread = y - hedge_ratio * x - intercept
        
        # Get or create OU model using pair-specific data
        try:
            ou_model, _, _ = self.get_pair_ou_params(pair, use_raw_data=use_raw_data)
        except:
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
    
    def get_rolling_statistics(self, pair: str, window: int = 60) -> pd.DataFrame:
        """Calculate rolling statistics for parameter stability monitoring."""
        t1, t2 = pair.split('/')
        pair_data = self.price_data[[t1, t2]].dropna()
        if len(pair_data) < window:
            return pd.DataFrame()
        log_spread = np.log(pair_data[t1] / pair_data[t2])
        
        rolling_mean = log_spread.rolling(window).mean()
        rolling_std = log_spread.rolling(window).std()
        
        # Rolling half-life estimation
        half_lives = []
        for i in range(window, len(log_spread)):
            chunk = log_spread.iloc[i-window:i]
            ou = self.fit_ou_parameters(chunk)
            half_lives.append(ou.half_life_days if ou.valid else np.nan)
        
        half_life_series = pd.Series([np.nan] * window + half_lives, index=log_spread.index)
        
        return pd.DataFrame({
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_half_life': half_life_series
        })
    
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
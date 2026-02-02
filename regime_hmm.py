"""
================================================================================
INSTITUTIONAL MARKET REGIME DETECTION SYSTEM
================================================================================
Hedge Fund Grade Hidden Markov Model Implementation

Features:
    1. Sticky HMM with Dirichlet Prior - Penalizes rapid regime switching
    2. Student-t Emissions - Robust to fat tails and outliers
    3. Robust Covariance - Ledoit-Wolf shrinkage
    4. Particle Filter - Online updates
    5. Jump Detection - Fast regime shift identification

4-State Regime Model:
    0: RISK-ON      - Expansion: Strong growth, moderate inflation
    1: RISK-OFF     - Contraction: Falling growth, risk aversion
    2: DEFLATION    - Crisis: Falling growth AND inflation
    3: STAGFLATION  - Stress: Weak growth, high inflation

Author: Klippinge Investment Trading Terminal
Version: 4.0 - Hedge Fund Grade
================================================================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import warnings
import os
import pickle
import time
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.special import logsumexp, gammaln
from scipy.stats import nbinom
from scipy.optimize import linear_sum_assignment

try:
    from sklearn.covariance import LedoitWolf
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# =============================================================================
# REGIME DEFINITIONS (3-STATE MODEL)
# =============================================================================

N_REGIMES = 3  # Global constant for number of regimes

REGIMES = {
    0: {
        'name': 'RISK-ON',
        'color': '#22c55e',  # Green
        'description': 'Expansion phase. Equities lead, credit spreads tight, volatility suppressed. '
                      'Favor cyclicals, small caps, high beta. Carry strategies perform well.',
        'assets': 'Long equities, risk assets',
        'expected_duration_months': 14.0,
        'expected_signs': {
            'equity_mom': 1, 'bond_mom': -1, 'vix': -1,
            'yield_curve': 1, 'credit_spread': 1,
        }
    },
    1: {
        'name': 'NEUTRAL',
        'color': '#f59e0b',  # Amber/Orange
        'description': 'Transition phase. Mixed signals, moderate volatility. '
                      'Balanced allocation, focus on quality and dividend yield.',
        'assets': 'Balanced, quality stocks, dividends',
        'expected_duration_months': 8.0,
        'expected_signs': {
            'equity_mom': 0, 'bond_mom': 0, 'vix': 0,
            'yield_curve': 0, 'credit_spread': 0,
        }
    },
    2: {
        'name': 'RISK-OFF',
        'color': '#ef4444',  # Red
        'description': 'Risk aversion / crisis phase. Defensive assets outperform, credit widens, '
                      'volatility spikes. Flight to quality - bonds, gold, cash, low beta.',
        'assets': 'Defensive, bonds, gold, cash',
        'expected_duration_months': 5.0,
        'expected_signs': {
            'equity_mom': -1, 'bond_mom': 1, 'vix': 1,
            'yield_curve': -1, 'credit_spread': -1,
        }
    }
}

# Legacy 4-regime definitions (kept for backward compatibility)
REGIMES_4STATE = {
    0: {'name': 'RISK-ON', 'color': '#22c55e'},
    1: {'name': 'RISK-OFF', 'color': '#f59e0b'},
    2: {'name': 'DEFLATION', 'color': '#3b82f6'},
    3: {'name': 'STAGFLATION', 'color': '#ef4444'},
}


# =============================================================================
# DURATION DISTRIBUTION
# =============================================================================

@dataclass
class DurationDistribution:
    distribution: str
    params: Dict[str, float]
    
    def pmf(self, d: int) -> float:
        if self.distribution == 'negative_binomial':
            r, p = self.params['r'], self.params['p']
            return nbinom.pmf(d - 1, r, p) if d >= 1 else 0.0
        return 0.0
    
    def survival(self, d: int) -> float:
        if self.distribution == 'negative_binomial':
            r, p = self.params['r'], self.params['p']
            return 1.0 - nbinom.cdf(d - 2, r, p) if d >= 2 else 1.0
        return 1.0
    
    def hazard(self, d: int) -> float:
        surv = self.survival(d)
        return self.pmf(d) / surv if surv > 1e-10 else 1.0
    
    def expected_duration(self) -> float:
        if self.distribution == 'negative_binomial':
            r, p = self.params['r'], self.params['p']
            return r * (1 - p) / p + 1
        return 1.0


# =============================================================================
# ROBUST COVARIANCE
# =============================================================================

class RobustCovarianceEstimator:
    def __init__(self):
        self.covariance_ = None
        
    def fit(self, X: np.ndarray) -> 'RobustCovarianceEstimator':
        n_samples, n_features = X.shape
        
        if SKLEARN_AVAILABLE and n_samples > n_features + 5:
            try:
                estimator = LedoitWolf().fit(X)
                self.covariance_ = estimator.covariance_
                return self
            except:
                pass
        
        self.covariance_ = np.cov(X.T) + np.eye(n_features) * 1e-4
        return self


# =============================================================================
# STUDENT-T EMISSION
# =============================================================================

class StudentTEmission:
    def __init__(self, n_features: int, nu: float = 5.0):
        self.n_features = n_features
        self.nu = nu
        self.mean_ = None
        self.scale_ = None
        self.precision_ = None
        
    def fit(self, X: np.ndarray, weights: np.ndarray = None) -> 'StudentTEmission':
        n_samples = len(X)
        
        if weights is None:
            weights = np.ones(n_samples)
        weights = weights / weights.sum()
        
        self.mean_ = np.average(X, axis=0, weights=weights)
        
        cov_est = RobustCovarianceEstimator().fit(X)
        self.scale_ = cov_est.covariance_
        
        for _ in range(10):
            u = self._compute_weights(X)
            w = weights * u
            w_sum = w.sum()
            
            self.mean_ = (w[:, None] * X).sum(axis=0) / w_sum
            X_centered = X - self.mean_
            self.scale_ = (w[:, None, None] * np.einsum('ni,nj->nij', X_centered, X_centered)).sum(axis=0)
            self.scale_ /= w_sum
            self.scale_ += np.eye(self.n_features) * 1e-6
        
        try:
            self.precision_ = np.linalg.inv(self.scale_)
        except:
            self.precision_ = np.eye(self.n_features)
        
        return self
    
    def _compute_weights(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        try:
            precision = np.linalg.inv(self.scale_)
            mahal = np.sum(X_centered @ precision * X_centered, axis=1)
        except:
            mahal = np.sum(X_centered ** 2, axis=1)
        return (self.nu + self.n_features) / (self.nu + mahal)
    
    def log_prob(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean_
        
        try:
            mahal = np.sum(X_centered @ self.precision_ * X_centered, axis=1)
        except:
            mahal = np.sum(X_centered ** 2, axis=1)
        
        sign, log_det = np.linalg.slogdet(self.scale_)
        if sign <= 0:
            log_det = np.sum(np.log(np.abs(np.diag(self.scale_)) + 1e-10))
        
        log_prob = (
            gammaln((self.nu + self.n_features) / 2) -
            gammaln(self.nu / 2) -
            0.5 * self.n_features * np.log(self.nu * np.pi) -
            0.5 * log_det -
            0.5 * (self.nu + self.n_features) * np.log(1 + mahal / self.nu)
        )
        return log_prob


# =============================================================================
# STICKY HMM
# =============================================================================

class StickyHMM:
    def __init__(self, 
                 n_states: int = N_REGIMES,  # Default to 3-state model
                 kappa: float = 15.0,  # Reduced for more realistic transitions
                 alpha: float = 1.0,
                 emission_type: str = 'student_t',
                 nu: float = 5.0):
        
        self.n_states = n_states
        self.kappa = kappa
        self.alpha = alpha
        self.emission_type = emission_type
        self.nu = nu
        
        self.initial_dist_ = None
        self.transition_matrix_ = None
        self.emissions_: List[StudentTEmission] = []
        self.alpha_prior_ = None
        
        self.states_ = None
        self.state_probs_ = None
        self.log_likelihood_ = None
        self.fitted = False
        
    def _initialize_priors(self):
        self.alpha_prior_ = np.full((self.n_states, self.n_states), self.alpha)
        np.fill_diagonal(self.alpha_prior_, self.alpha + self.kappa)
        self.initial_prior_ = np.ones(self.n_states)
        
    def fit(self, X: np.ndarray, progress_callback: Callable = None) -> 'StickyHMM':
        T, n_features = X.shape
        
        if T < 10:
            raise ValueError(f"Not enough data: {T} samples. Need at least 10.")
        
        if progress_callback:
            progress_callback(30, "Initializing Sticky HMM...")
        
        self._initialize_priors()
        self._initialize_emissions(X)
        self.transition_matrix_ = self._sample_transition_from_prior()
        self.initial_dist_ = np.ones(self.n_states) / self.n_states
        
        if progress_callback:
            progress_callback(40, "Running Variational EM...")
        
        prev_ll = -np.inf
        for iteration in range(50):
            log_emission = self._compute_log_emission(X)
            alpha, beta, log_likelihood = self._forward_backward(log_emission)
            
            log_gamma = alpha + beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            
            xi = self._compute_xi(alpha, beta, log_emission)
            self._m_step(X, gamma, xi)
            
            if abs(log_likelihood - prev_ll) < 1e-4:
                break
            prev_ll = log_likelihood
            
            if progress_callback and iteration % 10 == 0:
                progress_callback(40 + int(30 * iteration / 50), 
                                f"EM iteration {iteration}")
        
        self.log_likelihood_ = log_likelihood
        
        if progress_callback:
            progress_callback(75, "Viterbi decoding...")
        
        self.states_ = self._viterbi_decode(log_emission)
        
        # =====================================================================
        # PROBABILITY SMOOTHING - Prevents overconfident probabilities
        # This is critical for realistic uncertainty quantification
        # =====================================================================
        
        # Apply additional smoothing to gamma
        gamma_smooth = gamma.copy()
        
        # Ensure minimum probability floor (2% for each state)
        prob_floor = 0.02
        gamma_smooth = np.maximum(gamma_smooth, prob_floor)
        gamma_smooth /= gamma_smooth.sum(axis=1, keepdims=True)
        
        # Cap maximum probability at 85%
        prob_cap = 0.85
        gamma_smooth = np.minimum(gamma_smooth, prob_cap)
        gamma_smooth /= gamma_smooth.sum(axis=1, keepdims=True)
        
        # Temporal smoothing (EMA with recent observations)
        alpha_smooth = 0.7  # Weight on current observation
        for t in range(1, len(gamma_smooth)):
            gamma_smooth[t] = alpha_smooth * gamma_smooth[t] + (1 - alpha_smooth) * gamma_smooth[t-1]
        
        # Final normalization
        gamma_smooth /= gamma_smooth.sum(axis=1, keepdims=True)
        
        self.state_probs_ = gamma_smooth
        
        self._reorder_states_by_characteristics(X)
        self.fitted = True
        
        if progress_callback:
            progress_callback(85, "HMM fitting complete")
        
        return self
    
    def _initialize_emissions(self, X: np.ndarray):
        n_features = X.shape[1]
        
        if SKLEARN_AVAILABLE and len(X) >= self.n_states:
            try:
                kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
            except:
                labels = np.random.randint(0, self.n_states, size=len(X))
        else:
            labels = np.random.randint(0, self.n_states, size=len(X))
        
        self.emissions_ = []
        for s in range(self.n_states):
            mask = labels == s
            if self.emission_type == 'student_t':
                emission = StudentTEmission(n_features, nu=self.nu)
            else:
                emission = StudentTEmission(n_features, nu=100.0)
            
            if np.sum(mask) > n_features:
                emission.fit(X[mask])
            else:
                emission.fit(X)
            
            self.emissions_.append(emission)
    
    def _sample_transition_from_prior(self) -> np.ndarray:
        transition = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            transition[i] = np.random.dirichlet(self.alpha_prior_[i])
        return transition
    
    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        T = len(X)
        log_emission = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            log_emission[:, s] = self.emissions_[s].log_prob(X)
        return log_emission
    
    def _forward_backward(self, log_emission: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        T = log_emission.shape[0]
        
        # Temperature scaling on emissions to prevent overconfidence
        # Higher temperature = more uniform (less extreme) probabilities
        temperature = 2.0
        log_emission_scaled = log_emission / temperature
        
        log_trans = np.log(self.transition_matrix_ + 1e-10)
        log_init = np.log(self.initial_dist_ + 1e-10)
        
        alpha = np.zeros((T, self.n_states))
        alpha[0] = log_init + log_emission_scaled[0]
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = logsumexp(alpha[t-1] + log_trans[:, j]) + log_emission_scaled[t, j]
        
        beta = np.zeros((T, self.n_states))
        beta[-1] = 0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = logsumexp(log_trans[i] + log_emission_scaled[t+1] + beta[t+1])
        
        return alpha, beta, logsumexp(alpha[-1])
    
    def _compute_xi(self, alpha: np.ndarray, beta: np.ndarray, 
                    log_emission: np.ndarray) -> np.ndarray:
        T = len(alpha)
        log_trans = np.log(self.transition_matrix_ + 1e-10)
        xi = np.zeros((self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi_t = (alpha[t, i] + log_trans[i, j] + 
                               log_emission[t+1, j] + beta[t+1, j])
                    xi[i, j] += np.exp(log_xi_t - logsumexp(alpha[-1]))
        return xi
    
    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        self.initial_dist_ = gamma[0] + self.initial_prior_ - 1
        self.initial_dist_ = np.maximum(self.initial_dist_, 1e-10)
        self.initial_dist_ /= self.initial_dist_.sum()
        
        for i in range(self.n_states):
            posterior = xi[i] + self.alpha_prior_[i] - 1
            posterior = np.maximum(posterior, 1e-10)
            self.transition_matrix_[i] = posterior / posterior.sum()
        
        for s in range(self.n_states):
            weights = gamma[:, s]
            if weights.sum() > 1.0:
                self.emissions_[s].fit(X, weights)
    
    def _viterbi_decode(self, log_emission: np.ndarray) -> np.ndarray:
        T = log_emission.shape[0]
        log_trans = np.log(self.transition_matrix_ + 1e-10)
        log_init = np.log(self.initial_dist_ + 1e-10)
        
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = log_init + log_emission[0]
        
        for t in range(1, T):
            for j in range(self.n_states):
                trans = delta[t-1] + log_trans[:, j]
                psi[t, j] = np.argmax(trans)
                delta[t, j] = trans[psi[t, j]] + log_emission[t, j]
        
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def _reorder_states_by_characteristics(self, X: np.ndarray):
        """
        Reorder states to match expected regime characteristics.
        
        Uses optimal matching (Hungarian algorithm) for proper state labeling.
        
        3-State Model:
        - RISK-ON:  High equity (+), low VIX (-), weak bonds (-)
        - NEUTRAL:  Mixed/moderate signals
        - RISK-OFF: Low equity (-), high VIX (+), strong bonds (+)
        """
        n_states = len(self.emissions_)
        
        if n_states not in [3, 4]:
            print(f"Warning: Reordering not implemented for {n_states} states")
            return
        
        n_features = len(self.emissions_[0].mean_)
        
        # Expected z-score characteristics for each regime
        if n_states == 3:
            expected = {
                0: {'equity': +0.8, 'vix': -0.8, 'bonds': -0.3, 'name': 'RISK-ON'},
                1: {'equity':  0.0, 'vix':  0.0, 'bonds':  0.0, 'name': 'NEUTRAL'},
                2: {'equity': -0.5, 'vix': +0.8, 'bonds': +0.5, 'name': 'RISK-OFF'},
            }
        else:  # 4 states (legacy)
            expected = {
                0: {'equity': +1.0, 'vix': -1.0, 'bonds': -0.5, 'name': 'RISK-ON'},
                1: {'equity': -0.5, 'vix': +1.0, 'bonds': +0.5, 'name': 'RISK-OFF'},
                2: {'equity': -0.3, 'vix': +0.3, 'bonds': +1.0, 'name': 'DEFLATION'},
                3: {'equity': -0.5, 'vix': +0.5, 'bonds': -1.0, 'name': 'STAGFLATION'},
            }
        
        # Extract characteristics for each HMM state
        state_chars = []
        for e in self.emissions_:
            state_chars.append({
                'equity': e.mean_[0],
                'vix': e.mean_[3] if n_features > 3 else 0,
                'bonds': e.mean_[1] if n_features > 1 else 0,
            })
        
        # Build cost matrix for Hungarian algorithm
        cost_matrix = np.zeros((n_states, n_states))
        for hmm_state in range(n_states):
            for regime in range(n_states):
                sc = state_chars[hmm_state]
                exp = expected[regime]
                # Weighted squared differences
                cost = (2.0 * (sc['equity'] - exp['equity'])**2 +
                       1.5 * (sc['vix'] - exp['vix'])**2 +
                       1.0 * (sc['bonds'] - exp['bonds'])**2)
                cost_matrix[hmm_state, regime] = cost
        
        # Use Hungarian algorithm for optimal assignment
        hmm_indices, regime_indices = linear_sum_assignment(cost_matrix)
        
        # Build new_order: new_order[regime] = which HMM state becomes this regime
        new_order = np.zeros(n_states, dtype=int)
        for hmm_idx, regime_idx in zip(hmm_indices, regime_indices):
            new_order[regime_idx] = hmm_idx
        
        for regime_idx in range(n_states):
            hmm_idx = new_order[regime_idx]
            cost = cost_matrix[hmm_idx, regime_idx]
        
        # Apply reordering
        self.emissions_ = [self.emissions_[i] for i in new_order]
        
        new_trans = np.zeros_like(self.transition_matrix_)
        for i in range(n_states):
            for j in range(n_states):
                new_trans[i, j] = self.transition_matrix_[new_order[i], new_order[j]]
        self.transition_matrix_ = new_trans
        
        for i in range(n_states):
            self.transition_matrix_[i] /= self.transition_matrix_[i].sum()
        
        inverse_order = np.argsort(new_order)
        self.states_ = np.array([inverse_order[s] for s in self.states_])
        
        if self.state_probs_ is not None:
            self.state_probs_ = self.state_probs_[:, new_order]
        
        # Validate final assignment with warnings
        for i in range(n_states):
            e = self.emissions_[i]
            exp = expected[i]
            sc = state_chars[new_order[i]]
            
            # Check if assignment makes sense
            if n_states == 3 and i == 1:  # NEUTRAL - more lenient
                status = "✓"
            else:
                equity_ok = (sc['equity'] > 0) == (exp['equity'] > 0) or abs(sc['equity']) < 0.15
                vix_ok = (sc['vix'] > 0) == (exp['vix'] > 0) or abs(sc['vix']) < 0.25
                status = "✓" if (equity_ok and vix_ok) else "⚠️ MISMATCH"
            
    
    def get_transition_matrix(self) -> np.ndarray:
        return self.transition_matrix_.copy()
    
    @property
    def means(self) -> np.ndarray:
        return np.array([e.mean_ for e in self.emissions_])
    
    @property
    def covars(self) -> np.ndarray:
        return np.array([e.scale_ for e in self.emissions_])


# =============================================================================
# PARTICLE FILTER
# =============================================================================

class ParticleFilterHMM:
    def __init__(self, n_particles: int = 2000, n_states: int = N_REGIMES):
        self.n_particles = n_particles
        self.n_states = n_states
        self.particles = None
        self.weights = None
        self.means = None
        self.covars = None
        self.transition_matrix = None
        self.state_history: List[int] = []
        self.prob_history: List[np.ndarray] = []
        self.initialized = False
    
    def initialize(self, hmm_model, X: np.ndarray = None):
        if hasattr(hmm_model, 'means'):
            self.means = hmm_model.means.copy()
            self.covars = hmm_model.covars.copy()
        elif hasattr(hmm_model, 'emissions_'):
            self.means = np.array([e.mean_ for e in hmm_model.emissions_])
            self.covars = np.array([e.scale_ for e in hmm_model.emissions_])
        
        if hasattr(hmm_model, 'get_transition_matrix'):
            self.transition_matrix = hmm_model.get_transition_matrix()
        elif hasattr(hmm_model, 'transition_matrix_'):
            self.transition_matrix = hmm_model.transition_matrix_.copy()
        elif hasattr(hmm_model, 'transition_matrix'):
            self.transition_matrix = hmm_model.transition_matrix.copy()
        
        for i in range(self.n_states):
            self.transition_matrix[i] /= self.transition_matrix[i].sum()
        
        self.particles = np.random.randint(0, self.n_states, size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.initialized = True
        
        if X is not None and len(X) > 0:
            for obs in X[-50:]:
                self.update(obs)
    
    def update(self, observation: np.ndarray) -> Dict[str, Any]:
        if not self.initialized:
            raise ValueError("Not initialized")
        
        new_particles = np.zeros(self.n_particles, dtype=int)
        for i in range(self.n_particles):
            new_particles[i] = np.random.choice(
                self.n_states, p=self.transition_matrix[self.particles[i]]
            )
        self.particles = new_particles
        
        log_weights = np.array([
            self._log_emission_prob(observation, self.particles[i])
            for i in range(self.n_particles)
        ])
        log_weights -= logsumexp(log_weights)
        self.weights = np.exp(log_weights)
        
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n_particles / 2:
            self._resample()
        
        state_probs = np.zeros(self.n_states)
        for s in range(self.n_states):
            state_probs[s] = np.sum(self.weights[self.particles == s])
        
        current_state = int(np.argmax(state_probs))
        self.state_history.append(current_state)
        self.prob_history.append(state_probs.copy())
        
        return {'state': current_state, 'probabilities': state_probs, 'ess': ess}
    
    def _log_emission_prob(self, obs: np.ndarray, state: int) -> float:
        try:
            diff = obs - self.means[state]
            cov_inv = np.linalg.inv(self.covars[state])
            log_det = np.log(np.linalg.det(self.covars[state]) + 1e-10)
            mahal = diff @ cov_inv @ diff
            n = len(obs)
            return -0.5 * (n * np.log(2 * np.pi) + log_det + mahal)
        except:
            return -100.0
    
    def _resample(self):
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles


# =============================================================================
# TIME-VARYING TRANSITIONS
# =============================================================================

class TimeVaryingTransitionHMM:
    def __init__(self, n_states: int = N_REGIMES, n_context_features: int = 3):
        self.n_states = n_states
        self.n_context_features = n_context_features
        self.W = np.random.randn(n_states, n_states, n_context_features) * 0.1
        self.b = np.zeros((n_states, n_states))
        self._set_economic_priors()
        self.fitted = False
    
    def _set_economic_priors(self):
        """Set economic priors for time-varying transitions."""
        # For 3-state model: RISK-ON (0), NEUTRAL (1), RISK-OFF (2)
        # Higher VIX increases transition to RISK-OFF
        if self.n_states >= 3:
            self.W[:, min(2, self.n_states-1), 0] = 0.5  # VIX -> RISK-OFF
            self.W[:, 0, 0] = -0.3  # VIX -> not RISK-ON
        np.fill_diagonal(self.b, 2.0)  # Sticky diagonal
    
    def get_transition_matrix(self, context: np.ndarray) -> np.ndarray:
        logits = self.b + np.tensordot(self.W, context, axes=([2], [0]))
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    def fit(self, states: np.ndarray, contexts: np.ndarray, 
            n_iter: int = 100, learning_rate: float = 0.01):
        T = len(states)
        for _ in range(n_iter):
            for t in range(1, T):
                s_prev, s_curr = states[t-1], states[t]
                ctx = contexts[t-1]
                trans_mat = self.get_transition_matrix(ctx)
                for j in range(self.n_states):
                    grad = (1.0 if j == s_curr else 0.0) - trans_mat[s_prev, j]
                    self.b[s_prev, j] += learning_rate * grad
                    self.W[s_prev, j] += learning_rate * grad * ctx
        self.fitted = True
    
    def predict_regime_change_prob(self, current_state: int, context: np.ndarray,
                                    horizon: int = 12) -> Dict[str, Any]:
        trans_mat = self.get_transition_matrix(context)
        change_probs, cumulative_stay = [], 1.0
        
        for _ in range(horizon):
            stay_prob = trans_mat[current_state, current_state]
            change_probs.append(cumulative_stay * (1 - stay_prob))
            cumulative_stay *= stay_prob
        
        trans_row = trans_mat[current_state].copy()
        trans_row[current_state] = 0
        if trans_row.sum() > 0:
            trans_row /= trans_row.sum()
            next_state = int(np.argmax(trans_row))
        else:
            next_state = current_state
        
        return {
            'change_probability': change_probs,
            'most_likely_next': REGIMES[next_state]['name'],
            'next_state_id': next_state,
            'next_state_color': REGIMES[next_state]['color']
        }


# =============================================================================
# UNCERTAINTY QUANTIFIER
# =============================================================================

class UncertaintyQuantifier:
    def __init__(self, models: List = None):
        self.models = models or []
    
    def add_model(self, model):
        self.models.append(model)
    
    def compute_regime_stability(self, states: np.ndarray) -> Dict[str, Any]:
        transitions = np.sum(np.diff(states) != 0)
        
        # Get unique states dynamically
        unique_states = np.unique(states)
        durations = {s: [] for s in unique_states}
        current_state, current_duration = states[0], 1
        
        for s in states[1:]:
            if s == current_state:
                current_duration += 1
            else:
                durations[current_state].append(current_duration)
                current_state, current_duration = s, 1
        durations[current_state].append(current_duration)
        
        all_durations = [d for ds in durations.values() for d in ds]
        
        unique, counts = np.unique(states, return_counts=True)
        distribution = {}
        for s, c in zip(unique, counts):
            s_int = int(s)
            if s_int in REGIMES:
                distribution[REGIMES[s_int]['name']] = c / len(states)
            else:
                distribution[f'State_{s_int}'] = c / len(states)
        
        return {
            'total_transitions': int(transitions),
            'avg_regime_duration': float(np.mean(all_durations)) if all_durations else 0,
            'median_regime_duration': float(np.median(all_durations)) if all_durations else 0,
            'max_regime_duration': int(np.max(all_durations)) if all_durations else 0,
            'regime_distribution': distribution,
            'stability_score': 1.0 - (transitions / max(len(states) - 1, 1))
        }


# =============================================================================
# FEATURE ENGINEERING - ROBUST VERSION
# =============================================================================

class RegimeFeatures:
    """Robust feature engineering with individual ticker downloads."""
    
    TICKERS = {
        'SPY': 'S&P 500 ETF', 
        'TLT': '20+ Year Treasury ETF',
        'GLD': 'Gold ETF',
        '^VIX': 'VIX Index', 
        '^TNX': '10-Year Treasury Yield',
        '^IRX': '3-Month Treasury Yield', 
        'IWM': 'Russell 2000 ETF',
        'EEM': 'Emerging Markets ETF', 
        'HYG': 'High Yield Bond ETF', 
        'LQD': 'Investment Grade Bond ETF',
        'VUG': 'Growth ETF',
        'VTV': 'Value ETF',
    }
    
    FEATURE_NAMES = [
        'equity_mom', 'bond_mom', 'gold_mom', 'vix',
        'yield_curve', 'credit_spread', 'small_large', 
        'em_dm', 'growth_value'
    ]
    
    def __init__(self, lookback_years: int = 30, zscore_window: int = 24):
        self.lookback_years = lookback_years
        self.zscore_window = zscore_window
        self.raw_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
    
    def fetch_data(self, progress_callback: Callable = None) -> pd.DataFrame:
        """
        Fetch data with robust error handling.
        
        IMPORTANT: Uses DAILY data and resamples to monthly because yfinance's
        interval="1mo" has a known bug where data can be months behind.
        """
        if progress_callback:
            progress_callback(5, "Fetching market data...")
        
        all_data = {}
        tickers = list(self.TICKERS.keys())
        
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365 + 30)
        
        # Try batch download with DAILY data (more reliable than monthly)
        try:
            if progress_callback:
                progress_callback(6, "Downloading daily data...")
            
            data = yf.download(
                tickers, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval="1d",  # Daily data - more reliable!
                progress=False,
                threads=True,
                timeout=60,
                ignore_tz=True
            )
            
            if data is not None and len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data['Close']
                else:
                    closes = data
                
                # Resample daily to monthly (last trading day of month)
                # 'ME' for pandas >= 2.0, 'M' for older versions
                try:
                    monthly_closes = closes.resample('ME').last()
                except ValueError:
                    monthly_closes = closes.resample('M').last()
                
                # Check what we got
                for ticker in tickers:
                    if ticker in monthly_closes.columns:
                        series = monthly_closes[ticker].dropna()
                        if len(series) > 50:
                            all_data[ticker] = series
                
                if len(all_data) >= 4:
                    if progress_callback:
                        progress_callback(8, f"Batch download: {len(all_data)} tickers")
                        
        except Exception as e:
            print(f"Batch download failed: {e}")
        
        # If batch failed or incomplete, try individual downloads
        missing = [t for t in tickers if t not in all_data]
        
        if missing:
            if progress_callback:
                progress_callback(7, f"Downloading {len(missing)} missing tickers...")
            
            for i, ticker in enumerate(missing):
                try:
                    df = yf.download(
                        ticker, 
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval="1d",  # Daily data
                        progress=False,
                        timeout=30,
                        ignore_tz=True
                    )
                    if df is not None and len(df) > 50:
                        if 'Close' in df.columns:
                            # Resample to monthly (pandas version compatible)
                            try:
                                monthly = df['Close'].resample('ME').last().dropna()
                            except ValueError:
                                monthly = df['Close'].resample('M').last().dropna()
                            if len(monthly) > 50:
                                all_data[ticker] = monthly
                        elif len(df.columns) == 1:
                            try:
                                monthly = df.iloc[:, 0].resample('ME').last().dropna()
                            except ValueError:
                                monthly = df.iloc[:, 0].resample('M').last().dropna()
                            if len(monthly) > 50:
                                all_data[ticker] = monthly
                except Exception as e:
                    print(f"Failed to download {ticker}: {e}")
                
                time.sleep(0.2)  # Rate limiting
        
        if len(all_data) < 4:
            raise ValueError(f"Only got {len(all_data)} tickers. Need at least 4. Got: {list(all_data.keys())}")
        
        # Combine into DataFrame
        self.raw_data = pd.DataFrame(all_data)
        
        # Debug: Show actual date range
        
        if progress_callback:
            progress_callback(10, f"Downloaded {len(all_data)} tickers, {len(self.raw_data)} months")
        
        return self.raw_data
    
    def compute_features(self, momentum_window: int = 3, 
                        progress_callback: Callable = None) -> pd.DataFrame:
        """Compute features with graceful handling of missing data."""
        if progress_callback:
            progress_callback(12, "Computing features...")
        
        if self.raw_data is None or len(self.raw_data) == 0:
            self.fetch_data(progress_callback)
        
        df = self.raw_data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Print available columns for debugging
        
        # CORE FEATURES (required)
        # 1. Equity momentum
        if 'SPY' in df.columns:
            features['equity_mom'] = self._zscore(df['SPY'].pct_change(momentum_window))
        else:
            raise ValueError("SPY data required but not available")
        
        # 2. Bond momentum
        if 'TLT' in df.columns:
            features['bond_mom'] = self._zscore(df['TLT'].pct_change(momentum_window))
        
        # 3. Gold/Commodity momentum
        if 'GLD' in df.columns:
            features['gold_mom'] = self._zscore(df['GLD'].pct_change(momentum_window))
        
        # 4. VIX level
        if '^VIX' in df.columns:
            features['vix'] = self._zscore(df['^VIX'])
        
        # 5. Yield curve
        if '^TNX' in df.columns and '^IRX' in df.columns:
            features['yield_curve'] = self._zscore(df['^TNX'] - df['^IRX'])
        elif '^TNX' in df.columns:
            features['yield_curve'] = self._zscore(df['^TNX'])
        
        # 6. Credit spread (HYG/LQD ratio)
        if 'HYG' in df.columns and 'LQD' in df.columns:
            features['credit_spread'] = self._zscore(df['HYG'] / df['LQD'])
        
        # 7. Small/Large cap ratio
        if 'IWM' in df.columns and 'SPY' in df.columns:
            features['small_large'] = self._zscore(df['IWM'] / df['SPY'])
        
        # 8. EM/DM ratio
        if 'EEM' in df.columns and 'SPY' in df.columns:
            features['em_dm'] = self._zscore(df['EEM'] / df['SPY'])
        
        # 9. Growth/Value ratio
        if 'VUG' in df.columns and 'VTV' in df.columns:
            features['growth_value'] = self._zscore(df['VUG'] / df['VTV'])
        
        # Drop rows with NaN and clip outliers
        features = features.dropna()
        features = features.clip(-4, 4)
        
        # Require minimum features
        if len(features.columns) < 4:
            raise ValueError(f"Only {len(features.columns)} features computed. Need at least 4.")
        
        if len(features) < 50:
            raise ValueError(f"Only {len(features)} observations after cleaning. Need at least 50.")
        
        self.features = features
        
        if progress_callback:
            progress_callback(20, f"Computed {len(features.columns)} features, {len(features)} months")
        
        
        return features
    
    def _zscore(self, series: pd.Series) -> pd.Series:
        """Standard rolling z-score."""
        rolling_mean = series.rolling(window=self.zscore_window, min_periods=12).mean()
        rolling_std = series.rolling(window=self.zscore_window, min_periods=12).std()
        return (series - rolling_mean) / (rolling_std + 1e-8)
    
    def get_feature_matrix(self) -> np.ndarray:
        if self.features is None:
            self.compute_features()
        return self.features.values
    
    def get_features_df(self) -> pd.DataFrame:
        if self.features is None:
            self.compute_features()
        return self.features


# =============================================================================
# MAIN DETECTOR CLASS
# =============================================================================

class RegimeDetector:
    """
    Hedge Fund Grade Market Regime Detection.
    
    Usage:
        detector = RegimeDetector(model_type='hsmm')
        detector.fit(progress_callback=lambda p, m: print(f"[{p}%] {m}"))
        regime = detector.get_current_regime()
    """
    
    CACHE_PATH = os.path.expanduser("~/.regime_cache_monthly.pkl")
    
    def __init__(self,
                 model_type: str = 'hsmm',
                 lookback_years: int = 30,
                 use_particle_filter: bool = True,
                 use_tv_transitions: bool = True,
                 use_bayesian: bool = False,
                 stickiness: float = 15.0,  # Reduced from 40 for more realistic transitions
                 use_student_t: bool = True):
        
        self.model_type = model_type
        self.lookback_years = lookback_years
        self.use_particle_filter = use_particle_filter
        self.use_tv_transitions = use_tv_transitions
        self.stickiness = stickiness
        self.use_student_t = use_student_t
        
        self.features = RegimeFeatures(lookback_years=lookback_years)
        self.primary_model = None
        self.particle_filter: Optional[ParticleFilterHMM] = None
        self.tv_transitions: Optional[TimeVaryingTransitionHMM] = None
        self.uncertainty: Optional[UncertaintyQuantifier] = None
        
        self.states: Optional[pd.Series] = None
        self.regime_probs: Optional[np.ndarray] = None
        self.features_df: Optional[pd.DataFrame] = None
        
        self.fitted = False
    
    def fit(self, progress_callback: Callable = None) -> 'RegimeDetector':
        if progress_callback:
            progress_callback(0, "Starting regime detection...")
        
        # Fetch and compute features
        self.features.fetch_data(progress_callback)
        self.features_df = self.features.compute_features(progress_callback=progress_callback)
        X = self.features.get_feature_matrix()
        
        
        if len(X) < 10:
            raise ValueError(f"Not enough data: {len(X)} samples")
        
        if progress_callback:
            progress_callback(25, "Fitting Sticky HMM...")
        
        # Fit primary model
        emission_type = 'student_t' if self.use_student_t else 'gaussian'
        self.primary_model = StickyHMM(
            n_states=N_REGIMES,  # Use global constant (3-state model)
            kappa=self.stickiness,
            emission_type=emission_type,
            nu=5.0
        )
        self.primary_model.fit(X, progress_callback)
        
        self.states = pd.Series(self.primary_model.states_, index=self.features_df.index)
        self.regime_probs = self.primary_model.state_probs_
        
        # Particle filter
        if self.use_particle_filter:
            try:
                if progress_callback:
                    progress_callback(80, "Initializing particle filter...")
                self.particle_filter = ParticleFilterHMM()
                self.particle_filter.initialize(self.primary_model, X[-50:])
            except Exception as e:
                print(f"Particle filter error: {e}")
        
        # Time-varying transitions
        if self.use_tv_transitions and 'vix' in self.features_df.columns:
            try:
                if progress_callback:
                    progress_callback(85, "Fitting time-varying transitions...")
                self.tv_transitions = TimeVaryingTransitionHMM()
                
                # Find available context features
                context_cols = []
                for col in ['vix', 'yield_curve', 'credit_spread']:
                    if col in self.features_df.columns:
                        context_cols.append(col)
                
                if len(context_cols) >= 2:
                    # Pad to 3 features if needed
                    while len(context_cols) < 3:
                        context_cols.append(context_cols[0])
                    contexts = self.features_df[context_cols[:3]].values
                    self.tv_transitions.fit(self.states.values, contexts)
            except Exception as e:
                print(f"TV transitions error: {e}")
        
        self.uncertainty = UncertaintyQuantifier()
        self.uncertainty.add_model(self.primary_model)
        
        self.fitted = True
        
        if progress_callback:
            progress_callback(100, "Regime detection complete")
        
        self.save_model()
        return self
    
    def get_current_regime(self) -> Dict[str, Any]:
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        current_state = int(self.states.iloc[-1])
        current_probs = self.regime_probs[-1]
        
        raw_prob = float(current_probs[current_state])
        capped_prob = min(raw_prob, 0.90)
        
        probs = current_probs + 1e-10
        entropy = -np.sum(probs * np.log(probs))
        uncertainty = entropy / np.log(len(probs))
        
        current_duration = self._compute_current_duration()
        expected_duration = REGIMES[current_state].get('expected_duration_months', 8.0)
        expected_remaining = max(0, expected_duration - current_duration)
        
        primary = {
            'state_id': current_state,
            'name': REGIMES[current_state]['name'],
            'color': REGIMES[current_state]['color'],
            'probability': capped_prob,
            'raw_probability': raw_prob,
            'uncertainty': uncertainty,
            'all_probabilities': {i: float(current_probs[i]) for i in range(len(current_probs))},
            'current_duration': current_duration,
            'expected_duration': expected_duration,
            'expected_remaining': expected_remaining,
        }
        
        change_forecast = self._get_change_forecast(current_state)
        stability = self.uncertainty.compute_regime_stability(self.states.values)
        
        return {
            'primary': primary,
            'change_forecast': change_forecast,
            'stability': stability
        }
    
    def _compute_current_duration(self) -> int:
        if self.states is None or len(self.states) == 0:
            return 0
        current_state = self.states.iloc[-1]
        duration = 1
        for i in range(len(self.states) - 2, -1, -1):
            if self.states.iloc[i] == current_state:
                duration += 1
            else:
                break
        return duration
    
    def _get_change_forecast(self, current_state: int) -> Dict[str, Any]:
        trans_mat = self.get_transition_matrix()
        stay_prob = trans_mat[current_state, current_state]
        
        change_probs = []
        cumulative_stay = 1.0
        for _ in range(12):
            change_probs.append(cumulative_stay * (1 - stay_prob))
            cumulative_stay *= stay_prob
        
        # Calculate expected months until regime change
        # Cap at 120 months (10 years) to avoid unrealistic values from very sticky HMM
        if stay_prob < 0.9999:
            expected_months = 1.0 / (1 - stay_prob)
        else:
            expected_months = 120.0  # Default cap for extremely sticky regimes
        
        # Apply reasonable cap (max 10 years)
        expected_months = min(expected_months, 120.0)
        
        trans_row = trans_mat[current_state].copy()
        trans_row[current_state] = 0
        if trans_row.sum() > 0:
            trans_row /= trans_row.sum()
            next_state = int(np.argmax(trans_row))
        else:
            next_state = (current_state + 1) % N_REGIMES
        
        return {
            '1m': change_probs[0] if change_probs else 0.1,
            '3m': change_probs[2] if len(change_probs) > 2 else 0.3,
            '12m': change_probs[-1] if change_probs else 0.7,
            'expected_months': expected_months,
            'most_likely_next': REGIMES[next_state]['name'],
            'next_state_id': next_state,
            'next_state_color': REGIMES[next_state]['color']
        }
    
    def get_transition_matrix(self) -> np.ndarray:
        if hasattr(self.primary_model, 'get_transition_matrix'):
            return self.primary_model.get_transition_matrix()
        elif hasattr(self.primary_model, 'transition_matrix_'):
            return self.primary_model.transition_matrix_
        return np.full((N_REGIMES, N_REGIMES), 1.0 / N_REGIMES)
    
    def get_regime_history(self) -> pd.DataFrame:
        if self.features_df is None or self.states is None:
            raise ValueError("Model not fitted")
        
        df = self.features_df.copy()
        df['regime'] = self.states.values
        df['regime_name'] = df['regime'].map(lambda x: REGIMES.get(x, {'name': f'State_{x}'})['name'])
        
        for i in range(N_REGIMES):
            if i < self.regime_probs.shape[1]:
                df[f'prob_{REGIMES[i]["name"]}'] = self.regime_probs[:, i]
        
        return df
    
    def save_model(self, path: str = None):
        if not self.fitted:
            return
        
        path = path or self.CACHE_PATH
        cache_data = {
            'detector': self,
            'timestamp': datetime.now(),
            'model_type': self.model_type
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Save error: {e}")
    
    def load_model(self, path: str = None) -> bool:
        path = path or self.CACHE_PATH
        
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if 'detector' in cache_data:
                cached = cache_data['detector']
                self.primary_model = cached.primary_model
                self.states = cached.states
                self.regime_probs = cached.regime_probs
                self.features_df = cached.features_df
                self.particle_filter = getattr(cached, 'particle_filter', None)
                self.tv_transitions = getattr(cached, 'tv_transitions', None)
                self.uncertainty = getattr(cached, 'uncertainty', None)
                self.fitted = cached.fitted
                return True
                
        except Exception as e:
            print(f"Load error: {e}")
        
        return False


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

AdvancedRegimeDetector = RegimeDetector
SimpleGaussianHMM = StickyHMM
HiddenSemiMarkovModel = StickyHMM
BayesianHMM = StickyHMM


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HEDGE FUND GRADE MARKET REGIME DETECTION")
    print("=" * 70)
    
    # Delete cache
    cache_path = os.path.expanduser("~/.regime_cache_monthly.pkl")
    if os.path.exists(cache_path):
        print(f"\nDeleting cache: {cache_path}")
        os.remove(cache_path)
    
    detector = RegimeDetector(
        model_type='hsmm',
        lookback_years=30,
        stickiness=15.0,  # Reduced for more realistic transitions
        use_student_t=True
    )
    
    print("\nFitting model...")
    try:
        
        regime = detector.get_current_regime()
        primary = regime['primary']
        
        print(f"\n{'═' * 70}")
        print(f"CURRENT REGIME: {primary['name']}")
        print(f"{'═' * 70}")
        print(f"  Confidence:   {primary['probability']:.1%}")
        print(f"  Duration:     {primary['current_duration']} months")
        
        print(f"\nREGIME PROBABILITIES:")
        for state_id, prob in primary['all_probabilities'].items():
            name = REGIMES[state_id]['name']
            bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
            print(f"  {name:12s} {bar} {prob:5.1%}")
        
        print(f"\nTRANSITION MATRIX:")
        trans = detector.get_transition_matrix()
        for i in range(N_REGIMES):
            row = " ".join(f"{trans[i,j]:.1%}" for j in range(N_REGIMES))
        
        forecast = regime['change_forecast']
        print(f"\nCHANGE FORECAST:")
        print(f"  1-month:  {forecast['1m']:.1%}")
        print(f"  3-month:  {forecast['3m']:.1%}")
        print(f"  Next:     {forecast['most_likely_next']}")
        
        print(f"\n{'═' * 70}")
        print("SUCCESS! Model saved.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
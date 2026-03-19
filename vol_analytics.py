"""
Volatility Analytics for TTM Squeeze Straddle Strategy.

Features:
  1. Post-squeeze empirical volatility expansion (5y backtest)
  2. Yang-Zhang volatility estimator (OHLC-based)
  3. GARCH(1,1) forward volatility forecast term structure
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional

# Optional: arch package for GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


# ── Squeeze detection (replicated from squeeze.py for standalone backtest) ──

def _detect_squeeze_periods(close: pd.Series, high: pd.Series, low: pd.Series,
                            bb_length: int = 20, bb_mult: float = 2.0,
                            kc_length: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """Returnerar boolean Series: True nar BB ar inuti KC (squeeze on)."""
    # Bollinger Bands
    bb_mid = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Keltner Channels (EMA + ATR)
    kc_mid = close.ewm(span=kc_length, adjust=False).mean()
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(kc_length).mean()
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    return squeeze_on


# ── Feature 1: Post-Squeeze Empirical Volatility Expansion ────────────────

def compute_post_squeeze_expansion(close: pd.Series, high: pd.Series, low: pd.Series,
                                   forward_windows: list = None,
                                   min_squeeze_days: int = 3) -> dict:
    """
    Analysera historiska squeeze-perioder och mata realiserad volatilitet
    efter varje squeeze-exit vid olika forward-fonster.

    Returnerar dict med:
      {
        window_days: {
          'post_rv_mean': float,     # medel realiserad vol efter squeeze (annualiserad %)
          'post_rv_median': float,
          'during_rv_mean': float,   # medel vol under squeeze
          'expansion_mean': float,   # post/during ratio
          'expansion_median': float,
          'n_samples': int,
        },
        ...
      }
    """
    if forward_windows is None:
        forward_windows = [30, 60, 90, 120, 180]

    squeeze_on = _detect_squeeze_periods(close, high, low)

    # Hitta squeeze exits: True -> False transition
    shifted = squeeze_on.shift(1).fillna(False)
    exits = squeeze_on.index[(shifted == True) & (squeeze_on == False)]

    # Hitta squeeze starts for varje exit (for during-squeeze vol)
    squeeze_starts = squeeze_on.index[(squeeze_on == True) & (shifted == False)]

    log_returns = np.log(close / close.shift(1)).dropna()
    sqrt252 = np.sqrt(252)

    results = {}
    for window in forward_windows:
        post_rvs = []
        during_rvs = []

        for exit_date in exits:
            exit_idx = close.index.get_loc(exit_date)

            # Kontrollera att squeeze varade minst min_squeeze_days
            # Hitta narmaste start fore exit
            prior_starts = squeeze_starts[squeeze_starts < exit_date]
            if len(prior_starts) == 0:
                continue
            start_date = prior_starts[-1]
            start_idx = close.index.get_loc(start_date)
            squeeze_duration = exit_idx - start_idx
            if squeeze_duration < min_squeeze_days:
                continue

            # Post-squeeze realiserad vol
            end_idx = exit_idx + window
            if end_idx >= len(close):
                continue  # Inte tillrackligt med data framover

            post_returns = log_returns.iloc[exit_idx:end_idx]
            if len(post_returns) < window * 0.8:  # Krav 80% av fonstret
                continue
            post_rv = float(post_returns.std() * sqrt252 * 100)

            # During-squeeze realiserad vol
            during_returns = log_returns.iloc[start_idx:exit_idx]
            if len(during_returns) > 2:
                during_rv = float(during_returns.std() * sqrt252 * 100)
            else:
                during_rv = None

            post_rvs.append(post_rv)
            if during_rv is not None:
                during_rvs.append(during_rv)

        if post_rvs:
            post_arr = np.array(post_rvs)
            during_arr = np.array(during_rvs) if during_rvs else np.array([np.nan])
            during_mean = float(np.nanmean(during_arr))
            post_mean = float(np.mean(post_arr))

            results[window] = {
                'post_rv_mean': post_mean,
                'post_rv_median': float(np.median(post_arr)),
                'during_rv_mean': during_mean,
                'during_rv_median': float(np.nanmedian(during_arr)) if during_rvs else None,
                'expansion_mean': post_mean / during_mean if during_mean > 0 else None,
                'expansion_median': float(np.median(post_arr)) / float(np.nanmedian(during_arr))
                    if during_rvs and np.nanmedian(during_arr) > 0 else None,
                'n_samples': len(post_rvs),
            }
        else:
            results[window] = {
                'post_rv_mean': None,
                'post_rv_median': None,
                'during_rv_mean': None,
                'during_rv_median': None,
                'expansion_mean': None,
                'expansion_median': None,
                'n_samples': 0,
            }

    return results


# ── Feature 2: Yang-Zhang Volatility Estimator ────────────────────────────

def yang_zhang_vol(open_s: pd.Series, high_s: pd.Series, low_s: pd.Series,
                   close_s: pd.Series, window: int = 20) -> Optional[float]:
    """
    Yang-Zhang (2000) volatilitetsestimator. Anvander OHLC-priser.
    Mer effektiv an close-to-close, hanterar overnight gaps och intraday range.

    Returnerar annualiserad volatilitet i procent, eller None.
    """
    if len(close_s) < window + 1:
        return None

    # Ta de senaste window+1 raderna
    o = open_s.iloc[-(window + 1):].values
    h = high_s.iloc[-(window + 1):].values
    l = low_s.iloc[-(window + 1):].values
    c = close_s.iloc[-(window + 1):].values

    n = window

    # Overnight returns: log(open_t / close_{t-1})
    log_oc = np.log(o[1:] / c[:-1])

    # Close-to-close returns
    log_cc = np.log(c[1:] / c[:-1])

    # Open-to-close returns
    log_co = np.log(c[1:] / o[1:])

    # Rogers-Satchell variance
    log_ho = np.log(h[1:] / o[1:])
    log_lo = np.log(l[1:] / o[1:])
    log_hc = np.log(h[1:] / c[1:])
    log_lc = np.log(l[1:] / c[1:])
    rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

    # Overnight variance
    overnight_var = np.var(log_oc, ddof=1)

    # Close-to-open variance (open-to-close for same day)
    open_var = np.var(log_co, ddof=1)

    # Yang-Zhang k-factor
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    # Combined
    sigma2 = overnight_var + k * open_var + (1 - k) * rs_var

    if sigma2 <= 0:
        return None

    # Annualisera
    return float(np.sqrt(sigma2 * 252) * 100)


# ── Feature 3: GARCH(1,1) Forward Volatility Forecast ────────────────────

def garch_forecast_term_structure(returns: pd.Series,
                                  horizons: list = None) -> dict:
    """
    Fit GARCH(1,1) och generera forward volatilitetsprognos per horizon.

    Returnerar dict:
      {
        'current_vol': float,    # Nuvarande conditional vol (annualiserad %)
        'long_run_vol': float,   # Langsiktig GARCH-vol
        'term_structure': {
          horizon_days: {
            'garch_vol': float,  # Prognos for perioden (annualiserad %)
          },
          ...
        }
      }
    """
    if horizons is None:
        horizons = [30, 60, 90, 120, 180]

    max_h = max(horizons)
    result = {'current_vol': None, 'long_run_vol': None, 'term_structure': {}}

    if len(returns) < 100:
        return result

    # Rensa NaN
    rets = returns.dropna()
    if len(rets) < 100:
        return result

    scale = 100.0  # arch paketet vill ha returns i procent

    if ARCH_AVAILABLE:
        try:
            am = arch_model(rets * scale, vol='Garch', p=1, q=1,
                            mean='Zero', rescale=False)
            res = am.fit(disp='off', show_warning=False)

            # Nuvarande conditional vol
            cond_var = res.conditional_volatility.iloc[-1] ** 2
            result['current_vol'] = float(np.sqrt(cond_var / scale**2 * 252) * 100)

            # Langsiktig vol: omega / (1 - alpha - beta)
            omega = res.params.get('omega', 0)
            alpha = res.params.get('alpha[1]', 0)
            beta = res.params.get('beta[1]', 0)
            persist = alpha + beta
            if 0 < persist < 1 and omega > 0:
                lr_var = omega / (1 - persist)
                result['long_run_vol'] = float(np.sqrt(lr_var / scale**2 * 252) * 100)

            # Term structure forecast
            fcast = res.forecast(horizon=max_h)
            var_fcast = fcast.variance.iloc[-1].values  # array [h.1, h.2, ..., h.max_h]

            for h in horizons:
                # Genomsnittlig daglig varians over horisonten
                avg_var = np.mean(var_fcast[:h])
                vol = float(np.sqrt(avg_var / scale**2 * 252) * 100)
                result['term_structure'][h] = {'garch_vol': vol}

        except Exception as e:
            print(f"[VOL] GARCH fit error: {e}")
            # Fallback: variance targeting
            _variance_target_fallback(rets, horizons, result)
    else:
        _variance_target_fallback(rets, horizons, result)

    return result


def _variance_target_fallback(returns: pd.Series, horizons: list, result: dict):
    """Enkel variance targeting som fallback nar arch inte ar tillgangligt."""
    sqrt252 = np.sqrt(252)
    vol_20d = float(returns.iloc[-20:].std() * sqrt252 * 100)
    vol_full = float(returns.std() * sqrt252 * 100)

    result['current_vol'] = vol_20d
    result['long_run_vol'] = vol_full

    # Enkel mean-reversion: blenda kort och lang HV baserat pa horizon
    for h in horizons:
        weight_long = min(h / 252, 1.0)  # Langre horisont -> mer langsiktig vol
        blended = vol_20d * (1 - weight_long) + vol_full * weight_long
        result['term_structure'][h] = {'garch_vol': round(blended, 1)}


# ── Feature 4: Black-Scholes IV Solver + Vol Surface ──────────────────────

def _bs_price(S, K, T, r, sigma, is_call=True):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_implied_vol(price, S, K, T, r=0.02, is_call=True,
                   tol=1e-6, max_iter=50) -> Optional[float]:
    """Newton-Raphson BS implied volatility solver. Returns IV as decimal or None."""
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price < intrinsic:
        return None

    sigma = 0.3  # Initial guess
    for _ in range(max_iter):
        bs = _bs_price(S, K, T, r, sigma, is_call)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        if vega < 1e-12:
            break
        sigma -= (bs - price) / vega
        if sigma <= 0.001:
            sigma = 0.001
        if abs(bs - price) < tol:
            return sigma
    return sigma if abs(_bs_price(S, K, T, r, sigma, is_call) - price) < price * 0.05 else None


def build_vol_surface_data(chain_data: list, spot: float, r: float = 0.02) -> pd.DataFrame:
    """
    Bygg vol-surface DataFrame fran optionskedja.

    chain_data: lista av dicts med:
      {'strike': float, 'dte': int, 'c_price': float, 'p_price': float}

    Returnerar DataFrame med kolumner:
      Strike, DTE, Moneyness, C_IV, P_IV, MidIV
    """
    rows = []
    for opt in chain_data:
        K = opt['strike']
        dte = opt['dte']
        T = dte / 365.0
        if T <= 0.005 or K <= 0:
            continue

        moneyness = K / spot

        c_iv = None
        p_iv = None
        c_price = opt.get('c_price')
        p_price = opt.get('p_price')

        if c_price and c_price > 0:
            c_iv = bs_implied_vol(c_price, spot, K, T, r, is_call=True)
        if p_price and p_price > 0:
            p_iv = bs_implied_vol(p_price, spot, K, T, r, is_call=False)

        # Konvertera till procent
        if c_iv is not None:
            c_iv *= 100
        if p_iv is not None:
            p_iv *= 100

        # Mid IV: genomsnitt av call och put IV (om bada finns)
        if c_iv is not None and p_iv is not None:
            mid_iv = (c_iv + p_iv) / 2
        else:
            mid_iv = c_iv or p_iv

        if mid_iv is not None:
            rows.append({
                'Strike': K, 'DTE': dte, 'Moneyness': round(moneyness, 3),
                'C_IV': c_iv, 'P_IV': p_iv, 'MidIV': mid_iv,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['DTE', 'Strike']).reset_index(drop=True)


# ── Convenience: Analysera en ticker komplett ──────────────────────────────

def _to_series(s) -> pd.Series:
    """Konvertera DataFrame-kolumn (multi-index) till Series om nodvandigt."""
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s


def analyze_ticker(open_s: pd.Series, high_s: pd.Series, low_s: pd.Series,
                   close_s: pd.Series, forward_windows: list = None) -> dict:
    """
    Kor alla 3 analyser for en ticker.
    Returnerar dict med 'post_squeeze', 'yang_zhang', 'garch'.
    """
    if forward_windows is None:
        forward_windows = [30, 60, 90, 120, 180]

    # Saker att det ar Series (inte DataFrame med multi-index)
    open_s = _to_series(open_s)
    high_s = _to_series(high_s)
    low_s = _to_series(low_s)
    close_s = _to_series(close_s)

    result = {}

    # 1. Post-squeeze expansion
    result['post_squeeze'] = compute_post_squeeze_expansion(
        close_s, high_s, low_s, forward_windows=forward_windows)

    # 2. Yang-Zhang vol
    result['yz_20d'] = yang_zhang_vol(open_s, high_s, low_s, close_s, window=20)
    result['yz_60d'] = yang_zhang_vol(open_s, high_s, low_s, close_s, window=60)

    # 3. GARCH forecast
    log_returns = np.log(close_s / close_s.shift(1)).dropna()
    result['garch'] = garch_forecast_term_structure(log_returns, horizons=forward_windows)

    return result

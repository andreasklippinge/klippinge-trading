"""
EPS & Price Mean Reversion Analyzer
====================================
Analyserar sambandet mellan aktiekurs och EPS (Earnings Per Share).

Kör: python eps_mean_reversion.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")

# ─── Default tickers ───────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "JNJ", "JPM", "PG", "XOM", "V"]


# ═══════════════════════════════════════════════════════════════════
# 1. Data-hämtning
# ═══════════════════════════════════════════════════════════════════

def _get_quarterly_eps(ticker_obj) -> pd.Series | None:
    """Hämta kvartals-EPS — maximerar historik via Yahoo Finance timeseries API.

    Strategi (i prioritetsordning):
      1. Yahoo Finance fundamentals-timeseries API (kvartalsvisa, 10+ år)
      2. quarterly_income_stmt → Diluted EPS / Basic EPS (~5 kvartal)
      3. quarterly_income_stmt → Net Income / Diluted Shares (~5 kvartal)
      4. Annual income_stmt → syntetiska kvartal (förlänger bakåt)
    """

    ticker_name = getattr(ticker_obj, 'ticker', '?')
    best_eps = None

    def _clean_eps(eps_raw: pd.Series, label: str) -> pd.Series | None:
        """Normalisera, sortera, dedup, returnera om ≥4 kvartal."""
        eps = eps_raw.dropna()
        if len(eps) < 4:
            return None
        eps.index = pd.to_datetime(eps.index)
        if eps.index.tz is not None:
            eps.index = eps.index.tz_localize(None)
        eps = eps.sort_index()
        eps = eps[~eps.index.duplicated(keep="last")]
        eps = eps.astype(float)
        print(f"  {ticker_name}: {label} → {len(eps)} kvartal "
              f"({eps.index[0].strftime('%Y-%m')} → {eps.index[-1].strftime('%Y-%m')})")
        return eps

    def _update_best(candidate, label):
        nonlocal best_eps
        if candidate is not None and (best_eps is None or len(candidate) > len(best_eps)):
            best_eps = candidate
            return True
        return False

    # ── Metod 1: Yahoo Finance fundamentals-timeseries API ──────────────
    # Hämtar kvartalsvisa financial data direkt, kan ge 10+ år historik.
    # Använder yfinance's interna session för att hantera cookies/crumbs.
    try:
        import time as _time
        import json

        # Hämta session från yfinance-tickerobjektet
        session = None
        for attr in ['_data', '_session', 'session']:
            obj = getattr(ticker_obj, attr, None)
            if obj is not None:
                if hasattr(obj, 'get'):
                    session = obj
                    break
                elif hasattr(obj, 'session') and hasattr(obj.session, 'get'):
                    session = obj.session
                    break

        if session is None:
            # Fallback: skapa ny session via yfinance
            try:
                from yfinance.utils import requests as yf_requests
                session = yf_requests
            except ImportError:
                import requests
                session = requests.Session()
                session.headers.update({'User-Agent': 'Mozilla/5.0'})

        # Timeseries endpoint — hämta BÅDE quarterly och annual EPS
        ts_types = ("quarterlyDilutedEPS,quarterlyBasicEPS,"
                    "annualDilutedEPS,annualBasicEPS,"
                    "quarterlyNetIncome,quarterlyDilutedAverageShares,"
                    "annualNetIncome,annualDilutedAverageShares")
        period1 = 0  # Från epoch start = all historik
        period2 = int(_time.time())
        url = (f"https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/"
               f"finance/timeseries/{ticker_name}"
               f"?type={ts_types}&period1={period1}&period2={period2}")

        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('timeseries', {}).get('result', [])

            def _parse_ts(result_item, key):
                """Extrahera (dates, values) från timeseries result."""
                entries = result_item.get(key, [])
                if not entries:
                    return [], []
                dates, values = [], []
                for entry in entries:
                    if entry is None:
                        continue
                    date_str = entry.get('asOfDate')
                    # Värdet kan vara under 'reportedValue' eller 'raw'
                    val = None
                    for val_key in ['reportedValue', 'raw']:
                        v = entry.get(val_key)
                        if isinstance(v, dict):
                            val = v.get('raw')
                        elif isinstance(v, (int, float)):
                            val = v
                        if val is not None:
                            break
                    if date_str and val is not None:
                        dates.append(pd.Timestamp(date_str))
                        values.append(float(val))
                return dates, values

            # Samla alla serier
            quarterly_eps = None       # Riktiga kvartalsvisa EPS
            annual_eps = None          # Årlig EPS (för syntetiska kvartal)
            quarterly_ni = None        # Net Income kvartalsvis
            quarterly_shares = None    # Shares kvartalsvis
            annual_ni = None           # Net Income annual
            annual_shares = None       # Shares annual

            for result in results:
                for key in ['quarterlyDilutedEPS', 'quarterlyBasicEPS']:
                    if key in result:
                        d, v = _parse_ts(result, key)
                        if len(d) >= 4:
                            candidate = _clean_eps(
                                pd.Series(v, index=pd.DatetimeIndex(d)),
                                f"timeseries[{key}]")
                            _update_best(candidate, key)
                            if quarterly_eps is None or len(d) > len(quarterly_eps):
                                quarterly_eps = pd.Series(v, index=pd.DatetimeIndex(d))

                for key in ['annualDilutedEPS', 'annualBasicEPS']:
                    if key in result:
                        d, v = _parse_ts(result, key)
                        if len(d) >= 2 and (annual_eps is None or len(d) > len(annual_eps)):
                            annual_eps = pd.Series(v, index=pd.DatetimeIndex(d)).sort_index()
                            print(f"  {ticker_name}: timeseries[{key}] → {len(annual_eps)} annual "
                                  f"({annual_eps.index[0].strftime('%Y-%m')} → {annual_eps.index[-1].strftime('%Y-%m')})")

                # Net Income och Shares för beräkning
                for key, target in [('quarterlyNetIncome', 'q_ni'),
                                     ('quarterlyDilutedAverageShares', 'q_sh'),
                                     ('annualNetIncome', 'a_ni'),
                                     ('annualDilutedAverageShares', 'a_sh')]:
                    if key in result:
                        d, v = _parse_ts(result, key)
                        if len(d) > 0:
                            s = pd.Series(v, index=pd.DatetimeIndex(d)).sort_index()
                            if target == 'q_ni':
                                quarterly_ni = s
                            elif target == 'q_sh':
                                quarterly_shares = s
                            elif target == 'a_ni':
                                annual_ni = s
                            elif target == 'a_sh':
                                annual_shares = s

            # Beräkna annual EPS från NI/Shares om direkt EPS saknas
            if annual_eps is None and annual_ni is not None and annual_shares is not None:
                common = annual_ni.index.intersection(annual_shares.index)
                if len(common) >= 2:
                    annual_eps = (annual_ni[common] / annual_shares[common]).sort_index()
                    print(f"  {ticker_name}: timeseries computed annual EPS → {len(annual_eps)} år")

            # Kombinera: quarterly (riktiga) + annual (syntetiska kvartal bakåt)
            if annual_eps is not None and len(annual_eps) >= 2:
                annual_eps.index = pd.to_datetime(annual_eps.index)
                if annual_eps.index.tz is not None:
                    annual_eps.index = annual_eps.index.tz_localize(None)
                annual_eps = annual_eps.astype(float)

                # Syntetiska kvartal från annual EPS
                synthetic_q = []
                for date, val in annual_eps.items():
                    q_val = val / 4.0
                    for q_off in range(4):
                        q_date = date - pd.DateOffset(months=3 * q_off)
                        synthetic_q.append((q_date, q_val))

                if synthetic_q:
                    syn = pd.Series(
                        [v for _, v in synthetic_q],
                        index=pd.DatetimeIndex([d for d, _ in synthetic_q])
                    ).sort_index()
                    syn = syn[~syn.index.duplicated(keep="last")]

                    # Overlay: riktiga kvartal prioriteras, syntetiska fyller gap
                    if best_eps is not None and len(best_eps) > 0:
                        earliest_real = best_eps.index[0]
                        syn_before = syn[syn.index < earliest_real]
                        if len(syn_before) > 0:
                            combined = pd.concat([syn_before, best_eps]).sort_index()
                            combined = combined[~combined.index.duplicated(keep="last")]
                            print(f"  {ticker_name}: timeseries annual+quarterly → {len(combined)} kvartal "
                                  f"({combined.index[0].strftime('%Y-%m')} → {combined.index[-1].strftime('%Y-%m')})")
                            if len(combined) > len(best_eps):
                                best_eps = combined
                    elif len(syn) >= 8:
                        _update_best(
                            _clean_eps(syn, "timeseries annual→synthetic quarterly"),
                            "annual_synthetic")

            if best_eps is not None and len(best_eps) >= 20:
                print(f"  {ticker_name}: FINAL: {len(best_eps)} kvartal "
                      f"({best_eps.index[0].strftime('%Y-%m')} → {best_eps.index[-1].strftime('%Y-%m')})")
                return best_eps
        else:
            print(f"  {ticker_name}: timeseries API HTTP {resp.status_code}")
    except Exception as e:
        print(f"  {ticker_name}: timeseries API: {type(e).__name__}: {e}")

    # ── Fallback: quarterly_income_stmt (om timeseries misslyckades) ─────
    if best_eps is None:
        try:
            inc = ticker_obj.quarterly_income_stmt
            if inc is not None and len(inc.columns) > 0:
                for row_name in ["Diluted EPS", "Basic EPS"]:
                    if row_name in inc.index:
                        candidate = _clean_eps(inc.loc[row_name], f"quarterly_income_stmt[{row_name}]")
                        _update_best(candidate, row_name)
                        break

            # Förlänger med annual syntetiska kvartal
            if best_eps is not None:
                ann = ticker_obj.income_stmt
                if ann is not None and len(ann.columns) > 0:
                    ann_eps = None
                    for row_name in ["Diluted EPS", "Basic EPS"]:
                        if row_name in ann.index:
                            ann_eps = ann.loc[row_name].dropna().sort_index()
                            break
                    if ann_eps is not None and len(ann_eps) >= 2:
                        ann_eps.index = pd.to_datetime(ann_eps.index)
                        if ann_eps.index.tz is not None:
                            ann_eps.index = ann_eps.index.tz_localize(None)
                        ann_eps = ann_eps.astype(float)
                        synthetic_q = []
                        for date, val in ann_eps.items():
                            q_val = val / 4.0
                            for q_off in range(4):
                                q_date = date - pd.DateOffset(months=3 * q_off)
                                synthetic_q.append((q_date, q_val))
                        if synthetic_q:
                            syn = pd.Series(
                                [v for _, v in synthetic_q],
                                index=pd.DatetimeIndex([d for d, _ in synthetic_q])
                            ).sort_index()
                            syn = syn[~syn.index.duplicated(keep="last")]
                            earliest_real = best_eps.index[0]
                            syn_before = syn[syn.index < earliest_real]
                            if len(syn_before) > 0:
                                combined = pd.concat([syn_before, best_eps]).sort_index()
                                combined = combined[~combined.index.duplicated(keep="last")]
                                print(f"  {ticker_name}: fallback annual+quarterly → {len(combined)} kvartal")
                                best_eps = combined
        except Exception as e:
            print(f"  {ticker_name}: fallback quarterly_income_stmt: {e}")

    if best_eps is not None:
        print(f"  {ticker_name}: FINAL: {len(best_eps)} kvartal "
              f"({best_eps.index[0].strftime('%Y-%m')} → {best_eps.index[-1].strftime('%Y-%m')})")
    else:
        print(f"  {ticker_name}: Ingen EPS-data hittad")

    return best_eps


def fetch_eps_and_price(tickers: list, period: str = "10y") -> dict:
    """
    Hämtar kvartals-EPS och daglig kursdata, beräknar TTM EPS och P/E.
    Returnerar dict: ticker -> {price, eps_quarterly, ttm_eps, pe_ratio}
    """
    # Hämta prisdata i batch (timeout=30 för att undvika att hänga vid concurrent downloads)
    print(f"Hämtar prisdata för {len(tickers)} tickers...")
    if len(tickers) == 1:
        price_df = yf.download(tickers, period=period, progress=False, timeout=30)
        price_data = {tickers[0]: price_df["Close"]}
    else:
        price_df = yf.download(tickers, period=period, group_by="ticker", progress=False, timeout=30)
        price_data = {}
        for t in tickers:
            try:
                col = price_df[t]["Close"].dropna()
                if len(col) > 0:
                    price_data[t] = col
            except (KeyError, TypeError):
                pass

    results = {}
    for t in tickers:
        if t not in price_data or len(price_data[t]) < 100:
            print(f"  {t}: Otillräcklig prisdata, hoppar över")
            continue

        print(f"  {t}: Hämtar EPS-data...")
        try:
            ticker_obj = yf.Ticker(t)
            eps_series = _get_quarterly_eps(ticker_obj)

            if eps_series is None:
                print(f"  {t}: Kunde inte hämta tillräckligt med EPS-data")
                continue

            print(f"  {t}: {len(eps_series)} kvartals-EPS ({eps_series.index[0].strftime('%Y-%m')} → {eps_series.index[-1].strftime('%Y-%m')})")

            # Beräkna TTM EPS (rullande 4-kvartals summa)
            ttm_eps = eps_series.rolling(4).sum().dropna()

            if len(ttm_eps) < 2:
                print(f"  {t}: Otillräcklig TTM EPS-data")
                continue

            # Interpolera TTM EPS till daglig frekvens
            # Steg-interpolation: TTM EPS gäller från rapportdatum tills nästa rapport
            price = price_data[t].copy()
            price.index = pd.to_datetime(price.index)
            if price.index.tz is not None:
                price.index = price.index.tz_localize(None)
            if ttm_eps.index.tz is not None:
                ttm_eps.index = ttm_eps.index.tz_localize(None)

            # Skapa union av index, fyll framåt (step/ffill) — EPS ändras bara vid rapport
            combined_idx = price.index.union(ttm_eps.index).sort_values().drop_duplicates()
            ttm_daily = ttm_eps.reindex(combined_idx).ffill()
            ttm_daily = ttm_daily.reindex(price.index).dropna()

            # Beräkna P/E
            common_idx = price.index.intersection(ttm_daily.index)
            if len(common_idx) < 100:
                print(f"  {t}: Otillräcklig överlappande data ({len(common_idx)} dagar)")
                continue

            price_vals = price.loc[common_idx].values.flatten().astype(float)
            eps_vals = ttm_daily.loc[common_idx].values.flatten().astype(float)

            pe_ratio = pd.Series(price_vals / eps_vals, index=common_idx)
            # Filtrera: ta bort negativ EPS och extrema P/E
            mask = (eps_vals > 0) & np.isfinite(price_vals / eps_vals)
            pe_ratio = pe_ratio[mask]
            pe_ratio = pe_ratio[(pe_ratio > 0) & (pe_ratio < 200)]
            pe_ratio = pe_ratio.dropna()

            if len(pe_ratio) < 50:
                print(f"  {t}: Otillräcklig P/E-data efter filtrering ({len(pe_ratio)})")
                continue

            results[t] = {
                "price": price,
                "eps_quarterly": pd.DataFrame({"EPS": eps_series}),
                "ttm_eps": ttm_daily,
                "pe_ratio": pe_ratio,
            }
            print(f"  {t}: OK — {len(pe_ratio)} dagliga P/E-datapunkter")

        except Exception as e:
            print(f"  {t}: Fel vid datahämtning: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 2. Mean reversion-analys
# ═══════════════════════════════════════════════════════════════════

def hurst_exponent(ts: np.ndarray, max_lag: int = 100) -> float:
    """Beräkna Hurst-exponent via rescaled range (R/S)."""
    n = len(ts)
    if n < 20:
        return np.nan
    max_lag = min(max_lag, n // 2)
    lags = range(2, max_lag + 1)
    rs_values = []
    for lag in lags:
        chunks = [ts[i:i + lag] for i in range(0, n - lag + 1, lag)]
        if len(chunks) < 1:
            continue
        rs_chunk = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            devs = np.cumsum(chunk - mean_c)
            r = np.max(devs) - np.min(devs)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 5:
        return np.nan

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    coeffs = np.polyfit(log_lags, log_rs, 1)
    return coeffs[0]


def analyze_spread_mean_reversion(price: pd.Series, ttm_eps: pd.Series) -> dict | None:
    """
    Analysera mean reversion av spreaden mellan pris och EPS.
    Spread = residual från OLS-regression: Price = α + β·EPS + ε
    Om pris och EPS är kointegrerade mean-reverterar spreaden.
    """
    # Gemensamt index, filtrera bort negativ EPS
    common = price.index.intersection(ttm_eps.index)
    if len(common) < 100:
        return None

    p = price.loc[common].values.flatten().astype(float)
    e = ttm_eps.loc[common].values.flatten().astype(float)
    mask = (e > 0) & np.isfinite(p) & np.isfinite(e)
    p, e, idx = p[mask], e[mask], common[mask]

    if len(p) < 100:
        return None

    # OLS: Price = α + β·EPS → spread = Price - (α + β·EPS)
    X = np.column_stack([np.ones(len(e)), e])
    coeffs = np.linalg.lstsq(X, p, rcond=None)[0]
    alpha, beta = coeffs[0], coeffs[1]
    fitted = alpha + beta * e
    spread = p - fitted
    r_squared = 1 - np.sum(spread**2) / np.sum((p - np.mean(p))**2)

    spread_series = pd.Series(spread, index=idx)

    # ADF-test på spreaden (kointegrations-test)
    try:
        adf_stat, adf_pval, *_ = adfuller(spread, maxlag=int(len(spread) ** (1/3)))
    except Exception:
        adf_stat, adf_pval = np.nan, np.nan

    # Half-life
    try:
        s_lag = spread[:-1]
        s_diff = np.diff(spread)
        Xh = np.column_stack([np.ones(len(s_lag)), s_lag])
        b = np.linalg.lstsq(Xh, s_diff, rcond=None)[0][1]
        half_life = -np.log(2) / np.log(1 + b) if b < 0 and (1 + b) > 0 else np.inf
    except Exception:
        half_life = np.inf

    # Hurst
    h = hurst_exponent(spread)

    # Z-score
    s_mean = np.mean(spread)
    s_std = np.std(spread, ddof=1)
    current_z = (spread[-1] - s_mean) / s_std if s_std > 0 else 0.0

    return {
        "spread": spread_series,
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "current_spread": spread[-1],
        "spread_mean": s_mean,
        "spread_std": s_std,
        "z_score": current_z,
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "half_life": half_life,
        "hurst": h,
    }


def analyze_pe_mean_reversion(pe_series: pd.Series) -> dict:
    """
    Analysera mean reversion-egenskaper hos en P/E-serie.
    Returnerar dict med ADF, half-life, Hurst, z-score m.m.
    """
    pe = pe_series.dropna()
    if len(pe) < 50:
        return None

    pe_vals = pe.values.astype(float)

    # ADF-test
    try:
        adf_stat, adf_pval, *_ = adfuller(pe_vals, maxlag=int(len(pe_vals) ** (1/3)))
    except Exception:
        adf_stat, adf_pval = np.nan, np.nan

    # Half-life via AR(1): ΔPE = α + β·PE_{t-1} + ε
    try:
        pe_lag = pe_vals[:-1]
        pe_diff = np.diff(pe_vals)
        # OLS: pe_diff = a + b * pe_lag
        X = np.column_stack([np.ones(len(pe_lag)), pe_lag])
        beta = np.linalg.lstsq(X, pe_diff, rcond=None)[0]
        b = beta[1]
        if b < 0:
            half_life = -np.log(2) / np.log(1 + b) if (1 + b) > 0 else np.nan
        else:
            half_life = np.inf  # Ej mean-reverterande
    except Exception:
        half_life = np.nan

    # Hurst-exponent
    h = hurst_exponent(pe_vals)

    # Z-score (nuvarande relativt historiskt medel/std)
    pe_mean = np.mean(pe_vals)
    pe_std = np.std(pe_vals, ddof=1)
    current_pe = pe_vals[-1]
    z_score = (current_pe - pe_mean) / pe_std if pe_std > 0 else 0.0

    return {
        "current_pe": current_pe,
        "mean_pe": pe_mean,
        "std_pe": pe_std,
        "z_score": z_score,
        "adf_stat": adf_stat,
        "adf_pval": adf_pval,
        "half_life": half_life,
        "hurst": h,
        "band_1sigma_upper": pe_mean + pe_std,
        "band_1sigma_lower": pe_mean - pe_std,
        "band_2sigma_upper": pe_mean + 2 * pe_std,
        "band_2sigma_lower": pe_mean - 2 * pe_std,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Screening-tabell
# ═══════════════════════════════════════════════════════════════════

def screen_tickers(data: dict) -> pd.DataFrame:
    """
    Screening baserad på spread (Pris vs EPS) mean reversion.

    Kriterier för att passera:
      1. R² ≥ 0.80    — EPS förklarar minst 80% av prisvariationen
      2. ADF p ≤ 0.10  — spreaden visar tecken på stationäritet
      3. Half-life ≤ 252 — mean reversion inom ett år (handelsdagar)
      4. |Z-score| ≥ 1.0 — spreaden är minst 1σ från medel (aktiv signal)
    """
    rows = []
    for ticker, d in data.items():
        spread_stats = analyze_spread_mean_reversion(d["price"], d["ttm_eps"])
        if spread_stats is None:
            continue

        sp = spread_stats
        z = sp["z_score"]
        hl = sp["half_life"]

        # Kriterier
        pass_r2 = sp["r_squared"] >= 0.80
        pass_adf = sp["adf_pval"] <= 0.10
        pass_hl = np.isfinite(hl) and hl <= 252
        pass_z = abs(z) >= 1.0

        passed = sum([pass_r2, pass_adf, pass_hl, pass_z])

        # Signal baserad på z-score
        if z < -2.0:
            signal = "Starkt Undervärderad"
        elif z < -1.0:
            signal = "Undervärderad"
        elif z > 2.0:
            signal = "Starkt Övervärderad"
        elif z > 1.0:
            signal = "Övervärderad"
        else:
            signal = "Neutral"

        rows.append({
            "Ticker": ticker,
            "Spr Z": round(z, 2),
            "ADF p": round(sp["adf_pval"], 4) if not np.isnan(sp["adf_pval"]) else None,
            "HL(d)": round(hl, 0) if np.isfinite(hl) else "∞",
            "R²": round(sp["r_squared"], 3),
            "Hurst": round(sp["hurst"], 3) if not np.isnan(sp["hurst"]) else None,
            "β": round(sp["beta"], 1),
            "Pass": f"{passed}/4",
            "Signal": signal,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("Spr Z", key=abs, ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════
# 4. Plotly-visualisering
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#0a0e1a",
    "card": "#131829",
    "text": "#e0e6ed",
    "text_dim": "#8892a0",
    "accent": "#5dade2",
    "green": "#2ecc71",
    "red": "#e74c3c",
    "orange": "#f39c12",
    "purple": "#9b59b6",
    "band_1s": "rgba(93, 173, 226, 0.15)",
    "band_2s": "rgba(93, 173, 226, 0.07)",
    "grid": "rgba(255,255,255,0.06)",
}


def build_dashboard(data: dict, screening_df: pd.DataFrame) -> go.Figure:
    """Bygg interaktivt Plotly-dashboard med dropdown per aktie."""

    tickers = list(data.keys())
    if not tickers:
        print("Inga aktier att visualisera!")
        return None

    # 2 plot-rader + 1 tabell
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.30, 0.35, 0.35],
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": True}],
            [{}],
            [{"type": "table"}],
        ],
    )

    # Förberäkna data per ticker
    all_traces = {}
    for t in tickers:
        d = data[t]
        spread_stats = analyze_spread_mean_reversion(d["price"], d["ttm_eps"])
        if spread_stats is None:
            continue

        pe = d["pe_ratio"]
        price = d["price"].loc[pe.index[0]:pe.index[-1]]
        ttm = d["ttm_eps"].loc[pe.index[0]:pe.index[-1]]

        all_traces[t] = {
            "price": price,
            "ttm": ttm,
            "spread_stats": spread_stats,
        }

    active_tickers = list(all_traces.keys())
    if not active_tickers:
        print("Inga aktier med giltig data!")
        return None

    # Lägg till traces för ALLA tickers, göm alla utom den första
    trace_map = {}  # ticker -> list of trace indices
    trace_idx = 0

    for i, t in enumerate(active_tickers):
        td = all_traces[t]
        visible = (i == 0)
        trace_map[t] = []

        # --- Row 1: Pris + EPS (dubbel y-axel) ---
        fig.add_trace(
            go.Scatter(
                x=td["price"].index, y=td["price"].values.flatten(),
                name="Pris", line=dict(color=COLORS["accent"], width=1.5),
                visible=visible, showlegend=visible,
            ),
            row=1, col=1, secondary_y=False,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        fig.add_trace(
            go.Scatter(
                x=td["ttm"].index, y=td["ttm"].values.flatten(),
                name="TTM EPS", line=dict(color=COLORS["green"], width=1.5),
                visible=visible, showlegend=visible,
            ),
            row=1, col=1, secondary_y=True,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        # --- Row 2: Spread med σ-band ---
        sp = td["spread_stats"]
        spread = sp["spread"]
        sp_mean = sp["spread_mean"]
        sp_std = sp["spread_std"]

        # Spread-linje
        fig.add_trace(
            go.Scatter(
                x=spread.index, y=spread.values,
                name="Spread", line=dict(color=COLORS["purple"], width=1.5),
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        # Medel
        fig.add_trace(
            go.Scatter(
                x=[spread.index[0], spread.index[-1]], y=[sp_mean, sp_mean],
                line=dict(color=COLORS["text_dim"], dash="dash", width=1),
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        # ±1σ band
        fig.add_trace(
            go.Scatter(
                x=[spread.index[0], spread.index[-1]],
                y=[sp_mean + sp_std, sp_mean + sp_std],
                line=dict(color="rgba(155,89,182,0.4)", dash="dot", width=1),
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        fig.add_trace(
            go.Scatter(
                x=[spread.index[0], spread.index[-1]],
                y=[sp_mean - sp_std, sp_mean - sp_std],
                line=dict(color="rgba(155,89,182,0.4)", dash="dot", width=1),
                fill="tonexty", fillcolor="rgba(155,89,182,0.10)",
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        # ±2σ band
        fig.add_trace(
            go.Scatter(
                x=[spread.index[0], spread.index[-1]],
                y=[sp_mean + 2*sp_std, sp_mean + 2*sp_std],
                line=dict(color="rgba(155,89,182,0.2)", dash="dot", width=1),
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

        fig.add_trace(
            go.Scatter(
                x=[spread.index[0], spread.index[-1]],
                y=[sp_mean - 2*sp_std, sp_mean - 2*sp_std],
                line=dict(color="rgba(155,89,182,0.2)", dash="dot", width=1),
                fill="tonexty", fillcolor="rgba(155,89,182,0.05)",
                visible=visible, showlegend=False,
            ),
            row=2, col=1,
        )
        trace_map[t].append(trace_idx); trace_idx += 1

    # --- Row 3: Screening-tabell (alltid synlig) ---
    if len(screening_df) > 0:
        signal_colors = []
        for sig in screening_df["Signal"]:
            if "Undervärderad" in sig:
                signal_colors.append(COLORS["green"])
            elif "Övervärderad" in sig:
                signal_colors.append(COLORS["red"])
            else:
                signal_colors.append(COLORS["text_dim"])

        n_cols = len(screening_df.columns)
        cell_fill = [[COLORS["card"]] * len(screening_df) for _ in range(n_cols)]
        cell_font_color = [[COLORS["text"]] * len(screening_df) for _ in range(n_cols)]

        cols_list = list(screening_df.columns)
        sig_idx = cols_list.index("Signal") if "Signal" in cols_list else -1
        if sig_idx >= 0:
            cell_font_color[sig_idx] = signal_colors

        # Färga Pass-kolumnen
        pass_idx = cols_list.index("Pass") if "Pass" in cols_list else -1
        if pass_idx >= 0:
            pass_colors = []
            for v in screening_df["Pass"]:
                n = int(str(v).split("/")[0]) if "/" in str(v) else 0
                if n == 4:
                    pass_colors.append(COLORS["green"])
                elif n >= 3:
                    pass_colors.append(COLORS["orange"])
                else:
                    pass_colors.append(COLORS["red"])
            cell_font_color[pass_idx] = pass_colors

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in screening_df.columns],
                    fill_color="#1a2035",
                    font=dict(color=COLORS["text"], size=12, family="JetBrains Mono, monospace"),
                    align="center",
                    line_color=COLORS["grid"],
                ),
                cells=dict(
                    values=[screening_df[c].tolist() for c in screening_df.columns],
                    fill_color=cell_fill,
                    font=dict(color=cell_font_color, size=11, family="JetBrains Mono, monospace"),
                    align="center",
                    line_color=COLORS["grid"],
                    height=28,
                ),
            ),
            row=3, col=1,
        )

    # --- Dropdown-meny ---
    total_ticker_traces = trace_idx

    def _make_title(t):
        sp = all_traces[t]["spread_stats"]
        hl = f"{sp['half_life']:.0f}d" if np.isfinite(sp['half_life']) else "∞"
        return (
            f"<b>{t}</b> — Spread Z: {sp['z_score']:+.2f}  |  "
            f"ADF p: {sp['adf_pval']:.4f}  |  Half-life: {hl}  |  "
            f"R²: {sp['r_squared']:.3f}  |  β: {sp['beta']:.1f}"
        )

    buttons = []
    for i, t in enumerate(active_tickers):
        visibility = [False] * total_ticker_traces
        for idx in trace_map[t]:
            visibility[idx] = True
        # Tabell alltid synlig
        visibility.append(True)

        sp = all_traces[t]["spread_stats"]
        buttons.append(dict(
            label=f"{t}  (Spr Z={sp['z_score']:+.2f}, R²={sp['r_squared']:.2f})",
            method="update",
            args=[
                {"visible": visibility},
                {"title.text": _make_title(t)},
            ],
        ))

    # Layout
    first = active_tickers[0]

    fig.update_layout(
        title=dict(
            text=_make_title(first),
            font=dict(size=13, color=COLORS["text"], family="JetBrains Mono, monospace"),
            x=0.01,
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0, y=1.14,
            xanchor="left", yanchor="top",
            buttons=buttons,
            bgcolor="#1a2035",
            font=dict(color=COLORS["text"], size=11, family="JetBrains Mono, monospace"),
            bordercolor=COLORS["grid"],
        )],
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="JetBrains Mono, monospace"),
        height=1000,
        margin=dict(l=60, r=40, t=120, b=40),
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # Axis styling
    axis_common = dict(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
        showgrid=True,
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)

    fig.update_yaxes(title_text="Pris", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="TTM EPS", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Spread (USD)", row=2, col=1)

    # Subplot titlar
    subplot_labels = [
        (0.5, 1.0, "Pris & TTM EPS"),
        (0.5, 0.63, "Spread: Pris vs EPS (OLS-residual) med σ-band"),
    ]
    for x, y, text in subplot_labels:
        fig.add_annotation(
            text=text, xref="paper", yref="paper", x=x, y=y,
            showarrow=False,
            font=dict(size=12, color=COLORS["text_dim"], family="JetBrains Mono, monospace"),
        )

    return fig


# ═══════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EPS & Price Mean Reversion Analyzer")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS,
                        help="Tickers att analysera (default: AAPL MSFT GOOG JNJ JPM PG XOM V)")
    parser.add_argument("--period", default="10y", help="Historisk period (default: 10y)")
    parser.add_argument("--output", default=None, help="Spara HTML till fil (default: öppna i webbläsare)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    print(f"\n{'='*60}")
    print(f"  EPS & Price Mean Reversion Analyzer")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Period: {args.period}")
    print(f"{'='*60}\n")

    # Hämta data
    data = fetch_eps_and_price(tickers, period=args.period)

    if not data:
        print("\nIngen data kunde hämtas. Kontrollera tickers och internetanslutning.")
        sys.exit(1)

    # Screening-tabell
    print("\nBeräknar mean reversion-statistik...")
    screening_df = screen_tickers(data)

    if len(screening_df) > 0:
        print(f"\n{'─'*80}")
        print(screening_df.to_string(index=False))
        print(f"{'─'*80}")
    else:
        print("Inga aktier klarade screening-filtren.")

    # Bygg dashboard
    print("\nSkapar Plotly-dashboard...")
    fig = build_dashboard(data, screening_df)

    if fig is None:
        sys.exit(1)

    import os, webbrowser
    output_path = args.output or os.path.join(os.path.dirname(__file__), "Trading", "pe_mean_reversion.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"\nDashboard sparat till: {output_path}")
    webbrowser.open(f"file:///{os.path.abspath(output_path).replace(os.sep, '/')}")
    print("Öppnat i webbläsaren.")


if __name__ == "__main__":
    main()

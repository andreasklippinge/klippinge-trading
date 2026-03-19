"""
Nasdaq Nordic Options - Hämta optionskedjor och beräkna straddle-priser.

API-endpoints:
  Screener:     https://api.nasdaq.com/api/nordic/screener/shares?category=MAIN_MARKET&tableonly=true
  Search:       https://api.nasdaq.com/api/nordic/search?searchText={query}
  Option chain: https://api.nasdaq.com/api/nordic/instruments/{orderBookID}/option-chain

OMX option naming:
  Calls: C=Mar, D=Apr, E=May, F=Jun, G=Jul, H=Aug, I=Sep, J=Oct, K=Nov, L=Dec
  Puts:  O=Mar, P=Apr, Q=May, R=Jun, S=Jul, T=Aug, U=Sep, V=Oct, W=Nov, X=Dec
"""

import re
import math
import time
import requests
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, date

CALL_CODES = set('CDEFGHIJKL')
PUT_CODES = set('OPQRSTUVWX')

# Mappa yfinance-ticker (.ST) till Nasdaq Nordic-symbol
# Exempel: "AAK.ST" -> "AAK", "ASSA-B.ST" -> "ASSA B"
def _yf_to_nasdaq_symbol(yf_ticker: str) -> str:
    """Konvertera yfinance-ticker till Nasdaq Nordic symbol."""
    sym = yf_ticker.replace('.ST', '')
    sym = sym.replace('-', ' ')
    return sym


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.nasdaq.com/',
    })
    return session


# ── Ticker -> orderBookID mappning ──────────────────────────────────────────

def fetch_all_shares_screener(session: requests.Session = None) -> dict:
    """
    Hämta alla aktier från Nasdaq Nordic screener.
    Returnerar dict: nasdaq_symbol -> {orderbookId, fullName, isin, lastPrice, ...}
    """
    if session is None:
        session = _make_session()

    url = "https://api.nasdaq.com/api/nordic/screener/shares?category=MAIN_MARKET&tableonly=true"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = data.get('data', {}).get('table', {}).get('rows', [])
    result = {}
    for row in rows:
        symbol = row.get('symbol', '')
        if symbol:
            result[symbol.upper()] = {
                'orderbookId': row.get('orderbookId', ''),
                'fullName': row.get('fullName', ''),
                'isin': row.get('isin', ''),
                'lastPrice': row.get('lastPrice', ''),
                'currency': row.get('currency', ''),
            }
    return result


def build_ticker_mapping(yf_tickers: list, session: requests.Session = None) -> dict:
    """
    Bygg mappning: yf_ticker -> orderBookID.

    Steg 1: Screener (alla aktier i en request)
    Steg 2: Individuell search för tickers som inte matchade

    Returnerar dict: {"AAK.ST": "TX209114", ...}
    """
    if session is None:
        session = _make_session()

    print("[OPTIONS] Hamtar Nasdaq Nordic screener...")
    screener = fetch_all_shares_screener(session)
    print(f"[OPTIONS] Screener: {len(screener)} aktier")

    mapping = {}
    unmatched = []

    for yf_ticker in yf_tickers:
        nasdaq_sym = _yf_to_nasdaq_symbol(yf_ticker).upper()

        # Direkt match
        if nasdaq_sym in screener:
            mapping[yf_ticker] = screener[nasdaq_sym]['orderbookId']
            continue

        # Prova utan mellanslag (ASSA B -> ASSAB)
        no_space = nasdaq_sym.replace(' ', '')
        for key, val in screener.items():
            if key.replace(' ', '') == no_space:
                mapping[yf_ticker] = val['orderbookId']
                break
        else:
            unmatched.append(yf_ticker)

    # Steg 2: Search API för omatchade
    if unmatched:
        print(f"[OPTIONS] {len(unmatched)} omatchade, provar search API...")
        for yf_ticker in unmatched:
            nasdaq_sym = _yf_to_nasdaq_symbol(yf_ticker)
            try:
                url = f"https://api.nasdaq.com/api/nordic/search?searchText={nasdaq_sym}"
                r = session.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    groups = data.get('data', [])
                    for group in groups:
                        if 'Shares' in group.get('group', ''):
                            for inst in group.get('instruments', []):
                                if inst.get('symbol', '').upper().replace(' ', '') == nasdaq_sym.upper().replace(' ', ''):
                                    mapping[yf_ticker] = inst['orderbookId']
                                    break
                            if yf_ticker in mapping:
                                break
                time.sleep(0.3)  # Rate limit
            except Exception as e:
                print(f"[OPTIONS] Search failed for {yf_ticker}: {e}")

    matched = len(mapping)
    total = len(yf_tickers)
    still_unmatched = [t for t in yf_tickers if t not in mapping]
    print(f"[OPTIONS] Mappade {matched}/{total} tickers")
    if still_unmatched:
        print(f"[OPTIONS] Ej mappade: {still_unmatched}")

    return mapping


# ── Option chain ────────────────────────────────────────────────────────────

def classify_option(name: str) -> str:
    """Klassificera Call/Put fran OMX-optionsnamn."""
    m = re.search(r'\d([A-Z])\d', name)
    if not m:
        return 'UNKNOWN'
    code = m.group(1)
    if code in CALL_CODES:
        return 'CALL'
    elif code in PUT_CODES:
        return 'PUT'
    return 'UNKNOWN'


def fetch_option_chain(orderbook_id: str, session: requests.Session = None) -> dict:
    """Hamta option chain fran Nasdaq Nordic API."""
    if session is None:
        session = _make_session()
    url = f"https://api.nasdaq.com/api/nordic/instruments/{orderbook_id}/option-chain"
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def parse_option_chain(data: dict) -> pd.DataFrame:
    """Parsa API-svar till DataFrame med optionType (CALL/PUT)."""
    listing = data.get('data', {}).get('instrumentListing', {})
    rows = listing.get('rows', [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    options = df[df['assetClass'] != 'FUTURES_FORWARDS'].copy()
    if options.empty:
        return options

    options['optionType'] = options['fullName'].apply(classify_option)
    for col in ['strikePrice', 'bidPrice', 'askPrice', 'lastSalePrice']:
        if col in options.columns:
            options[col] = pd.to_numeric(options[col], errors='coerce')

    return options


def get_spot_from_chain(data: dict) -> Optional[float]:
    """Extrahera spot-pris fran futures i option chain-svaret."""
    listing = data.get('data', {}).get('instrumentListing', {})
    rows = listing.get('rows', [])
    futures = [r for r in rows if r.get('assetClass') == 'FUTURES_FORWARDS']
    if futures:
        for f in futures:
            for col in ['lastSalePrice', 'bidPrice']:
                val = pd.to_numeric(f.get(col, ''), errors='coerce')
                if pd.notna(val) and val > 0:
                    return float(val)
    return None


# ── Black-Scholes Implied Volatility ────────────────────────────────────────

def _bs_price(S, K, T, sigma, r=0.03, option_type='call'):
    """Black-Scholes optionspris."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _norm_cdf(x):
    """Standard normal CDF (snabb approximation)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def implied_volatility(price, S, K, T, r=0.03, option_type='call', tol=1e-5, max_iter=50):
    """
    Beräkna implied volatility med Newton-Raphson.
    price: optionspris (mid eller last)
    S: spot/underliggande
    K: strike
    T: tid till expiry (år)
    Returnerar IV som decimal (0.25 = 25%) eller None.
    """
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None

    # Intrinsic value check
    if option_type == 'call':
        intrinsic = max(S - K * math.exp(-r * T), 0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0)
    if price < intrinsic:
        return None

    sigma = 0.3  # Initial guess
    for _ in range(max_iter):
        bs = _bs_price(S, K, T, sigma, r, option_type)
        diff = bs - price
        if abs(diff) < tol:
            return sigma

        # Vega
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * _norm_pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if sigma <= 0.001:
            sigma = 0.001

    return sigma if 0.01 < sigma < 5.0 else None


def straddle_iv(call_price, put_price, S, K, T, r=0.03):
    """Beräkna genomsnittlig IV från straddle (call + put)."""
    iv_c = implied_volatility(call_price, S, K, T, r, 'call') if call_price and call_price > 0 else None
    iv_p = implied_volatility(put_price, S, K, T, r, 'put') if put_price and put_price > 0 else None

    if iv_c and iv_p:
        return (iv_c + iv_p) / 2
    return iv_c or iv_p


def _years_to_expiry(expiry_str: str) -> float:
    """Konvertera 'YYYY-MM-DD' till år från idag."""
    try:
        exp = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        delta = (exp - date.today()).days
        return max(delta, 1) / 365.0
    except (ValueError, TypeError):
        return 0.0


# ── Straddle-beräkning ──────────────────────────────────────────────────────

def build_straddle_table(options: pd.DataFrame, spot: float = None) -> pd.DataFrame:
    """Matcha Call+Put per (expiry, strike), beräkna straddle-priser och IV."""
    calls = options[options['optionType'] == 'CALL'].copy()
    puts = options[options['optionType'] == 'PUT'].copy()

    merged = calls.merge(
        puts, on=['expirationDate', 'strikePrice'],
        suffixes=('_call', '_put'), how='outer'
    )

    def _oi(col):
        return pd.to_numeric(
            merged.get(col, pd.Series(dtype=float)).astype(str).str.replace(',', ''),
            errors='coerce'
        ).fillna(0).astype(int)

    def _best(ask_col, last_col):
        ask = merged.get(ask_col, pd.Series(dtype=float))
        last = merged.get(last_col, pd.Series(dtype=float))
        return ask.fillna(last)

    c_price = _best('askPrice_call', 'lastSalePrice_call')
    p_price = _best('askPrice_put', 'lastSalePrice_put')

    result = pd.DataFrame({
        'Expiry': merged['expirationDate'],
        'Strike': merged['strikePrice'],
        'C_Bid': merged.get('bidPrice_call'),
        'C_Ask': merged.get('askPrice_call'),
        'C_Last': merged.get('lastSalePrice_call'),
        'P_Bid': merged.get('bidPrice_put'),
        'P_Ask': merged.get('askPrice_put'),
        'P_Last': merged.get('lastSalePrice_put'),
        'C_OI': _oi('openInterest_call'),
        'P_OI': _oi('openInterest_put'),
    })

    result['Straddle'] = c_price + p_price
    result['Cost_pct'] = (result['Straddle'] / result['Strike'] * 100).round(2)

    # Beräkna Implied Volatility per rad
    if spot and spot > 0:
        ivs = []
        for _, row in result.iterrows():
            T = _years_to_expiry(row['Expiry'])
            c_val = row['C_Last'] if pd.notna(row['C_Last']) else None
            p_val = row['P_Last'] if pd.notna(row['P_Last']) else None
            iv = straddle_iv(c_val, p_val, spot, row['Strike'], T)
            ivs.append(round(iv * 100, 1) if iv else None)
        result['IV'] = ivs
    else:
        result['IV'] = None

    return result.sort_values(['Expiry', 'Strike']).reset_index(drop=True)


def find_atm_straddle(straddle_df: pd.DataFrame, spot: float, expiry: str = None) -> Optional[pd.Series]:
    """Hitta narmaste ATM straddle for en given expiry (eller narmaste)."""
    priced = straddle_df[straddle_df['Straddle'].notna()]
    if priced.empty:
        return None

    if expiry:
        priced = priced[priced['Expiry'] == expiry]
        if priced.empty:
            return None

    idx = (priced['Strike'] - spot).abs().idxmin()
    return priced.loc[idx]


def get_atm_straddles_per_expiry(straddle_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Hitta narmaste ATM straddle per expiry."""
    priced = straddle_df[straddle_df['Straddle'].notna()]
    if priced.empty:
        return pd.DataFrame()

    atm_rows = []
    for expiry, group in priced.groupby('Expiry'):
        idx = (group['Strike'] - spot).abs().idxmin()
        atm_rows.append(group.loc[idx])
    return pd.DataFrame(atm_rows)


# ── Hög-nivå: Hämta straddle-data för en ticker ────────────────────────────

def get_straddle_summary(orderbook_id: str, spot: float = None,
                         session: requests.Session = None) -> dict:
    """
    Hamta komplett straddle-sammanfattning for en ticker.

    Returnerar dict med:
      - ticker_orderbook_id
      - spot (fran futures eller givet)
      - expiries: lista av datum
      - atm_straddles: DataFrame med ATM straddles per expiry
      - nearest_atm: Series med narmaste ATM straddle
      - all_straddles: full DataFrame
    """
    if session is None:
        session = _make_session()

    raw = fetch_option_chain(orderbook_id, session)
    options = parse_option_chain(raw)

    if options.empty:
        return {'error': 'no_options', 'n_options': 0}

    # Spot-pris
    if spot is None:
        spot = get_spot_from_chain(raw)
    if spot is None:
        # Uppskatta fran OI-viktat strike (kör utan IV först)
        straddles_tmp = build_straddle_table(options, spot=None)
        straddles_tmp['Total_OI'] = straddles_tmp['C_OI'] + straddles_tmp['P_OI']
        active = straddles_tmp[straddles_tmp['Total_OI'] > 0]
        if not active.empty:
            spot = round((active['Strike'] * active['Total_OI']).sum() / active['Total_OI'].sum(), 1)
        else:
            spot = float(options['strikePrice'].median())

    # Bygg straddle-tabell med IV (nu när spot är känt)
    straddles = build_straddle_table(options, spot=spot)

    expiries = sorted(options['expirationDate'].unique())
    atm_straddles = get_atm_straddles_per_expiry(straddles, spot)

    # nearest_atm: välj narmaste expiry som har IV (minst 7 dagar kvar)
    nearest_atm = None
    if not atm_straddles.empty:
        with_iv = atm_straddles[atm_straddles['IV'].notna()]
        if not with_iv.empty:
            nearest_atm = with_iv.iloc[0]
        else:
            nearest_atm = atm_straddles.iloc[0]

    return {
        'orderbook_id': orderbook_id,
        'spot': spot,
        'n_options': len(options),
        'n_calls': len(options[options['optionType'] == 'CALL']),
        'n_puts': len(options[options['optionType'] == 'PUT']),
        'expiries': expiries,
        'atm_straddles': atm_straddles,
        'nearest_atm': nearest_atm,
        'all_straddles': straddles,
    }


# ── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from app_config import OMXS_LARGE_CAP

    session = _make_session()

    # Bygg mappning
    mapping = build_ticker_mapping(OMXS_LARGE_CAP, session)

    print(f"\n{'='*60}")
    print(f"MAPPNING: {len(mapping)} av {len(OMXS_LARGE_CAP)} tickers")
    print(f"{'='*60}")
    for yf, obid in sorted(mapping.items()):
        print(f"  {yf:20s} -> {obid}")

    # Testa straddle for AAK
    if "AAK.ST" in mapping:
        print(f"\n{'='*60}")
        print("AAK STRADDLE-DATA")
        print(f"{'='*60}")
        summary = get_straddle_summary(mapping["AAK.ST"], session=session)
        print(f"Spot: {summary['spot']}")
        print(f"Options: {summary['n_options']} ({summary['n_calls']}C / {summary['n_puts']}P)")
        print(f"Expiries: {summary['expiries']}")
        if summary['nearest_atm'] is not None:
            atm = summary['nearest_atm']
            print(f"\nNarmaste ATM: Strike={atm['Strike']}, "
                  f"Straddle={atm['Straddle']:.1f}, Cost={atm['Cost_pct']:.1f}%")
        if not summary['atm_straddles'].empty:
            print("\nATM per expiry:")
            cols = ['Expiry', 'Strike', 'C_Last', 'P_Last', 'Straddle', 'Cost_pct', 'C_OI', 'P_OI']
            avail = [c for c in cols if c in summary['atm_straddles'].columns]
            print(summary['atm_straddles'][avail].to_string(index=False))

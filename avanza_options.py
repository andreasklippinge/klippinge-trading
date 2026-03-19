"""
Avanza Options - Hamta optionskedjor och berakna straddle-priser.

API-endpoints:
  Filter:  GET  https://www.avanza.se/_api/market-option-future-forward-list/filter-options
  Matris:  POST https://www.avanza.se/_api/market-option-future-forward-list/matrix
  Lista:   POST https://www.avanza.se/_api/market-option-future-forward-list/

Namnkodning: {TICKER}{ar}{manadsbokstav}{strike}
  Calls: A-L = Jan-Dec,  Puts: M-X = Jan-Dec
"""

import re
import math
import requests
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.avanza.se/_api/market-option-future-forward-list"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}


# ── Ticker-mappning ─────────────────────────────────────────────────────────

def _yf_to_avanza_name(yf_ticker: str) -> str:
    """Konvertera yfinance-ticker till Avanza-namn for matchning.
    'SAAB-B.ST' -> 'SAAB B', 'AAK.ST' -> 'AAK', 'HM-B.ST' -> 'H&M B'
    """
    name = yf_ticker.replace('.ST', '').replace('-', ' ')
    return name.upper()


# yfinance-ticker -> Avanza displayName (for de som inte matchar automatiskt)
_YF_TO_AVANZA_SPECIAL = {
    'ADDT-B.ST': 'Addtech B',
    'ALFA.ST': 'Alfa Laval',
    'ALLEI.ST': 'Alleima',
    'ASMO-B.ST': 'Asmodee',
    'BALD-B.ST': 'Balder B',
    'BETS-B.ST': 'Betsson B',
    'AZN.ST': 'AstraZeneca',
    'ATCO-A.ST': 'Atlas Copco A',
    'ATCO-B.ST': 'Atlas Copco B',
    'AZA.ST': 'Avanza Bank Holding',
    'AXFO.ST': 'Axfood',
    'BEIJ-B.ST': 'Beijer Ref B',  # kanske inte listad
    'BILL.ST': 'Billerud',
    'BOL.ST': 'Boliden',
    'BONEX.ST': 'BONESUPPORT',
    'CAST.ST': 'Castellum',
    'COFFEE-B.ST': 'Coffee Stain Group B',
    'DOM.ST': 'Dometic Group',
    'ELUX-B.ST': 'Electrolux B',
    'EKTA-B.ST': 'Elekta B',
    'EMBRAC-B.ST': 'Embracer Group B',
    'EPI-A.ST': 'Epiroc A',
    'ERIC-B.ST': 'Ericsson B',
    'EVO.ST': 'Evolution',
    'FABG.ST': 'Fabege',
    'GETI-B.ST': 'Getinge B',
    'HEXA-B.ST': 'Hexagon B',
    'HOLM-B.ST': 'Holmen B',
    'HM-B.ST': 'H&M B',
    'HUSQ-B.ST': 'Husqvarna B',
    'INDU-C.ST': 'Industrivärden C',
    'INVE-B.ST': 'Investor B',
    'JM.ST': 'JM',
    'KINV-B.ST': 'Kinnevik B',
    'LATO-B.ST': 'Latour B',
    'LIFCO-B.ST': 'Lifco B',
    'LUMI.ST': 'Lundin Mining Corporation',
    'MTG-B.ST': 'MTG B',
    'NCC-B.ST': 'NCC B',
    'NDA-SE.ST': 'Nordea Bank',
    'SAVE.ST': 'Nordnet',
    'NEWA-B.ST': 'New Wave B',
    'NIBE-B.ST': 'Nibe Industrier B',
    'NOBA.ST': 'NOBA Bank',
    'PEAB-B.ST': 'Peab B',  # kanske inte listad
    'RATO-B.ST': 'Ratos B',  # kanske inte listad
    'SAAB-B.ST': 'SAAB B',
    'SAGA-B.ST': 'Sagax B',
    'SBB-B.ST': 'SBB Norden B',
    'SAND.ST': 'Sandvik',
    'SCA-B.ST': 'SCA B',
    'SEB-A.ST': 'SEB A',
    'SECU-B.ST': 'Securitas B',
    'SINCH.ST': 'Sinch',
    'SKA-B.ST': 'Skanska B',
    'SKF-B.ST': 'SKF B',
    'SSAB-A.ST': 'SSAB A',
    'SSAB-B.ST': 'SSAB B',  # kanske inte listad (bara A)
    'STERV.ST': 'Stora Enso R',
    'SHB-A.ST': 'Handelsbanken A',
    'SWED-A.ST': 'Swedbank A',
    'SOBI.ST': 'Swedish Orphan Biovitrum',
    'TEL2-B.ST': 'Tele2 B',
    'TELIA.ST': 'Telia Company',
    '8TRA.ST': 'TRATON SE',
    'TREL-B.ST': 'Trelleborg B',
    'VERI.ST': 'Verisure',
    'VOLV-B.ST': 'Volvo B',
    'VOLCAR-B.ST': 'Volvo Car B',
}


def fetch_filter_options(session: requests.Session = None) -> dict:
    """Hamta alla tillgangliga underliggande instrument och expiry-datum."""
    s = session or requests.Session()
    r = s.get(f"{BASE_URL}/filter-options", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def build_ticker_mapping(yf_tickers: list, session: requests.Session = None) -> dict:
    """
    Mappa yfinance-tickers till Avanza orderbookId.
    Returnerar dict: {"SAAB-B.ST": "5401", ...}
    """
    s = session or requests.Session()
    print("[OPTIONS] Hamtar Avanza filter-options...")
    filters = fetch_filter_options(s)

    instruments = filters.get('underlyingInstruments', [])
    print(f"[OPTIONS] {len(instruments)} underliggande instrument")

    # Bygg lookup: normaliserat namn -> orderbookId
    # Avanza format: {value: "26268", displayName: "AAK", numberOfOrderbooks: 398}
    avanza_lookup = {}
    for inst in instruments:
        name = inst.get('displayName', '').upper()
        obid = str(inst.get('value', ''))
        if name and obid:
            avanza_lookup[name] = obid

    mapping = {}
    unmatched = []

    for yf_ticker in yf_tickers:
        matched = False

        # 1) Specialmappning (explicit namn)
        if yf_ticker in _YF_TO_AVANZA_SPECIAL:
            special_name = _YF_TO_AVANZA_SPECIAL[yf_ticker].upper()
            if special_name in avanza_lookup:
                mapping[yf_ticker] = avanza_lookup[special_name]
                matched = True

        # 2) Direkt: "SAAB-B.ST" -> "SAAB B"
        if not matched:
            av_name = _yf_to_avanza_name(yf_ticker)
            if av_name in avanza_lookup:
                mapping[yf_ticker] = avanza_lookup[av_name]
                matched = True

        # 3) Forsta ordet matchar
        if not matched:
            av_name = _yf_to_avanza_name(yf_ticker)
            first_word = av_name.split()[0]
            for key in avanza_lookup:
                if key.split()[0] == first_word:
                    # Kolla suffix om det finns (A/B/C/R)
                    parts = av_name.split()
                    if len(parts) > 1:
                        if key.endswith(parts[-1]):
                            mapping[yf_ticker] = avanza_lookup[key]
                            matched = True
                            break
                    else:
                        mapping[yf_ticker] = avanza_lookup[key]
                        matched = True
                        break

        if not matched:
            unmatched.append(yf_ticker)

    print(f"[OPTIONS] Mappade {len(mapping)}/{len(yf_tickers)} tickers")
    if unmatched:
        print(f"[OPTIONS] Ej mappade: {unmatched[:10]}")

    # Spara expiry-datumen for senare
    _cache_expiry_dates(filters)

    return mapping


_CACHED_EXPIRIES = []

def _cache_expiry_dates(filters: dict):
    """Cache tillgangliga expiry-datum fran filter-options."""
    global _CACHED_EXPIRIES
    _CACHED_EXPIRIES = []
    for group in filters.get('endDates', []):
        children = group.get('children', group.get('endDates', []))
        for ed in children:
            val = ed.get('value', '')
            if val and len(val) == 10:  # YYYY-MM-DD
                _CACHED_EXPIRIES.append(val)
    _CACHED_EXPIRIES.sort()


def get_cached_expiries() -> list:
    return _CACHED_EXPIRIES


# ── Option-data ─────────────────────────────────────────────────────────────

def fetch_option_matrix(orderbook_id: str, expiry: str = None,
                        session: requests.Session = None) -> dict:
    """Hamta option-matris (Call+Put matchade per strike)."""
    s = session or requests.Session()
    body = {
        "filter": {
            "underlyingInstruments": [orderbook_id],
            "optionTypes": ["STANDARD"],
            "endDates": [expiry] if expiry else [],
            "callIndicators": [],
        },
        "offset": 0,
        "limit": 200,
        "sortBy": {"field": "strikePrice", "order": "asc"},
    }
    r = s.post(f"{BASE_URL}/matrix", json=body, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_option_list(orderbook_id: str, expiry: str = None,
                      session: requests.Session = None) -> dict:
    """Hamta platt optionslista."""
    s = session or requests.Session()
    body = {
        "filter": {
            "underlyingInstruments": [orderbook_id],
            "optionTypes": ["STANDARD"],
            "endDates": [expiry] if expiry else [],
            "callIndicators": [],
        },
        "offset": 0,
        "limit": 200,
        "sortBy": {"field": "strikePrice", "order": "asc"},
    }
    r = s.post(f"{BASE_URL}/", json=body, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Avanza Option Analytics (greeks/IV) ────────────────────────────────────

def fetch_option_greeks(orderbook_id: str, session: requests.Session = None) -> dict:
    """Hamta greeks, IV och quote fran Avanza for en enskild option.
    Returnerar dict med greeks + quote-priser (ask/bid/theo).
    """
    s = session or requests.Session()
    result = {}

    # Hamta greeks fran details-endpoint
    try:
        r = s.get(f"https://www.avanza.se/_api/market-guide/option/{orderbook_id}/details",
                  headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            ga = data.get('optionAnalytics', {})
            if ga:
                result.update({
                    'iv_mid': ga.get('implicitMidVolatility'),
                    'iv_bid': ga.get('implicitBuyVolatility'),
                    'iv_ask': ga.get('implicitSellVolatility'),
                    'delta': ga.get('delta'),
                    'gamma': ga.get('gamma'),
                    'theta': ga.get('theta'),
                    'vega': ga.get('vega'),
                    'rho': ga.get('rho'),
                    'risk_free_rate': ga.get('riskFreeInterestRate'),
                    'theo_buy': ga.get('theoreticBuyPrice'),
                    'theo_sell': ga.get('theoreticSellPrice'),
                })
    except Exception:
        pass

    # Hamta quote (bid/ask/last — finns aven efter stangning)
    try:
        r = s.get(f"https://www.avanza.se/_api/market-guide/option/{orderbook_id}",
                  headers=HEADERS, timeout=10)
        if r.status_code == 200:
            q = r.json().get('quote', {})
            result['quote_buy'] = q.get('buy')
            result['quote_sell'] = q.get('sell')
            result['quote_last'] = q.get('last')
    except Exception:
        pass

    return result


# ── Straddle-berakning ──────────────────────────────────────────────────────

def build_straddle_table(matrix_data: dict, spot: float,
                         expiry: str = '') -> pd.DataFrame:
    """
    Bygg straddle-tabell fran Avanza matrix-svar.
    Returnerar DataFrame med strike, bid/ask, straddle.
    """
    matched = matrix_data.get('matchedOptions', [])
    if not matched:
        return pd.DataFrame()

    rows = []
    for m in matched:
        call = m.get('call') or {}
        put = m.get('put') or {}

        strike = call.get('strikePrice') or put.get('strikePrice')
        if not strike:
            continue

        c_bid = call.get('buyPrice')
        c_ask = call.get('sellPrice')
        c_vol = call.get('sellVolume', 0)
        p_bid = put.get('buyPrice')
        p_ask = put.get('sellPrice')
        p_vol = put.get('sellVolume', 0)

        # Straddle (ask-sida = vad det kostar att kopa)
        straddle_ask = None
        if isinstance(c_ask, (int, float)) and isinstance(p_ask, (int, float)):
            straddle_ask = c_ask + p_ask

        rows.append({
            'Expiry': expiry or '',
            'Strike': strike,
            'C_Name': call.get('name', ''),
            'C_Id': str(call.get('orderbookId', '')),
            'C_Bid': c_bid if isinstance(c_bid, (int, float)) and c_bid > 0 else None,
            'C_Ask': c_ask if isinstance(c_ask, (int, float)) else None,
            'P_Name': put.get('name', ''),
            'P_Id': str(put.get('orderbookId', '')),
            'P_Bid': p_bid if isinstance(p_bid, (int, float)) and p_bid > 0 else None,
            'P_Ask': p_ask if isinstance(p_ask, (int, float)) else None,
            'Straddle': straddle_ask,
            'Cost_pct': round(straddle_ask / strike * 100, 2) if straddle_ask and strike else None,
            'IV': None,  # Fylls i separat via fetch_option_greeks
            'C_Vol': c_vol or 0,
            'P_Vol': p_vol or 0,
        })

    return pd.DataFrame(rows)


def find_atm_straddle(straddle_df: pd.DataFrame, spot: float) -> Optional[pd.Series]:
    """Hitta narmaste ATM straddle (narmaste strike till spot)."""
    if straddle_df.empty:
        return None
    # Foredra rader med pris, men acceptera alla (priser fylls i fran greeks senare)
    has_ids = straddle_df[straddle_df['C_Id'].astype(str).str.len() > 0]
    candidates = has_ids if not has_ids.empty else straddle_df
    idx = (candidates['Strike'] - spot).abs().idxmin()
    return candidates.loc[idx]


# ── Hog-niva: Hamta straddle-data for en ticker ────────────────────────────

def get_straddle_summary(orderbook_id: str, spot: float = None,
                         session: requests.Session = None,
                         max_workers: int = 8) -> dict:
    """
    Hamta komplett straddle-sammanfattning for en ticker.
    Parallelliserad: matriser och greeks hamtas concurrent.

    Returnerar dict med:
      - spot, expiries, atm_straddles (DataFrame), nearest_atm, all_straddles
    """
    s = session or requests.Session()

    # Hamta lista for att fa alla expiry-datum och spot
    list_data = fetch_option_list(orderbook_id, session=s)
    underlying = list_data.get('underlyingInstrument', {})
    if spot is None:
        spot = underlying.get('lastPrice', 0)

    n_total = list_data.get('totalNumberOfOrderbooks', 0)

    # Hitta tillgangliga expiry-datum (bara de med options)
    filter_opts = list_data.get('filterOptions', {})
    expiry_dates = []
    for group in filter_opts.get('endDates', []):
        children = group.get('children', group.get('endDates', []))
        for ed in children:
            val = ed.get('value', '')
            n_opts = ed.get('numberOfOrderbooks', 0)
            if val and len(val) == 10 and n_opts > 0:
                expiry_dates.append(val)
    expiry_dates.sort()

    if not expiry_dates:
        return {'error': 'no_expiries', 'spot': spot, 'n_options': n_total}

    # ── Parallell: hamta alla matriser samtidigt ────────────────────────
    all_atm = []
    all_straddles = []
    expiries_to_fetch = expiry_dates[:8]

    def _fetch_matrix(expiry):
        try:
            matrix = fetch_option_matrix(orderbook_id, expiry=expiry, session=s)
            return expiry, matrix
        except Exception as e:
            print(f"[OPTIONS] Matrix error for {expiry}: {e}")
            return expiry, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        matrix_futures = {pool.submit(_fetch_matrix, exp): exp for exp in expiries_to_fetch}
        matrix_results = {}
        for fut in as_completed(matrix_futures):
            exp, matrix = fut.result()
            if matrix is not None:
                matrix_results[exp] = matrix

    # Bearbeta matriser (ordnade efter expiry)
    for expiry in expiries_to_fetch:
        matrix = matrix_results.get(expiry)
        if matrix is None:
            continue
        df = build_straddle_table(matrix, spot, expiry=expiry)
        if df.empty:
            continue
        all_straddles.append(df)
        atm = find_atm_straddle(df, spot)
        if atm is not None:
            all_atm.append(atm)

    atm_df = pd.DataFrame(all_atm) if all_atm else pd.DataFrame()
    full_df = pd.concat(all_straddles, ignore_index=True) if all_straddles else pd.DataFrame()

    # ── Parallell: hamta greeks for alla ATM-optioner samtidigt ─────────
    if not atm_df.empty:
        # Samla alla option-IDs som behover greeks
        greeks_jobs = []  # (row_idx, 'call'|'put', option_id)
        for idx, row in atm_df.iterrows():
            c_id = str(row.get('C_Id', ''))
            p_id = str(row.get('P_Id', ''))
            if c_id:
                greeks_jobs.append((idx, 'call', c_id))
            if p_id:
                greeks_jobs.append((idx, 'put', p_id))

        # Hamta alla parallellt
        greeks_results = {}  # (row_idx, 'call'|'put') -> dict

        def _fetch_greeks(job):
            row_idx, cp, opt_id = job
            try:
                return (row_idx, cp, fetch_option_greeks(opt_id, session=s))
            except Exception:
                return (row_idx, cp, {})

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_fetch_greeks, j) for j in greeks_jobs]
            for fut in as_completed(futs):
                row_idx, cp, data = fut.result()
                greeks_results[(row_idx, cp)] = data

        # Bygg greeks-kolumner fran parallella resultat
        greeks_cols = [
            'C_IV', 'C_IV_Bid', 'C_IV_Ask', 'C_Delta', 'C_Theta', 'C_Vega',
            'P_IV', 'P_IV_Bid', 'P_IV_Ask', 'P_Delta', 'P_Theta', 'P_Vega',
        ]
        greeks_data = {k: [] for k in greeks_cols}
        c_ask_fix = []
        p_ask_fix = []

        for idx, row in atm_df.iterrows():
            cg = greeks_results.get((idx, 'call'), {})
            pg = greeks_results.get((idx, 'put'), {})

            greeks_data['C_IV'].append(cg.get('iv_mid'))
            greeks_data['C_IV_Bid'].append(cg.get('iv_bid'))
            greeks_data['C_IV_Ask'].append(cg.get('iv_ask'))
            greeks_data['C_Delta'].append(cg.get('delta'))
            greeks_data['C_Theta'].append(cg.get('theta'))
            greeks_data['C_Vega'].append(cg.get('vega'))

            greeks_data['P_IV'].append(pg.get('iv_mid'))
            greeks_data['P_IV_Bid'].append(pg.get('iv_bid'))
            greeks_data['P_IV_Ask'].append(pg.get('iv_ask'))
            greeks_data['P_Delta'].append(pg.get('delta'))
            greeks_data['P_Theta'].append(pg.get('theta'))
            greeks_data['P_Vega'].append(pg.get('vega'))

            # Fallback-priser: quote_sell > theo_sell > quote_last
            c_price = row.get('C_Ask')
            if not isinstance(c_price, (int, float)) or pd.isna(c_price):
                c_price = cg.get('quote_sell') or cg.get('theo_sell') or cg.get('quote_last')
            c_ask_fix.append(c_price)

            p_price = row.get('P_Ask')
            if not isinstance(p_price, (int, float)) or pd.isna(p_price):
                p_price = pg.get('quote_sell') or pg.get('theo_sell') or pg.get('quote_last')
            p_ask_fix.append(p_price)

        atm_df = atm_df.copy()
        for k, v in greeks_data.items():
            atm_df[k] = v

        # Uppdatera priser och straddle med fallback-varden
        atm_df['C_Ask'] = c_ask_fix
        atm_df['P_Ask'] = p_ask_fix
        atm_df['Straddle'] = atm_df.apply(
            lambda r: (r['C_Ask'] + r['P_Ask'])
            if isinstance(r['C_Ask'], (int, float)) and isinstance(r['P_Ask'], (int, float))
            else None, axis=1)
        atm_df['Cost_pct'] = atm_df.apply(
            lambda r: round(r['Straddle'] / r['Strike'] * 100, 2)
            if pd.notna(r.get('Straddle')) and r.get('Strike', 0) > 0 else None, axis=1)

        # Medel-IV for convenience (anvands i squeeze-tabellen)
        atm_df['IV'] = atm_df.apply(
            lambda r: round((r['C_IV'] + r['P_IV']) / 2, 1)
            if pd.notna(r.get('C_IV')) and pd.notna(r.get('P_IV'))
            else (round(r['C_IV'], 1) if pd.notna(r.get('C_IV'))
                  else (round(r['P_IV'], 1) if pd.notna(r.get('P_IV')) else None)),
            axis=1)

    # Valj nearest_atm: foredra 60-180 dagar (3-6 manader), annars narmaste med IV
    nearest_atm = None
    if not atm_df.empty:
        with_iv = atm_df[atm_df['IV'].notna()]
        if not with_iv.empty:
            today = date.today()
            def _days_to_exp(exp_str):
                try:
                    return (datetime.strptime(str(exp_str)[:10], '%Y-%m-%d').date() - today).days
                except (ValueError, TypeError):
                    return 0
            with_iv = with_iv.copy()
            with_iv['_dte'] = with_iv['Expiry'].apply(_days_to_exp)
            ideal = with_iv[(with_iv['_dte'] >= 60) & (with_iv['_dte'] <= 180)]
            if not ideal.empty:
                nearest_atm = ideal.iloc[0].drop('_dte')
            else:
                ok = with_iv[with_iv['_dte'] >= 30]
                if not ok.empty:
                    nearest_atm = ok.iloc[0].drop('_dte')
                else:
                    nearest_atm = with_iv.iloc[0].drop('_dte')
        else:
            nearest_atm = atm_df.iloc[0]

    return {
        'orderbook_id': orderbook_id,
        'name': underlying.get('name', ''),
        'spot': spot,
        'n_options': n_total,
        'expiries': expiry_dates,
        'atm_straddles': atm_df,
        'nearest_atm': nearest_atm,
        'all_straddles': full_df,
    }


# ── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from app_config import OMXS_LARGE_CAP

    s = requests.Session()

    # Mappa tickers
    mapping = build_ticker_mapping(OMXS_LARGE_CAP, s)
    print(f"\nMappade {len(mapping)} tickers:")
    for yf, obid in sorted(mapping.items())[:10]:
        print(f"  {yf:20s} -> {obid}")

    # Testa SAAB B
    if "SAAB-B.ST" in mapping:
        print(f"\n{'='*60}")
        print("SAAB B STRADDLE")
        print(f"{'='*60}")
        summary = get_straddle_summary(mapping["SAAB-B.ST"], session=s)
        print(f"Spot: {summary['spot']}")
        print(f"Options: {summary['n_options']}")
        print(f"Expiries: {summary['expiries'][:6]}")

        if summary.get('nearest_atm') is not None:
            atm = summary['nearest_atm']
            print(f"\nNearest ATM: Strike={atm['Strike']}, "
                  f"Straddle={atm['Straddle']:.1f}, IV={atm.get('IV', '-')}%")

        if not summary['atm_straddles'].empty:
            print("\nATM per expiry:")
            cols = ['Expiry', 'Strike', 'C_Bid', 'C_Ask', 'P_Bid', 'P_Ask',
                    'Straddle', 'Cost_pct', 'IV']
            avail = [c for c in cols if c in summary['atm_straddles'].columns]
            print(summary['atm_straddles'][avail].to_string(index=False))

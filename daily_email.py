"""
Klippinge Investment Trading Terminal - Daily Summary Email
============================================================

Bygger och skickar ett HTML-formaterat dagligt sammanfattningsmail
via Gmail SMTP efter börsens stängning.

Designat för att likna rapporter från Goldman Sachs / JPMorgan:
- Mörk bakgrund, guld-accenter, monospace siffror
- Tabelbaserad layout (kompatibel med alla email-klienter)
- Kompakt 2-kolumns grid för marknadsdata
"""

import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

try:
    from PySide6.QtCore import QObject, Signal, Slot, QThread
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal as Signal, pyqtSlot as Slot, QThread


# ── Color palette ────────────────────────────────────────────────────────────
_C = {
    'bg':       '#0b0f19',
    'bg2':      '#0f1422',
    'bg3':      '#131a2b',
    'card':     '#111827',
    'border':   '#1c2537',
    'border_l': '#151d2e',
    'gold':     '#c9a96e',
    'gold_d':   '#a08450',
    'white':    '#e8ecf2',
    'text':     '#9aa5b8',
    'muted':    '#4a5568',
    'dim':      '#2d3748',
    'green':    '#34d399',
    'red':      '#f87171',
    'amber':    '#fbbf24',
    'mono':     "'JetBrains Mono','SF Mono','Consolas','Courier New',monospace",
    'sans':     "'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _chg(val: float, dec: int = 2) -> str:
    """Formaterad change med färg."""
    if val > 0:
        return f'<span style="color:{_C["green"]}">+{val:.{dec}f}%</span>'
    elif val < 0:
        return f'<span style="color:{_C["red"]}">{val:.{dec}f}%</span>'
    return f'<span style="color:{_C["muted"]}">0.00%</span>'


def _td_r(content: str, mono: bool = True) -> str:
    """Right-aligned table cell."""
    ff = f'font-family:{_C["mono"]};font-size:11px;' if mono else ''
    return f'<td style="text-align:right;padding:3px 0 3px 8px;border-bottom:1px solid {_C["border_l"]};{ff}">{content}</td>'


def _td_l(content: str, color: str = None) -> str:
    """Left-aligned table cell."""
    c = color or _C['text']
    return f'<td style="padding:3px 8px 3px 0;border-bottom:1px solid {_C["border_l"]};color:{c}">{content}</td>'


def _section_title(title: str) -> str:
    return (f'<tr><td colspan="10" style="padding:28px 0 10px 0;font-size:10px;font-weight:600;'
            f'color:{_C["muted"]};letter-spacing:2px;text-transform:uppercase;'
            f'border-bottom:1px solid {_C["border"]}">'
            f'<span style="display:inline-block;width:3px;height:10px;background:{_C["gold"]};'
            f'margin-right:8px;vertical-align:middle;border-radius:1px"></span>'
            f'{title}</td></tr>')


def _kpi_cell(label: str, value: str, color: str = None) -> str:
    """Single KPI box for table-based layout."""
    c = color or _C['white']
    return (f'<td style="background:{_C["card"]};border:1px solid {_C["border"]};border-radius:6px;'
            f'padding:14px 8px;text-align:center;width:25%">'
            f'<div style="font-size:9px;color:{_C["muted"]};letter-spacing:1.2px;'
            f'text-transform:uppercase;font-weight:500">{label}</div>'
            f'<div style="font-size:22px;font-weight:700;color:{c};margin-top:4px;'
            f'font-family:{_C["mono"]}">{value}</div></td>')


def _pct_bar_html(pct: float) -> str:
    """Percentile bar for volatility cards."""
    if pct > 80:   bar_c = _C['red']
    elif pct > 50:  bar_c = _C['amber']
    elif pct > 25:  bar_c = _C['gold']
    else:           bar_c = _C['green']
    w = max(3, min(100, pct))
    return (f'<div style="display:inline-block;vertical-align:middle;width:60px;height:4px;'
            f'background:{_C["bg"]};border-radius:2px;margin-right:4px">'
            f'<div style="width:{w}%;height:4px;background:{bar_c};border-radius:2px"></div></div>'
            f'<span style="color:{_C["text"]};font-size:10px">{pct:.0f}</span>')


def _sentiment_label(data: dict) -> tuple:
    """Beräkna marknadssentiment från index- och volatilitetsdata.
    Returnerar (label, color, breadth_pct)."""
    # Räkna positiva/negativa index (exkl. valutor/yields/crypto för renare signal)
    indices = data.get("indices", {})
    equity_regions = {'EUROPE', 'AMERICA', 'ASIA', 'OCEANIA'}
    pos_count = neg_count = 0
    for region, region_items in indices.items():
        if region not in equity_regions:
            continue
        for idx in region_items:
            if idx.get("change_pct", 0) > 0:
                pos_count += 1
            elif idx.get("change_pct", 0) < 0:
                neg_count += 1

    total = pos_count + neg_count
    if total == 0:
        return "NEUTRAL", _C['muted'], 0

    ratio = pos_count / total
    breadth = round(ratio * 100)
    # Justera med VIX-percentil
    vol = data.get("volatility", [])
    vix_pct = 50
    for v in vol:
        if v.get("name") == "VIX":
            vix_pct = v.get("percentile", 50)

    if ratio >= 0.7 and vix_pct < 60:
        return "RISK-ON", _C['green'], breadth
    elif ratio <= 0.3 or vix_pct > 80:
        return "RISK-OFF", _C['red'], breadth
    elif ratio >= 0.55:
        return "CONSTRUCTIVE", "#60a5fa", breadth
    elif ratio <= 0.45:
        return "CAUTIOUS", _C['amber'], breadth
    return "MIXED", _C['muted'], breadth


# ── HTML Builder ─────────────────────────────────────────────────────────────

def build_daily_summary_html(data: dict) -> str:
    """Bygger professionellt HTML-mail i investment bank-stil."""
    timestamp = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))
    version = data.get("version", "1.3.0")
    date_str = datetime.now().strftime("%A, %B %d, %Y")
    time_str = datetime.now().strftime("%H:%M CET")
    sentiment, sentiment_color, breadth = _sentiment_label(data)

    p = []

    # ── Document start ──
    p.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
body {{ margin:0; padding:0; background:{_C['bg']}; color:{_C['text']}; font-family:{_C['sans']};
       font-size:12px; line-height:1.5; -webkit-font-smoothing:antialiased; }}
a {{ color:{_C['gold']}; text-decoration:none; }}
</style></head><body>
<table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{_C['bg']}">
<tr><td align="center">
<table width="760" cellpadding="0" cellspacing="0" border="0" style="background:{_C['bg2']}">""")

    # ── Gold accent bar ──
    p.append(f'<tr><td style="height:3px;background:linear-gradient(90deg,{_C["gold_d"]},{_C["gold"]},{_C["gold_d"]})"></td></tr>')

    # ── Header ──
    p.append(f"""<tr><td style="padding:28px 32px 20px 32px;text-align:center">
<div style="font-size:11px;font-weight:700;letter-spacing:3px;color:{_C['gold']};margin-bottom:4px">KLIPPINGE INVESTMENT</div>
<div style="font-size:22px;font-weight:300;color:{_C['text']};letter-spacing:0.5px;margin-bottom:8px">Daily Market Intelligence</div>
<div style="font-size:11px;color:{_C['muted']};letter-spacing:0.5px">{date_str}</div>
<div style="font-size:10px;color:{_C['dim']};margin-top:2px">{time_str}</div>
</td></tr>""")

    # ── Executive Summary ribbon ──
    # Samla snabb statistik
    indices = data.get("indices", {})
    # Hitta nyckelindex
    all_idx = []
    for items in indices.values():
        all_idx.extend(items)
    key_map = {}
    for idx in all_idx:
        key_map[idx.get("name", "")] = idx

    spx = key_map.get("S&P 500", {})
    ndx = key_map.get("NASDAQ 100", {})
    omx = key_map.get("Stockholm", {})

    vol_data = data.get("volatility", [])
    vix_val = next((v.get("value", 0) for v in vol_data if v.get("name") == "VIX"), 0)

    scan = data.get("scan", {})
    n_signals = len(scan.get("signals", []))
    pf = data.get("portfolio", {})
    pf_pct = pf.get("total_pnl_pct", 0)
    pf_sign = "+" if pf_pct >= 0 else ""
    pf_color = _C['green'] if pf_pct >= 0 else _C['red']

    p.append(f"""<tr><td style="padding:0 36px">
<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background:{_C['card']};border:1px solid {_C['border']};border-radius:0 0 6px 6px">
<tr style="font-size:10px;color:{_C['muted']};letter-spacing:0.5px">
<td style="padding:12px 16px;border-right:1px solid {_C['border']}">
    <span style="text-transform:uppercase;letter-spacing:1px">Sentiment</span><br>
    <span style="font-size:13px;font-weight:700;color:{sentiment_color};letter-spacing:0.5px">{sentiment}</span></td>
<td style="padding:12px 16px;border-right:1px solid {_C['border']}">
    S&amp;P 500<br>
    <span style="font-family:{_C['mono']};font-size:12px;color:{_C['white']}">{spx.get('price',0):,.0f}</span>
    {_chg(spx.get('change_pct',0))}</td>
<td style="padding:12px 16px;border-right:1px solid {_C['border']}">
    OMX<br>
    <span style="font-family:{_C['mono']};font-size:12px;color:{_C['white']}">{omx.get('price',0):,.0f}</span>
    {_chg(omx.get('change_pct',0))}</td>
<td style="padding:12px 16px;border-right:1px solid {_C['border']}">
    VIX<br>
    <span style="font-family:{_C['mono']};font-size:12px;color:{_C['white']}">{vix_val:.1f}</span></td>
<td style="padding:12px 16px;border-right:1px solid {_C['border']}">
    NASDAQ<br>
    <span style="font-family:{_C['mono']};font-size:12px;color:{_C['white']}">{ndx.get('price',0):,.0f}</span>
    {_chg(ndx.get('change_pct',0))}</td>
<td style="padding:12px 16px;border-right:1px solid {_C['border']}">
    Portfolio<br>
    <span style="font-family:{_C['mono']};font-size:12px;color:{pf_color}">{pf_sign}{pf_pct:.2f}%</span></td>
<td style="padding:12px 16px">
    Breadth<br>
    <span style="font-family:{_C['mono']};font-size:12px;color:{_C['white']}">{breadth}%</span>
    <span style="font-size:9px;color:{_C['muted']}">adv</span></td>
</tr></table></td></tr>""")

    # ── Body start ──
    p.append(f'<tr><td style="padding:8px 36px 0"><table width="100%" cellpadding="0" cellspacing="0" border="0">')

    # ══════════════════════════════════════════════════════════════════════
    # 1. GLOBAL MARKETS — 2-kolumns tabell
    # ══════════════════════════════════════════════════════════════════════
    if indices:
        p.append(_section_title('Global Markets'))
        region_order = ['EUROPE', 'AMERICA', 'ASIA', 'OCEANIA', 'CURRENCIES', 'COMMODITIES', 'YIELDS', 'CRYPTO']
        # Par ihop regioner 2 och 2 för sida-vid-sida
        active_regions = [r for r in region_order if indices.get(r)]
        # Pad till jämnt antal
        if len(active_regions) % 2:
            active_regions.append(None)

        for i in range(0, len(active_regions), 2):
            left_region = active_regions[i]
            right_region = active_regions[i + 1] if i + 1 < len(active_regions) else None

            p.append('<tr><td colspan="10" style="padding:0"><table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>')

            # Vänster kolumn
            p.append(f'<td width="50%" style="vertical-align:top;padding:0 12px 12px 0">')
            if left_region:
                p.append(f'<div style="font-size:9px;font-weight:600;color:{_C["dim"]};letter-spacing:1.5px;'
                         f'text-transform:uppercase;padding:6px 0 4px">{left_region}</div>')
                p.append('<table width="100%" cellpadding="0" cellspacing="0" border="0" style="font-size:11px">')
                for idx in indices[left_region]:
                    chg = idx.get("change_pct", 0.0)
                    price = idx.get("price", 0)
                    # Formatera pris intelligent
                    if price >= 10000:
                        pf_str = f"{price:,.0f}"
                    elif price >= 100:
                        pf_str = f"{price:,.2f}"
                    else:
                        pf_str = f"{price:.4f}" if price < 1 else f"{price:.2f}"
                    p.append(f'<tr>{_td_l(idx["name"])}{_td_r(pf_str)}{_td_r(_chg(chg), mono=False)}</tr>')
                p.append('</table>')
            p.append('</td>')

            # Höger kolumn
            p.append(f'<td width="50%" style="vertical-align:top;padding:0 0 12px 12px">')
            if right_region:
                p.append(f'<div style="font-size:9px;font-weight:600;color:{_C["dim"]};letter-spacing:1.5px;'
                         f'text-transform:uppercase;padding:6px 0 4px">{right_region}</div>')
                p.append('<table width="100%" cellpadding="0" cellspacing="0" border="0" style="font-size:11px">')
                for idx in indices[right_region]:
                    chg = idx.get("change_pct", 0.0)
                    price = idx.get("price", 0)
                    if price >= 10000:
                        pf_str = f"{price:,.0f}"
                    elif price >= 100:
                        pf_str = f"{price:,.2f}"
                    else:
                        pf_str = f"{price:.4f}" if price < 1 else f"{price:.2f}"
                    p.append(f'<tr>{_td_l(idx["name"])}{_td_r(pf_str)}{_td_r(_chg(chg), mono=False)}</tr>')
                p.append('</table>')
            p.append('</td>')

            p.append('</tr></table></td></tr>')

    # ══════════════════════════════════════════════════════════════════════
    # 2. VOLATILITY REGIME
    # ══════════════════════════════════════════════════════════════════════
    vol = data.get("volatility", [])
    if vol:
        p.append(_section_title('Volatility Regime'))
        p.append(f'<tr><td colspan="10" style="padding:8px 0 4px"><table width="100%" cellpadding="0" cellspacing="8" border="0"><tr>')
        for v in vol:
            pct = v.get("percentile", 0)
            val = v.get("value", 0)
            med = v.get("median", 0)
            if pct >= 75:    val_c, border_c = _C['red'], _C['red']
            elif pct >= 50:  val_c, border_c = _C['amber'], _C['amber']
            else:           val_c, border_c = _C['green'], _C['green']

            p.append(f"""<td style="background:{_C['card']};border-left:1px solid {_C['border']};
                border-right:1px solid {_C['border']};border-bottom:1px solid {_C['border']};
                border-top:3px solid {border_c};border-radius:4px;padding:12px 14px;width:25%;vertical-align:top">
                <div style="font-size:9px;font-weight:600;color:{_C['muted']};letter-spacing:1.2px;text-transform:uppercase">
                    {v['name']}</div>
                <div style="font-size:24px;font-weight:700;color:{val_c};margin:4px 0 8px;
                    font-family:{_C['mono']}">{val:.2f}</div>
                <div style="font-size:10px;color:{_C['muted']};margin-bottom:2px">
                    Percentile <span style="color:{_C['text']};font-family:{_C['mono']}">{pct:.0f}</span></div>
                <div style="font-size:10px;color:{_C['muted']}">
                    Median <span style="color:{_C['text']};font-family:{_C['mono']}">{med:.1f}</span></div>
            </td>""")
        p.append('</tr></table></td></tr>')

    # ══════════════════════════════════════════════════════════════════════
    # 3. PORTFOLIO
    # ══════════════════════════════════════════════════════════════════════
    positions = pf.get("positions", [])
    if pf and (pf.get("total_pnl") is not None or positions):
        total_pnl = pf.get("total_pnl", 0)
        total_pct = pf.get("total_pnl_pct", 0)
        total_val = pf.get("total_value", 0)
        n_pos = len(positions)
        pnl_c = _C['green'] if total_pnl >= 0 else _C['red']
        sign = "+" if total_pnl >= 0 else ""

        p.append(_section_title('Portfolio'))

        # KPI row
        p.append(f'<tr><td colspan="10" style="padding:8px 0"><table width="100%" cellpadding="0" cellspacing="8" border="0"><tr>')
        p.append(_kpi_cell('Total P&L', f'{sign}{total_pct:.2f}%', pnl_c))
        p.append(_kpi_cell('Unrealized', f'{sign}{total_pnl:,.0f}', pnl_c))
        p.append(_kpi_cell('Positions', str(n_pos)))
        if total_val > 0:
            p.append(_kpi_cell('Exposure', f'{total_val:,.0f}'))
        p.append('</tr></table></td></tr>')

        # Positions table
        if positions:
            p.append(f"""<tr><td colspan="10" style="padding:0">
            <table width="100%" cellpadding="0" cellspacing="0" border="0" style="font-size:11px">
            <tr>
                <td style="padding:6px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid {_C['border']}">Pair</td>
                <td style="padding:6px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:right;border-bottom:1px solid {_C['border']}">P&L</td>
                <td style="padding:6px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:right;border-bottom:1px solid {_C['border']}">Z-score</td>
                <td style="padding:6px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:right;border-bottom:1px solid {_C['border']}">Side</td>
            </tr>""")
            for pos in positions[:10]:
                z = pos.get("z_score", 0)
                direction = pos.get("direction", "")
                dir_c = _C['green'] if direction == 'LONG' else _C['red'] if direction == 'SHORT' else _C['muted']
                p.append(f"""<tr>
                    <td style="padding:5px 0;border-bottom:1px solid {_C['border_l']};color:{_C['white']};font-weight:500">{pos.get('pair','')}</td>
                    <td style="padding:5px 0;border-bottom:1px solid {_C['border_l']};text-align:right;font-family:{_C['mono']}">{_chg(pos.get('pnl_pct',0))}</td>
                    <td style="padding:5px 0;border-bottom:1px solid {_C['border_l']};text-align:right;font-family:{_C['mono']};color:{_C['text']}">{z:.2f}</td>
                    <td style="padding:5px 0;border-bottom:1px solid {_C['border_l']};text-align:right">
                        <span style="font-size:9px;font-weight:600;color:{dir_c};letter-spacing:0.5px">{direction}</span></td>
                </tr>""")
            p.append('</table></td></tr>')

    # ══════════════════════════════════════════════════════════════════════
    # 4. PAIR SCANNING
    # ══════════════════════════════════════════════════════════════════════
    if scan:
        n_t = scan.get("n_tickers", 0)
        n_p = scan.get("n_pairs", 0)
        n_v = scan.get("n_viable", 0)
        signals = scan.get("signals", [])

        p.append(_section_title('Statistical Arbitrage'))

        # KPI row
        p.append(f'<tr><td colspan="10" style="padding:8px 0"><table width="100%" cellpadding="0" cellspacing="8" border="0"><tr>')
        p.append(_kpi_cell('Tickers', str(n_t)))
        p.append(_kpi_cell('Tested', f'{n_p:,}'))
        p.append(_kpi_cell('Viable', str(n_v), _C['gold']))
        p.append(_kpi_cell('Signals', str(len(signals)), _C['green'] if signals else _C['muted']))
        p.append('</tr></table></td></tr>')

        # Signal cards
        if signals:
            for s in signals:
                z = s.get("z", 0)
                z_c = _C['green'] if z > 0 else _C['red']
                p.append(f"""<tr><td colspan="10" style="padding:0 0 4px">
                <table width="100%" cellpadding="0" cellspacing="0" border="0"
                       style="background:{_C['card']};border:1px solid {_C['border']};border-left:3px solid {_C['gold']};border-radius:4px">
                <tr>
                    <td style="padding:10px 14px;color:{_C['white']};font-weight:600;font-size:12px">{s.get('pair','')}</td>
                    <td style="padding:10px 14px;text-align:right;font-family:{_C['mono']};font-size:11px;color:{_C['muted']}">
                        Z = <span style="color:{z_c}">{z:.2f}</span>
                        &nbsp;&middot;&nbsp; Opt = {s.get('opt_z',2.0):.2f}
                        </td>
                </tr></table></td></tr>""")
        else:
            p.append(f'<tr><td colspan="10" style="text-align:center;color:{_C["muted"]};padding:10px 0;font-size:11px">'
                     f'No active entry signals</td></tr>')

    # ══════════════════════════════════════════════════════════════════════
    # 5. EARNINGS CALENDAR
    # ══════════════════════════════════════════════════════════════════════
    earnings = data.get("earnings", [])
    if earnings:
        p.append(_section_title('Earnings Calendar'))
        p.append(f"""<tr><td colspan="10" style="padding:4px 0">
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="font-size:11px">
        <tr>
            <td style="padding:5px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid {_C['border']}">Company</td>
            <td style="padding:5px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:right;border-bottom:1px solid {_C['border']}">EPS Actual</td>
            <td style="padding:5px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:right;border-bottom:1px solid {_C['border']}">EPS Est.</td>
            <td style="padding:5px 0;color:{_C['dim']};font-size:9px;font-weight:500;letter-spacing:1px;text-transform:uppercase;text-align:right;border-bottom:1px solid {_C['border']}"></td>
        </tr>""")

        for e in earnings:
            eps_a = e.get("eps_actual")
            eps_e = e.get("eps_estimate")
            company = e.get("company", "")
            symbol = e.get("symbol", "")
            region = e.get("region", "").upper()

            # EPS actual — färgkoda beat/miss
            eps_a_str = f'<span style="color:{_C["muted"]}">—</span>'
            if eps_a is not None:
                beat = eps_a >= eps_e if eps_e is not None else True
                c = _C['green'] if beat else _C['red']
                eps_a_str = f'<span style="color:{c};font-family:{_C["mono"]}">{eps_a:.2f}</span>'

            eps_e_str = f'<span style="color:{_C["muted"]}">—</span>'
            if eps_e is not None:
                eps_e_str = f'<span style="font-family:{_C["mono"]};color:{_C["text"]}">{eps_e:.2f}</span>'

            # Symbol badge (visa om finns)
            sym_html = ""
            if symbol:
                sym_html = (f'<span style="font-family:{_C["mono"]};font-size:10px;color:{_C["gold"]};'
                            f'font-weight:600;margin-right:6px">{symbol}</span>')

            # Region badge
            reg_html = (f'<span style="display:inline-block;background:{_C["bg"]};color:{_C["muted"]};'
                        f'font-size:8px;font-weight:600;padding:1px 5px;border-radius:2px;'
                        f'letter-spacing:0.5px">{region}</span>')

            p.append(f"""<tr>
                <td style="padding:4px 0;border-bottom:1px solid {_C['border_l']};color:{_C['text']}">{sym_html}{company}</td>
                <td style="padding:4px 0;border-bottom:1px solid {_C['border_l']};text-align:right">{eps_a_str}</td>
                <td style="padding:4px 0;border-bottom:1px solid {_C['border_l']};text-align:right">{eps_e_str}</td>
                <td style="padding:4px 0;border-bottom:1px solid {_C['border_l']};text-align:right">{reg_html}</td>
            </tr>""")

        p.append('</table></td></tr>')

    elif data.get("_earnings_attempted"):
        p.append(_section_title('Earnings Calendar'))
        p.append(f'<tr><td colspan="10" style="text-align:center;color:{_C["muted"]};padding:12px 0;font-size:11px">'
                 f'No earnings reports scheduled for today</td></tr>')

    # ── Body end ──
    p.append('</table></td></tr>')

    # ── Disclaimer ──
    p.append(f"""<tr><td style="padding:20px 36px 12px;border-top:1px solid {_C['border']}">
<div style="font-size:9px;color:{_C['dim']};line-height:1.6;letter-spacing:0.2px">
This report is generated automatically by proprietary quantitative systems and is intended solely
for internal use. Market data is provided on a best-effort basis and may be delayed.
Past performance does not guarantee future results. Statistical arbitrage signals are model-driven
and should not be construed as investment advice.</div>
</td></tr>""")

    # ── Footer ──
    p.append(f"""<tr><td style="padding:8px 36px 16px">
<table width="100%" cellpadding="0" cellspacing="0" border="0"><tr>
<td style="font-size:9px;color:{_C['dim']};letter-spacing:0.3px">
    <span style="color:{_C['muted']};font-weight:600;letter-spacing:1px">KLIPPINGE INVESTMENT</span>
    <span style="color:{_C['dim']}">&nbsp;&nbsp;|&nbsp;&nbsp;Quantitative Strategies</span>
</td>
<td style="text-align:right;font-size:9px;color:{_C['dim']};letter-spacing:0.3px">
    {timestamp} &nbsp;&middot;&nbsp; Confidential
</td></tr></table></td></tr>""")

    # ── Gold accent bar bottom ──
    p.append(f'<tr><td style="height:2px;background:linear-gradient(90deg,{_C["gold_d"]},{_C["gold"]},{_C["gold_d"]})"></td></tr>')

    p.append('</table></td></tr></table></body></html>')

    return "\n".join(p)


# ── SMTP Sender ──────────────────────────────────────────────────────────────

def send_daily_email(config: dict, html: str, subject: str = None) -> None:
    """Skickar HTML-mail via Gmail SMTP."""
    sender = config.get("email_address", "")
    password = config.get("email_app_password", "")
    recipient = config.get("email_recipient", "")

    if not sender or not password or not recipient:
        raise ValueError("Email configuration incomplete (address, password, or recipient missing)")

    if subject is None:
        subject = f"Klippinge Daily Summary — {datetime.now():%Y-%m-%d}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Klippinge Investment <{sender}>"
    msg["To"] = recipient

    text_part = MIMEText("Daily summary from Klippinge Investment Trading Terminal. "
                         "View this email in an HTML-capable client.", "plain", "utf-8")
    html_part = MIMEText(html, "html", "utf-8")
    msg.attach(text_part)
    msg.attach(html_part)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())


# ── QThread Worker ───────────────────────────────────────────────────────────

class EmailWorker(QObject):
    """Worker thread för att skicka email utan att blockera GUI."""
    finished = Signal()
    success = Signal(str)
    error = Signal(str)

    def __init__(self, config: dict, html: str, subject: str = None):
        super().__init__()
        self.config = config
        self.html = html
        self.subject = subject

    @Slot()
    def run(self):
        try:
            send_daily_email(self.config, self.html, self.subject)
            self.success.emit("Daily summary email sent successfully")
        except smtplib.SMTPAuthenticationError:
            self.error.emit("Email authentication failed — check app password")
        except smtplib.SMTPException as e:
            self.error.emit(f"SMTP error: {e}")
        except Exception as e:
            self.error.emit(f"Email error: {e}")
            traceback.print_exc()
        finally:
            self.finished.emit()

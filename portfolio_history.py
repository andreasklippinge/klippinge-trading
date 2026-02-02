"""
Portfolio History & Benchmark Analysis Module
==============================================

Handles daily portfolio snapshots and benchmark comparison against S&P 500.

Features:
- Automatic daily snapshots at 22:00 (same time as scanner)
- Multi-period performance comparison (1d, 5d, 20d, 60d, 120d, YTD, 200d)
- S&P 500 benchmark comparison
- Discord notification integration

Author: Klippinge Investment
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Benchmark data unavailable.")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default file path for portfolio history (can be overridden)
DEFAULT_HISTORY_FILE = r"G:\Min enhet\Python\Aktieanalys\Python\Trading\portfolio_history.json"

# Benchmark ticker
BENCHMARK_TICKER = "^GSPC"  # S&P 500

# Performance periods in trading days
PERIODS = {
    '1D': 1,
    '5D': 5,
    '20D': 20,      # ~1 month
    '60D': 60,      # ~3 months
    '120D': 120,    # ~6 months
    'YTD': None,    # Calculated dynamically
    '200D': 200,    # ~10 months
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PortfolioSnapshot:
    """Single snapshot of portfolio state."""
    timestamp: str                    # ISO format datetime
    total_value: float                # Total portfolio value in SEK
    total_invested: float             # Total invested capital
    unrealized_pnl: float             # Unrealized P&L in SEK
    unrealized_pnl_pct: float         # Unrealized P&L in %
    n_positions: int                  # Number of open positions
    positions_summary: List[Dict]     # Brief summary of each position
    benchmark_price: Optional[float]  # S&P 500 close price
    notes: Optional[str] = None       # Optional notes for the snapshot
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioSnapshot':
        # Filter out unknown fields to handle forward compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass 
class PerformanceMetrics:
    """Performance metrics for a specific period."""
    period: str
    portfolio_return: float           # Portfolio return %
    benchmark_return: float           # S&P 500 return %
    excess_return: float              # Alpha (portfolio - benchmark)
    portfolio_value_change: float     # Absolute value change SEK
    start_date: str
    end_date: str
    trading_days: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# PORTFOLIO HISTORY MANAGER
# ============================================================================

class PortfolioHistoryManager:
    """
    Manages portfolio history snapshots and benchmark analysis.
    
    Usage:
        manager = PortfolioHistoryManager(history_file_path)
        
        # Take a snapshot (call at 22:00)
        manager.take_snapshot(portfolio_positions, mf_prices)
        
        # Get performance comparison
        performance = manager.get_performance_summary()
    """
    
    def __init__(self, history_file: str = DEFAULT_HISTORY_FILE):
        self.history_file = history_file
        self.snapshots: List[PortfolioSnapshot] = []
        self.benchmark_cache: Optional[pd.Series] = None
        self.benchmark_cache_date: Optional[datetime] = None
        
        self._load_history()
    
    def _load_history(self):
        """Load existing history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.snapshots = [
                    PortfolioSnapshot.from_dict(s) 
                    for s in data.get('snapshots', [])
                ]
                print(f"Loaded {len(self.snapshots)} historical snapshots")
            except Exception as e:
                print(f"Error loading portfolio history: {e}")
                self.snapshots = []
        else:
            self.snapshots = []
    
    def _save_history(self):
        """Save history to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'snapshots': [s.to_dict() for s in self.snapshots]
            }
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"Saved {len(self.snapshots)} snapshots to {self.history_file}")
        except Exception as e:
            print(f"Error saving portfolio history: {e}")
    
    def take_snapshot(self, portfolio: List[Dict], 
                      fetch_benchmark: bool = True) -> Optional[PortfolioSnapshot]:
        """
        Take a snapshot of current portfolio state.
        
        Args:
            portfolio: List of position dictionaries from portfolio_positions.json
            fetch_benchmark: Whether to fetch current S&P 500 price
            
        Returns:
            PortfolioSnapshot if successful, None otherwise
        """
        if not portfolio:
            print("No positions to snapshot")
            return None
        
        try:
            # Calculate portfolio metrics
            total_value = 0.0
            total_invested = 0.0
            positions_summary = []
            
            open_positions = [p for p in portfolio if p.get('status', 'OPEN') == 'OPEN']
            
            for pos in open_positions:
                # Y leg (MINI L = long certificate on stock)
                entry_y = pos.get('mf_entry_price_y', 0.0)
                current_y = pos.get('mf_current_price_y', 0.0)
                qty_y = pos.get('mf_qty_y', 0)
                
                # X leg (MINI S = short certificate on index)
                entry_x = pos.get('mf_entry_price_x', 0.0)
                current_x = pos.get('mf_current_price_x', 0.0)
                qty_x = pos.get('mf_qty_x', 0)
                
                # Invested capital = what we actually PAID for the certificates
                invested_y = entry_y * qty_y if entry_y > 0 and qty_y > 0 else 0
                invested_x = entry_x * qty_x if entry_x > 0 and qty_x > 0 else 0
                pos_invested = invested_y + invested_x
                
                # P&L calculation
                # Both Y and X legs are certificates we OWN, so same formula:
                # P&L = (current_price - entry_price) × quantity
                #
                # Y leg (MINI L): profits when certificate price rises (= stock rises)
                # X leg (MINI S): profits when certificate price rises (= index falls)
                pnl_y = (current_y - entry_y) * qty_y if entry_y > 0 and qty_y > 0 else 0
                pnl_x = (current_x - entry_x) * qty_x if entry_x > 0 and qty_x > 0 else 0
                
                pos_pnl = pnl_y + pnl_x
                pos_current_value = pos_invested + pos_pnl
                
                total_invested += pos_invested
                total_value += pos_current_value
                
                # Position summary
                pos_pnl_pct = (pos_pnl / pos_invested * 100) if pos_invested > 0 else 0
                positions_summary.append({
                    'pair': pos.get('pair', 'Unknown'),
                    'direction': pos.get('direction', 'LONG'),
                    'z_score': pos.get('current_z', 0),
                    'pnl_pct': round(pos_pnl_pct, 2),
                    'capital': round(pos_invested, 2)  # Use actual invested, not mf_total_capital
                })
            
            # Calculate unrealized P&L
            unrealized_pnl = total_value - total_invested
            unrealized_pnl_pct = (unrealized_pnl / total_invested * 100) if total_invested > 0 else 0
            
            # Fetch benchmark price
            benchmark_price = None
            if fetch_benchmark and YFINANCE_AVAILABLE:
                try:
                    ticker = yf.Ticker(BENCHMARK_TICKER)
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        benchmark_price = hist['Close'].iloc[-1]
                except Exception as e:
                    print(f"Error fetching benchmark price: {e}")
            
            # Create snapshot
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now().isoformat(),
                total_value=round(total_value, 2),
                total_invested=round(total_invested, 2),
                unrealized_pnl=round(unrealized_pnl, 2),
                unrealized_pnl_pct=round(unrealized_pnl_pct, 2),
                n_positions=len(open_positions),
                positions_summary=positions_summary,
                benchmark_price=benchmark_price
            )
            
            # Add to history
            self.snapshots.append(snapshot)
            
            # Keep only last 365 snapshots (1 year of daily data)
            if len(self.snapshots) > 365:
                self.snapshots = self.snapshots[-365:]
            
            # Save to file
            self._save_history()
            
            print(f"Snapshot taken: {len(open_positions)} positions, "
                  f"invested={total_invested:.0f} SEK, P&L={unrealized_pnl:+.0f} SEK ({unrealized_pnl_pct:+.2f}%)")
            
            return snapshot
            
        except Exception as e:
            print(f"Error taking snapshot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_benchmark_data(self, lookback_days: int = 250) -> Optional[pd.Series]:
        """Fetch S&P 500 price history."""
        if not YFINANCE_AVAILABLE:
            return None
        
        # Use cache if recent (within same day)
        now = datetime.now()
        if (self.benchmark_cache is not None and 
            self.benchmark_cache_date is not None and
            self.benchmark_cache_date.date() == now.date()):
            return self.benchmark_cache
        
        try:
            ticker = yf.Ticker(BENCHMARK_TICKER)
            end_date = now
            start_date = end_date - timedelta(days=lookback_days + 30)  # Buffer
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            prices = hist['Close']
            prices.index = prices.index.tz_localize(None)  # Remove timezone
            
            # Cache result
            self.benchmark_cache = prices
            self.benchmark_cache_date = now
            
            return prices
            
        except Exception as e:
            print(f"Error fetching benchmark data: {e}")
            return None
    
    def _get_ytd_days(self) -> int:
        """Calculate trading days since start of year."""
        now = datetime.now()
        year_start = datetime(now.year, 1, 1)
        calendar_days = (now - year_start).days
        # Approximately 252 trading days per year
        return max(1, int(calendar_days * 252 / 365))
    
    def get_performance_summary(self) -> Dict[str, PerformanceMetrics]:
        """
        Calculate performance for all periods vs S&P 500.
        
        Returns:
            Dict mapping period name to PerformanceMetrics
        """
        if len(self.snapshots) < 2:
            return {}
        
        # Get benchmark data
        benchmark_prices = self._get_benchmark_data(lookback_days=250)
        
        # Sort snapshots by timestamp
        sorted_snapshots = sorted(self.snapshots, key=lambda s: s.timestamp)
        
        # Build portfolio value series
        portfolio_dates = []
        portfolio_values = []
        
        for snapshot in sorted_snapshots:
            try:
                dt = datetime.fromisoformat(snapshot.timestamp)
                portfolio_dates.append(pd.Timestamp(dt.date()))
                portfolio_values.append(snapshot.total_value)
            except:
                continue
        
        if len(portfolio_values) < 2:
            return {}
        
        portfolio_series = pd.Series(portfolio_values, index=portfolio_dates)
        # Keep only last value per day (in case of multiple snapshots)
        portfolio_series = portfolio_series.groupby(portfolio_series.index).last()
        
        # Calculate performance for each period
        results = {}
        
        for period_name, days in PERIODS.items():
            # Handle YTD specially
            if days is None:
                days = self._get_ytd_days()
            
            metrics = self._calculate_period_performance(
                portfolio_series, 
                benchmark_prices,
                days,
                period_name
            )
            
            if metrics:
                results[period_name] = metrics
        
        return results
    
    def _calculate_period_performance(
        self, 
        portfolio_series: pd.Series,
        benchmark_prices: Optional[pd.Series],
        days: int,
        period_name: str
    ) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics for a specific period."""
        
        if len(portfolio_series) < 2:
            return None
        
        # Get end date (most recent)
        end_date = portfolio_series.index[-1]
        
        # Find start value (N days ago or earliest available)
        target_start = end_date - pd.Timedelta(days=days)
        
        # Get closest available start value
        earlier_dates = portfolio_series.index[portfolio_series.index <= target_start]
        
        if len(earlier_dates) > 0:
            start_date = earlier_dates[-1]
        else:
            # Use earliest available
            start_date = portfolio_series.index[0]
        
        start_value = portfolio_series.loc[start_date]
        end_value = portfolio_series.loc[end_date]
        
        # Portfolio return
        if start_value > 0:
            portfolio_return = ((end_value / start_value) - 1) * 100
        else:
            portfolio_return = 0.0
        
        # Benchmark return
        benchmark_return = 0.0
        if benchmark_prices is not None and len(benchmark_prices) > 0:
            # Find matching dates
            bench_end = benchmark_prices.index[benchmark_prices.index <= end_date]
            bench_start = benchmark_prices.index[benchmark_prices.index <= start_date]
            
            if len(bench_end) > 0 and len(bench_start) > 0:
                bench_end_price = benchmark_prices.loc[bench_end[-1]]
                bench_start_price = benchmark_prices.loc[bench_start[-1]]
                
                if bench_start_price > 0:
                    benchmark_return = ((bench_end_price / bench_start_price) - 1) * 100
        
        # Calculate actual trading days
        actual_days = len(portfolio_series.loc[start_date:end_date])
        
        return PerformanceMetrics(
            period=period_name,
            portfolio_return=round(portfolio_return, 2),
            benchmark_return=round(benchmark_return, 2),
            excess_return=round(portfolio_return - benchmark_return, 2),
            portfolio_value_change=round(end_value - start_value, 0),
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            trading_days=actual_days
        )
    
    def get_daily_change(self) -> Optional[Dict]:
        """Get today's portfolio change (for Discord notification)."""
        if len(self.snapshots) < 2:
            return None
        
        # Sort by timestamp
        sorted_snapshots = sorted(self.snapshots, key=lambda s: s.timestamp, reverse=True)
        
        today = sorted_snapshots[0]
        yesterday = sorted_snapshots[1]
        
        if today.total_value > 0 and yesterday.total_value > 0:
            daily_change_pct = ((today.total_value / yesterday.total_value) - 1) * 100
            daily_change_sek = today.total_value - yesterday.total_value
        else:
            daily_change_pct = 0
            daily_change_sek = 0
        
        # Get benchmark daily change
        benchmark_daily = 0.0
        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(BENCHMARK_TICKER)
                hist = ticker.history(period='5d')
                if len(hist) >= 2:
                    benchmark_daily = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100
            except:
                pass
        
        return {
            'portfolio_change_pct': round(daily_change_pct, 2),
            'portfolio_change_sek': round(daily_change_sek, 0),
            'benchmark_change_pct': round(benchmark_daily, 2),
            'excess_daily': round(daily_change_pct - benchmark_daily, 2),
            'total_value': today.total_value,
            'total_pnl_pct': today.unrealized_pnl_pct,
            'n_positions': today.n_positions,
            'positions': today.positions_summary
        }
    
    def get_discord_portfolio_fields(self) -> List[Dict]:
        """
        Generate Discord embed fields for portfolio status.
        
        Returns:
            List of Discord embed field dictionaries
        """
        fields = []
        
        # Get daily change
        daily = self.get_daily_change()
        if daily:
            # Portfolio status
            pnl_emoji = "🟢" if daily['total_pnl_pct'] >= 0 else "🔴"
            day_emoji = "📈" if daily['portfolio_change_pct'] >= 0 else "📉"
            
            status_text = (
                f"{pnl_emoji} **Total P&L:** {daily['total_pnl_pct']:+.2f}%\n"
                f"{day_emoji} **Today:** {daily['portfolio_change_pct']:+.2f}% "
                f"({daily['portfolio_change_sek']:+,.0f} SEK)\n"
                f"💼 **Positions:** {daily['n_positions']}\n"
                f"💰 **Value:** {daily['total_value']:,.0f} SEK"
            )
            
            fields.append({
                "name": "📊 Portfolio Status",
                "value": status_text,
                "inline": False
            })
            
            # Benchmark comparison
            alpha_emoji = "✅" if daily['excess_daily'] >= 0 else "⚠️"
            bench_text = (
                f"S&P 500: {daily['benchmark_change_pct']:+.2f}%\n"
                f"{alpha_emoji} Alpha: {daily['excess_daily']:+.2f}%"
            )
            
            fields.append({
                "name": "📈 vs Benchmark",
                "value": bench_text,
                "inline": True
            })
        
        # Get multi-period performance
        performance = self.get_performance_summary()
        if performance:
            perf_lines = []
            for period in ['1D', '5D', '20D', 'YTD']:
                if period in performance:
                    p = performance[period]
                    emoji = "🟢" if p.excess_return >= 0 else "🔴"
                    perf_lines.append(
                        f"**{period}:** {p.portfolio_return:+.1f}% "
                        f"({emoji} {p.excess_return:+.1f}% α)"
                    )
            
            if perf_lines:
                fields.append({
                    "name": "📊 Performance",
                    "value": "\n".join(perf_lines),
                    "inline": True
                })
        
        # Position details (brief)
        if daily and daily['positions']:
            pos_lines = []
            for pos in daily['positions'][:3]:  # Top 3 only
                z_emoji = "🟢" if pos['z_score'] < 0 else "🔴"
                pnl_emoji = "+" if pos['pnl_pct'] >= 0 else ""
                pos_lines.append(
                    f"{pos['pair']}: {pnl_emoji}{pos['pnl_pct']:.1f}% "
                    f"(Z: {pos['z_score']:.2f})"
                )
            
            if pos_lines:
                fields.append({
                    "name": "📋 Open Positions",
                    "value": "\n".join(pos_lines),
                    "inline": False
                })
        
        return fields
    
    def get_latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Get the most recent snapshot."""
        if not self.snapshots:
            return None
        return sorted(self.snapshots, key=lambda s: s.timestamp)[-1]
    
    def get_snapshot_count(self) -> int:
        """Get number of snapshots in history."""
        return len(self.snapshots)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_performance_table(performance: Dict[str, PerformanceMetrics]) -> str:
    """Format performance metrics as a text table."""
    if not performance:
        return "No performance data available"
    
    lines = [
        "Period    | Portfolio | Benchmark |   Alpha   | Days",
        "----------|-----------|-----------|-----------|-----"
    ]
    
    for period_name in ['1D', '5D', '20D', '60D', '120D', 'YTD', '200D']:
        if period_name not in performance:
            continue
        
        p = performance[period_name]
        alpha_sign = '+' if p.excess_return >= 0 else ''
        
        lines.append(
            f"{period_name:9} | {p.portfolio_return:+8.2f}% | "
            f"{p.benchmark_return:+8.2f}% | {alpha_sign}{p.excess_return:8.2f}% | {p.trading_days:4}"
        )
    
    return "\n".join(lines)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    print("Testing PortfolioHistoryManager...")
    
    # Create manager with temp file
    import tempfile
    temp_file = os.path.join(tempfile.gettempdir(), "test_portfolio_history.json")
    manager = PortfolioHistoryManager(temp_file)
    
    # Sample portfolio data
    sample_portfolio = [
        {
            'pair': 'AXP/^DJI',
            'direction': 'LONG',
            'status': 'OPEN',
            'current_z': -1.42,
            'mf_entry_price_y': 18.47,
            'mf_current_price_y': 18.88,
            'mf_qty_y': 209,
            'mf_entry_price_x': 69.99,
            'mf_current_price_x': 71.52,
            'mf_qty_x': 30,
            'mf_total_capital': 3000
        }
    ]
    
    # Take snapshot
    snapshot = manager.take_snapshot(sample_portfolio)
    
    if snapshot:
        print(f"\nSnapshot taken:")
        print(f"  Value: {snapshot.total_value} SEK")
        print(f"  P&L: {snapshot.unrealized_pnl_pct:+.2f}%")
        print(f"  Benchmark: {snapshot.benchmark_price}")
    
    # Get Discord fields
    fields = manager.get_discord_portfolio_fields()
    print(f"\nDiscord fields: {len(fields)}")
    for f in fields:
        print(f"  - {f['name']}")
    
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print("\nTest complete!")
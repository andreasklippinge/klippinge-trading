"""
Dynamisk Skärmskalning för Dashboard
=====================================

Denna modul detekterar skärmstorlek och justerar typografi/layout automatiskt.

INSTALLATION:
1. Importera i din dashboard_PyQt5.py högst upp:
   from screen_scaling import get_scaled_typography, get_scale_factor, apply_screen_scaling

2. Ersätt din TYPOGRAPHY-definition med:
   TYPOGRAPHY = get_scaled_typography()

3. I main(), före QApplication skapas, lägg till:
   apply_screen_scaling()

"""

import sys
from typing import Dict

# ============================================================================
# SCREEN SIZE DETECTION
# ============================================================================

def get_screen_info() -> Dict:
    """Get primary screen information.
    
    Returns:
        Dict with 'width', 'height', 'dpi', 'diagonal_inches'
    """
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QGuiApplication
    
    # Skapa temporär app om ingen finns (för att kunna läsa skärminfo)
    app = QApplication.instance()
    temp_app = None
    if app is None:
        temp_app = QApplication(sys.argv)
        app = temp_app
    
    try:
        screen = QGuiApplication.primaryScreen()
        geometry = screen.geometry()
        
        width = geometry.width()
        height = geometry.height()
        
        # Fysisk DPI (kan vara opålitligt på vissa system)
        physical_dpi = screen.physicalDotsPerInch()
        logical_dpi = screen.logicalDotsPerInch()
        
        # Beräkna ungefärlig diagonal i tum (antar 96 DPI som standard)
        # Detta är en approximation
        diagonal_pixels = (width**2 + height**2)**0.5
        diagonal_inches = diagonal_pixels / physical_dpi if physical_dpi > 0 else diagonal_pixels / 96
        
        return {
            'width': width,
            'height': height,
            'physical_dpi': physical_dpi,
            'logical_dpi': logical_dpi,
            'diagonal_inches': diagonal_inches,
            'device_pixel_ratio': screen.devicePixelRatio()
        }
    finally:
        if temp_app:
            temp_app.quit()


def get_scale_factor() -> float:
    """Calculate a scale factor based on screen size.
    
    Returns a multiplier:
    - 0.85 for small screens (laptops ~13-14")
    - 1.0 for medium screens (~24")
    - 1.15 for large screens (27"+)
    - 1.25 for very large screens (32"+)
    """
    try:
        info = get_screen_info()
        width = info['width']
        height = info['height']
        
        # Använd bredd som primär indikator
        # Vanliga upplösningar:
        # - 1366x768, 1920x1080: Laptop (13-15")
        # - 2560x1440: Medium-stor monitor (24-27")
        # - 3840x2160 (4K): Stor monitor (27-32"+)
        
        if width <= 1440:
            # Liten skärm (laptop)
            return 0.85
        elif width <= 1920:
            # HD/Full HD
            if height <= 1080:
                return 0.95  # Standard 1080p
            else:
                return 1.0
        elif width <= 2560:
            # QHD / 1440p
            return 1.05
        elif width <= 3440:
            # Ultrawide eller 4K
            return 1.15
        else:
            # 4K eller större
            return 1.25
            
    except Exception as e:
        print(f"[ScreenScaling] Error detecting screen: {e}")
        return 1.0


def get_screen_category() -> str:
    """Get a human-readable screen category.
    
    Returns:
        'small', 'medium', 'large', or 'xlarge'
    """
    factor = get_scale_factor()
    if factor < 0.9:
        return 'small'
    elif factor < 1.05:
        return 'medium'
    elif factor < 1.2:
        return 'large'
    else:
        return 'xlarge'


# ============================================================================
# SCALED TYPOGRAPHY
# ============================================================================

# Bas-typografi (för medium skärm ~1920x1080)
BASE_TYPOGRAPHY = {
    # Headers
    'header_large': 20,        # Main title (KLIPPINGE INVESTMENT)
    'header_section': 13,      # Section headers
    'header_sub': 12,          # Sub-headers within sections
    
    # Body text
    'body_large': 13,          # Primary body text
    'body_medium': 12,         # Standard text
    'body_small': 11,          # Secondary/muted text
    
    # Data display
    'metric_value': 22,        # Large metric values (Z-score, prices)
    'metric_label': 12,        # Metric labels
    'table_header': 12,        # Table column headers
    'table_cell': 13,          # Table cell content
    
    # UI elements
    'button': 12,              # Button text
    'input': 12,               # Input field text
    'tab': 13,                 # Tab labels
    'status': 11,              # Status indicators, timestamps
    'clock_time': 18,          # World clock time
    'clock_city': 12,          # World clock city names
}


def get_scaled_typography(scale_factor: float = None) -> Dict[str, int]:
    """Get typography dictionary scaled for current screen.
    
    Args:
        scale_factor: Optional manual scale factor. If None, auto-detects.
    
    Returns:
        Dictionary of font sizes scaled appropriately.
    """
    if scale_factor is None:
        scale_factor = get_scale_factor()
    
    scaled = {}
    for key, base_size in BASE_TYPOGRAPHY.items():
        # Skala och runda till närmaste heltal
        scaled[key] = max(8, round(base_size * scale_factor))
    
    # Scale factor applied
    return scaled


# ============================================================================
# LAYOUT RECOMMENDATIONS
# ============================================================================

def get_layout_recommendations() -> Dict:
    """Get layout recommendations based on screen size.
    
    Returns dict with:
    - 'news_feed_width': Recommended width for news feed
    - 'chart_height': Recommended chart height
    - 'table_row_height': Recommended table row height
    - 'card_min_height': Minimum height for metric cards
    - 'spacing': General spacing multiplier
    """
    factor = get_scale_factor()
    category = get_screen_category()
    
    recommendations = {
        'small': {
            'news_feed_width': (250, 320),     # min, max
            'chart_height': 200,
            'table_row_height': 28,
            'card_min_height': 100,
            'card_max_width': 180,
            'spacing': 0.8,
            'show_mode_in_vol_card': False,    # Dölj mode för att spara plats
        },
        'medium': {
            'news_feed_width': (300, 380),
            'chart_height': 250,
            'table_row_height': 32,
            'card_min_height': 125,
            'card_max_width': 200,
            'spacing': 1.0,
            'show_mode_in_vol_card': True,
        },
        'large': {
            'news_feed_width': (350, 450),
            'chart_height': 300,
            'table_row_height': 36,
            'card_min_height': 140,
            'card_max_width': 240,
            'spacing': 1.1,
            'show_mode_in_vol_card': True,
        },
        'xlarge': {
            'news_feed_width': (400, 500),
            'chart_height': 350,
            'table_row_height': 40,
            'card_min_height': 160,
            'card_max_width': 280,
            'spacing': 1.2,
            'show_mode_in_vol_card': True,
        }
    }
    
    return recommendations.get(category, recommendations['medium'])


# ============================================================================
# APPLICATION SETUP
# ============================================================================

def apply_screen_scaling():
    """Apply screen scaling attributes to QApplication.
    
    Call this BEFORE creating QApplication.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    
    # Aktivera High DPI skalning
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # På Windows, försök också sätta DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except:
        pass  # Inte på Windows eller funkar inte
    
    # High DPI scaling enabled


def get_scaled_stylesheet(base_stylesheet: str, scale_factor: float = None) -> str:
    """Scale numeric values in a stylesheet.
    
    This is a simple approach that scales pixel values.
    For more complex styling, consider using a proper CSS preprocessor.
    
    Args:
        base_stylesheet: The original stylesheet string
        scale_factor: Scale factor (auto-detected if None)
    
    Returns:
        Scaled stylesheet string
    """
    import re
    
    if scale_factor is None:
        scale_factor = get_scale_factor()
    
    def scale_px(match):
        value = int(match.group(1))
        scaled = max(1, round(value * scale_factor))
        return f"{scaled}px"
    
    # Skala alla px-värden (men inte för font-size som hanteras separat via TYPOGRAPHY)
    # Matcha t.ex. "padding: 10px" men undvik font-size
    # Detta är en förenklad approach
    scaled = re.sub(r'(\d+)px', scale_px, base_stylesheet)
    
    return scaled


# ============================================================================
# DEBUG / INFO
# ============================================================================

def print_screen_info():
    """Print screen information for debugging."""
    try:
        info = get_screen_info()
        factor = get_scale_factor()
        category = get_screen_category()
        
        print("=" * 50)
        print("SCREEN SCALING INFO")
        print("=" * 50)
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"Physical DPI: {info['physical_dpi']:.1f}")
        print(f"Logical DPI: {info['logical_dpi']:.1f}")
        print(f"Device Pixel Ratio: {info['device_pixel_ratio']:.2f}")
        print(f"Estimated diagonal: {info['diagonal_inches']:.1f}\"")
        print(f"Category: {category}")
        print(f"Scale factor: {factor:.2f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error getting screen info: {e}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Test screen scaling detection."""
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    print_screen_info()
    
    print("\nScaled Typography:")
    typography = get_scaled_typography()
    for key, value in typography.items():
        base = BASE_TYPOGRAPHY[key]
        print(f"  {key}: {base} -> {value}")
    
    print("\nLayout Recommendations:")
    recs = get_layout_recommendations()
    for key, value in recs.items():
        print(f"  {key}: {value}")
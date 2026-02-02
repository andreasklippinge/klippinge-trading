"""
Morgan Stanley Mini Futures Scraper
Bygger på befintlig scraping-logik med requests + BeautifulSoup
"""
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

# Cache
_minifutures_cache: pd.DataFrame = None
_minifutures_cache_time: datetime = None
_price_cache: dict = {}
_price_cache_time: datetime = None

SCRAPING_AVAILABLE = True


@dataclass
class ProductQuote:
    """Prisinfo för en Mini Future"""
    isin: str
    name: str
    buy_price: Optional[float]  # Köp (bid)
    sell_price: Optional[float]  # Sälj (ask)
    spread: Optional[float]
    currency: str = "SEK"
    timestamp: Optional[datetime] = None


def create_session() -> requests.Session:
    """Skapa en session med lämpliga headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
    })
    return session


def parse_ms_price(x: str) -> Optional[float]:
    """
    Parsa Morgan Stanley prisformat.
    
    Format:
        18,490      -> 18.490
        22 213,240  -> 22213.240
    
    Regel: mellanslag = tusentalsavgränsare, komma = decimal
    """
    if not isinstance(x, str):
        return None
    
    # Ta bort mellanslag (tusentalsavgränsare)
    clean = x.replace(" ", "").replace("\xa0", "")  # \xa0 = non-breaking space
    
    # Byt komma mot punkt (decimal)
    clean = clean.replace(",", ".")
    
    # Ta bort allt som inte är siffror eller punkt
    clean = re.sub(r"[^\d\.]", "", clean)
    
    try:
        return float(clean)
    except ValueError:
        return None


def fetch_product_detail(session: requests.Session, isin: str) -> Optional[ProductQuote]:
    """
    Hämta köp/sälj-pris för en specifik produkt.
    """
    url = f"https://etp.morganstanley.com/se/sv/product-details/{isin.lower()}"
    
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        buy_price = None
        sell_price = None
        name = None
        
        # Försök hitta produktnamn
        h1 = soup.find("h1")
        if h1:
            name = h1.get_text(strip=True)
        
        # Sök efter element med "KÖP" eller "SÄLJ"
        for elem in soup.find_all(["div", "span", "td", "button"]):
            text = elem.get_text(strip=True)
            
            # Leta efter KÖP-pris (exkludera "Utv. idag (KÖP)")
            if "KÖP" in text.upper() and "UTV" not in text.upper() and "IDAG" not in text.upper():
                # Matcha pris: siffror, mellanslag, komma följt av SEK
                price_match = re.search(r'([\d\s]+,[\d]+)\s*SEK', text)
                if price_match:
                    parsed = parse_ms_price(price_match.group(1))
                    if parsed:
                        buy_price = parsed
            
            # Leta efter SÄLJ-pris
            if "SÄLJ" in text.upper():
                price_match = re.search(r'([\d\s]+,[\d]+)\s*SEK', text)
                if price_match:
                    parsed = parse_ms_price(price_match.group(1))
                    if parsed:
                        sell_price = parsed
       
        if buy_price is not None or sell_price is not None:
            spread = None
            if buy_price and sell_price:
                spread = (sell_price / buy_price) - 1
            
            return ProductQuote(
                isin=isin.upper(),
                name=name or isin.upper(),
                buy_price=buy_price,
                sell_price=sell_price,
                spread=spread,
                timestamp=datetime.now()
            )
        
        return None
        
    except Exception as e:
        print(f"Fel vid hämtning av {isin}: {e}")
        return None


def get_buy_price(isin: str) -> Optional[float]:
    """Hämta köppris för en specifik ISIN."""
    session = create_session()
    quote = fetch_product_detail(session, isin)
    return quote.buy_price if quote else None


def get_quotes_batch(isins: list[str]) -> dict[str, ProductQuote]:
    """Hämta priser för flera ISINs parallellt."""
    session = create_session()
    results = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_isin = {
            executor.submit(fetch_product_detail, session, isin): isin 
            for isin in isins
        }
        
        for future in as_completed(future_to_isin):
            isin = future_to_isin[future]
            try:
                quote = future.result()
                if quote:
                    results[isin.upper()] = quote
            except Exception as e:
                print(f"Fel för {isin}: {e}")
    
    return results


# === Test ===
if __name__ == "__main__":
    print("Testar Morgan Stanley scraper...")
    print("=" * 50)
    
    # Test: Hämta köppris
    isin = "GB00BNV54681"
    print(f"\nHämtar pris för {isin}...")
    
    session = create_session()
    quote = fetch_product_detail(session, isin)
    
    if quote:
        print(f"  Namn: {quote.name}")
        print(f"  Köp: {quote.buy_price:,.3f} SEK" if quote.buy_price else "  Köp: N/A")
        print(f"  Sälj: {quote.sell_price:,.3f} SEK" if quote.sell_price else "  Sälj: N/A")
        print(f"  Spread: {quote.spread:.3%}" if quote.spread else "  Spread: N/A")
    else:
        print("  Kunde inte hämta pris")
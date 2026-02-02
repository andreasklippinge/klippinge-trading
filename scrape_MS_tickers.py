from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import pandas as pd

# Bas-URL (utan sidnummer)
# base_url = "https://etp.morganstanley.com/se/sv/produkter?f_pc=LeverageProducts&f_pt=MiniFuture&p_s=10"

base_url = "https://etp.morganstanley.com/se/sv/produkter?f_pc=LeverageProducts&f_pt=ConstantLeverage&f_factor_max=3&p_s=100"

# Konfigurera Selenium
options = Options()
options.headless = True  # Kör utan att öppna webbläsarfönster
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
driver = webdriver.Chrome(options=options)  # Kräver ChromeDriver installerat

try:
    # Lista för att lagra alla underliggande tillgångar
    all_underlying_assets = []

    # Loop genom alla 330 sidor
    for page_num in range(1, 21):  # 1 till 330
        url = f"{base_url}&p_n={page_num}"
        print(f"Hämtar data från sida {page_num} vid {time.strftime('%H:%M:%S')}...")

        # Öppna webbplatsen
        driver.get(url)

        # Vänta på att tabellen laddas (15 sekunder timeout)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except Exception as e:
            print(f"Timeout eller fel på sida {page_num}: {e}. Hoppar över denna sida och fortsätter...")
            time.sleep(5)  # Extra paus vid fel
            continue

        # Hämta sidans HTML
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Hitta tabellen
        table = soup.find("table")

        if table:
            # Hitta rubrikraden och identifiera index för kolumnen "Underliggande tillgång"
            headers = table.find("thead").find_all("th")
            header_texts = [header.text.strip().lower() for header in headers]
            underlying_col_index = None
            for i, header in enumerate(header_texts):
                if "underliggande" in header.lower() or "underlying" in header.lower():
                    underlying_col_index = i
                    print(f"Hittade kolumnen 'Underliggande tillgång' vid index {underlying_col_index} med rubrik: {header}")
                    break

            if underlying_col_index is not None:
                # Iterera över rader i tabellen (tbody)
                rows = table.find("tbody").find_all("tr")
                page_assets = []
                for row in rows:
                    columns = row.find_all("td")
                    if columns and len(columns) > underlying_col_index:
                        asset = columns[underlying_col_index].text.strip()
                        if asset:
                            page_assets.append(asset)

                # Lägg till unika tillgångar från denna sida
                all_underlying_assets.extend(page_assets)
                print(f"Hittade {len(page_assets)} underliggande tillgångar på sida {page_num}.")
            else:
                print(f"Kunde inte hitta kolumnen 'Underliggande tillgång' på sida {page_num}. Hoppar över...")
        else:
            print(f"Kunde inte hitta tabellen på sida {page_num}. Hoppar över...")

        # Fördröjning mellan sidladdningar för att undvika överbelastning
        time.sleep(5)

    # Ta bort dubbletter och sortera
    unique_assets = sorted(set(all_underlying_assets))
    if unique_assets:
        print(f"\nUnika underliggande tillgångar från {len(set(all_underlying_assets))} sidor:")
        for asset in unique_assets:
            print(f"- {asset}")

        # Spara till CSV
        df = pd.DataFrame(unique_assets, columns=["Underliggande Tillgång"])
        df.to_csv("underlying_assets_all.csv", index=False, encoding="utf-8")
        print("\nData sparad till 'underlying_assets_all.csv'")
    else:
        print("\nInga underliggande tillgångar hittades.")

except Exception as e:
    print(f"Ett fel uppstod: {e}")

finally:
    # Stäng webbläsaren
    driver.quit()

# Extra fördröjning för att säkerställa att allt är färdigt
time.sleep(1)
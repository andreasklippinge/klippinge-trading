# 🚀 Komplett Guide: Från Python-kod till distribuerbar Windows-programvara

## Klippinge Investment Trading Terminal — Byggguide

---

## Innehåll

1. [Översikt — Vad vi bygger](#1-översikt)
2. [Förberedelser — Installera verktyg](#2-förberedelser)
3. [Steg 1 — Sätt upp projektmappen](#3-steg-1-sätt-upp-projektmappen)
4. [Steg 2 — Testa att allt fungerar i utvecklingsläge](#4-steg-2-testa-i-utvecklingsläge)
5. [Steg 3 — Bygg .exe med PyInstaller](#5-steg-3-bygg-exe)
6. [Steg 4 — Testa den byggda .exe-filen](#6-steg-4-testa-exe)
7. [Steg 5 — Bygg en Windows-installer (valfritt)](#7-steg-5-installer)
8. [Steg 6 — Skapa ett GitHub-repo](#8-steg-6-github)
9. [Steg 7 — Automatiska byggen med GitHub Actions](#9-steg-7-github-actions)
10. [Steg 8 — Släpp din första version](#10-steg-8-release)
11. [Steg 9 — Skapa en nedladdningssida](#11-steg-9-hemsida)
12. [Felsökning](#12-felsökning)
13. [Checklista](#13-checklista)

---

## 1. Översikt

### Vad händer under huven?

```
DIN KOD (Python)                    PYINSTALLER                      ANVÄNDARE
┌─────────────────┐                ┌──────────────┐                 ┌───────────────────┐
│ dashboard_PyQt5  │                │ Analyserar   │                 │                   │
│ pairs_engine     │──────────────►│ alla imports  │────────────►   │ KlippingeTrading  │
│ regime_hmm       │  pyinstaller   │ buntar Python │  .exe + DLLer  │     .exe          │
│ app_config       │                │ + alla libs   │                 │                   │
│ auto_updater     │                │ till en mapp  │                 │ Dubbelklicka      │
└─────────────────┘                └──────────────┘                 │ och kör!          │
                                                                     └───────────────────┘
```

PyInstaller paketerar:
- Hela Python-tolken (python312.dll)
- Alla bibliotek (numpy, scipy, PyQt5, etc.)
- Dina .py-filer (kompilerade till .pyc)
- Datafiler (CSV, JSON, ICO)

Resultatet är en mapp (ca 250-400 MB) som kan köras på vilken Windows 10/11-dator som helst **utan att Python behöver vara installerat**.

### Vad har ändrats i din kod?

Jag har redan gjort dessa ändringar i den modifierade `dashboard_PyQt5.py` som medföljer:

| Ändring | Varför |
|---------|--------|
| `app_config.py` importeras | Alla sökvägar blir portabla |
| `G:\Min enhet\...` → `Paths.xxx()` | Fungerar på vilken dator som helst |
| Discord webhook → config-fil | Säkerhet — URL:en var synlig i koden |
| `logo.ico` → `Paths.logo_icon()` | Hittar ikonen oavsett installationsplats |
| Cache-filer → `%APPDATA%` | Varje användare får sin egen data |
| `main()` uppdaterad | Initialisering + auto-uppdatering |

---

## 2. Förberedelser

### 2.1 Kontrollera Python-version

Öppna en terminal (CMD eller PowerShell):

```powershell
python --version
```

Du behöver **Python 3.10 eller nyare**. Om du har 3.12 är det perfekt.

### 2.2 Installera PyInstaller

```powershell
pip install pyinstaller
```

Verifiera:

```powershell
pyinstaller --version
```

Du bör se något som `6.x.x`.

### 2.3 Installera alla beroenden

Se till att alla libs som din app behöver finns installerade:

```powershell
pip install PyQt5 pyqtgraph numpy pandas scipy statsmodels yfinance requests beautifulsoup4 openpyxl
```

### 2.4 (Valfritt) Installera Inno Setup

Om du vill skapa en riktig Windows-installer (med "Nästa → Nästa → Installera"-dialog):

1. Gå till https://jrsoftware.org/isinfo.php
2. Ladda ner Inno Setup 6
3. Installera med standardinställningar

### 2.5 (Valfritt) Installera Git

Om du vill använda GitHub för automatiska byggen och uppdateringar:

```powershell
# Kontrollera om git redan finns:
git --version

# Om inte installerat, ladda ner från https://git-scm.com/download/win
```

---

## 3. Steg 1 — Sätt upp projektmappen

### 3.1 Skapa en ren projektmapp

Skapa en ny mapp där allt ska ligga. Till exempel:

```
C:\Dev\KlippingeTrading\
```

### 3.2 Kopiera filerna

Kopiera **alla dessa filer** till projektmappen. Filerna du fått från mig (nya):

```
C:\Dev\KlippingeTrading\
├── app_config.py                          ← NY: Portabla sökvägar
├── auto_updater.py                        ← NY: Auto-uppdatering
├── build.py                               ← NY: Byggskript
├── requirements.txt                       ← NY: Beroenden
├── build\
│   ├── klippinge.spec                     ← NY: PyInstaller-konfiguration
│   └── installer.iss                      ← NY: Inno Setup-skript
└── .github\
    └── workflows\
        └── build-release.yml              ← NY: GitHub Actions
```

Plus dina befintliga filer (men använd den **modifierade** `dashboard_PyQt5.py`):

```
C:\Dev\KlippingeTrading\
├── dashboard_PyQt5.py                     ← MODIFIERAD VERSION
├── pairs_engine.py                        ← Oförändrad
├── regime_hmm.py                          ← Oförändrad
├── portfolio_history.py                   ← Oförändrad
├── scrape_prices_MS.py                    ← Oförändrad
├── scrape_MS_tickers.py                   ← Oförändrad
├── logo.ico                               ← Oförändrad
├── index_tickers.csv                      ← Oförändrad
├── underliggande_matchade_tickers.csv     ← Oförändrad
├── notification_config.json               ← Oförändrad
├── ib_ticker_mapping.json                 ← Oförändrad
├── news_cache.json                        ← Oförändrad
├── assets\
│   └── styles.css                         ← Oförändrad
└── Trading\
    ├── index_tickers.csv                  ← Oförändrad
    ├── portfolio_positions.json           ← Oförändrad
    ├── portfolio_history.json             ← Oförändrad
    ├── benchmark_cache.json               ← Oförändrad
    └── engine_cache.pkl                   ← Oförändrad
```

### 3.3 Konfigurera app_config.py

Öppna `app_config.py` och ändra **rad 17**:

```python
# FÖR:
GITHUB_REPO = "YOUR_GITHUB_USERNAME/klippinge-trading"

# EFTER (exempel):
GITHUB_REPO = "andreas-klippinge/trading-terminal"
```

Detta används av auto-uppdateraren för att hitta nya versioner.

### 3.4 Sätt upp Discord webhook (säkert)

Istället för att ha webhook-URL:en i koden, skapa/uppdatera filen `notification_config.json`:

```json
{
  "discord_webhook_url": "https://discord.com/api/webhooks/DIN_WEBHOOK_HÄR",
  "notifications_enabled": true
}
```

> ⚠️ **VIKTIGT:** Gå in i Discord-serverns inställningar och **rotera (regenerera) din webhook** eftersom den gamla URL:en har funnits i klartext i koden. Gamla URL:en bör betraktas som komprometterad.

---

## 4. Steg 2 — Testa i utvecklingsläge

Innan du bygger .exe, verifiera att allt fungerar med den nya konfigurationen:

```powershell
cd C:\Dev\KlippingeTrading
python dashboard_PyQt5.py
```

Du bör se output som:

```
============================================================
  Klippinge Investment Trading Terminal v1.0.0
============================================================
  Frozen:       False
  App dir:      C:\Dev\KlippingeTrading
  Install dir:  C:\Dev\KlippingeTrading
  User data:    C:\Users\Andreas\AppData\Roaming\KlippingeTrading
  Trading data: C:\Users\Andreas\AppData\Roaming\KlippingeTrading\Trading
  Logs:         C:\Users\Andreas\AppData\Roaming\KlippingeTrading\Logs
  Platform:     Windows 10
============================================================

  Initialized: C:\Users\Andreas\AppData\Roaming\KlippingeTrading\Trading\index_tickers.csv
  Initialized: C:\Users\Andreas\AppData\Roaming\KlippingeTrading\Trading\portfolio_positions.json
  ...
```

**Kontrollera att:**
- [ ] Appen startar utan felmeddelanden
- [ ] `Frozen: False` visas (vi är i utvecklingsläge)
- [ ] User data-mappen skapas i `%APPDATA%\KlippingeTrading\`
- [ ] Ticker-data laddas korrekt
- [ ] Portfolio-positioner visas

Om något inte fungerar, se [Felsökning](#12-felsökning) längst ner.

---

## 5. Steg 3 — Bygg .exe

### 5.1 Kör byggskriptet

```powershell
cd C:\Dev\KlippingeTrading
python build.py
```

Byggskriptet gör följande:
1. Kontrollerar att alla filer och beroenden finns
2. Kör PyInstaller med rätt inställningar
3. Skapar output i `dist\KlippingeTrading\`

Förvänta dig att det tar **3–10 minuter**.

### 5.2 Alternativ: single-file .exe

Om du vill ha en enda .exe-fil (lättare att dela, men långsammare att starta):

```powershell
python build.py --onefile
```

### 5.3 Alternativ: med portabel ZIP

```powershell
python build.py --zip
```

Skapar `dist\KlippingeTrading-v1.0.0-portable-win64.zip` som användare kan ladda ner och packa upp direkt.

### 5.4 Vad skapades?

Efter lyckat bygge har du:

```
dist\
└── KlippingeTrading\
    ├── KlippingeTrading.exe           ← Huvudprogrammet
    ├── python312.dll                   ← Python runtime
    ├── logo.ico                        ← App-ikon
    ├── index_tickers.csv               ← Bundled data
    ├── underliggande_matchade_tickers.csv
    ├── Trading\                        ← Default-data
    │   ├── index_tickers.csv
    │   ├── portfolio_positions.json
    │   └── benchmark_cache.json
    ├── assets\
    │   └── styles.css
    └── _internal\                      ← Python-bibliotek (numpy, scipy, etc.)
        ├── numpy\
        ├── scipy\
        ├── PyQt5\
        └── ... (hundratals filer)
```

---

## 6. Steg 4 — Testa den byggda .exe-filen

### 6.1 Kör .exe

```powershell
dist\KlippingeTrading\KlippingeTrading.exe
```

Eller dubbelklicka på filen i Utforskaren.

### 6.2 Kontrollera i output-fönstret

Om du byggt med `console=True` (för debugging), eller tittar i loggfilen:

```
%APPDATA%\KlippingeTrading\Logs\terminal_2025-01-31.log
```

Bör du se:

```
============================================================
  Klippinge Investment Trading Terminal v1.0.0
============================================================
  Frozen:       True          ← Bekräftar att vi kör som .exe
  App dir:      C:\Users\...\AppData\Local\Temp\_MEIxxxxxx
  Install dir:  C:\path\to\dist\KlippingeTrading
  User data:    C:\Users\Andreas\AppData\Roaming\KlippingeTrading
  ...
```

### 6.3 Checklista för testning

- [ ] Appen startar utan krasch
- [ ] Window-titeln visar "KLIPPINGE INVESTMENT TRADING TERMINAL"
- [ ] Ikonen visas korrekt i taskbar
- [ ] Ticker-data laddas (kontrollera att par visas i listan)
- [ ] Portfolio-positioner sparas och laddas korrekt
- [ ] Scheduled scan fungerar (ställ tillfälligt in tiden till om 2 minuter)
- [ ] HMM-analys kan köras
- [ ] News-flödet laddas
- [ ] Discord-notifikationer fungerar (om webhook konfigurerats)

### 6.4 Testa på en annan dator

Det ultimata testet! Kopiera hela `dist\KlippingeTrading\`-mappen till en USB-sticka eller zippa den och skicka till en kollega. Den ska fungera utan Python installerat.

---

## 7. Steg 5 — Bygg en Windows-installer (valfritt)

En installer ger en professionellare upplevelse: "Nästa → Välj mapp → Installera"-dialog, start-menygenväg, avinstallation via Kontrollpanelen.

### 7.1 Förutsättningar

- Du har redan byggt .exe (steg 3)
- Inno Setup 6 är installerat

### 7.2 Konfigurera installer.iss

Öppna `build\installer.iss` och gör dessa ändringar:

**Rad 14 — AppId:** Generera ett unikt GUID:
1. Gå till https://www.guidgenerator.com/
2. Klicka "Generate"
3. Kopiera resultatet

```iss
; FÖR:
AppId={{YOUR-UNIQUE-GUID-HERE}

; EFTER (exempel):
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
```

**Rad 19 — URL:** Ändra till ditt GitHub-repo:

```iss
#define MyAppURL "https://github.com/DITT_ANVÄNDARNAMN/klippinge-trading"
```

### 7.3 Bygg installern

**Alternativ A — Via byggskriptet:**

```powershell
python build.py --installer
```

**Alternativ B — Via Inno Setup GUI:**

1. Öppna Inno Setup Compiler (från Start-menyn)
2. File → Open → välj `build\installer.iss`
3. Build → Compile (eller tryck F9)

Resultatet hamnar i `dist\KlippingeTrading-v1.0.0-Setup.exe`.

### 7.4 Testa installern

1. Dubbelklicka `KlippingeTrading-v1.0.0-Setup.exe`
2. Gå igenom installationsguiden
3. Välj installationsplats (standard: `C:\Program Files\KlippingeTrading`)
4. Valfritt: skapa desktop-ikon
5. Klicka "Installera"
6. Starta programmet efter installation

Kontrollera att:
- [ ] Installationsguiden visas korrekt
- [ ] Programmet startar efter installation
- [ ] Start-menygenväg skapas
- [ ] Avinstallation fungerar (Kontrollpanelen → Avinstallera)

---

## 8. Steg 6 — Skapa ett GitHub-repo

GitHub ger dig två saker:
1. **Versionskontroll** för din kod
2. **Releases** som auto-uppdateraren kan kolla mot

### 8.1 Skapa ett GitHub-konto (om du inte har)

Gå till https://github.com och registrera dig.

### 8.2 Skapa ett nytt repo

1. Gå till https://github.com/new
2. Fyll i:
   - **Repository name:** `klippinge-trading` (eller vad du vill)
   - **Description:** "Professional pairs trading & statistical arbitrage terminal"
   - **Visibility:** `Private` (viktigast om du har proprietär kod)
3. Klicka "Create repository"

### 8.3 Skapa .gitignore

Skapa filen `C:\Dev\KlippingeTrading\.gitignore`:

```gitignore
# Build output
dist/
build/temp/
*.spec.bak

# Python cache
__pycache__/
*.pyc
*.pyo

# User data (ska inte versionshanteras)
Trading/engine_cache.pkl
Trading/portfolio_history.json
Trading/portfolio_positions.json
news_cache.json
.scheduler_last_run

# IDE
.vscode/
.idea/
*.swp

# OS
Thumbs.db
.DS_Store

# Secrets (VIKTIGT!)
notification_config.json
```

### 8.4 Initiera Git och pusha

```powershell
cd C:\Dev\KlippingeTrading

git init
git add .
git commit -m "Initial commit - v1.0.0"

# Koppla till GitHub (byt ut URL:en till ditt repo)
git remote add origin https://github.com/DITT_NAMN/klippinge-trading.git
git branch -M main
git push -u origin main
```

---

## 9. Steg 7 — Automatiska byggen med GitHub Actions

GitHub Actions bygger .exe + installer automatiskt varje gång du skapar en ny release-tag. Du behöver aldrig bygga manuellt igen!

### 9.1 Verifiera att workflow-filen finns

Kontrollera att denna fil finns i ditt repo:

```
.github/workflows/build-release.yml
```

Den ska redan finnas bland filerna du kopierade i steg 3.

### 9.2 Hur det fungerar

```
Du pushar en tag  →  GitHub Actions startar  →  Bygger .exe  →  Skapar Release
   v1.0.0              Windows VM                 PyInstaller      Med nedladdningslänk
                        Python 3.12               + Inno Setup
```

Workflow:en:
1. Startar en Windows-maskin i molnet
2. Installerar Python 3.12 + alla beroenden
3. Uppdaterar versionsnumret i `app_config.py`
4. Kör PyInstaller
5. Kör Inno Setup (installer)
6. Skapar en portabel ZIP
7. Publicerar allt som en GitHub Release

### 9.3 Verifiera att Actions fungerar

1. Gå till ditt repo på GitHub
2. Klicka på "Actions"-tabben
3. Du bör se workflow:en "Build & Release" listad
4. Den aktiveras automatiskt vid nästa tag-push

---

## 10. Steg 8 — Släpp din första version

### 10.1 Uppdatera versionsnummer

Öppna `app_config.py` och ändra vid behov:

```python
APP_VERSION = "1.0.0"
```

### 10.2 Skapa en release-tag

```powershell
cd C:\Dev\KlippingeTrading

# Committa alla ändringar
git add .
git commit -m "Release v1.0.0 - Initial public release"

# Skapa en versionstagg
git tag v1.0.0

# Pusha till GitHub
git push origin main --tags
```

### 10.3 Följ bygget

1. Gå till https://github.com/DITT_NAMN/klippinge-trading/actions
2. Du ser att "Build & Release" körs (gul cirkel)
3. Klicka in för att se loggen i realtid
4. När den är klar (grön bock) — gå till "Releases"

### 10.4 Verifiera releasen

Gå till: `https://github.com/DITT_NAMN/klippinge-trading/releases`

Du bör se:

```
Klippinge Trading Terminal v1.0.0
──────────────────────────────────
Assets:
  📦 KlippingeTrading-v1.0.0-Setup.exe     (installer)
  📦 KlippingeTrading-v1.0.0-portable-win64.zip  (portable)
```

### 10.5 Släpp uppdateringar framöver

Varje gång du vill släppa en ny version:

```powershell
# 1. Gör dina kodändringar
# 2. Uppdatera APP_VERSION i app_config.py till "1.1.0"

git add .
git commit -m "v1.1.0 - Lade till X, fixade Y"
git tag v1.1.0
git push origin main --tags
```

GitHub Actions bygger automatiskt och skapar en ny Release. Alla användare som kör appen ser en uppdateringsdialog vid nästa start!

---

## 11. Steg 9 — Skapa en nedladdningssida

### Alternativ A: Använd GitHub Releases direkt

Den enklaste lösningen — länka direkt till din release-sida:

```
https://github.com/DITT_NAMN/klippinge-trading/releases/latest
```

### Alternativ B: Enkel landningssida

Skapa en ren HTML-sida och hosta den via GitHub Pages:

1. Skapa en gren `gh-pages` i ditt repo
2. Lägg till en `index.html`
3. Aktivera GitHub Pages i repo-inställningarna

Eller använd en enkel hostingtjänst (Netlify, Vercel, etc.).

### Alternativ C: GitHub Pages med automatisk "latest"-länk

GitHub erbjuder en permanent URL som alltid pekar på senaste releasen:

```
https://github.com/DITT_NAMN/klippinge-trading/releases/latest/download/KlippingeTrading-v1.0.0-Setup.exe
```

> Obs: filnamnet ändras med varje version. Du kan skapa ett omdirigeringsskript eller använda GitHub API.

---

## 12. Felsökning

### Problem: "ModuleNotFoundError: No module named 'xxx'" vid körning av .exe

**Orsak:** PyInstaller hittade inte modulen automatiskt.

**Lösning:** Lägg till modulen i `build/klippinge.spec` under `hiddenimports`:

```python
hidden_imports = [
    ...
    'modulen_som_saknas',
]
```

Bygg om: `python build.py --clean`

### Problem: Antivirus flaggar .exe-filen

**Orsak:** PyInstaller-byggda .exe-filer triggar ibland falska positiva.

**Lösningar:**
1. Skicka in för whitelisting hos antivirustillverkaren
2. Signera .exe med ett kodsigneringscertifikat (ca $70-200/år)
3. Be användare lägga till undantag

### Problem: Appen hittar inte datafiler

**Symptom:** Tom tickerlista, inga par, krasch vid start.

**Debug:** Titta i loggfilen:
```
%APPDATA%\KlippingeTrading\Logs\terminal_YYYY-MM-DD.log
```

**Vanlig orsak:** Datafiler saknas i `datas`-listan i spec-filen. Kontrollera att alla CSV/JSON-filer listas.

### Problem: "Failed to execute script" utan felmeddelande

**Debug-metod:** Bygg med konsolfönster tillfälligt:

Ändra i `build/klippinge.spec`:
```python
# Ändra console från False till True temporärt
console=True,    # Visar felmeddelanden
```

Bygg om och kör. Nu visas felmeddelanden i ett konsollfönster.

### Problem: Appen är väldigt stor (500+ MB)

**Lösningar:**
1. Se till att `excludes` i spec-filen är korrekt (matplotlib, tkinter, etc.)
2. Kontrollera om PyQtWebEngine inkluderas (80+ MB) — behövs den?
3. Testa med UPX-komprimering (redan aktiverat i spec-filen)

### Problem: Appen startar långsamt (10+ sekunder)

**Orsak:** Single-file mode (`--onefile`) packar upp allt till en temp-mapp vid varje start.

**Lösning:** Använd directory mode (standard) istället. Det startar snabbare.

### Problem: "Windows protected your PC" (SmartScreen)

**Orsak:** .exe-filen är inte signerad.

**Lösning:** Klicka "More info" → "Run anyway". För att slippa detta permanent behöver du ett kodsigneringscertifikat.

---

## 13. Checklista

### Före bygge

- [ ] Python 3.10+ installerat
- [ ] `pip install pyinstaller` kört
- [ ] Alla beroenden installerade (`pip install -r requirements.txt`)
- [ ] `app_config.py` kopierad till projektmappen
- [ ] `auto_updater.py` kopierad till projektmappen
- [ ] `GITHUB_REPO` uppdaterad i `app_config.py`
- [ ] Discord webhook-URL borttagen från `dashboard_PyQt5.py`
- [ ] Discord webhook konfigurerad i `notification_config.json`
- [ ] Modifierad `dashboard_PyQt5.py` används (inte originalversionen)

### Bygge

- [ ] `python build.py` körs utan fel
- [ ] `dist\KlippingeTrading\KlippingeTrading.exe` existerar
- [ ] .exe startar och visar terminalen
- [ ] Data sparas i `%APPDATA%\KlippingeTrading\`
- [ ] Testat på dator utan Python

### Distribution

- [ ] GitHub-repo skapat
- [ ] `.gitignore` skapad (utesluter secrets och cache)
- [ ] `notification_config.json` exkluderad från repo
- [ ] Discord webhook roterad/regenererad
- [ ] Kod pushad till GitHub
- [ ] GitHub Actions workflow fungerar
- [ ] Första release-tag skapad (`v1.0.0`)
- [ ] Release publicerad med .exe/.zip

---

## Snabbreferens — Kommandon

```powershell
# ── Utveckling ──
python dashboard_PyQt5.py                    # Kör i utvecklingsläge

# ── Bygga ──
python build.py                              # Standard-build (mapp)
python build.py --onefile                    # Single .exe
python build.py --zip                        # + portabel ZIP
python build.py --installer                  # + Windows installer
python build.py --clean                      # Rensa gamla byggen först

# ── Git & Release ──
git add . && git commit -m "v1.1.0 - ..."   # Committa ändringar
git tag v1.1.0                                # Skapa version-tag
git push origin main --tags                   # Pusha → trigger bygge

# ── Debug ──
dist\KlippingeTrading\KlippingeTrading.exe   # Kör byggd app
type "%APPDATA%\KlippingeTrading\Logs\terminal_*.log"  # Läs loggar
```

---

*Guide skapad för Klippinge Investment Trading Terminal v1.0.0*

import os
import io
import json
import textwrap
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import pandas as pd
import cv2
import streamlit as st


def render_html_safely(html_content: str):
    """
    Cleans HTML content by removing leading whitespace from every line
    to guarantee Streamlit doesn't render it as a raw code block.
    """
    lines = [line.strip() for line in html_content.splitlines()]
    cleaned_html = "\n".join(lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)


# Check OpenCV built-in barcode engine availability
OPENCV_BARCODE_AVAILABLE = hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector")

# Safe import for easyocr
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception as e:
    EASYOCR_AVAILABLE = False
    EASYOCR_ERROR = str(e)

# Safe import for fuzzy matching
try:
    from thefuzz import fuzz
    THEFUZZ_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        THEFUZZ_AVAILABLE = True
    except ImportError:
        THEFUZZ_AVAILABLE = False


REQUIRED_COLUMNS = [
    "barcode", "keywords", "drug_name", "active_ingredient", "dosage", "uses", "contraindications"
]


# ==========================================
# STREAMLIT PAGE CONFIG & CUSTOM CSS STYLING
# ==========================================
st.set_page_config(
    page_title="MedGuard | Verified Medicine Verification",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Custom-drawn, tileable medicine-themed background pattern.
# Original SVG artwork (not a stock photo) kept at very low opacity so
# it reads as texture behind the light theme rather than a distraction.
# ------------------------------------------------------------------
_MED_BG_SVG = """<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>
  <g fill='none' stroke='#16A34A' stroke-width='1.3' stroke-opacity='0.07'>
    <!-- capsule -->
    <g transform='translate(20,25) rotate(35)'>
      <rect x='0' y='0' width='46' height='18' rx='9'/>
      <line x1='23' y1='0' x2='23' y2='18'/>
    </g>
    <!-- round tablet with score line -->
    <circle cx='165' cy='45' r='16'/>
    <line x1='153' y1='45' x2='177' y2='45'/>
    <!-- syringe -->
    <g transform='translate(120,120) rotate(-20)'>
      <rect x='0' y='0' width='34' height='10' rx='2'/>
      <line x1='34' y1='2.5' x2='44' y2='2.5'/>
      <line x1='34' y1='7.5' x2='44' y2='7.5'/>
      <line x1='0' y1='5' x2='-8' y2='5'/>
    </g>
    <!-- medical cross -->
    <g transform='translate(35,150)'>
      <rect x='7' y='0' width='8' height='24' rx='2'/>
      <rect x='0' y='7' width='22' height='8' rx='2'/>
    </g>
  </g>
</svg>"""
_MED_BG_DATA_URI = "data:image/svg+xml," + urllib.parse.quote(_MED_BG_SVG)

# Modern White + Green CSS Styling
_CSS_TEMPLATE = textwrap.dedent("""
    <style>
        /* Global Styles & Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #0F172A;
        }

        /* Medicine-themed background texture across the whole app */
        [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
            background-image:
                radial-gradient(circle at 12% 15%, rgba(22, 163, 74, 0.07) 0%, transparent 45%),
                radial-gradient(circle at 88% 85%, rgba(21, 128, 61, 0.06) 0%, transparent 45%),
                url("__MED_BG_DATA_URI__");
            background-repeat: no-repeat, no-repeat, repeat;
            background-size: auto, auto, 220px 220px;
            background-attachment: fixed;
        }

        [data-testid="stSidebar"] {
            background-color: #F0FDF4;
            border-right: 1px solid #BBF7D0;
        }

        /* App Header Banner */
        .header-container {
            position: relative;
            background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 60%, #DCFCE7 100%);
            padding: 2.25rem 2rem;
            border-radius: 16px;
            color: #0F172A;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(22, 163, 74, 0.12);
            border: 1px solid #BBF7D0;
            overflow: hidden;
        }
        /* Decorative background: soft green glow + faint dot grid, pure CSS (no external assets) */
        .header-container::before {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(22, 163, 74, 0.18) 0%, rgba(22, 163, 74, 0) 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .header-container::after {
            content: "";
            position: absolute;
            inset: 0;
            background-image: radial-gradient(rgba(22, 163, 74, 0.08) 1px, transparent 1px);
            background-size: 22px 22px;
            pointer-events: none;
        }
        .header-content {
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }
        .header-icon {
            flex-shrink: 0;
            width: 56px;
            height: 56px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(22, 163, 74, 0.10);
            border: 1px solid rgba(22, 163, 74, 0.35);
            border-radius: 14px;
        }
        .header-icon svg {
            width: 30px;
            height: 30px;
        }
        .header-title {
            font-size: 2.25rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #14532D;
        }
        .header-subtitle {
            color: #4B5563;
            font-size: 1.05rem;
            margin-top: 0.5rem;
            font-weight: 400;
        }
        .section-icon {
            width: 18px;
            height: 18px;
            vertical-align: -3px;
            margin-right: 4px;
            opacity: 0.9;
        }

        /* Badges */
        .badge-barcode {
            background-color: #DBEAFE;
            color: #1D4ED8;
            font-size: 0.85rem;
            padding: 4px 12px;
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #93C5FD;
        }
        .badge-fuzzy {
            background-color: #F3E8FF;
            color: #7E22CE;
            font-size: 0.85rem;
            padding: 4px 12px;
            border-radius: 8px;
            font-weight: 600;
            border: 1px solid #D8B4FE;
        }
        .badge-source-remote {
            background-color: #E0F2FE;
            color: #0369A1;
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
        }
        .badge-source-local {
            background-color: #F1F5F9;
            color: #334155;
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
        }

        /* Result Card Styles */
        .verified-card {
            background: linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%);
            border: 1px solid #4ADE80;
            border-radius: 16px;
            padding: 1.75rem;
            color: #14532D;
            margin-top: 1rem;
            box-shadow: 0 10px 20px rgba(22, 163, 74, 0.08);
        }
        .warning-card {
            background: linear-gradient(145deg, #FEF2F2 0%, #FEE2E2 100%);
            border: 1px solid #FCA5A5;
            border-radius: 16px;
            padding: 1.75rem;
            color: #7F1D1D;
            margin-top: 1rem;
            box-shadow: 0 10px 20px rgba(220, 38, 38, 0.08);
        }
        .med-title {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
            color: #14532D;
        }
        .field-label {
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            color: #15803D;
            margin-top: 1rem;
            margin-bottom: 0.25rem;
        }
        .field-value {
            font-size: 1rem;
            line-height: 1.5;
            color: #14532D;
            background: rgba(255, 255, 255, 0.6);
            padding: 0.75rem;
            border-radius: 8px;
            border: 1px solid rgba(22, 163, 74, 0.15);
        }

        /* Code Box for Raw OCR */
        .raw-box {
            background-color: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.9rem;
            color: #334155;
            white-space: pre-wrap;
            max-height: 250px;
            overflow-y: auto;
        }

        /* Splash / Intro Screen */
        .splash-screen {
            position: fixed;
            inset: 0;
            z-index: 9999;
            background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 60%, #DCFCE7 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            animation: splashFadeOut 1.9s ease forwards;
        }
        .splash-logo {
            width: 92px;
            height: 92px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(22, 163, 74, 0.10);
            border: 1px solid rgba(22, 163, 74, 0.35);
            border-radius: 22px;
            margin-bottom: 1.25rem;
            animation: splashPulse 1.9s ease forwards;
        }
        .splash-logo svg {
            width: 50px;
            height: 50px;
        }
        .splash-title {
            font-size: 2.75rem;
            font-weight: 700;
            color: #14532D;
            letter-spacing: 0.02em;
        }
        .splash-subtitle {
            color: #4B5563;
            font-size: 1rem;
            margin-top: 0.5rem;
        }
        @keyframes splashFadeOut {
            0%   { opacity: 1; }
            65%  { opacity: 1; }
            100% { opacity: 0; }
        }
        @keyframes splashPulse {
            0%   { transform: scale(0.9); opacity: 0; }
            25%  { transform: scale(1); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
    </style>
""")

st.markdown(
    _CSS_TEMPLATE.strip().replace("__MED_BG_DATA_URI__", _MED_BG_DATA_URI),
    unsafe_allow_html=True
)


# ==========================================
# OPENFDA API INTEGRATION (CACHED WITH TTL=3600)
# ==========================================

@st.cache_data(ttl=3600)
def search_openfda(query: str):
    """
    Search OpenFDA Drug Label API (https://api.fda.gov/drug/label.json).
    Parses and returns: Brand Name, Active Ingredient, Usage, Dosage, and Warnings.
    Cached with @st.cache_data(ttl=3600) for fast execution.
    """
    if not query or not str(query).strip():
        return None

    clean_q = str(query).strip().replace('"', '')
    if len(clean_q) < 3:
        return None

    urls_to_try = [
        f'https://api.fda.gov/drug/label.json?search=openfda.brand_name:"{clean_q}"+OR+openfda.generic_name:"{clean_q}"+OR+openfda.substance_name:"{clean_q}"&limit=1',
        f'https://api.fda.gov/drug/label.json?search=active_ingredient:"{clean_q}"+OR+openfda.brand_name:"{clean_q}"&limit=1',
        f'https://api.fda.gov/drug/label.json?search={clean_q}&limit=1'
    ]

    item = None
    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if results:
                    item = results[0]
                    break
        except Exception:
            continue

    if not item:
        return None

    openfda = item.get("openfda", {})

    brand_names = openfda.get("brand_name", [])
    brand_name = ", ".join(brand_names) if brand_names else (item.get("drug_name") or clean_q.title())

    generic_names = openfda.get("generic_name", [])
    active_ing = item.get("active_ingredient", openfda.get("substance_name", []))
    
    if isinstance(active_ing, list) and active_ing:
        active_ingredient = "; ".join([str(a).strip() for a in active_ing if str(a).strip()])
    elif generic_names:
        active_ingredient = ", ".join(generic_names)
    else:
        active_ingredient = str(active_ing).strip() if active_ing else "N/A"

    usage_list = item.get("indications_and_usage", item.get("purpose", []))
    if isinstance(usage_list, list):
        usage = " ".join([str(u).strip() for u in usage_list if str(u).strip()])
    else:
        usage = str(usage_list).strip()

    dosage_list = item.get("dosage_and_administration", [])
    if isinstance(dosage_list, list):
        dosage = " ".join([str(d).strip() for d in dosage_list if str(d).strip()])
    else:
        dosage = str(dosage_list).strip()

    warnings_list = item.get("warnings", item.get("warnings_and_cautions", []))
    if isinstance(warnings_list, list):
        warnings = " ".join([str(w).strip() for w in warnings_list if str(w).strip()])
    else:
        warnings = str(warnings_list).strip()

    # Drug-drug interactions, straight from the official FDA label text.
    # (Fallback source since NLM discontinued its dedicated Interaction API in Jan 2024.)
    interactions_list = item.get("drug_interactions", [])
    if isinstance(interactions_list, list):
        interactions = " ".join([str(i).strip() for i in interactions_list if str(i).strip()])
    else:
        interactions = str(interactions_list).strip()

    def format_text(val, default_msg="Not specified in OpenFDA label.", max_chars=350):
        if not val or val == "N/A":
            return default_msg
        val_clean = str(val).strip()
        if len(val_clean) > max_chars:
            return val_clean[:max_chars].strip() + "..."
        return val_clean

    return {
        "brand_name": brand_name,
        "generic_name": ", ".join(generic_names) if generic_names else "",
        "active_ingredient": format_text(active_ingredient, "Refer to packaging for active ingredient details.", 250),
        "usage": format_text(usage, "Refer to packaging for indications.", 350),
        "dosage": format_text(dosage, "Refer to packaging for dosage instructions.", 350),
        "warnings": format_text(warnings, "No specific warnings listed in OpenFDA record.", 350),
        "interactions": format_text(interactions, "", 400),  # empty string = not listed, handled at render time
    }


@st.cache_data(ttl=86400)
def lookup_ndc_by_barcode(barcode: str):
    """
    TIER 1 FALLBACK — When a scanned barcode is not found in the local database,
    query the FDA NDC (National Drug Code) Directory (api.fda.gov/drug/ndc.json).
    Many medicine barcodes (UPC-A / GS1) embed the 10 or 11-digit NDC code, so we
    try the raw scanned value plus the most likely embedded NDC substrings.
    Cached for 24 hours since the NDC Directory changes infrequently.
    """
    if not barcode or not str(barcode).strip():
        return None

    raw = str(barcode).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())

    candidates = [raw]
    if len(digits) >= 11:
        candidates.append(digits[-11:])
    if len(digits) >= 10:
        candidates.append(digits[-10:])
    # de-duplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for cand in candidates:
        url = f'https://api.fda.gov/drug/ndc.json?search=product_ndc:"{cand}"+OR+package_ndc:"{cand}"&limit=1'
        try:
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if results:
                    item = results[0]
                    active_ings = item.get("active_ingredients", [])
                    active_str = ", ".join(
                        f"{a.get('name', '')} {a.get('strength', '')}".strip()
                        for a in active_ings if a.get("name")
                    ) or "N/A"

                    dosage_form = item.get("dosage_form", "")
                    route = ", ".join(item.get("route", [])) if item.get("route") else ""
                    dosage_str = ", ".join(filter(None, [dosage_form, route])) or "Refer to official FDA label."

                    pharm_classes = item.get("pharm_class", [])
                    uses_str = "; ".join(pharm_classes) if pharm_classes else "Refer to official FDA label for indications."

                    return {
                        "drug_name": item.get("brand_name") or item.get("generic_name") or "Unknown (NDC Match)",
                        "active_ingredient": active_str,
                        "dosage": dosage_str,
                        "uses": uses_str,
                        "contraindications": "Not listed by NDC Directory — see official FDA label for full contraindications.",
                        "barcode": raw,
                        "keywords": [],
                        "_ndc_labeler": item.get("labeler_name", ""),
                        "_ndc_product_ndc": item.get("product_ndc", ""),
                    }
        except Exception:
            continue
    return None


@st.cache_data(ttl=86400)
def lookup_rxnorm_approximate(term: str, max_entries: int = 1):
    """
    TIER 2 FALLBACK — When local fuzzy matching (thefuzz) against the small local
    keyword list fails, delegate to RxNav's server-side approximate term matcher
    (rxnav.nlm.nih.gov/REST/approximateTerm.json). This handles OCR misspellings
    and partial names far better than naive string matching against a small list,
    since it draws on the full RxNorm drug name vocabulary.
    Cached for 24 hours.
    """
    if not term or not str(term).strip():
        return None
    clean_term = str(term).strip()
    url = f"https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term={urllib.parse.quote(clean_term)}&maxEntries={max_entries}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            if candidates:
                best = candidates[0]
                name = (best.get("name") or "").strip()
                rxcui = best.get("rxcui")
                score = best.get("score")
                if name and rxcui:
                    return {"name": name, "rxcui": rxcui, "score": int(score) if score else 0}
    except Exception:
        pass
    return None


def extract_brand_name(drug_name_str: str) -> str:
    if not drug_name_str:
        return ""
    # Split by slash / first to get the primary brand name
    first_part = drug_name_str.split("/")[0]
    words = [w for w in first_part.split() if len(w) >= 3 and w.isalpha()]
    return words[0] if words else first_part.split()[0]


def extract_generic_name(active_ingredient_str: str) -> str:
    if not active_ingredient_str:
        return ""
    # Split by comma first to get the first compound
    first_part = active_ingredient_str.split(",")[0]
    # Extract only alphabetical characters from words of length >= 3
    words = [w for w in first_part.split() if len(w) >= 3 and w.isalpha()]
    return words[0] if words else first_part.split()[0]


@st.cache_data(ttl=86400)
def lookup_rxnav(keyword: str):
    """
    Search RxNav API (https://rxnav.nlm.nih.gov/REST/rxcui.json?name={keyword})
    to extract RxCUI codes. Cached for 24 hours.
    """
    if not keyword or not str(keyword).strip():
        return None
    clean_keyword = str(keyword).strip()
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={urllib.parse.quote(clean_keyword)}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            rxnorm_ids = data.get("idGroup", {}).get("rxnormId", [])
            if rxnorm_ids:
                return rxnorm_ids
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def lookup_dailymed(keyword: str):
    """
    Search DailyMed API (https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={keyword})
    to extract official package insert links & SPL IDs. Cached for 24 hours.
    """
    if not keyword or not str(keyword).strip():
        return None
    clean_keyword = str(keyword).strip()
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={urllib.parse.quote(clean_keyword)}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            spl_list = data.get("spl", []) or data.get("data", [])
            results = []
            for spl in spl_list:
                setid = spl.get("setid")
                title = spl.get("title", "Unknown Label")
                if setid:
                    link = f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={setid}"
                    results.append({
                        "spl_id": setid,
                        "title": title,
                        "link": link
                    })
            if results:
                return results
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def lookup_wikipedia_vietnam(keyword: str):
    """
    Search Wikipedia Vietnam API
    (https://vi.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={keyword}&format=json)
    to extract Vietnamese summary text. Cached for 24 hours.
    """
    if not keyword or not str(keyword).strip():
        return None
    clean_keyword = str(keyword).strip()
    url = f"https://vi.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(clean_keyword)}&format=json"
    headers = {
        "User-Agent": "MedGuard/1.3 (contact@example.com) Python-requests/2.31"
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "").strip()
                    if extract:
                        return {
                            "title": page_data.get("title", clean_keyword),
                            "extract": extract
                        }
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def lookup_wikipedia_english(keyword: str):
    """
    Search Wikipedia English API (en.wikipedia.org) to extract an English
    summary. Same shape/behavior as lookup_wikipedia_vietnam, just a
    different-language endpoint. Cached for 24 hours.
    """
    if not keyword or not str(keyword).strip():
        return None
    clean_keyword = str(keyword).strip()
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(clean_keyword)}&format=json"
    headers = {
        "User-Agent": "MedGuard/1.3 (contact@example.com) Python-requests/2.31"
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "").strip()
                    if extract:
                        return {
                            "title": page_data.get("title", clean_keyword),
                            "extract": extract
                        }
    except Exception:
        pass
    return None


def fetch_all_external_apis(brand_query: str, generic_query: str):
    """
    Queries OpenFDA, RxNav, DailyMed, Wikipedia (EN) and Wikipedia (VI) in
    parallel, using appropriate queries and fallbacks.
    """
    results = {
        "openfda": None,
        "rxnav": None,
        "dailymed": None,
        "wikipedia": None,
        "wikipedia_en": None
    }
    if not brand_query and not generic_query:
        return results

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks
        # OpenFDA: brand query or generic query
        future_fda = executor.submit(search_openfda, brand_query or generic_query)
        # RxNav: generic query or brand query
        future_rxnav = executor.submit(lookup_rxnav, generic_query or brand_query)
        # DailyMed: brand query or generic query
        future_dailymed = executor.submit(lookup_dailymed, brand_query or generic_query)
        # Wikipedia (Vietnamese): generic query or brand query
        future_wiki = executor.submit(lookup_wikipedia_vietnam, generic_query or brand_query)
        # Wikipedia (English): generic query or brand query
        future_wiki_en = executor.submit(lookup_wikipedia_english, generic_query or brand_query)
        
        # Gather results
        results["openfda"] = future_fda.result()
        results["rxnav"] = future_rxnav.result()
        results["dailymed"] = future_dailymed.result()
        results["wikipedia"] = future_wiki.result()
        results["wikipedia_en"] = future_wiki_en.result()

    # Apply secondary fallbacks sequentially if one of them is empty and the queries are different
    if not results["openfda"] and generic_query and generic_query != brand_query:
        results["openfda"] = search_openfda(generic_query)
        
    if not results["rxnav"] and brand_query and brand_query != generic_query:
        results["rxnav"] = lookup_rxnav(brand_query)
        
    if not results["dailymed"] and generic_query and generic_query != brand_query:
        results["dailymed"] = lookup_dailymed(generic_query)
        
    if not results["wikipedia"] and brand_query and brand_query != generic_query:
        results["wikipedia"] = lookup_wikipedia_vietnam(brand_query)

    if not results["wikipedia_en"] and brand_query and brand_query != generic_query:
        results["wikipedia_en"] = lookup_wikipedia_english(brand_query)
        
    return results


# ==========================================
# MIL — BILINGUAL STRINGS (EN / VI)
# ==========================================

MIL_STRINGS = {
    "en": {
        # Claim audit
        "claim_audit_title": "⚠️ **Claim Audit Alert**",
        "claim_audit_body": "Suspicious / exaggerated phrases detected in scanned text: {phrases}  \nPlease verify through official sources before use.",
        # Credibility
        "cred_label": "CREDIBILITY",
        "cred_high_desc": "Verified by official FDA / DailyMed or exact Barcode match",
        "cred_med_desc": "Partial match via fuzzy OCR + Wikipedia reference",
        "cred_low_desc": "Unverified — OCR only, no official API confirmation",
        "cred_high": "HIGH",
        "cred_med": "MEDIUM",
        "cred_low": "LOW",
        # Recall
        "recall_title": "🚨 **FDA RECALL NOTICE FOUND**",
        "recall_firm": "Firm",
        "recall_reason": "Reason",
        "recall_status": "Status",
        "recall_date": "Date",
        # Cross-check table
        "xcheck_title": "📊 Cross-Check Table: OCR vs Official API",
        "xcheck_field": "Field",
        "xcheck_ocr": "OCR / Local DB",
        "xcheck_api": "Official API Data",
        "xcheck_status": "Status",
        "xcheck_drug": "Drug Name",
        "xcheck_ingredient": "Active Ingredient",
        "xcheck_dosage": "Dosage",
        "xcheck_warnings": "Warnings",
        "xcheck_checklabel": "Check label",
        "xcheck_match": "✅ Match",
        "xcheck_partial": "⚠️ Partial",
        "xcheck_notfound": "❌ Not Found",
        "xcheck_available": "✅ Available",
        "xcheck_seefda": "⚠️ See FDA",
        "xcheck_notverified": "❌ Not Verified",
        # MIL Guide
        "guide_title": "💡 MIL Guide: 4 Steps to Spot Fake / Unverified Medicines",
        "guide_body": """
**Step 1 — Check the Registration Code 📋**
Vietnamese medicines must carry a valid registration number printed on the label:
- `VD-XXXXX-XX` → Domestically produced medicine (manufactured in Vietnam)
- `VS-XXXXX-XX` → Traditional / herbal product (Thuốc đông y / thảo dược)
- `GC-XXXXX-XX` → Import permit (Giấy phép nhập khẩu — imported drug)
- `VN-XXXXX-XX` → Foreign-registered imported medicine

**Step 2 — Verify the Barcode Origin 🔢**
- Barcodes starting with **893** = Made in Vietnam
- Barcodes starting with **000–019** = Made in USA
- Barcodes starting with **400–440** = Made in Germany
- Barcodes starting with **690–699** = Made in China
- A barcode prefix that contradicts the stated country of origin is a **red flag**.

**Step 3 — Watch for Exaggerated or Misleading Claims ⚠️**
Suspect counterfeiting or misinformation if the label uses:
- "100% cure", "guaranteed cure" — "miracle drug" (thần dược), "specific cure" (đặc trị)
- "no side effects" — "heals all ailments" (bách bệnh), "secret formula" (bí quyết)
- "no prescription needed" (không cần đơn) — "instant improvement" (cải thiện ngay)
- No listed active ingredient, vague dosage, or pressure sales language

**Step 4 — Cross-Check with Official Sources 🌐**
- **OpenFDA**: [api.fda.gov](https://api.fda.gov) — U.S. FDA drug label database
- **DailyMed**: [dailymed.nlm.nih.gov](https://dailymed.nlm.nih.gov) — Official package inserts (NLM)
- **Drug Authority Vietnam (DAV)**: [dav.gov.vn](https://dav.gov.vn) — Vietnam Ministry of Health drug registry
- **WHO Prequalification**: [extranet.who.int](https://extranet.who.int/prequal) — WHO-certified medicines
        """,
    },
    "vi": {
        # Claim audit
        "claim_audit_title": "⚠️ **Cảnh báo kiểm tra tuyên bố**",
        "claim_audit_body": "Phát hiện cụm từ đáng ngờ / phóng đại trong văn bản quét: {phrases}  \nVui lòng xác minh qua nguồn chính thức trước khi sử dụng.",
        # Credibility
        "cred_label": "ĐỘ TIN CẬY",
        "cred_high_desc": "Đã xác minh qua FDA / DailyMed chính thức hoặc khớp mã vạch chính xác",
        "cred_med_desc": "Khớp một phần qua OCR mờ + tham chiếu Wikipedia",
        "cred_low_desc": "Chưa xác minh — chỉ OCR, không có xác nhận API chính thức",
        "cred_high": "CAO",
        "cred_med": "TRUNG BÌNH",
        "cred_low": "THẤP",
        # Recall
        "recall_title": "🚨 **ĐÃ TÌM THẤY THÔNG BÁO THU HỒI FDA**",
        "recall_firm": "Công ty",
        "recall_reason": "Lý do",
        "recall_status": "Trạng thái",
        "recall_date": "Ngày",
        # Cross-check table
        "xcheck_title": "📊 Bảng đối chiếu: OCR vs API chính thức",
        "xcheck_field": "Trường",
        "xcheck_ocr": "OCR / CSDL cục bộ",
        "xcheck_api": "Dữ liệu API chính thức",
        "xcheck_status": "Trạng thái",
        "xcheck_drug": "Tên thuốc",
        "xcheck_ingredient": "Hoạt chất",
        "xcheck_dosage": "Liều dùng",
        "xcheck_warnings": "Cảnh báo",
        "xcheck_checklabel": "Kiểm tra nhãn",
        "xcheck_match": "✅ Khớp",
        "xcheck_partial": "⚠️ Một phần",
        "xcheck_notfound": "❌ Không tìm thấy",
        "xcheck_available": "✅ Có sẵn",
        "xcheck_seefda": "⚠️ Xem FDA",
        "xcheck_notverified": "❌ Chưa xác minh",
        # MIL Guide
        "guide_title": "💡 Hướng dẫn MIL: 4 bước nhận biết thuốc giả / chưa được kiểm duyệt",
        "guide_body": """
**Bước 1 — Kiểm tra mã đăng ký 📋**
Thuốc lưu hành tại Việt Nam phải có số đăng ký hợp lệ in trên nhãn:
- `VD-XXXXX-XX` → Thuốc sản xuất trong nước
- `VS-XXXXX-XX` → Thuốc đông y / thảo dược truyền thống
- `GC-XXXXX-XX` → Thuốc nhập khẩu (Giấy phép nhập khẩu)
- `VN-XXXXX-XX` → Thuốc nhập khẩu đã đăng ký nước ngoài

**Bước 2 — Xác minh nguồn gốc mã vạch 🔢**
- Mã vạch bắt đầu bằng **893** = Sản xuất tại Việt Nam
- Mã vạch bắt đầu bằng **000–019** = Sản xuất tại Mỹ
- Mã vạch bắt đầu bằng **400–440** = Sản xuất tại Đức
- Mã vạch bắt đầu bằng **690–699** = Sản xuất tại Trung Quốc
- Tiền tố mã vạch mâu thuẫn với nước xuất xứ là **dấu hiệu đáng ngờ**.

**Bước 3 — Cảnh giác với các tuyên bố phóng đại ⚠️**
Nghi ngờ hàng giả hoặc thông tin sai lệch nếu nhãn thuốc có:
- "Thần dược" (miracle drug), "đặc trị" (specific cure), "chữa khỏi hoàn toàn"
- "Không tác dụng phụ", "bách bệnh" (cures all ailments), "bí quyết" (secret formula)
- "Không cần đơn thuốc", "cải thiện ngay" (instant improvement)
- Không liệt kê hoạt chất, liều dùng mơ hồ, hoặc ngôn ngữ tiếp thị áp lực

**Bước 4 — Đối chiếu với nguồn chính thức 🌐**
- **OpenFDA**: [api.fda.gov](https://api.fda.gov) — Cơ sở dữ liệu nhãn thuốc FDA Hoa Kỳ
- **DailyMed**: [dailymed.nlm.nih.gov](https://dailymed.nlm.nih.gov) — Tờ thông tin thuốc chính thức (NLM)
- **Cục Dược Việt Nam (DAV)**: [dav.gov.vn](https://dav.gov.vn) — Sổ đăng ký thuốc Bộ Y tế
- **WHO Prequalification**: [extranet.who.int](https://extranet.who.int/prequal) — Thuốc được WHO chứng nhận
        """,
    },
}

# Suspicious patterns — detection uses both languages regardless of UI lang
SUSPICIOUS_PATTERNS = [
    "100% cure", "miracle", "thần dược", "đặc trị", "no side effect",
    "guaranteed", "instant cure", "chữa khỏi", "đặc hiệu", "bí quyết",
    "không cần đơn", "cải thiện ngay", "thần kỳ", "bách bệnh", "100%"
]


# ==========================================
# MIL FEATURE 1 — CLAIM AUDIT
# ==========================================

def run_claim_audit(text: str) -> list:
    """Return list of suspicious phrases found in OCR text (case-insensitive)."""
    if not text:
        return []
    text_lower = text.lower()
    return [p for p in SUSPICIOUS_PATTERNS if p.lower() in text_lower]


# ==========================================
# MIL FEATURE 2 — CREDIBILITY SCORE
# ==========================================

def get_credibility_score(match_type, external_data, lang="en"):
    """
    Returns (label, color, emoji, description) based on data source quality.
    HIGH  = OpenFDA/DailyMed confirmed or Barcode exact match.
    MEDIUM = Wikipedia or fuzzy OCR fallback.
    LOW   = Unverified OCR only.
    """
    s = MIL_STRINGS[lang]
    has_fda  = bool(external_data and (external_data.get("openfda") or external_data.get("dailymed")))
    has_wiki = bool(external_data and (external_data.get("wikipedia") or external_data.get("wikipedia_en")))
    if match_type in ("BARCODE", "NDC_MATCH") or has_fda:
        return s["cred_high"], "#16a34a", "🟢", s["cred_high_desc"]
    elif match_type in ("FUZZY_OCR", "RXNORM_MATCH") and (has_wiki or has_fda):
        return s["cred_med"], "#ca8a04", "🟡", s["cred_med_desc"]
    elif match_type == "RXNORM_MATCH":
        return s["cred_med"], "#ca8a04", "🟡", s["cred_med_desc"]
    else:
        return s["cred_low"], "#dc2626", "🔴", s["cred_low_desc"]


def render_credibility_badge(match_type, external_data, lang="en"):
    """Render an inline credibility score badge."""
    s = MIL_STRINGS[lang]
    label, color, emoji, desc = get_credibility_score(match_type, external_data, lang)
    html = (
        f'<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,0.7);'
        f'border:1px solid {color};border-radius:9999px;padding:5px 14px;margin:0.5rem 0;">'
        f'<span style="font-size:1rem;">{emoji}</span>'
        f'<span style="font-weight:700;color:{color};font-size:0.85rem;">{s["cred_label"]}: {label}</span>'
        f'<span style="color:#64748B;font-size:0.78rem;"> — {desc}</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ==========================================
# MIL FEATURE 3 — RECALL ALERT
# ==========================================

@st.cache_data(ttl=3600)
def check_fda_recall(keyword: str):
    """Query FDA enforcement API for recall notices. Returns first result dict or None."""
    if not keyword or len(keyword.strip()) < 3:
        return None
    kw = keyword.strip().replace('"', '')
    url = f"https://api.fda.gov/drug/enforcement.json?search=product_description:{kw}&limit=1"
    try:
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                rec = results[0]
                return {
                    "recall_number": rec.get("recall_number", "N/A"),
                    "reason": rec.get("reason_for_recall", "N/A")[:250],
                    "status": rec.get("status", "N/A"),
                    "date": rec.get("recall_initiation_date", "N/A"),
                    "firm": rec.get("recalling_firm", "N/A"),
                }
    except Exception:
        pass
    return None


def render_recall_alert(keyword: str, lang="en"):
    """Check and display FDA recall notice if found."""
    if not keyword:
        return
    s = MIL_STRINGS[lang]
    recall = check_fda_recall(keyword)
    if recall:
        st.warning(
            f"{s['recall_title']} — Recall #{recall['recall_number']}  \n"
            f"**{s['recall_firm']}**: {recall['firm']}  \n"
            f"**{s['recall_reason']}**: {recall['reason']}  \n"
            f"**{s['recall_status']}**: {recall['status']} | **{s['recall_date']}**: {recall['date']}"
        )


# ==========================================
# MIL FEATURE 4 — CROSS-CHECK TABLE
# ==========================================

def render_cross_check_table(matched_med, ocr_text, external_data, lang="en"):
    """Render a compact OCR vs Official API vs Verification Status comparison table."""
    s = MIL_STRINGS[lang]
    openfda = external_data.get("openfda") if external_data else None
    ocr_snip = (ocr_text[:60] + "...") if ocr_text and len(ocr_text) > 60 else (ocr_text or "—")

    fields = [
        (s["xcheck_drug"],
         ocr_snip,
         openfda.get("brand_name", "—") if openfda else "—",
         matched_med.get("drug_name", "—") if matched_med else "—"),
        (s["xcheck_ingredient"],
         matched_med.get("active_ingredient", "—") if matched_med else "—",
         (openfda.get("active_ingredient", "—") or "—")[:80] if openfda else "—",
         s["xcheck_match"] if matched_med and openfda else (s["xcheck_partial"] if matched_med or openfda else s["xcheck_notfound"])),
        (s["xcheck_dosage"],
         matched_med.get("dosage", "—") if matched_med else "—",
         (openfda.get("dosage", "—") or "—")[:80] if openfda else "—",
         s["xcheck_available"] if matched_med or openfda else s["xcheck_notfound"]),
        (s["xcheck_warnings"],
         s["xcheck_checklabel"] if matched_med else "—",
         (openfda.get("warnings", "—") or "—")[:80] if openfda else "—",
         s["xcheck_seefda"] if openfda else s["xcheck_notverified"]),
    ]
    table_rows = [
        {s["xcheck_field"]: f, s["xcheck_ocr"]: o, s["xcheck_api"]: a, s["xcheck_status"]: st_}
        for f, o, a, st_ in fields
    ]
    with st.expander(s["xcheck_title"], expanded=False):
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ==========================================
# MIL FEATURE 5 — EDUCATIONAL GUIDE
# ==========================================

def render_mil_guide(lang="en"):
    """Collapsible MIL guide explaining how to spot fake/unverified medicines."""
    s = MIL_STRINGS[lang]
    with st.expander(s["guide_title"], expanded=False):
        st.markdown(s["guide_body"])


# ==========================================
# CACHED RESOURCE & DATA LOADERS (TTL = 3600s)
# ==========================================

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns, data types, and keyword lists in the DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    string_cols = ["barcode", "drug_name", "active_ingredient", "dosage", "uses", "contraindications"]
    for col in string_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    def parse_keywords(val):
        if isinstance(val, list):
            return [str(k).strip() for k in val if str(k).strip()]
        if isinstance(val, str) and val.strip():
            v_str = val.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    parsed = json.loads(v_str.replace("'", '"'))
                    if isinstance(parsed, list):
                        return [str(k).strip() for k in parsed]
                except Exception:
                    pass
            return [k.strip() for k in v_str.split(",") if k.strip()]
        return []

    df["keywords"] = df["keywords"].apply(parse_keywords)
    return df[REQUIRED_COLUMNS]


@st.cache_data(ttl=3600)
def load_medicine_database(source_url: str = None):
    """
    Load medicine records dynamically with a 1-hour cache TTL (ttl=3600).
    """
    df = None
    source_name = "Local File"
    status_msg = "Loaded successfully from local database.json"
    is_fallback = False

    if source_url and source_url.strip():
        url = source_url.strip()
        try:
            if "docs.google.com/spreadsheets" in url:
                if "/export" not in url:
                    if "/edit" in url:
                        url = url.split("/edit")[0] + "/export?format=csv"
                    else:
                        url = url.rstrip("/") + "/export?format=csv"

            if "csv" in url.lower() or "export?format=csv" in url.lower():
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                source_name = "Remote CSV / Google Sheets"
                status_msg = f"Successfully fetched live data from remote CSV."
            else:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                json_data = response.json()
                
                if isinstance(json_data, dict) and "medicines" in json_data:
                    records = json_data["medicines"]
                elif isinstance(json_data, list):
                    records = json_data
                else:
                    raise ValueError("JSON payload must be a list of objects or contain 'medicines' array.")
                
                df = pd.DataFrame(records)
                source_name = "Remote JSON API"
                status_msg = f"Successfully fetched live data from remote JSON API."

        except Exception as e:
            is_fallback = True
            source_name = "Local Fallback"
            status_msg = f"⚠️ Remote fetch failed ({str(e)}). Falling back to local database.json."
            df = None

    if df is None or df.empty:
        db_path = os.path.join(os.path.dirname(__file__), "database.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    records = data.get("medicines", [])
                    df = pd.DataFrame(records)
                if not is_fallback:
                    status_msg = "Loaded successfully from local database.json"
            except Exception as e:
                df = pd.DataFrame(columns=REQUIRED_COLUMNS)
                status_msg = f"❌ Error reading local database.json: {e}"
        else:
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
            status_msg = f"❌ Local database.json missing at {db_path}"

    df = standardize_dataframe(df)
    return df, source_name, status_msg, is_fallback


@st.cache_resource
def get_ocr_reader():
    """Cache and initialize EasyOCR reader instance."""
    if EASYOCR_AVAILABLE:
        try:
            return easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            st.error(f"Failed to initialize EasyOCR: {e}")
            return None
    return None


# ==========================================
# OPENCV BARCODE & QR DETECTION ENGINES
# ==========================================

def scan_barcode(image: Image.Image):
    """
    Detects 1D Barcodes and 2D QR Codes using OpenCV built-in detectors:
      1. cv2.barcode.BarcodeDetector() for 1D barcodes.
      2. cv2.QRCodeDetector() for QR codes.

    Completely removes pyzbar/zbar dependencies.
    """
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    barcodes_found = []
    draw_img = image.convert('RGB').copy()
    draw = ImageDraw.Draw(draw_img)

    # 1. 1D Barcode Detection via OpenCV cv2.barcode.BarcodeDetector
    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        try:
            barcode_detector = cv2.barcode.BarcodeDetector()
            res = barcode_detector.detectAndDecode(gray)
            
            if len(res) == 4:
                ok, decoded_info, decoded_type, points = res
            else:
                decoded_info, decoded_type, points = res
                ok = bool(decoded_info)

            if decoded_info:
                if isinstance(decoded_info, str):
                    decoded_info = [decoded_info]
                    decoded_type = [decoded_type] if decoded_type else ["BARCODE"]

                for i, info in enumerate(decoded_info):
                    info_str = str(info).strip()
                    if info_str:
                        b_type = str(decoded_type[i]) if i < len(decoded_type) and decoded_type[i] else "BARCODE"
                        
                        # Draw bounding line polygons if points array available
                        if points is not None and i < len(points):
                            pts_curr = points[i]
                            pts_list = [(int(p[0]), int(p[1])) for p in pts_curr]
                            pts_list.append(pts_list[0])
                            draw.line(pts_list, fill='#00FF66', width=4)

                        barcodes_found.append({
                            'data': info_str,
                            'type': b_type,
                            'rect': None
                        })
        except Exception:
            pass

    # 2. 2D QR Code Detection via OpenCV cv2.QRCodeDetector
    if hasattr(cv2, "QRCodeDetector"):
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(gray)
            if retval and decoded_info:
                for i, info in enumerate(decoded_info):
                    info_str = str(info).strip()
                    if info_str and not any(b['data'] == info_str for b in barcodes_found):
                        if points is not None and i < len(points):
                            pts_curr = points[i]
                            pts_list = [(int(p[0]), int(p[1])) for p in pts_curr]
                            pts_list.append(pts_list[0])
                            draw.line(pts_list, fill='#00FF66', width=4)

                        barcodes_found.append({
                            'data': info_str,
                            'type': 'QRCODE',
                            'rect': None
                        })
        except Exception:
            try:
                info_str, points, straight_qrcode = qr_detector.detectAndDecode(gray)
                if info_str and info_str.strip():
                    info_clean = info_str.strip()
                    if not any(b['data'] == info_clean for b in barcodes_found):
                        if points is not None:
                            pts_curr = points[0] if points.ndim == 3 else points
                            pts_list = [(int(p[0]), int(p[1])) for p in pts_curr]
                            pts_list.append(pts_list[0])
                            draw.line(pts_list, fill='#00FF66', width=4)

                        barcodes_found.append({
                            'data': info_clean,
                            'type': 'QRCODE',
                            'rect': None
                        })
            except Exception:
                pass

    return barcodes_found, draw_img


def preprocess_image_for_ocr(image: Image.Image) -> np.ndarray:
    """
    Lightweight, no-API image enhancement pipeline to improve OCR accuracy on
    real-world medicine photos (glare, low contrast, uneven lighting, slight blur).
    Pure OpenCV — no external service, so it never affects recognition coverage
    from the API side, only how cleanly EasyOCR can read the text.

    Steps: grayscale -> CLAHE local contrast boost -> denoise -> mild sharpen.
    """
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # CLAHE boosts local contrast — helps faded print / uneven lighting / glare
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)

    # Denoise while preserving edges — helps blurry or compressed camera photos
    denoised = cv2.fastNlMeansDenoising(contrasted, h=10)

    # Mild sharpening kernel to counter slight camera blur
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # EasyOCR expects a 3-channel image
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def scan_ocr_text(image: Image.Image):
    """
    Option B: Extract text using EasyOCR.

    Runs OCR on the original image first. If that pass returns too few
    confident tokens (a sign of glare/blur/low contrast — common with phone
    photos of medicine boxes), automatically retries once on an enhanced
    version of the same image and keeps whichever pass found more confident
    text. This adds at most one extra OCR pass, only when needed.
    """
    reader = get_ocr_reader()
    if not reader:
        return "", [], image

    img_np = np.array(image.convert('RGB'))

    def run_ocr(np_img):
        raw_results = reader.readtext(np_img)
        good = [r for r in raw_results if r[2] > 0.2]
        return raw_results, good

    results, good_results = run_ocr(img_np)

    # Fallback pass on enhanced image, only if the raw pass looks weak
    if len(good_results) < 2:
        try:
            enhanced_np = preprocess_image_for_ocr(image)
            enhanced_results, enhanced_good = run_ocr(enhanced_np)
            if len(enhanced_good) > len(good_results):
                results, good_results = enhanced_results, enhanced_good
        except Exception:
            pass  # preprocessing is best-effort; never block OCR on failure

    extracted_tokens = []
    draw_img = image.convert('RGB').copy()
    draw = ImageDraw.Draw(draw_img)
    
    for bbox, text, prob in results:
        if prob > 0.2:
            extracted_tokens.append(text)
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            pts.append(pts[0])
            draw.line(pts, fill='#38BDF8', width=2)
            
    raw_text_combined = " ".join(extracted_tokens)
    return raw_text_combined, results, draw_img


# ==========================================
# VERIFIED DATABASE MATCHING ENGINE (DATAFRAME)
# ==========================================

def match_medicine(barcodes: list, ocr_text: str, database_df: pd.DataFrame, fuzzy_threshold: int = 65):
    """
    Deterministic & Safe Matching Engine over standardized Pandas DataFrame.
    """
    if database_df is None or database_df.empty:
        return None, None, 0, "Medical Database is empty."

    # 1. Primary Exact Barcode Match
    for b in barcodes:
        code_str = str(b['data']).strip()
        matched_rows = database_df[database_df['barcode'] == code_str]
        if not matched_rows.empty:
            med_dict = matched_rows.iloc[0].to_dict()
            return med_dict, "BARCODE", 100, f"Exact Barcode Match ({b['type']}: {code_str})"

    # 2. Secondary Fuzzy OCR Text Match
    if not THEFUZZ_AVAILABLE or not ocr_text.strip():
        return None, None, 0, "No Barcode matched and OCR text was empty or fuzzy engine unavailable."

    clean_ocr = ocr_text.lower().strip()
    best_med = None
    best_score = 0
    best_reason = ""

    for _, row in database_df.iterrows():
        med = row.to_dict()
        scores = []
        
        keywords = med.get("keywords", [])
        if isinstance(keywords, list):
            for kw in keywords:
                kw_clean = str(kw).lower().strip()
                if kw_clean:
                    p_ratio = fuzz.partial_ratio(kw_clean, clean_ocr)
                    t_ratio = fuzz.token_set_ratio(kw_clean, clean_ocr)
                    max_kw = max(p_ratio, t_ratio)
                    scores.append((max_kw, f"Keyword '{kw}' (Score: {max_kw}%)"))
            
        drug_name = str(med.get("drug_name", "")).lower()
        if drug_name:
            dn_p = fuzz.partial_ratio(drug_name, clean_ocr)
            dn_t = fuzz.token_set_ratio(drug_name, clean_ocr)
            max_dn = max(dn_p, dn_t)
            scores.append((max_dn, f"Drug Name '{med['drug_name']}' (Score: {max_dn}%)"))

        active_ing = str(med.get("active_ingredient", "")).lower()
        if active_ing:
            ai_p = fuzz.partial_ratio(active_ing, clean_ocr)
            ai_t = fuzz.token_set_ratio(active_ing, clean_ocr)
            max_ai = max(ai_p, ai_t)
            scores.append((max_ai, f"Active Ingredient Match (Score: {max_ai}%)"))

        if scores:
            med_max_score, med_reason = max(scores, key=lambda x: x[0])
            if med_max_score > best_score:
                best_score = med_max_score
                best_med = med
                best_reason = med_reason

    if best_med and best_score >= fuzzy_threshold:
        return best_med, "FUZZY_OCR", best_score, f"Fuzzy Text Match via {best_reason}"

    return None, None, best_score, "No database record met the matching threshold."


def match_medicine_extended(barcodes: list, ocr_text: str, database_df: pd.DataFrame, fuzzy_threshold: int = 65):
    """
    Extended 4-tier matching pipeline to maximize recognition coverage:
      1. Local exact Barcode match        (match_medicine — BARCODE)
      2. Local fuzzy OCR match            (match_medicine — FUZZY_OCR)
      3. FDA NDC Directory barcode lookup (NEW — NDC_MATCH)   [when 1 & 2 fail]
      4. RxNav approximate name lookup    (NEW — RXNORM_MATCH) [when 1, 2 & 3 fail]

    Tiers 1-2 stay fully local/deterministic (no network). Tiers 3-4 only fire
    as a fallback, so the local verified database always takes priority.
    """
    med, mtype, conf, reason = match_medicine(barcodes, ocr_text, database_df, fuzzy_threshold)
    if med:
        return med, mtype, conf, reason

    # --- Tier 3: FDA NDC Directory fallback (via scanned barcode) ---
    for b in barcodes:
        ndc_med = lookup_ndc_by_barcode(b['data'])
        if ndc_med:
            return ndc_med, "NDC_MATCH", 90, f"FDA NDC Directory Match (Barcode: {b['data']})"

    # --- Tier 4: RxNorm approximate name fallback (via OCR tokens) ---
    if ocr_text and ocr_text.strip():
        words = [w for w in ocr_text.split() if len(w) >= 4 and w.isalpha()]
        for w in words[:5]:  # cap lookups to avoid excessive requests per scan
            approx = lookup_rxnorm_approximate(w)
            if approx:
                synth_med = {
                    "drug_name": approx["name"],
                    "active_ingredient": approx["name"],
                    "dosage": "Refer to official FDA label below.",
                    "uses": "Refer to official FDA label below.",
                    "contraindications": "Not available via RxNorm — see official FDA label for full contraindications.",
                    "barcode": "",
                    "keywords": [],
                }
                return (
                    synth_med, "RXNORM_MATCH", approx["score"],
                    f"RxNorm Approximate Match: OCR token '{w}' → '{approx['name']}' (RxCUI {approx['rxcui']}, Score: {approx['score']}%)"
                )

    return None, None, 0, "No match found in local database, FDA NDC Directory, or RxNorm."


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================

def main():
    # --- Splash / Intro Screen (shown once per browser session) ---
    if "splash_shown" not in st.session_state:
        st.session_state["splash_shown"] = False

    if not st.session_state["splash_shown"]:
        splash_placeholder = st.empty()
        with splash_placeholder.container():
            st.markdown(textwrap.dedent("""
                <div class="splash-screen">
                    <div class="splash-logo">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M12 2L4 5v6c0 5.25 3.4 9.74 8 11 4.6-1.26 8-5.75 8-11V5l-8-3z"
                                  stroke="#16A34A" stroke-width="1.5" stroke-linejoin="round" fill="rgba(22,163,74,0.08)"/>
                            <rect x="8.5" y="9.5" width="7" height="4" rx="2" transform="rotate(45 12 11.5)"
                                  stroke="#16A34A" stroke-width="1.4" fill="none"/>
                            <line x1="10.6" y1="9.6" x2="13.4" y2="13.4" stroke="#16A34A" stroke-width="1.4"/>
                        </svg>
                    </div>
                    <div class="splash-title">MedGuard</div>
                    <div class="splash-subtitle">Verified Medicine Verification</div>
                </div>
            """).strip(), unsafe_allow_html=True)
        time.sleep(1.9)  # matches the .splash-screen CSS animation duration
        splash_placeholder.empty()
        st.session_state["splash_shown"] = True

    # Header Banner
    st.markdown(textwrap.dedent("""
        <div class="header-container">
            <div class="header-content">
                <div class="header-icon">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L4 5v6c0 5.25 3.4 9.74 8 11 4.6-1.26 8-5.75 8-11V5l-8-3z"
                              stroke="#16A34A" stroke-width="1.5" stroke-linejoin="round" fill="rgba(22,163,74,0.08)"/>
                        <rect x="8.5" y="9.5" width="7" height="4" rx="2" transform="rotate(45 12 11.5)"
                              stroke="#16A34A" stroke-width="1.4" fill="none"/>
                        <line x1="10.6" y1="9.6" x2="13.4" y2="13.4" stroke="#16A34A" stroke-width="1.4"/>
                    </svg>
                </div>
                <div>
                    <div class="header-title">
                        MedGuard
                    </div>
                    <div class="header-subtitle">
                        Scan medicine packaging via OpenCV Barcode/QR Detector or Label Text OCR with OpenFDA live API integration.
                    </div>
                </div>
            </div>
        </div>
    """).strip(), unsafe_allow_html=True)

    # ── Language Selector ──────────────────────────────────────────────
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    # Sidebar
    with st.sidebar:
        st.markdown("### 🌍 Language / Ngôn ngữ")
        lang_choice = st.radio(
            label="",
            options=["🇬🇧 English", "🇻🇳 Tiếng Việt"],
            index=0 if st.session_state["lang"] == "en" else 1,
            horizontal=True,
            key="lang_radio",
            label_visibility="collapsed",
        )
        st.session_state["lang"] = "en" if lang_choice.startswith("🇬🇧") else "vi"
        lang = st.session_state["lang"]

        st.divider()
        st.header("🌐 Medical Database Source")
        
        remote_url_input = st.text_input(
            "Remote Data URL (Google Sheets CSV or JSON API):",
            placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv",
            help="Enter a public Google Sheets CSV export link or remote JSON API URL. Leave empty to use local database.json."
        )

        col_ref1, col_ref2 = st.columns([1, 1])
        with col_ref1:
            if st.button("🔄 Refresh Cache"):
                st.cache_data.clear()
                st.rerun()

        database_df, source_name, status_msg, is_fallback = load_medicine_database(source_url=remote_url_input)

        if is_fallback:
            st.warning(status_msg)
        else:
            if "Remote" in source_name:
                st.markdown(f'<span class="badge-source-remote">SOURCE: {source_name}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-source-local">SOURCE: {source_name}</span>', unsafe_allow_html=True)

        st.caption(f"Cache TTL: 3600 seconds (1 hour)")

        st.divider()
        st.header("⚙️ Engine Status")
        
        # Display OpenCV Barcode Engine Status
        st.success("✅ **OpenCV Barcode Engine**: Ready")

        if EASYOCR_AVAILABLE:
            st.success("✅ **EasyOCR Text Engine**: Ready (CPU)")
        else:
            st.error(f"❌ **EasyOCR**: Unavailable ({EASYOCR_ERROR})")

        if THEFUZZ_AVAILABLE:
            st.success("✅ **Fuzzywuzzy Engine**: Ready")
        else:
            st.error("❌ **thefuzz**: Missing")

        st.success("✅ **OpenFDA API**: Active (Cached)")

        st.divider()
        st.header("📚 Active Dataset")
        st.metric("Indexed Medicines", len(database_df))
        
        st.markdown("""
        **Safety Rules & Guarantees:**
        - 🔒 **Zero Generative AI**: Verified database & official OpenFDA records only.
        - ⚡ **OpenCV Powered**: Built-in 1D & 2D QR Barcode detection without external zbar DLL dependencies.
        - 🛡️ **Unknown Alert**: Instant warning for unverified packaging.
        """)

        st.divider()
        st.caption("MedGuard v1.3 • Streamlit + OpenCV + EasyOCR + OpenFDA")

    # Main Tabs
    tab_scan, tab_camera, tab_search = st.tabs([
        "🖼️ Scan Uploaded Image", 
        "📷 Live Camera Barcode", 
        "🔍 Manual Search & OpenFDA"
    ])

    # ----------------------------------------------------
    # TAB 1: UPLOAD & SCAN IMAGE
    # ----------------------------------------------------
    with tab_scan:
        st.subheader("Upload Packaging or Label Image")
        uploaded_file = st.file_uploader(
            "Select a medicine package image (JPG, PNG, JPEG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear picture of the barcode or active ingredient text."
        )

        if uploaded_file:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(img, caption="Original Uploaded Image", use_container_width=True)

                with st.spinner("🔍 Running OpenCV Barcode Engine & Text OCR..."):
                    barcodes, barcode_annotated_img = scan_barcode(img)
                    ocr_text, ocr_results, ocr_annotated_img = scan_ocr_text(img)

                    matched_med, match_type, confidence, match_reason = match_medicine_extended(
                        barcodes=barcodes,
                        ocr_text=ocr_text,
                        database_df=database_df
                    )

                    brand_query = ""
                    generic_query = ""
                    if matched_med:
                        brand_query = extract_brand_name(matched_med.get('drug_name', ''))
                        generic_query = extract_generic_name(matched_med.get('active_ingredient', ''))
                    elif ocr_text:
                        words = [w for w in ocr_text.split() if len(w) >= 4 and w.isalpha()]
                        if words:
                            brand_query = words[0]
                            generic_query = words[0]

                    external_data = fetch_all_external_apis(brand_query, generic_query)

                with col2:
                    st.subheader("🎯 Visual Detection Overlay")
                    if barcodes:
                        st.image(barcode_annotated_img, caption="✅ OpenCV Barcode / QR Detected (Green)", use_container_width=True)
                    else:
                        st.image(ocr_annotated_img, caption="🔤 OCR Text Regions Highlighted (Blue)", use_container_width=True)

                st.divider()

                render_verification_results(
                    matched_med=matched_med,
                    match_type=match_type,
                    confidence=confidence,
                    match_reason=match_reason,
                    barcodes=barcodes,
                    ocr_text=ocr_text,
                    external_data=external_data,
                    lang=lang,
                )

            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

    # ----------------------------------------------------
    # TAB 2: LIVE CAMERA BARCODE SCAN
    # ----------------------------------------------------
    with tab_camera:
        st.subheader("Take Photo with Camera")
        camera_img = st.camera_input("Position the medicine barcode or label in clear view:")

        if camera_img:
            try:
                img = Image.open(camera_img).convert("RGB")

                with st.spinner("⚡ Decoding barcode, label text, & OpenFDA..."):
                    barcodes, barcode_annotated_img = scan_barcode(img)
                    ocr_text, ocr_results, ocr_annotated_img = scan_ocr_text(img)

                    matched_med, match_type, confidence, match_reason = match_medicine_extended(
                        barcodes=barcodes,
                        ocr_text=ocr_text,
                        database_df=database_df
                    )

                    brand_query = ""
                    generic_query = ""
                    if matched_med:
                        brand_query = extract_brand_name(matched_med.get('drug_name', ''))
                        generic_query = extract_generic_name(matched_med.get('active_ingredient', ''))
                    elif ocr_text:
                        words = [w for w in ocr_text.split() if len(w) >= 4 and w.isalpha()]
                        if words:
                            brand_query = words[0]
                            generic_query = words[0]

                    external_data = fetch_all_external_apis(brand_query, generic_query)

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(barcode_annotated_img if barcodes else ocr_annotated_img, 
                             caption="Detected Bounding Regions", 
                             use_container_width=True)
                
                with col2:
                    st.markdown("#### Detected Raw Elements:")
                    if barcodes:
                        for b in barcodes:
                            st.info(f"🏷️ **Barcode ({b['type']})**: `{b['data']}`")
                    else:
                        st.caption("No barcode detected in camera snapshot.")

                    if ocr_text:
                        with st.expander("📄 View Raw Extracted OCR Text"):
                            st.code(ocr_text, language="text")

                st.divider()

                render_verification_results(
                    matched_med=matched_med,
                    match_type=match_type,
                    confidence=confidence,
                    match_reason=match_reason,
                    barcodes=barcodes,
                    ocr_text=ocr_text,
                    external_data=external_data,
                    lang=lang,
                )

            except Exception as e:
                st.error(f"❌ Camera processing error: {e}")

    # ----------------------------------------------------
    # TAB 3: MANUAL SEARCH & OPENFDA DIRECT LOOKUP
    # ----------------------------------------------------
    with tab_search:
        st.subheader("🔍 Local Database & OpenFDA Live Search")
        
        search_query = st.text_input(
            "Search by Drug Name, Active Ingredient, Barcode, or Keyword:",
            placeholder="e.g. Paracetamol, Amoxicillin, Ibuprofen"
        ).strip().lower()

        if search_query:
            with st.spinner("🌐 Querying public drug databases in parallel..."):
                external_direct = fetch_all_external_apis(search_query, search_query)
            
            render_external_data_cards(external_direct)

        st.divider()
        st.markdown("#### 📚 Local / Remote Verified Records")

        filtered_rows = []
        if not database_df.empty:
            for _, row in database_df.iterrows():
                med = row.to_dict()
                if not search_query:
                    filtered_rows.append(med)
                else:
                    searchable_str = f"{med['drug_name']} {med['active_ingredient']} {med['barcode']} {' '.join(med.get('keywords', []))}".lower()
                    if search_query in searchable_str or (THEFUZZ_AVAILABLE and fuzz.partial_ratio(search_query, searchable_str) >= 70):
                        filtered_rows.append(med)

        st.caption(f"Showing {len(filtered_rows)} of {len(database_df)} dataset entries")

        for med in filtered_rows:
            with st.expander(f"💊 **{med['drug_name']}** — Active Ingredient: *{med['active_ingredient']}*", expanded=True if search_query else False):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"**Barcode**: `{med['barcode']}`")
                    st.markdown(f"**Keywords**: `{', '.join(med.get('keywords', []))}`")
                with col_b:
                    st.markdown(f"**Dosage**: {med['dosage']}")
                    st.markdown(f"**Uses**: {med['uses']}")
                    st.markdown(f"**Contraindications**: {med['contraindications']}")

        st.divider()
        with st.expander("📊 View Standardized Pandas DataFrame Table"):
            st.dataframe(database_df, use_container_width=True)


# ==========================================
# RENDER VERIFICATION RESULTS UI
# ==========================================

def render_verification_results(matched_med, match_type, confidence, match_reason, barcodes, ocr_text, external_data=None, lang="en"):
    """
    Renders structured medical details if matched, or an 'Unknown Medicine' warning if unmatched,
    plus OpenFDA and other API live results if found.
    """
    if matched_med:
        badge_by_type = {
            "BARCODE": f'<span class="badge-barcode">MATCH TYPE: EXACT BARCODE ({confidence}%)</span>',
            "FUZZY_OCR": f'<span class="badge-fuzzy">MATCH TYPE: FUZZY OCR MATCH ({confidence}%)</span>',
            "NDC_MATCH": f'<span class="badge-source-remote">MATCH TYPE: FDA NDC DIRECTORY ({confidence}%)</span>',
            "RXNORM_MATCH": f'<span class="badge-fuzzy">MATCH TYPE: RXNORM APPROX. MATCH ({confidence}%)</span>',
        }
        badge_html = badge_by_type.get(match_type, f'<span class="badge-fuzzy">MATCH TYPE: {match_type} ({confidence}%)</span>')

        kw_str = ", ".join(matched_med.get("keywords", [])) if isinstance(matched_med.get("keywords"), list) else str(matched_med.get("keywords", ""))

        html = f"""
        <div class="verified-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #15803D; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                    <svg class="section-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L4 5v6c0 5.25 3.4 9.74 8 11 4.6-1.26 8-5.75 8-11V5l-8-3z" stroke="#15803D" stroke-width="1.6" stroke-linejoin="round"/>
                        <path d="M8.5 12l2.3 2.3L15.5 9.5" stroke="#15803D" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    VERIFIED DATABASE RECORD
                </span>
                {badge_html}
            </div>
            <div class="med-title">{matched_med['drug_name']}</div>
            <div style="font-size: 0.95rem; color: #6EE7B7; font-weight: 500; margin-bottom: 1rem;">
                Match Details: {match_reason}
            </div>
            <div class="field-label">🧪 Active Ingredient(s)</div>
            <div class="field-value">{matched_med['active_ingredient']}</div>
            <div class="field-label">📋 Recommended Dosage</div>
            <div class="field-value">{matched_med['dosage']}</div>
            <div class="field-label">🎯 Primary Uses &amp; Indications</div>
            <div class="field-value">{matched_med['uses']}</div>
            <div class="field-label">⚠️ Contraindications &amp; Safety Warnings</div>
            <div class="field-value" style="border-left: 4px solid #F59E0B; background: rgba(245, 158, 11, 0.1);">
                {matched_med['contraindications']}
            </div>
            <div style="margin-top: 1rem; font-size: 0.85rem; color: #9CA3AF;">
                <strong>Verified Barcode:</strong> <code>{matched_med['barcode']}</code> | <strong>Keywords:</strong> <code>{kw_str}</code>
            </div>
        </div>
        """
        render_html_safely(html)

    else:
        html = """
        <div class="warning-card">
            <div style="font-weight: 700; font-size: 1.25rem; color: #B91C1C; display: flex; align-items: center; gap: 8px;">
                <svg class="section-icon" style="width:22px;height:22px;" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 3L2 20h20L12 3z" stroke="#B91C1C" stroke-width="1.6" stroke-linejoin="round"/>
                    <line x1="12" y1="9.5" x2="12" y2="14" stroke="#B91C1C" stroke-width="1.6" stroke-linecap="round"/>
                    <circle cx="12" cy="17" r="0.9" fill="#B91C1C"/>
                </svg>
                UNKNOWN MEDICINE - NOT FOUND IN LOCAL/REMOTE DATABASE
            </div>
            <p style="margin-top: 0.5rem; font-size: 0.95rem; color: #FECACA;">
                No matching barcode or active ingredient was found in the verified medical database.
            </p>
        </div>
        """
        render_html_safely(html)
        
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown("##### 🏷️ Detected Barcode Raw Output:")
            if barcodes:
                for b in barcodes:
                    st.code(f"Type: {b['type']}\nData: {b['data']}", language="text")
            else:
                st.info("No barcode detected in image.")
                
        with col_u2:
            st.markdown("##### 🔤 Detected OCR Raw Text:")
            if ocr_text.strip():
                render_html_safely(f'<div class="raw-box">{ocr_text}</div>')
            else:
                st.info("No readable text extracted by OCR.")

    # --- MIL Feature 1: Claim Audit ---
    if ocr_text and ocr_text.strip():
        flagged = run_claim_audit(ocr_text)
        if flagged:
            s = MIL_STRINGS[lang]
            phrases_str = ", ".join([f"`{p}`" for p in flagged])
            st.warning(
                s["claim_audit_title"] + ": "
                + s["claim_audit_body"].format(phrases=phrases_str)
            )

    # --- MIL Feature 2: Credibility Score Badge ---
    render_credibility_badge(match_type, external_data, lang=lang)

    # --- MIL Feature 3: Recall Alert ---
    recall_kw = ""
    if matched_med and matched_med.get("drug_name"):
        recall_kw = matched_med["drug_name"].split("/")[0].split()[0]
    elif barcodes:
        recall_kw = barcodes[0]["data"][:30]
    render_recall_alert(recall_kw, lang=lang)

    # --- MIL Feature 4: Cross-Check Table ---
    if matched_med or (external_data and external_data.get("openfda")):
        render_cross_check_table(matched_med, ocr_text, external_data, lang=lang)

    if external_data:
        render_external_data_cards(external_data)

    # --- MIL Feature 5: Educational Guide ---
    render_mil_guide(lang=lang)


def render_rxnav_card(rxnav_data):
    """Renders the RxNav API drug identifiers card."""
    if not rxnav_data:
        return

    badges = "".join([f'<span style="background-color: #F3E8FF; color: #7E22CE; font-size: 0.85rem; padding: 4px 10px; border-radius: 6px; font-weight: 600; border: 1px solid #D8B4FE; margin-right: 6px; display: inline-block; margin-bottom: 6px;">CUI: {cui}</span>' for cui in rxnav_data])

    html = f"""
    <div style="background: linear-gradient(145deg, #FAF5FF 0%, #F3E8FF 100%); border: 1px solid #D8B4FE; border-radius: 16px; padding: 1.5rem; color: #581C87; margin-top: 1.25rem; box-shadow: 0 10px 20px rgba(168, 85, 247, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: #7E22CE; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                🧬 RXNAV RXNORM DRUG LOOKUP
            </span>
            <span style="background-color: #7E22CE; color: white; font-size: 0.75rem; padding: 3px 10px; border-radius: 9999px; font-weight: 600;">
                NLM API
            </span>
        </div>
        <div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #7E22CE; margin-bottom: 0.5rem;">
            RxNorm Concept Unique Identifiers (RxCUI)
        </div>
        <div>
            {badges}
        </div>
    </div>
    """
    render_html_safely(html)


def render_dailymed_card(dailymed_data):
    """Renders the DailyMed package inserts card."""
    if not dailymed_data:
        return

    links_html = ""
    for item in dailymed_data[:5]:  # Show top 5 records
        links_html += f"""
        <div style="margin-bottom: 0.75rem; padding: 0.6rem 0.8rem; background: rgba(255, 255, 255, 0.7); border-radius: 8px; border: 1px solid rgba(22, 163, 74, 0.15);">
            <div style="font-weight: 600; font-size: 0.95rem; color: #14532D;">{item['title']}</div>
            <div style="font-size: 0.8rem; color: #4B5563; margin-top: 0.25rem;">
                SPL ID: <code style="color: #15803D;">{item['spl_id']}</code>
            </div>
            <div style="margin-top: 0.4rem;">
                <a href="{item['link']}" target="_blank" style="color: #15803D; font-weight: 600; text-decoration: none; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px;">
                    🔗 View Official Package Insert &rarr;
                </a>
            </div>
        </div>
        """

    html = f"""
    <div style="background: linear-gradient(145deg, #F0FDF4 0%, #DCFCE7 100%); border: 1px solid #4ADE80; border-radius: 16px; padding: 1.5rem; color: #14532D; margin-top: 1.25rem; box-shadow: 0 10px 20px rgba(22, 163, 74, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: #15803D; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                📦 DAILYMED OFFICIAL FDA LABELS
            </span>
            <span style="background-color: #15803D; color: white; font-size: 0.75rem; padding: 3px 10px; border-radius: 9999px; font-weight: 600;">
                DAILYMED API
            </span>
        </div>
        <div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #15803D; margin-bottom: 0.5rem;">
            Package Inserts &amp; Product Details
        </div>
        {links_html}
    </div>
    """
    render_html_safely(html)


def render_wikipedia_card(wiki_data, lang: str = "vi"):
    """Renders a Wikipedia summary card. `lang` controls the badge/title only —
    pass the matching wiki_data (Vietnamese or English) for the language shown."""
    if not wiki_data:
        return

    label = "WIKIPEDIA VIETNAM SUMMARY" if lang == "vi" else "WIKIPEDIA ENGLISH SUMMARY"
    icon = "📝" if lang == "vi" else "📄"

    html = f"""
    <div style="background: linear-gradient(145deg, #F8FAFC 0%, #F1F5F9 100%); border: 1px solid #94A3B8; border-radius: 16px; padding: 1.5rem; color: #1E293B; margin-top: 1.25rem; box-shadow: 0 10px 20px rgba(100, 116, 139, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: #334155; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                {icon} {label}
            </span>
            <span style="background-color: #475569; color: white; font-size: 0.75rem; padding: 3px 10px; border-radius: 9999px; font-weight: 600;">
                WIKIPEDIA API
            </span>
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;">
            {wiki_data['title']}
        </div>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #334155; background: rgba(255,255,255,0.6); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(0,0,0,0.05); text-align: justify;">
            {wiki_data['extract']}
        </div>
    </div>
    """
    render_html_safely(html)


def render_openfda_card(openfda_data):
    """Renders the OpenFDA live API drug label card."""
    if not openfda_data:
        return

    html = f"""
    <div style="background: linear-gradient(145deg, #F0F9FF 0%, #E0F2FE 100%); border: 1px solid #38BDF8; border-radius: 16px; padding: 1.5rem; color: #0C4A6E; margin-top: 1.25rem; box-shadow: 0 10px 20px rgba(56, 189, 248, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #0284C7; font-weight: 700; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                🌐 OPENFDA PUBLIC DRUG LABEL MATCH
            </span>
            <span style="background-color: #0284C7; color: white; font-size: 0.75rem; padding: 3px 10px; border-radius: 9999px; font-weight: 600;">
                LIVE FDA API
            </span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #0C4A6E;">
            {openfda_data['brand_name']}
            <span style="font-size: 1rem; color: #64748B; font-weight: 400;">{f"({openfda_data['generic_name']})" if openfda_data.get('generic_name') else ""}</span>
        </div>
        <div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #0369A1; margin-top: 1rem; margin-bottom: 0.25rem;">
            🧪 Active Ingredient(s)
        </div>
        <div style="font-size: 0.95rem; background: rgba(255,255,255,0.65); padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid rgba(3,105,161,0.12);">
            {openfda_data['active_ingredient']}
        </div>
        <div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #0369A1; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            🎯 Indications &amp; Usage
        </div>
        <div style="font-size: 0.95rem; background: rgba(255,255,255,0.65); padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid rgba(3,105,161,0.12);">
            {openfda_data['usage']}
        </div>
        <div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #0369A1; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            📋 Dosage &amp; Administration
        </div>
        <div style="font-size: 0.95rem; background: rgba(255,255,255,0.65); padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid rgba(3,105,161,0.12);">
            {openfda_data['dosage']}
        </div>
        <div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #B91C1C; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            ⚠️ FDA Warnings &amp; Precautions
        </div>
        <div style="font-size: 0.95rem; background: rgba(239, 68, 68, 0.08); padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.2); color: #7F1D1D;">
            {openfda_data['warnings']}
        </div>
        {f'''<div style="font-weight: 600; text-transform: uppercase; font-size: 0.75rem; color: #C2410C; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            💊 Drug Interactions (Official FDA Label)
        </div>
        <div style="font-size: 0.95rem; background: rgba(249, 115, 22, 0.08); padding: 0.6rem 0.8rem; border-radius: 6px; border: 1px solid rgba(249, 115, 22, 0.25); color: #9A3412;">
            {openfda_data['interactions']}
        </div>''' if openfda_data.get('interactions') else ''}
    </div>
    """
    render_html_safely(html)


def render_external_data_cards(external_data):
    """Renders the combined output of all external APIs in a clean 2-column layout."""
    if not external_data:
        return

    has_data = any(external_data.values())
    if not has_data:
        st.info("ℹ️ No additional information found in public drug lookup APIs.")
        return

    st.markdown("### 🌐 Public Drug Reference & Lookup Results")

    ui_lang = st.session_state.get("lang", "en")

    col1, col2 = st.columns(2)
    with col1:
        if external_data.get("openfda"):
            render_openfda_card(external_data["openfda"])
        if external_data.get("dailymed"):
            render_dailymed_card(external_data["dailymed"])
    with col2:
        # Show only the Wikipedia card matching the app's current language
        # toggle (🇬🇧/🇻🇳) — never both at once. If the selected language has
        # no article, fall back to the other language rather than showing nothing.
        wiki_selected = external_data.get("wikipedia_en") if ui_lang == "en" else external_data.get("wikipedia")
        wiki_other = external_data.get("wikipedia") if ui_lang == "en" else external_data.get("wikipedia_en")
        other_lang = "vi" if ui_lang == "en" else "en"

        if wiki_selected:
            render_wikipedia_card(wiki_selected, lang=ui_lang)
        elif wiki_other:
            render_wikipedia_card(wiki_other, lang=other_lang)

        if external_data.get("rxnav"):
            render_rxnav_card(external_data["rxnav"])


if __name__ == "__main__":
    main()
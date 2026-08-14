import os
import io
import json
import textwrap
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
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

# Custom-drawn, tileable medicine-themed background pattern.
_MED_BG_SVG = """<svg xmlns='http://www.w3.org/2000/svg' width='220' height='220'>
  <g fill='none' stroke='#20B26C' stroke-width='1.2' stroke-opacity='0.08'>
    <g transform='translate(20,25) rotate(35)'>
      <rect x='0' y='0' width='46' height='18' rx='9'/>
      <line x1='23' y1='0' x2='23' y2='18'/>
    </g>
    <circle cx='165' cy='45' r='16'/>
    <line x1='153' y1='45' x2='177' y2='45'/>
    <g transform='translate(120,120) rotate(-20)'>
      <rect x='0' y='0' width='34' height='10' rx='2'/>
      <line x1='34' y1='2.5' x2='44' y2='2.5'/>
      <line x1='34' y1='7.5' x2='44' y2='7.5'/>
      <line x1='0' y1='5' x2='-8' y2='5'/>
    </g>
    <g transform='translate(35,150)'>
      <rect x='7' y='0' width='8' height='24' rx='2'/>
      <rect x='0' y='7' width='22' height='8' rx='2'/>
    </g>
  </g>
</svg>"""
_MED_BG_DATA_URI = "data:image/svg+xml," + urllib.parse.quote(_MED_BG_SVG)

_CSS_TEMPLATE = textwrap.dedent("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #1F2937;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #EBF8F5;
            background-image:
                radial-gradient(circle at 10% 10%, rgba(32, 178, 108, 0.12) 0%, transparent 50%),
                radial-gradient(circle at 90% 85%, rgba(27, 122, 120, 0.08) 0%, transparent 50%),
                url("__MED_BG_DATA_URI__");
            background-repeat: no-repeat, no-repeat, repeat;
            background-size: auto, auto, 220px 220px;
            background-attachment: fixed;
        }

        [data-testid="stSidebar"] {
            background-color: #F3FCFA;
            border-right: 1px solid #D1EBE3;
        }

        .header-container {
            position: relative;
            background: linear-gradient(135deg, #FFFFFF 0%, #EBF8F5 55%, #D6F5EC 100%);
            padding: 2.25rem 2rem;
            border-radius: 20px;
            color: #114B4E;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px -5px rgba(32, 178, 108, 0.12);
            border: 1px solid #BCEAD9;
            overflow: hidden;
            text-align: center;
        }
        .header-container::before {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(32, 178, 108, 0.18) 0%, rgba(32, 178, 108, 0) 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .header-content {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background-color: #D6F5EC;
            color: #176B6B;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 6px 18px;
            border-radius: 9999px;
            border: 1px solid #A3E4D1;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
            color: #176B6B;
            letter-spacing: -0.02em;
        }
        .header-subtitle {
            color: #4A6363;
            font-size: 1.05rem;
            max-width: 720px;
            margin: 0 auto;
            font-weight: 500;
            line-height: 1.6;
        }
        .section-icon {
            width: 18px;
            height: 18px;
            vertical-align: -3px;
            margin-right: 4px;
            opacity: 0.9;
        }

        .badge-barcode {
            background-color: #E0F2FE;
            color: #0369A1;
            font-size: 0.85rem;
            padding: 4px 14px;
            border-radius: 9999px;
            font-weight: 700;
            border: 1px solid #BAE6FD;
        }
        .badge-fuzzy {
            background-color: #F3E8FF;
            color: #7E22CE;
            font-size: 0.85rem;
            padding: 4px 14px;
            border-radius: 9999px;
            font-weight: 700;
            border: 1px solid #E9D5FF;
        }
        .badge-source-remote {
            background-color: #D6F5EC;
            color: #176B6B;
            font-size: 0.8rem;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
        }
        .badge-source-local {
            background-color: #F1F5F9;
            color: #334155;
            font-size: 0.8rem;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
        }

        .verified-card {
            background: linear-gradient(145deg, #FFFFFF 0%, #F0FAF7 100%);
            border: 1.5px solid #20B26C;
            border-radius: 18px;
            padding: 1.75rem;
            color: #135253;
            margin-top: 1rem;
            box-shadow: 0 12px 28px rgba(32, 178, 108, 0.10);
        }
        .warning-card {
            background: linear-gradient(145deg, #FEF2F2 0%, #FEE2E2 100%);
            border: 1.5px solid #FCA5A5;
            border-radius: 18px;
            padding: 1.75rem;
            color: #7F1D1D;
            margin-top: 1rem;
            box-shadow: 0 12px 28px rgba(220, 38, 38, 0.08);
        }
        .med-title {
            font-size: 1.85rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
            color: #176B6B;
        }
        .field-label {
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.75rem;
            color: #1A7A68;
            margin-top: 1rem;
            margin-bottom: 0.35rem;
        }
        .field-value {
            font-size: 1rem;
            line-height: 1.55;
            color: #1F2937;
            background: #FFFFFF;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            border: 1px solid #D1EBE3;
        }

        .disclaimer-banner {
            background-color: #FFFBEB;
            border: 1.5px solid #FDE68A;
            border-radius: 14px;
            padding: 1rem 1.5rem;
            margin-top: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            color: #92400E;
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.06);
        }
        .disclaimer-icon {
            font-size: 1.5rem;
            flex-shrink: 0;
        }
        .disclaimer-text {
            font-size: 0.88rem;
            line-height: 1.5;
            color: #B45309;
        }
        .disclaimer-text strong {
            color: #D97706;
            font-weight: 700;
        }

        /* Pill-shaped segmented tab control (reference-inspired) */
        [data-testid="stTabs"] div[data-baseweb="tab-list"] {
            background-color: #F3FCFA;
            border-radius: 9999px;
            padding: 6px;
            gap: 4px;
            border: 1px solid #D1EBE3;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            border-radius: 9999px;
            padding: 10px 22px;
            font-weight: 700;
            color: #4A6363;
            background-color: transparent;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #FFFFFF;
            color: #176B6B;
            box-shadow: 0 2px 10px rgba(32, 178, 108, 0.18);
        }
        [data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
            display: none;
        }
        [data-testid="stTabs"] div[data-baseweb="tab-border"] {
            display: none;
        }

        /* Pill-shaped buttons */
        .stButton > button {
            border-radius: 9999px !important;
            background-color: #20B26C !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 0.55rem 1.5rem !important;
            box-shadow: 0 4px 12px rgba(32, 178, 108, 0.25) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(32, 178, 108, 0.35) !important;
            color: #FFFFFF !important;
        }
        [data-testid="stFileUploaderDropzone"] button {
            border-radius: 9999px !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: #20B26C !important;
            font-weight: 800 !important;
        }

        /* Sidebar small-caps section eyebrow labels (nav grouping look) */
        .sidebar-eyebrow {
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #6B8F89;
            margin-top: 0.25rem;
            margin-bottom: -0.6rem;
        }

        .raw-box {
            background-color: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 10px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.9rem;
            color: #334155;
            white-space: pre-wrap;
            max-height: 250px;
            overflow-y: auto;
        }

        .splash-screen {
            position: fixed;
            inset: 0;
            z-index: 9999;
            background: linear-gradient(135deg, #FFFFFF 0%, #EBF8F5 60%, #D6F5EC 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            animation: splashFadeOut 1.9s ease forwards;
        }
        .splash-logo {
            width: 96px;
            height: 96px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #20B26C;
            border-radius: 24px;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 24px rgba(32, 178, 108, 0.3);
            animation: splashPulse 1.9s ease forwards;
        }
        .splash-logo svg {
            width: 52px;
            height: 52px;
            stroke: #FFFFFF;
        }
        .splash-title {
            font-size: 2.85rem;
            font-weight: 800;
            color: #176B6B;
            letter-spacing: -0.02em;
        }
        .splash-subtitle {
            color: #4A6363;
            font-size: 1.05rem;
            margin-top: 0.5rem;
            font-weight: 500;
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
# OPENFDA API INTEGRATION
# ==========================================

@st.cache_data(ttl=3600)
def search_openfda(query: str):
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
    usage = " ".join([str(u).strip() for u in usage_list if str(u).strip()]) if isinstance(usage_list, list) else str(usage_list).strip()

    dosage_list = item.get("dosage_and_administration", [])
    dosage = " ".join([str(d).strip() for d in dosage_list if str(d).strip()]) if isinstance(dosage_list, list) else str(dosage_list).strip()

    warnings_list = item.get("warnings", item.get("warnings_and_cautions", []))
    warnings = " ".join([str(w).strip() for w in warnings_list if str(w).strip()]) if isinstance(warnings_list, list) else str(warnings_list).strip()

    interactions_list = item.get("drug_interactions", [])
    interactions = " ".join([str(i).strip() for i in interactions_list if str(i).strip()]) if isinstance(interactions_list, list) else str(interactions_list).strip()

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
        "interactions": format_text(interactions, "", 400),
    }


@st.cache_data(ttl=86400)
def lookup_ndc_by_barcode(barcode: str):
    if not barcode or not str(barcode).strip():
        return None

    raw = str(barcode).strip()
    digits = "".join(ch for ch in raw if ch.isdigit())

    candidates = [raw]
    if len(digits) >= 11:
        candidates.append(digits[-11:])
    if len(digits) >= 10:
        candidates.append(digits[-10:])
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
    first_part = drug_name_str.split("/")[0]
    words = [w for w in first_part.split() if len(w) >= 3 and w.isalpha()]
    return words[0] if words else first_part.split()[0]


def extract_generic_name(active_ingredient_str: str) -> str:
    if not active_ingredient_str:
        return ""
    first_part = active_ingredient_str.split(",")[0]
    words = [w for w in first_part.split() if len(w) >= 3 and w.isalpha()]
    return words[0] if words else first_part.split()[0]


@st.cache_data(ttl=86400)
def lookup_rxnav(keyword: str):
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
    if not keyword or not str(keyword).strip():
        return None
    clean_keyword = str(keyword).strip()
    url = f"https://vi.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(clean_keyword)}&format=json"
    headers = {"User-Agent": "MedGuard/1.3 (contact@example.com) Python-requests/2.31"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "").strip()
                    if extract:
                        return {"title": page_data.get("title", clean_keyword), "extract": extract}
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400)
def lookup_wikipedia_english(keyword: str):
    if not keyword or not str(keyword).strip():
        return None
    clean_keyword = str(keyword).strip()
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={urllib.parse.quote(clean_keyword)}&format=json"
    headers = {"User-Agent": "MedGuard/1.3 (contact@example.com) Python-requests/2.31"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    extract = page_data.get("extract", "").strip()
                    if extract:
                        return {"title": page_data.get("title", clean_keyword), "extract": extract}
    except Exception:
        pass
    return None


def fetch_all_external_apis(brand_query: str, generic_query: str):
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
        future_fda = executor.submit(search_openfda, brand_query or generic_query)
        future_rxnav = executor.submit(lookup_rxnav, generic_query or brand_query)
        future_dailymed = executor.submit(lookup_dailymed, brand_query or generic_query)
        future_wiki = executor.submit(lookup_wikipedia_vietnam, generic_query or brand_query)
        future_wiki_en = executor.submit(lookup_wikipedia_english, generic_query or brand_query)
        
        results["openfda"] = future_fda.result()
        results["rxnav"] = future_rxnav.result()
        results["dailymed"] = future_dailymed.result()
        results["wikipedia"] = future_wiki.result()
        results["wikipedia_en"] = future_wiki_en.result()

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
# MIL — BILINGUAL STRINGS & AUDIT
# ==========================================

MIL_STRINGS = {
    "en": {
        "claim_audit_title": "⚠️ **Claim Audit Alert**",
        "claim_audit_body": "Suspicious / exaggerated phrases detected in scanned text: {phrases}  \nPlease verify through official sources before use.",
        "cred_label": "CREDIBILITY",
        "cred_high_desc": "Verified by official FDA / DailyMed or exact Barcode match",
        "cred_med_desc": "Partial match via fuzzy OCR + Wikipedia reference",
        "cred_low_desc": "Unverified — OCR only, no official API confirmation",
        "cred_high": "HIGH",
        "cred_med": "MEDIUM",
        "cred_low": "LOW",
        "recall_title": "🚨 **FDA RECALL NOTICE FOUND**",
        "recall_firm": "Firm",
        "recall_reason": "Reason",
        "recall_status": "Status",
        "recall_date": "Date",
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
        "disclaimer_title": "Medical Disclaimer:",
        "disclaimer_body": "All medical information and verification results provided on this application are for reference only and cannot replace professional doctors' diagnosis or treatment. When making medical decisions such as disease diagnosis or medication, be sure to consult a professional doctor.",
        "guide_title": "💡 MIL Guide: 4 Steps to Spot Fake / Unverified Medicines",
        "guide_body": """
**Step 1 — Check the Registration Code 📋**
Vietnamese medicines must carry a valid registration number printed on the label:
- `VD-XXXXX-XX` → Domestically produced medicine
- `VS-XXXXX-XX` → Traditional / herbal product
- `GC-XXXXX-XX` → Import permit
- `VN-XXXXX-XX` → Foreign-registered imported medicine

**Step 2 — Verify the Barcode Origin 🔢**
- **893** = Made in Vietnam | **000–019** = Made in USA | **400–440** = Made in Germany | **690–699** = Made in China

**Step 3 — Watch for Exaggerated Claims ⚠️**
- "100% cure", "miracle drug", "no side effects"

**Step 4 — Cross-Check with Official Sources 🌐**
- **OpenFDA**: [api.fda.gov](https://api.fda.gov) — U.S. FDA drug label database
- **DailyMed**: [dailymed.nlm.nih.gov](https://dailymed.nlm.nih.gov) — Official package inserts
        """,
    },
    "vi": {
        "claim_audit_title": "⚠️ **Cảnh báo kiểm tra tuyên bố**",
        "claim_audit_body": "Phát hiện cụm từ đáng ngờ / phóng đại trong văn bản quét: {phrases}  \nVui lòng xác minh qua nguồn chính thức trước khi sử dụng.",
        "cred_label": "ĐỘ TIN CẬY",
        "cred_high_desc": "Đã xác minh qua FDA / DailyMed chính thức hoặc khớp mã vạch chính xác",
        "cred_med_desc": "Khớp một phần qua OCR mờ + tham chiếu Wikipedia",
        "cred_low_desc": "Chưa xác minh — chỉ OCR, không có xác nhận API chính thức",
        "cred_high": "CAO",
        "cred_med": "TRUNG BÌNH",
        "cred_low": "THẤP",
        "recall_title": "🚨 **ĐÃ TÌM THẤY THÔNG BÁO THU HỒI FDA**",
        "recall_firm": "Công ty",
        "recall_reason": "Lý do",
        "recall_status": "Trạng thái",
        "recall_date": "Ngày",
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
        "disclaimer_title": "Cảnh báo y tế:",
        "disclaimer_body": "Tất cả thông tin y tế và kết quả xác minh trên ứng dụng này chỉ mang tính chất tham khảo và không thể thay thế cho chẩn đoán hoặc điều trị của bác sĩ chuyên khoa. Khi đưa ra các quyết định y tế như chẩn đoán bệnh hoặc dùng thuốc, hãy chắc chắn tham khảo ý kiến bác sĩ.",
        "guide_title": "💡 Hướng dẫn MIL: 4 bước nhận biết thuốc giả / chưa được kiểm duyệt",
        "guide_body": """
**Bước 1 — Kiểm tra mã đăng ký 📋**
- `VD-XXXXX-XX` → Thuốc sản xuất trong nước
- `VS-XXXXX-XX` → Thuốc đông y / thảo dược
- `GC-XXXXX-XX` → Thuốc nhập khẩu (Giấy phép)
- `VN-XXXXX-XX` → Thuốc nhập khẩu đã đăng ký

**Bước 2 — Xác minh nguồn gốc mã vạch 🔢**
- **893** = Việt Nam | **000–019** = Mỹ | **400–440** = Đức | **690–699** = Trung Quốc

**Bước 3 — Cảnh giác với các tuyên bố phóng đại ⚠️**
- "Thần dược", "chữa khỏi 100%", "không tác dụng phụ"

**Bước 4 — Đối chiếu với nguồn chính thức 🌐**
- **OpenFDA**: [api.fda.gov](https://api.fda.gov) | **DailyMed**: [dailymed.nlm.nih.gov](https://dailymed.nlm.nih.gov)
        """,
    },
}

SUSPICIOUS_PATTERNS = [
    "100% cure", "miracle", "thần dược", "đặc trị", "no side effect",
    "guaranteed", "instant cure", "chữa khỏi", "đặc hiệu", "bí quyết",
    "không cần đơn", "cải thiện ngay", "thần kỳ", "bách bệnh", "100%"
]


def run_claim_audit(text: str) -> list:
    if not text:
        return []
    text_lower = text.lower()
    return [p for p in SUSPICIOUS_PATTERNS if p.lower() in text_lower]


def get_credibility_score(match_type, external_data, lang="en"):
    s = MIL_STRINGS[lang]
    has_fda  = bool(external_data and (external_data.get("openfda") or external_data.get("dailymed")))
    has_wiki = bool(external_data and (external_data.get("wikipedia") or external_data.get("wikipedia_en")))
    if match_type in ("BARCODE", "NDC_MATCH") or has_fda:
        return s["cred_high"], "#20B26C", "🟢", s["cred_high_desc"]
    elif match_type in ("FUZZY_OCR", "RXNORM_MATCH") and (has_wiki or has_fda):
        return s["cred_med"], "#D97706", "🟡", s["cred_med_desc"]
    elif match_type == "RXNORM_MATCH":
        return s["cred_med"], "#D97706", "🟡", s["cred_med_desc"]
    else:
        return s["cred_low"], "#DC2626", "🔴", s["cred_low_desc"]


def render_credibility_badge(match_type, external_data, lang="en"):
    s = MIL_STRINGS[lang]
    label, color, emoji, desc = get_credibility_score(match_type, external_data, lang)
    html = (
        f'<div style="display:inline-flex;align-items:center;gap:8px;background:#FFFFFF;'
        f'border:1px solid {color};border-radius:9999px;padding:6px 16px;margin:0.75rem 0;box-shadow:0 2px 8px rgba(0,0,0,0.04);">'
        f'<span style="font-size:1rem;">{emoji}</span>'
        f'<span style="font-weight:700;color:{color};font-size:0.88rem;">{s["cred_label"]}: {label}</span>'
        f'<span style="color:#64748B;font-size:0.8rem;"> — {desc}</span>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def check_fda_recall(keyword: str):
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


def render_cross_check_table(matched_med, ocr_text, external_data, lang="en"):
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


def render_mil_guide(lang="en"):
    s = MIL_STRINGS[lang]
    with st.expander(s["guide_title"], expanded=False):
        st.markdown(s["guide_body"])


def render_disclaimer_banner(lang="en"):
    s = MIL_STRINGS[lang]
    html = f"""
    <div class="disclaimer-banner">
        <div class="disclaimer-icon">⚠️</div>
        <div class="disclaimer-text">
            <strong>{s['disclaimer_title']}</strong> {s['disclaimer_body']}
        </div>
    </div>
    """
    render_html_safely(html)


# ==========================================
# RESOURCE & DATA LOADERS
# ==========================================

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
    df = None
    source_name = "Local File"
    status_msg = "Loaded successfully from local database.json"
    is_fallback = False

    if source_url and source_url.strip():
        url = source_url.strip()
        try:
            if "docs.google.com/spreadsheets" in url:
                if "/export" not in url:
                    url = url.split("/edit")[0] + "/export?format=csv" if "/edit" in url else url.rstrip("/") + "/export?format=csv"

            if "csv" in url.lower() or "export?format=csv" in url.lower():
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                source_name = "Remote CSV / Google Sheets"
                status_msg = "Successfully fetched live data from remote CSV."
            else:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                json_data = response.json()
                
                records = json_data.get("medicines", []) if isinstance(json_data, dict) else json_data
                if not isinstance(records, list):
                    raise ValueError("JSON payload must be a list of objects or contain 'medicines' array.")
                
                df = pd.DataFrame(records)
                source_name = "Remote JSON API"
                status_msg = "Successfully fetched live data from remote JSON API."

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
    if EASYOCR_AVAILABLE:
        try:
            return easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            st.error(f"Failed to initialize EasyOCR: {e}")
            return None
    return None


# ==========================================
# OPENCV BARCODE & OCR DETECTION
# ==========================================

def scan_barcode(image: Image.Image):
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    barcodes_found = []
    draw_img = image.convert('RGB').copy()
    draw = ImageDraw.Draw(draw_img)

    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        try:
            barcode_detector = cv2.barcode.BarcodeDetector()
            res = barcode_detector.detectAndDecode(gray)
            
            if len(res) == 4:
                ok, decoded_info, decoded_type, points = res
            else:
                decoded_info, decoded_type, points = res

            if decoded_info:
                if isinstance(decoded_info, str):
                    decoded_info = [decoded_info]
                    decoded_type = [decoded_type] if decoded_type else ["BARCODE"]

                for i, info in enumerate(decoded_info):
                    info_str = str(info).strip()
                    if info_str:
                        b_type = str(decoded_type[i]) if i < len(decoded_type) and decoded_type[i] else "BARCODE"
                        
                        if points is not None and i < len(points):
                            pts_curr = points[i]
                            pts_list = [(int(p[0]), int(p[1])) for p in pts_curr]
                            pts_list.append(pts_list[0])
                            draw.line(pts_list, fill='#20B26C', width=4)

                        barcodes_found.append({'data': info_str, 'type': b_type, 'rect': None})
        except Exception:
            pass

    if hasattr(cv2, "QRCodeDetector"):
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)
            if retval and decoded_info:
                for i, info in enumerate(decoded_info):
                    info_str = str(info).strip()
                    if info_str and not any(b['data'] == info_str for b in barcodes_found):
                        if points is not None and i < len(points):
                            pts_curr = points[i]
                            pts_list = [(int(p[0]), int(p[1])) for p in pts_curr]
                            pts_list.append(pts_list[0])
                            draw.line(pts_list, fill='#20B26C', width=4)

                        barcodes_found.append({'data': info_str, 'type': 'QRCODE', 'rect': None})
        except Exception:
            try:
                info_str, points, _ = qr_detector.detectAndDecode(gray)
                if info_str and info_str.strip():
                    info_clean = info_str.strip()
                    if not any(b['data'] == info_clean for b in barcodes_found):
                        if points is not None:
                            pts_curr = points[0] if points.ndim == 3 else points
                            pts_list = [(int(p[0]), int(p[1])) for p in pts_curr]
                            pts_list.append(pts_list[0])
                            draw.line(pts_list, fill='#20B26C', width=4)

                        barcodes_found.append({'data': info_clean, 'type': 'QRCODE', 'rect': None})
            except Exception:
                pass

    return barcodes_found, draw_img


def preprocess_image_for_ocr(image: Image.Image) -> np.ndarray:
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrasted, h=10)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


def scan_ocr_text(image: Image.Image):
    reader = get_ocr_reader()
    if not reader:
        return "", [], image

    img_np = np.array(image.convert('RGB'))

    def run_ocr(np_img):
        raw_results = reader.readtext(np_img)
        good = [r for r in raw_results if r[2] > 0.2]
        return raw_results, good

    results, good_results = run_ocr(img_np)

    if len(good_results) < 2:
        try:
            enhanced_np = preprocess_image_for_ocr(image)
            enhanced_results, enhanced_good = run_ocr(enhanced_np)
            if len(enhanced_good) > len(good_results):
                results, good_results = enhanced_results, enhanced_good
        except Exception:
            pass

    extracted_tokens = []
    draw_img = image.convert('RGB').copy()
    draw = ImageDraw.Draw(draw_img)
    
    for bbox, text, prob in results:
        if prob > 0.2:
            extracted_tokens.append(text)
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            pts.append(pts[0])
            draw.line(pts, fill='#0284C7', width=2)
            
    raw_text_combined = " ".join(extracted_tokens)
    return raw_text_combined, results, draw_img


# ==========================================
# DATABASE MATCHING ENGINE
# ==========================================

def match_medicine(barcodes: list, ocr_text: str, database_df: pd.DataFrame, fuzzy_threshold: int = 65):
    if database_df is None or database_df.empty:
        return None, None, 0, "Medical Database is empty."

    for b in barcodes:
        code_str = str(b['data']).strip()
        matched_rows = database_df[database_df['barcode'] == code_str]
        if not matched_rows.empty:
            med_dict = matched_rows.iloc[0].to_dict()
            return med_dict, "BARCODE", 100, f"Exact Barcode Match ({b['type']}: {code_str})"

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
    med, mtype, conf, reason = match_medicine(barcodes, ocr_text, database_df, fuzzy_threshold)
    if med:
        return med, mtype, conf, reason

    for b in barcodes:
        ndc_med = lookup_ndc_by_barcode(b['data'])
        if ndc_med:
            return ndc_med, "NDC_MATCH", 90, f"FDA NDC Directory Match (Barcode: {b['data']})"

    if ocr_text and ocr_text.strip():
        words = [w for w in ocr_text.split() if len(w) >= 4 and w.isalpha()]
        for w in words[:5]:
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
                                  stroke="#FFFFFF" stroke-width="2" stroke-linejoin="round" fill="none"/>
                            <path d="M12 8v8M8 12h8" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div class="splash-title">MedGuard</div>
                    <div class="splash-subtitle">Start your journey to verified medicine safety</div>
                </div>
            """).strip(), unsafe_allow_html=True)
        time.sleep(1.9)
        splash_placeholder.empty()
        st.session_state["splash_shown"] = True

    st.markdown(textwrap.dedent("""
        <div class="header-container">
            <div class="header-content">
                <div class="header-badge">
                    🌱 Science Health · Quality Living
                </div>
                <div class="header-title">
                    MedGuard Medicine Verification
                </div>
                <div class="header-subtitle">
                    Bringing together scientific medicine knowledge, we offer instant verification, barcode recognition, and official OpenFDA live label lookup to protect your health every day.
                </div>
            </div>
        </div>
    """).strip(), unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    with st.sidebar:
        st.markdown('<div class="sidebar-eyebrow">Preferences</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="sidebar-eyebrow">Configuration</div>', unsafe_allow_html=True)
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

        st.caption("Cache TTL: 3600 seconds (1 hour)")

        st.divider()
        st.markdown('<div class="sidebar-eyebrow">System</div>', unsafe_allow_html=True)
        st.header("⚙️ Engine Status")
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
        st.markdown('<div class="sidebar-eyebrow">Data</div>', unsafe_allow_html=True)
        st.header("📚 Active Dataset")
        st.metric("Indexed Medicines", len(database_df))

        st.divider()
        st.caption("MedGuard v1.3 • Healthy Mint Light Theme")

    tab_scan, tab_camera, tab_search = st.tabs([
        "🖼️ Scan Uploaded Image", 
        "📷 Live Camera Barcode", 
        "🔍 Manual Search & OpenFDA"
    ])

    # TAB 1: UPLOAD & SCAN
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

    # TAB 2: LIVE CAMERA SCAN
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

    # TAB 3: MANUAL SEARCH
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
            with st.expander(f"💊 **{med['drug_name']}** — Active Ingredient: *{med['active_ingredient']}*", expanded=bool(search_query)):
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

    render_disclaimer_banner(lang=lang)


# ==========================================
# RENDER UI COMPONENTS
# ==========================================

def render_verification_results(matched_med, match_type, confidence, match_reason, barcodes, ocr_text, external_data=None, lang="en"):
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
                <span style="color: #20B26C; font-weight: 800; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                    <svg class="section-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L4 5v6c0 5.25 3.4 9.74 8 11 4.6-1.26 8-5.75 8-11V5l-8-3z" stroke="#20B26C" stroke-width="1.8" stroke-linejoin="round"/>
                        <path d="M8.5 12l2.3 2.3L15.5 9.5" stroke="#20B26C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    VERIFIED DATABASE RECORD
                </span>
                {badge_html}
            </div>
            <div class="med-title">{matched_med['drug_name']}</div>
            <div style="font-size: 0.95rem; color: #176B6B; font-weight: 600; margin-bottom: 1rem;">
                Match Details: {match_reason}
            </div>
            <div class="field-label">🧪 Active Ingredient(s)</div>
            <div class="field-value">{matched_med['active_ingredient']}</div>
            <div class="field-label">📋 Recommended Dosage</div>
            <div class="field-value">{matched_med['dosage']}</div>
            <div class="field-label">🎯 Primary Uses &amp; Indications</div>
            <div class="field-value">{matched_med['uses']}</div>
            <div class="field-label">⚠️ Contraindications &amp; Safety Warnings</div>
            <div class="field-value" style="border-left: 4px solid #D97706; background: #FFFBEB;">
                {matched_med['contraindications']}
            </div>
            <div style="margin-top: 1rem; font-size: 0.85rem; color: #64748B;">
                <strong>Verified Barcode:</strong> <code>{matched_med['barcode']}</code> | <strong>Keywords:</strong> <code>{kw_str}</code>
            </div>
        </div>
        """
        render_html_safely(html)

    else:
        html = """
        <div class="warning-card">
            <div style="font-weight: 800; font-size: 1.25rem; color: #B91C1C; display: flex; align-items: center; gap: 8px;">
                <svg class="section-icon" style="width:22px;height:22px;" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 3L2 20h20L12 3z" stroke="#B91C1C" stroke-width="1.8" stroke-linejoin="round"/>
                    <line x1="12" y1="9.5" x2="12" y2="14" stroke="#B91C1C" stroke-width="1.8" stroke-linecap="round"/>
                    <circle cx="12" cy="17" r="0.9" fill="#B91C1C"/>
                </svg>
                UNKNOWN MEDICINE - NOT FOUND IN DATABASE
            </div>
            <p style="margin-top: 0.5rem; font-size: 0.95rem; color: #7F1D1D;">
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

    if ocr_text and ocr_text.strip():
        flagged = run_claim_audit(ocr_text)
        if flagged:
            s = MIL_STRINGS[lang]
            phrases_str = ", ".join([f"`{p}`" for p in flagged])
            st.warning(
                s["claim_audit_title"] + ": " + s["claim_audit_body"].format(phrases=phrases_str)
            )

    render_credibility_badge(match_type, external_data, lang=lang)

    recall_kw = ""
    if matched_med and matched_med.get("drug_name"):
        recall_kw = matched_med["drug_name"].split("/")[0].split()[0]
    elif barcodes:
        recall_kw = barcodes[0]["data"][:30]
    render_recall_alert(recall_kw, lang=lang)

    if matched_med or (external_data and external_data.get("openfda")):
        render_cross_check_table(matched_med, ocr_text, external_data, lang=lang)

    if external_data:
        render_external_data_cards(external_data)

    render_mil_guide(lang=lang)


def render_rxnav_card(rxnav_data):
    if not rxnav_data:
        return

    badges = "".join([
        f'<span style="background-color: #F3E8FF; color: #7E22CE; font-size: 0.85rem; padding: 4px 12px; border-radius: 9999px; font-weight: 700; border: 1px solid #E9D5FF; margin-right: 6px; display: inline-block; margin-bottom: 6px;">CUI: {cui}</span>' 
        for cui in rxnav_data
    ])

    html = f"""
    <div style="background: linear-gradient(145deg, #FAF5FF 0%, #F3E8FF 100%); border: 1.5px solid #E9D5FF; border-radius: 18px; padding: 1.5rem; color: #581C87; margin-top: 1.25rem; box-shadow: 0 10px 24px rgba(168, 85, 247, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: #7E22CE; font-weight: 800; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                🧬 RXNAV RXNORM DRUG LOOKUP
            </span>
            <span style="background-color: #7E22CE; color: white; font-size: 0.75rem; padding: 4px 12px; border-radius: 9999px; font-weight: 700;">
                NLM API
            </span>
        </div>
        <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #7E22CE; margin-bottom: 0.5rem;">
            RxNorm Concept Unique Identifiers (RxCUI)
        </div>
        <div>
            {badges}
        </div>
    </div>
    """
    render_html_safely(html)


def render_dailymed_card(dailymed_data):
    if not dailymed_data:
        return

    links_html = ""
    for item in dailymed_data[:5]:
        links_html += f"""
        <div style="margin-bottom: 0.75rem; padding: 0.7rem 0.9rem; background: #FFFFFF; border-radius: 10px; border: 1px solid #D1EBE3;">
            <div style="font-weight: 700; font-size: 0.95rem; color: #176B6B;">{item['title']}</div>
            <div style="font-size: 0.8rem; color: #4B5563; margin-top: 0.25rem;">
                SPL ID: <code style="color: #20B26C;">{item['spl_id']}</code>
            </div>
            <div style="margin-top: 0.4rem;">
                <a href="{item['link']}" target="_blank" style="color: #20B26C; font-weight: 700; text-decoration: none; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 4px;">
                    🔗 View Official Package Insert &rarr;
                </a>
            </div>
        </div>
        """

    html = f"""
    <div style="background: linear-gradient(145deg, #FFFFFF 0%, #F0FAF7 100%); border: 1.5px solid #20B26C; border-radius: 18px; padding: 1.5rem; color: #135253; margin-top: 1.25rem; box-shadow: 0 10px 24px rgba(32, 178, 108, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: #176B6B; font-weight: 800; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                📦 DAILYMED OFFICIAL FDA LABELS
            </span>
            <span style="background-color: #20B26C; color: white; font-size: 0.75rem; padding: 4px 12px; border-radius: 9999px; font-weight: 700;">
                DAILYMED API
            </span>
        </div>
        <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #176B6B; margin-bottom: 0.5rem;">
            Package Inserts &amp; Product Details
        </div>
        {links_html}
    </div>
    """
    render_html_safely(html)


def render_wikipedia_card(wiki_data, lang: str = "vi"):
    if not wiki_data:
        return

    label = "WIKIPEDIA VIETNAM SUMMARY" if lang == "vi" else "WIKIPEDIA ENGLISH SUMMARY"
    icon = "📝" if lang == "vi" else "📄"

    html = f"""
    <div style="background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%); border: 1.5px solid #CBD5E1; border-radius: 18px; padding: 1.5rem; color: #1E293B; margin-top: 1.25rem; box-shadow: 0 10px 24px rgba(100, 116, 139, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <span style="color: #334155; font-weight: 800; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                {icon} {label}
            </span>
            <span style="background-color: #475569; color: white; font-size: 0.75rem; padding: 4px 12px; border-radius: 9999px; font-weight: 700;">
                WIKIPEDIA API
            </span>
        </div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A; margin-bottom: 0.5rem;">
            {wiki_data['title']}
        </div>
        <div style="font-size: 0.95rem; line-height: 1.6; color: #334155; background: #FFFFFF; padding: 0.8rem; border-radius: 10px; border: 1px solid #E2E8F0; text-align: justify;">
            {wiki_data['extract']}
        </div>
    </div>
    """
    render_html_safely(html)


def render_openfda_card(openfda_data):
    if not openfda_data:
        return

    html = f"""
    <div style="background: linear-gradient(145deg, #FFFFFF 0%, #E0F2FE 100%); border: 1.5px solid #38BDF8; border-radius: 18px; padding: 1.5rem; color: #0C4A6E; margin-top: 1.25rem; box-shadow: 0 10px 24px rgba(56, 189, 248, 0.08);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="color: #0284C7; font-weight: 800; font-size: 0.9rem; display: flex; align-items: center; gap: 6px;">
                🌐 OPENFDA PUBLIC DRUG LABEL MATCH
            </span>
            <span style="background-color: #0284C7; color: white; font-size: 0.75rem; padding: 4px 12px; border-radius: 9999px; font-weight: 700;">
                LIVE FDA API
            </span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: #0C4A6E;">
            {openfda_data['brand_name']}
            <span style="font-size: 1rem; color: #64748B; font-weight: 400;">{f"({openfda_data['generic_name']})" if openfda_data.get('generic_name') else ""}</span>
        </div>
        <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #0369A1; margin-top: 1rem; margin-bottom: 0.25rem;">
            🧪 Active Ingredient(s)
        </div>
        <div style="font-size: 0.95rem; background: #FFFFFF; padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid #BAE6FD;">
            {openfda_data['active_ingredient']}
        </div>
        <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #0369A1; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            🎯 Indications &amp; Usage
        </div>
        <div style="font-size: 0.95rem; background: #FFFFFF; padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid #BAE6FD;">
            {openfda_data['usage']}
        </div>
        <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #0369A1; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            📋 Dosage &amp; Administration
        </div>
        <div style="font-size: 0.95rem; background: #FFFFFF; padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid #BAE6FD;">
            {openfda_data['dosage']}
        </div>
        <div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #B91C1C; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            ⚠️ FDA Warnings &amp; Precautions
        </div>
        <div style="font-size: 0.95rem; background: #FEF2F2; padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid #FCA5A5; color: #7F1D1D;">
            {openfda_data['warnings']}
        </div>
        {f'''<div style="font-weight: 700; text-transform: uppercase; font-size: 0.75rem; color: #C2410C; margin-top: 0.75rem; margin-bottom: 0.25rem;">
            💊 Drug Interactions (Official FDA Label)
        </div>
        <div style="font-size: 0.95rem; background: #FFEDD5; padding: 0.7rem 0.9rem; border-radius: 8px; border: 1px solid #FDBA74; color: #9A3412;">
            {openfda_data['interactions']}
        </div>''' if openfda_data.get('interactions') else ''}
    </div>
    """
    render_html_safely(html)


def render_external_data_cards(external_data):
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
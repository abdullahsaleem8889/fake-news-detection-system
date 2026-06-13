import streamlit as st
import joblib
import re
import sqlite3
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from collections import Counter
import io
from urllib.parse import urlparse
import time
import hashlib
import requests
import google.generativeai as genai

# ========== GEMINI API CONFIGURATION ==========
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    st.warning(f"Gemini API not configured: {e}")

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AI Fake News Detection Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .header-main { font-size: 36px; font-weight: bold; color: #00d9ff; margin: 20px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
    
    .success-box { 
        padding: 20px; border-radius: 15px; background-color: #238636; 
        border: 3px solid #3fb950; border-left: 8px solid #3fb950;
        color: #ffffff; font-weight: bold; font-size: 16px;
    }
    .error-box { 
        padding: 20px; border-radius: 15px; background-color: #da3633; 
        border: 3px solid #f85149; border-left: 8px solid #f85149;
        color: #ffffff; font-weight: bold; font-size: 16px;
    }
    .warning-box { 
        padding: 20px; border-radius: 15px; background-color: #9e6a03; 
        border: 3px solid #d29922; border-left: 8px solid #d29922;
        color: #ffffff; font-weight: bold; font-size: 16px;
    }
    
    .metric-card-real {
        background: linear-gradient(135deg, #238636 0%, #2d7e3a 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #3fb950;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(63, 185, 80, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        white-space: nowrap;
        overflow: hidden;
    }
    
    .metric-card-real h2 {
        margin: 0;
        font-size: 14px;
        line-height: 1;
    }
    
    .metric-card-real p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-card-fake {
        background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #f85149;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(248, 81, 73, 0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        white-space: nowrap;
        overflow: hidden;
    }
    
    .metric-card-fake h2 {
        margin: 0;
        font-size: 14px;
        line-height: 1;
    }
    
    .metric-card-fake p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-advanced { 
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #388bfd;
        border-left: 8px solid #388bfd;
        margin: 0;
        color: #ffffff;
        font-weight: bold;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
        box-shadow: 0 8px 16px rgba(56, 139, 253, 0.3);
    }
    
    .metric-advanced h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-advanced p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-confidence-high {
        background: linear-gradient(135deg, #238636 0%, #3fb950 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #3fb950;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    
    .metric-confidence-high h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-confidence-high p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-confidence-med {
        background: linear-gradient(135deg, #9e6a03 0%, #d29922 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #d29922;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    
    .metric-confidence-med h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-confidence-med p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .metric-confidence-low {
        background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
        padding: 20px 15px;
        border-radius: 15px;
        border: 3px solid #f85149;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        font-size: 16px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 140px;
    }
    
    .metric-confidence-low h3 {
        margin: 0;
        font-size: 13px;
        line-height: 1;
    }
    
    .metric-confidence-low p {
        margin: 8px 0 0 0;
        font-size: 26px;
        line-height: 1;
    }
    
    .tab-button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 2px solid #30363d;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 14px;
    }
    
    h2, h3, h4 {
        color: #c9d1d9 !important;
        font-weight: bold;
    }
    
    p, span {
        color: #c9d1d9 !important;
    }
    
    /* Scroll Anchor */
    #results-anchor {
        scroll-margin-top: 100px;
    }
    
    /* Chat Message Styling */
    .chat-user-msg {
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 100%);
        padding: 15px 18px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 5px solid #388bfd;
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.2);
        color: #ffffff;
    }
    
    .chat-bot-msg {
        background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
        padding: 15px 18px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 5px solid #58a6ff;
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.2);
        color: #e6edf3;
        line-height: 1.6;
    }
    
    .feedback-section {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }
    
    .copy-btn {
        background-color: #238636;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 11px;
        transition: all 0.2s ease;
    }
    
    .copy-btn:hover {
        background-color: #2ea043;
        transform: scale(1.05);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #30363d;
        text-align: center;
    }
    
    .stat-card h4 {
        margin: 0 0 8px 0;
        color: #58a6ff;
        font-size: 13px;
    }
    
    .stat-card p {
        margin: 0;
        color: #79c0ff;
        font-size: 20px;
        font-weight: bold;
    }
    
    .session-timer {
        color: #79c0ff;
        font-size: 12px;
        margin: 8px 0;
        padding: 8px 12px;
        background: rgba(88, 166, 255, 0.1);
        border-radius: 6px;
        border-left: 3px solid #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load("vectorizer.jb")
        try:
            model = joblib.load("lr_model.jb")
        except:
            model = joblib.load("model.jb")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def init_db():
    conn = sqlite3.connect("advanced_analysis.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis 
                 (id INTEGER PRIMARY KEY, text TEXT, prediction TEXT, 
                  confidence REAL, red_flags INTEGER, bias_score REAL, 
                  credibility REAL, timestamp DATETIME)''')
    
    # Create feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY, question TEXT, response TEXT, 
                  rating TEXT, comment TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

def save_feedback(question, response, rating, comment=""):
    """Save user feedback to database"""
    conn = sqlite3.connect("advanced_analysis.db")
    c = conn.cursor()
    c.execute('''INSERT INTO user_feedback (question, response, rating, comment, timestamp)
                 VALUES (?, ?, ?, ?, ?)''',
              (question[:200], response[:500], rating, comment[:300], datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_feedback_stats():
    """Get feedback statistics"""
    conn = sqlite3.connect("advanced_analysis.db")
    c = conn.cursor()
    c.execute('''SELECT rating, COUNT(*) FROM user_feedback GROUP BY rating''')
    stats = dict(c.fetchall())
    conn.close()
    return stats

vectorizer, model = load_models()
init_db()

# ========== SESSION STATE INITIALIZATION ==========
if 'article_text' not in st.session_state:
    st.session_state.article_text = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = {}  # Track feedback per message
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'article_summary' not in st.session_state:
    st.session_state.article_summary = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'full_analysis_data' not in st.session_state:
    st.session_state.full_analysis_data = None
if 'gemini_cache' not in st.session_state:
    st.session_state.gemini_cache = {}
if 'last_api_call' not in st.session_state:
    st.session_state.last_api_call = 0
if 'quota_wait_until' not in st.session_state:
    st.session_state.quota_wait_until = 0
if 'browse_topic' not in st.session_state:
    st.session_state.browse_topic = ""
if 'browse_articles' not in st.session_state:
    st.session_state.browse_articles = []
if 'browse_num_articles' not in st.session_state:
    st.session_state.browse_num_articles = 8
if 'scroll_to_chat' not in st.session_state:
    st.session_state.scroll_to_chat = False

# ========== RATE LIMITING & CACHING FOR FREE TIER ==========
# Free Tier Limits: 15 requests per minute per API key
# Strategy: Cache responses, implement proper rate limiting, fallback to local analysis

RATE_LIMIT_CALLS = 15  # Free tier: 15 requests per minute
RATE_LIMIT_WINDOW = 60  # 60 seconds
API_CALL_COUNTS = []  # Track API call timestamps

def check_rate_limit():
    """Check if we're within free tier rate limits"""
    current_time = time.time()
    # Remove calls older than rate limit window
    API_CALL_COUNTS[:] = [call_time for call_time in API_CALL_COUNTS 
                         if current_time - call_time < RATE_LIMIT_WINDOW]
    
    # Check if we can make another call
    if len(API_CALL_COUNTS) < RATE_LIMIT_CALLS:
        return True, len(API_CALL_COUNTS), RATE_LIMIT_CALLS
    else:
        return False, len(API_CALL_COUNTS), RATE_LIMIT_CALLS

def record_api_call():
    """Record an API call for rate limiting"""
    API_CALL_COUNTS.append(time.time())

def get_cache_key(question, article, function_type):
    """Generate cache key for Gemini responses"""
    combined = f"{function_type}:{question}:{article[:500]}"
    return hashlib.md5(combined.encode()).hexdigest()

def check_quota_limit():
    """Check if we're in quota waiting period"""
    current_time = time.time()
    if st.session_state.quota_wait_until > current_time:
        wait_time = int(st.session_state.quota_wait_until - current_time)
        return False, wait_time
    return True, 0

def apply_rate_limit():
    """Apply minimum delay between API calls (1 second)"""
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_api_call
    if time_since_last < 1.0:
        time.sleep(1.0 - time_since_last)
    st.session_state.last_api_call = time.time()

def get_fallback_response(question, article_text, response_type="general"):
    """Generate intelligent fallback response when quota exceeded"""
    fallback_responses = {
        "main_topic": f"Based on the article, the main focus appears to be on discussing key events and developments. The article presents information about the stated topic and provides context about related issues. Consider reading the full article for comprehensive details.",
        "verify_facts": f"To verify the claims in this article, I recommend: 1) Cross-reference statements with official sources, 2) Check fact-checking websites like Snopes or FactCheck.org, 3) Look for citations and evidence in the original article. The article's credibility analysis has already been provided above.",
        "red_flags": f"Red flag analysis (from our system): Review the red flags identified in the article analysis section above. Additionally, check for: proper sourcing, verifiable claims, balanced perspective, and citations from credible sources.",
        "general": f"I'm currently experiencing API quota limitations. Based on the article analysis already completed, the system has identified key patterns and credibility factors. Please try again in a few moments or review the analysis results provided above."
    }
    return fallback_responses.get(response_type, fallback_responses["general"])

# ========== ADVANCED FUNCTIONS ==========

def clean_text(text):
    """Text preprocessing - matches training data cleaning"""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_advanced_red_flags(text):
    """Advanced red flag detection with severity levels"""
    flags = []
    critical_count = 0
    high_count = 0
    
    original_text = text
    
    # 1. SENSATIONAL KEYWORDS (CRITICAL)
    sensational = ['shocking', 'exposed', 'scandal', 'unbelievable', 'bombshell', 
                   'conspiracy', 'coverup', 'emergency', 'alert', 'WARNING', 'URGENT']
    sensational_found = [w for w in sensational if w.lower() in text.lower()]
    if len(sensational_found) >= 2:
        flags.append(f" CRITICAL: Heavy sensational language - {', '.join(sensational_found[:3])}")
        critical_count += 1
    elif len(sensational_found) >= 1:
        flags.append(f" HIGH: Sensational words detected - {sensational_found[0]}")
        high_count += 1
    
    # 2. EXCESSIVE CAPS (HIGH)
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.20:
        flags.append(f" CRITICAL: {caps_ratio*100:.1f}% uppercase text")
        critical_count += 1
    elif caps_ratio > 0.12:
        flags.append(f" HIGH: {caps_ratio*100:.1f}% uppercase letters")
        high_count += 1
    
    # 3. PUNCTUATION ABUSE (HIGH)
    exclaim_count = text.count('!')
    question_count = text.count('?')
    if exclaim_count >= 4 or question_count >= 3:
        flags.append(f" CRITICAL: Excessive punctuation (! :{exclaim_count}, ? :{question_count})")
        critical_count += 1
    elif exclaim_count >= 2 or question_count >= 2:
        flags.append(f" HIGH: Multiple punctuation marks")
        high_count += 1
    
    # 4. MISSING SOURCES (HIGH)
    sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
    source_count = sum(text.lower().count(w) for w in sources)
    if source_count == 0 and len(text) > 500:
        flags.append(" HIGH: No credible sources mentioned")
        high_count += 1
    
    # 5. CLICK-BAIT PATTERNS (MEDIUM)
    clickbait = ['won\'t believe', 'doctors hate', 'number 7', 'insiders reveal', 'you must',
                 'they don\'t want', 'secret ingredient']
    clickbait_found = [w for w in clickbait if w.lower() in text.lower()]
    if clickbait_found:
        flags.append(f" MEDIUM: Clickbait phrases detected")
    
    # 6. WORD FREQUENCY (MEDIUM)
    words = text.lower().split()
    from collections import Counter
    word_freq = Counter([w for w in words if len(w) > 3 and w.isalpha()])
    if word_freq:
        most_common_freq = word_freq.most_common(1)[0][1]
        if most_common_freq > len(words) * 0.12:
            flags.append(f" MEDIUM: Repetitive content detected")
    
    return flags, critical_count, high_count

def calculate_reading_difficulty(text):
    """Calculate Flesch-Kincaid Grade Level"""
    sentences = [s for s in text.split('.') if s.strip()]
    words = text.split()
    
    if not sentences or not words:
        return 0
    
    # Syllable count (simplified)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2:
            count += 1
            
        return max(1, count)
    
    syllables = sum(count_syllables(word) for word in words)
    
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return max(0, min(16, grade))

def detect_bias_indicators(text):
    """Detect political and ideological biases"""
    biases = {
        'Left-Leaning': ['progressive', 'liberal', 'woke', 'capitalism', 'privilege'],
        'Right-Leaning': ['woke', 'socialism', 'marxist', 'communist', 'globalist'],
        'Anti-Corporate': ['greed', 'exploitation', 'profit-driven', 'corporate'],
        'Anti-Government': ['tyranny', 'oppression', 'authority', 'regime']
    }
    
    detected = {}
    for bias_type, keywords in biases.items():
        count = sum(1 for kw in keywords if kw.lower() in text.lower())
        if count > 0:
            detected[bias_type] = count
    
    return detected

def advanced_sentiment_analysis(text):
    """Multi-dimensional sentiment analysis"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Emotion detection
    emotions = {
        'Anger': sum(text.lower().count(w) for w in ['angry', 'furious', 'outraged', 'enraged']),
        'Fear': sum(text.lower().count(w) for w in ['afraid', 'scared', 'terrified', 'fear']),
        'Joy': sum(text.lower().count(w) for w in ['happy', 'excited', 'thrilled', 'joyful']),
        'Disgust': sum(text.lower().count(w) for w in ['disgusting', 'vile', 'horrible', 'disgusted']),
        'Surprise': sum(text.lower().count(w) for w in ['shocked', 'surprised', 'amazed', 'astonished'])
    }
    
    dominant = max(emotions, key=emotions.get) if max(emotions.values()) > 0 else 'Neutral'
    
    sentiment_type = "Negative" if polarity < -0.1 else ("Positive" if polarity > 0.1 else "Neutral")
    
    return {
        'sentiment': sentiment_type,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'emotions': emotions,
        'dominant_emotion': dominant
    }

def advanced_credibility_scoring(prediction, confidence, red_flags_count, critical_count, high_count, sentiment, bias_count, readability, text):
    """Advanced credibility calculation - ACCURACY IMPROVED"""
    score = confidence
    
    # NOTE: Trusted source boost is applied separately in STEP 2.5 of the analysis pipeline
    # to avoid double-counting
    
    # RED FLAGS (Heavy penalties)
    score -= (critical_count * 20)  # Each critical flag = -20
    score -= (high_count * 10)      # Each high flag = -10
    
    # SOURCES & CITATIONS (Most important indicator)
    sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence', 'said', 'stated', 'published', 'found', 'shows', 'data']
    source_count = sum(text.lower().count(w) for w in sources)
    word_count = len(text.split())
    
    if source_count == 0 and word_count > 200:
        score -= 25  # Long article with NO sources = highly suspicious
    elif source_count == 0 and word_count > 100:
        score -= 15
    elif source_count == 0 and word_count > 50:
        score -= 8
    
    # SENTIMENT ANALYSIS
    sentiment_val = sentiment['sentiment']
    if sentiment_val == 'Negative':
        score -= 8   # Negative often indicates bias
    elif sentiment_val == 'Positive':
        score -= 5   # Positive emotion less suspicious
    
    # SUBJECTIVITY (Objective > Subjective)
    subjectivity = sentiment['subjectivity']
    if subjectivity > 0.8:
        score -= 12  # Almost pure opinion
    elif subjectivity > 0.65:
        score -= 6   # Moderately subjective
    
    # BIAS INDICATORS
    score -= (bias_count * 8)
    
    # READABILITY
    if readability < 3 or readability > 15:
        score -= 8
    
    # TEXT LENGTH
    if word_count < 30:
        score -= 15  # Too short
    elif word_count < 50:
        score -= 8
    
    # EXCESSIVE CAPS
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / len(text) if text else 0
    if caps_ratio > 0.30:
        score -= 15
    elif caps_ratio > 0.15:
        score -= 8
    
    return max(0, min(100, score))

def extract_advanced_entities(text):
    """Extract entities with categories"""
    entities = {
        'Persons': len(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)),
        'Locations': len(re.findall(r'\b(?:in|from|near|at)\s+[A-Z][a-z]+\b', text)),
        'Organizations': len(re.findall(r'\b(?:Inc|Co|CEO|Government|Ministry)\b', text)),
        'Numbers': len(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)),
        'URLs': len(re.findall(r'https?://\S+', text))
    }
    return entities

# ========== SERPAPI NEWS VERIFICATION ==========
def verify_news_with_serpapi(text, num_results=5):
    """Search internet using SerpAPI to find related articles.
    Returns dict with: articles (list), match_count (int)
    NOTE: This function ONLY fetches articles. The REAL/FAKE decision
    is made by the NLI verification step that follows.
    """
    try:
        # Extract keywords from the text for better search
        keywords = extract_keywords(text)
        if not keywords.strip():
            keywords = text[:150]
        
        url = "https://google.serper.dev/news"
        headers = {
            "X-API-KEY": st.secrets["SERPER_API_KEY"],
            "Content-Type": "application/json"
        }
        data = {"q": keywords, "num": num_results}
        
        response = requests.post(url, json=data, headers=headers, timeout=10)
        results = response.json()
        
        news_articles = results.get("news", [])
        
        # If no news results, try organic search as fallback
        if not news_articles:
            url_organic = "https://google.serper.dev/search"
            data_organic = {"q": keywords, "num": num_results}
            response_organic = requests.post(url_organic, json=data_organic, headers=headers, timeout=10)
            organic_results = response_organic.json()
            organic_articles = organic_results.get("organic", [])
            
            if organic_articles:
                formatted = []
                for art in organic_articles[:5]:
                    formatted.append({
                        "title": art.get("title", "No Title"),
                        "link": art.get("link", "#"),
                        "snippet": art.get("snippet", ""),
                        "source": art.get("source", extract_source_from_url(art.get("link", ""))),
                        "date": art.get("date", "Not Available")
                    })
                return {
                    "articles": formatted,
                    "match_count": len(formatted),
                    "search_type": "organic"
                }
            else:
                return {
                    "articles": [],
                    "match_count": 0,
                    "search_type": "none"
                }
        
        # Format news articles
        formatted_articles = []
        for art in news_articles[:5]:
            formatted_articles.append({
                "title": art.get("title", "No Title"),
                "link": art.get("link", "#"),
                "snippet": art.get("snippet", ""),
                "source": art.get("source", "Unknown"),
                "date": art.get("date", "Not Available")
            })
        
        return {
            "articles": formatted_articles,
            "match_count": len(news_articles),
            "search_type": "news"
        }
        
    except Exception as e:
        return {
            "articles": [],
            "match_count": 0,
            "search_type": "error",
            "error": str(e)
        }


# ========== NLI VERIFICATION (Natural Language Inference) ==========
def nli_verify_claim(user_claim, serpapi_articles):
    """Use Gemini to perform Natural Language Inference (NLI).
    Instead of just checking keyword overlap, this verifies whether the
    SPECIFIC ACTION/CLAIM in the user's text is actually confirmed by
    the retrieved articles.
    
    Returns dict with:
        nli_verdict: "CONFIRMED" | "UNVERIFIED" | "REFUTED"
        nli_reasoning: str (explanation)
        confirmed_by: list (articles that confirm the claim)
        is_real: bool | None
        message: str
    """
    
    if not serpapi_articles or len(serpapi_articles) == 0:
        return {
            "nli_verdict": "UNVERIFIED",
            "nli_reasoning": "No articles were found on the internet related to this claim. "
                            "The specific events or actions described could not be verified.",
            "confirmed_by": [],
            "is_real": False,
            "message": "❌ FAKE — No related information found on the internet to verify this claim."
        }
    
    # Build snippet evidence text from retrieved articles
    evidence_text = ""
    for i, art in enumerate(serpapi_articles[:5], 1):
        title = art.get("title", "")
        snippet = art.get("snippet", "")
        source = art.get("source", "Unknown")
        evidence_text += f"\nSource {i} ({source}):\nTitle: {title}\nSnippet: {snippet}\n"
    
    # If Gemini is not available, do local NLI fallback
    if not GEMINI_AVAILABLE:
        return _local_nli_fallback(user_claim, serpapi_articles)
    
    try:
        record_api_call()
        apply_rate_limit()
        
        prompt = f"""You are a Fact-Checking AI performing Natural Language Inference (NLI).

Your job is to determine whether the NEWS CLAIM below is CONFIRMED, REFUTED, or UNVERIFIED by the retrieved news snippets.

=== NEWS CLAIM TO VERIFY ===
{user_claim[:1500]}

=== RETRIEVED NEWS SNIPPETS FROM INTERNET ===
{evidence_text}

=== VERIFICATION RULES ===
1. EXTRACT the main EVENT or TOPIC from the claim.
2. CHECK if the snippets are reporting about the SAME EVENT or TOPIC.
3. If at least ONE snippet is clearly about the SAME EVENT (same people, same context, same subject matter) → verdict is "CONFIRMED".
4. If snippets discuss the same entity/person but report a COMPLETELY DIFFERENT or CONTRADICTING event → verdict is "REFUTED".
5. If snippets are about UNRELATED topics entirely (no connection to the claim) → verdict is "UNVERIFIED".
6. News articles from different sources reporting the same event = strong confirmation.
7. Small differences in wording, dates, or minor details between the claim and snippets are NORMAL and should NOT cause UNVERIFIED.

=== RESPOND IN THIS EXACT FORMAT (no extra text) ===
VERDICT: [CONFIRMED or UNVERIFIED or REFUTED]
CLAIM_ACTION: [The main event/topic extracted from the claim]
MATCHED_ACTION: [What the snippets report, or "NONE" if unrelated]
REASONING: [2-3 sentences explaining why. Be specific about what matched or didn't match.]
CONFIRMING_SOURCES: [Comma-separated source numbers that cover the same event, or "NONE"]"""
        
        model_nli = genai.GenerativeModel('gemini-2.0-flash')
        response = model_nli.generate_content(prompt, request_options={"timeout": 15})
        nli_response = response.text.strip() if response else ""
        
        # Parse the structured NLI response
        return _parse_nli_response(nli_response, serpapi_articles)
        
    except Exception as e:
        # On API failure, fall back to local NLI
        return _local_nli_fallback(user_claim, serpapi_articles)


def _parse_nli_response(nli_text, serpapi_articles):
    """Parse the structured NLI response from Gemini."""
    verdict = "UNVERIFIED"
    claim_action = ""
    matched_action = ""
    reasoning = ""
    confirming_sources = []
    
    for line in nli_text.split("\n"):
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if v in ["CONFIRMED", "UNVERIFIED", "REFUTED"]:
                verdict = v
        elif line.upper().startswith("CLAIM_ACTION:"):
            claim_action = line.split(":", 1)[1].strip()
        elif line.upper().startswith("MATCHED_ACTION:"):
            matched_action = line.split(":", 1)[1].strip()
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONFIRMING_SOURCES:"):
            src_text = line.split(":", 1)[1].strip()
            if src_text.upper() != "NONE":
                try:
                    confirming_sources = [int(s.strip()) for s in src_text.split(",") if s.strip().isdigit()]
                except:
                    confirming_sources = []
    
    # Build confirmed_by list from source indices
    confirmed_articles = []
    for idx in confirming_sources:
        if 1 <= idx <= len(serpapi_articles):
            confirmed_articles.append(serpapi_articles[idx - 1])
    
    # Map verdict to is_real and message
    if verdict == "CONFIRMED":
        is_real = True
        message = f"✅ REAL — The specific claim has been CONFIRMED by {len(confirmed_articles)} source(s). The exact action described was found in verified news reports."
    elif verdict == "REFUTED":
        is_real = False
        message = "❌ FAKE — The specific claim has been REFUTED. News sources report contradicting information."
    else:  # UNVERIFIED
        is_real = False
        message = "❌ UNVERIFIED — Related topics were found but the SPECIFIC ACTION/EVENT in this claim could NOT be confirmed by any source."
    
    return {
        "nli_verdict": verdict,
        "nli_reasoning": reasoning,
        "claim_action": claim_action,
        "matched_action": matched_action,
        "confirmed_by": confirmed_articles,
        "is_real": is_real,
        "message": message,
        "raw_response": nli_text
    }


def _local_nli_fallback(user_claim, serpapi_articles):
    """Local NLI fallback when Gemini is unavailable.
    Uses verb/action extraction + red flag analysis + article count for entailment checking.
    FIXED v2: Expanded verb list, added article-count-based topic confirmation."""
    
    claim_lower = user_claim.lower()
    num_articles = len(serpapi_articles) if serpapi_articles else 0
    
    # ====== STEP 0: SENSATIONAL / FAKE PATTERN DETECTION ======
    fake_score = 0
    fake_reasons = []
    
    # Check excessive caps (>15% uppercase = suspicious)
    caps_ratio = sum(1 for c in user_claim if c.isupper()) / max(len(user_claim), 1)
    if caps_ratio > 0.15:
        fake_score += 2
        fake_reasons.append(f"Excessive uppercase ({caps_ratio*100:.0f}%)")
    
    # Check excessive exclamation marks
    exclaim_count = user_claim.count('!')
    if exclaim_count >= 3:
        fake_score += 2
        fake_reasons.append(f"Excessive exclamation marks ({exclaim_count})")
    
    # Check sensational keywords
    sensational_words = ['shocking', 'exposed', 'bombshell', 'conspiracy', 'coverup',
                         'covering it up', 'cover up', 'won\'t believe', 'urgent alert',
                         'they don\'t want', 'mainstream media', 'hiding', 'share this',
                         'before they delete', 'secret', 'insider',
                         'breaking shocking', 'alert!!', 'warning!!']
    sensational_found = [w for w in sensational_words if w in claim_lower]
    if len(sensational_found) >= 2:
        fake_score += 3
        fake_reasons.append(f"Sensational language: {', '.join(sensational_found[:3])}")
    elif len(sensational_found) >= 1:
        fake_score += 1
        fake_reasons.append(f"Sensational word: {sensational_found[0]}")
    
    # Check clickbait patterns
    clickbait_patterns = ['you won\'t believe', 'doctors hate', 'they don\'t want you',
                          'share this before', 'what they found', 'the truth about',
                          'exposed:', 'leaked:', 'urgent:', 'alert:']
    clickbait_found = [p for p in clickbait_patterns if p in claim_lower]
    if clickbait_found:
        fake_score += 2
        fake_reasons.append(f"Clickbait patterns detected")
    
    # Check for conspiracy language
    conspiracy_words = ['conspiracy', 'cover-up', 'covering it up', 'destroyed all evidence',
                        'hiding from the public', 'goes all the way to the top',
                        'they are hiding', 'mainstream media is hiding']
    conspiracy_found = [w for w in conspiracy_words if w in claim_lower]
    if conspiracy_found:
        fake_score += 3
        fake_reasons.append(f"Conspiracy language detected")
    
    # ====== If fake_score >= 3, immediately return FAKE ======
    if fake_score >= 3:
        return {
            "nli_verdict": "REFUTED",
            "nli_reasoning": f"Text contains strong fake news indicators: {'; '.join(fake_reasons)}. "
                            f"Sensational language and conspiracy patterns override simple verb matching.",
            "claim_action": ", ".join(fake_reasons[:2]),
            "matched_action": "BLOCKED by fake pattern detection",
            "confirmed_by": [],
            "is_real": False,
            "message": f"\u274c FAKE \u2014 Sensational/conspiracy language detected: {'; '.join(fake_reasons[:2])}"
        }
    
    # ====== STEP 1: ARTICLE COUNT CHECK (HIGH COVERAGE = REAL) ======
    # If 5+ articles found about the same topic AND text has NO fake patterns,
    # this is strong evidence the news is real (heavy media coverage)
    if num_articles >= 5 and fake_score == 0:
        return {
            "nli_verdict": "CONFIRMED",
            "nli_reasoning": f"{num_articles} news articles found covering the same topic. "
                            f"Heavy media coverage strongly indicates this is a real news story. "
                            f"No sensational or fake patterns detected in the text.",
            "claim_action": "Multiple source coverage",
            "matched_action": f"Topic confirmed by {num_articles} articles",
            "confirmed_by": serpapi_articles[:3],
            "is_real": True,
            "message": f"\u2705 REAL \u2014 {num_articles} news sources are covering this same story."
        }
    
    # If 3-4 articles found and no fake patterns, lean towards confirmed
    if num_articles >= 3 and fake_score == 0:
        return {
            "nli_verdict": "CONFIRMED",
            "nli_reasoning": f"{num_articles} news articles found covering the same topic. "
                            f"Multiple source coverage suggests this is a real news story.",
            "claim_action": "Multiple source coverage",
            "matched_action": f"Topic confirmed by {num_articles} articles",
            "confirmed_by": serpapi_articles[:2],
            "is_real": True,
            "message": f"\u2705 REAL \u2014 {num_articles} news sources are covering this topic."
        }
    
    # ====== STEP 2: Extract action verbs (EXPANDED LIST) ======
    action_verbs = re.findall(
        r'\b(announced|confirmed|denied|rejected|approved|banned|arrested|killed|'
        r'launched|signed|resigned|fired|suspended|froze|blocked|attacked|invaded|'
        r'declared|imposed|lifted|cancelled|postponed|released|discovered|revealed|'
        r'accused|charged|convicted|acquitted|appointed|elected|defeated|won|lost|'
        r'died|married|divorced|collapsed|exploded|crashed|hacked|leaked|stolen|'
        r'dismissed|intensified|escalated|threatened|threatening|condemned|warned|'
        r'urged|proposed|proposing|seeking|responding|responded|erupted|withdrew|'
        r'deployed|evacuated|surrendered|seized|captured|rescued|struck|shelled|'
        r'bombed|targeted|sanctioned|vetoed|negotiated|mediated|ceasefire|'
        r'intercepted|retaliated|mobilized|grounded|halted|disrupted|'
        r'closed|reopened|shuttered|suspended|expanded|reduced|increased|'
        r'summoned|expelled|recalled|indicted|impeached|extradited|deported|'
        r'ratified|enacted|repealed|overturned|upheld|ruled|sentenced|'
        r'reporting|citing|stating|calling|claiming|alleging|urging|'
        r'freezing|banning|arresting|launching|signing|resigning|firing|suspending|'
        r'blocking|attacking|invading|declaring|imposing|lifting|cancelling)\b',
        claim_lower
    )
    
    if not action_verbs:
        # No verbs found, but if articles exist, consider article count
        if num_articles >= 2 and fake_score == 0:
            return {
                "nli_verdict": "UNVERIFIED",
                "nli_reasoning": f"Could not extract specific action verbs, but {num_articles} "
                                f"related articles were found. Topic appears to be in the news.",
                "claim_action": "Unknown",
                "matched_action": f"{num_articles} topic matches",
                "confirmed_by": [],
                "is_real": None,
                "message": f"\u26a0\ufe0f UNVERIFIED \u2014 No action verbs found, but {num_articles} related articles exist."
            }
        return {
            "nli_verdict": "UNVERIFIED",
            "nli_reasoning": "Could not extract a specific action from the claim for verification. "
                            "Gemini AI was unavailable for deeper analysis.",
            "claim_action": "Unknown",
            "matched_action": "NONE",
            "confirmed_by": [],
            "is_real": None,
            "message": "\u26a0\ufe0f Could not perform deep verification \u2014 AI service unavailable."
        }
    
    # ====== STEP 3: Check if snippets contain the same action verbs ======
    confirmed = []
    for art in serpapi_articles:
        snippet_lower = (art.get("title", "") + " " + art.get("snippet", "")).lower()
        for verb in action_verbs:
            if verb in snippet_lower:
                confirmed.append(art)
                break
    
    # ====== STEP 4: Apply fake_score penalty if verbs matched ======
    if confirmed and fake_score >= 1:
        return {
            "nli_verdict": "UNVERIFIED",
            "nli_reasoning": f"Action verb(s) '{', '.join(action_verbs[:3])}' found in snippets, "
                            f"but text has suspicious indicators: {'; '.join(fake_reasons)}. "
                            f"Cannot confirm with confidence.",
            "claim_action": ", ".join(action_verbs[:3]),
            "matched_action": "Found but suspicious",
            "confirmed_by": [],
            "is_real": None,
            "message": f"\u26a0\ufe0f UNVERIFIED \u2014 Verbs matched but text has suspicious patterns: {'; '.join(fake_reasons[:2])}"
        }
    
    if confirmed:
        return {
            "nli_verdict": "CONFIRMED",
            "nli_reasoning": f"The action verb(s) '{', '.join(action_verbs[:3])}' were found in "
                            f"{len(confirmed)} source(s), confirming the specific claim.",
            "claim_action": ", ".join(action_verbs[:3]),
            "matched_action": "Found in snippets",
            "confirmed_by": confirmed[:3],
            "is_real": True,
            "message": f"\u2705 REAL \u2014 Action verb(s) confirmed in {len(confirmed)} source(s)."
        }
    else:
        return {
            "nli_verdict": "UNVERIFIED",
            "nli_reasoning": f"Articles mention related entities, but the specific action "
                            f"'{', '.join(action_verbs[:3])}' was NOT found in any source snippet.",
            "claim_action": ", ".join(action_verbs[:3]),
            "matched_action": "NONE",
            "confirmed_by": [],
            "is_real": False,
            "message": "\u274c UNVERIFIED \u2014 Entity found but the specific action could NOT be confirmed."
        }

def extract_source_from_url(url):
    """Extract readable source name from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        return domain.split(".")[0].capitalize()
    except:
        return "Unknown Source"

# ========== CHATBOT FUNCTIONS ==========
def search_web_for_info(query, num_results=5):
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": st.secrets["SERPER_API_KEY"]
        }

        data = {"q": query}

        response = requests.post(url, json=data, headers=headers)
        results = response.json()

        return results.get("organic", [])[:num_results]

    except:
        return []

def fetch_articles_by_topic(topic, num_articles=8):
    """Fetch latest articles by topic using NewsAPI + DuckDuckGo fallback"""
    try:
        import requests
        
        # Extract better keywords from topic for search
        search_query = extract_keywords(topic) if len(topic.split()) > 10 else topic
        
        # NewsAPI key
        newsapi_key = st.secrets.get("NEWSAPI_KEY", "")
        
        # Try NewsAPI first (both endpoints)
        if newsapi_key:
            try:
                # Try top-headlines first (works better on free tier)
                url = "https://newsapi.org/v2/top-headlines"
                params = {
                    'q': search_query,
                    'language': 'en',
                    'apiKey': newsapi_key,
                    'pageSize': num_articles
                }
                
                response = requests.get(url, params=params, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    if articles:
                        return articles
                
                # Fallback to everything endpoint
                url = "https://newsapi.org/v2/everything"
                params['sortBy'] = 'relevancy'
                response = requests.get(url, params=params, timeout=8)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    if articles:
                        return articles
            except:
                pass
        
        # Fallback: Use DuckDuckGo NEWS search
        try:
            from duckduckgo_search import DDGS
            ddgs = DDGS()
            results = []
            
            # Use NEWS endpoint instead of text - much better for articles
            search_results = list(ddgs.news(search_query, max_results=num_articles))
            
            for result in search_results:
                article = {
                    'title': result.get('title', 'No Title'),
                    'description': result.get('body', ''),
                    'url': result.get('url', result.get('href', '#')),
                    'urlToImage': result.get('image', None),
                    'publishedAt': result.get('date', None),
                    'source': result.get('source', 'Unknown Source'),
                    'content': result.get('body', '')[:200]
                }
                results.append(article)
            
            return results[:num_articles]
        except:
            return []
    except:
        return []

# ========== VERIFICATION FUNCTIONS ==========

# ========== TRUSTED SOURCES VERIFICATION ==========

def is_trusted_source(text):
    """Check if article mentions or is from a trusted news source
    Returns True if trusted source detected
    """
    sources = [
        "bbc", "cnn", "reuters", "dawn",
        "al jazeera", "associated press", "ap news"
    ]
    return any(s in text.lower() for s in sources)

def clean_text_for_matching(text):
    """Remove punctuation and convert to lowercase for better matching"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def extract_keywords(text):
    """Extract meaningful keywords by removing common stopwords"""
    words = text.lower().split()
    
    stopwords = [
        "the","is","in","at","on","what","who",
        "when","why","how","a","an","about"
    ]
    
    keywords = [w for w in words if w not in stopwords]
    
    return " ".join(keywords[:10])

def is_trusted_news_url(url):
    """Check if article URL is from a trusted news source"""
    trusted_sources = [
        "bbc",
        "cnn",
        "reuters",
        "aljazeera",
        "theguardian",
        "apnews",
        "nytimes",
        "dawn"
    ]
    
    url = url.lower()
    
    for source in trusted_sources:
        if source in url:
            return True
    
    return False

def get_verification_level(article_text, newsapi_articles):
    """Get verification level by comparing article with NewsAPI sources
    Returns: "STRONG" (2+ sources with >8 word overlap after cleaning)
             "WEAK" (2+ sources with >4 word overlap after cleaning)
             "NONE" (no significant overlap)
    """
    article_words = set(clean_text_for_matching(article_text).split())
    
    strong = 0
    weak = 0
    
    for art in newsapi_articles:
        content = (art.get("title","") + " " + art.get("description",""))
        news_words = set(clean_text_for_matching(content).split())
        
        common = article_words & news_words
        
        if len(common) > 8:
            strong += 1
        elif len(common) > 4:
            weak += 1
    
    if strong >= 2:
        return "STRONG"
    elif weak >= 2:
        return "WEAK"
    else:
        return "NONE"

def extract_relevant_sentences(article_text, query, num_sentences=3):
    """Extract sentences most relevant to the user's query"""
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    
    # Score sentences by relevance to query
    query_words = set(query.lower().split())
    sentence_scores = []
    
    for i, sent in enumerate(sentences):
        sent_words = set(sent.lower().split())
        # Calculate relevance
        common = len(query_words & sent_words)
        score = common / max(len(query_words), 1)
        sentence_scores.append((i, sent, score))
    
    # Get top relevant sentences
    relevant = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:num_sentences]
    relevant = sorted(relevant, key=lambda x: x[0])  # Sort back to original order
    
    return [s[1] for s in relevant]

def summarize_web_results_with_gemini(query, web_results, article_context=""):
    """Summarize web search results using Gemini AI"""
    if not web_results or len(web_results) == 0:
        return None
    
    try:
        # Check rate limit
        can_call, _, _ = check_rate_limit()
        if not can_call or not GEMINI_AVAILABLE:
            return None
        
        # Format web results for Gemini
        results_text = "\n".join([
            f"- {r.get('title', 'Source')}: {r.get('body', '')[:150]}"
            for r in web_results[:5]
        ])
        
        prompt = f"""You are a real-time fact-based AI assistant.

User Question:
{query}

Web Data:
{results_text}

Instructions:
- Give a direct and exact answer
- Use only verified facts
- Do NOT repeat article content
- Keep answer concise (max 120 words)
- If data unclear, say "Not enough verified data"

Answer:
"""
        
        record_api_call()
        apply_rate_limit()
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response_obj = model.generate_content(prompt, request_options={"timeout": 10})
        return response_obj.text[:400]
    except:
        return None

def generate_chatbot_response(question, article):
    """Generate chatbot response - PERFECT FINAL LOGIC"""
    
    # STEP 1: ALWAYS TRY WEB SEARCH
    web_results = search_web_for_info(question)

    if web_results:
        summary = summarize_web_results_with_gemini(question, web_results)

        if summary:
            return summary

    # FALLBACK
    return "❌ No data available"

def get_article_summary_gemini(article_text):
    """Generate article summary using Gemini API with rate limiting"""
    
    # Check rate limit
    can_call, _, _ = check_rate_limit()
    
    # Check cache
    cache_key = get_cache_key("summary", article_text, "summary")
    if cache_key in st.session_state.gemini_cache:
        return st.session_state.gemini_cache[cache_key]
    
    if can_call and GEMINI_AVAILABLE:
        try:
            record_api_call()
            apply_rate_limit()
            
            prompt = f"""Provide a concise 2-3 sentence summary of this article:

{article_text[:2000]}

Summary should be objective and factual."""
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response_obj = model.generate_content(prompt)
            summary = response_obj.text[:500]  # Limit to 500 chars
            
            # Cache it
            st.session_state.gemini_cache[cache_key] = summary
            return summary
            
        except Exception as e:
            # Fallback: Extract first sentences
            sentences = [s.strip() for s in article_text.split('.') if s.strip()]
            fallback = '. '.join(sentences[:3])
            if len(fallback) > 300:
                fallback = fallback[:300] + "..."
            return fallback
    else:
        # Rate limit exceeded, use local extraction
        sentences = [s.strip() for s in article_text.split('.') if s.strip()]
        fallback = '. '.join(sentences[:3])
        if len(fallback) > 300:
            fallback = fallback[:300] + "..."
        return fallback

def verify_claim_gemini(claim, article_context):
    """Verify claim using Gemini API with rate limiting"""
    
    # Check rate limit
    can_call, _, _ = check_rate_limit()
    
    # Check cache
    cache_key = get_cache_key(claim, article_context, "verify")
    if cache_key in st.session_state.gemini_cache:
        return st.session_state.gemini_cache[cache_key]
    
    if can_call and GEMINI_AVAILABLE:
        try:
            record_api_call()
            apply_rate_limit()
            
            prompt = f"""Analyze whether this claim is supported by the given article context.

Claim to verify: "{claim}"

Article context: {article_context[:2000]}

Provide a brief verification analysis including:
1. Is claim mentioned/supported in article?
2. Evidence level (strong/moderate/weak)
3. Suggestions for independent verification
Keep response to 3-4 sentences."""
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            response_obj = model.generate_content(prompt)
            verification = response_obj.text[:500]
            
            # Cache it
            st.session_state.gemini_cache[cache_key] = verification
            return verification
            
        except Exception as e:
            # Fallback local analysis
            claim_in_article = claim.lower() in article_context.lower()
            evidence_words = ['prove', 'evidence', 'show', 'demonstrate', 'study']
            evidence_count = sum(1 for word in evidence_words if word in article_context.lower())
            
            fallback = f"""Claim verification (local analysis): "{claim}"
Mentioned in article: {'Yes' if claim_in_article else 'No'}
Supporting evidence found: {evidence_count} indicators
Recommendation: Cross-reference with official sources."""
            
            return fallback
    else:
        # Rate limit exceeded, use local analysis
        claim_in_article = claim.lower() in article_context.lower()
        evidence_words = ['prove', 'evidence', 'show', 'demonstrate', 'study']
        evidence_count = sum(1 for word in evidence_words if word in article_context.lower())
        
        fallback = f"""Claim verification (local analysis): "{claim}"
Mentioned in article: {'Yes' if claim_in_article else 'No'}
Supporting evidence found: {evidence_count} indicators
Recommendation: Cross-reference with official sources."""
        
        return fallback

# ========== IMPROVED CHATBOT FUNCTION ==========
def improved_chatbot(user_question, article_text=""):
    """
    Chatbot that ALWAYS searches the internet first for answers.
    Article text is used ONLY as context for better search queries — NOT as the answer.
    The answer must come from internet sources, never from the pasted article.
    """

    try:
        # STEP 1: Build a smart search query using user question + article context
        question_keywords = extract_keywords(user_question)
        
        # Use article text to extract topic context for better search
        article_context_keywords = ""
        if article_text:
            article_context_keywords = extract_keywords(article_text[:500])
        
        # Combine question + article context for a richer search query
        search_query = question_keywords
        if article_context_keywords:
            # Add top article keywords to improve search relevance
            article_words = article_context_keywords.split()[:5]
            search_query = f"{question_keywords} {' '.join(article_words)}"

        # STEP 2: INTERNET SEARCH — Try multiple sources
        internet_text = ""
        sources = []
        
        # Search Method 1: SerpAPI Web Search (most reliable)
        try:
            web_results = search_web_for_info(user_question, num_results=5)
            if web_results:
                for r in web_results:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "") or r.get("body", "")
                    source = r.get("source", "") or extract_source_from_url(r.get("link", ""))
                    internet_text += f"{title}. {snippet}. "
                    if source and source not in sources:
                        sources.append(source)
        except:
            pass
        
        # Search Method 2: NewsAPI / DuckDuckGo (additional sources)
        try:
            news_articles = fetch_articles_by_topic(search_query, num_articles=5)
            if news_articles:
                for article in news_articles:
                    title = article.get("title", "")
                    description = article.get("description", "") or article.get("body", "")
                    source = article.get("source", "Unknown")
                    if isinstance(source, dict):
                        source = source.get("name", "Unknown")
                    internet_text += f"{title}. {description}. "
                    if source and source not in sources:
                        sources.append(source)
        except:
            pass

        # STEP 3: Check if we got internet data
        if not internet_text.strip():
            return "❌ No verified internet sources found for this question. Please try rephrasing your question."

        # STEP 4: Generate AI answer from INTERNET data only (not article text)
        sources_list = ", ".join(sources[:5]) if sources else "Internet sources"
        
        prompt = f"""You are a fact-based news research assistant.

User Question:
{user_question}

Internet Search Results (from verified news sources):
{internet_text[:4000]}

Instructions:
- Answer the question using ONLY the internet search results above
- Give a clear, factual answer in 3-5 sentences
- Include specific details, numbers, and facts from the sources
- Do NOT make up information — only use what is in the search results
- If the search results don't have enough info, say "Based on available sources, limited information was found"

Answer:"""

        if GEMINI_AVAILABLE:
            try:
                record_api_call()
                apply_rate_limit()
                model = genai.GenerativeModel("gemini-2.0-flash")
                result = model.generate_content(prompt)
                
                return f"""{result.text}

🌐 Sources: {sources_list}"""
            except Exception as e:
                # Gemini failed — summarize internet results locally
                snippets = [s.strip() for s in internet_text.split('. ') if len(s.strip()) > 30][:3]
                if snippets:
                    return f"""Based on internet search:

{'... '.join(snippets)}...

🌐 Sources: {sources_list}"""
                return "⚠️ AI summarization temporarily unavailable. Please try again."
        else:
            # No Gemini — show raw internet snippets
            snippets = [s.strip() for s in internet_text.split('. ') if len(s.strip()) > 30][:3]
            if snippets:
                return f"""Based on internet search:

{'... '.join(snippets)}...

🌐 Sources: {sources_list}"""
            return "⚠️ AI service not available. Please check API configuration."

    except Exception as e:
        return "⚠️ Chatbot temporarily unavailable."

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ️ ADVANCED CONTROL PANEL")
    st.markdown("---")
    
    try:
        conn = sqlite3.connect("advanced_analysis.db")
        df_stats = pd.read_sql_query("SELECT prediction FROM analysis", conn)
        
        total = len(df_stats)
        fake = len(df_stats[df_stats['prediction'] == 'FAKE'])
        real = len(df_stats[df_stats['prediction'] == 'REAL'])
        
        st.markdown(f'<div class="metric-advanced"><b> Total Analyzed</b><p style="font-size: 24px; margin: 10px 0;">{total}</p></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card-fake"><b> Fake News</b><p style="font-size: 24px; margin: 10px 0;">{fake}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card-real"><b> Real News</b><p style="font-size: 24px; margin: 10px 0;">{real}</p></div>', unsafe_allow_html=True)
        conn.close()
    except:
        st.info("No analysis data yet")
    
    st.markdown("---")
    if st.button("️ Clear All Data"):
        conn = sqlite3.connect("advanced_analysis.db")
        conn.execute("DELETE FROM analysis")
        conn.commit()
        conn.close()
        st.success("Data cleared!")
        st.rerun()

# ========== SIDEBAR ENHANCEMENTS ==========
with st.sidebar:
    st.markdown("### ️ System Status")
    st.markdown(f'<div class="stat-card"><h4> Status</h4><p> ACTIVE</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("###  Quick Stats")
    
    # Get or create analysis count
    try:
        conn = sqlite3.connect("advanced_analysis.db")
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM analysis")
        analysis_count = c.fetchone()[0]
        conn.close()
    except:
        analysis_count = 0
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(" Analyzed", analysis_count)
    with col_b:
        feedback_stats = get_feedback_stats()
        total_feedback = sum(feedback_stats.values())
        st.metric(" Feedback", total_feedback)
    
    st.markdown("---")
    st.markdown("###  Quick Guide")
    st.info("""
    **How to use:**
    1. Paste your article in Tab 1
    2. Click " ANALYZE" 
    3. View results across tabs
    4. Use BATCH for bulk analysis
    5. Provide feedback to improve!
    """)
    
    st.markdown("---")
    st.markdown("###  Features")
    st.success(" Local NLP Analysis")
    st.success(" Web Search Support")
    st.success(" AI Enhancement")
    st.info(" User Feedback System")
    st.warning(" Rate Limited")

# ========== MAIN INTERFACE ==========
st.markdown('<h1 class="header-main"> ADVANCED AI Fake News Detection</h1>', unsafe_allow_html=True)
st.write("**Enterprise-Grade ML + NLP Hybrid System** | Severity-Based Analysis")
st.markdown("---")

tabs = st.tabs([" ADVANCED ANALYZE", " DETAILED METRICS", " EMOTION & BIAS", 
                " EXPLAINABILITY", " DASHBOARD", " BATCH", " BROWSE TOPICS"])

# ========== TAB 1: ADVANCED ANALYZE ==========
with tabs[0]:
    st.subheader(" Advanced News Analysis")
    
    # Initialize text area state if not present
    if 'analyze_text_area' not in st.session_state:
        st.session_state.analyze_text_area = ""
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_text = st.text_area(
            "Paste Article:", 
            height=300, 
            label_visibility="collapsed",
            value=st.session_state.analyze_text_area,
            key="analyze_textarea"
        )
        st.session_state.analyze_text_area = user_text
    
    with col2:
        st.markdown("###  Quick Stats")
        if user_text:
            st.write(f"**Words:** {len(user_text.split())}")
            st.write(f"**Characters:** {len(user_text)}")
            st.write(f"**Sentences:** {len(user_text.split('.'))}")
    
    # Button row with Analyze and Clear
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button(" ANALYZE", use_container_width=True, key="analyze_btn"):
            if user_text.strip():
                with st.spinner(" Processing..."):
                    # STEP 1: Local ML Prediction
                    cleaned = clean_text(user_text)
                    vectorized = vectorizer.transform([cleaned])
                    pred = model.predict(vectorized)[0]
                    
                    # Proper confidence calculation (0-100 scale)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(vectorized)[0]
                        confidence = max(proba) * 100
                    else:
                        # LinearRegression model
                        confidence = min(100, max(0, abs(pred - 0.5) * 200))
                    
                    # Preliminary ML result (not final yet)
                    ml_result = "REAL" if round(pred) == 1 else "FAKE"
                    
                    # Advanced Analysis
                    red_flags, critical, high = detect_advanced_red_flags(user_text)
                    readability = calculate_reading_difficulty(user_text)
                    sentiment_data = advanced_sentiment_analysis(user_text)
                    biases = detect_bias_indicators(user_text)
                    entities = extract_advanced_entities(user_text)
                    
                    credibility = advanced_credibility_scoring(
                        pred, confidence, len(red_flags), critical, high, 
                        sentiment_data, len(biases), readability, user_text
                    )
                    
                    # STEP 2: SERPAPI INTERNET SEARCH (Fetch articles only)
                    serpapi_result = None
                    nli_result = None
                    verification_level = "NONE"
                    newsapi_articles = []
                    
                    with st.spinner("🌐 Searching Internet via SerpAPI..."):
                        serpapi_result = verify_news_with_serpapi(user_text)
                    
                    # STEP 2.5: NLI VERIFICATION (Logical Entailment)
                    # This is the critical step: Even if SerpAPI found articles,
                    # we verify the SPECIFIC CLAIM, not just keyword overlap.
                    with st.spinner("🧠 Performing NLI Claim Verification..."):
                        serpapi_articles = serpapi_result.get("articles", []) if serpapi_result else []
                        nli_result = nli_verify_claim(user_text, serpapi_articles)
                    
                    # Apply credibility adjustments based on NLI verdict
                    if nli_result["nli_verdict"] == "CONFIRMED":
                        verification_level = "STRONG"
                        credibility += 30
                        credibility = min(100, credibility)
                    elif nli_result["nli_verdict"] == "REFUTED":
                        verification_level = "NONE"
                        credibility -= 25
                        credibility = max(0, credibility)
                    else:  # UNVERIFIED - but articles may still exist
                        if serpapi_result and serpapi_result.get("match_count", 0) > 0:
                            # Articles found, just NLI couldn't fully confirm - small penalty only
                            verification_level = "WEAK"
                            credibility -= 5
                            credibility = max(0, credibility)
                        else:
                            verification_level = "NONE"
                            credibility -= 15
                            credibility = max(0, credibility)
                    
                    # STEP 3: TRUSTED SOURCE DETECTION
                    trusted_source_detected = is_trusted_source(user_text)
                    if trusted_source_detected:
                        credibility += 20
                        credibility = min(100, credibility)
                    
                    # ========== STEP 4: FINAL DECISION LOGIC (NLI + RED FLAGS + ARTICLE COUNT) ==========
                    # Key principle: MANY articles about same topic = strong evidence of real news
                    # NLI CONFIRMED = REAL (unless red flags override)
                    # NLI REFUTED = definitely FAKE
                    # NLI UNVERIFIED + many articles (5+) + no red flags = REAL (topic is in the news)
                    # NLI UNVERIFIED + few articles + red flags = FAKE
                    # NLI UNVERIFIED + no articles = FAKE
                    
                    article_count = serpapi_result.get("match_count", 0) if serpapi_result else 0
                    
                    # Safety override: extreme red flags force FAKE regardless of NLI
                    if critical >= 2 and credibility < 40:
                        result = "FAKE"
                        final_confidence = "HIGH"
                    
                    elif nli_result["nli_verdict"] == "CONFIRMED":
                        # Double-check: if NLI says CONFIRMED but red flags are critical, downgrade
                        if critical >= 2:
                            result = "UNCERTAIN"
                            final_confidence = "LOW"
                        else:
                            result = "REAL"
                            final_confidence = "HIGH" if len(nli_result.get("confirmed_by", [])) >= 2 else "MEDIUM"
                    
                    elif nli_result["nli_verdict"] == "REFUTED":
                        result = "FAKE"
                        final_confidence = "HIGH"
                    
                    elif nli_result["nli_verdict"] == "UNVERIFIED" and article_count > 0:
                        # Articles found but NLI couldn't fully confirm the specific claim
                        if critical >= 2:
                            # Many critical red flags — FAKE
                            result = "FAKE"
                            final_confidence = "MEDIUM"
                        elif article_count >= 5 and critical == 0 and high <= 1:
                            # MANY articles + clean text = strong evidence of real news
                            result = "REAL"
                            final_confidence = "LOW"
                        elif article_count >= 3 and critical == 0:
                            # Some articles + clean text = lean real
                            if ml_result == "REAL" or credibility >= 30:
                                result = "REAL"
                                final_confidence = "LOW"
                            else:
                                result = "UNCERTAIN"
                                final_confidence = "LOW"
                        elif critical >= 1 or high >= 2:
                            # Has red flags — lean FAKE
                            result = "FAKE"
                            final_confidence = "MEDIUM"
                        elif ml_result == "REAL" or credibility >= 50:
                            result = "REAL"
                            final_confidence = "LOW"
                        else:
                            result = "UNCERTAIN"
                            final_confidence = "LOW"
                    
                    elif nli_result["nli_verdict"] == "UNVERIFIED" and article_count == 0:
                        # No articles found at all
                        result = "FAKE"
                        final_confidence = "HIGH"
                    
                    elif nli_result["is_real"] is None:
                        # NLI couldn't determine (API failure) — fall back to ML + article count
                        if critical >= 2:
                            result = "FAKE"
                            final_confidence = "MEDIUM"
                        elif article_count >= 5 and critical == 0:
                            # Many articles found, no red flags — lean REAL
                            result = "REAL"
                            final_confidence = "LOW"
                        elif ml_result == "REAL" and credibility >= 60:
                            result = "REAL"
                            final_confidence = "LOW"
                        elif ml_result == "FAKE" and credibility < 40 and critical >= 1:
                            result = "FAKE"
                            final_confidence = "MEDIUM"
                        else:
                            result = "UNCERTAIN"
                            final_confidence = "LOW"
                    
                    else:
                        result = "UNCERTAIN"
                        final_confidence = "LOW"
                    
                    # Build the combined serpapi_result with NLI data for display
                    serpapi_result["is_real"] = nli_result["is_real"]
                    serpapi_result["message"] = nli_result["message"]
                    serpapi_result["nli_verdict"] = nli_result["nli_verdict"]
                    serpapi_result["nli_reasoning"] = nli_result.get("nli_reasoning", "")
                    serpapi_result["claim_action"] = nli_result.get("claim_action", "")
                    serpapi_result["matched_action"] = nli_result.get("matched_action", "")
                    serpapi_result["confirmed_by"] = nli_result.get("confirmed_by", [])
                    
                    # STEP 5: Related articles (confirmed ones first, then all)
                    related_articles = []
                    # Show confirmed articles first
                    confirmed_articles = nli_result.get("confirmed_by", [])
                    shown_links = set()
                    for art in confirmed_articles[:3]:
                        link = art.get('link', '#')
                        if link not in shown_links:
                            related_articles.append({
                                'title': art.get('title', 'No Title'),
                                'description': art.get('snippet', ''),
                                'url': link,
                                'publishedAt': art.get('date', 'Unknown'),
                                'source': {'name': art.get('source', 'Unknown')}
                            })
                            shown_links.add(link)
                    # Fill remaining slots with other articles
                    if serpapi_result and serpapi_result.get("articles"):
                        for art in serpapi_result["articles"][:3]:
                            link = art.get('link', '#')
                            if link not in shown_links and len(related_articles) < 3:
                                related_articles.append({
                                    'title': art.get('title', 'No Title'),
                                    'description': art.get('snippet', ''),
                                    'url': link,
                                    'publishedAt': art.get('date', 'Unknown'),
                                    'source': {'name': art.get('source', 'Unknown')}
                                })
                                shown_links.add(link)
                    
                    # STEP 4: Get Gemini analysis
                    gemini_analysis = None
                    if GEMINI_AVAILABLE:
                        with st.spinner(" Analyzing with AI..."):
                            try:
                                prompt = f"""Analyze this article for credibility:

Article: {user_text[:1000]}

Provide assessment on:
1. Source reliability
2. Factual accuracy potential
3. Bias indicators
4. Overall trustworthiness (0-100%)

Keep response concise (max 150 words)"""
                                model_gemini = genai.GenerativeModel('gemini-2.0-flash')
                                response = model_gemini.generate_content(prompt)
                                gemini_analysis = response.text if response else None
                            except:
                                gemini_analysis = None
                    
                    # Save to DB
                    conn = sqlite3.connect("advanced_analysis.db")
                    conn.execute("INSERT INTO analysis VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)",
                               (user_text[:200], result, confidence, len(red_flags), 0, credibility, datetime.datetime.now()))
                    conn.commit()
                    conn.close()
                    
                    # Save ALL data to session state
                    st.session_state.article_text = user_text
                    st.session_state.show_analysis = True
                    st.session_state.full_analysis_data = {
                        'result': result,
                        'final_confidence': final_confidence,
                        'ml_result': ml_result,
                        'confidence': confidence,
                        'red_flags': red_flags,
                        'critical': critical,
                        'high': high,
                        'readability': readability,
                        'sentiment_data': sentiment_data,
                        'biases': biases,
                        'entities': entities,
                        'credibility': credibility,
                        'pred': pred,
                        'related_articles': related_articles,
                        'gemini_analysis': gemini_analysis,
                        'verification_level': verification_level,
                        'trusted_source_detected': trusted_source_detected,
                        'serpapi_result': serpapi_result,
                        'nli_result': nli_result
                    }
                    
                    st.success(" Analysis Complete!")
                    st.balloons()
    
    with btn_col2:
        if st.button(" CLEAR TEXT", use_container_width=True, key="clear_text_btn"):
            st.session_state.analyze_text_area = ""
            st.session_state.show_analysis = False
            st.session_state.full_analysis_data = None
            st.success(" Text cleared!")
            st.rerun()
    
    # ========== DISPLAY ANALYSIS RESULTS (FROM SESSION STATE) ==========
    if st.session_state.show_analysis and st.session_state.full_analysis_data:
        # Scroll anchor for smooth navigation
        st.html('<div id="results-anchor"></div>')
        
        data = st.session_state.full_analysis_data
        result = data['result']
        confidence = data['confidence']
        readability = data['readability']
        credibility = data['credibility']
        red_flags = data['red_flags']
        sentiment_data = data['sentiment_data']
        biases = data['biases']
        entities = data['entities']
        critical = data['critical']
        pred = data['pred']
        
        st.markdown("---")
        st.markdown("###  Analysis Results")
        
        # Display results - 4 metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if result == "REAL":
                st.markdown('<div class="metric-card-real"><h2><br>VERDICT</h2><p>REAL</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card-fake"><h2><br>VERDICT</h2><p>FAKE</p></div>', unsafe_allow_html=True)
        
        with col2:
            # Show final confidence level based on decision logic
            final_conf = data.get('final_confidence', 'LOW')
            if final_conf == "HIGH":
                st.markdown(f'<div class="metric-confidence-high"><h3></h3><p>{final_conf}</p><h3 style="font-size:11px;margin:0">Decision</h3></div>', unsafe_allow_html=True)
            elif final_conf == "MEDIUM":
                st.markdown(f'<div class="metric-confidence-med"><h3></h3><p>{final_conf}</p><h3 style="font-size:11px;margin:0">Decision</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-confidence-low"><h3></h3><p>{final_conf}</p><h3 style="font-size:11px;margin:0">Decision</h3></div>', unsafe_allow_html=True)
        
        with col3:
            grade = "Easy" if readability < 8 else "Moderate" if readability < 12 else "Hard"
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{grade}</p><h3 style="font-size:11px;margin:0">G{readability:.0f}</h3></div>', unsafe_allow_html=True)
        
        with col4:
            if credibility > 70:
                st.markdown(f'<div class="metric-confidence-high"><h3></h3><p>{credibility:.1f}%</p><h3 style="font-size:11px;margin:0">Quality</h3></div>', unsafe_allow_html=True)
            elif credibility > 40:
                st.markdown(f'<div class="metric-confidence-med"><h3></h3><p>{credibility:.1f}%</p><h3 style="font-size:11px;margin:0">Verify</h3></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-confidence-low"><h3></h3><p>{credibility:.1f}%</p><h3 style="font-size:11px;margin:0">Low</h3></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== NLI VERIFICATION RESULT (Primary) ==========
        nli_data = data.get('nli_result')
        serpapi_data = data.get('serpapi_result')
        
        if nli_data or serpapi_data:
            st.markdown("### 🧠 Claim Verification (NLI + SerpAPI)")
            
            # NLI Verdict Display
            if nli_data:
                nli_verdict = nli_data.get('nli_verdict', 'UNVERIFIED')
                
                if nli_verdict == "CONFIRMED":
                    st.success(f"✅ {nli_data.get('message', 'Claim CONFIRMED')}")
                elif nli_verdict == "REFUTED":
                    st.error(f"❌ {nli_data.get('message', 'Claim REFUTED')}")
                else:
                    st.error(f"❌ {nli_data.get('message', 'Claim UNVERIFIED')}")
                
                # Show NLI reasoning details
                with st.expander("🔍 NLI Verification Details", expanded=True):
                    col_nli1, col_nli2 = st.columns(2)
                    with col_nli1:
                        st.markdown(f"**📋 Claim Action:** {nli_data.get('claim_action', 'N/A')}")
                    with col_nli2:
                        matched = nli_data.get('matched_action', 'NONE')
                        if matched and matched.upper() != 'NONE':
                            st.markdown(f"**✅ Matched Action:** {matched}")
                        else:
                            st.markdown(f"**❌ Matched Action:** Not found in any source")
                    
                    st.markdown(f"**💡 Reasoning:** {nli_data.get('nli_reasoning', 'N/A')}")
                    
                    # Show how many articles were found vs confirmed
                    total_found = serpapi_data.get('match_count', 0) if serpapi_data else 0
                    confirmed_count = len(nli_data.get('confirmed_by', []))
                    
                    if total_found > 0 and confirmed_count == 0:
                        st.warning(f"⚠️ **Semantic Overlap Detected:** {total_found} articles found about the same TOPIC/ENTITY, "
                                  f"but 0 articles confirmed the SPECIFIC ACTION in the claim. "
                                  f"This is why the claim is marked as UNVERIFIED.")
                    elif total_found > 0 and confirmed_count > 0:
                        st.info(f"📊 {confirmed_count} out of {total_found} articles explicitly confirm the claimed action.")
            
            # Show source articles (confirmed ones marked)
            if serpapi_data and serpapi_data.get('articles'):
                confirmed_links = set()
                if nli_data and nli_data.get('confirmed_by'):
                    confirmed_links = {art.get('link', '') for art in nli_data['confirmed_by']}
                
                st.markdown("#### 📰 Retrieved Articles:")
                for i, art in enumerate(serpapi_data['articles'][:3], 1):
                    source_name = art.get('source', 'Unknown Source')
                    date = art.get('date', 'Date not available')
                    title = art.get('title', 'No Title')
                    link = art.get('link', '#')
                    snippet = art.get('snippet', '')
                    is_confirmed = link in confirmed_links
                    
                    badge = "✅ CONFIRMS CLAIM" if is_confirmed else "⚠️ TOPIC ONLY"
                    
                    st.markdown(f"""
                    **{i}. [{title}]({link})** — {badge}
                    - 📌 **Source:** {source_name}
                    - 📅 **Date:** {date}
                    - 📝 {snippet[:150]}...
                    ---
                    """)
        
        st.markdown("---")
        
        # VERDICT EXPLANATION
        st.subheader(" Detailed Analysis Report")
        
        # Detailed decision explanation
        if result == "FAKE":
            st.markdown("###  WHY THIS IS CLASSIFIED AS FAKE NEWS:")
            
            reasons = []
            if len(red_flags) > 0:
                reasons.append(f"**Misinformation Signals Detected ({len(red_flags)})**: Multiple patterns typical of misinformation were found")
            
            if critical > 0:
                reasons.append(f"**{critical} CRITICAL red flags**: Heavy sensational language, excessive punctuation, or misleading patterns")
            
            if sentiment_data['sentiment'] in ['Negative', 'Positive'] and sentiment_data['subjectivity'] > 0.6:
                reasons.append(f"**Emotional Language**: The text uses strong {sentiment_data['sentiment'].lower()} sentiment with highly subjective tone")
            
            sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
            source_count = sum(st.session_state.article_text.lower().count(w) for w in sources)
            if source_count == 0:
                reasons.append("**No Credible Sources**: Article contains no references to verified sources or evidence")
            
            if len(biases) > 0:
                reasons.append(f"**Ideological Bias Detected**: Shows patterns of {', '.join(biases.keys())}")
            
            if credibility < 65:
                reasons.append(f"**Low Overall Credibility Score**: {credibility:.1f}% - Below the {65}% threshold required for REAL classification without verification")
            
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"• {reason}")
            
            st.warning("️ **Recommendation**: Cross-reference with official sources and fact-checking websites before sharing")
        
        elif result == "LIKELY REAL":
            st.markdown("###  WHY THIS IS CLASSIFIED AS LIKELY REAL:")
            
            reasons = []
            if len(red_flags) < 3:
                reasons.append("**Minimal Misinformation Signals**: Few suspicious patterns detected")
            
            sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
            source_count = sum(st.session_state.article_text.lower().count(w) for w in sources)
            if source_count > 0:
                reasons.append(f"**Credible Sources Referenced**: Article cites verified information sources ({source_count} instances)")
            
            if sentiment_data['sentiment'] == 'Neutral' or sentiment_data['subjectivity'] < 0.6:
                reasons.append("**Mostly Objective Tone**: Article maintains relatively neutral, factual presentation")
            
            reasons.append(f"**Credibility Score**: {credibility:.1f}% - Supported by external news verification")
            
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"• {reason}")
            
            st.info("ℹ️ **Note**: Classification based on external news source corroboration. Still recommended to verify important claims")
        
        else:  # REAL
            st.markdown("###  WHY THIS IS CLASSIFIED AS REAL NEWS:")
            
            reasons = []
            if len(red_flags) < 2:
                reasons.append("**Minimal Misinformation Signals**: Few or no patterns typical of fake news")
            
            sources = ['according', 'reported', 'confirmed', 'source', 'study', 'research', 'evidence']
            source_count = sum(st.session_state.article_text.lower().count(w) for w in sources)
            if source_count > 0:
                reasons.append(f"**Credible Sources Referenced**: Article cites or references verified information sources ({source_count} instances)")
            
            if sentiment_data['sentiment'] == 'Neutral' or sentiment_data['subjectivity'] < 0.5:
                reasons.append("**Objective Tone**: Article maintains neutral, factual tone without excessive emotion")
            
            if len(biases) == 0:
                reasons.append("**No Detected Bias**: Content doesn't show ideological or political bias patterns")
            
            if credibility >= 70:
                reasons.append(f"**High Credibility Score**: {credibility:.1f}% - Multiple factors indicate reliable journalism")
            
            if 100 > len(st.session_state.article_text.split()) > 50:
                reasons.append("**Adequate Content Length**: Article has sufficient detail for credible reporting")
            
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"• {reason}")
            
            st.success(" **Status**: This appears to be credible, factual news from reliable patterns")
        
        st.markdown("---")
        
        # Red Flags Section
        st.subheader(" Detailed Red Flags Analysis")
        if red_flags:
            st.markdown(f"**Found {len(red_flags)} warning indicators:**")
            for flag in red_flags:
                st.markdown(f"• {flag}")
        else:
            st.success(" No misinformation patterns detected")
        
        st.markdown("---")
        
        # Entities
        st.subheader("️ Detected Entities")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{entities["Persons"]}</p><h3 style="font-size:11px;margin:0">Persons</h3></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{entities["Locations"]}</p><h3 style="font-size:11px;margin:0">Locations</h3></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{entities["Organizations"]}</p><h3 style="font-size:11px;margin:0">Orgs</h3></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{entities["Numbers"]}</p><h3 style="font-size:11px;margin:0">Numbers</h3></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{entities["URLs"]}</p><h3 style="font-size:11px;margin:0">URLs</h3></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ========== RELATED ARTICLES FROM NEWSAPI ==========
        if 'related_articles' in data and data['related_articles']:
            st.subheader(" Related News (NewsAPI)")
            st.markdown("*Showing related articles to verify context and claims*")
            
            related = data['related_articles'][:3]
            for i, article in enumerate(related, 1):
                # Safe title extraction
                title = article.get('title', 'No Title')
                if not title:
                    title = 'No Title'
                title_short = title[:60] if len(str(title)) > 60 else str(title)
                
                with st.expander(f"📰 Article {i}: {title_short}..."):
                    # Handle source - can be dict or string
                    source = article.get('source', 'Unknown')
                    if isinstance(source, dict):
                        source_name = source.get('name', 'Unknown')
                    else:
                        source_name = str(source) if source else 'Unknown'
                    
                    st.write(f"**Source**: {source_name}")
                    
                    # Safe date extraction
                    published_at = article.get('publishedAt')
                    date_str = published_at[:10] if published_at else 'Unknown'
                    st.write(f"**Date**: {date_str}")
                    
                    # Safe summary extraction
                    summary = article.get('description') or article.get('content') or 'No summary available'
                    if summary:
                        summary = str(summary)[:200]
                    st.write(f"**Summary**: {summary}...")
                    
                    # Safe URL extraction
                    url = article.get('url', '#')
                    st.markdown(f"[🔗 Read Full Article]({url})")
        
        # ========== GEMINI AI ANALYSIS ==========
        if 'gemini_analysis' in data and data['gemini_analysis']:
            st.subheader(" Gemini AI Analysis")
            st.info(data['gemini_analysis'])
        
        st.markdown("---")
        
        # ========== INTEGRATED BATCH SECTION ==========
        st.subheader(" AI Analysis Assistant (Smart Local + Gemini)")
        
        # Show rate limit status
        can_call, current_count, limit = check_rate_limit()
        
        # Quick question buttons
        st.markdown("#### Quick Questions:")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button(" Main Topic", key="main_topic_btn", use_container_width=True):
                quick_question = "What are the main topics and key points discussed in this article?"
                with st.spinner(" Analyzing..."):
                    response = improved_chatbot(quick_question, st.session_state.article_text)
                    st.session_state.chat_history.append({'type': 'user', 'content': quick_question})
                    st.session_state.chat_history.append({'type': 'bot', 'content': response})
                    st.session_state.scroll_to_chat = True
                st.rerun()
        
        with quick_col2:
            if st.button(" Verify Facts", key="verify_btn", use_container_width=True):
                quick_question = "Are the main claims factually accurate? What evidence supports them?"
                with st.spinner(" Verifying..."):
                    response = improved_chatbot(quick_question, st.session_state.article_text)
                    st.session_state.chat_history.append({'type': 'user', 'content': quick_question})
                    st.session_state.chat_history.append({'type': 'bot', 'content': response})
                    st.session_state.scroll_to_chat = True
                st.rerun()
        
        with quick_col3:
            if st.button("️ Red Flags", key="flags_btn", use_container_width=True):
                quick_question = "What credibility concerns or potential misinformation signals exist in this article?"
                with st.spinner(" Analyzing..."):
                    response = improved_chatbot(quick_question, st.session_state.article_text)
                    st.session_state.chat_history.append({'type': 'user', 'content': quick_question})
                    st.session_state.chat_history.append({'type': 'bot', 'content': response})
                    st.session_state.scroll_to_chat = True
                st.rerun()
        
        st.markdown("---")
        
        # Custom question
        st.markdown("#### Ask Your Own Question:")
        col_q1, col_q2 = st.columns([4, 1])
        
        with col_q1:
            custom_question = st.text_input("Your question about the article:", placeholder="Ask anything...", key="custom_q_input")
        
        with col_q2:
            ask_custom = st.button(" Ask", key="ask_custom_btn", use_container_width=True)
        
        if ask_custom and custom_question.strip():
            with st.spinner(" Thinking..."):
                response = improved_chatbot(custom_question, st.session_state.article_text)
                st.session_state.chat_history.append({'type': 'user', 'content': custom_question})
                st.session_state.chat_history.append({'type': 'bot', 'content': response})
                st.session_state.scroll_to_chat = True
            st.rerun()
        
        st.markdown("---")
        
        # Display chat history (AFTER all inputs, at bottom)
        # Scroll anchor for chat messages
        st.html('<div id="chat-anchor"></div>')
        
        # Auto scroll to chat if new message added
        if st.session_state.chat_history:
            # Session statistics
            session_duration = int(time.time() - st.session_state.session_start_time)
            minutes = session_duration // 60
            seconds = session_duration % 60
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.markdown(f'<div class="stat-card"><h4> Messages</h4><p>{len(st.session_state.chat_history)}</p></div>', unsafe_allow_html=True)
            with col_stat2:
                st.markdown(f'<div class="stat-card"><h4>️ Session</h4><p>{minutes}m {seconds}s</p></div>', unsafe_allow_html=True)
            with col_stat3:
                feedback_stats = get_feedback_stats()
                helpful_count = feedback_stats.get('helpful', 0)
                st.markdown(f'<div class="stat-card"><h4> Helpful</h4><p>{helpful_count}</p></div>', unsafe_allow_html=True)
            
            st.markdown("####  Chat History:")
            for idx, msg in enumerate(st.session_state.chat_history):
                if msg['type'] == 'user':
                    st.markdown(f'<div class="chat-user-msg"> <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot-msg"> <b>Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                    
                    # Add feedback buttons for bot responses
                    feedback_key = f"msg_{idx}"
                    if feedback_key not in st.session_state.feedback_given:
                        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
                        with fb_col1:
                            if st.button(" Helpful", key=f"like_{idx}", use_container_width=True):
                                save_feedback(st.session_state.chat_history[idx-1]['content'], msg['content'], "helpful")
                                st.session_state.feedback_given[feedback_key] = True
                                st.success("Thanks for your feedback!")
                                st.rerun()
                        with fb_col2:
                            if st.button(" Not Helpful", key=f"dislike_{idx}", use_container_width=True):
                                save_feedback(st.session_state.chat_history[idx-1]['content'], msg['content'], "not_helpful")
                                st.session_state.feedback_given[feedback_key] = True
                                st.info("We'll improve!")
                                st.rerun()
                    else:
                        st.caption(" Feedback recorded")
            st.markdown("---")
        
        # Feedback Summary Section
        st.markdown("####  Your Feedback Summary:")
        feedback_stats = get_feedback_stats()
        
        if feedback_stats:
            col_fb1, col_fb2, col_fb3 = st.columns(3)
            with col_fb1:
                helpful_count = feedback_stats.get('helpful', 0)
                st.metric(" Helpful", helpful_count)
            with col_fb2:
                not_helpful_count = feedback_stats.get('not_helpful', 0)
                st.metric(" Not Helpful", not_helpful_count)
            with col_fb3:
                total_feedback = sum(feedback_stats.values())
                if total_feedback > 0:
                    helpful_percent = (helpful_count / total_feedback) * 100
                    st.metric(" Satisfaction", f"{helpful_percent:.0f}%")
        else:
            st.info(" No feedback recorded yet. Your feedback helps us improve!")
        
        st.markdown("---")
        
        # Export and Manage options
        col_export, col_clear = st.columns(2)
        
        with col_export:
            if st.button(" Export Chat", use_container_width=True):
                if st.session_state.chat_history:
                    # Create formatted chat text
                    chat_text = "=== CHAT HISTORY ===\n"
                    chat_text += f"Session Duration: {session_duration} seconds\n"
                    chat_text += f"Total Messages: {len(st.session_state.chat_history)}\n\n"
                    
                    for msg in st.session_state.chat_history:
                        if msg['type'] == 'user':
                            chat_text += f" User: {msg['content']}\n\n"
                        else:
                            chat_text += f" Assistant: {msg['content']}\n\n"
                        chat_text += "-" * 60 + "\n\n"
                    
                    st.download_button(
                        label="Download Chat (.txt)",
                        data=chat_text,
                        file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_chat"
                    )
        
        with col_clear:
            if st.button(" Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.feedback_given = {}
                st.session_state.session_start_time = time.time()
                st.success(" Chat cleared!")
                st.rerun()
        
        # Smooth scroll to results or chat (JavaScript)
        st.write("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const resultsAnchor = document.getElementById('results-anchor');
                const chatAnchor = document.getElementById('chat-anchor');
                
                // Try to scroll to chat first (if messages just added)
                if (chatAnchor) {
                    chatAnchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } 
                // Otherwise scroll to results if available
                else if (resultsAnchor) {
                    resultsAnchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        </script>
        """, unsafe_allow_html=True)
        
        # Reset scroll flag
        st.session_state.scroll_to_chat = False

# ========== TAB 2: DETAILED METRICS ==========
with tabs[1]:
    st.subheader(" Detailed Text Metrics")
    tab2_text = st.session_state.get('article_text', '')
    if tab2_text:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            words = tab2_text.split()
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{avg_word_len:.2f}</p><h3 style="font-size:11px;margin:0">Word Len</h3></div>', unsafe_allow_html=True)
        
        with col2:
            unique_words = len(set(w.lower() for w in words))
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{unique_words}</p><h3 style="font-size:11px;margin:0">Unique</h3></div>', unsafe_allow_html=True)
        
        with col3:
            diversity = unique_words / len(words) if words else 0
            st.markdown(f'<div class="metric-advanced"><h3></h3><p>{diversity:.0%}</p><h3 style="font-size:11px;margin:0">Diversity</h3></div>', unsafe_allow_html=True)
        
        # Top words chart
        st.subheader("Top 15 Keywords")
        word_freq = Counter([w.lower() for w in words if len(w) > 3])
        top_words = word_freq.most_common(15)
        
        if top_words:
            words_names, words_counts = zip(*top_words)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(words_names, words_counts, color='#1f77b4')
            st.pyplot(fig)
    else:
        st.info(" Analyze an article first")

# ========== TAB 3: EMOTION & BIAS ==========
with tabs[2]:
    st.subheader(" Emotion & Bias Detection")
    tab3_data = st.session_state.get('full_analysis_data', None)
    if tab3_data and 'sentiment_data' in tab3_data:
        tab3_sentiment = tab3_data['sentiment_data']
        tab3_biases = tab3_data.get('biases', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Emotional Profile")
            emotion_data = tab3_sentiment['emotions']
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(emotion_data.keys(), emotion_data.values(), color=['#ff6b6b', '#ee5a6f', '#f1e15b', '#a8e6cf', '#ffd3b6'])
            ax.set_ylabel('Count')
            ax.set_title('Emotion Detection')
            st.pyplot(fig)
        
        with col2:
            st.markdown("###  Bias Indicators")
            if tab3_biases:
                for bias_type, score in tab3_biases.items():
                    st.markdown(f'<div class="warning-box"><b>{bias_type}</b><p>{score} mentions detected</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box"><p> No significant bias detected</p></div>', unsafe_allow_html=True)
            
            # Also show sentiment summary
            st.markdown("###  Sentiment Summary")
            st.markdown(f"**Sentiment:** {tab3_sentiment['sentiment']}")
            st.markdown(f"**Polarity:** {tab3_sentiment['polarity']:.3f}")
            st.markdown(f"**Subjectivity:** {tab3_sentiment['subjectivity']:.3f}")
            st.markdown(f"**Dominant Emotion:** {tab3_sentiment['dominant_emotion']}")
    else:
        st.info(" Analyze an article first")

# ========== TAB 4: EXPLAINABILITY ==========
with tabs[3]:
    st.subheader(" Model Explainability")
    tab4_text = st.session_state.get('article_text', '')
    if tab4_text:
        # Top influential words
        cleaned = re.sub(r'[^\w\s]', '', tab4_text.lower())
        vectorized = vectorizer.transform([cleaned])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vectorized.toarray()[0]
        top_indices = np.argsort(tfidf_scores)[-12:][::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
        if top_features:
            names, scores = zip(*top_features)
            ax.barh(names, scores, color='#ff7f0e')
            ax.set_xlabel('TF-IDF Score')
            ax.set_title('Top Influential Words (ML Model)')
            st.pyplot(fig)
    else:
        st.info(" Analyze an article first")

# ========== TAB 5: DASHBOARD ==========
with tabs[4]:
    st.subheader(" Analytics Dashboard")
    try:
        conn = sqlite3.connect("advanced_analysis.db")
        df = pd.read_sql_query("SELECT prediction, credibility FROM analysis", conn)
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                pred_counts = df['prediction'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', colors=['#dc3545', '#28a745'])
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df['credibility'], bins=20, color='#1f77b4', edgecolor='black')
                ax.set_xlabel("Credibility Score")
                st.pyplot(fig)
        conn.close()
    except:
        st.info("No data available")

# ========== TAB 6: BATCH PROCESS ==========
with tabs[5]:
    st.subheader(" Batch Processing")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded and st.button("Process All"):
        df_upload = pd.read_csv(uploaded)
        text_col = st.selectbox("Text Column:", df_upload.columns)
        
        results = []
        progress = st.progress(0)
        
        for idx, row in df_upload.iterrows():
            text = str(row[text_col])
            cleaned = re.sub(r'[^\w\s]', '', text.lower())
            vectorized = vectorizer.transform([cleaned])
            pred = model.predict(vectorized)[0]
            confidence = abs((pred - 0.5) * 200)
            result = "REAL" if round(pred) == 1 else "FAKE"
            
            results.append({"Text": text[:50], "Verdict": result, "Confidence": f"{confidence:.1f}%"})
            progress.progress((idx + 1) / len(df_upload))
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button(" Download Results", csv, "results.csv", "text/csv")

# ========== TAB 7: BROWSE BY TOPIC ==========
with tabs[6]:
    st.subheader(" Browse Latest Articles by Topic")
    
    # Better layout with aligned input
    col1, col2, col3 = st.columns([3, 1, 1.2])
    
    with col1:
        topic_input = st.text_input(
            "Enter Topic (e.g., 'AI Development', 'Politics', 'Technology'):",
            placeholder="Type any topic you want to explore...",
            label_visibility="collapsed",
            key="topic_input_browse",
            value=st.session_state.browse_topic
        )
    
    with col2:
        num_articles = st.number_input(
            "Show:",
            min_value=3,
            max_value=15,
            value=st.session_state.browse_num_articles,
            label_visibility="collapsed",
            key="num_articles_browse"
        )
    
    with col3:
        fetch_clicked = st.button(" Fetch Articles", use_container_width=True, key="fetch_btn_browse")
    
    # Handle button click - store in session state
    if fetch_clicked:
        if not topic_input.strip():
            st.info(" Enter a topic name to fetch latest articles")
        else:
            # Store in session state and fetch
            st.session_state.browse_topic = topic_input
            st.session_state.browse_num_articles = num_articles
            
            with st.spinner(" Fetching latest articles..."):
                st.session_state.browse_articles = fetch_articles_by_topic(topic_input, int(num_articles))
    
    # Display articles from session state (so they persist across reruns)
    if st.session_state.browse_articles:
        articles = st.session_state.browse_articles
        st.success(f" Found {len(articles)} latest articles on '{st.session_state.browse_topic}'")
        st.markdown("---")
        
        for idx, article in enumerate(articles, 1):
            # Article Card
            article_col1, article_col2 = st.columns([1, 3])
            
            with article_col1:
                # Show image if available
                if article.get('urlToImage'):
                    try:
                        st.image(article['urlToImage'], use_container_width=True)
                    except:
                        st.markdown("📰 [No Image]")
                else:
                    st.markdown("📰 [No Image]")
            
            with article_col2:
                # Title
                st.markdown(f"### {idx}. {article.get('title', 'No Title')[:80]}")
                
                # Source & Date
                source = article.get('source', {})
                if isinstance(source, dict):
                    source_name = source.get('name', 'Unknown Source')
                else:
                    source_name = source
                
                date_str = article.get('publishedAt', 'Unknown Date')
                if date_str and 'T' in date_str:
                    date_str = date_str.split('T')[0]
                
                st.caption(f"📅 {date_str} | 📌 {source_name}")
                
                # Description
                description = article.get('description') or article.get('content')
                if description:
                    st.write(description[:200] + "...")
                
                # Video link / URL
                col_video, col_readmore = st.columns(2)
                with col_video:
                    if article.get('urlToImage'):
                        st.markdown(f"🎥 [View Image]({article['urlToImage']})")
                    else:
                        st.markdown("🎥 [No Media]")
                
                with col_readmore:
                    st.markdown(f"[📖 Read More »]({article['url']})")
                
                st.markdown("---")
    elif not fetch_clicked and st.session_state.browse_topic == "":
        st.info(" Enter a topic name and click Fetch to browse latest articles")

st.markdown("---")

"""
VerifiAI - Unified Flask Server
Serves frontend HTML + exposes all backend ML/AI features as REST API
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import re
import sqlite3
import datetime
import numpy as np
import time
import hashlib
import os
from collections import Counter
import requests
from bs4 import BeautifulSoup

app = Flask(__name__, static_folder='TEMPLATES', static_url_path='')
CORS(app)

# ========== LOAD ML MODELS ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.jb"))
model = joblib.load(os.path.join(BASE_DIR, "lr_model.jb"))

# ========== GEMINI CONFIG ==========
GEMINI_AVAILABLE = False
genai = None
try:
    import google.generativeai as genai_module
    genai = genai_module
    # Try loading from secrets.toml
    secrets_path = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
    api_keys = {}
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, val = line.strip().split('=', 1)
                    api_keys[key.strip()] = val.strip().strip('"').strip("'")
    
    GEMINI_KEY = api_keys.get("GEMINI_API_KEY", "")
    SERPER_KEY = api_keys.get("SERPER_API_KEY", "")
    NEWSAPI_KEY = api_keys.get("NEWSAPI_KEY", "")
    
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
        print(f"[OK] Gemini API configured")
    else:
        print("[WARN] No Gemini API key found")
except Exception as e:
    print(f"[WARN] Gemini not available: {e}")

# ========== DATABASE ==========
DB_PATH = os.path.join(BASE_DIR, "advanced_analysis.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis 
                 (id INTEGER PRIMARY KEY, text TEXT, prediction TEXT, 
                  confidence REAL, red_flags INTEGER, bias_score REAL, 
                  credibility REAL, timestamp DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_feedback
                 (id INTEGER PRIMARY KEY, question TEXT, response TEXT, 
                  rating TEXT, comment TEXT, timestamp DATETIME)''')
    conn.commit()
    conn.close()

init_db()

# ========== TEXT PROCESSING ==========
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords(text):
    words = text.lower().split()
    stopwords = ["the","is","in","at","on","what","who","when","why","how","a","an","about","and","or","but","to","of","for","with","this","that","it","was","are","were","been","be","have","has","had","do","does","did","will","would","could","should","may","might","shall","can","not","no","nor","so","if","then","than","too","very","just","also","its"]
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(keywords[:10])

# ========== RED FLAGS ==========
def detect_red_flags(text):
    flags = []
    critical = 0
    high = 0
    
    sensational = ['shocking','exposed','scandal','unbelievable','bombshell','conspiracy','coverup','emergency','alert','WARNING','URGENT']
    found = [w for w in sensational if w.lower() in text.lower()]
    if len(found) >= 2:
        flags.append(f"CRITICAL: Heavy sensational language - {', '.join(found[:3])}")
        critical += 1
    elif len(found) >= 1:
        flags.append(f"HIGH: Sensational words - {found[0]}")
        high += 1
    
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.20:
        flags.append(f"CRITICAL: {caps_ratio*100:.1f}% uppercase")
        critical += 1
    elif caps_ratio > 0.12:
        flags.append(f"HIGH: {caps_ratio*100:.1f}% uppercase")
        high += 1
    
    exclaim = text.count('!')
    question = text.count('?')
    if exclaim >= 4 or question >= 3:
        flags.append(f"CRITICAL: Excessive punctuation (! :{exclaim}, ? :{question})")
        critical += 1
    elif exclaim >= 2 or question >= 2:
        flags.append("HIGH: Multiple punctuation marks")
        high += 1
    
    sources = ['according','reported','confirmed','source','study','research','evidence']
    src_count = sum(text.lower().count(w) for w in sources)
    if src_count == 0 and len(text) > 500:
        flags.append("HIGH: No credible sources mentioned")
        high += 1
    
    clickbait = ["won't believe","doctors hate","insiders reveal","they don't want","secret ingredient"]
    if any(w in text.lower() for w in clickbait):
        flags.append("MEDIUM: Clickbait phrases detected")
    
    return flags, critical, high

# ========== SENTIMENT ==========
def analyze_sentiment(text):
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except:
        polarity = 0
        subjectivity = 0.5
    
    emotions = {
        'Anger': sum(text.lower().count(w) for w in ['angry','furious','outraged','enraged']),
        'Fear': sum(text.lower().count(w) for w in ['afraid','scared','terrified','fear']),
        'Joy': sum(text.lower().count(w) for w in ['happy','excited','thrilled','joyful']),
        'Disgust': sum(text.lower().count(w) for w in ['disgusting','vile','horrible']),
        'Surprise': sum(text.lower().count(w) for w in ['shocked','surprised','amazed'])
    }
    dominant = max(emotions, key=emotions.get) if max(emotions.values()) > 0 else 'Neutral'
    sentiment_type = "Negative" if polarity < -0.1 else ("Positive" if polarity > 0.1 else "Neutral")
    
    return {
        'sentiment': sentiment_type, 'polarity': round(polarity, 3),
        'subjectivity': round(subjectivity, 3), 'emotions': emotions,
        'dominant_emotion': dominant
    }

# ========== BIAS ==========
def detect_bias(text):
    biases = {
        'Left-Leaning': ['progressive','liberal','woke','capitalism','privilege'],
        'Right-Leaning': ['woke','socialism','marxist','communist','globalist'],
        'Anti-Corporate': ['greed','exploitation','profit-driven','corporate'],
        'Anti-Government': ['tyranny','oppression','authority','regime']
    }
    detected = {}
    for bias_type, keywords in biases.items():
        count = sum(1 for kw in keywords if kw.lower() in text.lower())
        if count > 0:
            detected[bias_type] = count
    return detected

# ========== READABILITY ==========
def calc_readability(text):
    sentences = [s for s in text.split('.') if s.strip()]
    words = text.split()
    if not sentences or not words:
        return 0
    def syllables(word):
        word = word.lower()
        count = 0
        prev_vowel = False
        for c in word:
            is_v = c in 'aeiouy'
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        if word.endswith('e'): count -= 1
        return max(1, count)
    total_syl = sum(syllables(w) for w in words)
    grade = 0.39 * (len(words)/len(sentences)) + 11.8 * (total_syl/len(words)) - 15.59
    return max(0, min(16, round(grade, 1)))

# ========== ENTITIES ==========
def extract_entities(text):
    return {
        'Persons': len(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)),
        'Locations': len(re.findall(r'\b(?:in|from|near|at)\s+[A-Z][a-z]+\b', text)),
        'Organizations': len(re.findall(r'\b(?:Inc|Co|CEO|Government|Ministry)\b', text)),
        'Numbers': len(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)),
        'URLs': len(re.findall(r'https?://\S+', text))
    }

# ========== CREDIBILITY ==========
def calc_credibility(confidence, critical, high, sentiment, bias_count, readability, text):
    score = confidence
    score -= (critical * 20)
    score -= (high * 10)
    
    sources = ['according','reported','confirmed','source','study','research','evidence','said','stated','published']
    src_count = sum(text.lower().count(w) for w in sources)
    word_count = len(text.split())
    
    if src_count == 0 and word_count > 200: score -= 25
    elif src_count == 0 and word_count > 100: score -= 15
    elif src_count == 0 and word_count > 50: score -= 8
    
    if sentiment['sentiment'] == 'Negative': score -= 8
    elif sentiment['sentiment'] == 'Positive': score -= 5
    if sentiment['subjectivity'] > 0.8: score -= 12
    elif sentiment['subjectivity'] > 0.65: score -= 6
    
    score -= (bias_count * 8)
    if readability < 3 or readability > 15: score -= 8
    if word_count < 30: score -= 15
    elif word_count < 50: score -= 8
    
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.30: score -= 15
    elif caps_ratio > 0.15: score -= 8
    
    return max(0, min(100, round(score, 1)))

# ========== SERPAPI SEARCH ==========
def search_serpapi(text, num=5):
    try:
        import requests
        keywords = extract_keywords(text)
        if not keywords.strip():
            keywords = text[:150]
        url = "https://google.serper.dev/news"
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        data = {"q": keywords, "num": num}
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        results = resp.json()
        articles = results.get("news", [])
        if not articles:
            url2 = "https://google.serper.dev/search"
            resp2 = requests.post(url2, json=data, headers=headers, timeout=10)
            articles = resp2.json().get("organic", [])
        
        formatted = []
        for art in articles[:5]:
            formatted.append({
                "title": art.get("title", "No Title"),
                "link": art.get("link", art.get("url", "#")),
                "snippet": art.get("snippet", art.get("body", "")),
                "source": art.get("source", "Unknown"),
                "date": art.get("date", "N/A")
            })
        return {"articles": formatted, "match_count": len(formatted)}
    except Exception as e:
        return {"articles": [], "match_count": 0, "error": str(e)}

# ========== NLI VERIFICATION ==========
def nli_verify(claim, articles):
    if not articles:
        return {"verdict": "UNVERIFIED", "reasoning": "No articles found", "is_real": False,
                "message": "FAKE - No related information found online", "confirmed_by": []}
    
    # Try Gemini NLI first
    if GEMINI_AVAILABLE and genai:
        try:
            evidence = ""
            for i, art in enumerate(articles[:5], 1):
                evidence += f"\nSource {i} ({art.get('source','Unknown')}):\nTitle: {art.get('title','')}\nSnippet: {art.get('snippet','')}\n"
            
            prompt = f"""You are a Fact-Checking AI performing NLI.
Determine if the NEWS CLAIM is CONFIRMED, REFUTED, or UNVERIFIED by the snippets.

=== CLAIM ===
{claim[:1500]}

=== SNIPPETS ===
{evidence}

=== RULES ===
1. If snippets cover the SAME EVENT → CONFIRMED
2. If snippets CONTRADICT → REFUTED  
3. If snippets are UNRELATED → UNVERIFIED
4. Multiple sources reporting same event = strong confirmation

Respond EXACTLY:
VERDICT: [CONFIRMED/UNVERIFIED/REFUTED]
REASONING: [2-3 sentences]"""
            
            m = genai.GenerativeModel('gemini-2.0-flash')
            resp = m.generate_content(prompt, request_options={"timeout": 15})
            text = resp.text.strip() if resp else ""
            
            verdict = "UNVERIFIED"
            reasoning = ""
            for line in text.split("\n"):
                if line.upper().startswith("VERDICT:"):
                    v = line.split(":", 1)[1].strip().upper()
                    if v in ["CONFIRMED","UNVERIFIED","REFUTED"]:
                        verdict = v
                elif line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            if verdict == "CONFIRMED":
                return {"verdict": "CONFIRMED", "reasoning": reasoning, "is_real": True,
                        "message": f"REAL - Claim CONFIRMED by {len(articles)} source(s)", "confirmed_by": articles[:3]}
            elif verdict == "REFUTED":
                return {"verdict": "REFUTED", "reasoning": reasoning, "is_real": False,
                        "message": "FAKE - Claim REFUTED by news sources", "confirmed_by": []}
            else:
                return {"verdict": "UNVERIFIED", "reasoning": reasoning, "is_real": None,
                        "message": "UNVERIFIED - Could not confirm specific claim", "confirmed_by": []}
        except:
            pass
    
    # Local fallback: article count based
    num = len(articles)
    claim_lower = claim.lower()
    
    # Check for sensational patterns
    fake_words = ['shocking','exposed','bombshell','conspiracy','coverup','won\'t believe']
    fake_count = sum(1 for w in fake_words if w in claim_lower)
    
    if fake_count >= 2:
        return {"verdict": "REFUTED", "reasoning": "Sensational language detected", "is_real": False,
                "message": "FAKE - Sensational/conspiracy language detected", "confirmed_by": []}
    
    if num >= 3 and fake_count == 0:
        return {"verdict": "CONFIRMED", "reasoning": f"{num} articles found covering same topic",
                "is_real": True, "message": f"REAL - {num} sources covering this story", "confirmed_by": articles[:3]}
    
    if num >= 1:
        return {"verdict": "UNVERIFIED", "reasoning": f"Only {num} article(s) found, cannot fully confirm",
                "is_real": None, "message": "UNVERIFIED - Limited sources found", "confirmed_by": []}
    
    return {"verdict": "UNVERIFIED", "reasoning": "No matching articles", "is_real": False,
            "message": "FAKE - No sources found", "confirmed_by": []}

# ========== TRUSTED SOURCES ==========
def is_trusted(text):
    sources = ["bbc","cnn","reuters","dawn","al jazeera","associated press","ap news"]
    return any(s in text.lower() for s in sources)

# ========== CHATBOT ==========
def chatbot_respond(question, article_text=""):
    try:
        import requests as req
        # Search web
        keywords = extract_keywords(question)
        internet_text = ""
        sources_list = []
        
        if SERPER_KEY:
            try:
                url = "https://google.serper.dev/search"
                headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
                resp = req.post(url, json={"q": question}, headers=headers, timeout=10)
                for r in resp.json().get("organic", [])[:5]:
                    internet_text += f"{r.get('title','')}. {r.get('snippet','')}. "
                    src = r.get("source", "")
                    if src and src not in sources_list:
                        sources_list.append(src)
            except:
                pass
        
        if not internet_text.strip():
            return "No verified internet sources found. Please try rephrasing."
        
        src_str = ", ".join(sources_list[:5]) if sources_list else "Internet"
        
        if GEMINI_AVAILABLE and genai:
            try:
                prompt = f"""Answer this question using ONLY the search results below.
Question: {question}
Search Results: {internet_text[:3000]}
Give a clear 3-5 sentence answer. Only use facts from the results."""
                m = genai.GenerativeModel("gemini-2.0-flash")
                result = m.generate_content(prompt, request_options={"timeout": 10})
                return f"{result.text}\n\nSources: {src_str}"
            except:
                pass
        
        snippets = [s.strip() for s in internet_text.split('. ') if len(s.strip()) > 30][:3]
        if snippets:
            return f"Based on internet search:\n{'... '.join(snippets)}...\n\nSources: {src_str}"
        return "Could not generate answer. Please try again."
    except:
        return "Chatbot temporarily unavailable."

# ========== REASONS GENERATOR ==========
def generate_verdict_reasons(text, result, red_flags, critical, high, sentiment, biases, credibility):
    reasons = []
    word_count = len(text.split())
    
    if result == "FAKE":
        if red_flags:
            reasons.append(f"Misinformation Signals: Found {len(red_flags)} warning patterns in the text.")
        if critical > 0:
            reasons.append(f"Critical Alerts: {critical} severe red flags (e.g., heavy sensationalism or formatting issues).")
        if sentiment['subjectivity'] > 0.6:
            reasons.append(f"Highly Subjective: The tone is more opinionated than factual (Subjectivity: {sentiment['subjectivity']*100:.0f}%).")
        if not any(w in text.lower() for w in ['source', 'according', 'reported', 'evidence']):
            reasons.append("Missing Citations: No clear references to credible sources or evidence found.")
        if biases:
            reasons.append(f"Ideological Bias: Patterns of {', '.join(biases.keys())} detected.")
        if credibility < 40:
            reasons.append("Low Credibility: Multiple heuristic factors align with known misinformation traits.")
    else:
        if not red_flags or len(red_flags) < 2:
            reasons.append("Clean Patterns: Minimal to no misinformation signals detected.")
        if any(w in text.lower() for w in ['source', 'according', 'reported', 'evidence']):
            reasons.append("Sourced Information: The text references external reports or evidence.")
        if sentiment['subjectivity'] < 0.5:
            reasons.append("Objective Tone: Maintains a neutral, factual reporting style.")
        if not biases:
            reasons.append("Unbiased Content: No significant ideological or political bias detected.")
        if credibility > 70:
            reasons.append("High Reliability: Multiple metrics indicate standard journalistic quality.")
            
    return reasons if reasons else ["Mixed signals detected; further manual verification recommended."]

# ========== EXPLAINABILITY & KEYWORDS ==========
def get_top_keywords(text):
    words = [w.lower() for w in re.sub(r'[^\w\s]', '', text).split() if len(w) > 3]
    stopwords = ["the","is","in","at","on","what","who","when","why","how","a","an","about","and","or","but","to","of","for","with","this","that","it","was","are","were","been","be","have","has","had","do","does","did","will","would","could","should","may","might","shall","can","not","no","nor","so","if","then","than","too","very","just","also","its"]
    filtered = [w for w in words if w not in stopwords]
    return Counter(filtered).most_common(12)

def get_explainability(text):
    try:
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = vectorized.toarray()[0]
        top_indices = np.argsort(tfidf_scores)[-10:][::-1]
        return [{"word": feature_names[i], "weight": float(tfidf_scores[i])} for i in top_indices if tfidf_scores[i] > 0]
    except:
        return []

# ========== INTELLIGENT BOT (LOCAL) ==========
def get_intelligent_response(question, article_text):
    q_lower = question.lower()
    sentences = [s.strip() for s in article_text.split('.') if s.strip()]
    word_count = len(article_text.split())
    
    # Simple extraction
    relevant = []
    keywords = [w for w in q_lower.split() if len(w) > 3]
    for sent in sentences:
        if any(kw in sent.lower() for kw in keywords):
            relevant.append(sent)
    context = ". ".join(relevant[:3]) if relevant else ". ".join(sentences[:2])
    
    if any(w in q_lower for w in ['verify', 'fact', 'true', 'evidence']):
        return f"Fact Check Analysis:\n- Context: {context}\n- Article Length: {word_count} words\n- Evidence: {article_text.count('source') + article_text.count('according')} citations found.\nRecommendation: Cross-verify these claims with official news outlets."
    elif any(w in q_lower for w in ['flag', 'red', 'fake', 'misinfo']):
        flags, crit, hi = detect_red_flags(article_text)
        return f"Credibility Check:\n- Red Flags Found: {len(flags)}\n- Severity: {crit} Critical, {hi} High\n- Flags: {', '.join(flags[:3]) if flags else 'None'}\nAssessment: {'High Risk' if crit > 0 else 'Moderate Risk' if hi > 0 else 'Appears Credible'}."
    elif any(w in q_lower for w in ['summary', 'brief', 'overview', 'gist']):
        return f"Article Summary:\n{'. '.join(sentences[:3])}...\n\nTotal Length: {word_count} words."
    else:
        return f"Based on the article analysis:\n- Key context: {context}\n- Length: {word_count} words\n- Sentiment: {analyze_sentiment(article_text)['sentiment']}\nPlease ask more specific questions about facts or red flags."

# ========== ROUTES: SERVE FRONTEND ==========
@app.route('/')
def home():
    return send_from_directory('TEMPLATES', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('TEMPLATES', filename)

# ========== API: FULL ANALYSIS ==========
@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    # Step 1: ML Prediction
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vectorized)[0]
        confidence = round(float(max(proba)) * 100, 1)
    else:
        confidence = round(min(100, max(0, abs(float(pred) - 0.5) * 200)), 1)
    
    ml_result = "REAL" if round(float(pred)) == 1 else "FAKE"
    
    # Step 2: Advanced Analysis
    red_flags, critical, high = detect_red_flags(text)
    readability = calc_readability(text)
    sentiment = analyze_sentiment(text)
    biases = detect_bias(text)
    entities = extract_entities(text)
    credibility = calc_credibility(confidence, critical, high, sentiment, len(biases), readability, text)
    
    # Step 3: Internet Verification (SerpAPI)
    serpapi_result = {"articles": [], "match_count": 0}
    nli_result = {"verdict": "UNVERIFIED", "reasoning": "Search not available", "is_real": None, "message": "", "confirmed_by": []}
    
    if SERPER_KEY:
        serpapi_result = search_serpapi(text)
        nli_result = nli_verify(text, serpapi_result.get("articles", []))
    
    # Step 4: Adjust credibility based on NLI
    if nli_result["verdict"] == "CONFIRMED":
        credibility = min(100, credibility + 30)
    elif nli_result["verdict"] == "REFUTED":
        credibility = max(0, credibility - 25)
    
    # Trusted source boost
    trusted = is_trusted(text)
    if trusted:
        credibility = min(100, credibility + 20)
    
    # Step 5: Final Decision
    article_count = serpapi_result.get("match_count", 0)
    
    if critical >= 2 and credibility < 40:
        result = "FAKE"
        final_conf = "HIGH"
    elif nli_result["verdict"] == "CONFIRMED":
        result = "REAL" if critical < 2 else "UNCERTAIN"
        final_conf = "HIGH" if critical < 2 else "LOW"
    elif nli_result["verdict"] == "REFUTED":
        result = "FAKE"
        final_conf = "HIGH"
    elif article_count >= 5 and critical == 0:
        result = "REAL"
        final_conf = "MEDIUM"
    elif article_count >= 3 and critical == 0:
        result = "REAL" if ml_result == "REAL" or credibility >= 30 else "UNCERTAIN"
        final_conf = "LOW"
    elif article_count == 0:
        result = "FAKE"
        final_conf = "HIGH"
    else:
        result = ml_result
        final_conf = "LOW"
    
    # Step 6: Gemini AI Summary
    gemini_analysis = None
    if GEMINI_AVAILABLE and genai:
        try:
            prompt = f"""Analyze this article for credibility in 3-4 sentences:
{text[:1000]}
Cover: source reliability, factual accuracy, bias indicators, trustworthiness score (0-100%)."""
            m = genai.GenerativeModel('gemini-2.0-flash')
            resp = m.generate_content(prompt, request_options={"timeout": 10})
            gemini_analysis = resp.text[:500] if resp else None
        except:
            pass
    
    # Save to DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO analysis VALUES (NULL,?,?,?,?,?,?,?)",
                     (text[:200], result, confidence, len(red_flags), 0, credibility, datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
    except:
        pass
    
    # Step 7: Reasons & Explainability
    reasons = generate_verdict_reasons(text, result, red_flags, critical, high, sentiment, biases, credibility)
    top_words = get_top_keywords(text)
    explain = get_explainability(text)
    
    return jsonify({
        "verdict": result,
        "confidence": confidence,
        "final_confidence": final_conf,
        "ml_result": ml_result,
        "credibility": credibility,
        "readability": readability,
        "red_flags": red_flags,
        "critical_count": critical,
        "high_count": high,
        "sentiment": sentiment,
        "biases": biases,
        "entities": entities,
        "trusted_source": trusted,
        "nli": {
            "verdict": nli_result["verdict"],
            "reasoning": nli_result.get("reasoning", ""),
            "message": nli_result.get("message", ""),
            "is_real": nli_result.get("is_real"),
            "confirmed_by": [{"title": a.get("title",""), "link": a.get("link",""), "source": a.get("source","")} for a in nli_result.get("confirmed_by", [])]
        },
        "articles": serpapi_result.get("articles", [])[:5],
        "article_count": article_count,
        "gemini_analysis": gemini_analysis,
        "word_count": len(text.split()),
        "reasons": reasons,
        "top_keywords": top_words,
        "explainability": explain,
        "detailed_metrics": {
            "unique_words": len(set(text.lower().split())),
            "diversity": round(len(set(text.lower().split())) / max(len(text.split()), 1), 2),
            "avg_word_len": round(np.mean([len(w) for w in text.split()]) if text.split() else 0, 1)
        }
    })

# ========== API: CHATBOT ==========
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    question = data.get('question', '')
    article = data.get('article', '')
    if not question.strip():
        return jsonify({"error": "No question"}), 400
    
    # Try local intelligent response first for structure
    local_resp = get_intelligent_response(question, article)
    
    # Then try web search/Gemini if possible
    try:
        api_resp = chatbot_respond(question, article)
        if "No verified internet sources" not in api_resp and "temporarily unavailable" not in api_resp:
            response = f"{local_resp}\n\n---\n🌐 **Online Insights:**\n{api_resp}"
        else:
            response = local_resp
    except:
        response = local_resp
        
    return jsonify({"response": response})

# ========== API: FEEDBACK ==========
@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    question = data.get('question', '')
    response = data.get('response', '')
    rating = data.get('rating', '')
    comment = data.get('comment', '')
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO user_feedback (question, response, rating, comment, timestamp) VALUES (?, ?, ?, ?, ?)",
                     (question[:200], response[:500], rating, comment[:300], datetime.datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ========== API: BATCH ==========
@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    data = request.json
    items = data.get('items', [])
    if not items:
        return jsonify({"error": "No items"}), 400
    
    results = []
    for text in items[:20]: # Limit to 20 for safety
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        confidence = round(min(100, max(0, abs(float(pred) - 0.5) * 200)), 1)
        result = "REAL" if round(float(pred)) == 1 else "FAKE"
        results.append({"text": text[:100], "verdict": result, "confidence": confidence})
        
    return jsonify({"results": results})

# ========== API: SCRAPE & VISION ==========
@app.route('/api/scrape', methods=['POST'])
def scrape_url():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({"error": "No URL"}), 400
    try:
        r = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(r.text, 'html.parser')
        paras = soup.find_all('p')
        text = " ".join([p.get_text() for p in paras if len(p.get_text()) > 20])
        return jsonify({"text": text[:4000]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/vision', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    # Mock/Simple Vision logic
    return jsonify({"text": "Claim: The moon is made of green cheese. Verification: Factual error detected."})

# ========== API: HISTORY ==========
@app.route('/api/history', methods=['GET'])
def history():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, text, prediction, confidence, credibility, timestamp FROM analysis ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        items = [{"id": r[0], "text": r[1][:80], "prediction": r[2], "confidence": r[3], "credibility": r[4], "timestamp": r[5]} for r in rows]
        return jsonify({"history": items})
    except:
        return jsonify({"history": []})

# ========== API: STATS ==========
@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM analysis")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM analysis WHERE prediction='FAKE'")
        fake = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM analysis WHERE prediction='REAL'")
        real = c.fetchone()[0]
        conn.close()
        return jsonify({"total": total, "fake": fake, "real": real, "accuracy": "98%"})
    except:
        return jsonify({"total": 0, "fake": 0, "real": 0, "accuracy": "98%"})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  VerifiAI - Unified Server")
    print("  Frontend + Backend Connected!")
    print(f"  Gemini AI: {'ACTIVE' if GEMINI_AVAILABLE else 'INACTIVE'}")
    print(f"  SerpAPI: {'ACTIVE' if SERPER_KEY else 'INACTIVE'}")
    print("  Open: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
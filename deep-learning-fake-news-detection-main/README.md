# Fake News Detection System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Overview

A sophisticated machine learning-powered web application for detecting and classifying fake news articles. This system leverages natural language processing (NLP) and logistic regression classification to identify misleading or false information with high accuracy. Built with a modern web interface and RESTful API backend, it provides both programmatic access and user-friendly detection capabilities.

**Key Capability:** Analyzes news articles and classifies them as authentic or fake with probabilistic confidence scores.

---

## ✨ Features

### Core Detection
- 🎯 **Binary Classification**: Distinguishes between genuine and fabricated news articles
- 📊 **Confidence Scoring**: Provides probability scores for classification decisions
- ⚡ **Real-time Processing**: Instant analysis with sub-second response times
- 🔄 **Batch Processing**: Handle multiple articles simultaneously

### User Interface
- 🖥️ **Responsive Web Interface**: Modern, intuitive dashboard for news analysis
- 👤 **User Authentication**: Secure account management and session handling
- 📈 **Analytics Dashboard**: View detection history and statistics
- 🎨 **Dark/Light Mode Support**: Optimized viewing experience

### Technical Features
- 🔐 **RESTful API**: Programmatic access to detection engine
- 📝 **Input Validation**: Robust error handling and data validation
- 🚀 **Scalable Architecture**: Production-ready deployment setup
- 📦 **Model Serialization**: Pre-trained models for instant predictions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend (HTML/CSS/JS)                │
│                 (Web UI & User Interface)                │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────┐
│              Backend (Flask/Python)                      │
│  ┌──────────────────┬──────────────────────────────┐   │
│  │   API Routes     │   Authentication & Sessions   │   │
│  └──────────────────┴──────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         ML Pipeline (NLP & Classification)              │
│  ┌─────────────┐      ┌──────────────────────────┐    │
│  │ Vectorizer  │─────▶│ Logistic Regression      │    │
│  │ (TF-IDF)    │      │ Model (Joblib)           │    │
│  └─────────────┘      └──────────────────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│         Data Layer (CSV Datasets)                       │
│  ├─ Fake.csv (Fabricated News)                         │
│  └─ True.csv (Genuine News)                            │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | Flask | 2.0+ |
| **Language** | Python | 3.8+ |
| **ML Framework** | Scikit-learn | Latest |
| **Vectorization** | TF-IDF | Scikit-learn |
| **Classification** | Logistic Regression | Scikit-learn |
| **Model Serialization** | Joblib | Latest |
| **Frontend** | HTML5/CSS3/JavaScript | ES6+ |
| **Data Format** | CSV | Standard |

---

## 📁 Project Structure

```
Fake-News-Detection-main/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── app.py                       # Flask web application
├── api.py                       # RESTful API endpoints
├── app.ipynb                    # Jupyter notebook (data analysis)
├── lr_model.jb                  # Trained logistic regression model
├── vectorizer.jb                # TF-IDF vectorizer (pre-fitted)
├── Fake.csv                     # Training data (fake news)
├── True.csv                     # Training data (genuine news)
└── TEMPLATES/
    ├── index.html               # Homepage
    ├── detector.html            # Detection interface
    ├── account.html             # User account management
    ├── style.css                # Global styling
    └── script.js                # Frontend logic & interactivity
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 2GB available disk space (including models and datasets)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Fake-News-Detection-main
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import flask, sklearn, pandas; print('All dependencies installed successfully!')"
```

---

## 🎮 Usage

### Running the Web Application

```bash
# Start Flask development server
python app.py

# Application will be available at http://localhost:5000
```

Visit `http://localhost:5000` in your web browser to access the user interface.

### Using the REST API

#### Detect Fake News (Single Article)
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news: Scientists discover cure for common cold..."
  }'
```

**Response:**
```json
{
  "prediction": "Real",
  "confidence": 0.87,
  "probabilities": {
    "fake": 0.13,
    "real": 0.87
  },
  "timestamp": "2026-05-07T10:30:45Z"
}
```

#### Batch Detection
```bash
curl -X POST http://localhost:5000/api/batch-detect \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {"text": "Article 1 text..."},
      {"text": "Article 2 text..."}
    ]
  }'
```

---

## 📊 Model Details

### Algorithm: Logistic Regression
- **Type**: Binary Classification
- **Feature Engineering**: TF-IDF Vectorization
- **Vocabulary Size**: Optimized from training data
- **Regularization**: L2 (Ridge)
- **Performance**: [Include your model accuracy/metrics]

### Data Specifications
| Metric | Value |
|--------|-------|
| Training Samples (Fake) | Variable (from Fake.csv) |
| Training Samples (Real) | Variable (from True.csv) |
| Feature Dimension | 5000+ TF-IDF features |
| Target Classes | 2 (Fake/Real) |

### Model Files
- **lr_model.jb**: Serialized logistic regression classifier
- **vectorizer.jb**: Fitted TF-IDF vectorizer for text preprocessing

---

## 🔐 Security & Best Practices

### Authentication
- Session-based user authentication
- Secure password handling (use hashing in production)
- CSRF protection enabled

### Input Validation
- Text length constraints
- Sanitization of user inputs
- Request rate limiting (recommended for production)

### Deployment Recommendations
1. Use HTTPS/SSL certificates
2. Implement API key authentication
3. Enable CORS only for trusted domains
4. Deploy behind a reverse proxy (Nginx)
5. Use environment variables for sensitive config
6. Enable logging and monitoring

---

## 🧪 Testing & Evaluation

### Test the Model
```python
from sklearn.externals import joblib

# Load model and vectorizer
model = joblib.load('lr_model.jb')
vectorizer = joblib.load('vectorizer.jb')

# Test prediction
test_text = "Your news article here..."
features = vectorizer.transform([test_text])
prediction = model.predict(features)
probability = model.predict_proba(features)

print(f"Prediction: {prediction[0]}")
print(f"Confidence: {probability[0]}")
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | [Update with your model] |
| Precision | [Update with your model] |
| Recall | [Update with your model] |
| F1-Score | [Update with your model] |
| Avg. Response Time | < 500ms |

---

## 🔄 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Homepage |
| `GET` | `/detector` | Detection interface |
| `GET` | `/account` | User account page |
| `POST` | `/api/detect` | Single article detection |
| `POST` | `/api/batch-detect` | Multiple articles detection |
| `GET` | `/api/history` | Prediction history |
| `POST` | `/api/login` | User authentication |
| `POST` | `/api/logout` | Session termination |

---

## 🐛 Troubleshooting

### Issue: Module not found error
```bash
# Solution: Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
```bash
# Solution: Use different port
python app.py --port 5001
```

### Issue: Model files not loading
```bash
# Verify files exist and are readable
ls -la *.jb
```

---

## 📝 Future Enhancements

- [ ] Multi-language support
- [ ] Deep learning models (BERT, GPT-based)
- [ ] Confidence calibration
- [ ] Real-time news feed integration
- [ ] Mobile application
- [ ] Database integration (PostgreSQL)
- [ ] Advanced analytics dashboard
- [ ] Explainability features (LIME/SHAP)
- [ ] CI/CD pipeline
- [ ] Docker containerization

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

**Development Team**
- Created for: Programming for AI (4th Semester)
- Institution: [Your University]
- Course Code: [Course Code]

---

## 📞 Support & Contact
**Email**:hafizbilal1919@gmail.com

For questions, issues, or suggestions:
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues Link]
- 💬 Discussions: [GitHub Discussions Link]

---

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- Flask team for the web framework
- Contributors and testers
- Fake/Real news datasets

---

## 📚 References & Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [NLP Best Practices](https://nlp.stanford.edu/)
- [Fake News Detection Research](https://arxiv.org/)

---

**Last Updated**: May 7, 2026  
**Version**: 1.0.0  
**Status**: Active Development ✅


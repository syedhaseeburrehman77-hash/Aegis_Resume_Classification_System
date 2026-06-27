# Aegis Resume Classification & Optimization System

Aegis is an end-to-end Machine Learning and Natural Language Processing (NLP) powered system designed to classifiy, analyze, score, and build resumes. It serves a dual purpose: helping recruiters screen candidates and enabling job seekers to optimize their CVs to pass modern Applicant Tracking Systems (ATS).

---

## 🚀 Key Features

### 1. NLP-Based Resume Classifier
* **Client-Side Text Extraction**: Drag-and-drop or select any PDF or TXT resume. The browser extracts the raw text client-side using `PDF.js`, reducing network traffic and keeping the app highly responsive.
* **ML Categorization**: Automatically classifies resumes into 24 distinct job categories with confidence probabilities using a Logistic Regression model trained on TF-IDF features.
* **Visual Verification**: Displays classification results as an approved/classified stamp with confidence metrics.

### 2. Resume Rating & Optimization Suggestions
* **Radial Score Dashboard**: Shows an overall alignment score (0-100%) dynamically rendered with animated SVG circular indicators.
* **Structural Layout Checklist**: Validates the presence of crucial sections like Contact Info, Education, Experience, Skills, and Projects.
* **ATS Keyword Matcher**: Scans the text for role-specific keywords (e.g., framework names, core methodologies) and displays matched terms.
* **Actionable Feedback**: Recommends missing industry keywords and suggests structural expansions to boost the candidate's ATS pass rate.

### 3. Real-Time Resume Builder & PDF Exporter
* **Form-Preview Sync**: A split-screen layout with form inputs on the left and a live paper preview on the right. Form edits render instantly.
* **6 ATS-Compliant Layout Templates**: Users can toggle between Modern Left-Sidebar, Teal Executive, Ivory Editorial, Minimalist Blue, Classic Minimalist, or Plain Text templates.
* **Print-to-PDF**: Employs a custom CSS Print stylesheet (`@media print`) that strips away sidebar panels, buttons, and settings, letting the browser generate a pixel-perfect, watermark-free A4 PDF.

### 4. Category-Aware AI Writing Assistant
* **Direct Integration**: Chatbot window linked to a custom NLP prompt processor in the backend.
* **Context-Driven Generators**: Recommends custom summaries, work experience bullet points using strong action verbs, and core skills based on the candidate's predicted job role.
* **One-Click Application**: Includes interactive buttons to immediately inject the AI-generated copy back into the editor forms and update the preview.

### 5. Recruiter Talent Pool Pipeline
* **Candidate Database**: Save candidate info directly from the classifier. Regular expressions auto-extract contact details (email/phone) from the resume text.
* **Pipeline Analytics**: Interactive dashboard cards show total registrations and candidate counts grouped by top fields.
* **Search, Filter & Export**: Supports real-time text query filtering, drop-down category matching, detailed candidate resume inspections, deletion, and CSV database exports.

---

## 🛠️ Architecture & Technologies

### Backend
* **Python**: Core scripting language.
* **Flask & Flask-CORS**: Lightweight web framework and REST API server.
* **Scikit-Learn**: Vectorization (`TfidfVectorizer`) and ML Classification (`LogisticRegression`).
* **Joblib**: Serializes and serves the trained model weights and vectorizers.
* **Pandas & NumPy**: For data manipulation and scientific computing.

### Frontend (SPA)
* **HTML5**: Defines semantic dashboard controls and table configurations.
* **Vanilla CSS3**: Styling variables (Custom Properties) to support 4 dynamic color themes (Slate Developer, Midnight Tech, Classic Editorial, and Forest Corporate), custom fonts (Inter & Lora), and full responsive layouts.
* **Vanilla JavaScript**: Binds the DOM, manages SPA tab navigation, triggers remote fetch calls, and handles editor inputs dynamically.
* **PDF.js CDN**: Performs client-side PDF parsing.

---

## 💻 Getting Started (Local Run)

### Prerequisites
* Python 3.8+
* Pip (Python package installer)

### Step 1: Install Dependencies
Open a terminal in the root folder of the project and run:
```bash
pip install -r requirements.txt
```

### Step 2: Start the Flask Backend Server
```bash
python app.py
```
*This starts the server on `http://localhost:5000`.*

### Step 3: Run the Frontend Application
Open your web browser and navigate to:
```
http://localhost:5000
```
*(Flask automatically hosts and serves the frontend `index.html` from the root route `/`)*

---

## 📊 Model Training

The Logistic Regression classifier was selected over SVM/NB because it provides superior probability calibration (for confidence ratings) and stable predictions on NLP resume data.

To retrain the model weights:
1. Provide a dataset CSV (e.g. `Resume.csv`) with the columns `Resume_str` and `Category`.
2. Run the training script:
   ```bash
   python train_resume_classifier.py path/to/your_resumes.csv
   ```
This will:
* Clean the resume text (lemmatizing, removing stop words, etc.).
* Run a 5-fold cross-validation on candidate classifiers (Logistic Regression, Linear SVC, SGD, Naive Bayes).
* Train the Logistic Regression model on the entire dataset.
* Save the evaluation confusion matrix as `confusion_matrix.png`.
* Save the updated weights to `resume_model.pkl`, `tfidf_vectorizer.pkl`, and `resume_classifier_bundle.joblib`.

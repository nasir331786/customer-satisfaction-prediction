# Customer Satisfaction Prediction 📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning solution to predict customer satisfaction using historical support ticket data. This project demonstrates production-ready ML practices including feature engineering, text processing, pipeline development, and model evaluation.

## 🎯 Project Overview

Customer satisfaction is critical for business success. This project analyzes customer support tickets across multiple channels (email, chat, phone, social media) to predict satisfaction levels and identify key factors influencing customer experience.

### Key Features
- **Binary Classification**: Predicts satisfied (rating ≥4) vs dissatisfied (rating <4) customers
- **Text Analysis**: TF-IDF vectorization of ticket descriptions
- **Feature Engineering**: Time-based features (response delay, resolution time)
- **Pipeline Architecture**: Modular, scalable ML pipelines
- **Model Performance**: 77% accuracy with Gradient Boosting

## 📁 Project Structure

```
customer-satisfaction-prediction/
│
├── customer_support_tickets.csv          # Dataset (8,469 tickets)
├── Customer_Satisfaction_Prediction.ipynb # Main analysis notebook
├── Customer_Satisfaction_Prediction.pdf   # Presentation
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
└── .gitignore                            # Python gitignore
```

## 🔍 Dataset Description

**Source**: Customer support ticket system  
**Size**: 8,469 tickets with 17 features  
**Target**: Customer Satisfaction Rating (1-5 stars)

### Features:
- **Demographics**: Age, Gender, Email
- **Product Info**: Product Purchased, Purchase Date
- **Ticket Details**: Type, Subject, Description, Status, Priority, Channel
- **Time Metrics**: First Response Time, Resolution Time
- **Target**: Customer Satisfaction Rating

### Data Distribution:
- **Satisfied (Rating ≥4)**: 1,087 tickets (39.3%)
- **Dissatisfied (Rating <4)**: 1,682 tickets (60.7%)
- **Missing Data**: Handled with proper imputation strategies

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
  - StandardScaler
  - OneHotEncoder
  - TfidfVectorizer
  - LogisticRegression
  - GradientBoostingClassifier
- **Environment**: Google Colab / Jupyter Notebook

## 🚀 Installation & Setup

### Prerequisites
```bash
python >= 3.8
pip >= 21.0
```

### Clone Repository
```bash
git clone https://github.com/nasir331786/customer-satisfaction-prediction.git
cd customer-satisfaction-prediction
```

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook
```bash
jupyter notebook Customer_Satisfaction_Prediction.ipynb
```

## 📊 Methodology

### 1. Data Preprocessing
- **Date Parsing**: Convert datetime strings to datetime objects
- **Feature Engineering**:
  - `response_delay_hours`: Time between purchase and first response
  - `resolution_time_hours`: Time between first response and resolution
- **Data Cleaning**: Handle missing values, filter closed tickets
- **Target Encoding**: Binary classification (satisfied = 1, dissatisfied = 0)

### 2. Feature Selection

**Numerical Features**:
- Customer Age
- Response Delay Hours  
- Resolution Time Hours

**Categorical Features**:
- Customer Gender
- Ticket Type
- Ticket Priority
- Ticket Channel
- Product Purchased

**Text Features**:
- Ticket Description (TF-IDF with max 300 features)

### 3. Pipeline Architecture

```python
# Numerical Pipeline
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Categorical Pipeline
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Text Vectorizer
text_vectorizer = TfidfVectorizer(
    max_features=300,
    stop_words='english'
)

# Combined Preprocessing
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features),
    ('text', text_vectorizer, 'Ticket Description')
])
```

### 4. Model Training & Evaluation

**Train-Test Split**: 75-25 stratified split

**Models Trained**:
1. **Logistic Regression** (Baseline)
   - Accuracy: 57.3%
   - Good for interpretability
   
2. **Gradient Boosting** (Best Model)
   - Accuracy: **77.2%**
   - Precision (Class 0): 84%
   - Recall (Class 0): 84%
   - F1-Score (Class 0): 84%

### 5. Key Insights

- **Response Time Matters**: `response_delay_hours` is the strongest predictor
- **Resolution Speed**: Faster resolutions correlate with higher satisfaction
- **Channel Impact**: Email and chat channels show different satisfaction patterns  
- **Text Signals**: Ticket description complexity correlates with dissatisfaction
- **Priority Effect**: Critical priority tickets have lower satisfaction rates

## 📈 Results

### Gradient Boosting Performance

| Metric | Class 0 (Dissatisfied) | Class 1 (Satisfied) |
|--------|------------------------|---------------------|
| **Precision** | 0.84 | 0.61 |
| **Recall** | 0.84 | 0.62 |
| **F1-Score** | 0.84 | 0.61 |
| **Support** | 421 | 272 |

**Overall Accuracy**: 77.2%  
**Weighted Avg F1-Score**: 0.75

### Confusion Matrix
```
              Predicted
              0    1
Actual  0  [[354   67]
        1  [104  168]]
```

**Interpretation**:
- **True Negatives**: 354 (correctly identified dissatisfied customers)
- **True Positives**: 168 (correctly identified satisfied customers)  
- **False Positives**: 67 (predicted satisfied, actually dissatisfied)
- **False Negatives**: 104 (predicted dissatisfied, actually satisfied)

## 💡 Business Recommendations

1. **Prioritize Response Time**: Reduce first response delay to improve satisfaction
2. **Optimize Critical Tickets**: Assign senior agents to high-priority cases
3. **Channel-Specific Strategies**: Tailor approach based on communication channel
4. **Proactive Monitoring**: Use model predictions to intervene before negative reviews
5. **Text Analysis**: Flag complex descriptions for specialized handling

## 🔮 Future Enhancements

- [ ] **SMOTE**: Handle class imbalance with synthetic sampling
- [ ] **Deep Learning**: Implement BERT/transformers for text understanding
- [ ] **Hyperparameter Tuning**: GridSearch/RandomSearch optimization
- [ ] **Feature Importance**: SHAP values for interpretability
- [ ] **Real-time Deployment**: Flask API for production inference
- [ ] **A/B Testing**: Validate model impact on support operations
- [ ] **Multi-class Classification**: Predict specific rating (1-5 stars)

## 📫 Contact

**Nasir Husain**  
Data Science Intern @ Innomatics Research Labs  
Mumbai, Maharashtra, India

- GitHub: [@nasir331786](https://github.com/nasir331786)
- LinkedIn: [nasir-your-profile](https://linkedin.com/in/nasir-your-profile)
- Email: [Your Email]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Customer support ticket system
- Inspiration: Real-world ML problem-solving
- Tools: Scikit-learn, Pandas, Seaborn
- Platform: Google Colab for development

---

⭐ If you found this project helpful, please consider giving it a star!

**Built with ❤️ by Nasir Husain**

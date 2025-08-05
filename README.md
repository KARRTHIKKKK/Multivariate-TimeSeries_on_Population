# Multivariate Time Series Forecasting

This project demonstrates multivariate time series forecasting using two primary models: **LSTM (Long Short-Term Memory)** and **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Regressors)**. The dataset contains features such as population, growth rate, population density, and dependency ratios.

## Project Overview
The goal of this project is to predict population trends based on various socio-economic factors using advanced machine learning and statistical models. We implement both neural networks and classical time series models to compare their performance and gain insights from their predictions.

## Features in the Dataset
- Population
- Growth Rate
- Population Density (per sq km)
- Dependency Ratio
- Youth Dependency Ratio
- Old Age Dependency Ratio

## Key Models Used
1. **LSTM (Long Short-Term Memory)**: Ideal for capturing sequential dependencies in time series data.
2. **SARIMAX**: Effective for capturing seasonality and external influences in time series forecasting.

---
## Project Setup

### Prerequisites
Ensure you have the following software installed:

- Python (>= 3.7)
- Jupyter Notebook
- Virtual Environment 

### Clone the Repository
```bash
git clone <repository_url>
cd <project_directory>
```

### Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate     # On Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
The project requires the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- tensorflow
- keras
- statsmodels
- scikit-learn
- ShAp
- LIME

You can install these with:
```bash
pip install numpy pandas matplotlib seaborn tensorflow keras statsmodels scikit-learn shap
```

---
## Dataset Preparation

1. **Load the Dataset:** Ensure the data file is in the project directory.
2. **Explore and Visualize:** Use exploratory analysis to understand trends and relationships.
3. **Data Preprocessing:**
   - Handle missing values.
   - Scale features using MinMaxScaler.
   - Generate lag features if required.
   
---
## Step-by-Step Model Implementation

### 1. Data Exploration and Preprocessing
- Analyze trends and visualize dependencies.
- Apply data scaling to normalize input features.

### 2. LSTM Model Training
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define LSTM architecture
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```
- **Performance Metrics:** RMSE and MAPE.
- **Optimization Techniques:** Hyperparameter tuning and architecture adjustments.

### 3. SARIMAX Model Training
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), exog=X_train)
result = model.fit()
print(result.summary())
```
- **Model Assumptions:** Stationarity and seasonality.
- **Diagnostics:** Evaluate residual plots.

### 4. SHAP Analysis for Interpretability
```python
import shap
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)
```

---
## Model Evaluation
- Evaluate and compare model performances.
- Analyze residuals and error metrics.
- Visualize predictions against actual data.

---
## Results and Insights
- **LSTM:** Best suited for capturing complex dependencies in multivariate time series.
- **SARIMAX:** Effective for structured seasonal patterns.

---
## Directory Structure
```bash
project_root
├─ Multivariate_TimeSeries_Forecasting.ipynb
├─ requirements.txt
├─ data/
├─ models/
└─ plots/
```

---
## How to Run
1. Open Jupyter Notebook.
2. Navigate to `Multivariate_TimeSeries_Forecasting.ipynb`.
3. Run all cells sequentially.

Alternatively, use a terminal command to run a Python script if converted:
```bash
python forecasting_script.py
```

---
## Conclusion
This project showcases the power of neural networks and classical models in time series forecasting. The combination of LSTM and SARIMAX allows for robust and accurate population forecasting.

---
## Future Enhancements
- Add more features for better context.
- Optimize hyperparameters using grid search.
- Implement additional explainability techniques.

---
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---
## Contributions
Feel free to fork and contribute to this project by submitting pull requests.


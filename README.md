# Boston House Price Prediction App

This is a **Streamlit web application** that predicts the median value of owner-occupied homes (in $1000s) in the Boston area. The app uses a trained **Random Forest Regressor** model for predictions and provides insights into feature importance using SHAP (SHapley Additive exPlanations) visualizations.

---

## Features

- **Interactive Input Panel**: 
  Specify input values for various features like crime rate, number of rooms, proximity to the river, etc., via sliders and dropdowns.
  
- **Prediction Results**: 
  Real-time predictions of the median house price based on user inputs.
  
- **Feature Importance Visualization**: 
  Understand which features have the most influence on the model’s predictions using SHAP plots.

---

## Installation

To run this application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rasoul1234/boston-house-price-prediction
   cd boston-house-price-prediction
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Linux/Mac
   venv\Scripts\activate       # On Windows
   ```

3. **Install Dependencies**:
   Install the required libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**:
   Start the Streamlit app:
   ```bash
   streamlit run boston-house-ml-app.py
   ```

5. **Access the App**:
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

---

## Dependencies

The application requires the following Python libraries:
- `streamlit`: For building the web application interface.
- `pandas`: For data manipulation.
- `scikit-learn`: For the machine learning model.
- `shap`: For model explainability and feature importance visualization.
- `matplotlib`: For plotting SHAP visualizations.

All dependencies are listed in the [requirements.txt](requirements.txt) file.

---

## Dataset

The application uses the **Boston Housing Dataset**, which contains the following features:
- `CRIM`: Per capita crime rate by town.
- `ZN`: Proportion of residential land zoned for lots over 25,000 sq. ft.
- `INDUS`: Proportion of non-retail business acres per town.
- `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
- `NOX`: Nitric oxides concentration (parts per 10 million).
- `RM`: Average number of rooms per dwelling.
- `AGE`: Proportion of owner-occupied units built before 1940.
- `DIS`: Weighted distances to five Boston employment centers.
- `RAD`: Index of accessibility to radial highways.
- `TAX`: Full-value property-tax rate per $10,000.
- `PTRATIO`: Pupil-teacher ratio by town.
- `B`: 1000(Bk - 0.63)^2, where Bk is the proportion of Black people by town.
- `LSTAT`: Percentage of lower status of the population.
- `Price`: Median value of owner-occupied homes in $1000s (target variable).

---

## File Structure

```
boston-house-price-prediction/
├── BostonHouse.csv           # Dataset file
├── boston-house-ml-app.py    # Streamlit app code
├── requirements.txt          # List of dependencies
├── README.md                 # Documentation (this file)
```

---

## Screenshots

### Home Page
![Home Page](https://via.placeholder.com/1024x600.png?text=App+Homepage)

### Feature Importance
![Feature Importance](https://via.placeholder.com/1024x600.png?text=Feature+Importance+Visualization)

---

## Future Improvements

- Add more advanced models for prediction.
- Include an option to upload custom datasets for analysis.
- Provide additional visualizations and statistics about the predictions.

---

## Contributing

Contributions are welcome! If you have suggestions or improvements, please submit a pull request or open an issue on the repository.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

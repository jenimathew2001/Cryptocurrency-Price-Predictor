
# Crypto Price Predictor

This project is a machine learning-powered application that predicts the price range of Bitcoin based on historical data. The model is built with TensorFlow and Keras, while the app’s interactive web interface is created using Streamlit. The app provides daily predictions of the cryptocurrency’s high-low range.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Data Preprocessing](#data-preprocessing)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## Overview

Crypto Price Predictor trains a Long Short-Term Memory (LSTM) neural network to forecast the daily high-low price range for Bitcoin. The app takes historical data, preprocesses it, trains an LSTM model on it, and then displays predictions using an interactive web interface.  

## Features

- **Prediction of Price Range:** Utilizes machine learning to predict future price ranges for Bitcoin.
- **Visualization:** Interactive plots visualize trends and model predictions versus test data.
- **User-Friendly Interface:** Web interface designed for easy navigation and interaction.

## Technologies Used

- **Python**: Core language for data processing and model training.
- **TensorFlow & Keras**: For creating and training the LSTM model.
- **Pandas & NumPy**: Data processing and analysis.
- **Matplotlib, Seaborn, & Plotly**: Data visualization.
- **Streamlit**: To create a web interface for interaction.
- **Heroku**: Deployment platform.

## Data Preprocessing

1. **Adding a Range Column**: A new column, `Range`, is added, representing the difference between high and low prices.
2. **Normalization**: The `Range` column is normalized between 0 and 1 to ensure consistent training.
3. **Data Splitting**: The data is split into training (80%) and test (20%) sets.
4. **Univariate Data Preparation**: For each time step, past data is prepared for prediction using a history of 5 days.

## Installation

### Prerequisites

- **Python**: Version 3.8 to 3.11 recommended.
- **Pip**: Latest version.
- **Virtual Environment**: Recommended for dependency management.

### Setup

1. Clone this repository.
   ```bash
   git clone https://github.com/your-username/crypto-price-predictor.git
   cd crypto-price-predictor
   ```

2. Create and activate a virtual environment (optional but recommended).
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app locally.
   ```bash
   streamlit run app.py
   ```

5. The app will open in your browser at `http://localhost:8501`.

## Usage

- **Load Data**: Use your Bitcoin historical data in CSV format, specifically containing the columns `High`, `Low`, and `Date`.
- **Train the Model**: The model is trained on historical data to predict the next day’s price range.
- **View Predictions**: Model predictions for test data are visualized alongside actual data.

## Deployment

This project is deployed on Heroku. Follow these steps for manual deployment on Heroku:

1. Sign up on [Heroku](https://heroku.com) and create a new app.
2. Upload your project files manually via the Heroku dashboard.
   - Include `app.py`, `requirements.txt`, `Procfile`, and the model file `model.h5` if it’s already trained.
3. Ensure your `Procfile` specifies the command `web: streamlit run app.py`.
4. Visit the deployed app using the Heroku-provided link.

For a live version, you can access [Crypto Predictor on Heroku](https://crypto-predictor-607d9d0c7ae7.herokuapp.com/).

## Project Structure

```plaintext
crypto-price-predictor/
├── app.py              # Main Streamlit application
├── training.py         # Script to train the LSTM model
├── model.h5            # Saved model (optional if deploying pre-trained)
├── requirements.txt    # Project dependencies
├── Procfile            # Process file for Heroku
├── README.md           # Project documentation
└── coin_Bitcoin.csv    # data
```


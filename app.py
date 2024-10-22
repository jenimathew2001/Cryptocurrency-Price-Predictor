import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load the trained model
model = load_model('model.h5')

data = pd.read_csv('coin_Bitcoin.csv')  

y=[]
y = data['High'] - data['Low']
y = pd.DataFrame(y , columns = ["Range"])
data = pd.concat([data, y], axis=1)

# Normalizing the Range data
range_1 = data[['Range']]
min_max_scaler = MinMaxScaler()
norm_data = min_max_scaler.fit_transform(range_1.values)

# Function for univariate data preparation
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

# Function to prepare input data for the model
def prepare_input(user_input):
    # Convert input to numpy array and reshape for the model
    input_data = np.array(user_input).reshape(-1, 1)
    normalized_input = min_max_scaler.transform(input_data)
    return normalized_input.reshape((1, 5, 1))

# Streamlit app layout
st.title('Cryptocurrency Price Prediction')

st.write('Enter the Range values for the past 5 days:')

# Fixed input for 5 past days (since the model is trained for 5 inputs)
user_input = []
for i in range(5):
    value = st.number_input(f'Day {i + 1} Range Value:', value=0.0)
    user_input.append(value)

# Button to trigger prediction
if st.button('Predict'):
    # Prepare the input and make prediction
    x_input = prepare_input(user_input)
    prediction = model.predict(x_input)
    
    # Inverse transform the prediction to the original scale
    predicted_value = min_max_scaler.inverse_transform(prediction)
    predicted_range = predicted_value[0][0]
    
    # Display the prediction value
    st.write(f'The predicted Range for the next day is: **{predicted_range:.2f}**')
    
    # Plot historical data and the predicted value
    st.write("## Data Visualization with Prediction")

    # Plot the past 5 days
    fig = go.Figure()
    
    # Add the user input data (last 5 days)
    fig.add_trace(go.Scatter(
        x=[f'Day {i+1}' for i in range(5)],
        y=user_input,
        mode='lines+markers',
        name='Past 5 Days',
        line=dict(color='royalblue'),
        marker=dict(color='blue', size=8)
    ))
    
    # Add the predicted point (next day)
    fig.add_trace(go.Scatter(
        x=[f'Day {6}'],
        y=[predicted_range],
        mode='markers',
        name='Predicted Range (Next Day)',
        marker=dict(color='red', size=12, symbol='circle')
    ))

    # Customize the layout of the plot
    fig.update_layout(
        title='Cryptocurrency Range Prediction',
        xaxis_title='Days',
        yaxis_title='Range',
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor='white'
    )
    
    # Display the plot
    st.plotly_chart(fig)

    # Display a table showing input and prediction
    st.write("### Input Data and Prediction:")
    df_display = pd.DataFrame({
        'Day': [f'Day {i+1}' for i in range(5)] + ['Prediction (Next Day)'],
        'Range': user_input + [predicted_range]
    })
    st.table(df_display)

    # Justification/Explanation of the prediction
    st.write("""
    ### Prediction Explanation:
    - The model was trained to predict the next day's range based on the last 5 days of data.
    - The blue line represents the actual range values for the past 5 days.
    - The red dot on the plot shows the predicted value for the next day.
    """)

# Optionally, display original data for context
st.write("## Original Data Visualization")

# Plotting with Plotly (original dataset)
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Range'], mode='lines', name='Original Data', line=dict(color='lightgrey')))
st.plotly_chart(fig)




import numpy as np
import talib as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import os
from pathlib import Path
import tensorflow as tf
import random

sns.set_theme()

class Data:
    """
    This class is designed for handling and processing financial market data, particularly for 
    technical analysis and machine learning applications.

    Attributes:
    -----------
    file_path : str
        The path to the CSV file containing the market data.
    data : pandas.DataFrame or None
        A DataFrame to hold the market data after loading from the file. Initially set to None.

    Methods:
    --------
    __init__(self, file_path):
        Initializes the Data class with the provided file path.

    load_data(self):
        Loads the market data from the CSV file specified in file_path into the 'data' attribute.

    add_technical_indicators(self):
        Computes and adds various technical indicators (RSI, SMA, MACD, ATR) to the 'data' attribute. 
        These indicators are commonly used in market analysis.

    extract_data_for_NN(self):
        Prepares and returns a subset of the market data with relevant columns for Neural Network (NN) processing.
        This includes both the technical indicators and the basic market data columns such as Open, High, Low, Close, and Volume.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Loads the data from the given file path."""
        self.data = pd.read_csv(self.file_path)

    def add_technical_indicators(self):
        """Adds RSI and SMA technical indicators to the data."""
        self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)
        self.data['SMA_8'] = ta.SMA(self.data['Close'], timeperiod=8)
        self.data['SMA_22'] = ta.SMA(self.data['Close'], timeperiod=22)
        self.data['SMA_50'] = ta.SMA(self.data['Close'], timeperiod=50)
        self.data['SMA_110'] = ta.SMA(self.data['Close'], timeperiod=110)
        #self.data['SMA_200'] = ta.SMA(self.data['Close'], timeperiod=200)
        self.data['MACD'], self.data['MACDSignal'], macdhist = ta.MACD(self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['ATR'] = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)

    def extract_data_for_NN(self):
        """Extracts the necessary columns for Technical Analysis (TA)."""
        if self.data is None:
            self.load_data()

        # Extracting relevant columns
        ta_data = self.data[['Open', 'High', 'Low', 'Close', 'Volume', 
                             'RSI', 'SMA_8', 'SMA_22', 'SMA_50', 'SMA_110', 
                             'MACD', 'MACDSignal', 'ATR', 
                            'Label', 'Target']]
        
        ta_data = ta_data.dropna() # Dropping NaN rows

        return ta_data

class LSTMModel:
    """
    This class encapsulates the functionality for creating, training, and utilizing a Long Short-Term Memory (LSTM) neural network model for time series forecasting.

    Attributes:
    -----------
    data : pandas.DataFrame
        The dataset used for training the model.
    look_back : int
        The number of previous time steps to use as input variables to predict the next time period.
    model_name : str
        The name of the file to which the trained model will be saved.
    model : keras.Sequential or None
        The LSTM model. Initially set to None.
    X : numpy.ndarray or None
        Feature data for training the model. Initially set to None.
    y : numpy.ndarray or None
        Target data for training the model. Initially set to None.
    scaler : MinMaxScaler
        Feature scaler for normalizing the dataset.

    Methods:
    --------
    __init__(self, data, look_back=5, model_name='model_name.keras'):
        Initializes the LSTMModel class with data, look_back period, and model name.

    preprocess_data(self):
        Scales the features and prepares the dataset for LSTM model training.

    build_model(self):
        Constructs the LSTM neural network architecture, including LSTM layers, Dropout layers, and Dense layers.

    train_model(self):
        Executes the data preprocessing, builds the model, and trains the LSTM model using the provided dataset.
        The training process includes early stopping and cross-validation. The trained model is saved to the specified file.
    """

    def __init__(self, data, look_back=5, model_name='model_name.keras'):
        self.data = data
        self.look_back = look_back
        self.model_name = model_name
        self.model = None
        self.X = None
        self.y = None

    def preprocess_data(self):
        # Scale the features
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = self.scaler.fit_transform(self.data.drop(columns=['Label', 'Target']))
        joblib.dump(self.scaler, 'scaler.save')
        # Prepare the sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.look_back):
            X.append(scaled_data[i:(i + self.look_back)])
            y.append(self.data.iloc[i + self.look_back]['Target'])

        self.X = np.array(X)
        self.y = np.array(y)
        print(f'Shape of X: {self.X.shape}')

    def build_model(self):

        self.model = Sequential()
        # Adding more LSTM layers with 'return_sequences=True' for all but the last LSTM layer
        self.model.add(LSTM(175, activation='relu', return_sequences=True, input_shape=(self.look_back, self.data.shape[1] - 2)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, activation='relu', return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(75, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(50, activation='LeakyReLU'))
        self.model.add(Dense(1))  # Output layer+
        self.model.compile(optimizer='adam', loss='mae')

    def train_model(self):

        self.preprocess_data()
        self.build_model()

        # EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

        self.model.save(self.model_name)
        print(f"Model saved as {self.model_name}")

if __name__ == "__main__":

    """
    This script demonstrates the process of forecasting financial time series data using LSTM (Long Short-Term Memory) neural networks, specifically focusing on Bitcoin (BTC) price prediction.
    ------
    Usage:
    ------
    This block is meant to be executed as a standalone script and is tailored for users with knowledge in financial analysis and machine learning. It requires a specific CSV file format and utilizes libraries like TensorFlow, NumPy, and scikit-learn.

    Output:
    -------
    The script outputs the trained LSTM model and its performance metrics, offering insights into its effectiveness for Bitcoin price forecasting.
    """

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Create the Data Frame for training data
    data_frame = Data('./Data/BTC-USD_2014.csv')
    data_frame.load_data()
    data_frame.add_technical_indicators()
    ta_data = data_frame.extract_data_for_NN()

    model_name = 'lstm_model_proj.keras'
    lstm_model = LSTMModel(ta_data, look_back=2, model_name=model_name)
    lstm_model.train_model()
    tscv = TimeSeriesSplit(n_splits=5)

    # Metrics and Prediction Logic Remains Same
    final_predictions = []
    final_actuals = []
    mse_values = []
    mae_values = []
    r2_values = []
    mape_values = []
    max_error_values = []

    daily_mse_values_test = []
    daily_MAE_values_test = []
    days = []
    window_size = 7  
    rolling_r2_values = []

    accuracy_threshold = 0.05  # 5% threshold
    accurate_count_values = []
    inaccurate_count_values = []    

    if lstm_model.model is not None:  # Ensure model is available
        for train_index, test_index in tscv.split(lstm_model.X):
            prev_y_test, prev_predictions = None, None
        
            X_train, X_test = lstm_model.X[train_index], lstm_model.X[test_index]
            y_train, y_test = lstm_model.y[train_index], lstm_model.y[test_index]

            predictions = lstm_model.model.predict(X_test)
            final_predictions.extend(predictions.flatten())
            final_actuals.extend(y_test)

            # Calculate and store metrics
            mse_values.append(mean_squared_error(y_test, predictions))
            mae_values.append(mean_absolute_error(y_test, predictions))
            r2_values.append(r2_score(y_test, predictions))
            max_error_values.append(max_error(y_test, predictions))
            mape_values.append(mean_absolute_percentage_error(y_test, predictions))
            accurate_predictions = np.abs(y_test - predictions.flatten()) / y_test <= accuracy_threshold
            accurate_count = np.sum(accurate_predictions)
            inaccurate_count = len(y_test) - accurate_count
            accurate_count_values.append(accurate_count)
            inaccurate_count_values.append(inaccurate_count)

            y_test_r2, predictions_r2 = y_test, predictions

            if prev_y_test:
                y_test_r2 = prev_y_test + y_test
                predictions_r2 = prev_predictions + predictions

            if len(days) == 0:
                days.extend(train_index)
                rolling_r2_values.extend([np.nan for i in range(len(train_index))])
                daily_mse_values_test.extend([np.nan for i in range(len(train_index))])
                daily_MAE_values_test.extend([np.nan for i in range(len(train_index))])
            days.extend(test_index)

            for i in range(len(y_test)):
                day_mse = mean_squared_error([y_test[i]], [predictions[i]])
                daily_mse_values_test.append(day_mse)

                day_MAE = mean_absolute_error([y_test[i]], [predictions[i]])
                daily_MAE_values_test.append(day_MAE)

            for i in range(window_size, len(y_test_r2) + window_size):
                window_actual = y_test_r2[i - window_size:i]
                window_predicted = predictions_r2[i - window_size:i]
                rolling_r2_values.append(r2_score(window_actual, window_predicted))

            prev_y_test = y_test_r2[-window_size:]
            prev_predictions = predictions_r2[-window_size:]
  
    plt.plot(days, daily_mse_values_test)
    plt.xlabel("Days since november 15, 2014")
    plt.ylabel("MSE")
    cwd = os.getcwd()
    path = Path(cwd) / "Code" / "FigurePlots" / "LSTM"/ "MSE"
    plt.savefig(path / "daily_MSE_LSTM.png")
    plt.show()

    path = Path(cwd) / "Code" / "FigurePlots" / "LSTM"/ "MAE"
    plt.plot(days, daily_MAE_values_test)
    plt.xlabel("Days since november 15, 2014")
    plt.ylabel("MAE")
    plt.savefig(path / "daily_MAE_LSTM.png")
    plt.show()

    path = Path(cwd) / "Code" / "FigurePlots" / "LSTM"/ "R2"
    plt.plot(days, rolling_r2_values)
    plt.xlabel("Days since november 15, 2014")
    plt.ylabel("R2 score")
    plt.savefig(path / "daily_R2_LSTM.png")
    plt.show()

    # Compute average of the metrics
    average_mse = sum(mse_values) / len(mse_values)
    average_mae = sum(mae_values) / len(mae_values)
    average_r2 = round(sum(r2_values) / len(r2_values), 5)
    average_mape = sum(mape_values) / len(mape_values)
    average_max_error = sum(max_error_values) / len(max_error_values)
    total_accurate_predictions = sum(accurate_count_values)
    total_inaccurate_predictions = sum(inaccurate_count_values)
    average_accuracy = total_accurate_predictions / (total_accurate_predictions + total_inaccurate_predictions)


    print(f'Average MSE: {average_mse}')
    print(f'Average MAE: {average_mae}')
    print(f'Average R2: {average_r2}')
    print(f'Average MAPE: {average_mape}')
    print(f'Average MAPE: {average_max_error}')
    print(f'Average accuarcy:{average_accuracy}')

    # Setting days for plot
    days = 100 #* 30
    last_100_predictions = final_predictions[-days:]
    last_100_actuals = final_actuals[-days:]

    print(f"Average days: {days}")

    # Find the points of intersection
    cross_points_x = []
    cross_points_y = []
    for i in range(1, len(last_100_actuals)):
        if (last_100_actuals[i-1] < last_100_predictions[i-1] and last_100_actuals[i] > last_100_predictions[i]) or \
        (last_100_actuals[i-1] > last_100_predictions[i-1] and last_100_actuals[i] < last_100_predictions[i]):
            cross_points_x.append(i)
            cross_points_y.append(last_100_actuals[i])

    # Plotting
    plt.figure(figsize=(9, 5))
    plt.plot(last_100_actuals, label='Actual Prices', color='tab:blue', linestyle='-', linewidth=2)
    plt.plot(last_100_predictions, label='Predicted Prices', color='tab:red', linestyle='-.', linewidth=2)
    plt.scatter(cross_points_x, cross_points_y, color='tab:green', marker='x', s=100, label='Cross Points')
    # Configure plot details
    plt.title(f'Actual vs Predicted - Last {days} Days - R2: {average_r2}', fontsize=16)
    plt.xlabel(f'Time (Last {days} Days)', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    metrics_text = (
    f"Metrics:\n"
    f"R2: {average_r2:.2f}\n"
    f"MAE: {average_mae:.2f}\n"
    f"Max Error: {average_max_error:.2f}\n"
    f"MAPE: {average_mape:.2%}\n"  # Formatting MAPE as a percentage
    f'Average accuarcy:{average_accuracy:.2%}'
    )
    plt.text(0.02, 0.02, metrics_text, fontsize=10, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.1, bottom=0.2)
    # Save the plot
    plt.savefig(f'last_100_lstm.png', bbox_inches='tight')
    plt.show()
  

    # Setting days for plot
    days = 100 #* 30
    last_100_predictions = final_predictions[-days:]
    last_100_actuals = final_actuals[-days:]

    print(f"Average days: {days}")

    # Find the points of intersection
    cross_points_x = []
    cross_points_y = []
    for i in range(1, len(last_100_actuals)):
        if (last_100_actuals[i-1] < last_100_predictions[i-1] and last_100_actuals[i] > last_100_predictions[i]) or \
        (last_100_actuals[i-1] > last_100_predictions[i-1] and last_100_actuals[i] < last_100_predictions[i]):
            cross_points_x.append(i)
            cross_points_y.append(last_100_actuals[i])

    # Plotting

    plt.figure(figsize=(9, 5))
    plt.plot(last_100_actuals, label='Actual Prices', color='tab:blue', linestyle='-', linewidth=2)
    plt.plot(last_100_predictions, label='Predicted Prices', color='tab:red', linestyle='-.', linewidth=2)
    plt.scatter(cross_points_x, cross_points_y, color='tab:green', marker='x', s=100, label='Cross Points')
    # Configure plot details
    plt.title(f'Actual vs Predicted - Last {days} Days - R2: {average_r2}', fontsize=16)
    plt.xlabel(f'Time (Last {days} Days)', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    metrics_text = (
    f"Metrics:\n"
    f"R2: {average_r2:.2f}\n"
    f"MAE: {average_mae:.2f}\n"
    f"Max Error: {average_max_error:.2f}\n"
    f"MAPE: {average_mape:.2%}\n"  # Formatting MAPE as a percentage
    f'Average accuarcy:{average_accuracy:.2%}'
    )
    plt.text(0.02, 0.02, metrics_text, fontsize=10, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.1, bottom=0.2)
    # Save the plot
    plt.savefig(f'last_365_lstm.png', bbox_inches='tight')
    plt.show()
  

    #Plot the difference bewteen actual and predicted prices
    last_100_actuals = np.array(last_100_actuals)
    last_100_predictions = np.array(last_100_predictions)
    print(last_100_actuals)
    print(last_100_predictions)
    plt.plot(last_100_actuals - last_100_predictions)
    plt.xlabel(f'Time (Last {days} Days)', fontsize=14)
    plt.ylabel("Difference between actual and predicted price", fontsize=14)
    plt.show()
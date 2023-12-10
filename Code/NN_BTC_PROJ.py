import numpy as np
import talib as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

sns.set_theme()

class Data:
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

# Create the Data Frame for training data
data_frame = Data("/Users/miafrivik/Documents/GitHub/FYS_STK_Project_3/Data/BTC-USD_2014.csv")
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


if lstm_model.model is not None:  # Ensure model is available
    for train_index, test_index in tscv.split(lstm_model.X):
        X_train, X_test = lstm_model.X[train_index], lstm_model.X[test_index]
        y_train, y_test = lstm_model.y[train_index], lstm_model.y[test_index]

        predictions = lstm_model.model.predict(X_test)
        final_predictions.extend(predictions.flatten())
        final_actuals.extend(y_test)

        # Calculate and store metrics
        mse_values.append(mean_squared_error(y_test, predictions))
        mae_values.append(mean_absolute_error(y_test, predictions))
        r2_values.append(r2_score(y_test, predictions))
        mape_values.append(mean_absolute_percentage_error(y_test, predictions))


# Compute average of the metrics
average_mse = sum(mse_values) / len(mse_values)
average_mae = sum(mae_values) / len(mae_values)
average_r2 = round(sum(r2_values) / len(r2_values), 5)
average_mape = sum(mape_values) / len(mape_values)

print(f'Average MSE: {average_mse}')
print(f'Average MAE: {average_mae}')
print(f'Average R2: {average_r2}')
print(f'Average MAPE: {average_mape}')

# Setting days for plot
days = 365 #* 30
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
plt.subplots_adjust(left=0.1, bottom=0.2)
# Save the plot
plt.savefig(f'{model_name}.png', bbox_inches='tight')
plt.show()
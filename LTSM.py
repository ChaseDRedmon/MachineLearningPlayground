# univariate lstm example
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split


def prepare_data(timeseries_data, n_features):
    X, y = [], []
    for i in range(len(timeseries_data)):
        # find the end of this pattern
        end_ix = i + n_features
        # check if we are beyond the sequence
        if end_ix > len(timeseries_data) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Import sample dataset containing date, year, day of the week, day of the month, holidays and sales (i.e. #loans sold)
df = pd.read_csv('.\\data\\Sales.csv', delimiter=',', index_col='Date')

dates = df.index
# Split targets and features
Y = df.iloc[:, 4]
X = df.iloc[:, 0:4]

# Print first 5 rows from each dataframe
print(X.head(5))
print(Y.head(5))

# Count features for modelization
X_num_columns = len(X.columns)

# choose a number of time steps
n_steps = 10
# split into samples
X, y = prepare_data(X.values, n_steps)

print(X.shape)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 4
X = X.reshape(990, 10, 4)

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=1)

print("Training completed")

# Save trained model
#model.save("PBISalesDemo.h5")

# Predict known daily sales in order to check results
#predictions = model.predict(X)
#predictions_list = map(lambda x: x[0], predictions)
#predictions_series = pd.Series(predictions_list, index=dates)
#dates_series = pd.Series(dates)
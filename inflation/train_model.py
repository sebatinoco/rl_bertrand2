import pandas as pd
import numpy as np
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

from utils.lstm import LSTM
from utils.train import train
from utils.countries import countries
from utils.dates import low_date, high_date

# set seeds
random_state = 3380
torch.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
random.seed(random_state)

# read data
df = pd.read_excel('Inflation-data.xlsx', sheet_name='hcpi_m')
df = df.drop(columns = ['IMF Country Code', 'Indicator Type', 'Series Name', 'Country Code'])

value_vars = [col for col in df.columns if type(col) == int]

# rearrange data
df = pd.melt(df, id_vars = 'Country', value_vars=value_vars, value_name = 'inflation').rename(columns = {'variable': 'date'})
df['date'] = df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[-2:]), 1))

df = df.sort_values(['Country', 'date'])

# filter by country
df = df[df['Country'].isin(countries)]

# filter by date
low_filter = df['date'] >= low_date
high_filter = df['date'] <= high_date
df = df[low_filter & high_filter]

# Transform to m/m variation
df[f't-1'] = df.groupby('Country')['inflation'].shift(1)
df['inflation'] = (df['inflation'] - df['t-1']) / df['t-1']

# Generate lags
K = 3
for k in range(1, K + 1):
    df[f't-{k}'] = df.groupby('Country')['inflation'].shift(k)

# remove invalid values
df = df.replace(np.inf, np.nan)
df = df.replace(-np.inf, np.nan)
df = df.dropna()

# shuffle dataset
#df = df.sample(frac=1)

# generate X, y
t_cols = [col for col in df.columns if col[0] == 't']
X = df.apply(lambda x: [x[col] for col in t_cols], axis = 1).copy()
y = df.inflation.copy()

# transform to tensor
X = torch.tensor(X.to_list()).unsqueeze(dim = 2)
y = torch.tensor(y.to_list()).unsqueeze(dim = 1)

# split data
train_size = int(0.7 * len(X))
val_size = int(0.2 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_valid, y_valid = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# crear Tensor datasets
train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test, y_test)

# dataloaders
batch_size = 128

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# instantiate model
model = LSTM(input_size = 1, output_size = 1, hidden_size = 100, num_layers = 2, random_state = random_state, dropout = 0.2)

params = {
    'model': model,
    'trainloader': train_loader,
    'validloader': valid_loader,
    'criterion': torch.nn.MSELoss(),
    'optimizer': torch.optim.Adam(model.parameters(), lr = 1e-3),
    'epochs': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'random_state': random_state,
}

# train
train(**params)
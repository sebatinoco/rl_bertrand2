import pandas as pd
import numpy as np
import datetime
import random

def get_inflation_serie(start_year: int = 2000, end_year: int = 2020, country: str = None, random_state = 3380):
    
    random.seed(random_state)
    df_inflation = pd.read_excel('inflation/inflation-data.xlsx', sheet_name='hcpi_m')

    value_vars = [col for col in df_inflation.columns if type(col) == int]

    # rearrange data
    df_inflation = pd.melt(df_inflation, id_vars = 'Country', value_vars=value_vars, value_name = 'inflation').rename(columns = {'variable': 'date'})
    df_inflation['date'] = df_inflation['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[-2:]), 1))

    df_inflation = df_inflation.sort_values(['Country', 'date'])

    # Transform to m/m variation
    df_inflation[f't-1'] = df_inflation.groupby('Country')['inflation'].shift(1)
    df_inflation['inflation'] = (df_inflation['inflation'] - df_inflation['t-1']) / df_inflation['t-1']

    # remove invalid values
    df_inflation = df_inflation.replace(np.inf, np.nan)
    df_inflation = df_inflation.replace(-np.inf, np.nan)
    df_inflation = df_inflation.dropna(subset = 'inflation')

    countries = ['Italy', 'United States', 'Sweden', 'Canada', 'France', 'China', 'Singapore', 'Germany', 'Switzerland', 'Netherlands']

    if country is None:
        country = random.choice(countries)
        
    mask = (df_inflation['Country'] == country) & (df_inflation['date'] > datetime.date(start_year,1,1)) & (df_inflation['date'] < datetime.date(end_year, 1, 1))
    inflation_serie = df_inflation[mask]['inflation'].copy()

    return inflation_serie
import pandas as pd
import numpy as np

def load_and_preprocess(filepath):
    data_frame = pd.read_csv(filepath, encoding='cp949')

    data_frame.rename(columns={
        '날짜': 'date', 
        '시간': 'time', 
        '측정소명': 'station_name',
        '수온': 'temperature', 
        '용존산소': 'DO', 
        '총질소': 'TN',
        '총인': 'TP', 
        '총유기탄소': 'TOC', 
        '페놀': 'phenol',
        '시안': 'cyanide', 
        'pH': 'pH'
    }, inplace=True)

    data_frame['timestamp'] = pd.to_datetime(
        data_frame['date'].astype(str) + ' ' + data_frame['time'].astype(str),
        errors='coerce')
    data_frame['hour'] = data_frame['timestamp'].dt.hour
    data_frame['weekday'] = data_frame['timestamp'].dt.weekday
    data_frame['month'] = data_frame['timestamp'].dt.month

    data_frame = pd.get_dummies(data_frame, columns=['station_name'], drop_first=True)

    numeric_cols = ['temperature', 'DO', 'TN', 'TP', 'TOC', 'phenol', 'cyanide']
    for col in numeric_cols:
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')

    data_frame.dropna(inplace=True)

    features = numeric_cols + ['hour', 'weekday', 'month'] + \
               [col for col in data_frame.columns if col.startswith('station_name_')]

    train_set = data_frame[features]
    pH = data_frame['pH']

    return train_set, pH

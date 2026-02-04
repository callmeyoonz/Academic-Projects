import pandas as pd

# Define file paths
file_paths = [
    r"C:\Users\1000r\Downloads\jcsxmx3m3e0ra7cn.csv",
    r"C:\Users\1000r\Downloads\kh8sdgd9xtvxw6sx.csv",
    r"C:\Users\1000r\Downloads\nrtzjrt9ixhsbffu.csv"
]

def data_read(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df[df['SYM_ROOT'] == 'CSCO']
    return df[['DATE', 'TIME_M', 'PRICE', 'EX', 'TR_CORR', 'TR_SCOND']]

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates()

def remove_out_of_range_values(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['PRICE'] > 0]

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['PRICE', 'DATE', 'TIME_M']
    data = data.dropna(subset=required_columns)
    return data

def filter_hours(data: pd.DataFrame, start_time: str = '09:30:00', end_time: str = '16:00:00') -> pd.DataFrame:
    data['DATETIME'] = pd.to_datetime(data['DATE'] + ' ' + data['TIME_M'])
    data = data.set_index('DATETIME')
    data = data.between_time(start_time, end_time).reset_index(drop=False)
    return data

def filter_corrected_trades(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['TR_CORR'] == 0]

def filter_sale_conditions(data: pd.DataFrame) -> pd.DataFrame:
    valid_conditions = ['E', 'F']
    invalid_conditions = ['T', 'Z']
    data = data[data['TR_SCOND'].str.contains('|'.join(valid_conditions), na=False)]
    data = data[~data['TR_SCOND'].str.contains('|'.join(invalid_conditions), na=False)]
    return data

def handle_duplicate_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby('DATETIME', as_index=False).agg({
        'PRICE': 'median',
        'EX': 'first',
        'TR_CORR': 'first',
        'TR_SCOND': 'first',
        'DATE': 'first'
    })

def filter_most_frequent_exchange(data: pd.DataFrame, frequency: str = 'M'):
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['period'] = data['DATE'].dt.to_period(frequency)
    exchange_counts = data.groupby(['period', 'EX']).size().reset_index(name='count')
    most_frequent_exchanges = exchange_counts.loc[exchange_counts.groupby('period')['count'].idxmax()]
    data = pd.merge(data, most_frequent_exchanges, on='period', suffixes=('', '_y'))
    data = data[data['EX'] == data['EX_y']].drop(columns=['EX_y', 'count', 'period'])
    exchange_list = most_frequent_exchanges[['period', 'EX']].values.tolist()
    return data, exchange_list

def data_cleaning(data: pd.DataFrame):
    data = filter_hours(data)
    data = handle_missing_values(data)
    data = remove_out_of_range_values(data)
    data = filter_corrected_trades(data)
    data = filter_sale_conditions(data)
    data = handle_duplicate_timestamps(data)
    data, exchange_list = filter_most_frequent_exchange(data)
    return data, exchange_list

def data_concat(file_paths: list) -> pd.DataFrame:
    dfs = [data_read(file_path) for file_path in file_paths]
    all_data = pd.concat(dfs, ignore_index=True)
    if all_data.empty or 'DATE' not in all_data.columns:
        raise ValueError("'DATE' column is missing or all data is empty after concatenation.")
    return all_data

def resample(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index('DATETIME', inplace=True)
    resampled_data = data['PRICE'].resample('1S').ffill().reset_index()
    resampled_data = resampled_data.sort_values(by='DATETIME').reset_index(drop=True)
    return resampled_data

# Process all data
all_data = data_concat(file_paths)
cleaned_data, exchange_list = data_cleaning(all_data)
resampled_data = resample(cleaned_data)

# Save the final dataset
output_path = r"C:\Users\1000r\Downloads\final_resampled_data2.csv"
resampled_data.to_csv(output_path, index=False)





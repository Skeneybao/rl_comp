import glob
import os

import pandas as pd
from pandas_parallel_apply import SeriesParallel
from tqdm.auto import tqdm


def parse_event_time(event_time):
    event_time_str = f'{int(event_time):09d}'
    hours = int(event_time_str[:2])
    minutes = int(event_time_str[2:4])
    seconds = int(event_time_str[4:6])
    microseconds = int(event_time_str[6:]) * 100
    return pd.Timestamp(year=1900, month=1, day=1, hour=hours, minute=minutes, second=seconds,
                        microsecond=microseconds)


def nearest_5second(time):
    # Add 2 before the division to round to the nearest multiple of 5
    second = ((time.second + 2) // 5) * 5
    # Adjust for the case where seconds become 60
    if second == 60:
        if time.minute == 59:
            # If minute is 59 and second rounds to 60, increase the hour
            return time.replace(hour=(time.hour + 1) % 24, minute=0, second=0, microsecond=0)
        else:
            # Else just increase the minute
            return time.replace(minute=time.minute + 1, second=0, microsecond=0)
    else:
        return time.replace(second=second, microsecond=0)


def process_data(df):
    df = df[df['eventTime'] <= 145700000]

    df['datetime'] = SeriesParallel(df['eventTime'], n_cores=-1, pbar=False).apply(parse_event_time)

    df['nearest_5sec'] = SeriesParallel(df['datetime'], n_cores=-1, pbar=False).apply(nearest_5second)

    df['time_diff'] = abs((df['datetime'] - df['nearest_5sec']).dt.total_seconds())

    df_sorted = df.sort_values('time_diff').drop_duplicates(['code', 'nearest_5sec']).sort_index()

    df_result = df_sorted.drop(columns=['datetime', 'nearest_5sec', 'time_diff'])
    return df_result


if __name__ == '__main__':
    training_set = '/mnt/data3/rl-data/train_set/'
    output_path_base = '/mnt/data3/rl-data/train_set_nearest_5sec/'
    file_list = glob.glob(f"{training_set}*/*")

    for file in tqdm(file_list):
        print(f'processing {file}')
        file_relative_path = file.replace(training_set, '')
        output_path = f'{output_path_base}{file_relative_path}'

        # if os.path.exists(output_path):
        #     print(f'{output_path} already exists, skipping')
        #     continue

        df = pd.read_parquet(file)
        df_result = process_data(df)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        df_result.to_parquet(output_path)
        print(f'output to {output_path}')

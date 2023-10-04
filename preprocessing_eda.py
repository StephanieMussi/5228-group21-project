from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder



# alignment of flat_type for consistent spelling
def align_flat_type(df):
    df=df.replace('2 room','2-room')
    df=df.replace('3 room','3-room')
    df=df.replace('4 room','4-room')
    df=df.replace('5 room','5-room')
    return df

# process date data in df, it will modify df internally
def process_date(df):
    df['rent_approval_date'] = pd.to_datetime(df['rent_approval_date']).dt.year * 100 + pd.to_datetime(df['rent_approval_date']).dt.month
    scaler = MinMaxScaler()
    df['rent_approval_date'] = scaler.fit_transform(df[['rent_approval_date']])
    return 

# binary encoding of attribute
def binary_encoding(df, col_name):
    df = BinaryEncoder(cols=[col_name]).fit_transform(df)
    return df

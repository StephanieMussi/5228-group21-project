from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
from sklearn.preprocessing import TargetEncoder


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

# binary encoding of town
def binary_encoding(df_train, df_test, col_name):
    encoder = BinaryEncoder(cols=[col_name])
    df_train = encoder.fit_transform(df_train)
    
    num_rows = df_test.shape[0]
    pad_col = pd.DataFrame(np.zeros([num_rows, 1]), columns = ['pad'])
    df_test_pad = pd.concat([df_test, pad_col], axis = 1)
    df_test_pad = encoder.transform(df_test_pad)
    df_test = df_test_pad.iloc[:,:-1]
    
    return df_train, df_test

# target encoding of column
def target_encoding(df_train, df_test, col_name):
    encoder = TargetEncoder(smooth="auto", target_type='continuous')
    df_train[col_name] = encoder.fit_transform(df_train[col_name].values.reshape(-1, 1), df_train['monthly_rent'])
    df_test[col_name] = encoder.transform(df_test[col_name].values.reshape(-1, 1))
    return df_train, df_test
    

"""
One-hot encoding of flat_type
Example usage:
Assuming you have a DataFrame df and you want to one-hot encode the "Category" column
df = one_hot_encode_column(df, "Category", order, status)
order is for test set to get the same order with train data
status is either "train" or "test", although I did not check
"""
def one_hot_encode_column(df, column_name, order, status):
    # Extract the specified column from the DataFrame
    column = df[column_name]
    
    if status == "train":
        # Get unique values in the column
        unique_values = column.unique()

            
    if status == "test":
        unique_values = order
    
    # Create a new DataFrame with binary columns
    binary_columns = pd.DataFrame(0, columns=unique_values, index=df.index)

    
    # Set binary values based on one-hot encoding
    for value in unique_values:
        binary_columns.loc[column == value, value] = 1
    
    # Concatenate the binary columns into a single column with binary strings
    df = pd.concat([df, binary_columns], axis=1)
    df.drop(columns=[column_name], inplace=True)
    
    #Rename column
    for name in unique_values:
        df = df.rename(columns={name: f"{column_name}_{name}"})
    
    
    return df, unique_values




"""
Rank encoding of flat_model
Example usage:
Assuming you have a DataFrame df and you want to rank encode the "Category" column
df = rank_encode_column(df, "Category")
"""
def rank_encode_column(df, column_name, order, status, rank = 0):
    # Extract the specified column from the DataFrame
    column = df[column_name]
    
    if status == "train":
        # Get unique values in the column
        unique_values = column.unique()

        if rank != 0:
            unique_values = df.groupby(column_name)['monthly_rent'].mean().sort_values().reset_index()[column_name].tolist()
            
    if status == "test":
        unique_values = order
        
    value_to_rank = {value: rank for rank, value in enumerate(unique_values, 1)}
    
    # Create a new column with rank values
    df[column_name] = column.map(value_to_rank)
    
    return df, unique_values



from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from category_encoders import BinaryEncoder
import math


"""
1. sg_stock_prices
"""
# check if the date range in train/test data is fully covered in stock data
def is_full_coverage(df_auxi, df_train, df_test):
    date_auxi = df_auxi['date'].unique()
    date_train = df_train['rent_approval_date'].unique()
    date_test = df_test['rent_approval_date'].unique()
    print("Date in auxi data covers all in train data? {}".format(all(item in date_auxi for item in date_train)))
    print("Date in auxi data covers all in test data? {}".format(all(item in date_auxi for item in date_test)))
    
# get average adjusted close price for each year-month
def get_avg_adjusted_close(df_stock):
    df_stock['date'] = [x[0:7] for x in df_stock['date']]
    df_stock = df_stock.groupby('date', as_index=False)['adjusted_close'].mean()
    return df_stock

# add avg_stock_price column for train/test data
def insert_col_stock_price(df_stock, df):
    list_index_date = [df_stock.index[df_stock['date'] == x][0] for x in df['rent_approval_date']]
    list_avg_stock_price = df_stock['adjusted_close'][list_index_date].to_list()
    df['avg_stock_price'] = list_avg_stock_price
    

"""
2. sg-shopping-malls
"""
# naive approach of finding the number of shopping malls in radius of the df
def num_shopping_malls_df(df, df_shopping_malls, radius=2):
    num_list = []
    for i in df.index:
        lat = df['latitude'][i]
        long = df['longitude'][i]
        num_shopping = num_shopping_malls_pt(lat, long, df_shopping_malls, radius)
        num_list.append(num_shopping)
    df['num_shopping_malls'] = num_list
    return num_list

    
    
# find the number of shopping malls in radius of a point
def num_shopping_malls_pt(lat, long, df_shopping_malls, radius=2):
    house_position = (lat, long)

    distances = []
    for i in df_shopping_malls.index:
        shopping_pos = (df_shopping_malls['latitude'][i], df_shopping_malls['longitude'][i])
        distance = geodesic(house_position, shopping_pos).km
        distances.append(distance)

    distances.sort()
    num_shopping = 0
    for d in distances:
        if d <= radius:
            num_shopping += 1
        else:
            break

    return num_shopping

""" 
Kmeans approach of finding the number of shopping malls in radius of the df. it will modify df internally
inputs:
df: original df
df_shppoing malls: shopping mall df
radius: radius use for search for num of shopping mall
k: number of nearest center to count on 

return:
num_list: a list contain number of shopping mall of df
"""
def num_shopping_malls_df_Kmeans(df, df_shopping_malls, radius=2, k=3):
    
    # K-means
    kmeans = KMeans(n_clusters=14, random_state=42)
    kmeans.fit(df_shopping_malls)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    num_list = []
    for i in df.index:
        lat = df['latitude'][i]
        long = df['longitude'][i]
        #find the most three nearby centers
        nearby_centers = find_nearby_centers(lat, long, centers, k)
        df_partial_shopping = df_shopping_malls.iloc[np.where(np.isin(labels, nearby_centers))]
#         print(len(df_partial_shopping))
        num_shopping = num_shopping_malls_pt(lat, long, df_partial_shopping, radius)
        num_list.append(num_shopping)
    df['num_shopping_malls'] = num_list
    return num_list

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


# find k nearest centers
def find_nearby_centers(lat, long, centers, k):
    distances = []
    house_position = (lat, long)
    for i, center in enumerate(centers):
        distance = euclidean_distance(house_position, (centers[i, 0], centers[i, 1]))
        distances.append((distance, i))
    distances.sort()
    return [i[1] for i in distances[:k]]


"""
3. sg-mrt-existing-stations
"""
def calculate_min_distance(row, df_mrt):
    min_distance = float('inf')
    for _, df2_row in df_mrt.iterrows():
        loc1 = (row['latitude'], row['longitude'])
        loc2 = (df2_row['latitude'], df2_row['longitude'])
        distance = geodesic(loc1, loc2).kilometers
        if distance < min_distance:
            min_distance = distance
    return min_distance

"""
4. sg_coe_prices
"""
# get average coe price for each year-month
def get_avg_coe_price(df_coe):
    df_coe = df_coe.groupby('date', as_index=False)['price'].mean()
    return df_coe

# add avg_coe_price column for train/test data
def insert_col_coe_price(df_coe, df):
    list_index_date = [df_coe.index[df_coe['date'] == x][0] for x in df['rent_approval_date']]
    list_avg_coe_price = df_coe['price'][list_index_date].to_list()
    df['avg_coe_price'] = list_avg_coe_price
    
    
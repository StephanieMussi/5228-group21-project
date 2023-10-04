from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

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


# find k nearest centers
def find_nearby_centers(lat, long, centers, k):
    distances = []
    house_position = (lat, long)
    for i, center in enumerate(centers):
        distance = geodesic(house_position, (centers[i, 0], centers[i, 1]))
        distances.append((distance, i))
    distances.sort()
    return [i[1] for i in distances[:k]]


"""
process date data in df, it will modify df internally
"""
def process_date(df):
    df['rent_approval_date'] = pd.to_datetime(df['rent_approval_date']).dt.year * 100 + pd.to_datetime(df['rent_approval_date']).dt.month
    scaler = MinMaxScaler()
    df['rent_approval_date'] = scaler.fit_transform(df[['rent_approval_date']])
    return 


    
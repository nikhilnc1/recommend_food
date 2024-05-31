import firebase_admin
import pandas as pd
from firebase_admin import credentials
from flask import Flask, request, jsonify
import requests
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

# Initialize Firebase Admin SDK
cred = credentials.Certificate("intellicater-firebase-adminsdk-r3q36-28b3cdbc74.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://intellicater-default-rtdb.firebaseio.com/'
}, name='intellicater')

# Function to load data from JSON
def load_data_from_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"Failed to fetch data from {url}")

# Load ratings data from Firebase
ratings = load_data_from_json('https://intellicater-default-rtdb.firebaseio.com/Ratings.json')

# Create DataFrame from ratings data
df = pd.DataFrame(columns=['userID', 'itemID', 'rating'])
for user, items in ratings.items():
    for item, rating in items.items():
        df.loc[len(df)] = {'userID': user, 'itemID': item, 'rating': rating}

# Load menu data from Firebase
menu_data = load_data_from_json('https://intellicater-default-rtdb.firebaseio.com/menu.json')

# Create DataFrame from menu data
menu_items = []
for item_id, item_info in menu_data.items():
    menu_items.append({'itemID': item_id, 'name': item_info['foodName'], 'description': item_info['foodDescription'], 'price': item_info['foodPrice']})

menu_df = pd.DataFrame(menu_items)

# Load the data into the surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# Train the User-Based Collaborative Filtering Model
trainset, testset = train_test_split(data, test_size=0.25)
user_cf = KNNBasic(sim_options={'user_based': True})
user_cf.fit(trainset)

# Train the Item-Based Collaborative Filtering Model
item_cf = KNNBasic(sim_options={'user_based': False})
item_cf.fit(trainset)

# User-Based Recommendation Function
def user_based_recommendation(user_id, num_recommendations=5):
    user_inner_id = user_cf.trainset.to_inner_uid(user_id)
    user_neighbors = user_cf.get_neighbors(user_inner_id, k=num_recommendations)
    recommendations = []
    for neighbor in user_neighbors:
        neighbor_items = df[df['userID'] == user_cf.trainset.to_raw_uid(neighbor)]['itemID'].tolist()
        recommendations.extend(neighbor_items)
    return list(set(recommendations))[:num_recommendations]

# Item-Based Recommendation Function
def item_based_recommendation(user_id, num_recommendations=5):
    user_data = df[df['userID'] == user_id]
    if user_data.empty:
        return []

    last_rated_item = user_data.tail(1)['itemID'].values[0]
    item_inner_id = item_cf.trainset.to_inner_iid(last_rated_item)
    item_neighbors = item_cf.get_neighbors(item_inner_id, k=num_recommendations)
    recommendations = [item_cf.trainset.to_raw_iid(inner_id) for inner_id in item_neighbors]
    return recommendations

# Hybrid Recommendation Function
def hybrid_recommendation(user_id, num_recommendations=5):
    user_based = user_based_recommendation(user_id, num_recommendations)
    item_based = item_based_recommendation(user_id, num_recommendations)
    hybrid_recommendations = pd.Series(user_based + item_based).drop_duplicates().head(num_recommendations)
    return hybrid_recommendations.tolist()

# Helper Function for Recommendations
def recommend(userid):
    recommended_items = hybrid_recommendation(userid)
    recommended_item_ids = [item_id for item_id in recommended_items]
    return recommended_item_ids

app = Flask('intellicater')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    user_id = request.form.get('userID')
    recommended_items = recommend(user_id)
    response = {
        'user_id': user_id,
        'recommended_items': recommended_items
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

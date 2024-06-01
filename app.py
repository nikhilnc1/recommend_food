import firebase_admin
import pandas as pd
from firebase_admin import credentials
from flask import Flask, request, jsonify
import requests
import pickle
from surprise import KNNBasic

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


# Load the trained models from pickle files
with open('user_cf_model.pkl', 'rb') as f:
    user_cf = pickle.load(f)

with open('item_cf_model.pkl', 'rb') as f:
    item_cf = pickle.load(f)

# Recommendation Function
def hybrid_recommendation(user_id, num_recommendations=5):
    if user_id in df['userID'].unique():
        # For existing users
        user_inner_id = user_cf.trainset.to_inner_uid(user_id)
        user_neighbors = user_cf.get_neighbors(user_inner_id, k=num_recommendations)
        user_based_recommendations = [user_cf.trainset.to_raw_iid(neighbor) for neighbor in user_neighbors]

        item_inner_ids = []
        for item in user_based_recommendations:
            try:
                item_inner_id = item_cf.trainset.to_inner_iid(item)
                item_inner_ids.append(item_inner_id)
            except ValueError:
                continue

        item_neighbors = [item_cf.get_neighbors(item_inner_id, k=num_recommendations) for item_inner_id in item_inner_ids]
        item_based_recommendations = [item_cf.trainset.to_raw_iid(neighbor) for neighbors in item_neighbors for neighbor in neighbors]

        hybrid_recommendations = list(set(user_based_recommendations + item_based_recommendations))[:num_recommendations]
    else:
        # For new users (popularity-based recommendations)
        popular_items = df['itemID'].value_counts().index.tolist()
        hybrid_recommendations = popular_items[:num_recommendations]

    return hybrid_recommendations

# Flask Web Application
app = Flask('food_recommendation')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    user_id = request.form.get('userID')
    recommended_items = hybrid_recommendation(user_id)
    response = {
        'user_id': user_id,
        'recommended_items': recommended_items
    }
    return jsonify(response)

print(hybrid_recommendation("oa4g0UbCCuO7CJSxXpFs6PBnTft2"))

if __name__ == '__main__':
    app.run(debug=True)

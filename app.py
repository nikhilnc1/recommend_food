import firebase_admin
import pandas as pd
from firebase_admin import credentials
from flask import Flask, request, jsonify
import requests

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


def collaborative_filtering_recommendation(data, user_id, num_recommendations=4):
    user_data = data[data['userID'] == user_id]
    user_items = list(user_data['itemID'])

    # Filter out items already rated by the user
    available_items = data[~data['itemID'].isin(user_items)]

    # Group by itemID and compute the mean rating
    item_ratings = available_items.groupby('itemID')['rating'].mean().reset_index()

    # Sort the items based on the mean rating
    top_items = item_ratings.sort_values(by='rating', ascending=False).head(num_recommendations)

    return top_items['itemID']

def hybrid_recommendation(data, user_id, num_recommendations=10):
    # Get content-based recommendations
    #content_based = content_based_recommendation(data, user_id, num_recommendations)

    # Get collaborative filtering recommendations
    collaborative_filtering = collaborative_filtering_recommendation(data, user_id, num_recommendations)

    # Combine the recommendations
    #hybrid_recommendations = pd.concat([content_based, collaborative_filtering]).drop_duplicates().head(num_recommendations)

    return collaborative_filtering

def recommend(userid):
    recommended_items = hybrid_recommendation(df, userid)
    recommended_item_ids = []
    for item_id in recommended_items:
        recommended_item_ids.append(item_id)
    return recommended_item_ids

print(recommend('zG7g6bhvTWT3UvnCEMMHYSJnOdB2'))

# with open('food_recommend.pkl', 'rb') as f:
#     model = pickle.load(f)

app = Flask('intellicater')

@app.route('/recommendation', methods=['POST'])
def recommendation():

    user_id: str = request.form.get('userID')


    # Get recommendations using the loaded model
    recommended_items = recommend(user_id)  # Assuming your model returns a list
    # response = {
    #     'user_id': user_id,
    #     'recommended_items': 77  # Convert to list for JSON serialization
    # }

    # Directly return the list of recommended items as JSON
    return jsonify(menu=recommended_items)


if __name__ == '__main__':
    app.run(debug=True)

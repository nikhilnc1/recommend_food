import firebase_admin
import pandas as pd
from firebase_admin import credentials
from flask import Flask, request, jsonify
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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


menu_data = load_data_from_json('https://intellicater-default-rtdb.firebaseio.com/menu.json')

# Create DataFrame from menu data
menu_items = []
for item_id, item_info in menu_data.items():
    menu_items.append({'itemID': item_id, 'name': item_info['foodName'], 'description': item_info['foodDescription'], 'price': item_info['foodPrice']})


menu_df = pd.DataFrame(menu_items)

# TF-IDF Vectorization for content-based recommendation
tfidf = TfidfVectorizer(stop_words='english')
menu_df['description'] = menu_df['description'].fillna('')
tfidf_matrix = tfidf.fit_transform(menu_df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
def collaborative_filtering_recommendation(data, user_id, num_recommendations=5):
    user_data = data[data['userID'] == user_id]
    user_items = list(user_data['itemID'])

    # Filter out items already rated by the user
    available_items = data[~data['itemID'].isin(user_items)]

    # Group by itemID and compute the mean rating
    item_ratings = available_items.groupby('itemID')['rating'].mean().reset_index()

    # Sort the items based on the mean rating
    top_items = item_ratings.sort_values(by='rating', ascending=False).head(num_recommendations)

    return top_items['itemID']

def content_based_recommendation(user_id, num_recommendations=5):
    last_rated_item = df[df['userID'] == user_id].tail(1)['itemID'].values[0]
    idx = menu_df[menu_df['itemID'] == last_rated_item].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    item_indices = [i[0] for i in sim_scores]
    return menu_df['itemID'].iloc[item_indices].tolist()

def hybrid_recommendation(data, user_id, num_recommendations=5):
    content_based = content_based_recommendation(user_id, num_recommendations)
    collaborative_filtering = collaborative_filtering_recommendation(data, user_id, num_recommendations)
    hybrid_recommendations = pd.concat([pd.Series(content_based), pd.Series(collaborative_filtering)]).drop_duplicates().head(num_recommendations)
    return hybrid_recommendations.tolist()



def recommend(userid):
    recommended_items = hybrid_recommendation(df, userid)
    recommended_item_ids = []
    for item_id in recommended_items:
        recommended_item_ids.append(item_id)
    return recommended_item_ids

print(recommend('9XrD3k3EYmgQpCFomdhLPeJ79EJ3'))

# with open('food_recommend.pkl', 'rb') as f:
#     model = pickle.load(f)

app = Flask('intellicater')

@app.route('/recommendation', methods=['POST'])
def recommendation():

    user_id = request.form.get('userID')


    # Get recommendations using the loaded model
    recommended_items = recommend(user_id)  # Assuming your model returns a list
    response = {
        'user_id': user_id,
        'recommended_items': recommended_items  # Convert to list for JSON serialization
    }

    # Directly return the list of recommended items as JSON
    return jsonify(response)


if _name_ == '_main_':
    app.run(debug=True)

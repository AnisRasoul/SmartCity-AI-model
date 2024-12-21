import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)

class HotelRecommender:
    def __init__(self):
        self.hotel_data = None
        self.scaler = MinMaxScaler()
        self.mlb = MultiLabelBinarizer()
        self.feature_matrix = None
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
    def _clean_price(self, price):
        try:
            if isinstance(price, str):
                price = ''.join(c for c in price if c.isdigit() or c == '.')
                if len(price) > 10:
                    price = price[:10]
            return float(price)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error cleaning price {price}: {str(e)}")
            return 0.0

    def load_data(self, json_file_path):
        try:
            self.logger.info(f"Attempting to load data from {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
                self.logger.debug(f"File content first 100 chars: {file_content[:100]}")
                data = json.loads(file_content)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                data = data[0]
            self.logger.info(f"Successfully loaded JSON data with {len(data)} records")
            processed_hotels = []
            for hotel in data:
                if 'hotel_id' not in hotel:
                    self.logger.warning(f"Skipping hotel without hotel_id: {hotel}")
                    continue
                processed_hotel = {
                    'hotel_id': hotel['hotel_id'],
                    'hotel_name': hotel['hotel_name'],
                    'rating': float(hotel.get('rating', 0)),
                    'price': self._clean_price(hotel.get('price', '0')),
                    'amenities': hotel.get('amenities', [])
                }
                processed_hotels.append(processed_hotel)
            self.hotel_data = pd.DataFrame(processed_hotels)
            self.logger.info(f"DataFrame columns: {self.hotel_data.columns.tolist()}")
            self.logger.info(f"Successfully processed {len(self.hotel_data)} hotels")
            required_columns = ['hotel_id', 'hotel_name', 'rating', 'price', 'amenities']
            missing_columns = [col for col in required_columns if col not in self.hotel_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            self._create_feature_matrix()
        except FileNotFoundError:
            self.logger.error(f"File not found: {json_file_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error at line {e.lineno}, column {e.colno}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise

    def _create_feature_matrix(self):
        try:
            amenities_features = self.mlb.fit_transform(self.hotel_data['amenities'])
            numerical_features = self.scaler.fit_transform(self.hotel_data[['rating', 'price']].values)
            self.feature_matrix = np.hstack([numerical_features, amenities_features])
            self.logger.info("Feature matrix created successfully")
            self.logger.debug(f"Feature matrix shape: {self.feature_matrix.shape}")
        except Exception as e:
            self.logger.error(f"Error creating feature matrix: {str(e)}", exc_info=True)
            raise

    def get_recommendations(self, hotel_index=None, selected_amenities=None, budget=None, n_recommendations=5):
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Please load data first.")
        try:
            similarity_matrix = cosine_similarity(self.feature_matrix)
            if hotel_index is None:
                weights = self.hotel_data['rating'].values
                weights = weights / weights.sum()
                hotel_index = np.random.choice(len(self.hotel_data), p=weights)
            similar_indices = similarity_matrix[hotel_index].argsort()[::-1]
            recommended_hotels = []
            for idx in similar_indices:
                if idx == hotel_index:
                    continue
                hotel_info = self.hotel_data.iloc[idx]
                if budget is not None and hotel_info['price'] > budget:
                    continue
                if selected_amenities and not all(amenity in hotel_info['amenities '] for amenity in selected_amenities):
                    continue
                recommended_hotels.append({
                    'hotel_name': hotel_info['hotel_name'],
                    'amenities': hotel_info['amenities'],
                    'rating': hotel_info['rating'],
                    'price': hotel_info['price'],
                    'similarity_score': similarity_matrix[hotel_index][idx]
                })
                if len(recommended_hotels) >= n_recommendations:
                    break
            self.logger.info(f"Generated {len(recommended_hotels)} recommendations")
            return recommended_hotels
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            raise

@app.route('/load_data', methods=['POST'])
def load_data():
    json_file_path = request.json.get('file_path')
    recommender.load_data(json_file_path)
    return jsonify({"message": "Data loaded successfully"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    hotel_index = request.json.get('hotel_index', None)
    selected_amenities = request.json.get('selected_amenities', None)
    budget = request.json.get('budget', None)
    n_recommendations = request.json.get('n_recommendations', 5)
    try:
        recommendations = recommender.get_recommendations(
            hotel_index=hotel_index,
            selected_amenities=selected_amenities,
            budget=budget,
            n_recommendations=n_recommendations
        )
        return jsonify(recommendations), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    recommender = HotelRecommender()
    app.run(debug=True)
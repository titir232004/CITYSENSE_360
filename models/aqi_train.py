from aqi import city_data, initialize_system, train_city_model, load_city_data
import os

def train_all_cities():
    print("Loading city data...")
    load_city_data()
    for city in city_data:
        train_city_model(city)
    print("✅ ALL CITY MODELS TRAINED")

if __name__ == "__main__":
    train_all_cities()

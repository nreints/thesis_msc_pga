import pickle
import joblib


with open('h36m_test.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
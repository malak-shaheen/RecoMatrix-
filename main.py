from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

model_data = joblib.load("model.pkl")

svd = model_data["svd"]
user_factors = model_data["user_factors"]
item_factors = model_data["item_factors"]
user_mapping = model_data["user_mapping"]
item_mapping = model_data["item_mapping"]
item_category = model_data["item_category"]

print("ITEM MAPPING SAMPLE:")
print(list(item_mapping.items())[:10])

reverse_item_mapping = item_mapping

@app.get("/")
def home():
    return {"message": "Recommendation API is running"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 5):
    if user_id < 0 or user_id >= len(user_factors):
        raise HTTPException(status_code=404, detail="User not found")

    user_idx = user_id
    scores = np.dot(user_factors[user_idx], item_factors.T)
    top_indices = np.argsort(scores)[::-1][:top_k]

    recommendations = []
    for idx in top_indices:
        item_id = reverse_item_mapping[int(idx)]

        recommendations.append({
            "item_id": item_id,
            "score": round(float(scores[idx]), 3)
        })

    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": recommendations
    }
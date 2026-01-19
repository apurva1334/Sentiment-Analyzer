from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # safe for demo projects
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class TextInput(BaseModel):
    text: str

@app.post("/predict-sentiment")
def predict_sentiment(data: TextInput):
    X = vectorizer.transform([data.text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    return {
        "sentiment": pred,
        "confidence": round(float(prob), 2),
        "emoji": "😊" if pred == "positive" else "😞"
    }

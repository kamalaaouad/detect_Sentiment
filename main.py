import json

import uvicorn
from fastapi import FastAPI
from transformers import pipeline

path = './distilbert-base-uncased-finetuned-sst-2-english'

sentiment_analysis = pipeline("sentiment-analysis", model='ProsusAI/finbert')

"""sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = False
)
"""
app = FastAPI()


@app.post("/detect_sentiment")
async def detect_sentiment(sentence: str):
    res = sentiment_analysis(sentence)
    return json.dumps(res[0])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)

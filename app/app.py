from fastapi import FastAPI
from app.schemas import RankRequest
from app.main import rank_articles

app = FastAPI(title="Viral Ranking API")


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/rank")
def rank(data: RankRequest):
    ranked = rank_articles(data.articles)
    return {"ranked": ranked}
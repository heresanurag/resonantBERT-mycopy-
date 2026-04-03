from pydantic import BaseModel
from typing import List


class Article(BaseModel):
    headline: str
    content: str


class RankRequest(BaseModel):
    articles: List[Article]


class RankedArticle(BaseModel):
    headline: str
    content: str
    score: float
    rank: int
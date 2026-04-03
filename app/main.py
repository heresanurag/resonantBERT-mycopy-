from model.ranker import ViralRankNet
from model.loader import model_loader, device

# Initialize ranker
bert, tokenizer, sent_model = model_loader.load()
ranker = ViralRankNet(bert, tokenizer, sent_model, device)


def rank_articles(articles):
    """Rank articles by virality score"""
    headlines = [a.headline for a in articles]
    contents = [a.content for a in articles]
    
    scores = ranker.score(headlines, contents)
    
    # Sort by score (descending)
    ranked = sorted(
        zip(articles, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    results = []
    for i, (article, score) in enumerate(ranked, start=1):
        results.append({
            "headline": article.headline,
            "content": article.content,
            "score": float(score),
            "rank": i
        })
    
    return results
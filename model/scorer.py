"""
DEPRECATED: Use model/ranker.py instead

ViralScorer has been replaced with ViralRankNet for better architecture.
This file is kept for backwards compatibility only.
"""

from model.ranker import ViralRankNet
from model.loader import model_loader, device as device_


class ViralScorer:
    """Legacy interface - redirects to ViralRankNet"""
    
    def __init__(self):
        self.bert, self.tokenizer, self.sent_model = model_loader.load()
        self.ranker = ViralRankNet(self.bert, self.tokenizer, self.sent_model, device_)

    @property
    def score(self):
        """Redirect to ranker for backwards compatibility"""
        return self.ranker.score
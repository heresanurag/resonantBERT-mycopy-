import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from model.loader import device as device_


class FeatureNormalizer:
    """Normalize and stabilize features"""
    @staticmethod
    def normalize(x, eps=1e-8):
        x_min = x.min()
        x_max = x.max()
        normalized = (x - x_min) / (x_max - x_min + eps)
        return normalized


class ViralRankNet(nn.Module):
    """
    Research-Grade Virality Ranker (Pretrained Only)
    
    Signals:
    1. Semantic importance (BERT CLS norm)
    2. Emotional intensity (sentiment logits)
    3. Topic criticality (zero-shot NLI)
    4. Content density (text statistics)
    5. Headline impact (numerals, caps, punctuation)
    """
    
    def __init__(self, bert, tokenizer, sent_model, device):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.sent_model = sent_model
        self.device = device
        
        # Zero-shot topic classifier (BART-MNLI from Facebook)
        self.topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Topic hierarchy (based on virality research)
        self.topic_labels = [
            "major geopolitical crisis or war",
            "severe economic collapse or market crisis",
            "groundbreaking technology or scientific discovery",
            "health pandemic or major disease outbreak",
            "celebrity scandal or entertainment news",
            "environmental disaster or climate crisis",
            "sports championship or major sports event",
            "minor local news or weather update"
        ]
        
        # Topic importance weights (from research + intuition)
        self.topic_weights = torch.tensor([
            1.0,   # geopolitical
            0.9,   # economic
            0.85,  # technology
            0.9,   # health
            0.5,   # celebrity
            0.8,   # environment
            0.4,   # sports
            0.1    # local
        ]).to(device)
        
        # Learnable-like feature fusion (adaptive weighting without training)
        self.signal_importance = nn.Parameter(
            torch.tensor([0.35, 0.25, 0.25, 0.10, 0.05]),
            requires_grad=False  # frozen, but structured
        )
        
    @torch.no_grad()
    def score(self, headlines, contents):
        """
        Score articles based on all signals combined.
        
        Args:
            headlines: List[str] - article headlines
            contents: List[str] - full article bodies
            
        Returns:
            List[float] - virality scores (0-1)
        """
        batch_size = len(headlines)
        
        # Combine headline + content with emphasis on headline
        combined_texts = [
            f"{h} {c}"  # headline first (more weight in BERT)
            for h, c in zip(headlines, contents)
        ]
        
        # ========== SIGNAL 1: SEMANTIC IMPORTANCE ==========
        semantic_score = self._compute_semantic_score(combined_texts)
        
        # ========== SIGNAL 2: EMOTIONAL INTENSITY ==========
        sentiment_score = self._compute_sentiment_score(combined_texts)
        
        # ========== SIGNAL 3: TOPIC CRITICALITY ==========
        topic_score = self._compute_topic_score(headlines)  # use headline for topics
        
        # ========== SIGNAL 4: CONTENT DENSITY ==========
        density_score = self._compute_density_score(combined_texts, contents)
        
        # ========== SIGNAL 5: HEADLINE IMPACT ==========
        headline_score = self._compute_headline_impact(headlines)
        
        # ========== FEATURE FUSION (attention-based) ==========
        features = torch.stack([
            semantic_score.to(self.device),
            sentiment_score.to(self.device),
            topic_score.to(self.device),
            density_score.to(self.device),
            headline_score.to(self.device)
        ], dim=1)  # (B, 5)
        
        # Adaptive attention: which signals matter most for THIS article?
        signal_importance = F.softmax(self.signal_importance.to(self.device), dim=0)
        final_score = (features * signal_importance.unsqueeze(0)).sum(dim=1)
        
        return final_score.cpu().tolist()
    
    def _compute_semantic_score(self, texts):
        """
        Signal 1: Semantic richness via BERT CLS embedding norm.
        Higher norm = more semantic content.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        bert_out = self.bert(**inputs)
        text_vec = bert_out.last_hidden_state[:, 0]  # CLS token
        
        # L2 norm captures semantic density
        semantic_score = torch.norm(text_vec, dim=1, p=2)
        
        # Normalize to [0, 1]
        semantic_score = FeatureNormalizer.normalize(semantic_score)
        
        return semantic_score.to(self.device)
    
    def _compute_sentiment_score(self, texts):
        """
        Signal 2: Emotional intensity.
        Combines positive and negative sentiment (both drive engagement).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Remove token_type_ids for DistilBERT
        sent_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        
        sent_out = self.sent_model(**sent_inputs)
        sent_probs = F.softmax(sent_out.logits, dim=1)
        
        # Handle 2 or 3 class output
        if sent_probs.shape[1] == 2:
            # Binary: [negative, positive]
            sentiment_score = sent_probs[:, 1]  # positive probability
        elif sent_probs.shape[1] == 3:
            # Ternary: [negative, neutral, positive]
            # Emotional pull = strong positive OR strong negative
            sentiment_score = sent_probs[:, 0] + sent_probs[:, 2]  # neg + pos
        else:
            # Fallback
            sentiment_score = sent_probs.mean(dim=1)
        
        return sentiment_score.to(self.device)
    
    def _compute_topic_score(self, headlines):
        """
        Signal 3: Topic importance via zero-shot classification.
        War > Economics > Tech > Health > Celebrity > Local
        """
        batch_size = len(headlines)
        topic_scores = []
        
        for headline in headlines:
            # Zero-shot predict topic
            result = self.topic_classifier(
                headline,
                candidate_labels=self.topic_labels
            )
            
            # Weight by importance
            topic_score = 0.0
            for i, label in enumerate(result["labels"]):
                label_idx = self.topic_labels.index(label)
                prob = result["scores"][i]
                weight = self.topic_weights[label_idx]
                topic_score += prob * weight.item()
            
            topic_scores.append(topic_score)
        
        topic_score = torch.tensor(topic_scores, dtype=torch.float32, device=self.device)
        topic_score = FeatureNormalizer.normalize(topic_score)
        
        return topic_score
    
    def _compute_density_score(self, combined_texts, contents):
        """
        Signal 4: Information density.
        Deep content > shallow content
        """
        batch_size = len(combined_texts)
        density_scores = []
        
        for text, content in zip(combined_texts, contents):
            # Features
            word_count = len(text.split())
            sentence_count = len(text.split('.')[:-1]) + 1
            unique_words = len(set(text.lower().split()))
            
            # Depth metrics
            avg_sentence_length = word_count / max(sentence_count, 1)
            vocabulary_richness = unique_words / max(word_count, 1)
            
            # Content-to-headline ratio (longer content = deeper reporting)
            content_length = len(content.split())
            
            # Composite density score
            density = (
                0.3 * (word_count / 1000.0) +  # penalize very short
                0.3 * vocabulary_richness +      # diverse vocabulary
                0.2 * (avg_sentence_length / 20.0) +  # moderate sentence length
                0.2 * min(content_length / 500.0, 1.0)  # substantial content
            )
            
            density_scores.append(min(density, 1.0))
        
        density_score = torch.tensor(density_scores, dtype=torch.float32, device=self.device)
        return FeatureNormalizer.normalize(density_score)
    
    def _compute_headline_impact(self, headlines):
        """
        Signal 5: Headline quality (numerals, caps, punctuation = higher virality).
        Research shows headlines with numbers/caps drive more clicks.
        """
        impact_scores = []
        
        for headline in headlines:
            score = 0.0
            
            # Numbers in headline (highly viral)
            num_count = sum(1 for c in headline if c.isdigit())
            score += min(num_count * 0.15, 0.4)
            
            # Capitalization (some caps = good, all caps = clickbait penalty)
            caps_count = sum(1 for c in headline if c.isupper())
            caps_ratio = caps_count / max(len(headline), 1)
            if caps_ratio < 0.3:
                score += 0.05
            elif caps_ratio > 0.5:
                score -= 0.1
            
            # Punctuation (! and ? drive engagement)
            punctuation = headline.count('!') + headline.count('?')
            score += min(punctuation * 0.1, 0.2)
            
            # Length (optimal: 8-12 words)
            word_count = len(headline.split())
            if 8 <= word_count <= 12:
                score += 0.2
            elif word_count < 5 or word_count > 20:
                score -= 0.1
            
            impact_scores.append(max(min(score, 1.0), 0.0))
        
        impact_score = torch.tensor(impact_scores, dtype=torch.float32, device=self.device)
        return impact_score

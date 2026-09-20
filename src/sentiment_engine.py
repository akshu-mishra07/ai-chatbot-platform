"""Sentiment Analysis & Emotion-Adaptive Response Engine.

Provides real-time sentiment analysis, urgency detection, and emotion-adaptive
response recommendations using lightweight lexicon-based scoring.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentiment lexicons
POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "love", "happy", "pleased", "satisfied", "helpful", "thank",
    "thanks", "perfect", "awesome", "brilliant", "outstanding",
    "appreciate", "enjoy", "recommend", "best", "superb", "delighted"
]
NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "horrible", "hate", "angry", "upset",
    "disappointed", "frustrated", "annoyed", "worst", "poor", "useless",
    "broken", "fail", "failed", "waste", "disgusting", "pathetic",
    "ridiculous", "complaint", "unacceptable", "refund", "cancel"
]
URGENCY_WORDS = [
    "urgent", "emergency", "immediately", "asap", "critical", "help",
    "now", "hurry", "deadline", "important", "serious"
]

# Negation words that invert sentiment
NEGATION_WORDS = [
    "not", "never", "no", "without", "hardly", "barely", "scarcely",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't",
    "didn't", "haven't", "hasn't", "hadn't", "can't", "couldn't",
    "shouldn't", "won't", "wouldn't"
]

# Intensifier words
INTENSIFIERS = [
    "very", "extremely", "really", "so", "absolutely", "totally",
    "completely", "super", "highly", "exceptionally"
]

# Emoji mappings
SENTIMENT_EMOJIS = {
    "positive": "😊",
    "satisfied": "😄",
    "negative": "😞",
    "frustrated": "😠",
    "urgent": "🚨",
    "neutral": "😐"
}


class SentimentEngine:
    """Real-time sentiment analysis with emotion-adaptive response recommendations."""

    def __init__(self, history_size: int = 100) -> None:
        """Initialize the sentiment engine with a bounded history buffer.

        Args:
            history_size: Maximum number of historical sentiment records to retain.
        """
        self.history_size: int = history_size
        self.history: deque = deque(maxlen=history_size)
        self.session_stats: Dict[str, Any] = {
            "total_messages": 0,
            "sentiment_counts": {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "frustrated": 0,
                "satisfied": 0,
                "urgent": 0
            }
        }
        logger.info("SentimentEngine initialized with history_size=%d", history_size)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a text message.

        Performs lexicon-based scoring, detects intensity, checks for urgency,
        and records the result in the session history.

        Args:
            text: The user message string to analyze.

        Returns:
            Dict containing sentiment, score, intensity, matched_positive,
            matched_negative, is_urgent, emoji_indicator, and timestamp.
        """
        if not isinstance(text, str) or not text.strip():
            empty_result = {
                "sentiment": "neutral",
                "score": 0.0,
                "intensity": "mild",
                "matched_positive": [],
                "matched_negative": [],
                "is_urgent": False,
                "emoji_indicator": self.get_sentiment_emoji("neutral"),
                "timestamp": datetime.now().isoformat()
            }
            return empty_result

        # Normalize text and tokenize into words
        clean_text = text.lower()
        words = re.findall(r"\b[\w']+\b", clean_text)

        matched_positive: List[str] = []
        matched_negative: List[str] = []
        matched_urgency: List[str] = []

        pos_set = set(POSITIVE_WORDS)
        neg_set = set(NEGATIVE_WORDS)
        urg_set = set(URGENCY_WORDS)
        negation_set = set(NEGATION_WORDS)

        for i, word in enumerate(words):
            # Check for preceding negation (within 2 previous words)
            is_negated = False
            for prev_idx in range(max(0, i - 2), i):
                if words[prev_idx] in negation_set:
                    is_negated = True
                    break

            if word in pos_set:
                if is_negated:
                    matched_negative.append(f"not {word}")
                else:
                    matched_positive.append(word)
            elif word in neg_set:
                if is_negated:
                    matched_positive.append(f"not {word}")
                else:
                    matched_negative.append(word)

            if word in urg_set:
                matched_urgency.append(word)

        # Check for phrase-level urgency triggers
        if "asap" in clean_text and "asap" not in matched_urgency:
            matched_urgency.append("asap")
        if "right now" in clean_text and "now" not in matched_urgency:
            matched_urgency.append("right now")

        pos_count = len(matched_positive)
        neg_count = len(matched_negative)
        urg_count = len(matched_urgency)
        total_sentiment_words = pos_count + neg_count

        # Compound score calculation (-1.0 to 1.0)
        if total_sentiment_words > 0:
            raw_score = (pos_count - neg_count) / total_sentiment_words
            score = round(max(-1.0, min(1.0, raw_score)), 2)
        else:
            score = 0.0

        # Urgency flag
        is_urgent = urg_count > 0 or ("!" in text and urg_count > 0)

        # Intensity detection (mild, moderate, strong)
        has_intensifier = any(w in INTENSIFIERS for w in words)
        exclamation_count = text.count("!")
        is_shouting = text.isupper() and len(text.strip()) > 3

        if (
            abs(score) >= 0.7
            or total_sentiment_words >= 3
            or exclamation_count >= 2
            or (is_shouting and total_sentiment_words > 0)
            or (has_intensifier and total_sentiment_words > 0)
        ):
            intensity = "strong"
        elif abs(score) >= 0.3 or total_sentiment_words >= 2 or exclamation_count == 1 or has_intensifier:
            intensity = "moderate"
        else:
            intensity = "mild"

        # Sentiment classification
        strong_negatives = {
            "frustrated", "angry", "annoyed", "terrible", "horrible",
            "worst", "unacceptable", "hate", "disgusting"
        }
        strong_positives = {
            "amazing", "excellent", "wonderful", "fantastic", "superb",
            "delighted", "perfect", "outstanding", "brilliant"
        }

        has_strong_negative = any(w in strong_negatives for w in matched_negative)
        has_strong_positive = any(w in strong_positives for w in matched_positive)

        if is_urgent and score <= 0:
            sentiment = "urgent"
        elif score <= -0.5 or (neg_count >= 2 and score < 0) or (has_strong_negative and score < 0):
            sentiment = "frustrated"
        elif score < -0.1:
            sentiment = "negative"
        elif score >= 0.6 or (pos_count >= 2 and score > 0) or (has_strong_positive and score > 0):
            sentiment = "satisfied"
        elif score > 0.1:
            sentiment = "positive"
        elif is_urgent:
            sentiment = "urgent"
        else:
            sentiment = "neutral"

        emoji = self.get_sentiment_emoji(sentiment)
        timestamp = datetime.now().isoformat()

        result = {
            "sentiment": sentiment,
            "score": score,
            "intensity": intensity,
            "matched_positive": matched_positive,
            "matched_negative": matched_negative,
            "is_urgent": is_urgent,
            "emoji_indicator": emoji,
            "timestamp": timestamp,
        }

        # Update history and session statistics
        self.history.append(result)
        self.session_stats["total_messages"] += 1
        if sentiment in self.session_stats["sentiment_counts"]:
            self.session_stats["sentiment_counts"][sentiment] += 1

        logger.info("Analyzed sentiment: %s (score=%.2f, intensity=%s)", sentiment, score, intensity)
        return result

    def get_response_modifier(self, sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get response modification recommendations based on detected sentiment.

        Args:
            sentiment_result: Sentiment analysis result dictionary.

        Returns:
            Dict containing tone, temperature_adjustment, prefix_suggestion,
            suffix_suggestion, and escalate.
        """
        sentiment = sentiment_result.get("sentiment", "neutral")
        intensity = sentiment_result.get("intensity", "mild")
        is_urgent = sentiment_result.get("is_urgent", False)

        if sentiment == "frustrated":
            if intensity == "strong":
                prefix = "I am deeply sorry for the frustrating experience you've had. Let's get this resolved for you immediately."
            else:
                prefix = "I understand your frustration and apologize for the inconvenience. Let me help you with this right away."
            suffix = "If you would prefer, I can promptly escalate this ticket to a human supervisor."
            return {
                "tone": "empathetic, apologetic, and solution-focused",
                "temperature_adjustment": -0.2,
                "prefix_suggestion": prefix,
                "suffix_suggestion": suffix,
                "escalate": True
            }

        elif sentiment == "negative":
            prefix = "I apologize for any difficulty you are experiencing. I am here to help get this sorted out."
            suffix = "Please let me know if there's anything else I can do to assist you."
            return {
                "tone": "empathetic, attentive, and helpful",
                "temperature_adjustment": -0.1,
                "prefix_suggestion": prefix,
                "suffix_suggestion": suffix,
                "escalate": is_urgent
            }

        elif sentiment == "satisfied":
            prefix = "Thank you so much! I'm thrilled to hear that everything went well."
            suffix = "We truly appreciate your feedback! If you have a moment, please consider leaving us a review."
            return {
                "tone": "enthusiastic, warm, and appreciative",
                "temperature_adjustment": 0.1,
                "prefix_suggestion": prefix,
                "suffix_suggestion": suffix,
                "escalate": False
            }

        elif sentiment == "positive":
            prefix = "Thank you for reaching out! I'm glad I could assist you."
            suffix = "Have a wonderful day! Please let me know if there's anything else you need."
            return {
                "tone": "friendly, positive, and helpful",
                "temperature_adjustment": 0.0,
                "prefix_suggestion": prefix,
                "suffix_suggestion": suffix,
                "escalate": False
            }

        elif sentiment == "urgent":
            prefix = "I understand this is an urgent matter. I am prioritizing your request right now."
            suffix = "If immediate live support is required, please say 'agent' and I will transfer you instantly."
            return {
                "tone": "direct, rapid, concise, and prioritized",
                "temperature_adjustment": -0.3,
                "prefix_suggestion": prefix,
                "suffix_suggestion": suffix,
                "escalate": True
            }

        else:  # neutral
            return {
                "tone": "professional, courteous, and informative",
                "temperature_adjustment": 0.0,
                "prefix_suggestion": "",
                "suffix_suggestion": "Please feel free to ask if you have any further questions.",
                "escalate": False
            }

    def get_sentiment_emoji(self, sentiment: str) -> str:
        """Return appropriate emoji for sentiment category.

        Args:
            sentiment: Sentiment classification string.

        Returns:
            String containing the corresponding emoji.
        """
        return SENTIMENT_EMOJIS.get(sentiment.lower(), "😐")

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for the current session.

        Aggregates history to compute average score, sentiment distribution,
        trend (improving/declining/stable), and satisfaction rate.

        Returns:
            Dict containing total_messages, avg_score, distribution,
            trend, and satisfaction_rate.
        """
        total = len(self.history)
        if total == 0:
            return {
                "total_messages": 0,
                "avg_score": 0.0,
                "distribution": {k: 0 for k in SENTIMENT_EMOJIS.keys()},
                "trend": "stable",
                "satisfaction_rate": 0.0
            }

        scores = [item["score"] for item in self.history]
        avg_score = round(sum(scores) / total, 2)

        distribution: Dict[str, int] = {k: 0 for k in SENTIMENT_EMOJIS.keys()}
        for item in self.history:
            s = item.get("sentiment", "neutral")
            if s in distribution:
                distribution[s] += 1
            else:
                distribution[s] = 1

        # Trend calculation: compare first half vs second half
        if total >= 2:
            mid = total // 2
            first_half = scores[:mid]
            second_half = scores[mid:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            diff = avg_second - avg_first
            if diff > 0.15:
                trend = "improving"
            elif diff < -0.15:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Satisfaction rate (% of positive or satisfied messages)
        positive_count = distribution.get("positive", 0) + distribution.get("satisfied", 0)
        satisfaction_rate = round((positive_count / total) * 100, 1)

        return {
            "total_messages": total,
            "avg_score": avg_score,
            "distribution": distribution,
            "trend": trend,
            "satisfaction_rate": satisfaction_rate
        }

    def get_sentiment_trend(self) -> List[Dict[str, Any]]:
        """Return list of timestamp, score, and sentiment for charting.

        Returns:
            List of dicts with timestamp, score, sentiment, and emoji.
        """
        return [
            {
                "timestamp": item["timestamp"],
                "score": item["score"],
                "sentiment": item["sentiment"],
                "emoji": item["emoji_indicator"]
            }
            for item in self.history
        ]

    def reset_session(self) -> None:
        """Clear history and session statistics."""
        self.history.clear()
        self.session_stats = {
            "total_messages": 0,
            "sentiment_counts": {
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "frustrated": 0,
                "satisfied": 0,
                "urgent": 0
            }
        }
        logger.info("SentimentEngine session reset.")

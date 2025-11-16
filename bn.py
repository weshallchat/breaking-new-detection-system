#!/usr/bin/env python3
"""
Breaking News Detection Service using Zero-Shot Classification

Modern approach using transformer-based zero-shot classification for accurate
breaking news detection without training data.
"""

import asyncio
import csv
import json
import logging
import hashlib
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import threading

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NewsNotification:
    """Represents a news notification"""
    id: str
    title: str
    description: str
    pubDate: datetime
    link: str
    guid: str
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['pubDate'] = self.pubDate.isoformat()
        return data


@dataclass
class BreakingNewsClassification:
    """Classification result for a news item"""
    notification: NewsNotification
    is_breaking: bool
    breaking_score: float
    urgency_score: float
    importance_score: float
    category: str
    subcategories: Dict[str, float]
    detected_at: datetime
    expires_at: datetime
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'notification': self.notification.to_dict(),
            'is_breaking': self.is_breaking,
            'breaking_score': round(self.breaking_score, 3),
            'urgency_score': round(self.urgency_score, 3),
            'importance_score': round(self.importance_score, 3),
            'category': self.category,
            'subcategories': {k: round(v, 3) for k, v in self.subcategories.items()},
            'detected_at': self.detected_at.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }


class ZeroShotBreakingNewsClassifier:
    """
    Zero-shot classifier for breaking news detection using transformer models
    
    This classifier uses pre-trained language models to classify news without
    any training on breaking news data specifically.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-mnli",
                 breaking_threshold: float = 0.7,
                 expiry_hours: int = 4,
                 device: str = None):
        """
        Initialize the zero-shot classifier
        
        Args:
            model_name: HuggingFace model for zero-shot classification
            breaking_threshold: Minimum score to classify as breaking
            expiry_hours: Hours before breaking news expires
            device: Device to run model on (cuda/cpu/mps)
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing zero-shot classifier with {model_name} on {self.device}")
        
        # Initialize the zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        self.breaking_threshold = breaking_threshold
        self.expiry_duration = timedelta(hours=expiry_hours)
        
        # Define classification labels for different aspects
        self.breaking_labels = [
            "breaking news",
            "urgent development", 
            "major event",
            "routine news",
            "minor update",
            "background information"
        ]
        
        self.urgency_labels = [
            "extremely urgent",
            "very urgent",
            "urgent",
            "moderately urgent",
            "not urgent"
        ]
        
        self.importance_labels = [
            "globally significant",
            "nationally significant", 
            "regionally significant",
            "locally significant",
            "minor significance"
        ]
        
        self.category_labels = [
            "war and conflict",
            "natural disaster",
            "political crisis",
            "economic crisis",
            "public health emergency",
            "terrorism",
            "major accident",
            "technology breakthrough",
            "sports",
            "entertainment",
            "general news"
        ]
        
        # Cache for recent classifications to avoid reprocessing
        self.cache = {}
        self.cache_lock = threading.Lock()
        
    def classify_breaking(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """
        Classify if news is breaking using zero-shot classification
        
        Returns:
            Tuple of (is_breaking, confidence_score, label_scores)
        """
        result = self.classifier(
            text,
            candidate_labels=self.breaking_labels,
            hypothesis_template="This news is {}.",
            multi_label=False
        )
        
        # Create score dictionary
        scores = dict(zip(result['labels'], result['scores']))
        
        # Calculate breaking score (weight the positive labels)
        breaking_score = max(
            scores.get("breaking news", 0),
            scores.get("urgent development", 0),
            scores.get("major event", 0)
        ) # Normalize
        
        is_breaking = breaking_score >= self.breaking_threshold
        
        return is_breaking, breaking_score, scores
    
    def classify_urgency(self, text: str) -> Tuple[float, str]:
        """
        Classify urgency level of news
        
        Returns:
            Tuple of (urgency_score, urgency_label)
        """
        result = self.classifier(
            text,
            candidate_labels=self.urgency_labels,
            hypothesis_template="This news is {}.",
            multi_label=False
        )
        
        # Map labels to scores
        urgency_map = {
            "extremely urgent": 1.0,
            "very urgent": 0.8,
            "urgent": 0.6,
            "moderately urgent": 0.4,
            "not urgent": 0.2
        }
        
        top_label = result['labels'][0]
        urgency_score = urgency_map.get(top_label, 0.5)
        
        return urgency_score, top_label
    
    def classify_importance(self, text: str) -> Tuple[float, str]:
        """
        Classify importance/significance of news
        
        Returns:
            Tuple of (importance_score, importance_label)
        """
        result = self.classifier(
            text,
            candidate_labels=self.importance_labels,
            hypothesis_template="This news is {}.",
            multi_label=False
        )
        
        # Map labels to scores
        importance_map = {
            "globally significant": 1.0,
            "nationally significant": 0.8,
            "regionally significant": 0.6,
            "locally significant": 0.4,
            "minor significance": 0.2
        }
        
        top_label = result['labels'][0]
        importance_score = importance_map.get(top_label, 0.5)
        
        return importance_score, top_label
    
    def classify_category(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Classify news category
        
        Returns:
            Tuple of (main_category, category_scores)
        """
        result = self.classifier(
            text,
            candidate_labels=self.category_labels,
            hypothesis_template="This news is about {}.",
            multi_label=True  # Allow multiple categories
        )
        
        scores = dict(zip(result['labels'], result['scores']))
        main_category = result['labels'][0]
        
        return main_category, scores
    
    def classify_news(self, notification: NewsNotification) -> BreakingNewsClassification:
        """
        Perform complete classification of a news notification
        
        Returns:
            BreakingNewsClassification object with all classification results
        """
        # Check cache first
        cache_key = notification.id
        with self.cache_lock:
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                # Update timestamps
                cached.detected_at = datetime.now()
                cached.expires_at = datetime.now() + self.expiry_duration
                return cached
        
        # Combine title and description for better context
        text = f"{notification.title}. {notification.description}"
        
        # Perform multi-aspect classification
        logger.debug(f"Classifying: {notification.title[:60]}...")
        
        # 1. Breaking news classification
        is_breaking, breaking_score, breaking_scores = self.classify_breaking(text)
        
        # 2. Urgency classification
        urgency_score, urgency_label = self.classify_urgency(text)
        
        # 3. Importance classification  
        importance_score, importance_label = self.classify_importance(text)
        
        # 4. Category classification
        category, category_scores = self.classify_category(text)
        
        # Create classification result
        classification = BreakingNewsClassification(
            notification=notification,
            is_breaking=is_breaking,
            breaking_score=breaking_score,
            urgency_score=urgency_score,
            importance_score=importance_score,
            category=category,
            subcategories=category_scores,
            detected_at=datetime.now(),
            expires_at=datetime.now() + self.expiry_duration
        )
        
        # Cache the result
        with self.cache_lock:
            self.cache[cache_key] = classification
            # Limit cache size
            if len(self.cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(self.cache.keys(), 
                                   key=lambda k: self.cache[k].detected_at)[:100]
                for key in oldest_keys:
                    del self.cache[key]
        
        if is_breaking:
            logger.info(f"🚨 BREAKING: {notification.title[:60]}... "
                       f"(score: {breaking_score:.2f}, urgency: {urgency_score:.2f})")
        
        return classification


class AdvancedNewsStreamProcessor:
    """
    Advanced processor using zero-shot classification for breaking news
    """
    
    def __init__(self, classifier: ZeroShotBreakingNewsClassifier):
        """Initialize the processor"""
        self.classifier = classifier
        self.breaking_news: Dict[str, BreakingNewsClassification] = {}
        self.all_classifications: deque = deque(maxlen=10000)
        self.processed_count = 0
        self.breaking_count = 0
        self.lock = threading.Lock()
        
        # Statistics tracking
        self.category_counts = defaultdict(int)
        self.urgency_distribution = defaultdict(int)
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
    
    def process_notification(self, notification: NewsNotification) -> BreakingNewsClassification:
        """
        Process a notification using zero-shot classification
        
        Returns:
            Classification result
        """
        self.processed_count += 1
        
        # Classify the notification
        classification = self.classifier.classify_news(notification)
        
        # Store results
        with self.lock:
            self.all_classifications.append(classification)
            
            if classification.is_breaking:
                self.breaking_news[notification.id] = classification
                self.breaking_count += 1
            
            # Update statistics
            self.category_counts[classification.category] += 1
            urgency_bucket = int(classification.urgency_score * 5)
            self.urgency_distribution[urgency_bucket] += 1
        
        return classification
    
    def get_active_breaking_news(self, 
                                min_score: float = 0.0,
                                category: Optional[str] = None) -> List[BreakingNewsClassification]:
        """Get currently active breaking news with filters"""
        with self.lock:
            now = datetime.now()
            active = [
                item for item in self.breaking_news.values()
                if item.expires_at > now and item.breaking_score >= min_score
            ]
            
            if category:
                active = [item for item in active if item.category == category]
            
            # Sort by combined score (breaking * urgency * importance)
            active.sort(
                key=lambda x: x.breaking_score * x.urgency_score * x.importance_score,
                reverse=True
            )
            
            return active
    
    def get_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        active_breaking = len(self.get_active_breaking_news())
        
        with self.lock:
            # Calculate average scores for breaking news
            if self.breaking_news:
                avg_breaking_score = sum(item.breaking_score for item in self.breaking_news.values()) / len(self.breaking_news)
                avg_urgency = sum(item.urgency_score for item in self.breaking_news.values()) / len(self.breaking_news)
                avg_importance = sum(item.importance_score for item in self.breaking_news.values()) / len(self.breaking_news)
            else:
                avg_breaking_score = avg_urgency = avg_importance = 0
            
            return {
                'processed_count': self.processed_count,
                'breaking_count': self.breaking_count,
                'active_breaking_count': active_breaking,
                'detection_rate': self.breaking_count / max(1, self.processed_count),
                'avg_breaking_score': avg_breaking_score,
                'avg_urgency_score': avg_urgency,
                'avg_importance_score': avg_importance,
                'category_distribution': dict(self.category_counts),
                'urgency_distribution': dict(self.urgency_distribution)
            }
    
    def _cleanup_expired(self):
        """Background thread to cleanup expired breaking news"""
        while True:
            time.sleep(60)  # Check every minute
            with self.lock:
                now = datetime.now()
                expired_ids = [
                    id for id, item in self.breaking_news.items()
                    if item.expires_at < now - timedelta(hours=24)
                ]
                for id in expired_ids:
                    del self.breaking_news[id]
                
                if expired_ids:
                    logger.info(f"Cleaned up {len(expired_ids)} expired items")


class IntelligentNewsStreamSimulator:
    """
    Simulates real-time news stream with intelligent batching
    """
    
    def __init__(self, csv_file: str, processor: AdvancedNewsStreamProcessor):
        """Initialize the simulator"""
        self.csv_file = csv_file
        self.processor = processor
        self.running = False
        # self.speed_multiplier = 3600  # 1 hour = 1 second
    
    async def start_stream(self, start_date: datetime, end_date: datetime, batch_size: int = 5):
        """
        Start streaming with intelligent batching for efficiency
        
        Args:
            start_date: Start date for simulation
            end_date: End date for simulation  
            batch_size: Number of items to process in parallel
        """
        self.running = True
        notifications = self._load_notifications(start_date, end_date)
        
        if not notifications:
            logger.error("No notifications found in date range")
            return
        
        logger.info(f"Starting intelligent stream with {len(notifications)} notifications")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Batch size: {batch_size} (parallel processing)")
        
        # Group by hour for realistic streaming
        by_hour = defaultdict(list)
        for notif in notifications:
            hour_key = notif.pubDate.replace(minute=0, second=0, microsecond=0)
            by_hour[hour_key].append(notif)
        
        # Process in time order
        start_real_time = time.time()
        start_sim_time = min(by_hour.keys())
        
        for hour in sorted(by_hour.keys()):
            if not self.running:
                break
            
            # Calculate timing
            # sim_elapsed = (hour - start_sim_time).total_seconds()
            # real_target = sim_elapsed / self.speed_multiplier
            # real_elapsed = time.time() - start_real_time
            
            # if real_target > real_elapsed:
            #     await asyncio.sleep(real_target - real_elapsed)
            
            # Process batch of notifications for this hour
            batch = by_hour[hour]
            
            # Process in parallel batches for efficiency
            for i in range(0, len(batch), batch_size):
                chunk = batch[i:i+batch_size]
                
                # Process chunk
                for notification in chunk:
                    classification = self.processor.process_notification(notification)
                    
                    if classification.is_breaking:
                        logger.info(
                            f"⚡ BREAKING [{classification.category}]: "
                            f"{notification.title[:50]}... "
                            f"(scores: B={classification.breaking_score:.2f}, "
                            f"U={classification.urgency_score:.2f}, "
                            f"I={classification.importance_score:.2f})"
                        )
            
            # Log progress
            if self.processor.processed_count % 20 == 0:
                stats = self.processor.get_statistics()
                logger.info(
                    f"Progress: {stats['processed_count']} processed, "
                    f"{stats['breaking_count']} breaking "
                    f"({stats['detection_rate']*100:.1f}% rate)"
                )
        
        logger.info("Stream simulation completed")
    
    def _load_notifications(self, start_date: datetime, end_date: datetime) -> List[NewsNotification]:
        """Load notifications from CSV within date range"""
        notifications = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pub_date = datetime.strptime(row['pubDate'], '%a, %d %b %Y %H:%M:%S %Z')
                    
                    if start_date <= pub_date <= end_date:
                        notif = NewsNotification(
                            id=hashlib.md5(row['guid'].encode()).hexdigest(),
                            title=row['title'],
                            description=row['description'],
                            pubDate=pub_date,
                            link=row['link'],
                            guid=row['guid']
                        )
                        notifications.append(notif)
                except Exception as e:
                    logger.debug(f"Error parsing row: {e}")
        
        notifications.sort(key=lambda x: x.pubDate)
        return notifications
    
    def stop_stream(self):
        """Stop the stream"""
        self.running = False


async def main():
    """Main function demonstrating zero-shot breaking news classification"""
    
    logger.info("=" * 60)
    logger.info("ZERO-SHOT BREAKING NEWS DETECTION SYSTEM")
    logger.info("=" * 60)
    
    # Initialize the zero-shot classifier
    classifier = ZeroShotBreakingNewsClassifier(
        model_name="facebook/bart-large-mnli",  # Can also use "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        breaking_threshold=0.5,
        expiry_hours=4
    )
    
    # Initialize processor
    processor = AdvancedNewsStreamProcessor(classifier)
    
    # Initialize simulator
    simulator = IntelligentNewsStreamSimulator(
        'bbc_news.csv',
        processor
    )
    
    # Set date range (Ukraine conflict week)
    start_date = datetime(2022, 3, 7, 0, 0, 0)
    end_date = datetime(2022, 3, 14, 23, 59, 59)
    
    # Start streaming with intelligent batching
    logger.info(f"Starting simulation (1 hour = 1 second)")
    logger.info(f"Using zero-shot classification - no training required!")
    
    try:
        await simulator.start_stream(start_date, end_date, batch_size=3)
    except KeyboardInterrupt:
        logger.info("Stopping simulation...")
        simulator.stop_stream()
    
    # Final statistics and analysis
    stats = processor.get_statistics()
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total Processed: {stats['processed_count']}")
    logger.info(f"Breaking News Detected: {stats['breaking_count']}")
    logger.info(f"Detection Rate: {stats['detection_rate']*100:.1f}%")
    logger.info(f"Average Breaking Score: {stats['avg_breaking_score']:.3f}")
    logger.info(f"Average Urgency Score: {stats['avg_urgency_score']:.3f}")
    logger.info(f"Average Importance Score: {stats['avg_importance_score']:.3f}")
    
    # Show category distribution
    logger.info("\nCategory Distribution:")
    for category, count in sorted(stats['category_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {category}: {count} articles")
    
    # Show top breaking news
    active = processor.get_active_breaking_news()
    if active:
        logger.info(f"\nTop 5 Breaking News (by combined score):")
        for i, item in enumerate(active[:5], 1):
            combined_score = item.breaking_score * item.urgency_score * item.importance_score
            logger.info(
                f"{i}. [{item.category}] {item.notification.title[:60]}..."
            )
            logger.info(
                f"   Scores: Breaking={item.breaking_score:.2f}, "
                f"Urgency={item.urgency_score:.2f}, "
                f"Importance={item.importance_score:.2f}, "
                f"Combined={combined_score:.3f}"
            )


if __name__ == "__main__":
    asyncio.run(main())
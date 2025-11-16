#!/usr/bin/env python3
"""
FastAPI Server for Zero-Shot Breaking News Classification

Modern API using transformer-based classification for breaking news detection.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from bn import (
    ZeroShotBreakingNewsClassifier,
    AdvancedNewsStreamProcessor,
    IntelligentNewsStreamSimulator,
    NewsNotification,
    BreakingNewsClassification
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class NewsNotificationResponse(BaseModel):
    """API response model for news notifications"""
    id: str
    title: str
    description: str
    pubDate: str
    link: str
    guid: str


class BreakingNewsResponse(BaseModel):
    """API response model for breaking news with zero-shot classification"""
    notification: NewsNotificationResponse
    is_breaking: bool
    breaking_score: float = Field(description="Breaking news confidence (0-1)")
    urgency_score: float = Field(description="Urgency level (0-1)")
    importance_score: float = Field(description="Importance/significance (0-1)")
    combined_score: float = Field(description="Combined score (product of all scores)")
    category: str = Field(description="Primary news category")
    subcategories: Dict[str, float] = Field(description="All category scores")
    detected_at: str
    expires_at: str


class SystemStats(BaseModel):
    """Enhanced system statistics with ML metrics"""
    processed_count: int
    breaking_count: int
    active_breaking_count: int
    detection_rate: float
    avg_breaking_score: float
    avg_urgency_score: float
    avg_importance_score: float
    category_distribution: Dict[str, int]
    uptime_seconds: float
    model_info: Dict[str, str]


class ModelConfig(BaseModel):
    """Configuration for the ML model"""
    model_name: str = Field(default="facebook/bart-large-mnli", description="HuggingFace model name")
    breaking_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Breaking news threshold")
    expiry_hours: int = Field(default=4, ge=1, le=24, description="Hours until breaking news expires")


class StreamConfig(BaseModel):
    """Configuration for stream control"""
    speed_multiplier: int = Field(default=3600, description="Simulation speed")
    start_date: Optional[str] = Field(default=None, description="Start date (ISO)")
    end_date: Optional[str] = Field(default=None, description="End date (ISO)")
    batch_size: int = Field(default=3, ge=1, le=10, description="Batch processing size")


# Global instances
classifier: Optional[ZeroShotBreakingNewsClassifier] = None
processor: Optional[AdvancedNewsStreamProcessor] = None
simulator: Optional[IntelligentNewsStreamSimulator] = None
stream_task: Optional[asyncio.Task] = None
service_start_time: datetime = datetime.now()
current_model_name: str = "facebook/bart-large-mnli"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global classifier, processor, simulator, stream_task, service_start_time
    
    # Startup
    logger.info("Initializing Zero-Shot Breaking News Classification Service...")
    
    try:
        # Initialize classifier with transformer model
        classifier = ZeroShotBreakingNewsClassifier(
            model_name=current_model_name,
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
        
        service_start_time = datetime.now()
        
        # Start default stream
        start_date = datetime(1900, 1, 1, 0, 0, 0)
        end_date = datetime(2099, 12, 31, 23, 59, 59)
        
        stream_task = asyncio.create_task(
            simulator.start_stream(start_date, end_date, batch_size=3)
        )
        
        logger.info(f"Service initialized with model: {current_model_name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")
    if simulator:
        simulator.stop_stream()
    if stream_task:
        stream_task.cancel()


# Create FastAPI app
app = FastAPI(
    title="Zero-Shot Breaking News Classification API",
    description="Modern breaking news detection using transformer-based zero-shot classification",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - service health check"""
    return {
        "service": "Zero-Shot Breaking News Classification API",
        "status": "running",
        "version": "2.0.0",
        "model": current_model_name,
        "approach": "transformer-based zero-shot classification"
    }


@app.get("/breaking", response_model=List[BreakingNewsResponse], tags=["Breaking News"])
async def get_breaking_news(
    limit: int = Query(default=10, ge=1, le=50, description="Maximum items"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum breaking score"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    sort_by: str = Query(default="combined", description="Sort by: combined, breaking, urgency, importance")
):
    """
    Get active breaking news classified by zero-shot ML model
    
    Returns news items with multi-dimensional scoring from the transformer model.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Get active breaking news
    active_news = processor.get_active_breaking_news(
        min_score=min_score,
        category=category
    )
    
    # Sort based on parameter
    if sort_by == "breaking":
        active_news.sort(key=lambda x: x.breaking_score, reverse=True)
    elif sort_by == "urgency":
        active_news.sort(key=lambda x: x.urgency_score, reverse=True)
    elif sort_by == "importance":
        active_news.sort(key=lambda x: x.importance_score, reverse=True)
    else:  # combined (default)
        active_news.sort(
            key=lambda x: x.breaking_score * x.urgency_score * x.importance_score,
            reverse=True
        )
    
    # Limit results
    active_news = active_news[:limit]
    
    # Convert to response models
    response = []
    for item in active_news:
        combined = item.breaking_score * item.urgency_score * item.importance_score
        
        notif_response = NewsNotificationResponse(
            id=item.notification.id,
            title=item.notification.title,
            description=item.notification.description,
            pubDate=item.notification.pubDate.isoformat(),
            link=item.notification.link,
            guid=item.notification.guid
        )
        
        breaking_response = BreakingNewsResponse(
            notification=notif_response,
            is_breaking=item.is_breaking,
            breaking_score=round(item.breaking_score, 3),
            urgency_score=round(item.urgency_score, 3),
            importance_score=round(item.importance_score, 3),
            combined_score=round(combined, 3),
            category=item.category,
            subcategories={k: round(v, 3) for k, v in item.subcategories.items()},
            detected_at=item.detected_at.isoformat(),
            expires_at=item.expires_at.isoformat()
        )
        response.append(breaking_response)
    
    return response


@app.get("/breaking/{news_id}", response_model=BreakingNewsResponse, tags=["Breaking News"])
async def get_breaking_news_by_id(news_id: str):
    """Get specific breaking news item with full classification details"""
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    with processor.lock:
        if news_id not in processor.breaking_news:
            raise HTTPException(status_code=404, detail="Breaking news item not found")
        
        item = processor.breaking_news[news_id]
    
    combined = item.breaking_score * item.urgency_score * item.importance_score
    
    notif_response = NewsNotificationResponse(
        id=item.notification.id,
        title=item.notification.title,
        description=item.notification.description,
        pubDate=item.notification.pubDate.isoformat(),
        link=item.notification.link,
        guid=item.notification.guid
    )
    
    return BreakingNewsResponse(
        notification=notif_response,
        is_breaking=item.is_breaking,
        breaking_score=round(item.breaking_score, 3),
        urgency_score=round(item.urgency_score, 3),
        importance_score=round(item.importance_score, 3),
        combined_score=round(combined, 3),
        category=item.category,
        subcategories={k: round(v, 3) for k, v in item.subcategories.items()},
        detected_at=item.detected_at.isoformat(),
        expires_at=item.expires_at.isoformat()
    )


@app.post("/classify", response_model=BreakingNewsResponse, tags=["Classification"])
async def classify_text(
    title: str = Query(description="News title"),
    description: str = Query(default="", description="News description")
):
    """
    Classify arbitrary text using the zero-shot model
    
    Useful for testing the classification on custom input.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    # Create temporary notification
    notification = NewsNotification(
        id="temp_" + str(hash(title + description)),
        title=title,
        description=description,
        pubDate=datetime.now(),
        link="",
        guid=""
    )
    
    # Classify
    result = classifier.classify_news(notification)
    combined = result.breaking_score * result.urgency_score * result.importance_score
    
    notif_response = NewsNotificationResponse(
        id=result.notification.id,
        title=result.notification.title,
        description=result.notification.description,
        pubDate=result.notification.pubDate.isoformat(),
        link=result.notification.link,
        guid=result.notification.guid
    )
    
    return BreakingNewsResponse(
        notification=notif_response,
        is_breaking=result.is_breaking,
        breaking_score=round(result.breaking_score, 3),
        urgency_score=round(result.urgency_score, 3),
        importance_score=round(result.importance_score, 3),
        combined_score=round(combined, 3),
        category=result.category,
        subcategories={k: round(v, 3) for k, v in result.subcategories.items()},
        detected_at=result.detected_at.isoformat(),
        expires_at=result.expires_at.isoformat()
    )


@app.get("/stats", response_model=SystemStats, tags=["System"])
async def get_statistics():
    """Get comprehensive statistics including ML model performance"""
    if not processor or not classifier:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = processor.get_statistics()
    uptime = (datetime.now() - service_start_time).total_seconds()
    
    return SystemStats(
        processed_count=stats['processed_count'],
        breaking_count=stats['breaking_count'],
        active_breaking_count=stats['active_breaking_count'],
        detection_rate=round(stats['detection_rate'], 4),
        avg_breaking_score=round(stats['avg_breaking_score'], 3),
        avg_urgency_score=round(stats['avg_urgency_score'], 3),
        avg_importance_score=round(stats['avg_importance_score'], 3),
        category_distribution=stats['category_distribution'],
        uptime_seconds=round(uptime, 2),
        model_info={
            "name": current_model_name,
            "type": "zero-shot-classification",
            "device": classifier.device,
            "threshold": str(classifier.breaking_threshold),
            "cache_size": str(len(classifier.cache))
        }
    )


@app.get("/model/labels", tags=["Model"])
async def get_model_labels():
    """Get the labels used by the zero-shot classifier"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not initialized")
    
    return {
        "breaking_labels": classifier.breaking_labels,
        "urgency_labels": classifier.urgency_labels,
        "importance_labels": classifier.importance_labels,
        "category_labels": classifier.category_labels
    }


@app.post("/model/update", tags=["Model"])
async def update_model(config: ModelConfig):
    """
    Update model configuration (requires restart)
    
    Note: Changing the model will reinitialize the classifier.
    """
    global classifier, processor, current_model_name
    
    if not classifier:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    old_config = {
        "model_name": current_model_name,
        "threshold": classifier.breaking_threshold,
        "expiry_hours": classifier.expiry_duration.total_seconds() / 3600
    }
    
    # Update configuration
    if config.model_name != current_model_name:
        logger.info(f"Switching model from {current_model_name} to {config.model_name}")
        current_model_name = config.model_name
        
        # Reinitialize classifier with new model
        classifier = ZeroShotBreakingNewsClassifier(
            model_name=config.model_name,
            breaking_threshold=config.breaking_threshold,
            expiry_hours=config.expiry_hours
        )
        
        # Update processor with new classifier
        processor.classifier = classifier
    else:
        # Just update parameters
        classifier.breaking_threshold = config.breaking_threshold
        classifier.expiry_duration = timedelta(hours=config.expiry_hours)
    
    return {
        "status": "updated",
        "old_config": old_config,
        "new_config": {
            "model_name": config.model_name,
            "threshold": config.breaking_threshold,
            "expiry_hours": config.expiry_hours
        }
    }


@app.post("/stream/control", tags=["Stream Control"])
async def control_stream(config: StreamConfig, background_tasks: BackgroundTasks):
    """Control the news stream with intelligent batching"""
    global simulator, stream_task
    
    if not simulator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Cancel existing stream
    if stream_task and not stream_task.done():
        stream_task.cancel()
        await asyncio.sleep(0.1)
    
    # Configure new stream
    simulator.speed_multiplier = config.speed_multiplier
    
    # Parse dates
    if config.start_date:
        start_date = datetime.fromisoformat(config.start_date.replace('Z', '+00:00'))
    else:
        start_date = datetime(2022, 3, 7, 0, 0, 0)
    
    if config.end_date:
        end_date = datetime.fromisoformat(config.end_date.replace('Z', '+00:00'))
    else:
        end_date = start_date + timedelta(days=7)
    
    # Start new stream with batching
    stream_task = asyncio.create_task(
        simulator.start_stream(start_date, end_date, batch_size=config.batch_size)
    )
    
    return {
        "status": "stream_restarted",
        "speed_multiplier": config.speed_multiplier,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "batch_size": config.batch_size
    }


@app.get("/stream/sse", tags=["Stream"])
async def stream_breaking_news():
    """
    Server-Sent Events for real-time ML-classified breaking news
    
    Streams breaking news with full classification scores as detected.
    """
    if not processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    async def event_generator():
        """Generate SSE events with ML classification data"""
        last_count = processor.breaking_count
        
        while True:
            current_count = processor.breaking_count
            
            if current_count > last_count:
                # Get the latest breaking news
                active = processor.get_active_breaking_news()
                if active:
                    latest = active[0]
                    combined = latest.breaking_score * latest.urgency_score * latest.importance_score
                    
                    event_data = {
                        "type": "breaking_news",
                        "data": {
                            "id": latest.notification.id,
                            "title": latest.notification.title,
                            "scores": {
                                "breaking": round(latest.breaking_score, 3),
                                "urgency": round(latest.urgency_score, 3),
                                "importance": round(latest.importance_score, 3),
                                "combined": round(combined, 3)
                            },
                            "category": latest.category,
                            "detected_at": latest.detected_at.isoformat()
                        }
                    }
                    
                    yield f"data: {json.dumps(event_data)}\n\n"
                    last_count = current_count
            
            # Heartbeat
            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/config", tags=["System"])
async def get_configuration():
    """Get current system configuration including ML model details"""
    if not classifier or not simulator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "classifier": {
            "model_name": current_model_name,
            "device": classifier.device,
            "breaking_threshold": classifier.breaking_threshold,
            "expiry_hours": classifier.expiry_duration.total_seconds() / 3600,
            "cache_size": len(classifier.cache)
        },
        "stream": {
            "speed_multiplier": simulator.speed_multiplier,
            "is_running": simulator.running
        },
        "labels": {
            "breaking": len(classifier.breaking_labels),
            "urgency": len(classifier.urgency_labels),
            "importance": len(classifier.importance_labels),
            "categories": len(classifier.category_labels)
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
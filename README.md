# Zero-Shot Breaking News Detection System

## Overview

A modern, ML-powered breaking news detection system using **transformer-based zero-shot classification**. This approach requires no training data and uses state-of-the-art language models to intelligently classify news across multiple dimensions.

## Why Zero-Shot Classification?

Traditional rule-based systems suffer from:
- Rigid keyword matching
- Manual threshold tuning  
- Poor generalization
- Language/domain limitations

**Zero-shot classification advantages:**
- No training data required
- Nuanced understanding of context
- Multi-dimensional scoring
- Language-agnostic capabilities
- Continuous improvement with better models

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   News Stream   │────▶│   Transformer    │────▶│  Multi-Dimensional  │
│   (CSV/API)     │     │   Model (BART)   │     │   Classification    │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                            │
                                ┌───────────────────────────┼───────────────────────────┐
                                ▼                           ▼                           ▼
                        ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
                        │   Breaking   │          │   Urgency    │          │  Importance  │
                        │   Detection  │          │   Analysis   │          │   Scoring    │
                        └──────────────┘          └──────────────┘          └──────────────┘
                                                            │
                                                            ▼
                                                    ┌──────────────┐
                                                    │   FastAPI    │
                                                    │   REST API   │
                                                    └──────────────┘
```

## Classification Methodology

### Multi-Dimensional Analysis

The system classifies news across **four independent dimensions**:

#### 1. Breaking News Classification
Determines if news qualifies as "breaking" using labels:
- `breaking news`
- `urgent development`
- `major event`
- `routine news`
- `minor update`
- `background information`

#### 2. Urgency Assessment
Evaluates time-sensitivity:
- `extremely urgent` (1.0)
- `very urgent` (0.8)
- `urgent` (0.6)
- `moderately urgent` (0.4)
- `not urgent` (0.2)

#### 3. Importance Scoring
Assesses significance/impact:
- `globally significant` (1.0)
- `nationally significant` (0.8)
- `regionally significant` (0.6)
- `locally significant` (0.4)
- `minor significance` (0.2)

#### 4. Category Classification
Multi-label categorization:
- War and conflict
- Natural disaster
- Political crisis
- Economic crisis
- Public health emergency
- Technology breakthrough
- Sports, Entertainment, etc.

### Combined Scoring

**Final Score = Breaking × Urgency × Importance**

This multiplicative approach ensures all dimensions must score high for maximum breaking news priority.

## Installation

### Prerequisites
```bash
# Python 3.8+ required
python3 --version
```

### Install Dependencies
```bash
pip install --break-system-packages \
    transformers \
    torch \
    fastapi \
    uvicorn \
    pydantic \
    sentencepiece \
    protobuf
```

### GPU Support (Optional)
```bash
# For CUDA (NVIDIA GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2)
pip install torch  # MPS backend included
```

## Usage

### 1. Run Standalone Classifier
```bash
python3 bn.py
```

### 2. Start API Server
```bash
python3 api_server.py
```
API available at: `http://localhost:8000`

### 3. Quick Test
```bash
# Test classification on custom text
curl -X POST "http://localhost:8000/classify" \
  -G --data-urlencode "title=Massive earthquake hits major city" \
  --data-urlencode "description=7.8 magnitude earthquake devastates capital, thousands feared trapped"
```

### 4. Required Data File
Ensure `bbc_news.csv` is in the same directory as the scripts.


## API Endpoints

### Core Endpoints

#### `GET /breaking`
Returns ML-classified breaking news with multi-dimensional scores.

**Query Parameters:**
- `limit`: Max items (1-50)
- `min_score`: Minimum breaking score filter (0-1)
- `category`: Filter by category
- `sort_by`: Sort by `combined`, `breaking`, `urgency`, or `importance`

#### `POST /classify`
Classify arbitrary text in real-time.

**Parameters:**
- `title`: News headline
- `description`: News body (optional)

**Response Example:**
```json
{
  "is_breaking": true,
  "breaking_score": 0.295,
  "urgency_score": 0.4,
  "importance_score": 0.4,
  "combined_score": 0.047,
  "category": "natural disaster",
  "subcategories": {
    "natural disaster": 0.78,
    "political crisis": 0.12,
    "general news": 0.10
  }
}
```

Note: Actual scores may be significantly lower (0.2-0.4 range).
Default threshold of 0.7 may be too high. Consider using 0.3-0.5.

#### `GET /stats`
System statistics with ML model performance metrics.

#### `GET /model/labels`
View all classification labels used by the model.

#### `POST /model/update`
Update model configuration (model name, threshold, etc.).

## Model Options

### Recommended Models

1. **BART-large-MNLI** (Default)
   - Model: `facebook/bart-large-mnli`
   - Size: ~400M parameters
   - Performance: Excellent
   - Speed: Moderate

2. **DeBERTa-v3-MNLI**
   - Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
   - Size: ~180M parameters
   - Performance: State-of-the-art
   - Speed: Good

3. **DistilBART-MNLI**
   - Model: `valhalla/distilbart-mnli-12-1`
   - Size: ~140M parameters
   - Performance: Good
   - Speed: Fast

### Switching Models
```bash
curl -X POST "http://localhost:8000/model/update" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "breaking_threshold": 0.75,
    "expiry_hours": 4
  }'
```

## Design Decisions

### 1. **Zero-Shot Over Fine-Tuning**
- No labeled data required
- Immediate deployment
- Leverages pre-trained knowledge
- Easy to update labels/categories

### 2. **Multi-Dimensional Scoring**
- Separates breaking/urgency/importance
- More nuanced prioritization
- Better explainability
- Flexible filtering

### 3. **Hypothesis Templates**
- Custom templates for each dimension
- Better prompt engineering
- Improved classification accuracy

### 4. **Intelligent Batching**
- Process multiple articles in parallel
- Optimize GPU utilization
- Maintain temporal ordering

### 5. **Caching Strategy**
- In-memory result caching
- Avoid reprocessing
- Fast response times

## Customization

### Adjusting Classification Labels

Edit labels in `bn.py`:

```python
# Customize breaking news labels
self.breaking_labels = [
    "breaking news",
    "developing story",
    "just in",
    "routine update"
]

# Customize categories
self.category_labels = [
    "your custom category",
    "another category"
]
```

### Tuning Thresholds

```python
# Adjust sensitivity
classifier = ZeroShotBreakingNewsClassifier(
    breaking_threshold=0.6,  # Lower = more breaking news
    expiry_hours=6           # Longer expiry window
)
```

## Production Deployment

### Scaling Considerations

1. **GPU Deployment**
   ```bash
   # Use GPU for 10x faster inference
   docker run --gpus all -p 8000:8000 breaking-news-ml
   ```

2. **Model Optimization**
   - Use ONNX for faster inference
   - Implement model quantization
   - Deploy with TorchServe

3. **Caching Layer**
   - Redis for distributed caching
   - Edge caching for API responses

4. **Load Balancing**
   - Multiple model instances
   - Round-robin distribution
   - Health checks

### Monitoring

Key metrics to track:
- Classification latency (p50, p95, p99)
- Model confidence distributions
- Category balance
- Cache hit rates
- GPU utilization

## Comparison: Rule-Based vs Zero-Shot

| Aspect | Rule-Based | Zero-Shot ML |
|--------|-----------|--------------|
| Setup Time | Minutes | Minutes |
| Context Understanding | Poor | Excellent |
| Language Support | English only | Multilingual |
| Maintenance | High | Low |
| Customization | Code changes | Label changes |
| Speed (CPU) | <1ms | 50-200ms |
| Speed (GPU) | <1ms | 10-30ms |
| Scalability | Excellent | Good |

## Future Enhancements

1. **Fine-Tuning**
   - Collect labeled breaking news data
   - Fine-tune on domain-specific content
   - Improve classification accuracy

2. **Ensemble Models**
   - Combine multiple models
   - Weighted voting mechanism
   - Improved robustness

3. **Active Learning**
   - User feedback integration
   - Continuous model improvement
   - Adaptive thresholds

4. **Explainability**
   - Attention visualization
   - Feature importance
   - Classification reasoning

## Troubleshooting

### Out of Memory
```bash
# Use smaller model
export MODEL_NAME="valhalla/distilbart-mnli-12-1"

# Or reduce batch size
python3 api_server.py --batch-size 1
```

### Slow Performance
```bash
# Enable GPU
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Use quantized model
pip install optimum
```

### Model Download Issues
```bash
# Pre-download model
python3 -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
```

### No Breaking News Detected
If `/breaking` returns empty results:

1. **Check threshold** - Default threshold is 0.5
```bash
   curl -X POST "http://localhost:8000/model/update" \
     -H "Content-Type: application/json" \
     -d '{"breaking_threshold": 0.5}'
```

2. **Test actual scores** - Use `/classify` endpoint to see real scores
```bash
   curl -X POST "http://localhost:8000/classify?title=Test&description=Test"
```
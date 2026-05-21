# FoodFinder - Free Restaurant Discovery App

A free, open-source alternative to RestaurantInsight that uses:
- **OpenStreetMap** (free) - Restaurant data
- **Sentence-Transformers** (free) - Text embeddings
- **ChromaDB** (free) - Vector storage
- **Ollama** (free) - Local LLM (optional)

## Installation

```bash
cd foodfinder
pip install -r requirements.txt
```

## Running the App

### Basic Version:
```bash
streamlit run app.py
```

### RAG Version (with AI):
```bash
# First install Ollama (https://ollama.com)
ollama pull llama3.2

streamlit run rag_app.py
```

## Features

### app.py
- Search restaurants by location
- Filter by cuisine, hours, etc.
- Interactive map view
- Browse all results

### rag_app.py
- Natural language queries
- AI-powered recommendations
- Context-aware search results
- Works with or without Ollama

## API Alternatives (Free)

| Feature | RestaurantInsight (Paid) | FoodFinder (Free) |
|---------|-------------------------|-------------------|
| Restaurant Data | Google Maps API | OpenStreetMap |
| Text Embeddings | OpenAI CLIP | Sentence-Transformers |
| Vector Storage | DeepLake | ChromaDB |
| LLM | OpenAI API | Ollama (local) |

## Usage

1. Enter coordinates or select a city
2. Click "Find Restaurants"
3. Search or browse results
4. (RAG version) Ask AI questions

## Troubleshooting

### Issue: App works with small radius but fails with larger radius (3000m+)

**Error symptoms:**
- Timeout errors (504 Gateway Timeout)
- Rate limiting errors (429 Too Many Requests)
- App hangs or crashes with larger search radius
- Empty results or partial data

**Root causes:**
1. **API Timeout**: Larger radius = more restaurants to fetch, exceeding the 120s timeout
2. **Rate Limiting**: Overpass API throttles larger queries that take too long
3. **Memory Issues**: Too many restaurants to process at once causes memory exhaustion

**Solutions implemented:**

1. **Result Limiting**: Capped maximum results to 500 for large radii (3000m+)
   ```python
   max_results = 500 if radius > 3000 else 200
   ```

2. **Better Rate Limit Handling**: Added exponential backoff with longer wait times
   ```python
   wait_time = min(60, 2 ** attempt * 5)
   time.sleep(wait_time)
   ```

3. **Batch Processing**: Index restaurants in batches of 100 to avoid memory issues
   ```python
   for i in range(0, len(texts), batch_size):
       # Process batch
   ```

4. **Progress Feedback**: Added warning message for large radius queries

**Tips:**
- Use radius 2000m or less for faster results
- If rate limited, wait 1-2 minutes before retrying
- Close and restart the app to clear ChromaDB cache if issues persist

## I will be uploading more optimization for this code as the main issue im facing is increasing the distance.



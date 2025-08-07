# Ecommerce Retrieval Search System - Complete Architecture Documentation

## Project Overview
This is a comprehensive e-commerce product retrieval and search system that leverages advanced Natural Language Processing (NLP) and vector similarity search techniques to provide accurate and efficient product recommendations based on user text queries.

**Live Demo:** https://ecommerce-retrieval-search.streamlit.app/
**Documentation:** https://1drv.ms/w/s!AnO5FdGErMSuh7VXK1wgpzwajPXOGw?e=fAh5Qh

## System Architecture

### 1. High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   User Query    │───▶│  Text Processing │───▶│  Embedding          │
│   (Text Input)  │    │  & Preprocessing │    │  Generation         │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Results       │◀───│  Image Retrieval │◀───│  Similarity Search  │
│   Display       │    │  & Formatting    │    │  (FAISS Index)      │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### 2. Data Flow Architecture
```
Raw E-commerce Data (Flipkart Dataset)
           │
           ▼
┌─────────────────────────────┐
│    Data Preprocessing       │
│  • HTML Tag Removal         │
│  • Text Normalization       │
│  • Stopword Removal         │
│  • Text Concatenation       │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│    Embedding Generation     │
│  • GTE-Base-EN-v1.5 Model   │
│  • 768-dimensional vectors  │
│  • Batch Processing         │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│    Vector Index Building    │
│  • FAISS IndexFlatL2        │
│  • L2 Distance Metric       │
│  • Efficient Storage        │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│    Query Processing         │
│  • Same Preprocessing       │
│  • Query Embedding          │
│  • Similarity Search        │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│    Result Retrieval         │
│  • Top-K Similar Products   │
│  • Image URL Mapping        │
│  • Product Information      │
└─────────────────────────────┘
```

## Core Components

### 1. Data Processing Pipeline
- **Input:** Raw e-commerce product data from Flipkart dataset
- **Processing Steps:**
  1. Data cleaning and validation
  2. Text concatenation (name + description + category + specifications + brand)
  3. HTML tag removal
  4. Text normalization (lowercase)
  5. Punctuation removal
  6. Stopword filtering
- **Output:** Clean, preprocessed text ready for embedding

### 2. Embedding System
- **Model:** Alibaba-NLP/gte-base-en-v1.5
- **Dimensions:** 768
- **Pooling Strategy:** CLS token pooling
- **Batch Processing:** 32 samples per batch
- **Hardware:** CUDA-enabled GPU support with CPU fallback

### 3. Vector Storage & Retrieval
- **Index Type:** FAISS IndexFlatL2
- **Distance Metric:** L2 (Euclidean) distance
- **Storage:** Persistent disk storage with lazy loading
- **Search:** Top-K similarity search with configurable K value

### 4. Web Interface
- **Framework:** Streamlit
- **Features:**
  - Text input for queries
  - Real-time search results
  - Image display with product names
  - Responsive layout

## Technical Specifications

### Dependencies
```python
pandas          # Data manipulation
matplotlib      # Visualization
pillow          # Image processing
seaborn         # Statistical visualization
nltk            # Natural language processing
transformers    # Hugging Face transformers
torch           # PyTorch framework
faiss-cpu       # Vector similarity search
streamlit       # Web application framework
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Minimum 8GB RAM
- 2GB+ disk space for embeddings and index

### Performance Metrics
- **Dataset Size:** 20,000 products
- **Embedding Dimensions:** 768
- **Index Size:** ~59MB
- **Search Time:** <100ms per query
- **Preprocessing Time:** ~2-3 hours for full dataset

## File Structure
```
Ecommerce-Retrieval-Search/
├── app.py                      # Main Streamlit application
├── embeddings.py               # Embedding generation script
├── similarity.py               # FAISS index and search functions
├── eda.ipynb                   # Exploratory Data Analysis
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── Data/
│   └── data/
│       └── flipkart_com-ecommerce_sample.csv  # Raw dataset
├── Imgs/                       # Result screenshots
├── preprocessed_text.csv       # Cleaned text data
├── id2img.csv                  # Product ID to image mapping
├── embeddings.npy              # Generated embeddings
├── id_list.npy                 # Product ID list
└── index                       # FAISS index file
```

## API Endpoints & Functions

### Core Functions

#### `preprocess(tokenizer, model, text)`
- **Purpose:** Clean and embed user query text
- **Input:** Raw text query
- **Output:** 768-dimensional embedding vector
- **Process:**
  1. Text cleaning (HTML, punctuation, stopwords)
  2. Tokenization
  3. Model inference
  4. CLS token pooling

#### `find_similar(query_embedding, k=6)`
- **Purpose:** Find most similar products
- **Input:** Query embedding vector
- **Output:** Distances and indices of top-K products
- **Method:** FAISS L2 distance search

#### `get_images_list(df, uniq_ids)`
- **Purpose:** Retrieve product images and names
- **Input:** Product unique IDs
- **Output:** Image URLs and product names
- **Source:** Pre-processed image mapping CSV

## Data Schema

### Raw Dataset Schema (flipkart_com-ecommerce_sample.csv)
```
- uniq_id: Unique product identifier
- crawl_timestamp: Data collection timestamp
- product_url: Product page URL
- product_name: Product title
- product_category_tree: Category hierarchy
- pid: Product ID
- retail_price: Original price
- discounted_price: Sale price
- image: Product image URLs (JSON array)
- is_FK_Advantage_product: Flipkart advantage flag
- description: Product description
- product_rating: Individual product rating
- overall_rating: Overall rating
- brand: Product brand
- product_specifications: Technical specifications (JSON)
```

### Processed Data Schema
```
preprocessed_text.csv:
- uniq_id: Product identifier
- text_col: Concatenated and cleaned text

id2img.csv:
- uniq_id: Product identifier  
- image_new: Processed image URLs (JSON array)
```

## Search Algorithm

### 1. Query Processing
```python
def process_query(query):
    # 1. Text cleaning
    query = query.lower()
    query = re.sub('<[^>]+>', '', query)  # Remove HTML
    query = re.sub(r'[^\w\s]', '', query)  # Remove punctuation
    
    # 2. Stopword removal
    stop_words = set(stopwords.words('english'))
    query = ' '.join([word for word in query.split() if word not in stop_words])
    
    # 3. Embedding generation
    embedding = get_embeddings(tokenizer, model, query)
    return embedding
```

### 2. Similarity Search
```python
def similarity_search(query_embedding, k=6):
    # Load FAISS index
    index = load_faiss_index("index")
    
    # Perform search
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    return distances[0], indices[0]
```

### 3. Result Ranking
- **Primary Metric:** L2 (Euclidean) distance
- **Ranking:** Ascending order (lower distance = higher similarity)
- **Top-K Selection:** Configurable K value (default: 6)

## Deployment Architecture

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Production Deployment (Streamlit Cloud)
- **Platform:** Streamlit Cloud
- **URL:** https://ecommerce-retrieval-search.streamlit.app/
- **Auto-deployment:** GitHub integration
- **Scaling:** Automatic based on usage

## Performance Optimization

### 1. Embedding Generation
- **Batch Processing:** 32 samples per batch
- **GPU Acceleration:** CUDA support for faster inference
- **Memory Management:** Efficient tensor operations

### 2. Vector Search
- **FAISS Optimization:** IndexFlatL2 for exact search
- **Memory Mapping:** Lazy loading of large index files
- **Caching:** Persistent index storage

### 3. Web Interface
- **Streamlit Caching:** @st.cache for expensive operations
- **Image Loading:** Asynchronous image requests
- **Response Time:** <100ms typical search time

## Security Considerations

### 1. Input Validation
- Text length limits
- HTML sanitization
- SQL injection prevention

### 2. Rate Limiting
- Query frequency limits
- Resource usage monitoring

### 3. Data Privacy
- No personal data storage
- Anonymous query processing

## Monitoring & Analytics

### 1. Performance Metrics
- Query response time
- Search accuracy
- User engagement

### 2. Error Handling
- Graceful failure handling
- Error logging
- User feedback collection

## Future Enhancements

### 1. Multimodal Search
- **Vision-Language Models:** CLIP integration
- **Image-to-Image Search:** Visual similarity
- **Image-to-Text Queries:** Cross-modal search

### 2. Advanced Features
- **Re-ranking:** Learning-to-rank algorithms
- **Personalization:** User preference learning
- **Multilingual Support:** Multiple language queries

### 3. Scalability Improvements
- **Distributed Search:** Multi-node FAISS
- **Real-time Updates:** Incremental indexing
- **Load Balancing:** Multi-instance deployment

## Testing Strategy

### 1. Unit Tests
- Individual function testing
- Edge case handling
- Performance benchmarking

### 2. Integration Tests
- End-to-end workflow testing
- API endpoint validation
- Database connectivity

### 3. User Acceptance Testing
- Search relevance evaluation
- User interface testing
- Performance testing

This architecture provides a robust, scalable foundation for e-commerce product search with advanced NLP capabilities and efficient vector-based retrieval.
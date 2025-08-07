# Ecommerce Retrieval Search - Complete Source Codes

This document contains all the source codes for the Ecommerce Retrieval Search system with detailed explanations.

## Table of Contents
1. [Main Application (app.py)](#main-application-apppy)
2. [Embedding Generation (embeddings.py)](#embedding-generation-embeddingspy)
3. [Similarity Search (similarity.py)](#similarity-search-similaritypy)
4. [Requirements (requirements.txt)](#requirements-requirementstxt)
5. [Configuration Files](#configuration-files)
6. [Data Processing Scripts](#data-processing-scripts)

---

## Main Application (app.py)

This is the main Streamlit web application that provides the user interface for the search system.

```python
import os
import re
import torch
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import streamlit as st
import nltk
from nltk.corpus import stopwords
from similarity import find_similar
from transformers import AutoTokenizer, AutoModel

# Set device for computation (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cls_pooling(model_output):
    """
    Perform CLS token pooling on the model output.
    
    Args:
        model_output: Output from the transformer model
        
    Returns:
        torch.Tensor: CLS token embeddings (first token of each sequence)
    """
    return model_output.last_hidden_state[:, 0]

def get_embeddings(tokenizer, model, text):
    """
    Generate embeddings for given text using the pre-trained model.
    
    Args:
        tokenizer: Pre-trained tokenizer
        model: Pre-trained transformer model
        text: Input text to embed
        
    Returns:
        numpy.ndarray: Text embeddings
    """
    # Tokenize the input text
    encoded_input = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    )
    
    # Move tensors to the appropriate device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Generate embeddings without gradient computation
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Pool the output and move to CPU
    return cls_pooling(model_output).detach().cpu().numpy()

def preprocess(tokenizer, model, text):
    """
    Preprocess text and generate embeddings.
    
    Args:
        tokenizer: Pre-trained tokenizer
        model: Pre-trained transformer model
        text: Raw input text
        
    Returns:
        numpy.ndarray: Preprocessed text embeddings
    """
    # Download NLTK stopwords if not already present
    nltk.download('stopwords')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub('<[^>]+>', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Generate embeddings for the preprocessed text
    emb = get_embeddings(tokenizer, model, text)
    return emb

def get_images_list(df, uniq_ids):
    """
    Retrieve image URLs and product names for given product IDs.
    
    Args:
        df: DataFrame containing product information
        uniq_ids: List of unique product identifiers
        
    Returns:
        tuple: (list of image URLs, list of product names)
    """
    images_list = []
    product_names = []
    
    for id in uniq_ids:
        # Get image URLs for the product
        ls = df[df['uniq_id'] == id]['image'].values[0]
        ls = eval(ls)  # Convert string representation of list to actual list
        images_list.append(ls)
        
        # Get product name
        ls = df[df['uniq_id'] == id]['product_name'].values[0]
        product_names.append(ls)
    
    return images_list, product_names

def main():
    """
    Main function to run the Streamlit application.
    """
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
    model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
    model = model.to(device)
    
    # Load the dataset
    curr_dir = os.getcwd()
    data_path = os.path.join(curr_dir, "Data", "data", "flipkart_com-ecommerce_sample.csv")
    df = pd.read_csv(data_path)
    
    # Load pre-computed product IDs
    ids = np.load("id_list.npy")
    
    # Streamlit UI components
    st.title("Retrieval Search")
    user_text = st.text_area('Enter you query below', value="A red skirt")
    generate_response_btn = st.button('Search for products!')
    
    # Process user query and display results
    if generate_response_btn and user_text is not None:
        # Preprocess user query and generate embeddings
        emb = preprocess(tokenizer, model, user_text)
        
        # Find similar products
        distances, idx = find_similar(emb)
        
        # Get unique IDs of similar products
        uniq_ids = [ids[i] for i in idx]
        
        # Retrieve image URLs and product names
        images_links, product_names = get_images_list(df, uniq_ids)
        
        # Display the results
        st.write("**Products**:")
        for image_list in images_links:
            # Display product name
            st.write(product_names[images_links.index(image_list)])
            
            # Create columns for images
            cols = st.columns(len(image_list), gap="medium")
            
            # Display images in columns
            for i, image_link in enumerate(image_list):
                with cols[i]:
                    try:
                        # Fetch and display image
                        response = requests.get(image_link)
                        image = Image.open(BytesIO(response.content))
                        st.image(image)
                    except Exception as e:
                        st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()
```

### Key Features of app.py:
- **Streamlit Interface**: User-friendly web interface for search queries
- **Real-time Processing**: Processes user queries in real-time
- **Image Display**: Fetches and displays product images from URLs
- **Error Handling**: Graceful handling of image loading errors
- **GPU Support**: Automatically uses GPU if available for faster processing

---

## Embedding Generation (embeddings.py)

This script generates embeddings for the entire product dataset and saves them for efficient retrieval.

```python
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# Load preprocessed text data
df = pd.read_csv('preprocessed_text.csv')

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-base-en-v1.5")
model = AutoModel.from_pretrained("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

def cls_pooling(model_output):
    """
    Perform CLS token pooling on the model output.
    
    Args:
        model_output: Output from the transformer model
        
    Returns:
        torch.Tensor: CLS token embeddings (first token of each sequence)
        Shape: (batch_size, hidden_size)
    """
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text):
    """
    Generate embeddings for given text batch.
    
    Args:
        text: List of text strings or single text string
        
    Returns:
        numpy.ndarray: Text embeddings
    """
    # Tokenize the input text
    encoded_input = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    )
    # Shape of encoded_input: 
    #   {'input_ids': (batch_size, sequence_length),
    #    'attention_mask': (batch_size, sequence_length)}
    
    # Move tensors to the appropriate device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Generate embeddings without gradient computation
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Shape of model_output.last_hidden_state: (batch_size, sequence_length, hidden_size)
    
    # Pool the output and move to CPU
    return cls_pooling(model_output).detach().cpu().numpy()

# Configuration for batch processing
batch_size = 32  # Adjust batch size based on GPU memory

# Initialize lists for embeddings and ID mappings
all_embeddings = []
id_list = []

print("Generating embeddings for all products...")
print(f"Total products: {len(df)}")
print(f"Batch size: {batch_size}")

# Generate embeddings in batches
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    end_idx = min(i + batch_size, len(df))
    
    # Extract text and ID batches
    texts_batch = df['text_col'].iloc[i:end_idx].tolist()
    ids_batch = df['uniq_id'].iloc[i:end_idx].tolist()
    
    # Skip empty batches
    if not texts_batch:
        continue
    
    # Ensure that IDs and texts are of the same length
    if len(texts_batch) != len(ids_batch):
        raise ValueError("Mismatch between number of texts and IDs in the batch.")
    
    # Generate embeddings for the batch
    embeddings_batch = get_embeddings(texts_batch)
    all_embeddings.append(embeddings_batch)
    id_list.extend(ids_batch)

# Convert list of embeddings to a numpy array
embeddings = np.vstack(all_embeddings)

print(f"Generated embeddings shape: {embeddings.shape}")
print(f"Total products processed: {len(id_list)}")

# Save embeddings to a file
np.save('embeddings.npy', embeddings)
print("Embeddings saved to 'embeddings.npy'")

# Save ID list to a file
np.save('id_list.npy', np.array(id_list))
print("ID list saved to 'id_list.npy'")

print("Embedding generation completed successfully!")
```

### Key Features of embeddings.py:
- **Batch Processing**: Processes data in configurable batch sizes for memory efficiency
- **GPU Acceleration**: Utilizes GPU when available for faster embedding generation
- **Progress Tracking**: Shows progress using tqdm for long-running operations
- **Memory Management**: Efficient handling of large datasets
- **Data Validation**: Ensures consistency between text and ID batches

---

## Similarity Search (similarity.py)

This module handles the FAISS index creation and similarity search functionality.

```python
import os
import faiss
import numpy as np

def build_and_save_faiss_index(embeddings, index_path):
    """
    Build FAISS index from embeddings and save to disk.
    
    Args:
        embeddings (numpy.ndarray): Array of embeddings
        index_path (str): Path to save the index
        
    Returns:
        faiss.Index: Built FAISS index
    """
    # Get embedding dimension
    d = embeddings.shape[1]
    print(f"Building FAISS index with dimension: {d}")
    
    # Create FAISS index using L2 (Euclidean) distance
    index = faiss.IndexFlatL2(d)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    # Save index to disk
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
    
    return index

def load_faiss_index(index_path):
    """
    Load FAISS index from disk or build if not exists.
    
    Args:
        index_path (str): Path to the index file
        
    Returns:
        faiss.Index: Loaded or built FAISS index
    """
    print(f"Attempting to load index from {index_path}")
    
    # Check if index file exists
    if os.path.exists(index_path):
        # Load existing index
        index = faiss.read_index(index_path)
        print(f"Index loaded from {index_path}")
        print(f"Index type: {type(index)}")
        return index
    else:
        # Build index if not found
        print(f"No index found at {index_path}. Building Index...")
        
        # Load embeddings
        embeddings = np.load("embeddings.npy")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Build and save index
        index = build_and_save_faiss_index(embeddings, index_path)
        print(f"Built index type: {type(index)}")
        
        return index

def find_similar(query_embedding, k=6):
    """
    Find the most similar products to a query embedding.
    
    Args:
        query_embedding (numpy.ndarray): Query embedding vector
        k (int): Number of similar products to retrieve
        
    Returns:
        tuple: (distances, indices) of the k most similar products
    """
    # Load FAISS index
    index = load_faiss_index("index")
    
    # Perform similarity search
    # Reshape query embedding to (1, embedding_dim) for batch search
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    
    # Return results for single query (remove batch dimension)
    return distances[0], indices[0]

# Additional utility functions for index management

def get_index_info(index_path="index"):
    """
    Get information about the FAISS index.
    
    Args:
        index_path (str): Path to the index file
        
    Returns:
        dict: Index information
    """
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return {
            "index_type": type(index).__name__,
            "dimension": index.d,
            "total_vectors": index.ntotal,
            "is_trained": index.is_trained,
            "metric_type": index.metric_type
        }
    else:
        return {"error": "Index file not found"}

def rebuild_index(embeddings_path="embeddings.npy", index_path="index"):
    """
    Rebuild the FAISS index from embeddings.
    
    Args:
        embeddings_path (str): Path to embeddings file
        index_path (str): Path to save the new index
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        embeddings = np.load(embeddings_path)
        build_and_save_faiss_index(embeddings, index_path)
        return True
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        return False
```

### Key Features of similarity.py:
- **FAISS Integration**: Efficient vector similarity search using Facebook's FAISS library
- **Lazy Loading**: Loads index only when needed
- **Index Management**: Functions for building, loading, and managing FAISS indices
- **L2 Distance**: Uses Euclidean distance for similarity measurement
- **Scalable Search**: Handles large-scale vector search efficiently

---

## Requirements (requirements.txt)

This file lists all the Python dependencies required for the project.

```txt
pandas
matplotlib
pillow
seaborn
nltk
transformers
torch
faiss-cpu
streamlit
```

### Dependency Descriptions:
- **pandas**: Data manipulation and analysis library
- **matplotlib**: Plotting and visualization library
- **pillow**: Python Imaging Library for image processing
- **seaborn**: Statistical data visualization library
- **nltk**: Natural Language Toolkit for text processing
- **transformers**: Hugging Face library for transformer models
- **torch**: PyTorch deep learning framework
- **faiss-cpu**: Facebook AI Similarity Search library (CPU version)
- **streamlit**: Web application framework for data science

---

## Configuration Files

### .gitignore
```gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/
```

---

## Data Processing Scripts

### Data Preprocessing Pipeline

The data preprocessing involves several steps that are typically performed in the EDA notebook:

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess_text_data(df):
    """
    Preprocess the e-commerce dataset for embedding generation.
    
    Args:
        df (pd.DataFrame): Raw e-commerce dataset
        
    Returns:
        pd.DataFrame: Preprocessed dataset with text_col
    """
    # Download required NLTK data
    nltk.download('stopwords')
    
    # Initialize stemmer
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        """Clean and preprocess individual text."""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub('<[^>]+>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    # Create a combined text column
    text_columns = ['product_name', 'description', 'product_category_tree', 'brand']
    
    # Fill NaN values
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Combine all text fields
    df['combined_text'] = df[text_columns].apply(
        lambda x: ' '.join(x.astype(str)), axis=1
    )
    
    # Clean the combined text
    df['text_col'] = df['combined_text'].apply(clean_text)
    
    # Remove rows with empty text
    df = df[df['text_col'].str.len() > 0]
    
    # Select relevant columns
    processed_df = df[['uniq_id', 'text_col']].copy()
    
    return processed_df

def create_image_mapping(df):
    """
    Create mapping between product IDs and image URLs.
    
    Args:
        df (pd.DataFrame): Raw e-commerce dataset
        
    Returns:
        pd.DataFrame: Image mapping dataset
    """
    # Select relevant columns
    image_df = df[['uniq_id', 'image']].copy()
    
    # Remove rows with missing images
    image_df = image_df.dropna(subset=['image'])
    
    # Rename column for consistency
    image_df = image_df.rename(columns={'image': 'image_new'})
    
    return image_df

# Example usage:
if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv('Data/data/flipkart_com-ecommerce_sample.csv')
    
    # Preprocess text data
    processed_text = preprocess_text_data(df)
    processed_text.to_csv('preprocessed_text.csv', index=False)
    print(f"Saved {len(processed_text)} processed text records")
    
    # Create image mapping
    image_mapping = create_image_mapping(df)
    image_mapping.to_csv('id2img.csv', index=False)
    print(f"Saved {len(image_mapping)} image mappings")
```

---

## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/uday161616/Ecommerce-Retrieval-Search.git
cd Ecommerce-Retrieval-Search
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Required Data
- Ensure the Flipkart dataset is in `Data/data/flipkart_com-ecommerce_sample.csv`
- Run the preprocessing scripts to generate `preprocessed_text.csv` and `id2img.csv`

### Step 4: Generate Embeddings
```bash
python embeddings.py
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

---

## Usage Examples

### Basic Search Query
```python
# Example query processing
query = "red dress for women"
# This will be processed through:
# 1. Text cleaning and preprocessing
# 2. Embedding generation
# 3. Similarity search in FAISS index
# 4. Result retrieval and display
```

### Advanced Configuration
```python
# Modify search parameters
def custom_search(query, k=10, threshold=0.5):
    """Custom search with additional parameters."""
    emb = preprocess(tokenizer, model, query)
    distances, indices = find_similar(emb, k=k)
    
    # Filter by distance threshold
    valid_results = distances < threshold
    filtered_indices = indices[valid_results]
    filtered_distances = distances[valid_results]
    
    return filtered_distances, filtered_indices
```

This comprehensive source code documentation provides all the necessary files and explanations to understand and implement the Ecommerce Retrieval Search system.
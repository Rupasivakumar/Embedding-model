# Embedding Models: From Text to Meaningful Vector Representations

A small Python project that converts sentences into numerical vectors (embeddings) with a Sentence Transformer model and finds semantically similar sentences using cosine similarity.

## 1. Introduction
Computers process information numerically, but human language contains meaning, context, and relationships. An **embedding** converts text into a numerical representation called a **vector**, so machine-learning systems can compare and process the meaning of text.

```
Text → Embedding Model → Vector
```

## 2. What is an Embedding?
An embedding is a numerical representation of text that captures useful semantic information in a form a machine-learning model can process. A vector is simply an ordered list of numbers.

```
"I enjoy coding."
        ↓
Embedding Model
        ↓
[0.21, -0.45, 0.78, 0.12, ...]
```

## 3. Embedding Dimension
The embedding dimension is the number of numerical values in an embedding vector.

- Model: `all-MiniLM-L6-v2`
- Embedding dimension: **384**

Each sentence is converted into a vector of 384 numbers.

## 4. Sentence Transformers
A Sentence Transformer converts complete sentences into fixed-size embeddings:

```
Input Sentence → Tokenization → Transformer Encoder → Pooling → Embedding Vector
```

The Transformer processes the words in context, and pooling combines the token representations into one vector for the whole sentence.

## 5. Semantic Similarity
Semantic similarity compares sentences by meaning, not only by exact words. For example, "I enjoy coding." and "I like programming." have similar meanings, so their embeddings are expected to be close.

## 6. Cosine Similarity
Cosine similarity measures how closely two vectors are aligned. The program computes it between all sentence embeddings and displays the pairs whose similarity is above the chosen threshold. A higher value generally means the sentences are more similar in meaning.

## 7. Model Used
`all-MiniLM-L6-v2`, a Sentence Transformer model that produces **384-dimensional** sentence embeddings.

## 8. Applications of Embeddings
Semantic search, Retrieval-Augmented Generation (RAG), document similarity, question matching, recommendation systems, and clustering.

**RAG workflow**

```
Documents → Chunking → Embedding Model → Vectors → Vector Store

User Query → Embedding Model → Query Vector → Similarity Search → Relevant Context → LLM
```

## 9. Project Implementation
`app.py`:
1. Loads the Sentence Transformer model.
2. Defines multiple sentences.
3. Converts the sentences into embeddings.
4. Displays the embedding dimension.
5. Calculates cosine similarity.
6. Finds semantically similar sentence pairs.
7. Displays the similarity score.

## 10. Technologies Used
Python, Sentence Transformers, Scikit-learn, PyTorch

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```

## 11. Project Files
```
embedding model/
│
├── app.py
├── requirements.txt
├── README.md
└── output.pdf
```
- `app.py` – generates sentence embeddings and calculates cosine similarity.
- `requirements.txt` – Python dependencies.
- `output.pdf` – screenshot of the program output.

## 12. How to Run
```bash
# 1. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the program
python app.py

# Optional: change the similarity threshold (default 0.50)
python app.py --threshold 0.65
```
The model (~90 MB) is downloaded automatically on the first run, so an internet connection is needed once.

## 13. Key Takeaways
- An embedding converts text into a numerical vector representation.
- A vector's dimension is the number of numerical values it contains.
- Sentence Transformers convert complete sentences into fixed-size embeddings.
- Pooling combines token representations into a sentence-level vector.
- Semantically similar sentences generally have closer embeddings.
- Cosine similarity can be used to compare embedding vectors.
- Embeddings are an important component of semantic search and RAG systems.

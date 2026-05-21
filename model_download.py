from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading embedding model...")
SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model done")

print("Downloading reranker model...")
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Reranker model done")

print("\nAll models ready. You can now run: streamlit run app.py")
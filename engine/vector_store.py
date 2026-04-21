from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

TEXT_DIR = Path("data/text_data")
CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "documents"
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
MIN_CHUNK = 150
MAX_CHUNK = 1000


def chunk_text(text: str) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Merge đoạn quá ngắn vào đoạn kế tiếp
    merged = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < MIN_CHUNK:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = para
    if buffer:
        merged.append(buffer)

    # Split đoạn quá dài tại sentence boundary
    chunks = []
    for para in merged:
        if len(para) <= MAX_CHUNK:
            chunks.append(para)
        else:
            sentences = para.replace(". ", ".|").split("|")
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= MAX_CHUNK:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)

    return [c for c in chunks if c.strip()]


_collection = None

def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return _collection


def build_index(force: bool = False) -> chromadb.Collection:
    collection = get_collection()

    if collection.count() > 0 and not force:
        print(f"Index da ton tai: {collection.count()} chunks. Bo qua. Dung force=True de rebuild.")
        return collection

    # Chỉ lấy file .txt.clean, bỏ topics files
    files = [f for f in TEXT_DIR.glob("*.txt.clean") if "topics" not in f.name]

    all_ids, all_docs, all_metas = [], [], []

    for file in files:
        article_file = file.name.replace(".txt.clean", "")  # e.g. "S08_set1_a1"
        source_set = article_file[:3]     # "S08"
        text = file.read_text(encoding="latin-1").strip()

        # Dòng đầu tiên là tên article
        lines = text.split("\n")
        article_title = lines[0].strip() if lines else article_file

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_ids.append(f"{article_file}_chunk_{idx}")
            all_docs.append(chunk)
            all_metas.append({
                "article_file": article_file,
                "article_title": article_title,
                "source_set": source_set,
                "chunk_index": idx
            })

    # Insert theo batch
    batch_size = 100
    total = len(all_ids)
    for i in range(0, total, batch_size):
        collection.add(
            ids=all_ids[i:i + batch_size],
            documents=all_docs[i:i + batch_size],
            metadatas=all_metas[i:i + batch_size]
        )
        print(f"Indexed {min(i + batch_size, total)}/{total} chunks...")

    print(f"\nDone! {collection.count()} chunks tu {len(files)} files.")
    return collection


def search(query: str, top_k: int = 5) -> List[Dict]:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        output.append({
            "text": doc,
            "article_file": meta["article_file"],
            "article_title": meta.get("article_title", ""),
            "score": round(1 - dist, 4),
            "chunk_index": meta.get("chunk_index", 0)
        })

    return output


if __name__ == "__main__":
    build_index()
    print("\n--- Test search ---")
    results = search("When did Lincoln start his political career?", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['article_file']} | {r['text'][:100]}...")

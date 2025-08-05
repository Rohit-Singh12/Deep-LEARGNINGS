import faiss
import numpy as np
import pickle
from tqdm import tqdm
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

  
class LoadDocuments:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        file_path: str = None,
        sample_query: str = None,
        index_path: str = "faiss.index",
        metadata_path: str = "metadata.pkl",
    ):
        '''
        Initializes LoadDocumentsFAISS to load documents, generate embeddings,
        and store them in a FAISS index.

        Args:
            embedding_model_name (str): HuggingFace model name.
            file_path (str): Path to documents.
            sample_query (str): Optional sample query to test.
            index_path (str): File path to save FAISS index.
            metadata_path (str): File path to save metadata (text mappings).
        '''
        self.embedding_model_name = embedding_model_name
        self.file_path = file_path
        self.sample_query = sample_query
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.texts: List[str] = []
        self.index = None
        self.embedding_dim = 384  # Must match the embedding model's output
        self.embedding_model = self._get_embedding_model()

    def _get_embedding_model(self) -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            trust_remote_code=True,
            cache_folder="./hf_cache"
        )

    def _load_documents(self) -> List[str]:
        reader = SimpleDirectoryReader(input_files=[self.file_path])
        documents = reader.load_data()

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(documents)

        return [node.text for node in nodes]

    def _batch_iterate(self, items: List, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i: i + batch_size]

    def _generate_embeddings(
        self, texts: List[str], batch_size: int = 64
    ) -> np.ndarray:
        all_embeddings = []
        for batch in tqdm(
            self._batch_iterate(texts, batch_size),
            total=(len(texts) + batch_size - 1) // batch_size,
            desc="Generating embeddings",
        ):
            batch_embeddings = self.embedding_model.get_text_embedding_batch(batch)
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings).astype("float32")

    def _create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def initialize_index(self):
        print(f"Loading documents from: {self.file_path}")
        self.texts = self._load_documents()

        print("Generating embeddings...")
        embeddings = self._generate_embeddings(self.texts)

        print("Creating FAISS index...")
        self.index = self._create_index(embeddings)

        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.texts, f)

        if self.sample_query:
            print("\nSample query result:")
            print(self.query(self.sample_query))

        print("Indexing complete.")

    def query(self, query_text: str, top_k: int = 5) -> str:
        if self.index is None:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.texts = pickle.load(f)

        query_vector = self.embedding_model.get_query_embedding(query_text)
        query_vector = np.array(query_vector).reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]

        return "\n\n--\n\n".join(results)

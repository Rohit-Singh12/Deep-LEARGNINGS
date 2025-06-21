from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, HnswConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
import argparse

class LoadDocuments:
    def __init__(self, 
                 client_url, 
                 embedding_model_name, 
                 file_path,
                 sample_query=None,
                 collection_name="knowledge_base"):
        self.embeddings = []
        self.client_url = client_url
        self.embedding_model_name = embedding_model_name
        self.file_path = file_path
        self.sample_query = sample_query
        self.collection_name = collection_name
        self.client = self.create_client()

    def batch_iterate(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    def load_documents(self):
        # Load documents
        print(f"Loading documents from {self.file_path}")
        reader = SimpleDirectoryReader(input_files=[self.file_path])
        documents = reader.load_data()

        # Split documents into chunks
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(documents)
        texts = [node.text for node in nodes]
        print(f"Loaded {len(nodes)} nodes and nodes 0 {texts[0]}" )
        return texts

    def create_client(self):
        client = QdrantClient(url=self.client_url, timeout=10000)
        return client

    def create_index(self):
        #delete collection if it exists
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        # Create a Qdrant collection if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, 
                                            distance=Distance.COSINE))
        assert self.client.collection_exists(self.collection_name)
        
        return self.client, self.collection_name

    def get_embedding_model(self):
        return HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            trust_remote_code=True,
            cache_folder='./hf_cache'
        )
    
    def get_batch_embeddings(self, embedding_model, context):
        return embedding_model.get_text_embedding_batch(context)

    def generate_embeddings(self, embedding_model, nodes, batch_size=64):
        
        for batch_context in tqdm(self.batch_iterate(nodes, batch_size), total=len(nodes) // batch_size, desc="Generating embeddings"):
            batch_embeddings = self.get_batch_embeddings(embedding_model, batch_context)
            self.embeddings.extend(batch_embeddings)
        
    def add_embeddings_to_qdrant(self, embeddings, nodes, batch_size=64):
        if not self.client.collection_exists(self.collection_name):
            self.create_index()
        id_counter = 0
        for batch_context, batch_embedding in tqdm(zip(self.batch_iterate(nodes, batch_size), self.batch_iterate(embeddings, batch_size)), total=len(nodes) // batch_size, desc="Adding embeddings to Qdrant"):
            points = [PointStruct(id=id_counter + i, vector=embedding, payload={"text": context}) for i, (context, embedding) in enumerate(zip(batch_context, batch_embedding))]
            self.client.upsert(collection_name=self.collection_name, points=points)
            id_counter += len(batch_embedding)
        
        # Use HNSW if index count is more than 2000
        if len(embeddings) > 2000:
            self.client.update_collection(
                collection_name=self.collection_name,
                hnsw_config=HnswConfig(m=16, ef_construct=100)
            )

    def query_qdrant(self, embedding_model, query, top_k=5):
        print(f"Querying Qdrant with {query}")
        query_embedding = embedding_model.get_query_embedding(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        ).points
        dict_res = [dict(res) for res in results]
        context = []
        for data in dict_res:
            context.append(data['payload']['text'])
        return "\n\n--\n\n".join(context)


    def initialize_index(self):
        nodes = self.load_documents()
        print("Embedding model...")
        embedding_model = self.get_embedding_model()
        print("Generating embeddings...")
        self.generate_embeddings(embedding_model, nodes)
        print("Adding embeddings to Qdrant...")
        self.add_embeddings_to_qdrant(self.embeddings, nodes)
        if self.sample_query:
            self.query_qdrant(embedding_model, self.sample_query)

    def query(self, text):
        embedding_model = self.get_embedding_model()
        context = self.query_qdrant(embedding_model, text)
        return context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize the knowledge base index.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the document file to be loaded.")
    parser.add_argument("--collection_name", type=str, default="knowledge_base", help="Name of the Qdrant collection.")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="HuggingFace model name for embeddings.")
    parser.add_argument("--quadrant_url", type=str, default="http://localhost:6333", help="URL of the Qdrant server.")
    parser.add_argument("--sample_query", type=str, help="Sample query to test the index.")
    args = parser.parse_args()
    loader = LoadDocuments(args.quadrant_url,
                           args.embedding_model,
                           args.file_path,
                           args.sample_query,
                           args.collection_name)
    loader.initialize_index()

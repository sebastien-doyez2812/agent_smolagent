from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
import nest_asyncio
from llama_index.llms.ollama import Ollama
import os
from dotenv import load_dotenv


# For the hugging face Embedding:
# load_dotenv(".env")
# hf_token = os.getenv("HF_TOKEN")



PATH = "llama_index/RAG_folder"
reader = SimpleDirectoryReader(input_dir=PATH)

documents = reader.load_data()

db = chromadb.PersistentClient(path = "./my_chroma_db")
chroma_collection = db.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection= chroma_collection)

pipeline = IngestionPipeline(
    transformations= [
        SentenceSplitter(chunk_size= 25, 
                         chunk_overlap= 10),
        OllamaEmbedding(model_name="llama2", base_url= "http://localhost:11434")
        #HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    ],
    vector_store= vector_store
)

nodes = pipeline.run(document = documents)

#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = OllamaEmbedding(model_name="llama2", base_url= "http://localhost:11434")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)


nest_asyncio.apply() # Need for query engine
llm = Ollama(model="gemma")
query_engine = index.as_query_engine(
    llm= llm,
    response_mode = "tree_summarize"
)


response = query_engine.query("Comment sortir de la prison de monopoly?")
print(response)
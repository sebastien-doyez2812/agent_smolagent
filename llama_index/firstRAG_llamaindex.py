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
from PyPDF2 import PdfReader

# For the hugging face Embedding:
# load_dotenv(".env")
# hf_token = os.getenv("HF_TOKEN")



def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


PATH = "C:/Users/doyez/Documents/agent_smolagent/llama_index/RAG_folder/Une_Courte_Piece_de_Theatre.pdf"

# reader = SimpleDirectoryReader(input_dir=PATH)

# documents = reader.load_data()
pdf_text = extract_text_from_pdf(PATH)
documents = [Document(text = pdf_text)]
for doc in documents:
    print("Document extrait :", doc.text[:200])

db = chromadb.PersistentClient(path = "./my_chroma_db")
chroma_collection = db.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection= chroma_collection)

pipeline = IngestionPipeline(
    transformations= [
        SentenceSplitter(chunk_size= 50, 
                         chunk_overlap= 0),
        OllamaEmbedding(model_name="llama2", base_url= "http://localhost:11434")
        #HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    ],
    vector_store= vector_store
)

nodes = pipeline.run(document = documents)
print(f"{len(nodes)} noeuds insérés dans la base vectorielle.")
#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = OllamaEmbedding(model_name="llama2", base_url= "http://localhost:11434")
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)


nest_asyncio.apply() # Need for query engine
llm = Ollama(model="gemma", context_window= 100)
query_engine = index.as_query_engine(
    llm= llm,
    response_mode = "tree_summarize"
)


response = query_engine.query("Qui sont les personnages?")
print(response)


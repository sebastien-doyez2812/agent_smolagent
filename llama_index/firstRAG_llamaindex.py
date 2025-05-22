from llama_index.core import SimpleDirectoryReader

PATH = "llama_index/RAG_folder"
reader = SimpleDirectoryReader(input_dir=PATH)

document = reader.load_data()
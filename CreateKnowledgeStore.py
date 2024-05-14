
import os
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] =  os.getenv('OPENAI_API_KEY')
embeddingsModel=OpenAIEmbeddings()
chroma_db = Chroma(persist_directory=f"./WebSync-Vector-Store", embedding_function=OpenAIEmbeddings())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

def TestAPISQouta():
    try:
        embeddings = embeddingsModel.embed_documents(
                [
                    "Test"
                ]
            )
        return True
    except:
        return False


def num_tokens_from_string(string: str, encoding_name="text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def CountTOkensFromDocs(documents):
    tokens_Count = 0
    for document in documents:

        tokens_Count += num_tokens_from_string(document.page_content)
    return tokens_Count

def GetDocumentsFromURL(user_id,chatbot_id,urls):
    try:
        loader = WebBaseLoader(urls.split(','))
        documents = loader.load()
        for document in documents:
            document.metadata['user_id']=str(user_id)
            document.metadata['chatbot_id']=str(chatbot_id)
        return documents,CountTOkensFromDocs(documents)
    except Exception as e:
        raise Exception(str(e))
    return False,-1

def create_chatbot(documents):
    try:
        texts = text_splitter.split_documents(documents)
        chroma_db.add_documents(texts)
        return True
    except Exception as e:
        print(e)
        pass
    return False

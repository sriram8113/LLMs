#!pip install langchain
#!pip install openai
#!pip install PyPDF2
#!pip install faiss-cpu
#!pip install tiktoken
#!pip install unstructured
#!pip install chromadb

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
from constants import openai_key , serpapi_key

import os
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPAPI_API_KEY"] = serpapi_key  #for google search


# provide the path of  pdf file/files.
pdfreader = PdfReader('budget_speech.pdf')
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
#converting text into embedding and loading 
document_search = FAISS.from_texts(texts, embeddings)



chain = load_qa_chain(OpenAI(), chain_type="stuff")

#example query
query = "Ask a question from th eabove uploaded pdf"
docs = document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)


loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
data = loader.load()

index = VectorstoreIndexCreator().from_loaders([loader])
query = "Explain me about Attention is all you need"
index.query(query)
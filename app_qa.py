import os
import openai
import langchain
import pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain


load_dotenv()


## Lets Read the document
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('documents/')


def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs


documents=chunk_data(docs=doc)



embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
embeddings

vectors=embeddings.embed_query("How are you?")
len(vectors)

pinecone.init(
    api_key="923d5299-ab4c-4407-bfe6-7f439d9a9cb9",
    environment="gcp-starter"
)
index_name="langchainvector"

index=Pinecone.from_documents(doc,embeddings,index_name=index_name)

def retrieve_query(query,k=2):
    matching_results=index.similarity_search(query,k=k)
    return matching_results


#question and answer chain
llm=OpenAI(model_name="text-davinci-003",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response

our_query = "How much the agriculture target will be increased by how many crore?"
answer = retrieve_answers(our_query)
print(answer)




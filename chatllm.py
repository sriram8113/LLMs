import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,SystemMessage,AIMessage


chatllm=ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.6,model='gpt-3.5-turbo')


#system message is like saying chatbot to act as one
#human message is like asking question to that above chatbot
#AI message is like response from chatbot

chatllm([
SystemMessage(content="Yor are a comedian AI assitant"),
HumanMessage(content="Please provide some comedy punchlines on AI")
])
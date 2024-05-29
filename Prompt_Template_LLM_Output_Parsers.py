import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema import HumanMessage,SystemMessage,AIMessage


chatllm=ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.6,model='gpt-3.5-turbo')


#changing the output format of the chatbot or output from AI message .
class Commaseperatedoutput(BaseOutputParser):
    def parse(self,text:str):
        return text.strip().split(",")
    

#systemtemplate
template="Your are a helpful assistant. When the user given any input , you should generate 5 words synonyms in a comma seperated list"
#human template
human_template="{text}"
chatprompt=ChatPromptTemplate.from_messages([
    ("system",template),
    ("human",human_template)
])

chain=chatprompt|chatllm|Commaseperatedoutput()
chain.invoke({"text":"intelligent"})
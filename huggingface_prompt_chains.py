from langchain.llms import OpenAI
from langchain.llms import LLM
from huggingface_prompt_chains import HuggingFaceHub


import os

#temperature 0 means not taking any risk and 1 mean taking risks like giving may be wrong info 
#value more towards one the more creartive the model is going to be
llm = OpenAI(openai_api_key = os.environ["OPENAI_API_KEY"], temperature=0.6)
text = 'what is capital of india'
llm.predict(text)


#using open source models from huggingface

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ''
llm_huggingface = HuggingFaceHub(repo_id ="google/flan-t5-large", model_kwargs = {"temperature":0.6, "max_length":64})
output = llm_huggingface.predict('can you tell me capital of russia')
print(output)

#using prompt templates
from langchain.prompts import PromptTemplate

prompt_template=PromptTemplate(input_variables=['country'],template="Tell me the capital of this {country}")
prompt_template.format(country="India")

from langchain.chains import LLMChain
chain=LLMChain(llm=llm,prompt=prompt_template)
print(chain.run("India"))

#using mutiple chains 

capital_template=PromptTemplate(input_variables=['country'],template="Please tell me the capital of the {country}")

capital_chain=LLMChain(llm=llm,prompt=capital_template)

famous_template=PromptTemplate(input_variables=['capital'],template="Suggest me some amazing places to visit in {capital}")

famous_chain=LLMChain(llm=llm,prompt=famous_template)

from langchain.chains import SimpleSequentialChain
chain=SimpleSequentialChain(chains=[capital_chain,famous_chain])

chain.run("India")


#using sequential chain
#here we use output key

capital_template=PromptTemplate(input_variables=['country'],template="Please tell me the capital of the {country}")

capital_chain=LLMChain(llm=llm,prompt=capital_template,output_key="capital")

famous_template=PromptTemplate(input_variables=['capital'],template="Suggest me some amazing places to visit in {capital}")

famous_chain=LLMChain(llm=llm,prompt=famous_template,output_key="places")


from langchain.chains import SequentialChain
chain=SequentialChain(chains=[capital_chain,famous_chain],
                            input_variables=['country'],
                            output_variables=['capital',"places"])

chain({'country':"India"})



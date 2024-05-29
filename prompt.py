import os
from constants import openai_key
from langchain import PromptTemplate, FewShotPromptTemplate

from langchain.llms import OpenAI
from langchain.chains import LLMChain


os.enivron['OPENAI_API_KEY'] = openai_key


demo_template  = '''I want you to act as a acting financial advisor for people. In easy way explain the basics of{financial_concept}'''

prompt = PromptTemplate(
    input_variables = ['financial_concept'],
    template = demo_template
)

prompt.format(financial_concept = 'stock market')


llm = OpenAI(temperature=0.7)
chain1 = LLMChain(llm=llm, prompt = prompt, verbose=True)



examples = [
    {'word' :'happy', 'antonym' : 'sad'},
    {'word' :'tall', 'antonym' : 'short'},
]

example_template = """Word: {word}
Antonym: {antonym}
"""

prompt = PromptTemplate(
    input_variables = ['word', 'antonym'],
    template = example_template
)

few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    exam_prompt=example_template,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Give the antonym of every input\n",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Word: {input}\nAntonym: ",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["input"],
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n",
)

few_shot_prompt.format(input = 'big')

chain=LLMChain(llm=llm,prompt=few_shot_prompt)
chain({'input':"big"})
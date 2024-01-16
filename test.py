from dotenv import load_dotenv
from langchain import hub
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
import validators
import faiss
import pickle

load_dotenv()

# # Prepare template for prompt
# template = """
# Based on the user's query and base URL, generate a url to search results.
# You do not need to use single words from the query if it's long. Extract the main idea from the query and use that to generate the URL.
# Only output one URL.
#
# Example:
# user's query: transpose table
# base URL: https://pandas.pydata.org/docs/search.html?q=
# https://pandas.pydata.org/docs/search.html?q=transpose+table#
#
# user's query: {query}
# base URL: {base_url}
# """
#
# prompt = PromptTemplate(
#     template=template,
#     input_variables=['query', 'base_url']
# )
#
# model_input = prompt.format_prompt(
#     query='how can I group by two columns',
#     base_url='https://pandas.pydata.org/docs/search.html?q='
# )
#
# model = OpenAI(temperature=0)
# generated_url = model.invoke(model_input.to_string())
# print(generated_url)
#
# # check if output is a valid URL
# if not validators.url(generated_url):
#     raise ValueError('Output is not a valid URL!')

# Get the search results
# Load HTML
generated_url = "https://pandas.pydata.org/docs/search.html?q=group+by+two+columns#"  # for testing

options = ChromeOptions()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(2)
driver.get(generated_url)
html = driver.page_source
driver.quit()

# Extract the search results
soup = BeautifulSoup(html, 'html.parser')

urls = []
count = 0
max_links = 3

for ultag in soup.find_all('ul', {'class': 'search'}):
    for litag in ultag.find_all('li'):
        a_tag = litag.find('a')
        # Check if 'a' tag is found and print its 'href' attribute
        if a_tag and a_tag.has_attr('href'):
            urls.append(f"https://pandas.pydata.org/docs/{a_tag.attrs['href']}")
            count += 1
            if count == max_links:
                break

loader = AsyncChromiumLoader(urls)
data = loader.load()
html2text = Html2TextTransformer()
data_transformed = html2text.transform_documents(data)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False
# )
documents = text_splitter.split_documents(data_transformed)

# markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# documents = []
# for doc in data_transformed:
#     documents.extend(markdown_splitter.split_text(doc[0].page_content))

# Embedding
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
vectordb.save_local("faiss_store")

# Load the vector store
vectordb = FAISS.load_local("faiss_store", OpenAIEmbeddings())

retriever = vectordb.as_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)

template = """You are an outstanding assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise. 
Please include the coding example only in Python as well. 
Create a new example if the example you find in the context is not enough. Please provide what it returns as well.
Include source at the end.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("How can I group by two columns?")
print(result)





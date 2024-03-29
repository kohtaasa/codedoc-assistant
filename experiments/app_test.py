import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from googlesearch import search
from urllib.parse import urlparse
import validators

with st.sidebar:
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
        "2. Select a package/framework📄\n"
        "3. Ask a question about the document💬"
    )
    st.markdown("---")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        key="chatbot_api_key",
        type="password",
        placeholder="Paste your OpenAI API key here"
    )

st.title("💬 Code Knowledge Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def get_domain(url: str) -> str:
    parsed_url = urlparse(url)
    return parsed_url.netloc


def extract_content(query: str, base_url: str) -> list:
    # Prepare template for prompt
    template = """
    Based on the user's query and base URL, generate an url to search results.
    You do not need to use single words from the query if it's long. Extract the main idea from the query and use that to generate the URL.
    Only output one URL.

    Example:
    user's query: transpose table
    base URL: https://pandas.pydata.org/docs/search.html?q=
    https://pandas.pydata.org/docs/search.html?q=transpose+table
    
    user's query: how to rename columns
    base URL: https://pandas.pydata.org/docs/search.html?q=
    https://pandas.pydata.org/docs/search.html?q=rename

    user's query: {query}
    base URL: {base_url}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['query', 'base_url']
    )

    model_input = prompt.format_prompt(
        query=query,
        base_url=base_url
    )

    model = OpenAI(temperature=0, openai_api_key=openai_api_key)
    generated_url = model.invoke(model_input.to_string()).strip()
    # check if output is a valid URL
    if not validators.url(generated_url):
        raise ValueError(f'Output is not a valid URL! {generated_url}')

    # Get the search results
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
    domain = get_domain(base_url)

    for ultag in soup.find_all('ul', {'class': 'search'}):
        for litag in ultag.find_all('li'):
            a_tag = litag.find('a')
            # Check if 'a' tag is found and append its 'href' attribute
            if a_tag and a_tag.has_attr('href'):
                urls.append(f"{domain}{a_tag.attrs['href']}")
                count += 1
                if count == max_links:
                    break

    google_res = search(query, num_results=3, lang="en")

    for i in google_res:
        if i not in urls and domain in i:
            urls.append(i)
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
    documents = text_splitter.split_documents(data_transformed)

    return documents


def save_documents_in_vectorstore(documents: list) -> FAISS:
    # Save documents in vectorstore
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vectordb


prompt = st.chat_input()
if prompt:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    documents = extract_content(prompt, "https://pandas.pydata.org/docs/search.html?q=")
    vectordb = save_documents_in_vectorstore(documents)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, openai_api_key=openai_api_key)
    system_prompt = """You are an outstanding assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise. 
    Please include the coding example only in Python as well. 
    Create a new example if the example you find in the context is not enough. Please provide what it returns as well.
    Make sure include source, or sources if you refer to multiple contexts, at the end.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    qa_prompt = ChatPromptTemplate.from_template(system_prompt)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | qa_prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)


def reset_conversation():
    st.session_state.messages = None
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


st.button('Reset Chat', on_click=reset_conversation)

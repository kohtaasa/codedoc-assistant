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
from playwright.async_api import async_playwright
from googlesearch import search
from urllib.parse import urlparse

import streamlit as st
import asyncio
import validators


def setup_sidebar():
    """
    Setup the streamlit sidebar
    """
    with st.sidebar:
        st.image("data/app_logo.png")
        st.markdown(
            "## How to use\n"
            "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n" 
            "2. Select a library/package/framework📄\n"
            "3. Ask a question about the library💬"
        )
        st.markdown("---")
        st.markdown("Which library/package/framework do you want to search?")
        st.session_state.option = st.selectbox(
            "Which library/package/framework do you want to search?",
            ("Pandas", "Numpy", "Matplotlib", "Scikit-learn", "PyTorch"),
            index=None,
            placeholder="Select a library to search",
            label_visibility="collapsed",
        )
        st.session_state.api_key = st.text_input(
            "OpenAI API Key",
            key="chatbot_api_key",
            type="password",
            placeholder="Paste your OpenAI API key here"
        )


def get_domain(url: str) -> str:
    """
    Get the domain from a URL
    Use this to filter out Google search results from other domains
    :param url: URL
    :return: domain
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc


async def get_html(url: str) -> str:
    """
    Get HTML from a URL using Playwright
    :param url:
    :return: html content
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)
        html = await page.content()
        await browser.close()
    return html


def extract_content(query: str, base_url: str, option: str) -> list:
    """
    Extract content from a website based on a query
    :param query:
    :param base_url:
    :param option: library/package/framework selected
    :return: list of documents (Search results content)
    """
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
        https://pandas.pydata.org/docs/search.html?q=rename+columns

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

    model = OpenAI(temperature=0, openai_api_key=st.session_state.api_key)
    generated_url = model.invoke(model_input.to_string()).strip()
    # check if output is a valid URL
    # if not validators.url(generated_url):
    #     raise ValueError(f'Output is not a valid URL! {generated_url}')

    # Get the search results
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    html = loop.run_until_complete(get_html(generated_url))

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

    google_res = search(f"{query} {option}", num_results=3, lang="en")

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


def save_documents_in_vectorstore(documents: list, openai_api_key: str) -> FAISS:
    """
    Save documents in vectorstore to be used for similarity search
    :param openai_api_key:
    :param documents:
    :return: vectorstore instance
    """
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vectordb


def reset_conversation():
    st.session_state.messages = None
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


# Base URLs for different packages/frameworks
base_urls = {"Pandas": "https://pandas.pydata.org/docs/search.html?q=",
             "Numpy": "https://numpy.org/doc/stable/search.html?q=",
             "Matplotlib": "https://matplotlib.org/stable/search.html?q=",
             "Scikit-learn": "https://scikit-learn.org/stable/search.html?q=",
             "PyTorch": "https://pytorch.org/docs/stable/search.html?q="}

### Main Execution
setup_sidebar()
if st.session_state.option and st.session_state.option in base_urls:
    st.image(f"data/{st.session_state.option}.png", width=400)
st.title("📖 CodeDoc Assistant")
st.caption("Chatbot that helps you find answers to popular python libraries/packages/frameworks "
           "from their official documentations!")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask a question!")
if prompt:
    if not st.session_state.api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not st.session_state.option:
        st.info("Please select a library/package/framework to continue.")
        st.stop()

    client = OpenAI(api_key=st.session_state.api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    base_url = base_urls[st.session_state.option]
    with st.status("Searching..."):
        st.write("Fetching data from official documentation...")
        documents = extract_content(prompt, base_url, st.session_state.option)
        st.write("Saving data in vectorstore...")
        vectordb = save_documents_in_vectorstore(documents, st.session_state.api_key)
        st.write("Generating response...")
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, openai_api_key=st.session_state.api_key)
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

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = rag_chain.stream(prompt)
        for chunk in assistant_response:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.button('Clear Chat', on_click=reset_conversation)

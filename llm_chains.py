from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory




def load_normal_chain():
    """
    Loads a normal conversational chain without document context
    """


    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=3
    )


    llm = OpenAI(temperature=0.0)


    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=None,
        memory=memory
    )


    return chain




def load_pdf_chat_chain(pdf_path):
    """
    Loads a conversational chain for chatting with a PDF
    """


    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()


    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)


    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Vector store
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings
    )


    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=3
    )


    llm = OpenAI(temperature=0.0)


    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


    return chain


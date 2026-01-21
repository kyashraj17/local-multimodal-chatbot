from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from operator import itemgetter
import chromadb
from utils import load_config
from prompt_templates import memory_prompt_template, pdf_chat_prompt

config = load_config()

def create_llm():
    return CTransformers(
        model=config["ctransformers"]["model_path"]["large"],
        model_type="mistral",
        config=config["ctransformers"]["model_config"]
    )

def create_embeddings():
    return HuggingFaceInstructEmbeddings(
        model_name=config["embeddings_path"]
    )

def load_vectordb(embeddings):
    client = chromadb.PersistentClient(
        path=config["chromadb"]["chromadb_path"]
    )

    return Chroma(
        client=client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings
    )

class chatChain:
    def __init__(self):
        llm = create_llm()
        prompt = PromptTemplate.from_template(memory_prompt_template)
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, user_input, chat_history):
        return self.chain.invoke({
            "human_input": user_input,
            "history": chat_history
        })["text"]

class pdfChatChain:
    def __init__(self):
        llm = create_llm()
        vectordb = load_vectordb(create_embeddings())
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )

    def run(self, user_input, chat_history):
        return self.chain.run(user_input)

def load_normal_chain():
    return chatChain()

def load_pdf_chat_chain():
    return pdfChatChain()


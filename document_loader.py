from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
# from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None
    memory = None

    def __init__(self):
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Initialize memory

        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helpful chatbot that answers questions based on the context provided.
            Be conversational and engaging.  Refer to previous parts of the conversation if relevant.
            If the context doesn't contain the answer, say that you don't know.
            Use your expertise to answer generic questions, refer to the document provided to you to answer questions related to the document.
            Answer in English and respond to the questions. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Chat History: {chat_history}
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = FAISS.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # Create a Conversational Retrieval Chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.model,
            retriever=self.retriever,
            memory=self.memory,
            get_chat_history=lambda h : h,
            return_source_documents=False,
        )



    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        result = self.chain({"question": query})
        return result["answer"]

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.memory.clear() # Clear the memory too!

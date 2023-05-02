from fastapi import FastAPI, Response,status,FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
os.environ["OPENAI_API_KEY"] = ""

class FAQ(BaseModel):
    topic: str

app = FastAPI()

@app.post("/api/pdf_file_upload")
async def uploadfile(file: UploadFile = File(),ask_question: str = Form()):
    content = await file.read()  # Read the file contents
    with BytesIO(content) as f:
        reader = PdfReader(f)

        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()

        docsearch = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        docs = docsearch.similarity_search(ask_question)

        answer = chain.run(input_documents=docs, question=ask_question)

        return answer
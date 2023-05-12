FROM python:3.9

RUN pip install --upgrade langchain openai -q
RUN pip install unstructured -q
RUN pip install unstructured[local-inference] -q
RUN pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pinecone-client
RUN pip install "unstructured[local-inference]"
RUN pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
RUN pip install layoutparser[layoutmodels,tesseract]
RUN pip install "unstructured[local-inference]"
RUN pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
RUN pip install layoutparser[layoutmodels,tesseract]
RUN apt-get update && \
    apt-get install -y libmagic-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr libreoffice pandoc
RUN pip install tiktoken -q
RUN pip install faiss-cpu
RUN pip -q install langchain huggingface_hub transformers sentence_transformers accelerate bitsandbytes

WORKDIR /app

COPY requirements.txt /app

# Upgrade pip and install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . /app


EXPOSE 8000

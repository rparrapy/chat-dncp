from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

loader = UnstructuredFileLoader("pbc.pdf", mode="elements")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
index = FAISS.from_documents(docs, embeddings)

query = "Cuál es el el plazo de vigencia de la garantía de fiel cumplimiento?"
docs = index.similarity_search(query)


endpoint_url = (
    "https://api-inference.huggingface.co/models/mrm8488/bert-multi-cased-finetuned-xquadv1"
)
hf = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    huggingfacehub_api_token="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"
)

chain = load_qa_chain(hf, chain_type="stuff")
chain.run(input_documents=docs, question=query)

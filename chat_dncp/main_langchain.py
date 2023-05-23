from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

loader = UnstructuredFileLoader("pbc.pdf", mode="elements")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
index = FAISS.from_documents(docs, embeddings)

query = "Cuál es el el plazo de vigencia de la garantía de fiel cumplimiento?"
docs = index.similarity_search(query)


chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
print(chain.run(input_documents=docs, question=query))

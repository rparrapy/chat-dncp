from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


class OpenAIQAModel(object):
    """OpenAI API LLM."""

    def __init__(self, context):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
        )
        chunks = text_splitter.split_text(context)

        embeddings = OpenAIEmbeddings()
        self.index = FAISS.from_texts(chunks, embeddings)
        self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

    def query(self, question):
        docs = self.index.similarity_search(question)
        return self.chain.run(input_documents=docs, question=question)

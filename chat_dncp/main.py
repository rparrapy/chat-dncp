# importing required modules
from io import BytesIO

import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="deepset/xlm-roberta-large-squad2",
    tokenizer="deepset/xlm-roberta-large-squad2",
)

st.set_page_config(page_title="DNCP Chat - Demo", page_icon=":robot:")

st.header("DNCP Chat - Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def query(context, question):
    return qa_pipeline({"context": context, "question": question})


def get_text():
    input_text = st.text_input("Preguntame algo: ", "", key="input", on_change=clear_text)
    return input_text

if "question" not in st.session_state:
    st.session_state["question"] = ""

def clear_text():
    st.session_state["question"] = st.session_state["input"]
    st.session_state["input"] = ""

uploaded_file = st.file_uploader("Selecciona un archivo")

if uploaded_file:
    bytes_data = uploaded_file.getvalue()
    bytesio = BytesIO(bytes_data)

    # creating a pdf reader object
    reader = PdfReader(bytesio)

    # getting a specific page from the pdf file
    text = "\n".join([page.extract_text() for page in reader.pages])

    user_input = get_text()

    if st.session_state["question"]:
        with st.spinner('El bot está pensando...'):
            output = query(text, st.session_state["question"])
            st.session_state.past.append(st.session_state["question"])
            st.session_state.generated.append(output["answer"])

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            st.markdown(st.session_state["past"][i])
            st.markdown(st.session_state["generated"][i])

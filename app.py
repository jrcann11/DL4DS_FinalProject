import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize the database and model
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
model = ChatOpenAI(model="gpt-4-turbo")

# Streamlit UI
st.title("Sudan Agricultural Advisor")
st.caption("Ask me any questions related to agriculture in Sudan!")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["contexts"] = []

def send_query(query_text):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        st.warning("Unable to find matching results.")
        return "I don't have enough information to answer that question.", None

    context_text = "\n\n---\n\n".join([f"{doc.metadata.get('source', 'Unknown source')}:\n{doc.page_content}" for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = model.predict(prompt)
    return response_text, context_text

query = st.text_input("Enter your question here:", key="query")
if st.button("Ask"):
    response, context = send_query(query)
    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["contexts"].append({"role": "assistant", "content": context})
    st.session_state["messages"].append({"role": "assistant", "content": response})
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if context:
        with st.expander("Show Context Used"):
            st.markdown(context)
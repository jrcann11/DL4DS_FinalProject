import argparse
import numpy as np
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
import streamlit as st

from trulens_eval import TruChain, Tru
from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
tru = Tru()

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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def send_query(query_text, model):
    retriever = db.as_retriever()
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    context_text = "\n\n---\n\n".join([f"{doc.metadata.get('source', 'Unknown source')}:\n{doc.page_content}" for doc, _score in results])
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain, context_text

query = st.text_input("Enter your question here:", key="query")
if st.button("Ask"):
    rag_chain, context_text = send_query(query, model)
    rag_chain.invoke(query)
    
    context = App.select_context(rag_chain)
    
    #Evaluation functions
    provider = OpenAI()
    grounded = Groundedness(groundedness_provider=OpenAI())
    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons)
        .on(context.collect())
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = (
        Feedback(provider.relevance)
        .on_input_output()
    )
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons)
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    tru_recorder = TruChain(rag_chain,
        app_id='SudanAgri_ChatApplication',
        feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness])

    response, tru_record = tru_recorder.with_record(rag_chain.invoke, query)
    
    with tru_recorder as recording:
        llm_response = response
    print(llm_response)

    st.session_state["messages"].append({"role": "user", "content": query})
    st.session_state["contexts"].append({"role": "assistant", "content": context_text})
    st.session_state["messages"].append({"role": "assistant", "content": response})
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if context_text:
        with st.expander("Show Context Used"):
            st.markdown(context_text)
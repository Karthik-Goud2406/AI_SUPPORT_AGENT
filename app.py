import streamlit as st
import time
from rag_pipeline import get_answer

st.set_page_config(page_title="AI Support Agent")

st.title("🤖 AI Support Agent")

query = st.text_input("Ask your question:")

col1, col2 = st.columns(2)

submit = col1.button("Submit")
clear = col2.button("Clear Chat")

if clear:
    st.success("Chat cleared!")

if submit:
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        start_time = time.time()

        response = get_answer(query)

        end_time = time.time()
        latency = round(end_time - start_time, 2)

        st.write("### 🤖 Answer:")
        st.write(response)

        st.caption(f"⏱ Response time: {latency} seconds")
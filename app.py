import streamlit as st
from main import run_query

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("RAG Chatbot Using ONNX and Groq")
st.write("Ask questions based on your ingested PDFs.")

query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    with st.spinner("Searching and generating answer..."):
        try:
            result = run_query(query)
            st.success("Done!")
            
            st.markdown("### Answer")
            st.write(result['answer'])
            
            st.markdown("### Sources")
            for source in result['sources']:
                with st.expander(f"Source: {source['source']} (Score: {source['score']:.2f})"):
                    st.write(f"**Preview:** {source['preview']}")
        except Exception as e:
            st.error(f"Error querying the pipeline: {e}")

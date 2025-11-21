from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

load_dotenv()


def data_loader(video_id, question):
    try:
        # LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

        # 1. Transcript fetch
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        transcript = fetched.to_raw_data()

        # 2. Clean transcript
        full_text = " ".join(entry["text"] for entry in transcript)
        full_text = full_text.replace("\xa0", " ")
        full_text = re.sub(r"\s+", " ", full_text)

        # 3. Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.create_documents([full_text])

        # 4. Embeddings + FAISS
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embedding_model)

        # 5. Retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # 6. Prompt
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            ONLY use the transcript context to answer.
            If the context is insufficient, reply: "I don't know."

            Transcript:
            {context}

            Question: {question}
            """,
            input_variables=["context", "question"]
        )

        # 7. Build RAG pipeline (in your style)
        rag_chain = (
            {
                "context": retriever | (lambda docs: " ".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # 8. Invoke chain
        return rag_chain.invoke(question)

    except TranscriptsDisabled:
        return "No transcript available for this video"



st.title("🎥 YouTube RAG Assistant")
st.write("Ask questions from any YouTube video using transcript based RAG.")

# User Inputs
video_id = st.text_input("YouTube Video ID", placeholder="Enter video ID, e.g. hmtuvNfytjM")
question = st.text_input("Your Question", placeholder="How can I help you?")

# Button
if st.button("Generate Answer"):
    if not video_id.strip():
        st.error("Please enter a video ID.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            answer = data_loader(video_id, question)
        st.text_area("Answer", answer, height=300)

# TEST
# print(data_loader("7ARBJQn6QkM", "Who has been interviewed in this video"))

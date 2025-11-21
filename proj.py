from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import os 
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.prompts import PromptTemplate 
from langchain_community.vectorstores import FAISS
import re
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def data_loader(video_id):
    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=1.25, )
    try:
        # data loader
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id)
        # list of dicts
        transcript = fetched.to_raw_data()  
        # Flatten all text
        full_text = " ".join([entry["text"] for entry in transcript])
        full_text = full_text.replace('\xa0', ' ')
        full_text = re.sub(r'\s+', ' ', full_text)

        # splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
        chunks = splitter.create_documents([full_text])

        # embeddings
        embeddingz = OpenAIEmbeddings(model = "text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddingz)

        # retriever
        retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 2})
        # return retriever.invoke('what is the core topic of this video?')
        

        # prompt decleration
        prompt = PromptTemplate(
            template= """
            You are a helpful assistant.
            Answer only from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}

            Question : {Question}
            """, input_variables=['context', 'Question']
        )

        # setting up the fields for prompt
        Question = "Please Summarize this video in 5 lines?"
        retrived_docs = retriever.invoke(Question)
        
        # joining the page content of retrieved doc to feed in context.
        context_text =" ".join(PC.page_content for PC in retrived_docs)

        # final prompt which contains context + question
        final_prompt = prompt.invoke({'context' : context_text, 'Question' : Question})


        parser = StrOutputParser()

        
        answer = llm.invoke(final_prompt)
        return answer.content

    
    except TranscriptsDisabled:
        return "No transcript available for this video"
    
print(data_loader("hmtuvNfytjM"))

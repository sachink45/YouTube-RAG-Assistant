## YouTube RAG Assistant

# Project Summary

> YouTube RAG Assistant is a Retrieval-Augmented Generation (RAG) application.  
> It fetches YouTube video transcripts, converts them into embeddings, retrieves relevant context, and uses an LLM to answer questions or summarize videos.  
> You can ask any question related to a video or request a summary.  
> Workflow includes fetching, cleaning, chunking, embedding, vector storage, and response generation.  

---

## Environment Setup

# .env File

> Create a `.env` file in the project root and add your OpenAI API key:  


---

## Installation & Running the Project

# 1. Clone the Repository

> ```bash
> git clone https://github.com/sachink45/YouTube-RAG-Assistant.git
> cd YouTube-RAG-Assistant
> ```

# 2. Create and Activate Virtual Environment

> **Windows:**  
> ```bash
> python -m venv myenv
> myenv\Scripts\activate
> ```  

> **Mac/Linux:**  
> ```bash
> python3 -m venv myenv
> source myenv/bin/activate
> ```

# 3. Install Dependencies

> ```bash
> pip install -r requirements.txt
> ```

# 4. Add Your API Key

> Ensure `.env` contains:  
> ```
> OPENAI_API_KEY=your_openai_api_key
> ```

# 5. Run the Streamlit App

> ```bash
> streamlit run app.py
> ```

# 6. Access the App

> Open in browser:  
> ```
> http://localhost:8501
> ```

---

## Tech Stack

> Python  
> Streamlit  
> LangChain  
> OpenAI GPT Models  
> FAISS Vector Store  
> YouTube Transcript API  

---

## License

> MIT License

# app.py (Deployable Version)

import os
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import arxiv
import fitz # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. SETUP: API KEYS, MODELS, AND DATABASE ---

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please add it to your .env file or Hugging Face Spaces secrets.")
    genai.configure(api_key=google_api_key)
except ValueError as e:
    print(f"Error: {e}")

# Initialize global models and database client to avoid reloading on each function call
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to vector database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="scientific_papers")

# Configure the Gemini generative model
generation_config = {"temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config)


def setup_database():
    """
    Checks if the database is populated. If not, it downloads papers from arXiv,
    processes them, and stores their embeddings in ChromaDB.
    This function runs only once when the Space is starting up.
    """
    if collection.count() > 0:
        print("Database already populated.")
        return

    print("Database is empty. Populating with new papers from arXiv...")

    # 1. Download Papers
    if not os.path.exists('papers'):
        os.makedirs('papers')

    search = arxiv.Search(query="cat:q-bio.GN", max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
    for result in search.results():
        try:
            filename = f"{result.entry_id.split('/')[-1]}.pdf"
            result.download_pdf(dirpath="./papers", filename=filename)
            print(f"Downloaded: {result.title}")
        except Exception as e:
            print(f"Failed to download {result.title}. Reason: {e}")

    # 2. Process PDFs and Create Chunks
    documents = []
    for filename in os.listdir('papers'):
        if filename.endswith(".pdf"):
            filepath = os.path.join('papers', filename)
            doc = fitz.open(filepath)
            title = doc.metadata.get('title', 'No Title Found')
            full_text = "".join(page.get_text() for page in doc)
            doc.close()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(full_text)

            for i, chunk_text in enumerate(chunks):
                documents.append({"text": chunk_text, "metadata": {"source": filename, "title": title}})
    
    if not documents:
        print("No documents were processed. Halting database setup.")
        return

    # 3. Create Embeddings and Store in ChromaDB
    print(f"Creating embeddings for {len(documents)} chunks...")
    embeddings = embedding_model.encode([doc['text'] for doc in documents], show_progress_bar=True)
    ids = [f"chunk_{i}" for i in range(len(documents))]

    collection.add(
        embeddings=embeddings,
        documents=[doc['text'] for doc in documents],
        metadatas=[doc['metadata'] for doc in documents],
        ids=ids
    )
    print("✅ Database setup complete!")


# --- 2. CORE RAG FUNCTION ---

def get_research_backed_answer(query):
    """
    This function takes a user query, retrieves relevant documents,
    and generates an answer using the Gemini LLM.
    """
    if not query:
        return "Please ask a question."
        
    query_embedding = embedding_model.encode([query])[0].tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]
    context = "\n\n---\n\n".join(retrieved_docs)
    
    prompt_template = f"""
    You are a specialized scientific assistant for biology. Your task is to answer the user's question based *only* on the provided scientific research excerpts.
    Here is the relevant context from research papers:
    ---
    {context}
    ---
    Based on the context provided, please answer the following question:
    Question: {query}
    After providing the answer, list the sources you used in a "Citations" section by citing the title of the paper.
    """

    try:
        response = model.generate_content(prompt_template)
        answer = response.text
        
        # --- CORRECTED CITATION LOGIC ---
        # The previous version had a buggy line here. This is the correct, simplified logic.
        unique_titles = set(meta['title'] for meta in retrieved_metadatas if meta.get('title'))
        citations = "\n\n**Citations:**\n" + "\n".join(f"- {title}" for title in unique_titles)
        
        return answer + citations
    except Exception as e:
        # The str(e) will now show the actual error message in the UI
        return f"An error occurred while generating the answer: {str(e)}"

# --- 3. GRADIO UI AND APP LAUNCH ---

iface = gr.Interface(
    fn=get_research_backed_answer,
    inputs=gr.Textbox(lines=2, placeholder="e.g., What are the latest findings in CRISPR gene editing?"),
    outputs=gr.Markdown(label="Answer with Citations"),
    title="🔬 Scientific RAG powered by Gemini",
    description="Ask a question about quantitative biology. The AI will answer based on recent arXiv papers and provide citations.",
    theme="soft"
)

# Run the setup function before launching the UI
if __name__ == "__main__":
    setup_database()
    print("Launching Gradio App...")
    iface.launch()
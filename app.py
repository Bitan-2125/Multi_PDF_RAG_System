import os
import io
import time
import base64
from typing import List, Tuple

import gradio as gr
import fitz  # PyMuPDF
from PIL import Image

# Core LangChain imports
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import CharacterTextSplitter

# Community & Google Integrations
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Environment Management
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# ----------------------------- CONFIG -----------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ GOOGLE_API_KEY not found! "
        "Make sure you have a .env file with GOOGLE_API_KEY=your_key_here"
    )

# UPDATED: Using Gemini 2.0 Flash
VISION_MODEL = "gemini-2.0-flash"
LLM_MODEL    = "gemini-2.0-flash"
EMBED_MODEL  = "models/text-embedding-004" 

# Persistence configuration
PERSIST_DIR  = os.environ.get("PERSIST_DIR", "faiss_index_gemini")
TOP_K        = int(os.environ.get("TOP_K", "5"))

# ----------------------------- HELPERS -----------------------------

def _caption_image_gemini(img: Image.Image, prompt: str) -> str:
    """
    Uses Gemini 2.0 Flash to caption an image.
    """
    try:
        # 1. Optimization: Resize if too large (Gemini 2.0 handles big images, but this saves bandwidth)
        img = img.copy()
        if img.width > 1024 or img.height > 1024:
            img.thumbnail((1024, 1024))
        
        # 2. Encode image to base64
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 3. Initialize Vision Model
        vision_llm = ChatGoogleGenerativeAI(
            model=VISION_MODEL, 
            temperature=0.3, 
            google_api_key=GOOGLE_API_KEY
        )

        # 4. Create Multimodal Message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ]
        )

        # 5. Invoke
        response = vision_llm.invoke([message])
        return response.content.strip()

    except Exception as e:
        return f"[Image Captioning Failed: {str(e)}]"

def _extract_pdf(pdfs: List[bytes]) -> Tuple[List[Document], List[str]]:
    """Extract text + caption images from PDFs."""
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1200, chunk_overlap=200)
    docs: List[Document] = []
    captions_all: List[str] = []

    for file_bytes in pdfs:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pdf_name = getattr(file_bytes, "name", "uploaded.pdf")

        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            
            # --- Text Extraction ---
            text = page.get_text("text") or ""
            if text.strip():
                for chunk in splitter.split_text(text):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": pdf_name, "page": page_idx + 1, "type": "text"}
                    ))

            # --- Image Extraction ---
            image_list = page.get_images(full=True)
            
            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Filter tiny images (icons, logos)
                if len(image_bytes) < 2048: 
                    continue

                pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Filter small dimensions
                if pil.width < 100 or pil.height < 100:
                    continue

                # Generate Caption
                # Sleep briefly to avoid hitting Free Tier rate limits if processing many images
                time.sleep(1.0) 
                
                caption = _caption_image_gemini(
                    pil,
                    prompt=(
                        "Analyze this academic image extracted from a paper. "
                        "If it's a graph/chart, extract the axis labels, data trends, and key values. "
                        "If it's a diagram, explain the system flow. "
                        "Do not just say 'image of a graph', provide the data."
                    )
                )

                captions_all.append(caption)
                
                # Store the caption as a Document so it can be retrieved via search
                docs.append(Document(
                    page_content=f"**[IMAGE ANALYSIS - Page {page_idx+1}]**\n{caption}",
                    metadata={"source": pdf_name, "page": page_idx + 1, "type": "image"}
                ))
                
    return docs, captions_all

def _build_or_load_store(docs: List[Document]):
    """Create FAISS index using Google Embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, 
        google_api_key=GOOGLE_API_KEY
    )
    
    # Create vector store in memory (rebuilds on every upload for this demo)
    vs = FAISS.from_documents(docs, embedding=embeddings)
    return vs

def _answer_query(vs: FAISS, question: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Retrieve context and answer using Gemini 2.0 Flash."""
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    retrieved = retriever.invoke(question)
    
    context_str = "\n\n".join(
        f"--- SOURCE: {d.metadata.get('source','file')} (Page {d.metadata.get('page','?')}) [{d.metadata.get('type','text')}] ---\n{d.page_content}"
        for d in retrieved
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY
    )
    
    prompt = f"""You are an expert academic researcher using Gemini 2.0. 
Answer the question using ONLY the provided context. 
The context contains raw text from papers AND automatic descriptions of images/charts.

If the context mentions "[IMAGE ANALYSIS]", it means the data comes from a chart or graph in the paper. Use this data to answer questions about results or trends.

CONTEXT:
{context_str}

QUESTION: 
{question}

ANSWER:
"""
    
    response = llm.invoke(prompt)
    return response.content, [(f"{d.metadata.get('source','file')} (p.{d.metadata.get('page','?')})", d.page_content[:300] + "...") for d in retrieved]

# ----------------------------- GRADIO UI -----------------------------
class Engine:
    def __init__(self):
        self.vs = None

    def index_pdfs(self, files, progress=gr.Progress(track_tqdm=False)):
        if not files:
            return "⚠️ No files uploaded.", gr.update(visible=False)

        progress(0, desc="Reading PDFs & Analyzing Images with Gemini 2.0...")
        
        byte_files = []
        for f in files:
            # Handle Gradio file objects
            if isinstance(f, str): # file path
                 with open(f, "rb") as fp:
                    byte_files.append(fp.read())
            elif hasattr(f, "read"): # file object
                byte_files.append(f.read())
            elif hasattr(f, "name"): # temp file wrapper
                with open(f.name, "rb") as fp:
                    byte_files.append(fp.read())

        try:
            docs, caps = _extract_pdf(byte_files)
        except Exception as e:
            return f"❌ Error: {str(e)}", gr.update(visible=False)

        if not docs:
             return "⚠️ No text or images found in these PDFs.", gr.update(visible=False)

        progress(0.7, desc="Embedding data...")
        self.vs = _build_or_load_store(docs)

        return f"✅ Success! Indexed {len(docs)} segments ({len(caps)} images analyzed).", gr.update(visible=True)

    def chat(self, message, history):
        if self.vs is None:
            return history + [[message, "⚠️ Please upload and process a PDF first."]]
        
        try:
            answer, srcs = _answer_query(self.vs, message)
            
            # Format sources for display
            source_text = "\n".join([f"- {s[0]}" for s in srcs])
            full_resp = f"{answer}\n\n**Sources:**\n{source_text}"
            
            return history + [[message, full_resp]]
        except Exception as e:
            return history + [[message, f"❌ API Error: {str(e)}"]]

engine = Engine()

with gr.Blocks(title="Gemini  Flash RAG") as demo:
    gr.Markdown("# ⚡ Gemini  Flash RAG (Multimodal)")
    gr.Markdown("Upload academic papers. This tool extracts text AND uses **Gemini 2.0** to 'see' and describe every chart/graph.")

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
            btn = gr.Button("Analyze PDFs", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Ask a question")
            clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        # We need to pass the full history structure properly or just the last message
        # The engine.chat method expects (message, history) and returns new history
        new_history = engine.chat(user_message, history[:-1])
        return new_history

    # Chain the events
    btn.click(engine.index_pdfs, inputs=[files], outputs=[status, chatbot])
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
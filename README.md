# 🎓 Academic RAG with Gemini 2.0 Flash & LangChain

A powerful **Multimodal Retrieval-Augmented Generation (RAG)** system designed for academic research.

Unlike standard RAG systems that only read text, this tool **sees charts, graphs, and diagrams** inside your research papers. It uses **Gemini 2.0 Flash** to generate detailed captions of every visual element — making visual data fully searchable.

---

## 🚀 Live Demo

Try the app on Hugging Face Spaces:

👉 Coming Soon. (Actually, there is some problem with my hugging face account.will provide the link as soon as it resolve .)



---

## ⚙️ Features

### 📄 PDF Text Extraction  
Parses dense academic text using **PyMuPDF** and **LangChain**.

### 📊 Visual Analysis  
Automatically detects images, charts, diagrams, and graphs.

### 👁️ Multimodal Indexing  
Uses **Gemini 2.0 Flash** to caption charts and figures, turning them into searchable text.

### 🔍 Semantic Search  
Embeds both text and image captions using Google **text-embedding-004**.

### 💬 Context-Aware Chat  
Hybrid retrieval that mixes text context + image context for more accurate Q&A.

### ⚡ Optimized  
Built-in rate-limit handling for Google’s free-tier API usage.

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| LLM & Vision | Google Gemini 2.0 Flash |
| Embeddings | Google text-embedding-004 |
| Orchestration | LangChain |
| Vector Store | FAISS |
| UI | Gradio |
| PDF Processing | PyMuPDF (Fitz) |
| Image Handling | Pillow |

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
https://github.com/Bitan-2125/Multi_PDF_RAG_System.git
cd Multi_PDF_RAG_System

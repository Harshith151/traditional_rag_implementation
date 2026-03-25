import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from Notebook.search import RAGSearch
from Notebook.data_loader import load_all_documents

load_dotenv()

# ---------- Global Configuration ----------
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_LLM = "gpt-4.1-mini"
DATA_DIR = "data/pdf"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Rebuild Index Function ----------
def rebuild_index(files, chunk_size, chunk_overlap):
    try:
        print(f"[INFO] Rebuilding index with chunk_size={chunk_size}, overlap={chunk_overlap}")
        os.makedirs(DATA_DIR, exist_ok=True)

        if not files:
            return " Please upload at least one PDF file."

        # Copy uploaded files into /data/pdf
        for file in files:
            src_path = file.name if hasattr(file, "name") else file
            dst_path = os.path.join(DATA_DIR, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            print(f"[INFO] Copied file: {dst_path}")

        # Load and rebuild FAISS
        rag = RAGSearch()
        docs = load_all_documents("data")
        total_docs = len(docs)

        rag.vectorstore.build_from_documents(docs)
        vector_count = len(rag.vectorstore.vectors) if hasattr(rag.vectorstore, "vectors") else "?"
        print(f"[INFO] Indexed {total_docs} documents into {vector_count} vectors.")

        return f" **Index successfully rebuilt!**<br> **Papers Indexed:** {total_docs}<br>🧩 **Chunks Created:** {vector_count}"

    except Exception as e:
        print("[ERROR] Rebuilding index failed:", e)
        return f" **Error rebuilding index:** {str(e)}"


# ---------- Chat Function ----------
def chat_infer(user_message, top_k, llm_model, history):
    if history is None:
        history = []

    try:
        print(f"[USER QUERY] {user_message}")
        rag = RAGSearch(llm_model=llm_model)
        print(f"[INFO] Querying vector store for: '{user_message}'")
        answer = rag.search_and_summarize(user_message, top_k=int(top_k))
        print(f"[RESPONSE LENGTH] {len(answer)} characters")

        # Add messages to chat
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})

        return history, history, gr.update(value="", interactive=True)

    except Exception as e:
        print("[ERROR]", e)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f" Error: {str(e)}"})
        return history, history, gr.update(value="", interactive=True)


# ---------- Clear Chat ----------
def clear_chat():
    return [], gr.update(value="")


# ---------- UI Layout ----------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray")) as demo:
    gr.Markdown("## IEEE Research Assistant")
    gr.Markdown(
        "Ask questions from your IEEE research paper collection. "
        "This assistant retrieves papers from your local FAISS store and summarizes them using ChatGPT."
    )

    with gr.Row():
        # ----- Sidebar -----
        with gr.Column(scale=0.35):
            gr.Markdown(" **Upload Papers (PDF)**")
            file_input = gr.File(label="Select PDF(s)", file_count="multiple", file_types=[".pdf"])
            rebuild_button = gr.Button(" Rebuild Index", variant="primary")

            vector_status = gr.Markdown(" Waiting for index build...")

            with gr.Accordion(" Settings", open=True):
                topk_slider = gr.Slider(1, 12, value=DEFAULT_TOP_K, step=1, label="Top-K Chunks per Query")
                chunk_slider = gr.Slider(500, 3000, value=DEFAULT_CHUNK_SIZE, step=100, label="Chunk Size")
                overlap_slider = gr.Slider(50, 500, value=DEFAULT_CHUNK_OVERLAP, step=10, label="Chunk Overlap")
                llm_dropdown = gr.Dropdown(
                    ["gpt-4.1-mini", "gpt-4", "gpt-3.5-turbo"],
                    value=DEFAULT_LLM,
                    label="LLM (ChatGPT)",
                )

        # ----- Chat Section -----
        with gr.Column(scale=0.65):
            chatbot = gr.Chatbot(label=" Chatbot", height=500, type="messages")
            chatbot.autoscroll = True

            msg = gr.Textbox(
                label="Textbox",
                placeholder='Ask anything... e.g. "What are interface design guidelines for AI-based components?"',
                interactive=True,
            )

            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button(" Clear Chat")

    # ----- State for persistent conversation -----
    state = gr.State([])

    # Bind actions
    msg.submit(chat_infer, [msg, topk_slider, llm_dropdown, state], [chatbot, state, msg])
    send.click(chat_infer, [msg, topk_slider, llm_dropdown, state], [chatbot, state, msg])
    clear.click(clear_chat, None, [chatbot, msg, state])
    rebuild_button.click(rebuild_index, [file_input, chunk_slider, overlap_slider], [vector_status])

print(" Launching IEEE Research Assistant...")
demo.launch(server_name="0.0.0.0", share=True)

import os
from dotenv import load_dotenv
from .vectorstore import FaissVectorStore
from langchain_openai import ChatOpenAI
from .data_loader import load_all_documents

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-20B"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from .data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please add your OpenAI API key to .env file.")
        self.llm = ChatOpenAI(api_key=openai_api_key, model_name=llm_model)
        print(f"[INFO] OpenAI LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 10) -> str:
        """
        Performs hybrid retrieval: literal keyword + semantic FAISS vector search.
        Prioritizes exact text matches like names, IDs, or rare entities.
        """
        from collections import defaultdict

        # --- Step 1: semantic vector search ---
        semantic_results = self.vectorstore.query(query, top_k=top_k)

        # --- Step 2: literal match scan (case-insensitive) ---
        keyword_hits = []
        q_lower = query.lower()
        all_meta = getattr(self.vectorstore, "metadata_list", [])

        for meta in all_meta:
            text = meta.get("text", "").lower()
            if q_lower in text:
                keyword_hits.append({"metadata": meta, "text": meta.get("text", "")})

        # --- Step 3: Merge + deduplicate ---
        combined = []
        seen = set()
        for r in keyword_hits + semantic_results:
            meta = r["metadata"]
            key = meta.get("source", "") + meta.get("text", "")[:80]
            if key not in seen:
                combined.append(r)
                seen.add(key)

        # --- Debugging info ---
        print(f"[INFO] Found {len(keyword_hits)} literal matches for '{query}'")
        print("[DEBUG] Top retrieved docs:")
        for i, r in enumerate(combined[:5]):
            meta = r["metadata"]
            preview = meta.get("text", "")[:150].replace("\n", " ")
            print(f"{i+1}. {meta.get('source', '?')} → {preview}")

        # --- Step 4: Group by paper and summarize ---
        grouped = defaultdict(list)
        for r in combined:
            meta = r["metadata"]
            if not meta:
                continue
            grouped[f"{meta.get('source','?')}||{meta.get('author','?')}"].append(meta.get("text",""))

        summaries = []
        for key, texts in grouped.items():
            source, author = key.split("||")
            merged_text = "\n".join(texts)
            summaries.append(f"Paper: {source}\nAuthor(s): {author}\n{merged_text}\n")

        context = "\n\n".join(summaries)

        import re
        highlighted_context = re.sub(
            query, f"**{query}**", context, flags=re.IGNORECASE
        )


        # --- Step 5: Build summarization prompt ---
        prompt = f"""
        You are an academic summarization assistant.

        Task:
        For the query: '{query}', summarize relevant findings.
        Highlight how each paper mentions or relates to '{query}'.
        Include paper titles and authors. Prioritize literal matches if any exist.

        Context:
        {highlighted_context}
        """

        response = self.llm.invoke([prompt])
        return response.content



# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is Generative AI Summarization and Conversations?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

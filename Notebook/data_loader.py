from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()

            reader = PdfReader(str(pdf_file))
            info = reader.metadata
            author = info.get("/Author", None)
            title = info.get("/Title", None)

            # 👇 Improved fallback extraction
            if not author or not title:
                first_page = reader.pages[0]
                lines = first_page.extract_text().split("\n")
                lines = [l.strip() for l in lines if l.strip()]

                # Skip header lines like conference info
                # Find where title block starts (usually after conference line)
                start_idx = 1 if "IEEE" in lines[0] or "ACM" in lines[0] else 0

                # Title: consecutive lines until we hit a line that contains commas or @ (likely authors)
                title_lines = []
                for line in lines[start_idx:]:
                    if "," in line or "@" in line or "University" in line:
                        break
                    title_lines.append(line)

                title = " ".join(title_lines)
                # Authors: the next line(s) after title
                possible_authors = lines[len(title_lines) + start_idx]
                author = possible_authors

            for doc in loaded:
                doc.metadata["source"] = title.strip()
                doc.metadata["author"] = author.strip()
                doc.metadata["file_name"] = pdf_file.name

            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            print(f"[DEBUG] Extracted Title: {title}")
            print(f"[DEBUG] Extracted Author(s): {author}")

        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("../data/pdf")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
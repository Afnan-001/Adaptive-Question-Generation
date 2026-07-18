import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths
MAIN_DIR = Path(__file__).resolve().parents[1]
BOOKS_FOLDER = MAIN_DIR / "books"
OUTPUT_FILE = MAIN_DIR / "outputs" / "data" / "dbms_chunks.json"


# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------
# Extract Text Page-wise
# ---------------------------
def extract_text_from_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        cleaned = clean_text(text)

        if cleaned:  # avoid empty pages
            pages_data.append({
                "page_number": page_number,
                "text": cleaned
            })

    return pages_data


# ---------------------------
# Main Pipeline
# ---------------------------
def main():

    # Ensure books exist
    pdf_files = list(BOOKS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in 'books' folder.")
        return

    print(f"✅ Found {len(pdf_files)} PDF(s)")

    # Text splitter (character-based with overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []
    total_chunks = 0

    # Process PDFs
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):

        try:
            print(f"\n📘 Processing: {pdf_file.name}")

            pages_data = extract_text_from_pdf(pdf_file)

            for page in pages_data:

                chunks = splitter.split_text(page["text"])

                for idx, chunk in enumerate(chunks):

                    chunk_record = {
                        "chunk_id": f"{pdf_file.stem}_p{page['page_number']}_c{idx}",
                        "book_name": pdf_file.name,
                        "page_number": page["page_number"],
                        "text": chunk
                    }

                    all_chunks.append(chunk_record)
                    total_chunks += 1

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {total_chunks} chunks to {OUTPUT_FILE}")


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()

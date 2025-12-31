from src.document_processor import DocumentProcessor
import os


pdf_folder = "data/pdfs"


pdfs = [f"{pdf_folder}/{f}" for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

print("PDF files found:", pdfs)  


processor = DocumentProcessor(chunk_size=900, chunk_overlap=150)
chunks = processor.process_pdfs(pdfs)


for i, c in enumerate(chunks, 1):
    print(f"\n--- CHUNK {i} ---")
    print(c.page_content)


output_file = "chunks_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, c in enumerate(chunks, 1):
        f.write(f"\n--- CHUNK {i} ---\n")
        f.write(c.page_content + "\n")

print(f"\n✅ Processed {len(pdfs)} PDF files together.")
print(f"✅ Saved {len(chunks)} chunks into text file: {output_file}")


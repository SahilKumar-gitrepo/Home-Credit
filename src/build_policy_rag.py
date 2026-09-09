from pathlib import Path
import json
import shutil

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = BASE_DIR / "policies"
VECTOR_DIR = BASE_DIR / "data" / "policy_vector_db"

METADATA_FILE = POLICY_DIR / "policy_metadata.json"


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "credit_underwriting_policies"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ============================================================
# START
# ============================================================

print("=" * 70)
print("BUILDING POLICY RAG")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Missing metadata file:\n{METADATA_FILE}"
    )

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_records = json.load(f)

metadata_by_id = {
    item["document_id"]: item
    for item in metadata_records
}


# ============================================================
# FIND PDF FILES
# ============================================================

pdf_files = sorted(
    POLICY_DIR.glob("*.pdf")
)

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in:\n{POLICY_DIR}"
    )

print(f"\nFound {len(pdf_files)} PDF document(s):")

for path in pdf_files:
    print(f"  - {path.name}")


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

documents = []

for pdf_path in pdf_files:

    print(f"\nReading: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))

    print(f"Pages: {len(reader.pages)}")

    document_id = pdf_path.stem.upper()

    metadata = metadata_by_id.get(
        document_id,
        {
            "document_id": document_id,
            "title": pdf_path.stem,
            "authority": "Unknown",
            "source_type": "regulatory_guideline",
            "version_date": "",
            "jurisdiction": "India"
        }
    )

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": metadata["document_id"],
                    "title": metadata["title"],
                    "authority": metadata["authority"],
                    "source_type": metadata["source_type"],
                    "version_date": metadata["version_date"],
                    "jurisdiction": metadata["jurisdiction"],
                    "source_file": pdf_path.name,
                    "page": page_number
                }
            )
        )


print(f"\nExtracted pages with text: {len(documents)}")


# ============================================================
# SPLIT INTO CHUNKS
# ============================================================

print("\nSplitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(documents)

print(f"Created chunks: {len(chunks)}")


# ============================================================
# SHOW SAMPLE
# ============================================================

if not chunks:
    raise RuntimeError(
        "No text chunks were created. "
        "The PDF may be scanned/image-only."
    )

print("\nSample chunk:")
print("-" * 70)

print(chunks[0].page_content[:700])

print("\nMetadata:")
print(chunks[0].metadata)


# ============================================================
# EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# REMOVE OLD DATABASE
# ============================================================

if VECTOR_DIR.exists():

    print("\nRemoving existing vector database...")

    shutil.rmtree(VECTOR_DIR)


VECTOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

print("\nCreating Chroma vector database...")

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=str(VECTOR_DIR)
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("POLICY RAG BUILD COMPLETE")
print("=" * 70)

print(f"PDF documents : {len(pdf_files)}")
print(f"Pages         : {len(documents)}")
print(f"Chunks        : {len(chunks)}")
print(f"Database      : {VECTOR_DIR}")
print(f"Collection    : {COLLECTION_NAME}")
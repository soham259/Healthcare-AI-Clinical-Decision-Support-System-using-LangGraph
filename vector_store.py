"""
vector_store.py — Local FAISS vector store for clinical guideline PDFs.

WHY THIS FILE EXISTS:
The project previously retrieved "clinical guideline evidence" by web-searching
WHO/ADA/CDC/NICE websites via Tavily (see the old guideline_retrieval.py).
That is being replaced with retrieval against a local, offline corpus of
official guideline PDFs the user has already downloaded, e.g.:

    documents/
        ADA/standards-of-care-2026.pdf
        WHO/hypertension-guideline.pdf
        WHO/cholesterol-guideline.pdf
        CDC/diabetes-management.pdf
        Merck_Manual/Merck_Manual.pdf

This module is solely responsible for:
  - Recursively discovering every PDF under documents/
  - Loading each PDF page-by-page (keeping real page numbers as metadata)
  - Chunking pages into retrieval-sized pieces
  - Embedding chunks with OllamaEmbeddings (same embedding model already
    used elsewhere in workflow.py, so no new dependency is introduced)
  - Persisting the FAISS index to disk so it is only built once
  - Loading a persisted index back
  - Running similarity search and returning chunks + metadata (organization,
    title, page, chapter, similarity_score) ready for guideline_retrieval.py
    to wrap into GuidelineSource objects.

Nothing in this file talks to workflow.py, Streamlit, or the graph directly —
it is a pure retrieval utility, kept separate per the project's "one concern,
one module" convention (mirrors how evidence_ranking.py / explainability.py
are structured).
"""
from __future__ import annotations

import glob
import os
import time
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PyMuPDF (fitz) is dramatically faster than pypdf for large PDFs (the Merck
# Manual is 3000+ pages) — pypdf parses PDF structure in pure Python page by
# page with no internal progress reporting, which is what made loading look
# "frozen". We prefer PyMuPDFLoader when the `pymupdf` package is available
# and transparently fall back to PyPDFLoader otherwise, so nothing breaks if
# it isn't installed.
try:
    from langchain_community.document_loaders import PyMuPDFLoader as _FastPDFLoader
    _FAST_LOADER_AVAILABLE = True
except Exception:
    _FAST_LOADER_AVAILABLE = False

from langchain_community.document_loaders import PyPDFLoader

# Root folder containing one subfolder per issuing organization, each holding
# that organization's guideline PDF(s). Overridable via env var so this works
# the same in dev/CI/prod without code changes.
DOCUMENTS_DIR = os.environ.get("CLINICAL_DOCUMENTS_DIR", "documents")

# Where the built FAISS index is persisted so it doesn't have to be rebuilt
# (re-embedding every PDF) on every Streamlit rerun / workflow invocation.
VECTOR_STORE_DIR = os.environ.get("CLINICAL_VECTOR_STORE_DIR", "vector_store_index")

# Folder name (documents/<THIS>/...) -> display name used everywhere else in
# the app (badges, PDF report, Streamlit "Supported By" section, etc.)
_ORG_DISPLAY_NAMES = {
    "ADA": "ADA",
    "AHA": "AHA",
    "WHO": "WHO",
    "NICE": "NICE",
    "CDC": "CDC",
    "MERCK_MANUAL": "Merck Manual",
    "MERCK": "Merck Manual",
}


def _org_from_folder(folder_name: str) -> str:
    key = folder_name.strip().upper().replace(" ", "_")
    return _ORG_DISPLAY_NAMES.get(key, folder_name.replace("_", " ").strip().title())


def _title_from_filename(file_path: str) -> str:
    name = os.path.splitext(os.path.basename(file_path))[0]
    return name.replace("_", " ").replace("-", " ").strip().title()


def _discover_pdfs(documents_dir: str = DOCUMENTS_DIR, exclude_orgs: Optional[List[str]] = None) -> List[str]:
    pattern = os.path.join(documents_dir, "**", "*.pdf")
    pdf_paths = sorted(glob.glob(pattern, recursive=True))
    if not exclude_orgs:
        return pdf_paths
    excluded = {o.strip().upper() for o in exclude_orgs}
    filtered = []
    for p in pdf_paths:
        rel = os.path.relpath(p, documents_dir)
        org_folder = rel.split(os.sep)[0] if os.sep in rel else "Unknown"
        if org_folder.strip().upper() not in excluded:
            filtered.append(p)
    return filtered


def _load_pdf_pages(pdf_path: str) -> List[Document]:
    """Loads a single PDF's pages, preferring the much faster PyMuPDFLoader
    and falling back to PyPDFLoader if pymupdf isn't installed."""
    if _FAST_LOADER_AVAILABLE:
        try:
            return _FastPDFLoader(pdf_path).load()
        except Exception as e:
            print(f"[vector_store] PyMuPDFLoader failed for '{pdf_path}' ({e}); falling back to PyPDFLoader.")
    return PyPDFLoader(pdf_path).load()


def load_documents(documents_dir: str = DOCUMENTS_DIR, exclude_orgs: Optional[List[str]] = None) -> List[Document]:
    """
    Recursively loads every PDF under documents_dir. Each PDF page becomes one
    LangChain Document, tagged with:
      - organization  (derived from the immediate parent folder, e.g. "ADA")
      - title         (derived from the filename)
      - page          (1-indexed page number)
      - file_path     (original PDF path, for debugging/traceability)

    `exclude_orgs` (e.g. ["Merck_Manual"]) lets you skip a large, slow-to-embed
    PDF while validating the rest of the pipeline quickly.

    Returns an empty list (never raises) if the folder is missing or empty,
    so callers can fail gracefully instead of crashing the workflow.

    Prints per-file progress (name, page count, elapsed time) since large
    PDFs (e.g. a 3000+ page manual) can otherwise look "frozen" for minutes
    with no visible feedback.
    """
    all_docs: List[Document] = []

    if not os.path.isdir(documents_dir):
        print(f"[vector_store] documents_dir '{documents_dir}' does not exist.")
        return all_docs

    pdf_paths = _discover_pdfs(documents_dir, exclude_orgs=exclude_orgs)
    loader_name = "PyMuPDFLoader (fast)" if _FAST_LOADER_AVAILABLE else "PyPDFLoader (slower, pure-Python fallback)"
    print(f"[vector_store] Loading {len(pdf_paths)} PDF(s) using {loader_name}...")

    for i, pdf_path in enumerate(pdf_paths, start=1):
        rel_path = os.path.relpath(pdf_path, documents_dir)
        parts = rel_path.split(os.sep)
        org_folder = parts[0] if len(parts) > 1 else "Unknown"
        organization = _org_from_folder(org_folder)
        title = _title_from_filename(pdf_path)

        print(f"[vector_store] ({i}/{len(pdf_paths)}) Loading '{rel_path}'...", flush=True)
        start = time.time()
        try:
            pages = _load_pdf_pages(pdf_path)
        except Exception as e:
            print(f"[vector_store] Failed to load '{pdf_path}': {e}")
            continue
        elapsed = time.time() - start
        print(f"[vector_store]   -> {len(pages)} page(s) loaded in {elapsed:.1f}s", flush=True)

        for page_doc in pages:
            # PyPDFLoader/PyMuPDFLoader "page" metadata is 0-indexed; display as 1-indexed.
            page_number = page_doc.metadata.get("page", 0) + 1
            page_doc.metadata.update({
                "source": organization,
                "organization": organization,
                "title": title,
                "page": page_number,
                "file_path": pdf_path,
            })
            all_docs.append(page_doc)

    return all_docs


def build_vector_store(
    embeddings,
    documents_dir: str = DOCUMENTS_DIR,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    embed_batch_size: int = 40,
    exclude_orgs: Optional[List[str]] = None,
) -> Optional[FAISS]:
    """
    Loads all PDFs, splits them into chunks, embeds them, and builds a fresh
    FAISS index. Returns None (never raises) if no PDFs were found, so the
    caller can degrade gracefully rather than crash fetch_clinical_guidelines.

    Embedding is done in small batches (embed_batch_size chunks at a time,
    default 40) instead of one single blocking call across every chunk. This
    is what actually shows progress + an ETA on large corpora (e.g. a
    3000+ page manual can be tens of thousands of chunks) instead of Ollama
    silently churning for many minutes with no feedback.

    `exclude_orgs` (e.g. ["Merck_Manual"]) lets you skip a large, slow PDF to
    validate the rest of the pipeline quickly first.
    """
    docs = load_documents(documents_dir, exclude_orgs=exclude_orgs)
    if not docs:
        print(
            f"[vector_store] No PDFs found under '{documents_dir}/'. "
            "Local clinical guideline retrieval will return no results until "
            "PDFs are added there (see module docstring for the expected layout)."
        )
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[vector_store] Split {len(docs)} page(s) into {len(chunks)} chunk(s).")

    # Same sanitization already used for PubMed chunks in workflow.py, kept
    # consistent so both retrieval paths behave the same way.
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    total = len(chunks)
    print(f"[vector_store] Embedding {total} chunk(s) via Ollama in batches of {embed_batch_size}...", flush=True)

    store: Optional[FAISS] = None
    start_all = time.time()
    for batch_start in range(0, total, embed_batch_size):
        batch = chunks[batch_start:batch_start + embed_batch_size]
        batch_t0 = time.time()

        if store is None:
            store = FAISS.from_documents(documents=batch, embedding=embeddings)
        else:
            store.add_documents(batch)

        done = min(batch_start + embed_batch_size, total)
        batch_elapsed = time.time() - batch_t0
        elapsed_total = time.time() - start_all
        avg_per_chunk = elapsed_total / done
        remaining_s = (total - done) * avg_per_chunk
        pct = done / total * 100
        print(
            f"[vector_store]   {done}/{total} chunks embedded ({pct:.0f}%) "
            f"| this batch: {batch_elapsed:.1f}s | ETA: {remaining_s/60:.1f} min",
            flush=True,
        )

    print(f"[vector_store] Embedding complete: {total} chunks in {(time.time() - start_all)/60:.1f} min.")
    return store


def save_vector_store(store: FAISS, path: str = VECTOR_STORE_DIR) -> None:
    store.save_local(path)


def load_vector_store(embeddings, path: str = VECTOR_STORE_DIR) -> Optional[FAISS]:
    """Loads a previously persisted FAISS index. Returns None if it doesn't
    exist yet or fails to load (e.g. embedding model mismatch)."""
    if not os.path.isdir(path):
        return None
    try:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[vector_store] Failed to load existing index at '{path}': {e}")
        return None


def get_or_build_vector_store(
    embeddings,
    documents_dir: str = DOCUMENTS_DIR,
    persist_path: str = VECTOR_STORE_DIR,
    force_rebuild: bool = False,
    exclude_orgs: Optional[List[str]] = None,
) -> Optional[FAISS]:
    """
    Convenience entry point used by guideline_retrieval.py:
      1. Try loading a persisted index from persist_path.
      2. If absent (or force_rebuild=True), build one from documents_dir and
         persist it for next time.
    Returns None only if there are no PDFs to index at all.

    `exclude_orgs` only affects step 2 (a fresh build) — useful for a quick
    validation pass that skips a huge, slow-to-embed PDF.
    """
    if not force_rebuild:
        store = load_vector_store(embeddings, persist_path)
        if store is not None:
            return store

    store = build_vector_store(embeddings, documents_dir, exclude_orgs=exclude_orgs)
    if store is not None:
        try:
            save_vector_store(store, persist_path)
        except Exception as e:
            print(f"[vector_store] Failed to persist vector store to '{persist_path}': {e}")
    return store


def similarity_search(store: Optional[FAISS], query: str, k: int = 5) -> List[dict]:
    """
    Runs similarity search against the given FAISS store and returns a list of
    plain dicts (not LangChain Documents) so callers don't need to import
    FAISS/Document types themselves:

        {
            "source": "ADA",
            "title": "Standards Of Care 2026",
            "page": 42,
            "chapter": "",
            "similarity_score": 0.94,
            "content": "...chunk text...",
        }

    Returns [] if store is None (no PDFs indexed yet) or the query is empty.
    """
    if store is None or not query:
        return []

    try:
        results = store.similarity_search_with_relevance_scores(query, k=k)
    except Exception:
        # Older/newer FAISS+LangChain combinations sometimes don't support
        # relevance-normalized scores; fall back to unscored search rather
        # than failing retrieval entirely.
        try:
            docs = store.similarity_search(query, k=k)
            results = [(d, None) for d in docs]
        except Exception as e:
            print(f"[vector_store] similarity_search failed: {e}")
            return []

    chunks = []
    for doc, score in results:
        similarity = 0.75 if score is None else max(0.0, min(1.0, float(score)))
        chunks.append({
            "source": doc.metadata.get("organization") or doc.metadata.get("source", "Unknown"),
            "title": doc.metadata.get("title", "Untitled"),
            "page": doc.metadata.get("page"),
            "chapter": doc.metadata.get("chapter", ""),
            "similarity_score": round(similarity, 3),
            "content": doc.page_content,
        })
    return chunks
"""
diagnose_guidelines.py — run this from the SAME folder you launch
`streamlit run app.py` from (your project root), using the SAME venv:

    python diagnose_guidelines.py

It checks, in order, every place local guideline retrieval could be
silently failing, and tells you exactly which one it is.
"""
import os
import sys

print("=" * 70)
print("STEP 1: Working directory / documents folder")
print("=" * 70)
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

documents_dir = os.environ.get("CLINICAL_DOCUMENTS_DIR", "documents")
abs_documents_dir = os.path.abspath(documents_dir)
print(f"Looking for guideline PDFs in: {abs_documents_dir}")

if not os.path.isdir(abs_documents_dir):
    print("❌ FOLDER DOES NOT EXIST. This is almost certainly the problem.")
    print(f"   Create it at: {abs_documents_dir}")
    print("   Expected layout:")
    print("     documents/ADA/standards-of-care-2026.pdf")
    print("     documents/WHO/hypertension-guideline.pdf")
    print("     documents/CDC/diabetes-management.pdf")
    print("     documents/Merck_Manual/Merck_Manual.pdf")
    sys.exit(1)

print("✅ Folder exists.")

print()
print("=" * 70)
print("STEP 2: PDFs found")
print("=" * 70)
import glob
pdfs = sorted(glob.glob(os.path.join(abs_documents_dir, "**", "*.pdf"), recursive=True))
if not pdfs:
    print("❌ NO PDFs FOUND under documents/. This is the problem.")
    print("   Files must be directly inside an org subfolder, e.g.:")
    print("     documents/ADA/standards-of-care-2026.pdf   <- OK")
    print("     documents/standards-of-care-2026.pdf        <- will be tagged org='Unknown', still works")
    sys.exit(1)

print(f"✅ Found {len(pdfs)} PDF(s):")
for p in pdfs:
    print(f"   - {p}")

print()
print("=" * 70)
print("STEP 3: pypdf installed?")
print("=" * 70)
try:
    import pypdf  # noqa: F401
    print(f"✅ pypdf installed (version {pypdf.__version__})")
except ImportError:
    print("❌ pypdf NOT installed. Run: pip install pypdf")
    sys.exit(1)

print()
print("=" * 70)
print("STEP 4: Loading PDFs via vector_store.load_documents()")
print("=" * 70)
try:
    import vector_store
except ImportError as e:
    print(f"❌ Could not import vector_store.py: {e}")
    print("   Make sure vector_store.py is in the same folder as this script.")
    sys.exit(1)

docs = vector_store.load_documents(documents_dir, exclude_orgs=["Merck_Manual"])
if not docs:
    print("❌ load_documents() returned 0 documents even though PDFs exist.")
    print("   Check the [vector_store] error messages printed above this line.")
    sys.exit(1)

print(f"✅ Loaded {len(docs)} page-documents total (Merck Manual skipped for this quick check).")
orgs = sorted({d.metadata.get("organization") for d in docs})
print(f"   Organizations detected: {orgs}")

print()
print("=" * 70)
print("STEP 5: Building/loading the FAISS index (this calls Ollama embeddings)")
print("=" * 70)
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    print(f"   Using embed model '{OLLAMA_EMBED_MODEL}' at {OLLAMA_BASE_URL}")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
except Exception as e:
    print(f"❌ Could not initialize OllamaEmbeddings: {e}")
    sys.exit(1)

store = vector_store.get_or_build_vector_store(
    embeddings, documents_dir=documents_dir, force_rebuild=True, exclude_orgs=["Merck_Manual"]
)
if store is None:
    print("❌ get_or_build_vector_store() returned None. See errors above.")
    sys.exit(1)

print("✅ FAISS index built successfully (Merck Manual excluded from this quick check).")

print()
print("=" * 70)
print("STEP 6: Running a test similarity search")
print("=" * 70)
test_query = "hypertension diabetes cardiovascular risk management"
results = vector_store.similarity_search(store, test_query, k=5)
if not results:
    print(f"❌ similarity_search('{test_query}') returned 0 results.")
    sys.exit(1)

print(f"✅ Retrieved {len(results)} chunk(s) for query: '{test_query}'")
for r in results:
    print(f"   - [{r['source']}] {r['title']} (page {r['page']}) similarity={r['similarity_score']}")

print()
print("=" * 70)
print("ALL CHECKS PASSED (quick check, Merck Manual excluded) — local")
print("guideline retrieval works for ADA/CDC/WHO. Next: run")
print("  python diagnose_guidelines_full.py")
print("(or just launch the Streamlit app) to build the FULL index including")
print("the Merck Manual — expect several minutes for that one file, with")
print("live per-batch progress + ETA now printed to the console.")
print("=" * 70)
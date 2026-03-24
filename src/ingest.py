from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("*/*.pdf"))
    if not pdf_files:
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
    print(f"found {len(pdf_files)} PDF files to process in {pdf_directory}")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} Pages")
        except Exception as e:
            print(f"Error: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents, chunk_size=600, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import litellm

loader = PyPDFLoader("documents/Mémoire_Carbon_calculator (15).pdf")
pages = loader.load()

full_text = "\n".join([p.page_content for p in pages])

docs = [Document(page_content=full_text)]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
)

query = "You are the Ai who will clear the document and give a summary of the document, give me the most important information in the document and give me a summary of the document without forget any important information to describe my work. The return must be clear and concise"
chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks[10:15]):
    response = litellm.completion(
        model="ollama/qwen2.5:1.5b",
        temperature=0.1,
        messages=[
            {"role": "system", "content": chunk.page_content},
            {"role": "user", "content": query},
        ],
    )
    print(f"\n\nChunk {i}:\n{response.choices[0].message.content}\n{'-'*40}\n")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import asyncio
import litellm
import time


async def process_chunk(i, chunk, query):
    response = await litellm.acompletion(
        model="ollama/qwen2.5:7b",
        temperature=0.1,
        max_tokens=300,
        messages=[
            {"role": "system", "content": chunk.page_content},
            {"role": "user", "content": query},
        ],
    )
    return f"\n\nChunk {i}:\n{response.choices[0].message.content}\n{'-'*40}\n"

async def main():
    loader = PyPDFLoader("documents/CGAN.pdf")
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from the PDF.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )

    query = "You are the Ai who will clear the document and give a summary of the document, give me the most important information in the document and give me a summary of the document without forget any important information to describe my work. The return must be clear and concise. Make a short summary"
    chunks = splitter.split_documents(pages)
    print("now")
    timer = time.time()
    task = [process_chunk(i,chunk,query) for i, chunk in enumerate(chunks[10:30])]
    results = await asyncio.gather(*task)
    for res in results:
        print(f"\n{res}\n{'-'*40}")
    
    print(f"Total time: {time.time() - timer:.2f} seconds")

if __name__ == ("__main__"):
    asyncio.run(main())
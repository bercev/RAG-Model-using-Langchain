import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain.schema import Document
from langchain_chroma import Chroma #db
from dotenv import load_dotenv

# data path
load_dotenv()
FIRECRAWL_API_KEY=os.getenv('FIRECRAWL_API_KEY')
CHROMA_PATH = "chroma"

def main():
    #val = input("Do you want to create a db? y/n \t")
    #if val.lower() == 'y':
    #    generate_data_store()
    #else:
    #    return
    generate_data_store()

# creating db
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# loads the documents into a variable, where the documents are the .md files
def load_documents():
    loader = DirectoryLoader("data/aws_lambda_docs", glob="*.md") # IN THE FUTURE REPLACE WITH A BETTER CALL?
    loader2 = DirectoryLoader("data/books", glob="*.md")
    documents = loader.load() + loader2.load()
    return documents


# splits the documents into smaller chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter( # recursive character splitter
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # to get meta data --
    #document = chunks[10]
    #print(document.page_content)
    #print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    #clear out original data base
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # creating a new db
    db = Chroma.from_documents(
        documents=chunks, 
        persist_directory=CHROMA_PATH,
        embedding=GPT4AllEmbeddings()
        ) # creating vectors
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# with FireCrawlLoader, scrapes the urls into documents
# not needed, could use in the future
def web_scraper(urls):

    # one big document of all urls
    docs = [FireCrawlLoader(api_key=FIRECRAWL_API_KEY, url=url, mode="scrape").load() for url in urls]
    
    # split the documents
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


if __name__=="__main__":
    main()

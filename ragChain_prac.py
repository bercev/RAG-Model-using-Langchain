import argparse
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import gpt4all
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

# handling .env variables
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

# handling paths
CHROMA_PATH = "chroma"

# prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context. Uppercase and lowercase letters do matter:

{context}

---

Answer the question based on the above context: {question}
"""
def main():   
    # create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # prepare the db
    embedding_function = GPT4AllEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # search the DB
    results = db.similarity_search_with_relevance_scores(query=query_text, k=3)
    print(f"SCORE: {results[0][1]}")
    if len(results) == 0 or results[0][1] < 0.5: # code shows that query is invalid for provided context/db
        print(f"Unable to find matching results.")
        return
    
    # formatting and printing the prompt and context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # LLM data retrieval
    llm = ChatOllama(model="llama3", format= "str", temperature=0)
    response_text = llm.invoke(prompt)

    # formatting and printing the response
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}" # get rid of the .content to get response metadata
    print(formatted_response)


if __name__ == "__main__":
    main()

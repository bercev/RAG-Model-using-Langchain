import argparse
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import warnings
from tavily import TavilyClient
from langchain.schema import AIMessage
load_dotenv()

# handling .env variables
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# handling paths
CHROMA_PATH = "chroma"

# Convert warnings to exceptions
warnings.filterwarnings("ignore", category=UserWarning)
# prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context. Uppercase and lowercase letters do matter:

{context}

---

Answer the question based on the above context: {question}
"""

prompt_to_ask_for_context = PromptTemplate(template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as releveant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \nGive a binary score 'yes' or 'no' score to indicate whether the document is relevant 
to the question. \n Provide the binary score as a JSON with a single key 'score' and no premable or explanation. 
<|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the retrieved document: \n\n {document} \n\n
Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
""", input_variables=["question", "document"]
)

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

    # formatting context text and response text ahead of time:
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    response_text = ""
    
    # code shows that query is invalid for provided context/db, if the score is low, then ask the LLM the usability of context to answer the question.
    # If LLM determines that the context is not viable, then do a web search (worst case)
    print(f"\t\t\t\t\t\tSCORE: {results[0][1]}")
    if len(results) == 0 or results[0][1] < 0.6: 
        print(f"\t\t\t\t\t\tUnable to find matching results... Asking LLM if context are usable/viable")
        ans = determine_validity_of_context(query_text, context_text)
        print(f"\t\t\t\t\t\tAnswer from LLM: {ans}")
        if ans.get('score') == 'no':
            print("\t\t\t\t\t\tLLM determines that context isn't valid, doing a web search for relevant response: ")
            # web search via tavily AI
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY) # instantiating TavilyClient
            response_from_tavily = tavily_client.search(query_text, include_answer=True) # searching the web
            response_text = response_from_tavily.get('answer')
            sources = ""
            for dictionaries in response_from_tavily.get("results"):
                sources += "Title:\t" + dictionaries.get("title") + "\t\t\t\t\tUrl: "+ dictionaries.get("url") + "\n"
            print_prompt_and_context(query_text, context_text=None)
            print_response(response_text, results, sources)
            return
      
    # if the score is valid enough from the db and verified by the LLM, then go straight into the response from the LLM
    llm = ChatOllama(model="llama3", format= "str", temperature=0)
    prompt = print_prompt_and_context(query_text, context_text)
    response_text = llm.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results] # getting sources from db
    print_response(response_text, results)

def print_prompt_and_context(query_text, context_text=None):
    # formatting and printing the prompt and context
    if context_text != None:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        print(prompt)
        return prompt
    print("\n\nQuestion: " + query_text)
    return query_text

# print the response if the response was given from the LLM or Travily AI
def print_response(response_text, results, sources):
    if isinstance(response_text, AIMessage):
        response_text = response_text.content
    formatted_response = f"Response: {response_text}\nSources:\n{sources}" # get rid of the .content to get response metadata
    print(formatted_response)

# determines the validity of the documents, which is the input
def determine_validity_of_context(query_text, context_text):
    llm = ChatOllama(model="llama3", format="json", temperature=0) # creating llm
    retrieval_grader = prompt_to_ask_for_context | llm | JsonOutputParser() # invoking llm with prompt template
    ans = retrieval_grader.invoke({"question": query_text, "document": context_text}) # getting answer
    return ans

if __name__ == "__main__":
    main()

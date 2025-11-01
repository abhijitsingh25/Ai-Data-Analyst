#Step1: Extract schema
from sqlalchemy import create_engine, inspect
import json
import re
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()

db_url = 'sqlite:///amazon.db'
def extract_schema(db_url):
    engine = create_engine(db_url)
    inspector = inspect(engine)
    schema = {}

    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        schema[table_name] = [col['name'] for col in columns]
    return json.dumps(schema)

#Step2: Text to SQL(Deepseek using ollama)
from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
groq_api_key = os.getenv("GROQ_API_KEY")


def text_to_sql(schema, prompt):
    SYSTEM_PROMPT = """
    You are an expert SQL generator. Given a database schema and a user prompt, generate a valid SQL query that answers the prompt. 
    Only use the tables and columns provided in the schema. ALWAYS ensure the SQL syntax is correct and avoid using any unsupported features. 
    Output only the SQL as your response will be directly used to query data from the database. No preamble please. Do not use <think> tags.
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Schema:\n{schema}\n\nQuestion: {user_prompt}\n\nSQL Query:")
    ])

    # model = OllamaLLM(model="deepseek-r1:8b", temperature=0) #coz we want deterministic output, ceos doesnt like creative answers
    model = ChatGroq(
        model="openai/gpt-oss-20b",   # or llama3-70b / mixtral-8x7b
        temperature=0,
        api_key=groq_api_key
    )
    

    chain = prompt_template | model
    raw_response = chain.invoke({"schema": schema, "user_prompt": prompt})

    # Extract content string
    response_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

    # Remove <think> tags if the model accidentally generates them
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    return cleaned.strip() #remove blank spaces
##TEST THE QUERY GENERATION
# schema = extract_schema(db_url)
# prompt = "Tell me the name of all the customers"
# print(text_to_sql(schema, prompt))

def get_data_from_database(prompt):
    schema = extract_schema(db_url)
    sql_query = text_to_sql(schema, prompt)
    conn = sqlite3.connect("amazon.db")
    cursor = conn.cursor()
    res = cursor.execute(sql_query)
    results = res.fetchall()
    conn.close()
    return results

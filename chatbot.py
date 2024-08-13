import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate

# Read the csv file
# df = pd.read_csv("data/df.csv")

# Data base engine
db_path = str("data/wdc_db.db")
db_path = f"sqlite:///{db_path}"
engine = create_engine(db_path)

# Our DataBase
db = SQLDatabase(engine=engine)

# LLM 
os.environ["GOOGLE_API_KEY"] = "AIzaSyAX74j5WwUFJQtcoJErqp9_w8JjVsRfDf0"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# LangChain Agent
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Context + Template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant about disasters. If you don't know a response don't say it. 
            And tell the user how you can help him but if it's a greeting greet the user """,
        ),
        ("human", "{input}"),
    ]
)

# Streamlit app
import streamlit as st 

st.title("Disaster Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("This bot help you to find informations about disasters ?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    query = user_input
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        chain = prompt | agent_executor 
        response = chain.invoke({"input": query})['output']
        st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})








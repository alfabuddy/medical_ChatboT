from flask import Flask, render_template, request
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
import os
from dotenv import load_dotenv
from src.prompt import system_prompt # Assuming system_prompt is in src/prompt.py

# Initialize Flask app
# If deploying to Elastic Beanstalk, it's better to name this 'application'
# For local testing or Render, 'app' is fine.
application = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# --- Initialize LangChain components ---

# Download embeddings model
embeddings = download_embeddings()

# Initialize Pinecone vector store
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize the LLM with a slightly more creative temperature
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

# 1. Create a History-Aware Retriever Chain
# This prompt helps the LLM rephrase the user's question to be a standalone question
# based on the chat history.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the retriever that considers history
history_aware_retriever = create_history_aware_retriever(
    llm, docsearch.as_retriever(), contextualize_q_prompt
)


# 2. Create the Final Answering Chain
# This is the prompt for the final answer generation, using your original system_prompt
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), # Using your imported system_prompt
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# This chain takes the question and retrieved documents and generates an answer
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# 3. Combine them into the final RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- Flask Routes ---

@application.route("/")
def index():
    return render_template('chat.html')

@application.route("/get", methods=["POST"])
def chat():
    # Get the user's message and the chat history from the form data
    msg = request.form.get("msg")
    history_list = request.form.getlist('history[]')
    
    if not msg:
        return "Error: No message received", 400

    # Recreate the chat_history object from the flat list sent by the frontend
    chat_history = []
    for i in range(0, len(history_list), 2):
        if i+1 < len(history_list):
            chat_history.append(HumanMessage(content=history_list[i]))
            chat_history.append(AIMessage(content=history_list[i+1]))

    # Invoke the RAG chain with the new input and history
    response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
    
    answer = response.get("answer", "Sorry, I couldn't find an answer.")
    print("Response:", answer)
    return str(answer)


if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8080, debug=True)
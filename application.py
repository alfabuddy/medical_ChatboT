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
from src.prompt import system_prompt

# Initialize Flask app
application = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate environment variables
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing required environment variables. Please check your .env file.")

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

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

# Create a History-Aware Retriever Chain
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

history_aware_retriever = create_history_aware_retriever(
    llm, docsearch.as_retriever(), contextualize_q_prompt
)

# Create the Final Answering Chain
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine them into the final RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- Flask Routes ---

@application.route("/")
def index():
    return render_template('chat.html')

@application.route("/get", methods=["POST"])
def chat():
    try:
        # Get the user's message and the chat history
        msg = request.form.get("msg")
        history_list = request.form.getlist('history[]')
        
        if not msg:
            return "Error: No message received", 400

        # Recreate the chat_history object
        chat_history = []
        for i in range(0, len(history_list), 2):
            if i+1 < len(history_list):
                chat_history.append(HumanMessage(content=history_list[i]))
                chat_history.append(AIMessage(content=history_list[i+1]))

        # Invoke the RAG chain
        response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
        
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        return str(answer)
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return "Sorry, an error occurred. Please try again.", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    application.run(host="0.0.0.0", port=port, debug=False)
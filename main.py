from flask import Flask, request, jsonify, render_template_string
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq

app = Flask(__name__)

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    "brainlox_courses_faiss",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  # Ensures safe loading
)

# Set up Groq LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key="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"
)

# RetrievalQA setup
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# HTML + CSS + JS inside Python (All-in-One Code)
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }
        #chatbox {
            width: 50%;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        .message {
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
        }
        .user {
            background: #007bff;
            color: white;
            text-align: right;
        }
        .bot {
            background: #ddd;
            text-align: left;
        }
        #userInput {
            width: 70%;
            padding: 10px;
        }
        #sendBtn {
            padding: 10px 15px;
            cursor: pointer;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h2>AI Chatbot</h2>
    <div id="chatbox"></div>
    <input type="text" id="userInput" placeholder="Type your message...">
    <button id="sendBtn" onclick="sendMessage()">Send</button>

    <script>
        function sendMessage() {
            var userInput = document.getElementById("userInput").value;
            if (userInput.trim() === "") return;

            var chatbox = document.getElementById("chatbox");
            chatbox.innerHTML += '<div class="message user">' + userInput + '</div>';

            fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: userInput })
            })
            .then(response => response.json())
            .then(data => {
                chatbox.innerHTML += '<div class="message bot">' + data.response + '</div>';
                chatbox.scrollTop = chatbox.scrollHeight;
            });

            document.getElementById("userInput").value = "";
        }
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    response = qa_chain.run(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

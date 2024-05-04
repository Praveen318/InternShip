# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import openai
import io
from pdfminer.high_level import extract_text
from llama_index import Document, VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.storage.storage_context import StorageContext
import pymongo
import regex as re
from datetime import datetime
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


# MongoDB configuration
mongo_uri = "mongodb+srv://Praveen0309:Praveen1408@cluster1.hidzlai.mongodb.net/"
database_name = "Try1"

mongodb_client = pymongo.MongoClient(mongo_uri)
file_store = MongoDBAtlasVectorSearch(mongodb_client, database_name, collection_name = "file_store")
storage_context = StorageContext.from_defaults(vector_store=file_store)

# OpenAI API key
openai.api_key = "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"

# Flask app initialization
app = Flask(__name__)

chat_history_collection = mongodb_client[database_name]["Chat"]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get/")
# Function for the bot response
def get_bot_response():
    try:
        query = request.args.get('msg')
        standalone_query = ''    
        if len(chat_history) != 0:
            memory_messages = [{"role": "system", "content": """ Using the chat history and a user input given to you, rephrase the user input into a standalone message that is from the user. 
            If the user gives an input which has no reference in the chat history, then the standalone message should be the same user input. 
            If the user input is a question, then the standalone message also should be a question which asks the same thing as the user is asking.
            Only rephrase the user input and strictly avoid responding to it.
            Do not add the words 'standalone message' in the output."""}, \
                {"role": "user", "content": "chat history: " + str(chat_history[:3]) + " user input: " + str(query)}]
            
            standalone_query = generate_response(memory_messages)
            query = standalone_query
        
            
        retrieved_data = get_context(query)

        print(query)
        print('=' * 50)
        print("Context : ", retrieved_data)
        print('=' * 50)
        articulation_messages = [{"role": "system", "content": """ You are called My bot.  
        A user will give you a context and a question. 
        Answer the question only based on the facts in the given context. 
        Ensure that the answer is relevant to the question.
        If the question cannot be answered correctly using the context then say 'Sorry, no relevant context provided'.
        If no context is provided then answer the question without using context.
        Avoid sensitive / controversial subjects.
        Don't share confidential information.
        Maintain polite and neutral language.
        If the question is ambiguous, or you are not sure how the question can be answered by the context, politely ask the user to rephrase the question.'
        """
        }, \
                {"role": "user", "content": "context: " + str(retrieved_data) + " question: " + str(query)}]
        
        result = generate_response(articulation_messages)
        chat_history.append(["user: " + str(query), "My bot: " + str(result)])

        save_to_database(session_id="sample_session1", messages=[
            {"content": query, "from": "user"},
            {"content": result, "from": "bot"}
        ])

        return result
    except Exception as e:
        print(f"Error during file upload: {e}")

        return "Please upload some pdf to continue chat"

def save_to_database(session_id, messages):
    current_time = datetime.now()

    # Check if the session already exists in the database
    existing_session = chat_history_collection.find_one({"session": session_id})

    if existing_session:
        # Session exists, append messages to the existing session
        chat_history_collection.update_one(
            {"session": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "content": msg["content"],
                                "from": msg["from"],
                                "createdAt": current_time
                            } for msg in messages
                        ],
                    }
                }
            }
        )
    else:
        # Session does not exist, create a new session
        chat_history_collection.insert_one({
            "session": session_id,
            "title": messages[0]["content"],
            "messages": [
                {
                    "content": msg["content"],
                    "from": msg["from"],
                    "createdAt": current_time
                } for msg in messages
            ]
        })

def get_context(question):
    try:
        document = [Document(text="none")]
        index = VectorStoreIndex.from_documents(document, storage_context=storage_context)
        retrieved_text_chunks = index.as_retriever().retrieve(question)
        retrieved_data = ''
        for chunk in retrieved_text_chunks:
            retrieved_data += chunk.node.text
        cleaned_paragraph = re.sub(r'\n+', ' ', retrieved_data)
        cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph).strip()
        return cleaned_paragraph
    except Exception as e:
        print(f"An error occurred while retrieving data: {e}")
        return ""
    

def generate_response(messages):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.5,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response.choices[0].message.content

chat_history = []    
retrieved_data = ''
index = None

@app.route('/upload/', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        paragraphs = []
        with io.BytesIO(file.read()) as file_stream:
            text = extract_text(file_stream)
            lines = text.splitlines()
            current_paragraph = []
            for line in lines:
                if line.strip():  # Check for non-empty lines
                    current_paragraph.append(line)
                else:
                    if current_paragraph:
                        paragraphs.append("\n".join(current_paragraph))
                        current_paragraph = []

            # Append the last paragraph, if any
            if current_paragraph:
                paragraphs.append("\n".join(current_paragraph))

        # Process paragraphs
        for i, paragraph in enumerate(paragraphs):
            print(f"Processing Paragraph {i + 1}:\n{paragraph}")

        documents = [Document(text=t) for t in paragraphs]
        # embedding_name = "hkunlp/instructor-xl"

        # embed_model = HuggingFaceInstructEmbeddings(model_name=embedding_name)


        global index
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        print("Processing complete.")
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


# Run the Flask app
if __name__ == "__main__":
    app.run(port=5500, debug=True)



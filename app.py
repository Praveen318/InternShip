# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import openai
import io
from pdfminer.high_level import extract_text
from llama_index import Document, VectorStoreIndex
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import MongoDBChatMessageHistory
from llama_index.storage.storage_context import StorageContext
import pymongo
import regex as re
from datetime import datetime4



# MongoDB configuration
mongo_uri = "mongodb+srv://Username:Password@cluster1.hidzlai.mongodb.net/"
atabase_name = "Try1"

mongodb_client = pymongo.MongoClient(mongo_uri)
file_store = MongoDBAtlasVectorSearch(mongodb_client, database_name, collection_name = "file_store")
storage_context = StorageContext.from_defaults(vector_store=file_store)

# OpenAI API key
openai.api_key = "Key"

# Flask app initialization
app = Flask(__name__)

# Initialize message history
# message_history = MongoDBChatMessageHistory(
#     connection_string=mongo_uri, database_name=database_name, collection_name = "Chat_history" ,session_id="chat1"
# )

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
        print(retrieved_data)
        print('=' * 50)
        articulation_messages = [{"role": "system", "content": """ You are called My bot.  
        A user will give you a context and a question. 
        Answer the question only based on the facts in the given context. 
        Ensure that the answer is relevant to the question.
        If the question cannot be answered correctly using the context then say 'Sorry, I don't know the answer'. 
        Avoid sensitive / controversial subjects.
        Don't share confidential information.
        Maintain polite and neutral language.
        If the question is ambiguous, or you are not sure how the question can be answered by the context, politely ask the user to rephrase the question.'
        """
        }, \
                {"role": "user", "content": "context: " + str(retrieved_data) + " question: " + str(query)}]
        
        result = generate_response(articulation_messages)
        chat_history.append(["user: " + str(query), "My bot: " + str(result)])

        # message_history.add_user_message(query)
        # message_history.add_ai_message(str(result))

        save_to_database(session_id="sample_session1", messages=[
            {"content": query, "from": "user"},
            {"content": result, "from": "bot"}
        ])

        unique_sessions = chat_history_collection.distinct("session")

        # Extract session IDs along with their titles
        session_id_title_list = []

        for session_id in unique_sessions:
            # Find one document for the session to extract the title
            session_data = chat_history_collection.find_one({"session": session_id})
            
            # Extract the title from the document
            title = session_data.get("title", "Unknown Title")

            # Append session ID and title to the list
            session_id_title_list.append(f"{session_id} : {title}")

        # Print or use the list as needed
        print(session_id_title_list)

        print(result)
        print(chat_history)

        session_id_to_retrieve = "sample_session1"
        
        # Retrieve all messages for the specified session ID
        session_messages = chat_history_collection.find({"session": session_id_to_retrieve})
        content_from_list = []
        messages = session_data.get("messages", [])
        # content_from_list = [[message.get("from", ""), message.get("content", "")] for message in messages]
        for message in messages:
            from_content_pair = [f"{message['from']}: {message['content']}"]
            content_from_list.append(from_content_pair)
        # Print or use the list as needed
        print(content_from_list)
        # messages_for_session = chat_history_collection.find({"session": session_id_to_retrieve})
        # print(type(messages_for_session))    
        # # Iterate through the messages
        # for message in messages_for_session:
        #     print(message)
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
                                # "id": str(uuid.uuid4()),  # Generate a unique ID for each message
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
                    # "id": str(uuid.uuid4()),  # Generate a unique ID for each message
                    "createdAt": current_time
                } for msg in messages
            ]
        })


def get_context(question):
    # retriever = index.as_retriever()
    # retrieved_text_chunks = retriever.retrieve(question)
    print("get_context")
    # query_engine = index.as_query_engine(similarity_top_k=20)
    # print(query_engine)
    # response = query_engine.query("What does the author think of web frameworks?")
    # print(response)
    retrieved_text_chunks = index.as_retriever().retrieve(question)

    retrieved_data = ''
    for chunk in retrieved_text_chunks:
        retrieved_data += chunk.node.text
    
    # Remove newline characters and extra spaces
    cleaned_paragraph = re.sub(r'\n+', ' ', retrieved_data)
    cleaned_paragraph = re.sub(r'\s+', ' ', cleaned_paragraph).strip()
    
    return cleaned_paragraph

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

            # Enhanced paragraph splitting using a combination of techniques
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
        # , embedding=embed_model)
        # index = VectorStoreIndex.from_documents(documents, embedding=embed_model)
        
        print("Processing complete.")
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


# Run the Flask app
if __name__ == "__main__":
    app.run(port=5500, debug=True)


# from pymongo import MongoClient

# # Connect to MongoDB
# mongodb_client = MongoClient('mongodb://localhost:27017/')
# database_name = 'your_database'
# collection_name = 'file_store'

# db = mongodb_client[database_name]
# collection = db[collection_name]

# # Store documents with unique ID and their chunks
# for i, doc in enumerate(documents):
#     doc_id = f'doc_{i}'  # Create a unique ID for each document
#     collection.insert_one({'_id': doc_id, 'document': doc})

# # Check if index exists, if not create a new one
# if 'index' not in collection.list_indexes():
#     index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#     collection.insert_one({'_id': 'index', 'index': index})
# else:
#     index = collection.find_one({'_id': 'index'})['index']

# # Retrieve documents
# retrieved_text_chunks = index.as_retriever().retrieve(question)

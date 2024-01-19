# Import necessary libraries
from flask import Flask, render_template, request, redirect, send_from_directory, flash
import openai
import os
import time
from werkzeug.utils import secure_filename

# Set the OpenAI API key
openai.api_key = "api_key"
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
# Function for the bot response
def get_bot_response():
    try:
        query = request.args.get('msg')
        standalone_query = ''    
        if len(chat_history) != 0:
            memory_messages = [{"role":"system","content":""" Using the chat history and a user input given to you, rephrase the user input into a standalone message that is from the user. 
            If the user gives an input which has no reference in the chat history, then the standalone message should be the same user input. 
            If the user input is a question, then the standalone message also should be a question which asks the same thing as the user is asking.
            Only rephrase the user input and strictly avoid responding to it.
            Do not add the words 'standalone message' in the output."""}, \
                {"role":"user","content":"chat history: " + str(chat_history[:3]) + " user input: " + str(query)}]
            

            standalone_query = generate_response(memory_messages)
            query = standalone_query
        
            
        retrieved_data = get_context(query)

        print(query)
        print('='*50)
        print(retrieved_data)
        print('='*50)
        articulation_messages = [{"role":"system","content":""" You are called My bot.  
        A user will give you a context and a question. 
        Answer the question only based on the facts in the given context. 
        Ensure that the answer is relevant to the question.
        If the question cannot be answered correctly using the context then say 'Sorry, I don't know the answer'. 
        Greet the user politely when the user interacts with you.
        Do not do any direct comparison with Airtel, Idea, BSNL, MTNL, Vodafone or other competitors.
        Avoid sensitive / controversial subjects.
        Don't share confidential information.
        Maintain polite and neutral language.
        If the question is ambiguous, or you are not sure how the question can be answered by the context, politely ask the user to rephrase the question.'
        """
        }, \
                {"role":"user","content": "context: " + str(retrieved_data) + " question: "+str(query)}]
        
        #print(query)
        result = generate_response(articulation_messages)
        chat_history.append(["user: "+str(query),"My bot: "+str(result)])
        
        print(result)
        print(chat_history)

        
        return result
    except:
        return "Please upload some pdf to continue chat"


import os
# import PyPDF2
import io
from pdfminer.high_level import extract_text
from llama_index import Document, VectorStoreIndex
from llama_index import set_global_service_context, ServiceContext
import numpy as np
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import AzureOpenAIEmbedding
import regex as re
from flask import jsonify
from llama_index import PromptHelper
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


def get_context(question):
    retriever = index.as_retriever()
    retrieved_text_chunks = retriever.retrieve(question)
    
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
    messages = messages,
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

@app.route('/upload', methods=['POST'])
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
        embedding_name = "hkunlp/instructor-xl"

        embed_model = HuggingFaceInstructEmbeddings(model_name=embedding_name)
        
        # api_key = 'api_key'
        # azure_endpoint = 'endpoint'
        # api_version = "version"

        # print(documents)

        # embed_model = AzureOpenAIEmbedding(
        #     model="text-embedding-ada-002",
        #     deployment_name="my-custom-embedding",
        #     api_key=api_key,
        #     azure_endpoint=azure_endpoint,
        #     api_version=api_version,
        # )
        print(documents)
        print("RAm")
        global index
        # service_context = ServiceContext.from_defaults(
        #     embed_model=embed_model,
        # )
        # set_global_service_context(service_context)
        index = VectorStoreIndex.from_documents(documents, embedding=embed_model) 
        print("PRem")
        print(type(index))       
        print("SDddff",index)

      
    #     return jsonify({'status': 'success'})
    # except:
    #     return jsonify({'status': 'error'})            
        
        
        return render_template("index.html")
    except:
        print("dfhbk")
        return render_template("index.html")


# Run the Flask app
if __name__ == "__main__":
    app.run(port=5500, debug= True)
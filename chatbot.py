import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def format_response(result, source_documents):
    """Format the chatbot response with proper structure"""
    formatted_response = f"**Answer:**\n{result}\n\n"
    
    if source_documents:
        formatted_response += "**Source Documents:**\n"
        for i, doc in enumerate(source_documents, 1):
            # Extract metadata
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'Unknown page')
            
            # Clean up the content - remove extra whitespace and format nicely
            content = doc.page_content.strip()
            content = ' '.join(content.split())  # Remove extra whitespace
            
            # Truncate if too long
            if len(content) > 200:
                content = content[:200] + "..."
            
            formatted_response += f"\n**Source {i}:**\n"
            formatted_response += f"- **File:** {source.split('/')[-1] if '/' in source else source}\n"
            formatted_response += f"- **Page:** {page}\n"
            formatted_response += f"- **Content:** {content}\n"
    
    return formatted_response


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def load_llm(huggingface_repo_id):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.5
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    return llm


def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """

        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID),  # Fixed: removed HF_TOKEN parameter
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            
            # Format the response properly
            formatted_response = format_response(result, source_documents)
            
            st.chat_message('assistant').markdown(formatted_response)
            st.session_state.messages.append({'role':'assistant', 'content': formatted_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
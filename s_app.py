import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

st.set_page_config(page_title="Chat with Princeton Review AI Assistant",
                       page_icon=":books:")

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model='gpt-4-turbo-preview',
    temperature=0,
    max_tokens=1000
)

embedding_llm = OpenAIEmbeddings()

st.cache_resource
vector_store = Chroma(
    embedding_function=embedding_llm,
    persist_directory='db/00'
)
general_system_template = r""" 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, please reply "I can't answer this question , as this question is out of my context, Please Try to ask a question related to University SelecION, Exam prepatation and education planning " from your own knowledge base 
 ----
{context}
----
"""
general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
user_question=""
qa_prompt = ChatPromptTemplate.from_messages( user_question )



class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict) -> None:
        return super().save_context(inputs, {'response': outputs['answer']})


def get_conversation_chain(vectorstore):
    memory = AnswerConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': qa_prompt}
    )
    return conversation_chain
st.session_state.conversation = get_conversation_chain(vector_store)

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.title("Chat with Princeton Review AI Assistant :books:")


    user_question=st.chat_input("Ask the Princeton Review assistant !")
    
    if user_question:
        response = st.session_state.conversation.invoke({'question': user_question})
        answer = response['answer']
        source_documents=''
        #source_documents = response['source_documents']

        st.session_state.chat_history.append({"role": "user", "content": user_question})

        assistant_content = answer
        if source_documents:
            assistant_content += "\n\n**Source Documents:**\n"
            for doc in source_documents:
                if len(doc.page_content) >= 30:
                    assistant_content += f"- Source: {doc.metadata['source']} - Page {doc.metadata['page']}\n"

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_content})


    for message in st.session_state.chat_history:
        role, content = message["role"], message["content"]
        with st.chat_message(role):
            if role == "user":
                st.write(f"👤 : {content}")
            else:
                st.write(f"🤖 : {content}")


if __name__ == '__main__':
    main()

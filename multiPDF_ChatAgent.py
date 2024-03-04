
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langsmith import Client

client = Client()

# Function to split PDF into chunks for summarisation
def get_tokenSplit(pages):
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
  chunk = {}
  total_tokens = 0
  string = ""
  i = 0
  for page in pages:
    tokens_page = len(enc.encode(page.page_content))
    total_tokens += tokens_page
    if total_tokens < 15000:
      string = string + " " + page.page_content
      chunk[i] = string
    else:
      #print(chunk.keys(), ":", total_tokens)
      string = page.page_content
      total_tokens = 0
      i += 1
  return chunk

#Function to get Section(Token Splits) Summaries
def get_sectionSummaries(chunks):
  from langchain_openai import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate

  chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k")

  template_string = """You are the editor of Harvard Business Review.\
    Read the section of the document added below and summarise it in 500 words or less.
    The summary should contain only the topics discussed in the document and the main insights. Make sure not to exceed 150 words.
    text: ```{text}```
    """

  sectionSummaries = {}
  for chunk in chunks:
    print (chunk)
    from langchain.prompts import ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(template_string)
    book_section = prompt_template.format_messages(text=chunks[chunk])
    section_summary = chat.invoke(book_section)
    sectionSummaries[chunk] = section_summary

  stringSectionSummaries = ""
  for sectionSummary in sectionSummaries:
    stringSectionSummaries = stringSectionSummaries + " " + sectionSummaries[sectionSummary].content

  return sectionSummaries, stringSectionSummaries

#Function to get Book Summary from Section Summaries
def get_bookSummaries(stringSectionSummaries):
  from langchain_openai import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate

  chat = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k")

  template_string2 = """You are the editor of Harvard Business Review.\
  Read the below section summaries of a document and summarise the complete document based on the section summaries.
  The summary should contain only the topics discussed in the document and the main insights. The response should not refer to the section summaries
  and only present a complete summary of the document. Mention the word summary.
  text: ```{text}```
  """

  prompt_template2 = ChatPromptTemplate.from_template(template_string2)
  book_sectionSummaries = prompt_template2.format_messages(
                    text=stringSectionSummaries)
  book_summary = chat.invoke(book_sectionSummaries)
  return book_summary

# Function to get Splits for VectorDB
def get_splitsForVectorDB(pages):
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1500,
      chunk_overlap = 150
  )
  splits = text_splitter.split_documents(pages)
  return splits

#Function to create of vector store
def create_VectorStore():
  from langchain_openai import OpenAIEmbeddings
  from langchain_community.vectorstores import Chroma

  embeddings = OpenAIEmbeddings()
  
  persist_directory = 'chroma/'
  vectordb=Chroma(
      #persist_directory=persist_directory,
      embedding_function = embeddings
  )
  return vectordb

# Function to add documents to vector store
def add_toVectorStore(vectordb,splits):
  from langchain_openai import OpenAIEmbeddings
  from langchain_community.vectorstores import Chroma
  
  embeddings = OpenAIEmbeddings()
  vectordb.add_documents(
          documents=splits,
          embedding=embeddings
          )
  return vectordb


#Function to answer questions with memory
def question_answerWithMemory (vectordb):
  from langchain.prompts import PromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain.memory import ConversationBufferMemory
  from langchain.chains import ConversationalRetrievalChain

  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  memory = ConversationBufferMemory(
      memory_key="chat_history",
      return_messages=True
  )

  # Run chain
  retriever=vectordb.as_retriever()
  qa = ConversationalRetrievalChain.from_llm(
      llm,
      retriever=retriever,
      memory=memory
  )
  return qa


#Function to load PDF
def loadpdf(uploaded_file):
  file_name = uploaded_file.name
  temp_file = f"./{file_name}"
  with open(temp_file, "wb") as file:
      file.write(uploaded_file.getvalue())
  return temp_file

def main():
  st.set_page_config(page_title = "Chat with your documents")
  st.title('Chat with your documents!')
  text = st.chat_input("Type your message here...")

  st.sidebar.header("Settings")

  uploaded_file = st.sidebar.file_uploader('Choose your PDF file', type = "pdf",accept_multiple_files=True)

  # session state
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = [
          AIMessage(content="Welcome! 🌟 Feel free to upload a document, and I'll assist you with any questions or insights you need. Whether it's summarizing content, answering queries, or discussing key points, I'm here to help enhance your understanding. Simply upload your document, and let's get started on our insightful journey together!"),
      ]

  if uploaded_file:
    pages = []
    for f in uploaded_file:
      temp_file = loadpdf(f)
      loader = PyPDFLoader(temp_file)
      pages.append(loader.load())
    if 'vs' not in st.session_state:
      st.session_state.doc_count = len(pages)

  #Managing new document uploads
  if not uploaded_file:
    if 'vs' in st.session_state:
      del st.session_state.vs
      print("VectorDB deleted")
      st.session_state.doc_count = 0
      del st.session_state.book_summary
      
  
  #Generate Summaries
  if uploaded_file and 'vs' not in st.session_state:
    #Call functions for summarisation
    with st.sidebar:
      with st.spinner('Generating Summary...'):
        st.session_state.book_summary = []
        for page in pages:
          chunks = get_tokenSplit(page)
          sectionSummaries, stringSectionSummaries = get_sectionSummaries(chunks)
          st.session_state.book_summary.append(get_bookSummaries(stringSectionSummaries))
          
    for i in range(len(st.session_state.book_summary)):      
      doc_name = pages[i][0].metadata['source']
      summary = st.session_state.book_summary[i].content
      st.session_state.chat_history.append(AIMessage(content=doc_name[2:]+" "+summary))

    #Call functions for creating VectorDB
    vectordb = create_VectorStore()
    for page in pages:
      splits = get_splitsForVectorDB(page)
      vectordb = add_toVectorStore(vectordb,splits)
    vectordb.persist()

    # saving the vector store in the streamlit session state (to be persistent between reruns)
    st.session_state.vs = vectordb
    st.sidebar.success('Uploaded, chunked and embedded successfully.')

  if uploaded_file and 'vs' in st.session_state:
    updated_doc_count = len(pages)
    if updated_doc_count > st.session_state.doc_count:
      for i in range(st.session_state.doc_count,updated_doc_count):
        print("Uploaded new document")
        temp_file = loadpdf(uploaded_file[i])
        loader = PyPDFLoader(temp_file)
        pages.append(loader.load())
        with st.sidebar:
          with st.spinner('Generating Summary...'):
            chunks = get_tokenSplit(pages[i])
            sectionSummaries, stringSectionSummaries = get_sectionSummaries(chunks)
            st.session_state.book_summary.append(get_bookSummaries(stringSectionSummaries))
            doc_name = pages[i][0].metadata['source']
            summary = st.session_state.book_summary[i].content
            st.session_state.chat_history.append(AIMessage(content=doc_name[2:]+" "+summary))
        splits = get_splitsForVectorDB(pages[i])
        vectordb = st.session_state.vs
        vectordb = add_toVectorStore(vectordb,splits)
        vectordb.persist()
        st.session_state.vs = vectordb
        st.sidebar.success('Uploaded, chunked and embedded successfully.')
    st.session_state.doc_count = updated_doc_count


  #if submitted and 'vs' in st.session_state:
  if text is not None and text !="" and 'vs' in st.session_state:
      question= text
      vectordb = st.session_state.vs
      if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = question_answerWithMemory(vectordb)
      result = st.session_state.qa_chain.invoke({"question": question})
      st.session_state.chat_history.append(HumanMessage(content=question))
      st.session_state.chat_history.append(AIMessage(content=result['answer']))

  # conversation
  for message in st.session_state.chat_history:
      if isinstance(message, AIMessage):
          with st.chat_message("Bot"):
              st.write(message.content)
      elif isinstance(message, HumanMessage):
          with st.chat_message("User"):
              st.write(message.content)


if __name__ == "__main__":
    main()

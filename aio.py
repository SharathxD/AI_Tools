import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import langchain_google_genai
import youtube_transcript_api

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None

if 'pdf_chat_history' not in st.session_state:
    st.session_state['pdf_chat_history'] = []

if 'pdf_chat_mode' not in st.session_state:
    st.session_state['pdf_chat_mode'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.set_page_config(page_title="Tools AI", page_icon="🧑‍💻", layout="wide")

# Function to configure Google API key
def configure_api_key(api_key):
    st.session_state['api_key'] = api_key
    genai.configure(api_key=api_key)

# Function to get Gemini chat response
def get_gemini_response(question):
    if st.session_state['api_key']:
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])
        response = chat.send_message(question, stream=True)
        return response
    else:
        st.error("API Key is not added. Please add your API key in the navbar.")

# Function to get PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get vector store
def get_vector_store(text_chunks):
    embeddings = langchain_google_genai.GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.session_state['api_key'])
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get conversational chain
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to extract YouTube transcript in all available languages
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_list = youtube_transcript_api.YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = ""
        for transcript_info in transcript_list:
            try:
                transcript_text = transcript_info.fetch()
                for i in transcript_text:
                    transcript += " " + i["text"]
            except Exception as e:
                st.warning(f"Could not retrieve transcript for language {transcript_info.language}: {str(e)}")
        if transcript:
            return transcript
        else:
            st.error("No transcripts available.")
            return None
    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        return None

# Function to generate summary from transcript
def generate_gemini_content(transcript_text, prompt):
    if st.session_state['api_key']:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text
    else:
        st.error("API Key is not configured. Please add your API key in the navbar.")
        return None

# Streamlit interface

try:
    st.title("AI Tools ")

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Go to", ["Home", "ChatBot", "Image Captioning", "PDF Reader", "YouTube Summarizer"])

        st.header("API Key Configuration")
        api_key_input = st.text_input("Enter your Google API Key", type="password")
        if st.button("Set API Key"):
            configure_api_key(api_key_input)
            st.success("API Key set successfully!")
        st.markdown("Don't have an API key? Generate [Here](https://aistudio.google.com/app/apikey)")

    if page == "Home":
        st.header("Welcome to the AI Tools Application")
        images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg", "img6.jpg"]
        cols = st.columns(3)
        
        for idx, image_path in enumerate(images):
            with cols[idx % 3]:
                st.image(image_path, use_column_width=True)
    
    if page == "ChatBot":
        st.title("ChatBot Service")
        user_input = st.text_input("Input:", key="input")
        submit = st.button("Ask the Question")
        if submit and user_input:
            response = get_gemini_response(user_input)
            st.session_state['chat_history'].append(("You", user_input))
            st.subheader("Response")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))

        st.subheader("Chat History")
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")

    elif page == "Image Captioning":
        st.title("Generate Caption with Hashtags")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None and st.button('Upload'):
            try:
                if st.session_state['api_key']:
                    genai.configure(api_key=st.session_state['api_key'])
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    img = Image.open(uploaded_file)
                    caption = model.generate_content(["Generate a detailed caption that accurately describes the content, mood, and potential story of the image in english", img])
                    tags = model.generate_content(["Generate 10 trending hashtags for the image in a line in English", img])
                    st.image(img, caption=f"Caption: {caption.text}")
                    st.write(f"Tags: {tags.text}")
                else:
                    st.error("API Key is not configured. Please add your API key in the navbar.")
            except Exception as e:
                st.error(f"Failed to generate caption due to: {str(e)}")

    elif page == "PDF Reader":
        st.header("PDF Reader")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
            st.write("PDF text and vector store created successfully!")

            col1, col2 = st.columns(2)
            with col1:
                summarize_button = st.button("SUMMARIZE")
            with col2:
                chat_button = st.button("CHAT")

            if summarize_button:
                with st.spinner('Summarizing...'):
                    summary = generate_gemini_content(text, "Provide a detailed summary of the following text, ensuring all key points, arguments, and supporting details are included. Maintain the original text's structure and flow as much as possible")
                    if summary:
                        st.subheader("Summary")
                        st.write(summary)

            if chat_button:
                st.session_state['pdf_chat_mode'] = True

            if st.session_state['pdf_chat_mode']:
                st.subheader("PDF QnA Chat")
                
                with st.form("pdf_chat_form", clear_on_submit=True):
                    question = st.text_input("Ask a question about the PDF:")
                    submit_question = st.form_submit_button("Send")

                if submit_question and question:
                    try:
                        chain = get_conversational_chain(api_key_input)
                        vector_store = FAISS.load_local(
                            "faiss_index",
                            embeddings=langchain_google_genai.GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", 
                                google_api_key=st.session_state['api_key']
                            ),
                            allow_dangerous_deserialization=True
                        )
                        docs = vector_store.similarity_search(question)
                        answer = chain.run(input_documents=docs, question=question)

                        # Save to chat history
                        st.session_state['pdf_chat_history'].append(("You", question))
                        st.session_state['pdf_chat_history'].append(("Bot", answer))

                        # Display the answer
                        st.subheader("Answer")
                        st.write(answer)
                    except Exception as e:
                        if not st.session_state['api_key']:
                            st.error("Enter your API key.")
                        else:
                            st.error(f"OOPS! SOMETHING WENT WRONG: {str(e)}")

    elif page == "YouTube Summarizer":
        st.header("YouTube Video Summarizer")
        youtube_link = st.text_input("Enter the YouTube Video URL:")
        if youtube_link:
            video_id = youtube_link.split("=")[1]
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        if st.button("Get Summary"):
            transcript_text = extract_transcript_details(youtube_link)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, "Provide a detailed summary of the following text, ensuring all key points, arguments, and supporting details are included. Maintain the original text's structure and flow as much as possible")
                if summary:
                    st.write(summary)

except Exception:
    st.error("OOPS! SOMETHING WENT WRONG.")
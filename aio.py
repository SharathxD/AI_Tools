import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import langchain_google_genai
import youtube_transcript_api

if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.set_page_config(page_title="Tools AI", page_icon="🌐", layout="wide")

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
        st.error("API Key is not configured. Please add your API key in the navbar.")

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
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = genai.ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to extract YouTube transcript
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id, languages=('en',))
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript
    except Exception:
        st.error("Error extracting transcript: No Transcription available in English! Other languages coming soon.")
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
        images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg", "image6.jpg"]
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
                    caption = model.generate_content(["Write a short caption with 20 words for the image in English", img])
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
            st.write("PDF text and vector store created successfully! text feature coming SOON")

    elif page == "YouTube Summarizer":
        st.header("YouTube Video Summarizer")
        youtube_link = st.text_input("Enter the YouTube Video URL:")
        if youtube_link:
            video_id = youtube_link.split("=")[1]
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        if st.button("Get Summary"):
            transcript_text = extract_transcript_details(youtube_link)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, "summarize this text pointwise")
                if summary:
                    st.write(summary)

except Exception as e:
    st.error("OOPS! SOMETHING WENT WRONG.")
    st.error(f"Error details: {str(e)}")

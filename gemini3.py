# Import necessary libraries
import streamlit as st  # For creating web apps
import os  # For interacting with the operating system
from langchain_google_genai import ChatGoogleGenerativeAI  # To use the Gemini LLM
import google.generativeai as genai  # For configuring Google's Generative AI API
from langchain.chains import ConversationChain  # To manage conversations
from langchain.chains.conversation.memory import ConversationSummaryMemory  # To remember conversation history
from streamlit_chat import message  # To display chat messages in the app
from dotenv import load_dotenv  # To load environment variables from a file

# headers = {
#     "authorization": st.secret["GOOGLE_API_KEY"],
#     "content-type" : "application/json"
# }

# Load environment variables and configure API key (assuming .env file is present)
load_dotenv()

os.environ["GOOGLE_API_KEY"] = "AIzaSyAXOjM6OYg0bDS-NckePUAK_IAb_15G_z0"

os.getenv("GOOGLE_API_KEY")  # Access the Google API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure the API key

# Create a Gemini model instance with adjusted temperature for more creative responses
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Set up Streamlit session state to track conversation and message history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = None  # Initialize conversation if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  # Initialize message list if it doesn't exist

# Configure Streamlit app title, icon, and heading
st.set_page_config(page_title="Gemini LangChain Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align:center;'>Gemini LangChain Chatbot</h1>", unsafe_allow_html=True)

# Function to get responses from the model, creating a conversation chain if necessary
def get_response(user_input):
    if st.session_state['conversation'] is None:
        st.session_state['conversation'] = ConversationChain(
            llm=model,
            verbose=True,
            memory=ConversationSummaryMemory(llm=model)
        )  # Create a conversation chain if it doesn't exist

    response = st.session_state['conversation'].predict(input=user_input)  # Get response from the model
    return response

# Create containers for organizing the UI elements
response_container = st.container()  # Container for chat messages
container = st.container()  # Container for input form

# Create a form for user input and displaying responses
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Your chat goes here:", key='input')  # Input box for user
        submit_button = st.form_submit_button(label='Send')  # Submit button
        if submit_button:
            st.session_state['messages'].append(user_input)  # Add user input to message list
            model_response = get_response(user_input)  # Get model response
            st.session_state['messages'].append(model_response)  # Add model response to message list

            # Display messages in the chat container, alternating between user and AI
            with response_container:
                for i in range(len(st.session_state['messages'])):
                    if (i % 2) == 0:
                        message(st.session_state['messages'][i], is_user=True, key=str(i) + 'user')
                    else:
                        message(st.session_state['messages'][i], key=str(i) + '_AI')


# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = []

# Button to clear chat history
st.button("Clear Chat", on_click=clear_chat_history)

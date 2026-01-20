import streamlit as st
import sqlite3
from llm_chains import load_normal_chain, load_pdf_chat_chain
from audio_handler import transcribe_audio
from image_handler import handle_image
from pdf_handler import add_documents_to_db
from database_operations import *
from utils import get_timestamp, load_config, get_avatar
from html_templates import css
from streamlit_mic_recorder import mic_recorder

config = load_config()

@st.cache_resource
def load_chain():
    if st.session_state.pdf_chat:
        return load_pdf_chat_chain()
    return load_normal_chain()

def main():
    st.title("Cleora")
    st.write(css, unsafe_allow_html=True)

    if "db_conn" not in st.session_state:
        st.session_state.db_conn = sqlite3.connect(
            config["chat_sessions_database_path"], check_same_thread=False
        )
        st.session_state.session_key = "new_session"

    user_input = st.chat_input("Type your message here")

    if user_input:
        llm = load_chain()
        answer = llm.run(
            user_input,
            load_last_k_text_messages(
                st.session_state.session_key,
                config["chat_config"]["chat_memory_length"]
            )
        )
        save_text_message(st.session_state.session_key, "human", user_input)
        save_text_message(st.session_state.session_key, "ai", answer)
        st.chat_message("ai").write(answer)

if __name__ == "__main__":
    main()

import os
import logging
import streamlit as st
from forum_rag import ForumRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def rag():
    return ForumRAG()


@st.cache_data(show_spinner=False)
def generate_response(query_text):
    try:
        output = rag().input(query_text)
        # logger.info(f"Output: {output}")
        return output
    except Exception as e:
        st.exception(e)


def forum_topics():
    # Page title
    st.set_page_config(page_title="👶🏼 Kinder Forum")
    st.title("👶🏼 Kinder Themen 👶🏼")
    result = ""
    with st.form("myform"):
        query_text = st.text_input("Worum geht es?", placeholder="Baby schreit nachts.")
        password = st.text_input("Key", type="password")
        submitted = st.form_submit_button("Suche")
    if submitted:
        if password == "schnurri2023":
            with st.spinner("Suchen..."):
                response = generate_response(query_text)
                result = response
        else:
            st.error("Falscher Key", icon="🚨")
    if result != "":
        st.info(response)


if __name__ == "__main__":
    forum_topics()

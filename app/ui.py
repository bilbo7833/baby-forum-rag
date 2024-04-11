import pathlib
import logging
import streamlit as st
from forum_rag import ForumRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def rag():
    return ForumRAG()


@st.cache_data
def generate_response(query_text):
    try:
        output = rag().input(query_text)
        logger.info(f"Output: {output}")
        return output
    except Exception as e:
        st.exception(e)


def forum_topics():
    files = [f for f in pathlib.Path().iterdir()]
    print(files)
    # Page title
    st.set_page_config(page_title="👶🏼 Kinder Forum")
    st.title("👶🏼 Kinder Themen 👶🏼")
    result = ""
    with st.form("myform"):
        query_text = st.text_input("Worum geht es?", placeholder="Baby schreit nachts.")
        password = st.text_input("Key", type="password")
        submitted = st.form_submit_button("Suche")
    if submitted and (password == "schnurri2023"):
        with st.spinner("Suchen..."):
            response = generate_response(query_text)
            result = response
    if result != "":
        st.info(response)


if __name__ == "__main__":
    forum_topics()

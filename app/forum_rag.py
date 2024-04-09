import logging
import os
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# TODO: Wrap documents in <conversation> tags
def format_context(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class ForumRAG:

    EMBEDDING_MODEL = "aari1995/German_Semantic_STS_V2"
    INDEX_PATH = "../res/forum_index"
    MAX_TOKENS = 4000
    PROMPT_TEMPLATE = """Du bist ein hilfreicher Assistent für junge Eltern, der Informationen aus Foren sammelt, um Eltern bei der Beantwortung von Fragen über ihre Kinder zu helfen.

            Verwende die folgenden Konversationen aus Elternforen, um die Frage des Benutzers zu beantworten. Jede Zeile beginnt mit dem Namen des Benutzers, gefolgt von ":" und dann dem Kommentar, zum Beispiel so: "John: Bei mir ist es genauso."
            Verschiedene Konversationen können sich auf dasselbe Thema beziehen.
            <conversations>
            {context}
            </conversations>
            
            Wenn du die Unterhaltungen im Forum zitierst, gib bitte den Benutzernamen in deiner Antwort an. Hier ist ein Beispiel in <example> Tags:
            <example>
            Viele Nutzer sagen dass es normal ist. Cari234 sagt, z.B, dass sie täglich überfordert ist. Lomo2 hat gleiche Erfahrungen.
            </example>

            Schreib eine klare Antwort auf diese Frage:
            <question>
            {question}
            </question>

            Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt.
            Wenn du die Frage beantwortest, nenn bitte konkrete Beispiele und Tipps aus den Forendiskussionen und verallgemeinere die Details nicht.

            Bevor du antwortest, gehe die folgenden Schritte durch:
            1. Fasse jede Unterhaltung zusammen und extrahiere die relevanten Informationen, Details und Beispiele, die sich auf die Benutzerfrage beziehen. Gib die Zusammenfassungen in <summary>-Tags aus.
            2. Beantworte die Benutzerfrage auf der Grundlage der Zusammenfassungen von Schritt 1. Füge alle relevanten Informationen, Details und Beispiele ein. Behalte die Antwort auf maximal 5 Sätze. Gib diesen Text im <answer>-Tag aus.
            """

    def __print_matches(self, matches):
        for p in matches:
            print(f"\n\nMatch with similarity {p[1]}:\n{p[0].page_content}")

    def __get_rag_chain(self, vectorstore, llm):
        retriever = vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: format_context(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain)
        return rag_chain_with_source

    def __init__(self):
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embedding_model_wrapper = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        vector_store = FAISS.load_local(
            self.INDEX_PATH,
            embedding_model_wrapper,
            allow_dangerous_deserialization=True,
        )
        # openai_api_key = os.environ.get("OPENAI_API_KEY")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        # gpt4_preview = ChatOpenAI(
        #     model_name="gpt-4-1106-preview",
        #     openai_api_key=openai_api_key,
        #     max_tokens=self.MAX_TOKENS,
        # )
        # gpt4 = ChatOpenAI(
        #     model_name="gpt-4",
        #     openai_api_key=openai_api_key,
        #     max_tokens=self.MAX_TOKENS,
        # )
        # gpt3_5 = ChatOpenAI(
        #     model_name="gpt-3.5-turbo-0125",
        #     openai_api_key=openai_api_key,
        #     max_tokens=self.MAX_TOKENS,
        # )

        # opus = ChatAnthropic(
        #     model="claude-3-opus-20240229",
        #     anthropic_api_key=anthropic_api_key,
        #     max_tokens=self.MAX_TOKENS,
        # )
        # sonnet = ChatAnthropic(
        #     model="claude-3-sonnet-20240229",
        #     anthropic_api_key=anthropic_api_key,
        #     max_tokens=self.MAX_TOKENS,
        # )
        haiku = ChatAnthropic(
            model="claude-3-haiku-20240307",
            anthropic_api_key=anthropic_api_key,
            max_tokens=self.MAX_TOKENS,
        )

        # rag_opus = self.__get_rag_chain(vector_store, opus)
        # rag_sonnet = self.__get_rag_chain(vector_store, sonnet)
        rag_haiku = self.__get_rag_chain(vector_store, haiku)
        # rag_gpt4_preview = self.__get_rag_chain(vector_store, gpt4_preview)
        # rag_gpt4 = self.__get_rag_chain(vector_store, gpt4)
        # rag_gpt3_5 = self.__get_rag_chain(vector_store, gpt3_5)

        self.rag = rag_haiku

    def input(self, input_string):
        answer = self.rag.invoke(input_string)
        logger.info(f"RAG Answer: {answer}")
        # print(f"RAG Answer: {answer}")
        return answer["answer"]

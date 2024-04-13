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
    context = (
        "<conversation>"
        + "</conversation>\n\n<conversation>".join(doc.page_content for doc in docs)
        + "</conversation>"
    )
    logger.info(context)
    return context


class ForumRAG:

    EMBEDDING_MODEL = "aari1995/German_Semantic_STS_V2"
    INDEX_PATH = "res/forum_index"
    MAX_TOKENS = 4096
    PROMPT_TEMPLATE = """Du bist ein hilfreicher deutscher Assistent für junge Eltern, der Informationen aus Foren sammelt, um Eltern bei der Beantwortung von Fragen über ihre Kinder zu helfen.

            Verwende die folgenden Konversationen aus Elternforen, um den Nutzer über ein bestimmtes Thema zu helfen. Jede Konversation ist in einem <conversation>-Tag enthalten. Jede Zeile beginnt mit dem Namen des Benutzers, gefolgt von ":" und dann dem Kommentar, zum Beispiel so: "John: Bei mir ist es genauso."
            Verschiedene Konversationen können sich auf dasselbe Thema beziehen.
            <conversations>
            {context}
            </conversations>
            
            Wenn du die Unterhaltungen im Forum zitierst, gib bitte den Benutzernamen in deiner Antwort an. Hier ist ein Beispiel in <example> Tags:
            <example>
            Viele Nutzer sagen dass es normal ist Kinder nachts zu stillen. Cari234 sagt, z.B, dass sie täglich überfordert ist. Lomo2 hat gleiche Erfahrungen.
            </example>
            <example>
            Die Konversationen beinhalten keine Hinweise ob es normal ist, dass Kinder nachts Wachfenster haben.
            </example>

            Hier ist deine Aufgabe: Welche relevanten Informationen kannst du aus den oben genannten Gesprächen zu diesem Thema entnehmen? Das Thema ist unten in <question>-Tags:
            <question>
            {question}
            </question>

            Um deine Aufgabe zu erledigen, gehe die folgenden Schritte durch:
            1. Erstelle eine umfassende Zusammenfassung für jede Konversation in <conversation> Tags oben. Die Zusammenfassungen sollte alle wichtigen Punkte und Hauptgedanken des Originaltextes die sich auf diesem Aufgabe Thema beziehen abdecken und gleichzeitig die Informationen in einem prägnanten und leicht verständlichen Format zusammenfassen.
            Achte bitte darauf, dass die Zusammenfassung die Usernamen, relevante Details und Beispiele enthält, die die Hauptgedanken unterstützen, und vermeide unnötige Informationen oder Wiederholungen.
            2. Erledige deine Aufgabe auf der Grundlage der Zusammenfassungen von Schritt 1. Füge alle relevanten Informationen, Details und Beispiele ein die aus der Zusammenfassungen rauskommen und sich auf diesem Thema beziehen. Behalte die Antwort auf maximal 5 Sätze.

            Wenn du die Aufgabe erledigst, nenn bitte konkrete Beispiele und Tipps aus den Forendiskussionen und verallgemeinere die Details nicht. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt.

            Fang deine Antwort mit "Das sagen andere Nutzer dazu:" an. Danach, gib die Zusammenfassungen von Schritt 1 aus als Bulletpoint-Liste aus. Gib dann deine zusammenfassende Antwort gemäß Schritt 2 in eine neue Zeile ein.
            Hier sind Beispiele wie deine Antworten formattiert werden sollen, in <answer-example>-Tags:

            <answer-example>
            Das sagen andere Nutzer dazu:
            - jomda erklärt, dass Babys manchmal schreien, wenn ältere Geschwister versorgt werden müssen oder wenn das Baby nachts wach wird, weil der Betreuer kurz etwas erledigen muss.
            - Mami83 berichtet, dass ihr Baby auch Phasen hatte, in denen es sehr unruhig war und häufig nachts aufwachte. Caro34 bestätigt.

            Das sagen die Nutzer zusammengefasst: Häufiges nächtliches Aufwachen und Unruhe sind bei Babys oft normal und hängen mit deren Entwicklung und Bedürfnissen zusammen. Die Situation kann vorübergehend sehr herausfordernd sein, aber die Eltern müssen durchhalten, da es mit der Zeit wieder besser wird.
            </answer-example>
            <answer-example>
            Das sagen andere Nutzer dazu:
            - bilbo45 sagt, dass solche schwierigen Phasen normal sind und immer wieder kommen können, da sich Babys ständig weiterentwickeln. Sie ermutigt, durchzuhalten, da es mit der Zeit besser wird.
            - Micebwn beschreibt, dass ihr 9 Monate altes Baby seit Tagen nachts weinend aufwacht.

            Aus den Gesprächen geht hervor, dass bei den meisten Kindern dieser spezielle Test mit schwarzen und weißen Punkten, bei denen das Kind etwas Bestimmtes erkennen und zeigen soll, im Alter von 11 Monaten noch nicht funktioniert. Die Eltern berichten, dass ihre Kinder stattdessen eher versuchen, mit den Fingern in den Mund der Ärzte zu kommen oder generell eher an der Untersuchung interessiert sind als an dem Test.
            </answer-example>    

            Erledige jetzt bitte deine Aufgabe bezüglich des Themas von oben.
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
        sonnet = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=anthropic_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        haiku = ChatAnthropic(
            model="claude-3-haiku-20240307",
            anthropic_api_key=anthropic_api_key,
            max_tokens=self.MAX_TOKENS,
        )

        # rag_opus = self.__get_rag_chain(vector_store, opus)
        rag_sonnet = self.__get_rag_chain(vector_store, sonnet)
        rag_haiku = self.__get_rag_chain(vector_store, haiku)
        # rag_gpt4_preview = self.__get_rag_chain(vector_store, gpt4_preview)
        # rag_gpt4 = self.__get_rag_chain(vector_store, gpt4)
        # rag_gpt3_5 = self.__get_rag_chain(vector_store, gpt3_5)

        self.rag = rag_haiku

    def input(self, input_string):
        answer = self.rag.invoke(input_string)
        # logger.info(f"RAG Answer: {answer}")
        return answer["answer"]

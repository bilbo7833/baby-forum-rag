import logging
import os
import weaviate
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    Settings,
    PromptTemplate,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


# # TODO: Wrap documents in <conversation> tags
# def format_context(docs):
#     context = (
#         "<conversation>"
#         + "</conversation>\n\n<conversation>".join(doc.page_content for doc in docs)
#         + "</conversation>"
#     )
#     # logger.info(context)
#     return context


class ForumRAG:

    EMBEDDING_MODEL = "aari1995/German_Semantic_STS_V2"
    INDEX_NAME = "BabyForum"
    MAX_TOKENS = 4096
    RETRIEVED_RESULTS = 20
    SIMILARITY_THRESHOLD = 0.75
    PROMPT_TEMPLATE = """Du bist ein hilfreicher deutscher Assistent für junge Eltern, der Informationen aus Foren sammelt, um Eltern bei der Beantwortung von Fragen über ihre Kinder zu helfen.

            Verwende die folgenden Konversationen aus Elternforen, um den Nutzer über ein bestimmtes Thema zu helfen. Jede Zeile beginnt mit dem Namen des Benutzers, gefolgt von ":" und dann dem Kommentar, zum Beispiel so: "John: Bei mir ist es genauso."
            Verschiedene Konversationen können sich auf dasselbe Thema beziehen.
            <conversations>
            {context_str}
            </conversations>
            
            Wenn du die Unterhaltungen im Forum zitierst, gib bitte den Benutzernamen in deiner Antwort an. Hier sind Beispiele in <example> Tags:
            <example>
            Viele Nutzer sagen dass es normal ist Kinder nachts zu stillen. Cari234 sagt, z.B, dass sie täglich überfordert ist. Lomo2 hat gleiche Erfahrungen.
            </example>
            <example>
            Die Konversationen beinhalten keine Hinweise ob es normal ist, dass Kinder nachts Wachfenster haben.
            </example>

            Hier ist deine Aufgabe: Welche relevanten Informationen kannst du aus den oben genannten Gesprächen zu diesem Thema entnehmen? Das Thema ist unten in <question>-Tags:
            <question>
            {query_str}
            </question>

            Um deine Aufgabe zu erledigen, gehe die folgenden Schritte durch:
            1. Erstelle eine umfassende Zusammenfassung für jede Konversation in <conversation> Tags oben. Die Zusammenfassungen sollte alle wichtigen Punkte und Hauptgedanken des Originaltextes die sich auf diesem Aufgabe Thema beziehen abdecken und gleichzeitig die Informationen in einem prägnanten und leicht verständlichen Format zusammenfassen.
            Achte bitte darauf, dass die Zusammenfassung die Usernamen, relevante Details und Beispiele enthält, die die Hauptgedanken unterstützen, und vermeide unnötige Informationen oder Wiederholungen.
            2. Erledige deine Aufgabe auf der Grundlage der Zusammenfassungen von Schritt 1. Füge alle relevanten Informationen, Details und Beispiele ein die aus der Zusammenfassungen rauskommen und sich auf diesem Thema beziehen. Behalte die Antwort auf maximal 5 Sätze.

            Wenn du die Aufgabe erledigst, nenn bitte konkrete Beispiele und Tipps aus den Forendiskussionen und verallgemeinere die Details nicht. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt.
            Fang deine Antwort mit "Das sagen andere Nutzer dazu:" an. Danach, gib die Zusammenfassungen von Schritt 1 aus als Bulletpoint-Liste aus. Für jede Konversation soll es ein Bulletpoint geben. Gib dann deine zusammenfassende Antwort gemäß Schritt 2 in eine neue Zeile ein.
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

    def __print_matches(self, sources):
        logger.info(f"{len(sources)} RAG Answer Sources:")
        count = 0
        for source in sources:
            logger.info(
                f"Source #{count} has score {source.score}:\n{source.text[:100]}"
            )
            count += 1

    def get_llm(self):
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        Settings.tokenizer = Anthropic().tokenizer
        haiku = Anthropic(
            model="claude-3-haiku-20240307",
            api_key=anthropic_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        sonnet = Anthropic(
            model="claude-3-sonnet-20240229",
            api_key=anthropic_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        opus = Anthropic(
            model="claude-3-opus-20240229",
            api_key=anthropic_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        gpt_35 = OpenAI(
            model="gpt-3.5-turbo-0125",
            api_key=openai_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        gpt_4_turbo = OpenAI(
            model="gpt-4-turbo-2024-04-09",
            api_key=openai_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        gpt_4 = OpenAI(
            model="gpt-4",
            api_key=openai_api_key,
            max_tokens=self.MAX_TOKENS,
        )
        return opus

    def __init__(self):

        embedding_model = HuggingFaceEmbedding(model_name=self.EMBEDDING_MODEL)
        llm = self.get_llm()
        Settings.llm = llm
        Settings.embed_model = embedding_model
        client = weaviate.Client("http://localhost:8080")
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            embed_model=embedding_model,
            index_name=self.INDEX_NAME,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.RETRIEVED_RESULTS,
        )
        response_synthesizer = get_response_synthesizer(
            llm=llm, response_mode="simple_summarize"
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=self.SIMILARITY_THRESHOLD)
            ],
        )
        qa_prompt_tmpl = PromptTemplate(self.PROMPT_TEMPLATE)
        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        self.rag = query_engine

    def input(self, input_string):
        answer = self.rag.query(input_string)
        self.__print_matches(answer.source_nodes)
        return answer.response

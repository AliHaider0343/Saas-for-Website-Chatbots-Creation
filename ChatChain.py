import ast
import json
from google import generativeai as genai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from langchain.schema.retriever import BaseRetriever
from pydantic import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate



safety_settings_NONE = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

embedding_function = OpenAIEmbeddings()
chroma_db = Chroma(persist_directory=f"./WebSync-Vector-Store", embedding_function=OpenAIEmbeddings())

RESPONSE_TEMPLATE = """
You are a Helpful AI Assistant who have the Knowledge about the his Business and Your Task is to Generate a comprehensive and informative answer of 80 words or less for the given question based solely on the provided context and Act Like you are Offering that Product. 

Must Follow the Below Formatting Instructions:
1. You should use bullet points in your answer for readability. 
2. Put citations where they apply rather than putting them all at the end. Must ensure that you add the citations for each of the Relevenat Answer block.
3. Use proper alignment and indentation and make the format of the answer in the most suitable way.
4. you should provide a answer either in different paragraphs, bullet points or Tables where applicable.

Must Follow the Below Response Instructions:
1. You must only use information from the provided Context Information. 
2. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. 
3. Combine search results together into a coherent answer.
4. Must Cite your answer  using seperate {{Doc ID}} and {{Time Stamp}} notations for reference to exact Context Chunk. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
5. If different results refer to different entities within the same name, write separate answers for each entity.
6. If there is nothing in the context relevant to the question at hand, just say "I donot have any information about it because it isn't provided in my context i do apologize for in convenience." Don't try to make up an answer.
7. Anything between the following 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
8. Respond to the Greeting Messages Properly.
9. User the URLS and Links where needed in response. 

<context>
    {context}
<context/>

Answer in Markdown:

"""
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


refrence_docuemnts_sources=[]

class ChatRequest(BaseModel):
    chatbot_id: str
    temperature: str
    model: str
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None

def create_retriever_chain(llm: BaseLanguageModel, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()).with_config(
            run_name="CondenseQuestion", )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
            ),
            (
                    RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    )
                    | retriever
            ).with_config(run_name="RetrievalChainWithNoHistory"),
        ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    global refrence_docuemnts_sources
    if len(refrence_docuemnts_sources) > 0:
        refrence_docuemnts_sources = []

    formatted_docs = []
    for i, doc in enumerate(docs):
        refrence_docuemnts_sources.append({
            'Context-Information': doc.page_content,
            'MetaData': doc.metadata,
        })
        doc_string = f"<doc id='{i}' metadata='{doc.metadata}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)

def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history
def create_chain(llm: BaseLanguageModel,retriever: BaseRetriever,) -> Runnable:
    retriever_chain = create_retriever_chain(
            llm,
            retriever,
        ).with_config(run_name="FindDocs")
    _context = RunnableMap(
            {
                "context": retriever_chain | format_docs,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
        ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESPONSE_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
            run_name="GenerateResponse",
        )
    return (
                {
                    "question": RunnableLambda(itemgetter("question")).with_config(
                        run_name="Itemgetter:question"
                    ),
                    "chat_history": RunnableLambda(serialize_history).with_config(
                        run_name="SerializeHistory"
                    ),
                }
                | _context

                | response_synthesizer
        )

def Get_Conversation_chain(chatbot_id,temperature,model,question,chat_history):
    ids=[]
    ids.append(str(chatbot_id))
    retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5,"filter":{'chatbot_id': {'$in': ids}}})
    if model == 'gemini-pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True,temperature=float(temperature))
        llm.client = genai.GenerativeModel(model_name='gemini-pro', safety_settings=safety_settings_NONE)
    else:
        llm = ChatOpenAI(
        model=model,
        streaming=True,
        temperature=float(temperature),)
    answer_chain = create_chain(
        llm,
        retriever,
    )
    answer = answer_chain.invoke( {"question": question, "chat_history":chat_history})
    return answer, refrence_docuemnts_sources

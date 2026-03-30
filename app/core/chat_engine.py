import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.utils.config import settings
from app.core.qa_chain import QAChain

logger = logging.getLogger(__name__)

REFORMULATION_SYSTEM_PROMPT = """You are an expert query reformulator for a biomedical hybrid search engine.
Your sole purpose is to take a conversational user input and rewrite it into a standalone, highly specific search query.

CRITICAL RULES:
1. Look at the immediate Chat History to resolve any pronouns (e.g., "it", "they", "this treatment", "these papers", "their findings") or implicit context in the User's newest Input.
2. The output MUST be a standalone string that could be typed into Google Scholar without the listener needing to know the chat history.
3. DO NOT answer the user's question. ONLY output the rewritten query.
4. If the generic User Input does not require history to be understood (e.g., "What is semaglutide?"), just output the original input unchanged.

🔥 RULE 5 - MANDATORY PMID PRESERVATION 🔥:
If the user's follow-up question refers to specific papers, authors, or findings mentioned by the AI in the PREVIOUS TURN (e.g., "Can you summarize their main findings?", "Tell me more about the first paper"), you MUST extract ALL [PMID: XXXXXX] from the AI's historical answer and explicitly append them to your rewritten query in the format 'PMID: XXXXXX'.

Example 1:
History: User: "Does semaglutide help with weight loss?" -> AI: "Yes, it reduces body weight significantly."
New Input: "What are its side effects?"
Output: "What are the side effects of semaglutide in weight loss treatment?"

Example 2:
History: User: "Are there papers on GLP-1 and cardiovascular outcomes?" -> AI: "Yes, Smith discusses MACE reduction [PMID: 40648782] and Jones discusses cardiac safety [PMID: 31594285]."
New Input: "Can you summarize their findings?"
Output: "Can you summarize the findings of Smith [PMID: 40648782] and Jones [PMID: 31594285] regarding GLP-1 cardiovascular outcomes?"
"""

class ChatEngine:
    """Wraps the RAG pipeline in a conversational memory layer that reformulates
    queries based on chat history before passing them to the retrieval engine."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.qa_chain = QAChain()
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0,
        )
        
        self.sessions: Dict[str, List[Any]] = {}
        
        self.reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", REFORMULATION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Rewrite this input to be a standalone query: {input}")
        ])
        
    def _ensure_session(self, session_id: str) -> None:
        if session_id not in self.sessions:
            self.sessions[session_id] = []

    def _trim_session(self, session_id: str) -> None:
        if len(self.sessions[session_id]) > 10:
            self.sessions[session_id] = self.sessions[session_id][-10:]

    def _reformulate_query(self, user_input: str, session_id: str) -> str:
        """Uses the LLM to resolve pronouns and contextualize the user query."""
        chat_history = self.sessions.get(session_id, [])
        if not chat_history:
            return user_input

        logger.info(f"Reformulating query based on history: '{user_input}'")
        chain = self.reformulation_prompt | self.llm
        
        response = chain.invoke({
            "history": chat_history,
            "input": user_input
        })
        
        rewritten_query = response.content.strip()
        logger.info(f"Rewritten Query: '{rewritten_query}'")
        return rewritten_query
        
    def chat(self, user_input: str, session_id: str = "default", filters: dict = None) -> str:
        """Non-streaming conversational RAG."""
        self._ensure_session(session_id)
        standalone_query = self._reformulate_query(user_input, session_id)
        self.sessions[session_id].append(HumanMessage(content=user_input))
        answer, strategy = self.qa_chain.query(standalone_query, filters=filters)
        self.sessions[session_id].append(AIMessage(content=answer))
        self._trim_session(session_id)
        return answer

    def stream_chat(self, user_input: str, session_id: str = "default", filters: dict = None):
        """Streaming conversational RAG. Yields SSE json strings."""
        self._ensure_session(session_id)
        yield json.dumps({"type": "status", "message": "Analyzing query context..."}) + "\n\n"
        standalone_query = self._reformulate_query(user_input, session_id)
        self.sessions[session_id].append(HumanMessage(content=user_input))
        full_answer = ""
        for chunk_str in self.qa_chain.stream_query(standalone_query, filters=filters):
            yield chunk_str
            try:
                data = json.loads(chunk_str.strip())
                if data.get("type") == "token":
                    full_answer += data.get("content", "")
            except json.JSONDecodeError:
                pass
        self.sessions[session_id].append(AIMessage(content=full_answer))
        self._trim_session(session_id)
                
    def clear_history(self, session_id: str = "default"):
        self.sessions.pop(session_id, None)
        logger.info(f"Chat history cleared for session {session_id}.")

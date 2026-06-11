"""Prompt assembly for the RAG generation step."""
from __future__ import annotations
 
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.documents import Document
 
from src.generation.citation_enforcer import build_context_block
 
SYSTEM_TEMPLATE = """\
You are a precise, helpful assistant that answers questions using ONLY the provided context documents.
 
RULES:
1. Every factual claim MUST be followed by a citation in the format [doc-N] where N matches the document number.
2. If multiple documents support a claim, cite all of them: [doc-1][doc-3].
3. If the answer cannot be found in the context, respond with: "I don't have enough information in the provided documents to answer this question."
4. Do NOT use prior knowledge beyond what is in the context.
5. Be concise and direct. Avoid restating the question.
6. If the question asks for a list, format it as a bullet list with citations per bullet.
 
CONTEXT DOCUMENTS:
{context}
"""

CHAT_SYSTEM_TEMPLATE = """\
You are a precise assistant. Use only the provided context. Cite every fact with [doc-N].
 
Conversation history is provided for context only — do not re-answer old questions.
 
CONTEXT:
{context}
"""

def build_rag_messages(
        question: str,
        docs: list[Document],
        chat_history: list[dict] | None = None,
) -> list[BaseMessage]:
    """Assemble the full message list for a RAG query."""
    context = build_context_block(docs)

    if chat_history:
        system_content = CHAT_SYSTEM_TEMPLATE.format(context=context)
        messages: list[BaseMessage] = [SystemMessage(content=system_content)]
        for turn in chat_history[-6:]: # last 3 exchanges
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))
    else:
        system_content = SYSTEM_TEMPLATE.format(context= context)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content= question),
        ]

    return messages

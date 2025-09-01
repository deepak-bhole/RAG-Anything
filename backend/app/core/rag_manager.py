from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from typing import Dict, Any
import ollama
import logging
import inspect

logger = logging.getLogger(__name__)

rag_instances: Dict[str, RAGAnything] = {}
rag_configs: Dict[str, Dict[str, Any]] = {}

def get_rag_instance(config: Dict[str, Any]) -> RAGAnything:
    """
    Initializes or retrieves a cached RAG-Anything instance. This version includes
    robust argument cleaning to prevent keyword conflicts.
    """
    working_dir = config.get("working_dir", "./rag_storage_ollama")

    if working_dir in rag_instances:
        return rag_instances[working_dir]

    rag_config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=config.get("parser", "mineru"),
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs['host'] = config.get("host", "http://localhost:11434")
        kwargs['hashing_kv'] = type('obj', (object,), {'global_config': {'llm_model_name': config.get("llm_model", "llama3:latest")}})()
        return ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        ollama_client = ollama.AsyncClient(host=config.get("host", "http://localhost:11434"))
        
        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            final_messages.extend(history_messages)

        if messages: 
                final_messages.extend(messages)
                user_message = next((m for m in reversed(final_messages) if m.get("role") == "user"), None)
                
                if user_message and 'content' in user_message and isinstance(user_message['content'], list):
                    text_content = ""
                    for item in user_message['content']:
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if "base64," in url:
                                image_data = url.split("base64,")[1]
                    user_message['content'] = text_content
                
                if user_message and image_data:
                    user_message['images'] = [image_data]

        else: 
            user_message = {"role": "user", "content": prompt}
            if image_data:
                user_message["images"] = [image_data]
            final_messages.append(user_message)

        try:
            response = await ollama_client.chat(
                model=config.get("vision_model", "llava:latest"),
                messages=final_messages,
                **kwargs
            )
            return response["message"]["content"]
        finally:
            await ollama_client._client.aclose()

    embedding_func = EmbeddingFunc(
        embedding_dim=config.get("embedding_dim", 768),
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts,
            embed_model=config.get("embedding_model", "nomic-embed-text:latest"),
            host=config.get("host", "http://localhost:11434")
        ),
    )


    rag_instance = RAGAnything(
        config=rag_config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    
    rag_instances[working_dir] = rag_instance
    rag_configs[working_dir] = config
    
    return rag_instance
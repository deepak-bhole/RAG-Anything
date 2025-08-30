#!/usr/bin/env python
"""
Example script demonstrating the integration of RAGAnything with Ollama.

This example shows how to:
1.  Configure RAGAnything to use Ollama for LLM, vision, and embedding models.
2.  Process documents with the multimodal RAG pipeline.
3.  Perform text and multimodal queries using a local Ollama instance.
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
import base64

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Import the Ollama functions from LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env", override=False)

# Import ollama for the vision model function
try:
    import ollama
except ImportError:
    raise ImportError("Ollama is not installed. Please install it with 'pip install ollama'")


def configure_logging():
    """Configure logging for the application"""
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_ollama_example.log"))

    print(f"\nRAGAnything Ollama example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {"formatter": "default", "class": "logging.StreamHandler", "stream": "ext://sys.stderr"},
            "file": {
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "lightrag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        },
    })
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    host: str,
    llm_model: str,
    vision_model: str,
    embedding_model: str,
    embedding_dim: int,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything using Ollama models.
    """
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage_ollama",
            parser=parser,
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        logger.info("RAGAnything Configuration:" + os.getenv("LIGHTRAG_KV_STORAGE"))

        # Define LLM model function for Ollama
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # Pass host and a mock hashing_kv to satisfy the ollama_model_complete interface
            kwargs['host'] = host
            kwargs['hashing_kv'] = type('obj', (object,), {'global_config': {'llm_model_name': llm_model}})()
            return ollama_model_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs,
            )

        async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
            ollama_client = ollama.AsyncClient(host=host)
            
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                final_messages.extend(history_messages)

            # Handle different input formats
            if messages: # RAGAnything might pass a pre-formatted messages list
                    final_messages.extend(messages)
                    # Find the last user message to potentially add an image to it
                    # This part assumes the last message is the user prompt with the image context
                    user_message = next((m for m in reversed(final_messages) if m.get("role") == "user"), None)
                    
                    if user_message and 'content' in user_message and isinstance(user_message['content'], list):
                        # Extract text and image from OpenAI-style message format
                        text_content = ""
                        # This loop rebuilds the text and finds the image
                        for item in user_message['content']:
                            if item.get("type") == "text":
                                text_content += item.get("text", "")
                            elif item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if "base64," in url:
                                    # Set image_data from the message content
                                    image_data = url.split("base64,")[1]
                        # Replace the list-based content with simple text for Ollama
                        user_message['content'] = text_content
                    
                    # Add the extracted image data to the user message for Ollama
                    if user_message and image_data:
                        user_message['images'] = [image_data]

            else: # Standard prompt and image_data format
                # CORRECTED PART: Create the user message and add the image data to it directly
                user_message = {"role": "user", "content": prompt}
                if image_data:
                    user_message["images"] = [image_data]
                final_messages.append(user_message)

            try:
                # CORRECTED PART: The invalid 'images' keyword argument is removed from the call
                response = await ollama_client.chat(
                    model=vision_model,
                    messages=final_messages,
                    **kwargs
                )
                return response["message"]["content"]
            finally:
                await ollama_client._client.aclose()

        # Define embedding function for Ollama
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embedding_model,
                host=host,
            ),
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        logger.info("\nQuerying processed document with Ollama:")

        text_queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]
        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

        logger.info("\n[Multimodal Query]: Analyzing performance data in context of document")
        # multimodal_result = await rag.aquery_with_multimodal(
        #     "Compare this performance data with any similar results mentioned in the document",
        #     multimodal_content=[
        #         {"type": "table", "table_data": "Method,Accuracy,Processing_Time\nRAGAnything,95.2%,120ms\nTraditional_RAG,87.3%,180ms\nBaseline,82.1%,200ms", "table_caption": "Performance comparison results"}
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"Answer: {multimodal_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="RAGAnything Ollama Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument("--working_dir", "-w", default="./rag_storage_ollama", help="Working directory path")
    parser.add_argument("--output", "-o", default="./output_ollama", help="Output directory path")
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Ollama host URL")
    parser.add_argument("--llm-model", default=os.getenv("OLLAMA_LLM_MODEL", "llama3:latest"), help="Ollama model for text tasks")
    parser.add_argument("--vision-model", default=os.getenv("OLLAMA_VISION_MODEL", "llava:latest"), help="Ollama model for vision tasks")
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"), help="Ollama model for embeddings")
    parser.add_argument("--embedding-dim", type=int, default=os.getenv("OLLAMA_EMBEDDING_DIM", 768), help="Dimension of the embedding model")
    parser.add_argument("--parser", default=os.getenv("PARSER", "mineru"), help="Document parser to use (e.g., mineru, docling)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    asyncio.run(
        process_with_rag(
            file_path=args.file_path,
            output_dir=args.output,
            host=args.host,
            llm_model=args.llm_model,
            vision_model=args.vision_model,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            working_dir=args.working_dir,
            parser=args.parser,
        )
    )

if __name__ == "__main__":
    configure_logging()
    print("RAGAnything Ollama Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline using Ollama")
    print("=" * 30)
    main()
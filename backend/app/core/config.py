from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """
    Application settings.
    """
    PROJECT_NAME: str = "RAG-Anything API"
    API_V1_STR: str = "/api/v1"
    
    # CORS Origins - Update with your frontend URL
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    class Config:
        case_sensitive = True

settings = Settings()
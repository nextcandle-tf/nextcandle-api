"""Configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Model
    model_path: str = "./models/encoder.pth"
    embeddings_path: str = "./models/embeddings.pkl"
    embedding_dim: int = 64
    use_gpu: bool = True

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:4000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

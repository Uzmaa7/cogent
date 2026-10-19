from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, model_validator



class ModelConfig(BaseModel):
    name: str = "openai/gpt-oss-120b"
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256_000


class Config(BaseModel):

    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)

    max_turns: int = 100
    

    developer_instructions: str | None = None
    user_instructions: str | None = None

    debug: bool = False



    @property
    def api_key(self) -> str | None:
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY")
        # print(f"DEBUG [Config.api_key]: Raw key from env -> {repr(key)}")
        if key:
            cleaned = key.strip().strip('"').strip("'")
            # print(f"DEBUG [Config.api_key]: Cleaned key length -> {len(cleaned)}, prefix -> {cleaned[:5]}...")
            return cleaned
        return None
        

    @property
    def base_url(self) -> str | None:
        return os.environ.get("BASE_URL")

    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self.model.name = value

    

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: str) -> None:
        self.model.temperature = value

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            errors.append("No API key found. Set GROQ_API_KEY environment variable in .env file.")

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors

   
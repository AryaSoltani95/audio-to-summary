from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from openai import OpenAI
from dataclasses import dataclass
import gc
import torch
import sys
from pathlib import Path


class BaseTranscriber(ABC):
    load_dotenv(override=True)
    data_folder = str(Path(__file__).parent.parent / "data")
    print(f"data folder: {data_folder}")

    """force all classes to implement transcribe method"""
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        pass


class LocalTranscriber(BaseTranscriber):
    def __init__(self):
        super().__init__()
        """transcribe audio using local model the default model is whisper-medium.en"""
        self.model_name: str = "openai/whisper-medium.en"
        self.transcript: str = None

    def token(self) -> None:
        """load token from .env file"""
        try:
            load_dotenv()
            self.token = os.getenv("HUGGINGFACE_TRANSCRIBER_TOKEN")
        except Exception as e:
            print(e)

    def login(self) -> None:
        """login to huggingface hub"""
        try:
            login(self.token)
        except Exception as e:
            print(e)

    def model(self) -> None:
        """select model"""
        try:
            self.result = pipeline("automatic-speech-recognition", model=self.model_name, dtype=torch.float16, device='cuda', return_timestamps=True)
        except Exception as e:
            print(e)

    def memory_cleaner(self) -> None:
        """clean memory(RAM and GPU)"""
        try:
            del self.result
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)

    def transcribe(self, audio_path: str) -> None:
        """transcribe audio using local model"""
        self.token()
        self.login()
        self.model()
        try:
            self.transcript = self.result(fr"{self.data_folder}/{audio_path}")
        except Exception as e:
            print(e)
        self.memory_cleaner()
        return self.transcript["text"]


class CloudTranscriber(BaseTranscriber):
    "this class uses cloud model to transcribe audio"
    def __init__(self):
        super().__init__()
        self.transcript:str = None
        self.model: str = None
        self.model_name: str = "gapgpt/whisper-1"

    def openai(self) -> None:
        """get url and token from .env file and create openai client object"""
        try:
            load_dotenv(override=True)
            url = os.getenv("TRANSCRIBER_URL")
            assert url, "TRANSCRIBER_URL is not set in .env file"

            token = os.getenv("TRANSCRIBER_TOKEN")
            assert token, "TRANSCRIBER_TOKEN is not set in .env file"

            self.client = OpenAI(base_url=url, api_key=token)
        except Exception as e:
            print(e)
            sys.exit(1)

    def transcribe(self, audio_file: str) -> str:
        """transcribe audio using openai api"""
        self.openai()
        try:
            # load_dotenv(override=True)
            model = os.getenv("CLOUD_TRANSCRIBER_MODEL")
            self.openai()
            with open(f"{self.data_folder}/{audio_file}", "rb") as f:
                result = self.client.audio.transcriptions.create(
                file=f,
                model=model,
                response_format="text",
                    )
        except Exception as e:
            print(e)
            sys.exit(1)

        with open(fr"{self.data_folder} / test.txt", "w") as f:
                f.write(result)
                
        return result["text"]
        

if __name__ == "__main__":
    transcriber = CloudTranscriber()
    result = transcriber.transcribe("test.mp3")
    
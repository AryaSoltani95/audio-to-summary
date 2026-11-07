from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from openai import OpenAI
import gc
import torch
import sys
from IPython.display import display, Markdown


class BaseClass(ABC):

    """force all classes to implement the __init__ method"""
    @abstractmethod
    def __init__(self, transcipt: str):
        self.transcript = transcipt
        self.data_folder = os.environ.get("DATA_FOLDER")
        self.model = None

    """force all classes to implement the summarize method"""
    @abstractmethod
    def summarizer(self):
        pass


class LocalSummarizer(BaseClass):
    def __init__(self, transcript: str):
        super().__init__(transcript)
        self.model = os.environ.get("SUMMARIZER_MODEL")

    def prompt_generator(self, system_prompt=None, user_prompt=None):
        if not system_prompt:
            system_prompt = """
                You are an analytical summarization engine for conversation transcripts.

                Your role:
                1. Extract executive summaries, key decisions, and critical discussion points.
                2. Track participants and explicitly map tasks/action items to the responsible individuals.
                3. Highlight risks, blockers, and dates/deadlines mentioned.

                Operational constraints:
                - Rely strictly on facts stated in the transcript. Never infer or invent names, deadlines, or tasks.
                - If speaker names are unavailable, use existing labels (e.g., "Speaker 1").
                - Format all outputs strictly using the following Markdown schema:

                ### Executive Summary
                [Brief high-level overview]

                ### Key Points & Decisions
                - [Key point / decision]

                ### Action Items & Assignees
                - **Assignee**: [Task description] | Deadline: [Value or "None mentioned"]

                ### Critical Context & Follow-ups
                - [Pending issues, deadlines, or risks]
            """

        if not user_prompt:
            user_prompt = f"""
                Summarize the following conversation transcript, including key decisions, 
                action items, and critical discussion points.
                Also, track participants and map tasks/action items to the responsible individuals.
                Highlight risks, blockers, and dates/deadlines mentioned.
                conversation transcript as below :
               {self.transcript}
                """

        self.prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def quantize_provider(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def model_provider(self):
        assert self.model, "You dont choose model, please fill the summarization model part in pyproject.toml"
        self.prompt_generator()
        self.quantization_config = None
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer.apply_chat_template(self.prompt, return_tensors="pt", return_dict=True).to("cuda")
        self.input = inputs["input_ids"]
        self.streamer = TextStreamer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(self.model, quantization_config=self.quantization_config, device_map="cuda")

    def summarizer(self):
        self.quantize_provider()
        self.model_provider()
        outputs = self.model.generate(self.input, max_new_tokens=1000, streamer=self.streamer)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        display(Markdown(summary))
        return summary


class CloudTranscriber(BaseClass):
    "this class uses cloud model to transcribe audio"
    def __init__(self):
        super().__init__()
        self.model = os.environ.get("CLOUD_SUMMARIZER_MODEL")

    def openai(self) -> None:
        """get url and token from .env file and create openai client object"""
        try:
            load_dotenv(override=True)
            url = os.getenv("SUMMARIZER_URL")
            assert url, "TRANSCRIBER_URL is not set in .env file"

            token = os.getenv("SUMMARIZER_TOKEN")
            assert token, "TRANSCRIBER_TOKEN is not set in .env file"

            self.client = OpenAI(base_url=url, api_key=token)
        except Exception as e:
            print(e)
            sys.exit(1)

    def summarizer(self, audio_file: str) -> str:
        """transcribe audio using openai api"""
        try:
            load_dotenv(override=True)
            model = os.getenv("CLOUD_SUMMARIZER_MODEL")
            self.openai()
            result = self.client.audio.transcriptions.create(
                file=f"{self.data_folder}/{audio_file}",
                model=model,
                response_format="text",
            )
        except Exception as e:
            print(e)

        return result["text"]
        


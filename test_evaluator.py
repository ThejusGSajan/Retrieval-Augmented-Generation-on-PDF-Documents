import os
os.environ['HF_HOME'] = 'D:\\.cache\\huggingface\\'
from dotenv import load_dotenv
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepeval
from deepeval import assert_test
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")
login(token=hf_token)

class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cpu" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)

dataset = EvaluationDataset()
dataset.add_test_cases_from_json_file(
    file_path="evaluation_data.json",
    input_key_name="input",
    actual_output_key_name="output",
    expected_output_key_name="ground_truth"
)

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
@pytest.mark.asyncio
def test_chat_model(test_case: LLMTestCase):
    correctness_metric = GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "Heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are not okay"
        ],
        evaluation_params=[
          LLMTestCaseParams.INPUT,
          LLMTestCaseParams.ACTUAL_OUTPUT,
          LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        model="mistral_7b"
    )
    assert_test(test_case, [correctness_metric])

@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")
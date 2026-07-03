from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You solve numeric math and unit-conversion questions. Be concise and accurate. "
                    "Return exactly one line in this format: <answer>NUMBER</answer>. "
                    "Do not include units, words, or extra text outside the tag. "
                    "Use decimal numbers when needed."
                ),
            },
            {"role": "user", "content": "Can you change 2 hour to its equivalent in min?"},
            {
                "role": "assistant",
                "content": "<answer>120</answer>",
            },
            {"role": "user", "content": "Please convert 6 mi/h into m/s."},
            {
                "role": "assistant",
                "content": "<answer>2.68224</answer>",
            },
            {"role": "user", "content": question},
        ]

        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})

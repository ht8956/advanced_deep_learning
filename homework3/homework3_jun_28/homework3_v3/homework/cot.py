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
                    "You are a helpful math tutor. Solve the problem briefly and correctly. "
                    "Be concise. Show minimal reasoning, then give only the final numeric value "
                    "inside <answer></answer>."
                ),
            },
            {"role": "user", "content": "What is 7 * 8 + 6?"},
            {
                "role": "assistant",
                "content": "7 * 8 = 56, then 56 + 6 = 62. <answer>62</answer>",
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

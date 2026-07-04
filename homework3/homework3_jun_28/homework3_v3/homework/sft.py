from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    rounded_answer = round(float(answer), 3)
    return {
        "question": prompt,
        "answer": f"<answer>{rounded_answer}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str = "sft_runs",
    **kwargs,
):
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

    lora_rank = int(kwargs.pop("r", 8))
    lora_alpha = int(kwargs.pop("lora_alpha", 4 * lora_rank))

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_cfg)

    # Avoid known issues with gradient checkpointing on accelerated backends.
    if llm.device != "cpu" and hasattr(llm.model, "enable_input_require_grads"):
        llm.model.enable_input_require_grads()

    llm.model.config.use_cache = False

    trainset = Dataset("train")
    tokenized_dataset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    train_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=float(kwargs.pop("learning_rate", 1e-4)),
        num_train_epochs=float(kwargs.pop("num_train_epochs", 5)),
        per_device_train_batch_size=int(kwargs.pop("per_device_train_batch_size", 32)),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        logging_steps=int(kwargs.pop("logging_steps", 10)),
        save_strategy="epoch",
        weight_decay=float(kwargs.pop("weight_decay", 0.01)),
    )

    trainer = Trainer(
        model=llm.model,
        args=train_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

    final_model_dir = Path(__file__).parent / "sft_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))

    test_model(str(final_model_dir))


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})

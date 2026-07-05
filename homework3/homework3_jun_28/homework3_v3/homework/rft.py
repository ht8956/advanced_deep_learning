from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "rft_runs",
    **kwargs,
):
    import json
    from pathlib import Path

    from .datagen import generate_dataset
    from .sft import TokenizedDataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    llm = BaseLLM()

    lora_rank = int(kwargs.pop("r", 24))
    lora_alpha = int(kwargs.pop("lora_alpha", 4 * lora_rank))

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm.model = get_peft_model(llm.model, lora_cfg)

    if llm.device != "cpu" and hasattr(llm.model, "enable_input_require_grads"):
        llm.model.enable_input_require_grads()

    llm.model.config.use_cache = False

    repo_root = Path(__file__).parent.parent
    rft_data_path = repo_root / "data" / "rft.json"
    if not rft_data_path.exists():
        generate_dataset(
            str(rft_data_path),
            oversample=int(kwargs.pop("oversample", 10)),
            temperature=float(kwargs.pop("temperature", 0.6)),
        )

    with rft_data_path.open() as f:
        trainset = json.load(f)

    def format_example(question: str, _answer: float, reasoning: str) -> dict[str, str]:
        return {
            "question": question,
            "answer": reasoning,
        }

    tokenized_dataset = TokenizedDataset(llm.tokenizer, trainset, format_example)

    train_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        learning_rate=float(kwargs.pop("learning_rate", 3e-4)),
        num_train_epochs=float(kwargs.pop("num_train_epochs", 5)),
        per_device_train_batch_size=int(kwargs.pop("per_device_train_batch_size", 32)),
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=llm.model,
        args=train_args,
        train_dataset=tokenized_dataset,
    )
    trainer.train()

    final_model_dir = Path(__file__).parent / "rft_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))

    test_model(str(final_model_dir))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})

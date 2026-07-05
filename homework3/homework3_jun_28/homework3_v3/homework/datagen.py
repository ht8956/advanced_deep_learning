def generate_dataset(
    output_json: str = "data/rft.json",
    oversample: int = 20,
    temperature: float = 0.4,
    checkpoint: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
):
    import json
    from pathlib import Path

    from tqdm import tqdm

    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    repo_root = Path(__file__).parent.parent
    out_path = Path(output_json)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = CoTModel(checkpoint=checkpoint)
    dataset = Dataset("train")

    output_data: list[list[str | float]] = []
    for question, correct_answer in tqdm(dataset.data, total=len(dataset), desc="Selecting RFT samples"):
        prompt = model.format_prompt(question)
        # Generate per question to keep memory bounded on smaller GPUs.
        candidate_generations = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

        selected_generation = None
        for generation in candidate_generations:
            parsed_answer = model.parse_answer(generation)
            if is_answer_valid(parsed_answer, correct_answer):
                selected_generation = generation.strip()
                break

        if selected_generation is not None:
            output_data.append([question, correct_answer, selected_generation])

    with out_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"saved {len(output_data)} samples to {out_path}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)

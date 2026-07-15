from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
    find_view_image,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    kart_objects = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not kart_objects:
        return []

    ego_kart = next((kart for kart in kart_objects if kart["is_center_kart"]), kart_objects[0])
    other_karts = [kart for kart in kart_objects if kart["instance_id"] != ego_kart["instance_id"]]
    track_name = extract_track_info(info_path)

    captions = [
        f"There are {len(kart_objects)} karts in the scenario.",
        f"The track is {track_name}.",
    ]

    if ego_kart.get("kart_name"):
        captions.insert(0, f"{ego_kart['kart_name']} is the ego car.")

    for kart in other_karts:
        if not kart.get("kart_name"):
            continue

        delta_x = kart["center"][0] - ego_kart["center"][0]
        delta_y = kart["center"][1] - ego_kart["center"][1]

        horizontal_position = "left" if delta_x < 0 else "right"
        depth_position = "front" if delta_y < 0 else "back"

        captions.append(f"{kart['kart_name']} is {depth_position} and to the {horizontal_position} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_split(split: str = "train", data_dir: str | None = None) -> list[str]:
    """
    Generate caption files for a dataset split.

    Args:
        split: Dataset split to process. Use train for CLIP training data.
        data_dir: Optional override for dataset root directory.

    Returns:
        List of generated caption file paths.
    """
    dataset_root = Path(data_dir) if data_dir is not None else Path(__file__).parent.parent / "data"
    split_dir = dataset_root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    generated_files = []
    info_files = sorted(split_dir.glob("*_info.json"))

    for info_path in info_files:
        base_name = info_path.stem.replace("_info", "")
        all_captions = []

        with open(info_path) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))
        for view_index in range(num_views):
            image_path = find_view_image(info_path, view_index)
            if image_path is None:
                continue

            captions = generate_caption(str(info_path), view_index)
            for caption in captions:
                all_captions.append(
                    {
                        "caption": caption,
                        "image_file": f"{split}/{image_path.name}",
                    }
                )

        if not all_captions:
            print(f"Skipped {info_path}: no captions were generated")
            continue

        output_path = split_dir / f"{base_name}_captions.json"
        with open(output_path, "w") as f:
            json.dump(all_captions, f, indent=2)

        generated_files.append(str(output_path))
        print(f"Wrote {len(all_captions)} captions to {output_path}")

    return generated_files


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_split})


if __name__ == "__main__":
    main()

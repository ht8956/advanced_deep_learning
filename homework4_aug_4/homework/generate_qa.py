import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def find_view_image(info_path: Path, view_index: int) -> Path | None:
    """
    Find the corresponding rendered image for a frame/view using common extensions.
    """
    base_name = info_path.stem.replace("_info", "")
    for extension in ("jpg", "jpeg", "png"):
        candidate = info_path.parent / f"{base_name}_{view_index:02d}_im.{extension}"
        if candidate.exists():
            return candidate
    return None


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)

    detections = info.get("detections", [])
    if view_index < 0 or view_index >= len(detections):
        return []

    def iter_dicts(value):
        if isinstance(value, dict):
            yield value
            for nested_value in value.values():
                yield from iter_dicts(nested_value)
        elif isinstance(value, list):
            for item in value:
                yield from iter_dicts(item)

    kart_names = {}

    # Prefer explicit kart lists when present to avoid noisy id->name mappings.
    for list_key in ("karts", "players", "racers"):
        entries = info.get(list_key)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            id_value = None
            for id_key in ("track_id", "instance_id", "kart_id", "id"):
                if id_key in item:
                    try:
                        id_value = int(item[id_key])
                    except (TypeError, ValueError):
                        id_value = None
                    break
            if id_value is None:
                continue

            name_value = None
            for name_key in ("kart_name", "kart", "name"):
                candidate = item.get(name_key)
                if isinstance(candidate, str) and candidate.strip():
                    name_value = candidate.strip().lower()
                    break
            if name_value is not None:
                kart_names[id_value] = name_value

    for item in iter_dicts(info):
        id_value = None
        for id_key in ("track_id", "instance_id", "kart_id", "id"):
            if id_key in item:
                try:
                    id_value = int(item[id_key])
                except (TypeError, ValueError):
                    id_value = None
                break

        if id_value is None:
            continue

        name_value = None
        for name_key in ("kart_name", "kart", "name"):
            candidate = item.get(name_key)
            if isinstance(candidate, str) and candidate.strip():
                name_value = candidate.strip().lower()
                break

        if name_value is None:
            continue

        # Only accept recursive fallbacks when the dict itself is kart-specific.
        if any(key in item for key in ("kart_name", "kart", "kart_id")):
            kart_names[id_value] = name_value

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    image_center = (img_width / 2, img_height / 2)
    kart_objects = []

    for detection in detections[view_index]:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = float(x1) * scale_x
        y1_scaled = float(y1) * scale_y
        x2_scaled = float(x2) * scale_x
        y2_scaled = float(y2) * scale_y

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        center = ((x1_scaled + x2_scaled) / 2, (y1_scaled + y2_scaled) / 2)
        kart_objects.append(
            {
                "instance_id": track_id,
                "kart_name": kart_names.get(track_id),
                "center": center,
                "distance_to_center": (center[0] - image_center[0]) ** 2 + (center[1] - image_center[1]) ** 2,
            }
        )

    if not kart_objects:
        return []

    center_index = min(range(len(kart_objects)), key=lambda index: kart_objects[index]["distance_to_center"])
    for index, kart in enumerate(kart_objects):
        kart["is_center_kart"] = index == center_index
        del kart["distance_to_center"]

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)

    def normalize_track_name(value):
        if not isinstance(value, str):
            return None
        track_name = Path(value.strip()).name
        return track_name.lower() if track_name else None

    for key in ("track_name", "track", "course_name", "course", "map_name", "map", "level_name", "level"):
        value = info.get(key)
        if isinstance(value, str):
            track_name = normalize_track_name(value)
            if track_name:
                return track_name
        if isinstance(value, dict):
            for nested_key in ("name", "id", "track_name"):
                track_name = normalize_track_name(value.get(nested_key))
                if track_name:
                    return track_name

    stack = [info]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key in {"track_name", "track", "course_name", "course", "map_name", "map", "level_name", "level"}:
                    track_name = normalize_track_name(value)
                    if track_name:
                        return track_name
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)

    return "unknown"


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    kart_objects = extract_kart_objects(info_path, view_index, img_width=img_width, img_height=img_height)
    if not kart_objects:
        return []

    # Staff tip: the ego car is the kart closest to the center of the image.
    ego_kart = next((kart for kart in kart_objects if kart["is_center_kart"]), kart_objects[0])
    other_karts = [kart for kart in kart_objects if kart["instance_id"] != ego_kart["instance_id"]]
    track_name = extract_track_info(info_path)

    qa_pairs = [
        {"question": "How many karts are there in the scenario?", "answer": str(len(kart_objects))},
        {"question": "What track is this?", "answer": track_name},
    ]

    if ego_kart["kart_name"]:
        qa_pairs.insert(0, {"question": "What kart is the ego car?", "answer": ego_kart["kart_name"]})

    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0

    for kart in other_karts:
        delta_x = kart["center"][0] - ego_kart["center"][0]
        delta_y = kart["center"][1] - ego_kart["center"][1]

        horizontal_position = "left" if delta_x < 0 else "right"
        depth_position = "front" if delta_y < 0 else "back"
        relative_position = f"{depth_position} and {horizontal_position}"

        if horizontal_position == "left":
            left_count += 1
        else:
            right_count += 1

        if depth_position == "front":
            front_count += 1
        else:
            behind_count += 1

        if kart["kart_name"]:
            qa_pairs.extend(
                [
                    {
                        "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                        "answer": horizontal_position,
                    },
                    {
                        "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                        "answer": depth_position,
                    },
                    {
                        "question": f"Where is {kart['kart_name']} relative to the ego car?",
                        "answer": relative_position,
                    },
                ]
            )

    qa_pairs.extend(
        [
            {"question": "How many karts are to the left of the ego car?", "answer": str(left_count)},
            {"question": "How many karts are to the right of the ego car?", "answer": str(right_count)},
            {"question": "How many karts are in front of the ego car?", "answer": str(front_count)},
            {"question": "How many karts are behind the ego car?", "answer": str(behind_count)},
        ]
    )

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    image_file = find_view_image(info_path, view_index)
    if image_file is None:
        raise FileNotFoundError(
            f"No image found for view {view_index} near {info_path}. Expected *_im.jpg, *_im.jpeg, or *_im.png"
        )

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_split(split: str = "train", data_dir: str | None = None) -> list[str]:
    """
    Generate QA pair files for a dataset split.

    Args:
        split: Dataset split to process. Use `train` for homework training data.
        data_dir: Optional override for the dataset root directory.

    Returns:
        List of generated QA pair file paths.
    """
    dataset_root = Path(data_dir) if data_dir is not None else Path(__file__).parent.parent / "data"
    split_dir = dataset_root / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    generated_files = []
    info_files = sorted(split_dir.glob("*_info.json"))

    for info_path in info_files:
        base_name = info_path.stem.replace("_info", "")
        all_qa_pairs = []

        with open(info_path) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))
        for view_index in range(num_views):
            image_path = find_view_image(info_path, view_index)
            if image_path is None:
                continue

            image_name = image_path.name

            qa_pairs = generate_qa_pairs(str(info_path), view_index)
            for qa_pair in qa_pairs:
                all_qa_pairs.append(
                    {
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "image_file": f"{split}/{image_name}",
                    }
                )

        if not all_qa_pairs:
            print(f"Skipped {info_path}: no QA pairs were generated")
            continue

        output_path = split_dir / f"{base_name}_qa_pairs.json"
        with open(output_path, "w") as f:
            json.dump(all_qa_pairs, f, indent=2)

        generated_files.append(str(output_path))
        print(f"Wrote {len(all_qa_pairs)} QA pairs to {output_path}")

    return generated_files


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate": generate_split})


if __name__ == "__main__":
    main()

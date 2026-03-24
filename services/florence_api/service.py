import torch
from PIL import Image

from contracts import Detection
from models import FlorenceResources


def detect_with_florence(
    image: Image.Image,
    resources: FlorenceResources,
) -> list[Detection]:
    prompt = "<OD>"
    inputs = resources.processor(text=prompt, images=image, return_tensors="pt").to(
        resources.device
    )

    with torch.no_grad():
        generated_ids = resources.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=False,
        )

    generated_text = resources.processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]
    parsed = resources.processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height),
    )
    task_output = parsed.get(prompt, {}) if isinstance(parsed, dict) else {}
    labels = task_output.get("labels", [])
    bboxes = task_output.get("bboxes", [])

    detections: list[Detection] = []
    for label, bbox in zip(labels, bboxes):
        if (
            not isinstance(label, str)
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        detections.append(
            Detection(
                label=label.strip(),
                bbox=[float(v) for v in bbox],
            )
        )
    return detections
import math
import re

import torch
from PIL import Image

from contracts import Detection
from models import FlorenceResources


def _find_matched_token_indices(
    current_span: tuple[int, int],
    token_spans: list[tuple[int, int]],
) -> list[int]:
    return [
        index
        for index, span in enumerate(token_spans)
        if not (span[1] <= current_span[0] or span[0] >= current_span[1])
    ]


def _extract_detection_scores_from_transition(
    resources: FlorenceResources,
    sequence: torch.Tensor,
    transition_scores: torch.Tensor,
) -> list[float]:
    post_processor = getattr(resources.processor, "post_processor", None)
    if post_processor is None or not hasattr(post_processor, "decode_with_spans"):
        raise ValueError("Processor post-processor cannot decode spans for score extraction.")

    sequence_list = sequence.tolist()
    bos_token_id = getattr(resources.processor.tokenizer, "bos_token_id", None)
    if sequence_list and bos_token_id is not None and sequence_list[0] == bos_token_id:
        sequence_list = sequence_list[1:]

    score_values = transition_scores.tolist()

    if len(score_values) + 1 == len(sequence_list):
        # For encoder-decoder generation, transition scores can omit the first decoded token.
        sequence_list = sequence_list[1:]
    if len(score_values) == len(sequence_list) + 1:
        score_values = score_values[1:]
    if len(score_values) != len(sequence_list):
        raise ValueError(
            f"Transition score length mismatch: scores={len(score_values)} sequence={len(sequence_list)}."
        )
    text, spans = post_processor.decode_with_spans(sequence_list)

    phrase_pattern = r"([^<]+(?:<loc_\d+>){4,})"
    phrase_text_pattern = r"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)"
    box_pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"

    phrases = re.findall(phrase_pattern, text)
    cur_span = 0
    scores: list[float] = []
    for phrase_text in phrases:
        phrase_text_strip = phrase_text.replace("<ground>", "", 1)
        phrase_text_strip = phrase_text_strip.replace("<obj>", "", 1)
        if phrase_text_strip == "":
            cur_span += len(phrase_text)
            continue

        phrase_match = re.search(phrase_text_pattern, phrase_text_strip)
        if phrase_match is None:
            cur_span += len(phrase_text)
            continue

        bboxes_parsed = list(re.finditer(box_pattern, phrase_text))
        if len(bboxes_parsed) == 0:
            cur_span += len(phrase_text)
            continue

        bbox_end_spans = [bbox_match.span(0) for bbox_match in bboxes_parsed]
        for span_start, span_end in bbox_end_spans:
            token_indices = _find_matched_token_indices(
                (span_start + cur_span, span_end + cur_span),
                spans,
            )
            if len(token_indices) == 0:
                raise ValueError("Unable to align bbox tokens with transition scores.")
            loc_scores = [score_values[token_i] for token_i in token_indices]
            avg_log_score = sum(loc_scores) / len(loc_scores)
            scores.append(float(math.exp(avg_log_score)))

        cur_span += len(phrase_text)

    return scores


def detect_with_florence(
    image: Image.Image,
    resources: FlorenceResources,
) -> list[Detection]:
    prompt = "<OD>"
    inputs = resources.processor(text=prompt, images=image, return_tensors="pt").to(
        resources.device
    )

    with torch.no_grad():
        generated = resources.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    transition_scores = resources.model.compute_transition_scores(
        sequences=generated.sequences,
        scores=generated.scores,
        beam_indices=generated.beam_indices,
    )

    sequence = generated.sequences[0]
    sequence_scores = transition_scores[0]
    image_size = (image.width, image.height)
    try:
        parsed = resources.processor.post_process_generation(
            sequence=sequence,
            transition_beam_score=sequence_scores,
            task=prompt,
            image_size=image_size,
        )
    except TypeError:
        parsed = resources.processor.post_process_generation(
            sequence=sequence,
            task=prompt,
            image_size=image_size,
        )

    task_output = parsed.get(prompt, {}) if isinstance(parsed, dict) else {}
    labels = task_output.get("labels")
    bboxes = task_output.get("bboxes")
    scores = task_output.get("scores")

    if scores is None:
        scores = _extract_detection_scores_from_transition(
            resources=resources,
            sequence=sequence,
            transition_scores=sequence_scores,
        )

    if not isinstance(labels, list) or not isinstance(bboxes, list) or not isinstance(scores, list):
        raise ValueError("Post-processing must return labels, bboxes and scores lists.")
    if not (len(labels) == len(bboxes) == len(scores)):
        raise ValueError(
            f"Post-processing length mismatch: labels={len(labels)} bboxes={len(bboxes)} scores={len(scores)}."
        )
    if len(labels) == 0:
        return []

    detections: list[Detection] = []
    for label, bbox, score in zip(labels, bboxes, scores):
        if not isinstance(label, str):
            raise ValueError("Detected label must be a string.")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("Detected bbox must be a 4-element list.")
        if not isinstance(score, (float, int)):
            raise ValueError("Detected score must be numeric.")

        detections.append(
            Detection(
                label=label.strip(),
                bbox=[float(v) for v in bbox],
                score=float(score),
            )
        )

    return detections
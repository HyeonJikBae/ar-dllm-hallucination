#!/usr/bin/env python3
"""Run the PopQA knowledge screen with instructed short-answer QA prompts."""

from __future__ import annotations

import sys

import run_prompt_x_experiment as pipeline


QA_INSTRUCTION = (
    "Answer the following question with only the short factual answer. "
    "Do not provide any explanation."
)


def qa(question: str) -> str:
    """Use one identical instructed QA completion frame for every model."""
    return f"{QA_INSTRUCTION}\n\nQuestion: {question}\nAnswer:"


# Five semantically equivalent question paraphrases per verified PopQA relation.
# These are a separate prompt-format condition; they do not replace or mix with
# the five declarative completion prompts used in the original experiment.
QA_RELATION_TEMPLATE_SETS: dict[str, tuple[str, ...]] = {
    "author": (
        qa("Who wrote {subject}?"),
        qa("Who is the author of {subject}?"),
        qa("Which person wrote {subject}?"),
        qa("Who was {subject} written by?"),
        qa("What writer authored {subject}?"),
    ),
    "capital": (
        qa("What is the capital of {subject}?"),
        qa("Which city is the capital of {subject}?"),
        qa("What city serves as the capital of {subject}?"),
        qa("Which place is {subject}'s capital?"),
        qa("What is the capital city of {subject}?"),
    ),
    "capital of": (
        qa("What place is {subject} the capital of?"),
        qa("{subject} is the capital of which place?"),
        qa("For which region or country is {subject} the capital?"),
        qa("Which place has {subject} as its capital?"),
        qa("What territory does {subject} serve as the capital of?"),
    ),
    "color": (
        qa("What color is {subject}?"),
        qa("What is the color of {subject}?"),
        qa("Which color is associated with {subject}?"),
        qa("What is {subject}'s identifying color?"),
        qa("Which color does {subject} have?"),
    ),
    "composer": (
        qa("Who composed {subject}?"),
        qa("Who is the composer of {subject}?"),
        qa("Which person composed {subject}?"),
        qa("Who wrote the music for {subject}?"),
        qa("What composer created the music for {subject}?"),
    ),
    "country": (
        qa("In which country is {subject} located?"),
        qa("What country contains {subject}?"),
        qa("Which nation is {subject} located in?"),
        qa("{subject} is situated in which country?"),
        qa("What is the country associated with {subject}'s location?"),
    ),
    "director": (
        qa("Who directed {subject}?"),
        qa("Who was the director of {subject}?"),
        qa("Which person directed {subject}?"),
        qa("What filmmaker directed {subject}?"),
        qa("Who served as the director of {subject}?"),
    ),
    "father": (
        qa("Who is the father of {subject}?"),
        qa("Who is {subject}'s father?"),
        qa("Which person is identified as {subject}'s father?"),
        qa("Who is the paternal parent of {subject}?"),
        qa("What is the name of {subject}'s father?"),
    ),
    "genre": (
        qa("What is the genre of {subject}?"),
        qa("Which genre does {subject} belong to?"),
        qa("How is the genre of {subject} classified?"),
        qa("What genre is associated with {subject}?"),
        qa("In which genre is {subject} classified?"),
    ),
    "mother": (
        qa("Who is the mother of {subject}?"),
        qa("Who is {subject}'s mother?"),
        qa("Which person is identified as {subject}'s mother?"),
        qa("Who is the maternal parent of {subject}?"),
        qa("What is the name of {subject}'s mother?"),
    ),
    "occupation": (
        qa("What is the occupation of {subject}?"),
        qa("What does {subject} do for a living?"),
        qa("Which profession is associated with {subject}?"),
        qa("What is {subject}'s profession?"),
        qa("In what occupation does {subject} work?"),
    ),
    "place of birth": (
        qa("Where was {subject} born?"),
        qa("What is the birthplace of {subject}?"),
        qa("In which city was {subject} born?"),
        qa("What place was {subject} born in?"),
        qa("What is {subject}'s city of birth?"),
    ),
    "producer": (
        qa("Who produced {subject}?"),
        qa("Who was the producer of {subject}?"),
        qa("Which person produced {subject}?"),
        qa("Who handled the production of {subject}?"),
        qa("What producer was responsible for {subject}?"),
    ),
    "religion": (
        qa("What is the religion of {subject}?"),
        qa("Which religion is associated with {subject}?"),
        qa("What faith does {subject} follow?"),
        qa("What is {subject}'s religious affiliation?"),
        qa("To which religion does {subject} belong?"),
    ),
    "screenwriter": (
        qa("Who wrote the screenplay for {subject}?"),
        qa("Who was the screenwriter of {subject}?"),
        qa("Which person wrote {subject}'s screenplay?"),
        qa("Who wrote the script for {subject}?"),
        qa("What screenwriter was responsible for {subject}?"),
    ),
    "sport": (
        qa("What sport does {subject} play?"),
        qa("Which sport is played by {subject}?"),
        qa("What is {subject}'s sport?"),
        qa("Which athletic sport is associated with {subject}?"),
        qa("What game does {subject} play competitively?"),
    ),
}


def main() -> None:
    if set(QA_RELATION_TEMPLATE_SETS) != set(pipeline.RELATION_TEMPLATE_SETS):
        raise RuntimeError("QA relation coverage differs from the verified completion relation set")
    if any(len(templates) != 5 for templates in QA_RELATION_TEMPLATE_SETS.values()):
        raise RuntimeError("Every QA relation must have exactly five templates")
    pipeline.RELATION_TEMPLATE_SETS = QA_RELATION_TEMPLATE_SETS
    pipeline.__doc__ = (
        "Screen PopQA factual recall with five instructed question paraphrases "
        "per fact and model."
    )
    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", "results/popqa_qa_instructed_knowledge_smoke"])
    pipeline.main()


if __name__ == "__main__":
    main()

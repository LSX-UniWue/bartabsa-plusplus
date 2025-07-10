# Script for converting JSON data to XMI format
# Note: This script requires the wuenlp package for XMI processing
import argparse
import json
import os
from pathlib import Path

# Set up logger
from loguru import logger
from wuenlp.impl.UIMANLPStructs import UIMADocument, UIMAEntityReference, UIMASentimentTuple
from wuenlp.utils import style


def convert_file(input_path: Path, output_path: Path):
    with open(input_path) as json_file:
        data = json.load(json_file)

        sentences = [itm["raw_words"] + "\n" for itm in data]
        text = "".join(sentences)

    document = UIMADocument.from_text(text)

    last_sentence_end = 0
    for sample in data:
        sentence = sample["raw_words"] + "\n"
        sentence_start = text.index(sentence, last_sentence_end)
        uima_sentence = document.create_sentence(sentence_start, sentence_start + len(sentence), add_to_document=True)

        prev_end = last_sentence_end
        for word in sample["words"]:
            start = text.index(word, prev_end)
            prev_end = start + len(word)
            document.create_token(start, prev_end, True)

        for a, o in zip(sample["aspects"], sample["opinions"]):
            assert a["index"] == o["index"]

            aspect_start = a["from"]
            aspect_end = a["to"] - 1
            aspect = document.create_entity_reference(uima_sentence.tokens[aspect_start].begin, uima_sentence.tokens[aspect_end].end, "aspect", True)

            opinion_start = o["from"]
            opinion_end = o["to"] - 1
            opinion: style.UIMAEntityReference = document.create_entity_reference(
                uima_sentence.tokens[opinion_start].begin, uima_sentence.tokens[opinion_end].end, "opinion", True
            )

            document.create_sentiment_tuple(
                sentence_start,
                sentence_start + len(sentence),
                target=aspect,
                expression=opinion,
                sentiment=a["polarity"],
                add_to_document=True,
            )

        last_sentence_end += len(sentence)

    sentiment_style = {
        "style": "STYLE_ARC",
        "background": "rgb(255,0,0)",
        "foreground": "rgb(255,0,0)",
        "position": "top",
        "arcHeight": 20,
        "from": "Expression",
        "to": "Target",
        "label": "Sentiment",
        "font": "15px arial",
    }

    entity_style = {
        "style": "STYLE_BOX",
        "background": "#71FF19",
        "foreground": "#FFC019",
        "ReferredEntity": {
            "style": "STYLE_STRING",
            "value": "function get_value(anno){return anno.features.Name;};",
            "position": "bottom-right",
            "foreground": "#004EFF",
        },
        "ReferenceType": {
            "style": "STYLE_STRING",
            "position": "bottom-right",
            "foreground": "#FF00E9",
        },
    }

    style.add_style_to_document(document, custom_styles={UIMASentimentTuple: sentiment_style, UIMAEntityReference: entity_style})

    document.serialize(output_path)


def process_datasets(input_folder: Path, output_folder: Path, datasets: list):
    logger.info(f"Converting datasets: {datasets}")
    assert input_folder.exists(), "Input folder does not exist"
    assert output_folder.exists(), "Output folder does not exist"

    for dataset in datasets:
        input_path = input_folder / dataset
        output_path = output_folder / dataset

        assert input_path.exists(), f"Dataset path {input_path} does not exist"
        os.makedirs(output_path, exist_ok=True)

        logger.info(f"Processing dataset '{dataset}'")
        for file in os.listdir(input_path):
            input_file = input_path / file
            output_file = output_path / ("valid.xmi" if file == "dev.json" else file.replace(".json", ".xmi"))
            logger.info(f"Processing file {file} (input path: '{input_file}', output path: '{output_file}')")
            convert_file(input_file, output_file)


def main(args):
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    datasets = [args.dataset] if args.dataset else os.listdir(input_folder)
    process_datasets(input_folder, output_folder, datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ABSA JSON files to XMI files suitable for bartabsa implementation and viewable in webathen"
    )
    parser.add_argument("-in", "--input-folder", default="data/json/pengb", help="Path to input folder")
    parser.add_argument("-out", "--output-folder", default="/tmp", help="Path to output folder")
    parser.add_argument("-d", "--dataset", default=None, help="Dataset name (None for all in folder)")
    args = parser.parse_args()
    main(args)

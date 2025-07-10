import json
from logging import getLogger
from pathlib import Path
from typing import Any, Tuple

from wuenlp.impl.UIMANLPStructs import UIMAAnnotationError, UIMADocument, UIMAEntityReference, UIMASentimentTuple
from wuenlp.utils import style

logger = getLogger("lightning.pytorch")


def create_labels(
    document: UIMADocument, uima_sentence, label: Tuple[int, int, int, int, str], label_type: str, sentence_start: int, sentence_end: int
) -> None:
    """
    Creates sentiment labels for a given document and sentence.

    Args:
        document (UIMADocument): The UIMA document to add annotations to.
        uima_sentence: The UIMA sentence object to which the labels belong.
        label (Tuple[int, int, int, int, str]): A tuple containing label information:
            - opinion start index
            - opinion end index
            - aspect start index
            - aspect end index
            - sentiment label
        label_type (str): The type of the label (e.g., 'gold', 'pred').
        sentence_start (int): The start index of the sentence in the document.
        sentence_end (int): The end index of the sentence in the document.
    """
    aspect_start = label[2]
    aspect_end = label[3]
    aspect = document.create_entity_reference(
        uima_sentence.tokens[aspect_start].begin, uima_sentence.tokens[aspect_end].end, f"aspect_{label_type}", True
    )

    opinion_start = label[0]
    opinion_end = label[1]
    opinion = document.create_entity_reference(
        uima_sentence.tokens[opinion_start].begin, uima_sentence.tokens[opinion_end].end, f"opinion_{label_type}", True
    )

    document.create_sentiment_tuple(
        sentence_start,
        sentence_end,
        target=aspect,
        expression=opinion,  # type: ignore
        sentiment=label[4],
        add_to_document=True,
    )


def convert_to_xmi(data: list[Any], output_path: Path):
    """
    Converts a list of prediction data to an XMI file for visualization in WebAthen.

    The function creates a UIMADocument with sentences containing both the original labels
    and predicted labels, along with their corresponding tokens and sentiment annotations.

    Args:
        data (list[Any]): A list of dictionaries containing prediction data. Each dictionary should include
                          keys such as 'raw_words', 'words', 'original_labels', 'decoded_labels', and 'main_loss'.
        output_path (Path): The file path where the XMI file will be saved.
    """

    # Sort the list by loss to have the most important samples at the top
    data = sorted(data, key=lambda x: x["main_loss"], reverse=True)

    sentences = ["LABEL: " + itm["raw_words"] + "\n" + "PRED: " + itm["raw_words"] + "\n" for itm in data]
    text = "".join(sentences)
    document = UIMADocument.from_text(text)

    last_sentence_end = 0
    for sample in data:
        sentence = "LABEL: " + sample["raw_words"] + "\n"
        sentence_start = text.index(sentence, last_sentence_end)
        uima_sentence = document.create_sentence(sentence_start, sentence_start + len(sentence), add_to_document=True)

        prev_end = last_sentence_end
        for word in sample["words"]:
            start = text.index(word, prev_end)
            prev_end = start + len(word)
            document.create_token(start, prev_end, True)

        for label in sample["original_labels"]:
            create_labels(document, uima_sentence, label, "label", sentence_start, sentence_start + len(sentence))

        last_sentence_end += len(sentence)
        sentence = "PRED: " + sample["raw_words"] + "\n"
        sentence_start = text.index(sentence, last_sentence_end)

        uima_sentence = document.create_sentence(sentence_start, sentence_start + len(sentence), add_to_document=True)

        prev_end = last_sentence_end
        for word in sample["words"]:
            start = text.index(word, prev_end)
            prev_end = start + len(word)
            document.create_token(start, prev_end, True)

        for label in sample["decoded_labels"]:
            add = "prediction_correct" if label in sample["original_labels"] else "prediction_incorrect"
            create_labels(document, uima_sentence, label, add, sentence_start, sentence_start + len(sentence))

        # Add the loss as a label to PRED token
        document.create_annotation_error(sentence_start, sentence_start + 4, round(sample["main_loss"], 2), "loss", True)

        last_sentence_end += len(sentence)

    sentiment_style = {
        "style": "STYLE_ARC",
        "AdditionalFeatures": {"style": "STYLE_STRING", "position": "top", "foreground": "#ff0000", "margin": "-10"},
        "background": "function(annotation) { return annotation.features.ReferenceType === 'prediction_correct' ? 'rgb(0,255,0)' : (annotation.features.ReferenceType === 'prediction_incorrect' ? 'rgb(255,0,0)' : 'rgb(128,128,128)'); }",
        "foreground": "function(annotation) { return annotation.features.Expression.features.ReferenceType.endsWith('prediction_correct') ? 'rgb(0,255,0)' : (annotation.features.Expression.features.ReferenceType.endsWith('prediction_incorrect') ? 'rgb(255,0,0)' : 'rgb(128,128,128)'); }",
        "position": "top",
        "arcHeight": 10,
        "from": "Expression",
        "to": "Target",
        "label": "Sentiment",
        "font": "15px arial",
    }

    entity_style = {
        "style": "STYLE_BOX",
        "AdditionalFeatures": {"style": "STYLE_STRING", "position": "top", "foreground": "#ff0000", "margin": "-10"},
        "background": "function(annotation) { return annotation.features.ReferenceType.endsWith('prediction_correct') ? '#00FF00' : (annotation.features.ReferenceType.endsWith('prediction_incorrect') ? '#FF0000' : '#808080'); }",
        "foreground": "function(annotation) { return annotation.features.ReferenceType.endsWith('prediction_correct') ? '#00FF00' : (annotation.features.ReferenceType.endsWith('prediction_incorrect') ? '#FF0000' : '#808080'); }",
        "ReferenceType": {
            "style": "STYLE_STRING",
            "position": "bottom-right",
            "value": "function get_value(anno){return anno.split('_')[0];};",
            "foreground": "function(annotation) { return annotation.features.ReferenceType.endsWith('prediction_correct') ? '#00FF00' : (annotation.features.ReferenceType.endsWith('prediction_incorrect') ? '#FF0000' : '#808080'); }",
        },
    }

    error_style = [
        {
            "style": "STYLE_BACKGROUND",
            "Comment": {"style": "STYLE_STRING", "position": "top", "foreground": "#ff0000", "margin": "-10"},
            "ErrorType": {"style": "STYLE_STRING", "position": "bottom-right", "foreground": "#ff0000"},
            "background": "rgba(255,0,0,0.2)",
        },
        {
            "style": "BORDER",
        },
    ]

    style.add_style_to_document(
        document, custom_styles={UIMASentimentTuple: sentiment_style, UIMAEntityReference: entity_style, UIMAAnnotationError: error_style}
    )
    document.serialize(output_path)


if __name__ == "__main__":
    # For testing purposes
    with open("/Users/tomvolker/localProjects/ba/absa/bartabsa-lightning/predictions.json", "r") as file:
        data = json.load(file)
    print(type(data))
    convert_to_xmi(data, Path("/Users/tomvolker/localProjects/ba/absa/bartabsa-lightning/predictions2.xmi"))

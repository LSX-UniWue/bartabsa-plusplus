import json
import os


def clean_text(text):
    # Remove any unwanted characters or formatting
    return text.strip()


def create_finetuning_data(input_file, output_file):
    print(f"Processing {input_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        data = json.load(infile)

        for item in data:
            raw_text = item["raw_words"]
            aspects = item["aspects"]
            opinions = item["opinions"]

            # Create a prompt that includes the raw text and asks for aspect and sentiment
            prompt = f"Analyze the following text for aspect-based sentiment:\n\n{raw_text}\n\nIdentify the aspects and their associated sentiments."

            # Create the expected response
            response = "Here's the aspect-based sentiment analysis:\n\n"
            for aspect in aspects:
                aspect_term = " ".join(aspect["term"])
                polarity = aspect["polarity"]
                response += f"Aspect: {aspect_term}\nSentiment: {polarity}\n"

                # Find associated opinion if any
                for opinion in opinions:
                    if opinion["index"] == aspect["index"]:
                        opinion_term = " ".join(opinion["term"])
                        response += f"Opinion: {opinion_term}\n"
                response += "\n"

            # Create the finetuning sample
            finetuning_sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI trained to perform aspect-based sentiment analysis. Identify aspects in the given text and determine their associated sentiments and opinions.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response.strip()},
                ]
            }

            # Write the finetuning sample to the output file
            json.dump(finetuning_sample, outfile)
            outfile.write("\n")

    print(f"Conversion complete. Output written to {output_file}")


# Usage
# Get the scripts directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(scripts_dir, "..")
annotator = "pengb"
datasets = ["14lap", "14res", "15res", "16res"]
for dataset in datasets:
    for split in ["train", "valid", "test"]:
        in_folder = f"{data_path}/absa/{annotator}/json/{dataset}/"
        out_folder = f"{data_path}/finetuning/{annotator}/{dataset}/"
        create_finetuning_data(in_folder + split + ".json", out_folder + dataset + "_" + split + "_finetuning.jsonl")

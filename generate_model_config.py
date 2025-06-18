import csv
import json
import re

# File paths
csv_file_path = "/home/weilinwa/AI/aws/test_scripts/tuner/mteb_leaderboard_models.csv"
large_models_json_file_path = "/home/weilinwa/AI/aws/test_scripts/tuner/large_models_embed.json"
small_models_json_file_path = "/home/weilinwa/AI/aws/test_scripts/tuner/small_models_embed.json"

# Initialize lists to store model data
large_models = []
small_models = []

# Read the CSV file
with open(csv_file_path, "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        # Skip rows with missing or invalid data
        if not row["Model"] or not row["Memory Usage (MB)"] or not row["Embedding Dimensions"]:
            continue

        # Safely convert memory usage to int, skip row if conversion fails
        try:
            mem_usage_mb = int(row["Memory Usage (MB)"])
        except (ValueError, TypeError):
            continue

        # Input string
        input_string = row["Model"].strip()

        # Extract the desired part using regex
        match = re.search(r"\((https://huggingface\.co/([^/]+/[^)]+))\)", input_string)
        if match:
            name = match.group(2)  # Extract the part after "https://huggingface.co/"
        else:
            continue  # Skip if no match found

        # Extract and format the data
        model_data = {
            "model": name,
            "dtype": "bfloat16",  # Assuming dtype is always bfloat16
            "test_parameters": {
                "max_tokens": int(row["Max Tokens"]) if row["Max Tokens"].isdigit() else 512,
                "embedding_dimension": int(row["Embedding Dimensions"]) if row["Embedding Dimensions"].isdigit() else -1,
                "num_parameters": row["Number of Parameters"],
                "mem_usage(MB)": mem_usage_mb,
                "mteb_rank": int(row["Rank (Borda)"]),
                "kv_cache": 0,  # Assuming kv_cache is always 0
                "benchmark_tests": [
                    {"inp_tokens": 100, "concurrency": 1},
                    {"inp_tokens": 1000, "concurrency": 1},
                ],
            },
        }

        # Split models based on memory usage
        if mem_usage_mb > 1000:
            continue
        if mem_usage_mb > 400:
            large_models.append(model_data)
        else:
            small_models.append(model_data)

# Write the large models to a JSON file
with open(large_models_json_file_path, "w") as large_json_file:
    json.dump(large_models, large_json_file, indent=4)

# Write the small models to a JSON file
with open(small_models_json_file_path, "w") as small_json_file:
    json.dump(small_models, small_json_file, indent=4)

print(f"Large models JSON file generated at: {large_models_json_file_path}")
print(f"Small models JSON file generated at: {small_models_json_file_path}")
import json
import os

current_file_name = "running_log"


# helper function to log any errors/outputs to the console and to a text file
def log_print(*args, **kwargs):
    with open(current_file_name + "_CAIR_output_log.txt", "a") as log_file:
        print(*args, **kwargs, file=log_file)  # Save to file
    print(*args, **kwargs)  # Print to console


def read_json(filePath):
    try:
        with open(filePath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        log_print(f"Error decoding JSON: {e}")
    except FileNotFoundError:
        log_print(f"File not found: {filePath}")
        raise

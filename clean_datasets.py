import re
import os
import glob

from config import token

def split_text_natural(text, max_length=10000):
    """Split text into chunks without breaking sentences or paragraphs."""
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n"):
        if len(current_chunk) + len(paragraph) + 1 <= max_length:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to process text using GPT API
def process_text_with_gpt(text_chunk):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a skilled text editor and mythology researcher."},
            {"role": "user", "content": f"""
You are a skilled text editor and mythology researcher. Your task is to strictly clean, format, and extract mythological content from the provided text. Follow these instructions carefully:

1. Remove all irrelevant sections such as:
   - Table of contents
   - Chapter titles, headers, and footers
   - Page numbers
   - Bibliographies, notes, and references
   - Illustrations and captions
   - Dedications and prefaces

2. Focus on extracting only the coherent narrative content directly related to myths, legends, and their historical context. Preserve the original content without adding or inventing any new text.

3. Ensure the final text:
   - Is free of unnecessary symbols, formatting artifacts, and noise.
   - Retains coherent paragraph structures and sentence flow.
   - Excludes any content unrelated to mythology.
   - Does not include any comments, summaries, interpretations, or explanatory notes.

4. If the provided text contains no mythology-related content:
   - Return an empty result as a plain empty string `""`.
   - Do not provide any explanation, commentary, or placeholder text.

5. Do not include elements like chapter numbers or section labels unless they are integral to the mythological context.

**Only return the cleaned and extracted text. If no relevant content exists, return an empty string `""`. Do not provide any additional output under any circumstances.**

Here is the text for cleaning and extraction:
Text:
{text_chunk}
"""}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content

# Function to validate the cleaned text to ensure no extraneous output
def validate_cleaned_text(cleaned_text):
    validation_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quality assurance assistant for text formatting."},
            {"role": "user", "content": f"""
Please review the following text to ensure that:
1. There are no extraneous comments, such as "The provided text does not contain any mythology-related content" or other unnecessary explanations.

Here is the text for validation:
Text:
{cleaned_text}

If the text contains any issues, provide a list of lines that include the problems. Otherwise, confirm that the text is clean.
"""}
        ]
    )
    return validation_response.choices[0].message.content

# Combine all .txt files in the specified directory
data_dir = "/content/drive/MyDrive/LEVI/GPT2/datasets/levi-rawdata"
combined_text = ""
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            combined_text += file.read() + "\n"

# Split the text into chunks with natural boundaries
text_chunks = split_text_natural(combined_text, max_length=10000)

# Specify the number of chunks to process before saving intermediate results
chunks_per_save = 10
output_dir = "/content/drive/MyDrive/LEVI/GPT2/datasets/levi-rawdata-clean"
os.makedirs(output_dir, exist_ok=True)

processed_files = glob.glob(os.path.join(output_dir, "intermediate_*.txt"))
processed_indices = []

# Find the last processed chunk based on filenames
for file_path in processed_files:
    file_name = os.path.basename(file_path)
    match = file_name.split("_")[1].split(".")[0]
    try:
        processed_indices.append(int(match) * 10)  # Multiply by 10 (assuming chunks_per_save = 10)
    except ValueError:
        continue

# Determine the starting point for resuming
start_chunk = max(processed_indices) if processed_indices else 0

# Resume processing from the last incomplete chunk
chunks_per_save = 10
extracted_mythology = ""

for i, chunk in enumerate(text_chunks[start_chunk:], start=start_chunk):
    print(f"Processing chunk {i+1}/{len(text_chunks)}...")
    formatted_chunk = process_text_with_gpt(chunk)

    # Validate the formatted chunk
    validation_result = validate_cleaned_text(formatted_chunk)

    if "clean" in validation_result.lower():
        # If clean, add to the final result
        extracted_mythology += formatted_chunk.replace('""', '') + "\n"
    else:
        print(f"Validation issue detected in chunk {i+1}:")
        print(validation_result)

    # Save intermediate results every `chunks_per_save` chunks
    if (i + 1) % chunks_per_save == 0 or i == len(text_chunks) - 1:
        batch_number = (i + 1) // chunks_per_save
        intermediate_path = os.path.join(output_dir, f"intermediate_{str(batch_number).zfill(4)}.txt")
        with open(intermediate_path, "w", encoding="utf-8") as f:
            f.write(extracted_mythology)
        print(f"Intermediate results saved to {intermediate_path}")
        extracted_mythology = ""  # Reset for next batch

print("All chunks processed and saved.")

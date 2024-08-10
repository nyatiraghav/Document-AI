from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from pdf2image import convert_from_path
import pytesseract
import torch
import pandas as pd
from PIL import Image

# Load the fine-tuned model and processor
model_name = "nyati29/layoutlmv3-finetuned-funsd"  # Replace with your actual model name
processor = LayoutLMv3Processor.from_pretrained(model_name)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

# Function to extract text and bounding boxes from an image using Tesseract
def extract_text_and_bboxes(image):
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    bboxes = []

    # Get image dimensions
    image_width, image_height = image.size

    for i in range(len(ocr_result["text"])):
        word = ocr_result["text"][i].strip()
        if word:  # Only consider non-empty words
            x, y, w, h = (ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i])
            # Normalize the bounding box coordinates to 0-1000 range
            normalized_bbox = [
                int(1000 * x / image_width),  # left
                int(1000 * y / image_height), # top
                int(1000 * (x + w) / image_width),  # right
                int(1000 * (y + h) / image_height)  # bottom
            ]
            if normalized_bbox != [0, 0, 0, 0]:  # Filter out invalid bounding boxes
                words.append(word)
                bboxes.append(normalized_bbox)

    return words, bboxes

# Function to process a single PDF file and extract information
def extract_information_from_pdf(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)

    extracted_data = []

    for page_number, page in enumerate(pages):
        # Extract text and bounding boxes from the image
        words, bboxes = extract_text_and_bboxes(page)

        if not words:  # Skip empty pages
            continue

        # Preprocess the image and text
        encoding = processor(images=page, text=words, boxes=bboxes, return_tensors="pt", truncation=True)

        # Remove "Ġ" from tokens
        tokens = [token.replace("Ġ", "") for token in encoding.tokens()]

        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)

        # Extract the predicted labels
        predictions = outputs.logits.argmax(-1).squeeze().tolist()

        # Extract tokens and corresponding predictions
        bboxes = encoding.bbox.squeeze().tolist()

        for token, bbox, prediction in zip(tokens, bboxes, predictions):
            if token not in processor.tokenizer.all_special_tokens:
                extracted_data.append({
                    "Token": token,
                    "Prediction": prediction,
                    "Bounding Box": bbox,
                    "Page": page_number + 1
                })

    return extracted_data

# Function to save extracted information to Excel
def save_to_excel(extracted_data, output_excel_path):
    df = pd.DataFrame(extracted_data)
    df.to_excel(output_excel_path, index=False)

# Main script
if __name__ == "__main__":
    pdf_path = "OD328547910227093100 (1).pdf"  # Replace with your PDF file path
    output_excel_path = "output.xlsx"  # Replace with your desired output Excel file path

    # Extract information from the PDF
    extracted_data = extract_information_from_pdf(pdf_path)

    # Save the extracted information to an Excel file
    save_to_excel(extracted_data, output_excel_path)

    print(f"Information extracted and saved to {output_excel_path}")

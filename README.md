# Document AI System for Automated Information Extraction

This repository contains code and resources for a Document AI system that automatically extracts and processes information from industry-specific documents. The system uses a fine-tuned LayoutLMv3 model to handle various document formats, extract key details, and classify documents based on predefined categories.

## Contents

- fine_tuning.ipynb: Notebook used to fine-tune the LayoutLMv3 model.
- main.py: Script to process PDF files and extract information into an Excel sheet.
- input.pdf: Sample input PDF file.
- output.xlsx: Sample output Excel file.

## How to Use

1. *Fine-Tune the Model*:
    - Run fine_tuning.ipynb to fine-tune the LayoutLMv3 model on Funsd dataset.

2. *Process PDF Files*:
    - Run main.py to process PDF files and extract information into an Excel sheet.

## Model and Processor

The fine-tuned model and processor are available on the Hugging Face Hub:
- [Model](https://huggingface.co/nyati29/layoutlmv3-finetuned-funsd)
- [Processor](https://huggingface.co/nyati29/layoutlmv3-finetuned-funsd)
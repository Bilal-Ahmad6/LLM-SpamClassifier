# Spam Classifier using GPT-2 Fine-tuning

A clean, well-structured implementation of SMS spam detection using GPT-2 fine-tuning with interactive prediction capabilities.


## Files

- `spam_classifier.py` - Main implementation with all functionality
- `test_spam_detection.py` - Simple test script for quick testing
- `requirements.txt` - Required dependencies
- `README.md` - This file

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the `gpt_download3.py` file in the same directory (for loading pre-trained GPT-2 weights)

## Usage

### Option 1: Full Training and Interactive Mode

Run the complete training pipeline with interactive spam detection:

```bash
python spam_classifier.py
```

### Option 2: Web Application

Run the Flask-based web app for SMS spam classification:

```bash
cd webapp
python app.py
```

Open your browser and go to [http://localhost:5000](http://localhost:5000) to use the web interface.

### Option 3: Generate Technical Report

Generate a comprehensive technical report in DOCX format:

```bash
python Technical_Report_Spam_Classifier.py
```

The report will be saved as `Spam_Classifier_Technical_Report.docx` in the project directory.

## GitHub Repository Setup

To upload this project to GitHub:

1. Initialize a git repository:
   ```bash
   git init
   ```
2. Add all files:
   ```bash
   git add .
   ```
3. Commit your changes:
   ```bash
   git commit -m "Initial commit: GPT-2 SMS Spam Classifier with web app and report"
   ```
4. Create a new repository on GitHub (via the website).
5. Link your local repo to GitHub:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

Replace `yourusername` and `your-repo-name` with your GitHub username and desired repository name.

This will:
1. Download the SMS spam dataset
2. Prepare and balance the data
3. Load pre-trained GPT-2 weights
4. Fine-tune the model for spam classification
5. Launch interactive spam detection interface

### Option 2: Quick Testing

Test the spam detection with pre-defined examples and interactive input:

```bash
python test_spam_detection.py
```

This will:
1. Load a trained model (if available) or use untrained model
2. Test with sample spam/legitimate messages
3. Allow you to enter your own text for classification

## Interactive Spam Detection

Once running, you can enter any text and get instant classification:

```
Enter text: You are a winner! Click here to claim your prize!
Classification: SPAM
Confidence: 85.23%
⚠️  This message appears to be spam!

Enter text: Hi, can you pick me up from work at 6 PM?
Classification: NOT SPAM
Confidence: 92.15%
✅ This message appears to be legitimate.
```

## Model Architecture

- **Base Model**: GPT-2 Small (124M parameters)
- **Classification Head**: Linear layer with 2 outputs (spam/not spam)
- **Fine-tuning**: Only last transformer block and final layer norm are trained
- **Tokenization**: GPT-2 tokenizer using tiktoken

## Dataset

Uses the SMS Spam Collection Dataset from UCI:
- 5,574 SMS messages
- Balanced dataset (equal spam/legitimate messages)
- 70% train, 10% validation, 20% test split

## Performance

Typical results after training:
- Training accuracy: ~95-98%
- Validation accuracy: ~94-97%
- Test accuracy: ~93-96%

## Key Improvements Made

1. **Code Organization**: Separated into logical sections with clear comments
2. **Error Fixes**: Fixed all type hints and potential runtime errors
3. **Interactive Feature**: Added user-friendly spam detection interface
4. **Better Documentation**: Comprehensive docstrings and comments
5. **Cleaner Structure**: Removed unnecessary code and improved readability
6. **Type Safety**: Added proper type hints throughout
7. **Error Handling**: Robust error handling for user inputs

## Example Output

```
Using device: cuda
Dataset already exists at sms_spam_collection/SMSSpamCollection.tsv
Dataset loaded: 5574 samples
Label distribution:
ham     4825
spam     747
Name: Label, dtype: int64
Train: 1046, Validation: 149, Test: 299

Starting training...
Epoch 1: Train accuracy 95.23% | Val accuracy 94.63%
Epoch 2: Train accuracy 96.18% | Val accuracy 95.30%
...

Training completed in 2.45 minutes.

Final Results:
Training accuracy: 97.23%
Validation accuracy: 96.64%
Test accuracy: 96.32%

============================================================
SPAM DETECTION INTERFACE
============================================================
Enter text to classify as spam or not spam.
Type 'quit' to exit.
------------------------------------------------------------

Enter text: You've won a free iPhone! Click here to claim!
Classification: SPAM
Confidence: 89.45%
⚠️  This message appears to be spam!
```

## Troubleshooting

- **Import Error**: Make sure `gpt_download3.py` is in the same directory
- **CUDA Error**: The code automatically falls back to CPU if CUDA is not available
- **Memory Error**: Reduce batch size in the training configuration
- **Download Error**: Check internet connection for dataset download

## License

This code is provided for educational purposes. The SMS dataset is from UCI Machine Learning Repository.

## Folder Structure

```
├── gpt_download3.py
├── README.md
├── sms_spam_collection.zip
├── Spam_Classifier_Technical_Report.docx
├── spam_classifier.pth
├── spam_classifier.py
├── Technical_Report_Spam_Classifier.py
├── test.csv
├── train.csv
├── validation.csv
├── __pycache__/
│   ├── cute_handdrawn_gui.cpython-311.pyc
│   ├── gpt_download3.cpython-311.pyc
│   ├── spam_classifier.cpython-311.pyc
│   └── spam_detector_gui.cpython-311.pyc
├── gpt2/
│   └── 124M/
│       ├── checkpoint
│       ├── encoder.json
│       ├── hparams.json
│       ├── model.ckpt.data-00000-of-00001
│       ├── model.ckpt.index
│       ├── model.ckpt.meta
│       └── vocab.bpe
├── sms_spam_collection/
│   ├── readme
│   └── SMSSpamCollection.tsv
├── static/
│   └── images/
│       ├── spam_ham_chart.png (removed)
│       └── ...
├── tutorials/
│   ├── Architecture Classification finetuning.ipynb
│   ├── architecture.py
│   ├── classificationHead.py
│   ├── finetuned_settingData.py
│   └── LossCalc.py
├── webapp/
│   ├── app.py
│   ├── run_webapp.bat
│   ├── static/
│   │   ├── barchart.svg
│   │   ├── piechart.svg
│   │   ├── scatter.svg
│   │   ├── sketchy_robot.svg
│   │   ├── sketchy.css
│   │   ├── sketchy.js
│   │   ├── spa.js
│   │   ├── spam_illustration.svg
│   │   └── style.css
│   └── templates/
│       ├── base.html
│       ├── classify.html
│       ├── home.html
│       ├── how-it-works.html
│       ├── index.html
│       ├── result.html
│       └── statistics.html
```
# Historical Document OCR System 📜

![App Demo](demo.gif) <!-- You can add your video demo here -->

A sophisticated Optical Character Recognition (OCR) system specifically designed for processing historical documents. This application leverages Google's Gemini AI for accurate text extraction and entity recognition from historical documents.

## ✨ Features

### 🔍 Core Functionality
- **Advanced OCR Processing**: Utilizing Google Gemini AI for accurate text extraction
- **Entity Recognition**: Automatically identifies and extracts key entities from documents
- **Multi-Format Support**: Handles various image formats (PNG, JPG, JPEG, WEBP)
- **Batch Processing**: Process multiple documents efficiently

### 💫 User Interface
- **Modern Web Interface**: Clean, responsive design built with Flask and Bootstrap
- **Real-time Processing**: Immediate feedback on document processing
- **Interactive Transcription View**: 
  - Line numbers for easy reference
  - Copy functionality for extracted text
  - Download option for transcriptions
  - Page-wise organization

### 🛠 Technical Features
- **Smart Error Handling**: Robust error management system
- **Progress Tracking**: Real-time processing status updates
- **Flexible Output Formats**: Structured data output in various formats

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Google Gemini API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/historic-doc-ocr.git
cd historic-doc-ocr
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the root directory and add:
```env
GEMINI_API_KEY=your_api_key_here
```

### Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
historic-doc-ocr/
├── app.py              # Flask application main file
├── gemini_client.py    # Google Gemini AI client
├── main.py            # Core processing functions
├── prompts.py         # AI prompts configuration
├── schemas.py         # Data schemas
├── similarity.py      # Text similarity functions
├── static/           # Static files (CSS, JS)
├── templates/        # HTML templates
│   ├── index.html    # Upload interface
│   └── transcription.html  # Results view
├── input_images/     # Temporary storage for uploads
└── output/          # Processed results
    ├── transcriptions/  # Raw transcriptions
    └── final_outputs/   # Processed entities
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key

### Supported File Types
- PNG
- JPG/JPEG
- WEBP

## 📝 Usage Guide

1. **Upload Document**
   - Click the upload button on the home page
   - Select your historical document image
   - Click "Process Document"

2. **View Results**
   - Wait for processing to complete
   - View the transcribed text with line numbers
   - Use the toolbar to:
     - Copy text
     - Download transcription
     - Toggle line numbers

3. **Batch Processing**
   - Multiple documents can be processed in sequence
   - Results are stored in separate files

## 🎯 Features in Detail

### OCR Processing
- High-accuracy text extraction
- Maintains document formatting
- Handles various font styles and sizes

### Entity Recognition
- Identifies key information:
  - Names
  - Dates
  - Locations
  - Organizations
  - Key terms

### User Interface
- Responsive design
- Dark/light mode support
- Intuitive navigation
- Progress indicators
- Error notifications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎬 Demo


https://github.com/user-attachments/assets/55b4001a-f09a-4aa3-8f1d-8ee3ab9cad19




## 🙏 Acknowledgments

- Google Gemini AI for OCR processing
- Flask framework
- Bootstrap for UI components
- All contributors and testers

## 📞 Support

For support, please open an issue in the GitHub repository or contact muhammadhanan23230@gmail.com.

---

Built with ❤️ using Python, Flask, and Google Gemini AI

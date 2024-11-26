# Ask Your PDF 📄

**Ask Your PDF** is a powerful web app that allows users to upload PDF documents and ask questions about their content. The app leverages advanced AI models to extract meaningful information from PDF files, enabling users to get context-based answers in real-time. 

This project uses **Streamlit** for the frontend, **PyPDF2** to extract text from PDFs, and **Google Generative AI** for advanced question-answering capabilities. The backend also integrates with **LangChain** for managing embeddings and queries.

---

## Features

- **PDF Upload & Parsing**: Upload multiple PDF files for processing.
- **Context-Based Question Answering**: Ask questions related to the content of your PDFs, and get precise, context-driven answers.
- **PDF Preview**: View a preview of your PDFs in the sidebar.
- **Advanced AI**: Utilizes **Google Gemini Pro** model for generating answers based on document context.

---


## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:

   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required libraries**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:

   Create a `.env` file in the root directory and add your Google API key:

   ```
   GOOGLE_API_KEY=your_google_api_key
   ```

---

## Running the App

After setting up, you can run the app using the following command:

```bash
streamlit run app.py
```

This will launch the app in your default web browser. From there, you can upload PDFs and ask questions about the content!

---

## How It Works

1. **Upload PDF Files**: The user can upload one or more PDF files using the file uploader widget.
2. **PDF Content Extraction**: The text is extracted from the PDF using **PyPDF2**, and large texts are split into chunks for efficient processing.
3. **Embedding Generation**: The chunks are embedded using **Google Generative AI** embeddings and stored in a **FAISS** index.
4. **Query Processing**: The user can input a query, and the app retrieves relevant context from the indexed documents to answer the question using **Google Gemini Pro** for response generation.
5. **Answer Generation**: The response is generated based on the provided context and presented to the user in the main app interface.

---

## Customization

- **Model Selection**: The app uses **Google Gemini Pro** for question-answering. You can change the model by modifying the `get_conversational_chain` function.
- **PDF Parsing**: You can adjust the PDF parsing logic (like text chunking) by modifying the `get_pdf_text` and `get_text_chunks` functions.
- **Styling**: Customize the Streamlit interface by updating the embedded CSS styles in the `main` function.

---

## License

This project is licensed under the MIT License.

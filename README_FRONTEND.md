# Vietnamese Poem Generator Chat

A modern chatbot-style frontend for the Vietnamese Poem Generator API, featuring a beautiful UI similar to ChatGPT/Gemini.

## Features

- 🎨 **Modern Chatbot UI**: Clean, responsive design similar to ChatGPT/Gemini
- 💬 **Real-time Chat**: Interactive chat interface with message history
- 📝 **Poem Type Selection**: Choose from 5 Vietnamese poem types:
  - Thơ bốn chữ (4-character poetry)
  - Thơ bảy chữ (7-character poetry)
  - Thơ lục bát (6-8 character poetry)
  - Thơ tám chữ (8-character poetry)
  - Thơ năm chữ (5-character poetry)
- ⚙️ **Adjustable Parameters**: Control generation parameters with sliders:
  - Max Tokens
  - Temperature
  - Top-K
  - Top-P
- 💾 **Chat History**: Persistent chat history stored in browser localStorage
- 📤 **Export Functionality**: Export conversations as JSON files
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🌐 **CORS Support**: Configured for cross-origin requests

## File Structure

```
nlp_vietnamese/
├── index.html          # Main HTML file
├── styles.css          # CSS styling
├── script.js           # JavaScript functionality
├── serve_frontend.py   # Frontend server script
├── generation_api.py   # FastAPI backend
└── ... (other files)
```

## Setup and Running

### 1. Start the FastAPI Backend

First, make sure your FastAPI backend is running:

```bash
python generation_api.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend Server

In a new terminal, run the frontend server:

```bash
python serve_frontend.py
```

The frontend will be available at `http://localhost:3000`

### 3. Alternative: Direct File Access

You can also open `index.html` directly in your browser, but you'll need to ensure CORS is properly configured.

## Usage

1. **Select Poem Type**: Choose your desired Vietnamese poem type from the dropdown
2. **Enter Prompt**: Type your prompt in the text area
3. **Adjust Parameters**: Use the sliders to fine-tune generation parameters
4. **Send Message**: Click the send button or press Enter
5. **View Results**: Generated poems will appear in the chat interface
6. **Manage Chat**: Use the Clear Chat and Export buttons in the header

## API Integration

The frontend communicates with your FastAPI backend at `http://localhost:8000`. The API endpoint expects:

**POST /generate**
```json
{
  "prompt": "thơ lục bát: về mùa xuân",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9
}
```

And returns:
```json
{
  "generated_text": "Generated poem content..."
}
```

## Troubleshooting

### Model Loading Issues
If the model fails to load:
1. Check that your model files are in the correct location
2. Verify the `config.yaml` path is correct
3. Ensure all dependencies are installed

### Chat History Not Saving
If chat history isn't persisting:
1. Check browser localStorage support
2. Ensure the browser allows localStorage for the site
3. Check browser console for JavaScript errors

## Development

### Testing

Test the application by:
1. Running both servers
2. Opening the frontend in multiple browser tabs
3. Testing different poem types and parameters
4. Verifying chat history persistence
5. Testing export functionality

## License

This frontend is part of the Vietnamese Poem Generator project.

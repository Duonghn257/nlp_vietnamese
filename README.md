# Vietnamese NLP Project

A comprehensive Vietnamese Natural Language Processing project with custom tokenizer, transformer model, and training pipeline.

## Features

- **Custom Vietnamese Tokenizer**: BPE-based tokenizer with Vietnamese-specific optimizations
- **Word Segmentation**: Integration with NlpHUST Vietnamese word segmentation model
- **Transformer Model**: Decoder-only transformer for text generation
- **Training Pipeline**: Complete training and evaluation pipeline
- **YAML Configuration**: Flexible configuration management using YAML files
- **Streamlit Chat App**: Interactive web interface for poem generation

## Configuration System

The project uses a YAML-based configuration system for easy parameter management. The main configuration file is `config.yaml`:

### Configuration Structure

```yaml
# Data configuration
data:
  data_folder: "data/clean_data"
  tokenizer_file: "vietnamese_tokenizer.json"
  vocab_size: 25000
  max_seq_len: 512
  train_split: 0.8

# Model configuration
model:
  d_model: 768
  n_heads: 12
  n_layers: 12
  d_ff: 3072
  dropout: 0.1

# Training configuration
training:
  batch_size: 16
  learning_rate: 3e-5
  weight_decay: 0.01
  num_epochs: 50
  warmup_steps: 5000
  device: "auto"  # 'cuda', 'cpu', 'mps', or 'auto'

# Generation configuration
generation:
  temperature: 0.8
  top_k: 10
  top_p: 0.9
  max_new_tokens: 256

# Save configuration
save:
  model_save_path: "vietnamese_transformer_best.pt"
  config_save_path: "training_config.json"
```

### Using Configuration

```python
from src.helpers import setup_training_config

# Load default configuration
config = setup_training_config("config.yaml")

# Override specific values
config['batch_size'] = 32
config['learning_rate'] = 1e-4

# Use in training
model = VietnameseTransformer(**config)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Duonghn257/nlp_vietnamese.git
cd nlp_vietnamese
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the main training script:
```bash
python main.py
```

The script will:
1. Load configuration from `config.yaml`
2. Prepare the dataset
3. Build and train the tokenizer
4. Initialize the transformer model
5. Train the model
6. Test generation
7. Save results

### Streamlit Chat App

After training your model, you can use the interactive Streamlit chat app:

1. Install Streamlit (if not already installed):
```bash
pip install streamlit
```

2. Run the chat app:
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and go to `http://localhost:8501`

#### Chat App Features:
- **Interactive Chat Interface**: Chat with your Vietnamese poem generator
- **Parameter Controls**: Adjust `max_new_tokens`, `temperature`, `top_k`, and `top_p` in real-time
- **Conversation History**: All conversations are saved in session
- **Export Conversations**: Save your chat history as JSON files
- **Model Management**: Initialize and manage your trained model

#### Usage Steps:
1. Click "🚀 Initialize Model" in the sidebar
2. Adjust generation parameters using the sliders
3. Type your prompt in the chat input
4. View the generated poem response
5. Save your conversation when finished

### Web Interface (Backend + Frontend)

For a more advanced web interface with streaming generation, you can run both the backend API and frontend server:

#### 1. Start the Backend API Server

First, start the FastAPI backend server:

```bash
python generation_api.py
```

The backend will be available at `http://localhost:8000`

#### 2. Start the Frontend Server

In a new terminal, start the frontend server:

```bash
python serve_frontend.py
```

The frontend will be available at `http://localhost:3000`

#### 3. Access the Web Interface

1. Open your browser and go to `http://localhost:3000`
2. The interface will automatically connect to the backend API
3. Enter your poem prompt and adjust generation parameters
4. Watch the poem generate in real-time with streaming

#### Web Interface Features:
- **Real-time Streaming**: See the poem generate word by word
- **Modern UI**: Beautiful, responsive interface
- **Parameter Controls**: Adjust generation parameters in real-time
- **Poetry Types**: Select different Vietnamese poetry styles
- **Conversation History**: Keep track of all generated poems
- **Mobile Responsive**: Works on desktop and mobile devices

#### Troubleshooting:
- Make sure both servers are running simultaneously
- Check that ports 8000 and 3000 are available
- If the frontend can't connect to the backend, check the API URL in `index.html`
- For CORS issues, ensure the backend CORS settings are correct

### Custom Configuration

Create a custom configuration file:
```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml with your preferred settings
```

Use custom configuration:
```python
config = setup_training_config("my_config.yaml")
```

### Example Usage

See `example_config_usage.py` for detailed examples of how to use the configuration system.

## Project Structure

```
nlp_vietnamese/
├── config.yaml              # Main configuration file
├── main.py                  # Main training script
├── requirements.txt          # Python dependencies
├── generation_api.py        # FastAPI backend server
├── serve_frontend.py        # Frontend server
├── index.html              # Frontend interface
├── streamlit_app.py        # Streamlit chat interface
├── run_app.py              # Streamlit app runner
├── src/
│   ├── __init__.py
│   ├── tokenizer.py         # Vietnamese tokenizer
│   ├── model.py            # Transformer model
│   ├── dataset.py          # Dataset handling
│   ├── trainer.py          # Training pipeline
│   ├── helpers.py          # Utility functions
│   ├── chat.py             # Poem generation interface
│   └── word_piece.py       # WordPiece tokenizer
├── data/                    # Training data (ignored by git)
├── notebooks1/             # Jupyter notebooks
└── example_config_usage.py # Configuration examples
```

## Configuration Parameters

### Data Configuration
- `data_folder`: Path to training data directory
- `tokenizer_file`: Path to save/load tokenizer
- `vocab_size`: Vocabulary size for tokenizer
- `max_seq_len`: Maximum sequence length
- `train_split`: Training/validation split ratio

### Model Configuration
- `d_model`: Model dimension
- `n_heads`: Number of attention heads
- `n_layers`: Number of transformer layers
- `d_ff`: Feed-forward dimension
- `dropout`: Dropout rate

### Training Configuration
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay for regularization
- `num_epochs`: Number of training epochs
- `warmup_steps`: Learning rate warmup steps
- `device`: Training device ('cuda', 'cpu', 'mps', 'auto')

### Generation Configuration
- `temperature`: Sampling temperature
- `top_k`: Top-k sampling parameter
- `top_p`: Top-p (nucleus) sampling parameter
- `max_new_tokens`: Maximum tokens to generate

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update configuration if needed
5. Submit a pull request

## License

This project is licensed under the MIT License. 
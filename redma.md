# 📊 Data Analysis & AI Chatbot Platform

A comprehensive desktop application for SME businesses to analyze data and get AI-powered insights.

## Features

✅ **Data Loading** - Upload Excel files and preview data  
✅ **Data Understanding** - Automatic EDA with missing values, correlations, and outliers  
✅ **Data Cleaning** - Handle missing values, remove outliers, encode categories, scale data  
✅ **Visualization** - Create interactive charts (line, bar, pie, histogram, boxplot, scatter, etc.)  
✅ **AI Chatbot** - Ask questions about your data using local LLMs with RAG  
✅ **RAG Pipeline** - Retrieval Augmented Generation using ChromaDB and Ollama  

## Project Structure

```
Analsis-everything/
├── app/
│   └── data_loader.py          # Excel file loading
├── analysis/
│   ├── eda.py                  # Exploratory Data Analysis
│   └── visualization.py        # Chart creation
├── pipelines/
│   └── cleaning.py             # Data cleaning operations
├── models/
│   ├── data_to_text.py         # Convert data to text for LLM
│   └── rag_pipeline.py         # RAG pipeline with ChromaDB
├── ui/streamlit/
│   └── app.py                  # Main Streamlit application
├── config/
│   └── settings.yml            # Configuration file
├── data/                       # Data storage
├── requirements.txt            # Python dependencies
├── create_sample_data.py       # Generate sample Excel files
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.10+
- Ollama (for local LLMs)
- pip

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Ollama

Download and install Ollama from: https://ollama.ai

### Step 3: Pull LLM Models

```bash
# Main LLM model
ollama pull qwen2.5:7b

# Embedding model
ollama pull nomic-embed-text:latest

# Optional: Other models
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### Step 4: Create Sample Data (Optional)

```bash
python create_sample_data.py
```

This creates 5 sample Excel files:
- `sample_sales.xlsx` - Sales transactions
- `sample_hr.xlsx` - Employee data
- `sample_finance.xlsx` - Financial transactions
- `sample_inventory.xlsx` - Inventory management
- `sample_customers.xlsx` - Customer information

## Usage

### Start the Application

```bash
cd ui/streamlit
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Workflow

1. **Data Loading** 📤
   - Upload an Excel file
   - Select a sheet
   - Preview the data

2. **Data Understanding** 🔍
   - View summary statistics
   - Analyze missing values
   - Check correlations
   - Detect outliers
   - Profile individual columns

3. **Data Cleaning** 🧹
   - Handle missing values (mean, median, mode, drop, forward fill, backward fill)
   - Remove outliers (IQR, Z-score)
   - Encode categorical variables (label, one-hot)
   - Scale numeric variables (standard, min-max)
   - Remove duplicates

4. **Visualization** 📈
   - Create interactive charts
   - Supported types: Line, Bar, Pie, Histogram, Boxplot, Scatter, Distribution, Trend, Comparison
   - Customize axes and parameters

5. **AI Chatbot** 🤖
   - Ask natural language questions about your data
   - Get answers powered by local LLMs
   - See retrieved context documents
   - Chat history maintained

## Configuration

Edit `config/settings.yml` to customize:

```yaml
llm:
  model: "qwen2.5:7b"           # Main LLM model
  embedding_model: "nomic-embed-text:latest"  # Embedding model
  temperature: 0.7              # LLM temperature
  max_tokens: 1024              # Max response length

rag:
  chunk_size: 500               # Text chunk size
  top_k: 5                      # Number of retrieved documents
```

## Supported Data Sources (Phase 1: Excel Only)

Currently supports:
- ✅ Excel (.xlsx, .xls)

Coming soon:
- CSV
- SQL Server
- PostgreSQL
- Oracle
- MySQL
- Google Sheets
- APIs

## LLM Models Available

The application works with any Ollama model. Recommended:

**For Speed:**
- `qwen2.5:7b` (4.7 GB) - Fast, good quality
- `mistral:7b` (4.4 GB) - Very fast
- `llama3.1:8b` (4.9 GB) - Good balance

**For Quality:**
- `qwen2.5:14b-instruct` (9.0 GB) - Better responses
- `deepseek-r1:14b` (9.0 GB) - Reasoning focused

**For Embeddings:**
- `nomic-embed-text:latest` (274 MB) - Recommended
- `bge-m3:latest` (1.2 GB) - Multilingual

## Troubleshooting

### Ollama Connection Error
```
Error: Could not connect to Ollama
```
**Solution:** Make sure Ollama is running
```bash
ollama serve
```

### Out of Memory
If you get memory errors, use smaller models:
```bash
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

### Slow Performance
- Use smaller models (7B instead of 14B)
- Reduce `chunk_size` in settings.yml
- Reduce `top_k` for fewer retrieved documents

### ChromaDB Issues
Clear the database:
```bash
rm -rf data/chroma_db
```

## Performance Tips

1. **For Large Datasets:**
   - Limit preview rows
   - Use sampling for visualization
   - Reduce RAG chunk size

2. **For Better Responses:**
   - Use larger models (14B)
   - Increase `top_k` in RAG settings
   - Provide more context in questions

3. **For Faster Processing:**
   - Use smaller models (7B)
   - Reduce number of documents in RAG
   - Use GPU acceleration if available

## API Usage (Advanced)

You can use the modules programmatically:

```python
from app.data_loader import ExcelLoader
from analysis.eda import EDAAnalyzer
from pipelines.cleaning import DataCleaner
from analysis.visualization import Visualizer
from models.rag_pipeline import RAGPipeline

# Load data
loader = ExcelLoader("data.xlsx")
df = loader.load_sheet("Sheet1")

# Analyze
analyzer = EDAAnalyzer(df)
print(analyzer.get_summary_report())

# Clean
cleaner = DataCleaner(df)
cleaner.handle_missing_values(strategy="mean")
df_clean = cleaner.get_cleaned_df()

# Visualize
visualizer = Visualizer(df_clean)
fig = visualizer.line_chart("Date", "Sales")

# RAG
rag = RAGPipeline()
rag.add_documents(["Sample document 1", "Sample document 2"])
answer, docs = rag.generate_answer("What is the total sales?")
```

## Development Roadmap

### Phase 1 ✅ (Current)
- Excel data loading
- Basic EDA
- Data cleaning
- Visualization
- RAG chatbot

### Phase 2 (Next)
- CSV support
- SQL Server connector
- PostgreSQL connector
- Advanced analytics

### Phase 3
- Oracle support
- MySQL support
- Google Sheets integration
- API connectors

### Phase 4
- Desktop app (Electron/Tauri)
- Advanced ML models
- Custom model training
- Export reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.

## Acknowledgments

- Built with Streamlit
- LLMs powered by Ollama
- Vector DB: ChromaDB
- Data processing: Pandas, NumPy
- Visualization: Plotly

---

**Made for SME Businesses** 🚀

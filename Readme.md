# TDS RAG Query API - Quick Setup Guide

## 🚀 Automated Setup

This project includes an automated setup script that will install all dependencies and configure your environment.

### Method 1: Python Setup Script (Recommended)

```bash
python setup.py
```

### Method 2: Windows Batch Script

Double-click `setup.bat` or run:
```cmd
setup.bat
```

## What the Setup Script Does

1. ✅ **Checks Python version** (requires Python 3.8+)
2. ✅ **Installs all required packages** from requirements.txt
3. ✅ **Creates/updates .env file** with all necessary variables
4. ✅ **Preserves existing API key** if .env already exists
5. ✅ **Verifies project structure** is correct
6. ✅ **Checks database file** exists

## 🔑 After Running Setup

### 1. Set Your API Key

Open the `.env` file and replace `YOUR_AIPIPE_API_KEY_HERE` with your actual API key:

```env
API_KEY=your_actual_api_key_here
```

Get your API key from: [https://aipipe.org/login](https://aipipe.org/login)

### 2. Test the Setup

**Local Testing:**
```bash
python src/app.py
```
Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

**Vercel Deployment:**
```bash
vercel --prod
```

### 3. Run Promptfoo Evaluation

Update `promptfooconfig.yaml` with your Vercel URL and run:
```bash
npx promptfoo eval
```

## 📁 Project Structure After Setup

```
TDS-Project-1/
├── setup.py                 # Automated setup script
├── setup.bat                # Windows batch setup
├── .env                     # Environment configuration
├── .env.template            # Template for environment variables
├── requirements.txt         # Python dependencies
├── src/
│   └── app.py              # Main FastAPI application
├── api/
│   └── handler.py          # Vercel handler
├── data/
│   └── knowledge_base_compressed.db
└── promptfooconfig.yaml    # Evaluation configuration
```

## 🛠 Manual Setup (Alternative)

If you prefer manual setup:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy environment template:**
   ```bash
   cp .env.template .env
   ```

3. **Edit .env file** and add your API key

## 🔧 Environment Variables

The setup script configures these variables:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `API_KEY` | `YOUR_AIPIPE_API_KEY_HERE` | AIPipe API token (required) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model for responses |
| `SIMILARITY_THRESHOLD` | `0.68` | Similarity threshold for search |
| `MAX_RESULTS` | `15` | Maximum search results |
| `MAX_CONTEXT_CHUNKS` | `4` | Context chunks for RAG |
| `REQUEST_TIMEOUT` | `30` | API request timeout (seconds) |
| `MAX_RETRIES` | `3` | Maximum retry attempts |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## ❗ Troubleshooting

### Python Version Error
- Ensure Python 3.8+ is installed
- Check: `python --version`

### Package Installation Error
- Update pip: `python -m pip install --upgrade pip`
- Try: `pip install --user -r requirements.txt`

### Database Not Found
- Ensure `knowledge_base_compressed.db` is in the `data/` folder
- Check file permissions

### API Key Issues
- Verify your API key is valid at [https://aipipe.org](https://aipipe.org)
- Ensure no extra spaces in the .env file
- API key should be on a single line

## 🎯 Next Steps

After successful setup:

1. **Test locally** to ensure everything works
2. **Deploy to Vercel** for production use
3. **Run Promptfoo evaluation** to test your RAG system
4. **Customize configuration** as needed for your use case

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all files are in the correct locations
3. Ensure your API key is valid
4. Check the console output for specific error messages

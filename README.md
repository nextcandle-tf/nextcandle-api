# NextCandle API

The backend API service for NextCandle, powered by Python and PyTorch.

## 💡 Overview
This service handles cryptocurrency pattern recognition using deep learning (PyTorch) and provides RESTful endpoints for the frontend application. It manages user authentication, subscription data, and real-time market data retrieval from Binance.

## 🛠️ Tech Stack
*   **Language**: Python 3.11+
*   **Web Framework**: Flask / Gunicorn
*   **AI/ML**: PyTorch (PatternEncoder, Triplet Loss)
*   **Data Source**: CCXT (Binance API)
*   **Database**: SQLite (User data, Cache)

## 🚀 Getting Started

### Prerequisites
*   Python 3.11+
*   Virtual Environment (venv) recommended

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/nextcandle-tf/nextcandle-api.git
    cd nextcandle-api
    ```
2.  Create and activate virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Server
```bash
# Development
python pattern_api.py

# Production (Gunicorn)
gunicorn -c gunicorn_config.py pattern_api:app
```

## 📂 Project Structure
*   `pattern_api.py`: Main Flask application entry point.
*   `database.py`: Database connection and ORM-like management.
*   `users.db`: SQLite database for user info (gitignored).
*   `gunicorn_config.py`: Production server configuration.
*   `requirements.txt`: Python dependencies.

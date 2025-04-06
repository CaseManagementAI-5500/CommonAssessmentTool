# Team TicTech – Common Assessment Tool Backend

## 🎯 Project Goal

Create and containerize a backend API to support the CaseManagement service, allowing social workers to retrieve and update client information, assess intervention impacts, and improve employment outcomes using predictive models.

---

## 🧠 User Story

> As a user of the backend API, I want to call endpoints that can retrieve, update, and delete information of clients who have already registered with the CaseManagment service so that I can more efficiently help them make better decisions on how to be gainfully employed.

---

## ✅ Acceptance Criteria

- Provide REST API endpoints so that the Frontend can use them to get information on existing clients.
- Document how to use the REST API.
- Choose and create a database to hold client information.
- Add tests.
- Enable the backend to run in a Docker container on any OS (macOS, Windows, Linux).
- Enable support for Docker Compose.
- Update README with instructions to run using Docker.

---

## 🧪 Features

This project contains:

- ✅ RESTful FastAPI backend for client management and intervention planning
- ✅ Predictive ML model based on dummy data
- ✅ Admin-authenticated endpoints for case workers
- ✅ Dynamic model switching API (Logistic, Random Forest, Decision Tree)
- ✅ Docker + Docker Compose support for easy deployment

---

## 🚀 How to Use (Locally Without Docker)

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Initialize the database with dummy data**:
   ```bash
   python initialize_data.py
   ```

5. **Open the Swagger UI**:
   - http://127.0.0.1:8000/docs

6. **Login credentials**:
   - Username: `admin`
   - Password: `admin123`

---

## 🔐 Available Endpoints (Swagger UI)

- **Create User**
- **Get Clients**
- **Get Client by ID**
- **Update Client**
- **Delete Client**
- **Get Clients by Criteria**
- **Get Clients by Services**
- **Get Client Services**
- **Get Clients by Success Rate**
- **Get Clients by Case Worker**
- **Update Client Services**
- **Create Case Assignment**
- **ML Model APIs**:
  - `GET /ml/models` – List available models
  - `GET /ml/model` – Get current model
  - `POST /ml/model` – Switch model
  - `POST /ml/predict` – Predict success rate and interventions

---

## 🐳 Running with Docker

> Use Docker to containerize and run the backend on any platform (macOS, Windows, Linux)

### 🧱 Build Docker Image
```bash
docker build -t common-assessment-tool-backend .
```

### ▶️ Run with Docker
```bash
docker run -p 8000:8000 common-assessment-tool-backend
```

Then visit:  
http://localhost:8000/docs

---

## ⚙️ Running with Docker Compose

> Recommended for managing containers easily

### 🛠 Start the app
```bash
docker-compose up --build
```

### 🛑 Stop the app
```bash
docker-compose down
```

---

## 🗂 .dockerignore (Optional but Recommended)

To speed up Docker builds, include a `.dockerignore` file:
```
__pycache__/
*.pyc
*.pyo
*.pyd
venv/
.env
.git/
.DS_Store
*.sqlite3
```

---

## 📌 Notes

- This backend supports integration with a separate Vue or React frontend.
- The ML models can be retrained using the `service/model.py` module.
- Default database uses SQLite (`sql_app.db`), which is ideal for local testing.

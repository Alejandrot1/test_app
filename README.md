# FastAPI and React App

## Overview

This project consists of a FastAPI backend and a minimal React frontend.

### Backend

- The backend is built using FastAPI and runs on port 8000.
- It provides an endpoint to create items.

### Frontend

- The frontend is built using React and runs on port 3000.
- It allows users to create items and send them to the FastAPI backend.

### Getting Started

1. **Backend Setup**:
   - Navigate to the `backend` directory.
   - Install the dependencies:
          pip install -r requirements.txt
        - Run the FastAPI server:
          uvicorn main:app --reload
     
2. **Frontend Setup**:
   - Navigate to the `frontend` directory.
   - Install the dependencies:
          npm install
        - Start the React app:
          npm start
     
3. **Access the Application**:
   - Open your browser and navigate to `http://localhost:3000` to access the React frontend.
   - The FastAPI backend can be accessed at `http://localhost:8000`.
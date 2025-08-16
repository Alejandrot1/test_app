# Flask Web App

## Build and Run

1. Build the Docker image:
      docker build -t flask-web-app .
   
2. Run the Docker container:
      docker run -p 5000:5000 flask-web-app
   
3. Access the app at `http://localhost:5000`.

## Database Setup

- The app uses SQLite for user management. The database will be created automatically on the first run.
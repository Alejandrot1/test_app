## GitHub Actions Deployment

This project includes a GitHub Actions workflow for deploying to Google Cloud Run.

### Secrets Required

- `GCP_PROJECT_ID`: Your Google Cloud project ID.
- `GCP_SA_KEY`: Base64 encoded service account key JSON for authentication.

### Deployment Steps

1. Push changes to the `main` branch.
2. The workflow will automatically build and deploy the application to Google Cloud Run.
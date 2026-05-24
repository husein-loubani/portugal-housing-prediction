# Deployment guide: Portugal Housing Price Prediction

A step-by-step walkthrough for first-time deployers. Covers local Docker testing
and Google Cloud Run.

---

## Prerequisites

- A Docker engine. Either Docker Desktop (https://www.docker.com/products/docker-desktop/)
  or colima (`brew install colima docker`). This project was tested with colima.
- A Google account for GCP (the free tier covers this, $300 in credit)
- A trained model at `deployment/model.pkl` (the notebook saves it there)

> This project has been verified locally: the image builds (~715 MB), the
> container runs, and `/health` and `/predict` both respond correctly.

---

## Part 1: local Docker testing

Test locally before pushing anything to the cloud.

### 0. Start the Docker engine

If you use colima (not Docker Desktop), start the engine first:

```bash
colima start
docker info   # confirms the daemon is up
```

### 1. Build the Docker image

```bash
cd deployment/
docker build -t portugal-housing-api .
```

### 2. Run the container locally

```bash
docker run -p 8080:8080 portugal-housing-api
```

You should see uvicorn start and listen on port 8080.

### 3. Test the endpoint

In a new terminal:

```bash
curl http://localhost:8080/health
# Expected: {"status":"ok"}

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"total_area": 100, "parking": 1, "construction_year": 2010,
       "total_rooms": 4, "living_area": 80, "number_of_bathrooms": 2,
       "district": "Lisboa", "city": "Lisboa", "town": "Arroios",
       "type": "Apartment", "energy_certificate": "B", "elevator": true}'
# Expected: {"predicted_price": 285000.0, "currency": "EUR"}  (price varies)
```

### 4. Check the auto-generated API docs

Open http://localhost:8080/docs in a browser. FastAPI exposes an interactive Swagger UI where you can fire off requests without writing curl by hand.

### 5. Stop the container

Press Ctrl+C in the terminal running Docker, or:

```bash
docker ps                       # find the container ID
docker stop <container-id>
```

---

## Part 2: Google Cloud Run deployment

### 1. Create a GCP account

Sign up at https://cloud.google.com/free. You get $300 in credits, valid for 90 days. A credit card is required for verification, but you are not charged unless you exceed the free tier.

### 2. Install the gcloud CLI

```bash
# macOS (Homebrew)
brew install --cask google-cloud-sdk

# Or download the installer:
# https://cloud.google.com/sdk/docs/install
```

After install, restart your terminal.

### 3. Authenticate and create a project

```bash
gcloud auth login
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Create a project (the project ID must be globally unique)
gcloud projects create portugal-housing-PROJECT_ID --name="Portugal Housing API"

# Set this project as default
gcloud config set project portugal-housing-PROJECT_ID
```

Replace `PROJECT_ID` with something unique (e.g., your name + a number).

### 4. Link a billing account

Cloud Run still requires a billing account even when usage stays inside the free tier:

```bash
# List billing accounts
gcloud billing accounts list

# Link the project (replace BILLING_ID with the ID from above)
gcloud billing projects link portugal-housing-PROJECT_ID --billing-account=BILLING_ID
```

### 5. Enable required APIs

```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com
```

### 6. Create an Artifact Registry repository

```bash
gcloud artifacts repositories create housing-repo \
  --repository-format=docker \
  --location=europe-west1 \
  --description="Portugal Housing API container images"
```

### 7. Build and push the Docker image

```bash
# Tag the image for Artifact Registry
docker build -t europe-west1-docker.pkg.dev/portugal-housing-PROJECT_ID/housing-repo/api:latest .

# Push it
docker push europe-west1-docker.pkg.dev/portugal-housing-PROJECT_ID/housing-repo/api:latest
```

The first push takes a few minutes; the image is around 500 MB.

### 8. Deploy to Cloud Run

```bash
gcloud run deploy portugal-housing-api \
  --image=europe-west1-docker.pkg.dev/portugal-housing-PROJECT_ID/housing-repo/api:latest \
  --region=europe-west1 \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --port=8080 \
  --max-instances=2
```

The `--max-instances=2` cap is the safety net against runaway costs. Once deployment finishes, gcloud prints a URL like:

```
Service URL: https://portugal-housing-api-xxxxx-ew.a.run.app
```

### 9. Test the deployed endpoint

```bash
SERVICE_URL=https://portugal-housing-api-xxxxx-ew.a.run.app  # replace with your URL

curl $SERVICE_URL/health

curl -X POST $SERVICE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"total_area": 100, "parking": 1, "construction_year": 2010,
       "total_rooms": 4, "living_area": 80, "number_of_bathrooms": 2,
       "district": "Lisboa", "city": "Lisboa", "town": "Arroios",
       "type": "Apartment", "energy_certificate": "B", "elevator": true}'
```

Visit `$SERVICE_URL/docs` in a browser for the interactive API.

### 10. Monitor and view logs

```bash
# View recent logs
gcloud run services logs read portugal-housing-api --region=europe-west1 --limit=50

# Or use the GCP web console:
# https://console.cloud.google.com/run
```

---

## Part 3: cleanup (avoid charges)

Cloud Run is per-request, but the container image in Artifact Registry takes up storage that does count. To leave nothing billable behind:

```bash
# Delete the Cloud Run service
gcloud run services delete portugal-housing-api --region=europe-west1 --quiet

# Delete the Artifact Registry repository (and the image inside)
gcloud artifacts repositories delete housing-repo --location=europe-west1 --quiet

# Optionally, delete the entire project (removes everything)
gcloud projects delete portugal-housing-PROJECT_ID
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Permission denied` on docker push | Not authenticated to Artifact Registry | `gcloud auth configure-docker europe-west1-docker.pkg.dev` |
| Container fails to start on Cloud Run | Model file missing or too big | Verify `model.pkl` exists and image build succeeded; increase `--memory` if needed |
| `Service Unavailable` on first request | Cold start (Cloud Run scales to zero) | Wait 5-10 seconds and retry; subsequent requests are fast |
| Billing not enabled | Account verification incomplete | Visit https://console.cloud.google.com/billing |

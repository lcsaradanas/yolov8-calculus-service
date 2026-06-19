# Dental Calculus Severity Classification Service

A FastAPI-based inference service that uses a YOLOv8 model to detect and classify dental calculus severity from uploaded dental images. The service processes images, performs object detection using a trained YOLOv8 model, and returns the detected calculus count together with an oral health assessment.

---

## Features

* Dental calculus detection using YOLOv8
* Image processing and inference through FastAPI
* Docker containerization for portable deployment
* Automated CI/CD pipeline using GitHub Actions
* Automatic cloud deployment through Render
* Linting and Docker build validation before deployment

---

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dockerignore
├── .flake8
├── .gitignore
├── Dockerfile
├── app.py
├── mbest.pt
├── requirements.txt
├── requirements-full.txt
```

---

## Technology Stack

### Backend

* Python 3.11
* FastAPI
* Uvicorn

### Machine Learning

* YOLOv8
* PyTorch
* OpenCV

### DevOps

* Docker
* GitHub Actions
* Render

---

## API Overview

### Health Check

```http
GET /
```

Returns a response indicating that the service is running.

### Image Prediction

```http
POST /predict
```

Accepts a dental image and returns:

* Calculus detection results
* Calculus count
* Confidence values
* Oral health classification

---

## Local Development

### Clone Repository

```bash
git clone <repository-url>
cd yolov8-calculus-service
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
uvicorn app:app --reload
```

Application will be available at:

```text
http://localhost:8000
```

---

## Docker Deployment

### Build Image

```bash
docker build -t dental-calculus-service .
```

### Run Container

```bash
docker run -p 8000:8000 dental-calculus-service
```

---

## CI/CD Pipeline

The project uses GitHub Actions to automate code quality checks and deployment validation.

### Pipeline Stages

1. Source code checkout
2. Python environment setup
3. Dependency installation
4. Flake8 linting
5. Docker image build validation
6. Automatic deployment through Render

Workflow file:

```text
.github/workflows/ci.yml
```

---

## Deployment

The application is deployed using Render with Docker.

Deployment workflow:

```text
Developer Push
      ↓
GitHub Repository
      ↓
GitHub Actions
      ↓
Flake8 Linting
      ↓
Docker Build Validation
      ↓
Render Auto Deployment
      ↓
Production Service
```

---

## Model Information

Model File:

```text
mbest.pt
```

Model Type:

* YOLOv8
* Custom-trained dental calculus detection model
* Roboflow Dataset: https://universe.roboflow.com/dental-iex5i/calculus-tzg18


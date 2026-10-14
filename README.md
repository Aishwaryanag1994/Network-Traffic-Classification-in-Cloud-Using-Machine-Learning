# Network Traffic Classification in Cloud Using Machine Learning

## Overview
This project focuses on classifying network traffic in cloud environments using machine learning. It combines traffic classification models with containerized deployment and monitoring tools to support traffic analysis and cloud infrastructure observability.

## Key Features
- Machine learning-based network traffic classification
- KMeans clustering and Isolation Forest anomaly detection
- Random Forest traffic classification models
- Docker-based containerization
- Kubernetes deployment and autoscaling configurations
- Monitoring and visualization using Prometheus, Grafana, and the ELK Stack

## Technologies Used
- **Programming:** Python
- **Machine Learning:** Scikit-learn, KMeans, Isolation Forest, Random Forest
- **Containerization:** Docker
- **Orchestration:** Kubernetes
- **Monitoring:** Prometheus, Grafana, Elasticsearch, Logstash, Kibana
- **API:** Python-based API

## Project Structure
- `src/` — application, data processing, feature engineering, and training code
- `ml/` — machine learning scripts and evaluation utilities
- `models/` — saved machine learning models
- `k8s/` and `kubernetes/` — deployment, autoscaling, and monitoring configurations
- `configs/` — monitoring configuration
- `elk/` and `kibana/` — log collection and dashboard configuration
- `reports/` — evaluation metrics and visualizations

## Setup
1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the project configuration to prepare the required machine learning models and datasets.
4. Configure the application and monitoring components before running the services.

## Notes
The large model file `models/rf_protocol_classifier.joblib` is excluded from this repository because it exceeds GitHub's standard file size limit. Generate it using the project's training scripts or arrange separate model storage before running components that require it.

## Author
Aishwarya Nagesh
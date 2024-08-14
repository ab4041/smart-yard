# Walmart Supply Chain Yard Management System

## Overview

This project leverages computer vision technology to enhance logistics operations at Walmart's supply chain yards. The system uses high-resolution cameras for real-time visual monitoring of trucks and cargo. It employs advanced image analysis algorithms to extract critical information, such as truck identifiers, license plates, and cargo details. This data is continuously processed, logged, and reported to improve operational efficiency, security, and compliance.

#

## Project Details

This project aims to create a robust system for real-time monitoring and management of logistics operations. By leveraging advanced technologies, it provides:


- Continuous data capture and processing.
- Real-time insights and automated reporting.
- Enhanced operational efficiency and security.

<br> 

## Features

- **Operational Efficiency Optimization**:
  - Real-time truck arrival and departure updates.
  - Traffic pattern and truck speed analysis for route optimization.

- **Cargo and Security Management**:
  - Verification and logging of cargo details against shipping manifests.
  - Real-time monitoring to detect unauthorized access.

- **Workflow and Compliance Monitoring**:
  - Tracking and logging compliance with transportation regulations.
  - Identifying inefficiencies in workflows.

- **Maintenance and Diagnostics**:
  - Predictive maintenance alerts for truck usage and performance.



<br> 

###  Architecture and Components

1. **Data Capture**:
   - **CCTV Cameras**: High-resolution cameras capture real-time video feeds of trucks entering the yard.
   - **RFID Scanners and IoT Sensors**: Capture additional data related to truck and cargo information.

2. **Data Streaming**:
   - **Apache Kafka**: Manages the real-time streaming of data from CCTV cameras, RFID scanners, and IoT sensors.

3. **Data Storage**:
   - **Snowflake / Spark Databricks / Oracle**: Stores the continuously captured and processed data in a data warehouse.

4. **Data Processing**:
   - **Amazon SageMaker /Google Vertex AI /Metaflow**: Performs continuous data modeling using computer vision technologies.
   - **OpenCV, YOLO v8, Ultra Analytics**: Processes images to detect and track trucks and cargo, extract license plates and labels, and analyze data.

5. **Data Logging and Reporting**:
   - **PostgreSQL**: Stores extracted information in a structured log table for easy access and reporting.

6. **Load Balancer**:
   - **Automatic Scaling**: Increases instances based on the volume of trucks and data processing requirements.


<br>

## Installation

### Python Requirements

To set up the environment for this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/deveshruttala/yard-management-ai.git
   cd your-repository
   ```
2. **Create a Virtual Environment (optional but recommended)** : 
   ```
   python -m venv venv source venv/bin/activate  # On Windows use `venv\Scripts\activate`

   ```
3. **Install Requirements** : 
   ```
   pip install -r requirements.txt
   ```


### Running the Main File 

The main Python file for this project is main.py. It handles the overall data flow,processing with computer vision models, and logging data.
<br>

To run the application, use:
   ```
   python main.py
   streamlit run main.py
   ```


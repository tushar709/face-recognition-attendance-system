# Face Recognition Attendance System

A Python-based Face Recognition Attendance System using OpenCV.

## Features

- Face Detection
- Face Recognition
- Attendance Marking
- Date and Time Recording
- CSV File Storage
- Real-Time Webcam Detection

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas

## Project Structure

```bash
Attendance_System/
│
├── dataset/
├── trainer/
├── Attendance.csv
├── train.py
├── attendance.py
├── main.py
└── README.md
```

## Installation

Install required libraries:

```bash
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install pandas
```

## How to Run

### Step 1: Collect Face Data

```bash
python train.py
```

### Step 2: Train the Model

```bash
python attendance.py
```

### Step 3: Start Attendance System

```bash
python main.py
```

## Output

- Detects face using webcam
- Recognizes candidate
- Displays name
- Saves attendance with date and time

## Future Improvements

- GUI Interface
- Database Integration
- Cloud Attendance
- Email Notification
- Deep Learning Face Recognition

## Author

Tushar Sen

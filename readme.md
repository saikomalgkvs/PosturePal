# POSE - Posture Observation and Smart Evaluation  

**PosturePal** is a smart application designed to monitor and improve your posture using real-time webcam analysis and machine learning. It provides real-time feedback, customizable alerts, and detailed statistics to help users maintain good posture and prevent health issues caused by prolonged bad posture.  

## Features  

1. **Real-Time Posture Monitoring**  
    - Detects good and bad posture using a webcam and Mediapipe.  
    - Provides real-time feedback on posture status.  

2. **Customizable Alerts**  
    - Configurable alert duration for posture correction.  
    - Notifications to remind users to correct their posture.  

3. **Custom Posture Data Collection**  
    - Allows users to capture their own good and bad posture data.  
    - Trains a personalized machine learning model for advanced posture detection.  

4. **Statistics and Visualization**  
    - Displays total monitoring time, good posture time, and bad posture time.  
    - Includes a pie chart for posture breakdown.  
    - Weekly activity bar graph for posture trends.  

5. **Settings Customization**  
    - Toggle notifications and monitoring modes (Normal/Advanced).  
    - Save and load user preferences.  

## Technical Details  

1. **Programming Language**  
    - Python  

2. **Libraries and Frameworks**  
    - GUI: CustomTkinter  
    - Machine Learning: Scikit-learn  
    - Pose Detection: Mediapipe  
    - Data Visualization: Matplotlib, mplcursors  
    - Notifications: Plyer  

3. **Database**  
    - SQLite for storing daily posture statistics.  

## Installation  

1. **Prerequisites**  
    - Python 3.10 or higher installed on your system.  

2. **Clone the Repository**  
    ```bash  
    git clone https://github.com/your-repo/posturepal.git  
    cd posturepal  
    ```  

3. **Install Dependencies**  
    Install the required Python libraries using pip:  
    ```bash  
    pip install -r requirements.txt  
    ```  

4. **Run the Application**  
    Run the application using the following command:  
    ```bash  
    python app.py  
    ```  

## Usage  

1. **Monitor Tab**  
    - Click "Start Monitoring" to begin posture monitoring.  
    - View real-time posture status (Good/Bad).  

2. **Statistics Tab**  
    - View total monitoring time, good posture time, and bad posture time.  
    - Analyze posture trends using the pie chart and weekly activity bar graph.  

3. **Settings Tab**  
    - Customize alert duration and toggle notifications.  
    - Switch between Normal and Advanced monitoring modes.  

4. **Custom Posture Data**  
    - Capture good and bad posture data in the "Capture Posture Data" window.  
    - Train a personalized machine learning model for posture detection.  
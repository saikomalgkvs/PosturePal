import pickle
import time
import tkinter as tk
import customtkinter as ctk  # Import CustomTkinter
from tkinter import messagebox
import threading
import cv2
import mediapipe as mp
import pandas as pd
import os
import json
from plyer import notification
from timer import Timer
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import timedelta
import mplcursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from captureData import capture_data
from training import train_model
import sqlite3
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PosturePalApp:
    def __init__(self, root):
        # Set the theme and appearance mode
        ctk.set_appearance_mode("Dark")  # Options: "Dark", "Light", "System"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        self.root = root
        self.root.title("POSE - Posture Observation and Smart Evaluation")
        self.root.geometry("1400X800")  # Adjusted for better layout
        self.root.iconbitmap(os.path.join(BASE_DIR, "assets", "icon.ico"))
        
        self.root.resizable(False, False) #NOT RESIZEABLE

        self.is_monitoring = False
        self.total_time = 0
        self.good_posture_time = 0
        self.corrections = 0
        self.alert_duration = 5  # seconds

        self.good_posture_data = []
        self.bad_posture_data = []
        self.model = None

        self.start_time = 0
        self.good_time = 0
        self.bad_time = 0

        # Paths
        self.SETTINGS_FILE = "settings.json"
        self.MODEL_FILE = os.path.join(BASE_DIR, "models", "posture_model.pkl")

        if os.path.exists(self.MODEL_FILE):
            with open(self.MODEL_FILE, "rb") as model_file:
                self.model = pickle.load(model_file)

        "Initialize the database"
        self.initialize_database()
        self.initialize_statistics()
        
        # Create a modern tabbed interface
        self.notebook = ctk.CTkTabview(self.root, width=800, height=500)
        self.notebook.pack(fill="both", expand=True)

        self.monitor_tab = self.notebook.add("Monitor")
        self.stats_tab = self.notebook.add("Statistics")
        self.settings_tab = self.notebook.add("Settings")

        """ Monitor Tab """
        self.status_label = ctk.CTkLabel(self.monitor_tab, text="Click 'Start Monitoring' to begin.", font=("Roboto", 16))
        self.status_label.pack(pady=20)

        self.toggle_button = ctk.CTkButton(
            self.monitor_tab,
            text="Start Monitoring",
            font=("Roboto", 14),
            fg_color="blue",
            text_color="white",
            command=self.toggle_monitoring,
        )
        self.toggle_button.pack(pady=10)

        self.posture_status_label = ctk.CTkLabel(self.monitor_tab, text="Posture Status: None", font=("Roboto", 14))
        self.posture_status_label.pack(pady=5)
        
        """ Statistics Tab """
        self.stats_frame = ctk.CTkFrame(self.stats_tab)
        self.stats_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.total_time_label = ctk.CTkLabel(self.stats_frame, text="Total Monitoring Time: {}".format(self.time_string(self.total_time)), font=("Roboto", 14))
        self.total_time_label.grid(row=0, column=0, padx=5, pady=5, columnspan=2)

        self.good_posture_label = ctk.CTkLabel(self.stats_frame, text="Good Posture Time: {}".format(self.time_string(self.good_posture_time)), font=("Roboto", 14))
        self.good_posture_label.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        self.bad_posture_label = ctk.CTkLabel(self.stats_frame, text="Bad Posture Time: {}".format(self.time_string(self.bad_time)), font=("Roboto", 14))
        self.bad_posture_label.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        self.corrections_label = ctk.CTkLabel(self.stats_frame, text="Corrections: {}".format(self.corrections), font=("Roboto", 14))
        self.corrections_label.grid(row=3, column=0, padx=5, pady=5, columnspan=2)

        #create a pie chart to show the percentage of good and bad posture
        self.pie_chart_label = ctk.CTkLabel(self.stats_frame, text="Posture Breakdown", font=("Roboto", 16))
        self.pie_chart_label.grid(row=4, column=0, padx=5, pady=5)
        self.pie_chart_canvas = None

        #weekly activity graph
        self.weekly_activity_label = ctk.CTkLabel(self.stats_frame, text="Weekly Activity", font=("Roboto", 16))
        self.weekly_activity_label.grid(row=4, column=1, padx=5, pady=5)
        self.weekly_activity_canvas = None

        self.update_pie_chart()
        self.update_barchart()

        """ Settings Tab """
        self.settings_frame = ctk.CTkFrame(self.settings_tab)
        self.settings_frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.alert_duration_label = ctk.CTkLabel(self.settings_frame, text="Alert Duration (seconds):", font=("Roboto", 14))
        self.alert_duration_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.alert_duration_var = ctk.IntVar(value=self.alert_duration)
        self.alert_duration_entry = ctk.CTkEntry(self.settings_frame, textvariable=self.alert_duration_var, width=100)
        self.alert_duration_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.update_alert_button = ctk.CTkButton(
            self.settings_frame, text="Update", command=self.update_alert_duration, fg_color="green"
        )
        self.update_alert_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        self.monitoring_mode_label = ctk.CTkLabel(self.settings_frame, text="Monitoring Mode:", font=("Roboto", 14))
        self.monitoring_mode_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.monitoring_mode_var = ctk.StringVar(value="normal")
        self.normal_mode_radio = ctk.CTkRadioButton(
            self.settings_frame, text="Normal Mode", variable=self.monitoring_mode_var, value="normal"
        )
        self.normal_mode_radio.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.advanced_mode_radio = ctk.CTkRadioButton(
            self.settings_frame, text="Advanced Mode", variable=self.monitoring_mode_var, value="advanced"
        )
        self.advanced_mode_radio.grid(row=1, column=2, padx=10, pady=10, sticky="w")

        self.notifications_label = ctk.CTkLabel(self.settings_frame, text="Enable Notifications:", font=("Roboto", 14))
        self.notifications_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.notifications_var = ctk.BooleanVar(value=True)
        self.notifications_check = ctk.CTkCheckBox(self.settings_frame, text="" ,variable=self.notifications_var)
        self.notifications_check.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.capture_data_label = ctk.CTkLabel(
            self.settings_frame, text="Customize your Good and Bad Posture Data", font=("Roboto", 14)
        )
        self.capture_data_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")

        self.capture_data_button = ctk.CTkButton(
            self.settings_frame, text="Capture Posture Data", command=self.customize_posture, fg_color="blue"
        )
        self.capture_data_button.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        """ Function Calls """
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.SETTINGS_FILE):
            with open(self.SETTINGS_FILE, "r") as settings_file:
                settings = json.load(settings_file)
                self.alert_duration = settings.get("alert_duration", 5)
                self.alert_duration_var.set(self.alert_duration)
                self.notifications_var.set(settings.get("notifications", True))
                self.monitoring_mode_var.set(settings.get("monitoring_mode", "normal"))
        else:
            # Default settings if the file does not exist
            self.alert_duration_var.set(self.alert_duration)
            self.notifications_var.set(True)
            self.monitoring_mode_var.set("normal")

    def initialize_database(self):
        self.conn = sqlite3.connect(os.path.join(BASE_DIR, "posture_data.db"))
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS posture_data (
                date TEXT PRIMARY KEY,
                good_posture_duration INTEGER DEFAULT 0,
                bad_posture_duration INTEGER DEFAULT 0,
                total_monitoring_time INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def update_database(self):
        today = date.today().strftime("%Y-%m-%d")
        
        # Calculate the new elapsed times
        new_good_posture_time = self.good_time.get_elapsed_time()
        new_total_time = self.start_time.get_elapsed_time()
        new_bad_time = new_total_time - new_good_posture_time
    
        # Use the existing connection and cursor
        self.cursor.execute("SELECT * FROM posture_data WHERE date = ?", (today,))
        row = self.cursor.fetchone()
        
        if row:
            # Update the existing row with only the new elapsed times
            self.cursor.execute("""
            UPDATE posture_data
            SET good_posture_duration = good_posture_duration + ?,
                bad_posture_duration = bad_posture_duration + ?,
                total_monitoring_time = total_monitoring_time + ?
            WHERE date = ?
            """, (int(new_good_posture_time), int(new_bad_time), int(new_total_time), today))
        else:
            # Insert a new row
            self.cursor.execute("""
            INSERT INTO posture_data (date, good_posture_duration, bad_posture_duration, total_monitoring_time)
            VALUES (?, ?, ?, ?)
            """, (today, int(new_good_posture_time), int(new_bad_time), int(new_total_time)))
        
        self.conn.commit()

    def fetch_weekly_data(self):
        # Get the current week's dates
        today = date.today()
        week_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    
        # Fetch data from the database
        weekly_data = []
        for day in week_dates:
            self.cursor.execute("SELECT good_posture_duration, bad_posture_duration FROM posture_data WHERE date = ?", (day,))
            row = self.cursor.fetchone()
            if row:
                weekly_data.append((day, row[0], row[1]))  # (date, good_posture_duration, bad_posture_duration)
            else:
                weekly_data.append((day, 0, 0))  # Default to 0 if no data exists for the day
    
        return weekly_data

    def update_barchart(self):
        # Fetch weekly data
        weekly_data = self.fetch_weekly_data()

        # Extract data for the graph
        days = [data[0] for data in weekly_data]
        good_posture = [data[1] for data in weekly_data] 
        bad_posture = [data[2] for data in weekly_data]

        # Create the bar graph
        figure, ax = plt.subplots(figsize=(7.5, 4))
        bar_width = 0.4
        x = range(len(days))

        good_bars = ax.bar(x, good_posture, bar_width, label="Good Posture", color="green")
        bad_bars = ax.bar(x, bad_posture, bar_width, bottom=good_posture, label="Bad Posture", color="red")

        # Customize the graph
        ax.set_title("Weekly Activity", fontsize=16)
        ax.set_xlabel("Days", fontsize=12)
        ax.set_ylabel("Time (minutes)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([day[5:] for day in days])  # Show only MM-DD
        ax.legend()

        # Add interactivity with mplcursors
        cursor = mplcursors.cursor([good_bars, bad_bars], hover=True)
        @cursor.connect("add")
        def on_add(sel):
            index = sel.index
            day = days[index]
            sel.annotation.set_text(
                f"{day}\nGood Posture: {self.time_string(good_posture[index])}\nBad Posture: {self.time_string(bad_posture[index])}"
            )
        if self.weekly_activity_canvas is not None:
            self.weekly_activity_canvas.get_tk_widget().destroy()
            self.weekly_activity_canvas = None

        # Embed the graph in the Tkinter app
        self.weekly_activity_canvas = FigureCanvasTkAgg(figure, self.stats_frame)
        self.weekly_activity_canvas.get_tk_widget().grid(row=5, column=1, padx=5, pady=5)
        self.weekly_activity_canvas.draw()

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def save_settings(self):
        settings = {
            "alert_duration": self.alert_duration_var.get(),
            "notifications": self.notifications_var.get(),
            "monitoring_mode": self.monitoring_mode_var.get(),
        }
        with open(self.SETTINGS_FILE, "w") as settings_file:
            json.dump(settings, settings_file)

    def customize_posture(self):
        # Create a new window for capturing data
        self.capture_window = ctk.CTkToplevel(self.root)
        self.capture_window.iconbitmap(os.path.join(BASE_DIR, "assets", "icon.ico"))
        self.capture_window.title("Capture Posture Data")
        self.capture_window.geometry("500x300")

        self.capture_window.lift()
        self.capture_window.focus_force()
        
        # Create labels and entry fields for good and bad posture
        caution_label = ctk.CTkLabel(
            self.capture_window,
            text="Make sure you are in good/bad posture before capturing data.",
            text_color="red",
            font=("Roboto", 20),
            pady=10,
            padx=10,
        )
        caution_label.grid(row=0, column=0, columnspan=2, pady=5, padx=10)

        # show the number of good postures
        self.good_posture_count_label = ctk.CTkLabel(self.capture_window, text="Good Postures: {}".format(len(self.good_posture_data)))
        self.good_posture_count_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.good_posture_button = ctk.CTkButton(self.capture_window, text="Capture Good Posture", command=lambda: self.start_capture("good"))
        self.good_posture_button.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.bad_posture_count_label = ctk.CTkLabel(self.capture_window, text="Bad Postures: {}".format(len(self.bad_posture_data)))
        self.bad_posture_count_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.bad_posture_button = ctk.CTkButton(
            self.capture_window, text="Capture Bad Posture", command=lambda: self.start_capture("bad")
        )
        self.bad_posture_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")


        self.train_model_button = ctk.CTkButton(self.capture_window, text="Train Model", command=self.train_model_util)
        self.train_model_button.grid(row=3, column=0, columnspan=2, pady=10, padx=10)

    def start_capture(self, posture_type):
        if posture_type == "good":
            self.good_posture_data.append(capture_data("good", 30))
            self.good_posture_count_label.configure(text="Good Postures: {}".format(len(self.good_posture_data)))
            messagebox.showinfo("Success", "Good posture data captured successfully.")
        elif posture_type == "bad":
            self.bad_posture_data.append(capture_data("bad", 30))
            self.bad_posture_count_label.configure(text="Bad Postures: {}".format(len(self.bad_posture_data)))
            messagebox.showinfo("Success", "Bad posture data captured successfully.")
        else:
            messagebox.showerror("Error", "Invalid posture type.")

    def train_model_util(self):
        if self.good_posture_data and self.bad_posture_data:
            self.model, accuracy = train_model(self.bad_posture_data, self.good_posture_data)
            with open(self.MODEL_FILE, "wb") as model_file:
                pickle.dump(self.model, model_file)
            messagebox.showinfo("Success", "Model saved successfully with accuracy {:.2f}.".format(accuracy*100))
        else:
            messagebox.showerror("Error", "Please capture good and bad posture data first.")

    def toggle_monitoring(self):
        if self.is_monitoring:
            self.is_monitoring = False
            time.sleep(1)  # Give some time for the thread to stop
            self.toggle_button.configure(text="Start Monitoring", fg_color="blue")
            self.status_label.configure(text="Monitoring Stopped.")
            self.posture_status_label.configure(text="Posture Status: None", text_color="white")
            self.posture_status_label.update()
            self.update_statistics()
            self.update_database()  # Update the database with the monitoring data

        else:
            self.is_monitoring = True
            self.toggle_button.configure(text="Stop Monitoring", fg_color="red")
            self.status_label.configure(text="Loading...")
            threading.Thread(target=self.start_monitoring).start()
            self.start_time = Timer()
            self.start_time.start()
            self.good_time = Timer()
            self.good_time.start()

    def start_monitoring(self):
        cap = cv2.VideoCapture(0)
        thresholds = [0.25, 2]
        self.status_label.configure(text="Monitoring...")

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        leaning_forward_start_time = None
        leaning_forward_notified = False

        while self.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                head = landmarks[mp_pose.PoseLandmark.NOSE]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

                origin_x, origin_y, origin_z = head.x, head.y, head.z
                row = {
                    "left_ear_x": landmarks[mp_pose.PoseLandmark.LEFT_EAR].x - origin_x,
                    "left_ear_y": landmarks[mp_pose.PoseLandmark.LEFT_EAR].y - origin_y,
                    "left_ear_z": landmarks[mp_pose.PoseLandmark.LEFT_EAR].z - origin_z,
                    "right_ear_x": landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x - origin_x,
                    "right_ear_y": landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y - origin_y,
                    "right_ear_z": landmarks[mp_pose.PoseLandmark.RIGHT_EAR].z - origin_z,
                    "left_mouth_x": landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].x - origin_x,
                    "left_mouth_y": landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].y - origin_y,
                    "left_mouth_z": landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].z - origin_z,
                    "right_mouth_x": landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].x - origin_x,
                    "right_mouth_y": landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].y - origin_y,
                    "right_mouth_z": landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].z - origin_z,
                    "left_shoulder_x": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - origin_x,
                    "left_shoulder_y": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y - origin_y,
                    "left_shoulder_z": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z - origin_z,
                    "right_shoulder_x": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x - origin_x,
                    "right_shoulder_y": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - origin_y,
                    "right_shoulder_z": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z - origin_z,
                }
                # Load the posture model
                posture_model = self.model

                row_df = pd.DataFrame([row])  # Convert the row data to a DataFrame for prediction

                mode = self.monitoring_mode_var.get()

                isIncorrectPosture = False

                if mode == "normal":
                    isIncorrectPosture = (shoulder_y - head.y < thresholds[0]) or (abs(head.z) > thresholds[1])
                elif mode == "advanced" and posture_model is not None:
                    predicted_posture = posture_model.predict(row_df)[0]
                    isIncorrectPosture = predicted_posture == 0
                else:
                    messagebox.showerror("Error", "Model not trained. Please capture data first.")
                    return

                # Check if the predicted posture is bad
                if isIncorrectPosture:
                    self.good_time.pause()

                    self.posture_status_label.configure(text="Posture Status: Bad Posture", text_color="red")
                    self.posture_status_label.update()

                    if leaning_forward_start_time is None:
                        leaning_forward_start_time = Timer()
                        leaning_forward_start_time.start()

                    elif leaning_forward_start_time.get_elapsed_time() >= self.alert_duration and not leaning_forward_notified:
                        if self.notifications_var.get():
                            notification.notify(title="Posture Alert", message="Correct you posture!", timeout=5)

                        self.corrections += 1
                        self.corrections_label.configure(text="Corrections: {}".format(self.corrections))
                        self.corrections_label.update()

                        leaning_forward_notified = True

                else:
                    self.good_time.resume()

                    leaning_forward_start_time = None
                    leaning_forward_notified = False

                    self.posture_status_label.configure(text="Posture Status: Good Posture", text_color="green")
                    self.posture_status_label.update()

    def initialize_statistics(self):
        # update the total_time, good_posture_time, and bad_posture_time with data from the database with today's date
        today = date.today().strftime("%Y-%m-%d")
        self.cursor.execute("SELECT * FROM posture_data WHERE date = ?", (today,))
        row = self.cursor.fetchone()
        if row:
            self.total_time = row[3]
            self.good_posture_time = row[1]
            self.bad_time = row[2]

    def update_statistics(self):

        if self.start_time is not None:
            self.total_time += self.start_time.get_elapsed_time()
            self.good_posture_time += self.good_time.get_elapsed_time()

            self.total_time_label.configure(text="Total Monitoring Time: {}".format(self.time_string(self.total_time)))
            self.total_time_label.update()

            self.good_posture_label.configure(text="Good Posture Time: {}".format(self.time_string(self.good_posture_time)))
            self.good_posture_label.update()

            # Update bad posture time
            self.bad_time = self.total_time - self.good_posture_time
            self.bad_posture_label.configure(text="Bad Posture Time: {}".format(self.time_string(self.bad_time)))
            self.bad_posture_label.update()

    def update_pie_chart(self):
        # Calculate percentages for the pie chart
        if self.total_time > 0:
            good_percentage = (self.good_posture_time / self.total_time) * 100
            bad_percentage = (self.bad_time / self.total_time) * 100
        else:
            good_percentage = 0
            bad_percentage = 0

        # Create the pie chart
        figure = Figure(figsize=(6, 4), dpi=100)
        subplot = figure.add_subplot(111)
        subplot.pie(
            [good_percentage, bad_percentage],
            labels=["Good Posture", "Bad Posture"],
            autopct="%1.1f%%",
            colors=["green", "red"],
            startangle=90,
        )
        subplot.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.

        if self.pie_chart_canvas is not None:
            self.pie_chart_canvas.get_tk_widget().destroy()
            self.pie_chart_canvas = None

        self.pie_chart_canvas = FigureCanvasTkAgg(figure, self.stats_frame)
        self.pie_chart_canvas.get_tk_widget().grid(row=5, column=0, padx=5, pady=5)
        self.pie_chart_canvas.draw()
        
    def update_alert_duration(self):
        try:
            new_duration = self.alert_duration_var.get()
            if new_duration > 0:
                self.alert_duration = new_duration
                self.save_settings()
                messagebox.showinfo("Success", f"Alert duration updated to {self.alert_duration} seconds.")
            else:
                messagebox.showerror("Error", "Please enter a positive number.")
        except tk.TclError:
            messagebox.showerror("Error", "Invalid input. Please enter a valid number.")

    def toggle_monitoring_mode(self):
        self.save_settings()

    def toggle_notifications(self):
        self.save_settings()

    def time_string(self, seconds):
        """Convert seconds to a string format."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h : {int(minutes)}m : {int(seconds)}s"

              
if __name__ == "__main__":
    root = ctk.CTk()
    app = PosturePalApp(root)
    root.mainloop()
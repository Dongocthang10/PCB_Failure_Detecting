import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class PCBDefectDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PCB Defect Detection System")
        self.root.geometry("1000x700")
        self.root.resizable(False, False)
        
        # Variables
        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="output")
        self.source_type = tk.StringVar(value="video")
        self.camera_index = tk.IntVar(value=0)
        
        self.is_running = False
        self.is_paused = False
        self.detection_thread = None
        self.cap = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # ========== HEADER ==========
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🔍 PCB Defect Detection System",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # ========== MAIN CONTAINER ==========
        main_container = tk.Frame(self.root, bg="#ecf0f1")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT PANEL - Settings
        left_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        settings_label = tk.Label(
            left_panel,
            text="⚙️ Settings",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        settings_label.pack(pady=10)
        
        # Model Selection
        self.create_file_selector(
            left_panel,
            "YOLO Model:",
            self.model_path,
            lambda: filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
            )
        )
        
        # Source Type
        source_frame = tk.LabelFrame(left_panel, text="Source Type", bg="white", padx=10, pady=5)
        source_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Radiobutton(
            source_frame,
            text="Video File",
            variable=self.source_type,
            value="video",
            bg="white",
            command=self.toggle_source
        ).pack(anchor=tk.W)
        
        tk.Radiobutton(
            source_frame,
            text="Camera",
            variable=self.source_type,
            value="camera",
            bg="white",
            command=self.toggle_source
        ).pack(anchor=tk.W)
        
        # Video Selection
        self.video_selector_frame = tk.Frame(left_panel, bg="white")
        self.video_selector_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_file_selector(
            self.video_selector_frame,
            "Video File:",
            self.video_path,
            lambda: filedialog.askopenfilename(
                title="Select Video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mkv"), ("All Files", "*.*")]
            ),
            use_pack=True
        )
        
        # Camera Selection
        self.camera_selector_frame = tk.Frame(left_panel, bg="white")
        
        tk.Label(
            self.camera_selector_frame,
            text="Camera Index:",
            bg="white",
            font=("Arial", 9)
        ).pack(anchor=tk.W)
        
        tk.Spinbox(
            self.camera_selector_frame,
            from_=0,
            to=10,
            textvariable=self.camera_index,
            width=10
        ).pack(anchor=tk.W, pady=2)
        
        # Output Directory
        self.create_file_selector(
            left_panel,
            "Output Directory:",
            self.output_dir,
            lambda: filedialog.askdirectory(title="Select Output Directory")
        )
        
        # Control Buttons
        control_frame = tk.Frame(left_panel, bg="white")
        control_frame.pack(fill=tk.X, padx=10, pady=15)
        
        self.start_btn = tk.Button(
            control_frame,
            text="▶ Start",
            command=self.start_detection,
            bg="#27ae60",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=3,
            width=12
        )
        self.start_btn.pack(pady=5)
        
        self.pause_btn = tk.Button(
            control_frame,
            text="⏸ Pause",
            command=self.pause_detection,
            bg="#f39c12",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=3,
            width=12,
            state=tk.DISABLED
        )
        self.pause_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(
            control_frame,
            text="⏹ Stop",
            command=self.stop_detection,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold"),
            relief=tk.RAISED,
            bd=3,
            width=12,
            state=tk.DISABLED
        )
        self.stop_btn.pack(pady=5)
        
        # RIGHT PANEL - Display
        right_panel = tk.Frame(main_container, bg="white", relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        display_label = tk.Label(
            right_panel,
            text="📹 Live Detection",
            font=("Arial", 14, "bold"),
            bg="white"
        )
        display_label.pack(pady=10)
        
        # Video Display
        self.video_label = tk.Label(right_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status Bar
        self.status_frame = tk.Frame(right_panel, bg="#34495e", height=80)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="⚪ Ready",
            font=("Arial", 10),
            bg="#34495e",
            fg="white"
        )
        self.status_label.pack(pady=5)
        
        self.stats_label = tk.Label(
            self.status_frame,
            text="Frame: 0 | PCB: 0 | Errors: 0",
            font=("Arial", 9),
            bg="#34495e",
            fg="#ecf0f1"
        )
        self.stats_label.pack()
        
        # Progress Bar
        self.progress = ttk.Progressbar(
            self.status_frame,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=5)
        
    def create_file_selector(self, parent, label_text, var, command, use_pack=False):
        frame = tk.Frame(parent, bg="white")
        if use_pack:
            frame.pack(fill=tk.X, pady=5)
        else:
            frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            frame,
            text=label_text,
            bg="white",
            font=("Arial", 9)
        ).pack(anchor=tk.W)
        
        entry_frame = tk.Frame(frame, bg="white")
        entry_frame.pack(fill=tk.X)
        
        entry = tk.Entry(
            entry_frame,
            textvariable=var,
            font=("Arial", 9),
            width=25
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=2)
        
        btn = tk.Button(
            entry_frame,
            text="Browse",
            command=lambda: var.set(command()),
            bg="#3498db",
            fg="white",
            font=("Arial", 8),
            relief=tk.RAISED
        )
        btn.pack(side=tk.RIGHT, padx=(5, 0))
        
    def toggle_source(self):
        if self.source_type.get() == "video":
            self.video_selector_frame.pack(fill=tk.X, padx=10, pady=5)
            self.camera_selector_frame.pack_forget()
        else:
            self.video_selector_frame.pack_forget()
            self.camera_selector_frame.pack(fill=tk.X, padx=10, pady=5)
            
    def start_detection(self):
        # Validation
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a YOLO model!")
            return
            
        if self.source_type.get() == "video" and not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file!")
            return
            
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("Error", "Model file not found!")
            return
            
        # Start detection in separate thread
        self.is_running = True
        self.is_paused = False
        
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.progress.start()
        self.update_status("🟢 Running...", "green")
        
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()
        
    def pause_detection(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.update_status("🟡 Paused", "orange")
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.update_status("🟢 Running...", "green")
            
    def stop_detection(self):
        self.is_running = False
        self.is_paused = False
        
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        
        self.progress.stop()
        self.update_status("🔴 Stopped", "red")
        
        if self.cap:
            self.cap.release()
            
    def update_status(self, text, color="white"):
        self.status_label.config(text=text)
        
    def update_stats(self, frame_idx, pcb_idx, error_count):
        self.stats_label.config(
            text=f"Frame: {frame_idx} | PCB: {pcb_idx} | Errors: {error_count}"
        )
        
    def run_detection(self):
        try:
            # Setup directories
            output_dir = self.output_dir.get()
            os.makedirs(f"{output_dir}/logs", exist_ok=True)
            os.makedirs(f"{output_dir}/error_frames", exist_ok=True)
            os.makedirs(f"{output_dir}/error_pcbs", exist_ok=True)
            os.makedirs(f"{output_dir}/heatmap", exist_ok=True)
            
            # Load model
            model = YOLO(self.model_path.get())
            
            # Open video source
            if self.source_type.get() == "video":
                self.cap = cv2.VideoCapture(self.video_path.get())
            else:
                self.cap = cv2.VideoCapture(self.camera_index.get())
                
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open video source!")
                self.stop_detection()
                return
                
            # Log file
            log_file = open(f"{output_dir}/logs/pcb_log.txt", "w", encoding="utf-8")
            log_file.write("=== PCB DEFECT LOG ===\n")
            log_file.write(f"Model: {self.model_path.get()}\n")
            log_file.write(f"Start: {datetime.datetime.now()}\n\n")
            
            # Heatmap init
            ret, sample_frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot read frame!")
                self.stop_detection()
                return
                
            heatmap_acc = np.zeros(
                (sample_frame.shape[0], sample_frame.shape[1]),
                dtype=np.float32
            )
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frame_idx = 0
            pcb_idx = 0
            total_errors = 0
            
            while self.is_running:
                if self.is_paused:
                    continue
                    
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                
                # PCB detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for contour in contours:
                    if cv2.contourArea(contour) < 3000:
                        continue
                        
                    pcb_idx += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    pcb_part = frame[y:y+h, x:x+w]
                    
                    results = model(pcb_part, conf=0.4)
                    annotated = results[0].plot()
                    
                    error_counts = {}
                    
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        cls_name = model.names[cls]
                        error_counts[cls_name] = error_counts.get(cls_name, 0) + 1
                        
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        heatmap_acc[y+by1:y+by2, x+bx1:x+bx2] += 1.0
                        
                    if error_counts:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        total_errors += sum(error_counts.values())
                        
                        log_line = (
                            f"[{timestamp}] "
                            f"Frame:{frame_idx} | PCB:{pcb_idx} | "
                            + " | ".join([f"{k}:{v}" for k, v in error_counts.items()])
                            + "\n"
                        )
                        
                        log_file.write(log_line)
                        log_file.flush()
                        
                        cv2.imwrite(
                            f"{output_dir}/error_frames/frame_{frame_idx:05d}.jpg",
                            frame
                        )
                        
                        cv2.imwrite(
                            f"{output_dir}/error_pcbs/pcb_f{frame_idx:05d}_{pcb_idx:03d}.jpg",
                            annotated
                        )
                        
                    frame[0:h, 0:w] = annotated
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                # Update GUI
                self.update_stats(frame_idx, pcb_idx, total_errors)
                self.display_frame(frame)
                
            # Generate heatmap
            heatmap_norm = cv2.normalize(
                heatmap_acc, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(sample_frame, 0.6, heatmap_color, 0.4, 0)
            
            cv2.imwrite(f"{output_dir}/heatmap/heatmap_result.jpg", overlay)
            
            log_file.write(f"\nEnd: {datetime.datetime.now()}\n")
            log_file.close()
            
            self.cap.release()
            
            messagebox.showinfo("Complete", "Detection completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.stop_detection()
            
    def display_frame(self, frame):
        # Resize frame to fit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        max_width = 640
        max_height = 480
        
        scale = min(max_width/w, max_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
    def on_closing(self):
        if self.is_running:
            if messagebox.askokcancel("Quit", "Detection is running. Do you want to quit?"):
                self.stop_detection()
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PCBDefectDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
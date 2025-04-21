import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from Extract_TDR import s_param_to_Impedance, adjust_Z

class PlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Curve Plotter")
        self.shift_amount = 0  # Tracks horizontal shift
        self.x1_data = None  # Stores x data
        self.x2_data = None  # Stores y data
        self.data1 = None  # Stores loaded data
        self.data2 = None  # Stores loaded data
        self.file_path = None
        self.x_limits = None  # Stores custom x-axis limits
        self.y_limits = None  # Stores custom y-axis limits
        self.shift_step = 1E-8  # Default shift step size
        self.adjust_state = False  # Tracks adjust/reset state

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # File selection button
        self.select_button = tk.Button(self.main_frame, text="Select S-param File", command=self.load_file)
        self.select_button.pack(pady=5)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

        # Bind resize event to adjust figure size
        self.canvas.get_tk_widget().bind('<Configure>', self.on_resize)

        # Arrow buttons frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=5)

        # Left and Right arrow buttons
        self.left_button = tk.Button(self.button_frame, text="← Shift Left", command=self.shift_left)
        self.left_button.pack(side=tk.LEFT, padx=5)
        self.right_button = tk.Button(self.button_frame, text="Shift Right →", command=self.shift_right)
        self.right_button.pack(side=tk.LEFT, padx=5)

        self.adjust_button = tk.Button(self.main_frame, text="adjust curve", command=self.adjust_curve)
        self.adjust_button.pack(pady=5)
    
        # X-axis limit controls
        self.x_limit_frame = tk.Frame(self.main_frame)
        self.x_limit_frame.pack(pady=5)
        tk.Label(self.x_limit_frame, text="X-axis Limits (min, max):").pack(side=tk.LEFT)
        self.x_min_entry = tk.Entry(self.x_limit_frame, width=10)
        self.x_min_entry.pack(side=tk.LEFT, padx=5)
        self.x_max_entry = tk.Entry(self.x_limit_frame, width=10)
        self.x_max_entry.pack(side=tk.LEFT, padx=5)
        self.x_limit_button = tk.Button(self.x_limit_frame, text="Set X Limits", command=self.set_x_limits)
        self.x_limit_button.pack(side=tk.LEFT, padx=5)

        # Y-axis limit controls
        self.y_limit_frame = tk.Frame(self.main_frame)
        self.y_limit_frame.pack(pady=5)
        tk.Label(self.y_limit_frame, text="Y-axis Limits (min, max):").pack(side=tk.LEFT)
        self.y_min_entry = tk.Entry(self.y_limit_frame, width=10)
        self.y_min_entry.pack(side=tk.LEFT, padx=5)
        self.y_max_entry = tk.Entry(self.y_limit_frame, width=10)
        self.y_max_entry.pack(side=tk.LEFT, padx=5)
        self.y_limit_button = tk.Button(self.y_limit_frame, text="Set Y Limits", command=self.set_y_limits)
        self.y_limit_button.pack(side=tk.LEFT, padx=5)

        # Shift step size controls
        self.step_frame = tk.Frame(self.main_frame)
        self.step_frame.pack(pady=5)
        tk.Label(self.step_frame, text="Shift Step Size:").pack(side=tk.LEFT)
        self.step_entry = tk.Entry(self.step_frame, width=10)
        self.step_entry.insert(0, "0.1")  # Default value
        self.step_entry.pack(side=tk.LEFT, padx=5)
        self.step_button = tk.Button(self.step_frame, text="Set Step Size", command=self.set_step_size)
        self.step_button.pack(side=tk.LEFT, padx=5)

    def on_resize(self, event):
        # Adjust figure size based on canvas size and DPI
        widget = self.canvas.get_tk_widget()
        width_px = widget.winfo_width()  # Canvas width in pixels
        height_px = widget.winfo_height()  # Canvas height in pixels
        dpi = self.fig.dpi  # Get figure DPI
        if width_px > 100 and height_px > 100:  # Avoid resizing to invalid dimensions
            # Convert pixels to inches, scaling to fit canvas
            width_in = width_px / dpi * 0.85  # Reduced scaling to ensure fit
            height_in = height_px / dpi * 0.85
            # Cap figure size to avoid excessive scaling
            max_width_in = 20  # Max width in inches
            max_height_in = 15  # Max height in inches
            width_in = min(width_in, max_width_in)
            height_in = min(height_in, max_height_in)
            self.fig.set_size_inches(width_in, height_in)
            self.canvas.draw()

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("S-param file", "*.s4p")])
        if self.file_path:
            try:
                # Assume s_param_to_Impedance returns x1_data, data1, data2, Z0
                (self.x1_data, self.data1, self.data2, Z0) = s_param_to_Impedance(self.file_path)
                self.x2_data = self.x1_data
                self.x2_data = self.x2_data[::-1]  # Reverse x2_data
                self.shift_amount = 0  # Reset shift
                self.adjust_state = False  # Reset adjust state
                self.x_limits = None  # Reset x-axis limits
                self.y_limits = None  # Reset y-axis limits
                self.update_plot()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def adjust_curve(self):
        if self.data1 is not None and self.data2 is not None:
            self.adjust_state = not self.adjust_state  # Toggle state
            if self.adjust_state:
                # Apply adjustment
                self.data1 = adjust_Z(self.data1, self.x1_data)
                self.x2_data = self.x2_data[::-1]  # Reverse x2_data
                self.data2 = adjust_Z(self.data2, self.x2_data)
                self.x2_data = self.x2_data[::-1]  # Reverse x2_data
                self.adjust_button.config(text="Reset Curve")
            else:
                # Reload original data to reset
                (self.x1_data, self.data1, self.data2, Z0) = s_param_to_Impedance(self.file_path)
                self.x2_data = self.x1_data[::-1]  # Reapply reversal
                self.adjust_button.config(text="Adjust Curve")
            self.update_plot()
    

    def set_x_limits(self):
        try:
            x_min = float(self.x_min_entry.get())
            x_max = float(self.x_max_entry.get())
            if x_min >= x_max:
                messagebox.showerror("Error", "X min must be less than X max")
                return
            self.x_limits = (x_min, x_max)
            self.update_plot()
        except ValueError:
            messagebox.showerror("Error", "Invalid X-axis limits. Enter numeric values.")

    def set_y_limits(self):
        try:
            y_min = float(self.y_min_entry.get())
            y_max = float(self.y_max_entry.get())
            if y_min >= y_max:
                messagebox.showerror("Error", "Y min must be less than Y max")
                return
            self.y_limits = (y_min, y_max)
            self.update_plot()
        except ValueError:
            messagebox.showerror("Error", "Invalid Y-axis limits. Enter numeric values.")

    def set_step_size(self):
        try:
            step = float(self.step_entry.get())
            if step <= 0:
                messagebox.showerror("Error", "Step size must be positive")
                return
            self.shift_step = step
        except ValueError:
            messagebox.showerror("Error", "Invalid step size. Enter a numeric value.")

    def update_plot(self):
        self.ax.clear()
        if self.data1 is not None and self.data2 is not None:
            # Plot data with current shift
            shifted_x = self.x2_data + self.shift_amount
            self.ax.plot(self.x1_data, self.data1, 'b-', label='TDR 1')
            self.ax.plot(shifted_x, self.data2, 'r-', label='TDR 2')
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Impedance (Ω)')
            self.ax.set_title('TDR')
            self.ax.grid(True)
            self.ax.legend()

            # Apply custom axis limits if set
            if self.x_limits:
                self.ax.set_xlim(self.x_limits)
            if self.y_limits:
                self.ax.set_ylim(self.y_limits)

        self.canvas.draw()

    def shift_left(self):
        if self.data1 is not None:
            self.shift_amount -= self.shift_step  # Shift left by step size
            self.update_plot()

    def shift_right(self):
        if self.data1 is not None:
            self.shift_amount += self.shift_step  # Shift right by step size
            self.update_plot()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotGUI(root)
    root.mainloop()
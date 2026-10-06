"""Tkinter input window (original behaviour kept, input cleaned up)."""
import tkinter as tk
from tkinter import messagebox


class InputGUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Object Detection System")
        self.root.geometry("500x300")
        self.root.configure(bg="#222831")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.objects = None

        tk.Label(self.root, text="AI Object Detection", font=("Arial", 18, "bold"),
                 fg="white", bg="#222831").pack(pady=15)
        tk.Label(self.root, text="Enter object names separated by commas (e.g. apple, cup)",
                 font=("Arial", 11), fg="white", bg="#222831").pack()

        self.entry = tk.Entry(self.root, width=35, font=("Arial", 14))
        self.entry.pack(pady=20)
        self.entry.focus()
        self.entry.bind("<Return>", lambda _e: self.start())

        tk.Button(self.root, text="Start Detection", font=("Arial", 13, "bold"),
                  bg="#00ADB5", fg="white", padx=20, pady=8,
                  command=self.start).pack(pady=20)

    def start(self):
        text = self.entry.get().strip()
        if not text:
            messagebox.showerror("Error", "Please enter at least one object.")
            return
        names = [o.strip().lower() for o in text.split(",") if o.strip()]
        self.objects = list(dict.fromkeys(names))   # drop empties and duplicates
        self.root.destroy()

    def get_objects(self):
        self.root.mainloop()
        return self.objects

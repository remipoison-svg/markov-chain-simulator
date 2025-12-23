#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:33:45 2025

@author: remipoison
"""

import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
import numpy as np
import tempfile, os, subprocess, shutil
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import shutil
import subprocess
import os
import sys
from pathlib import Path

import sys, os
from pathlib import Path

def _maybe_prepend_meipass_to_path():
    try:
        if hasattr(sys, "_MEIPASS"):
            bin_dir = Path(sys._MEIPASS) / "bin"
            if bin_dir.exists():
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

_maybe_prepend_meipass_to_path()


# --- Helpers LaTeX + messagebox (ajoutés automatiquement) --------------------
def _run_pdflatex(tex_file: str, workdir: str, extra_env=None):
    """
    Compile tex_file dans workdir.
    Essaie pdflatex; si introuvable, ajoute /Library/TeX/texbin au PATH (macOS).
    Si toujours absent, tente 'tectonic' (conda-forge). Sinon, lève FileNotFoundError.
    """
    import os, shutil, subprocess
    from pathlib import Path

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        texbin = "/Library/TeX/texbin"
        if Path(texbin).exists():
            env["PATH"] = texbin + os.pathsep + env.get("PATH", "")
            pdflatex = shutil.which("pdflatex", path=env["PATH"])

    if pdflatex:
        return subprocess.run(
            [pdflatex, "-interaction=nonstopmode", tex_file],
            cwd=workdir, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

    tectonic = shutil.which("tectonic", path=env.get("PATH", ""))
    if tectonic:
        return subprocess.run(
            [tectonic, tex_file],
            cwd=workdir, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

    raise FileNotFoundError(
        "Aucun moteur LaTeX trouvé (pdflatex ni tectonic). "
        "Installez MacTeX/BasicTeX ou `conda install -c conda-forge tectonic`, "
        "ou ajoutez /Library/TeX/texbin au PATH."
    )

def _read_latex_log(workdir: str, tex_file: str) -> str:
    base = Path(tex_file).with_suffix("")
    log_path = Path(workdir) / (base.name + ".log")
    try:
        if log_path.exists():
            return log_path.read_text(errors="ignore")
    except Exception:
        pass
    return ""

def _show_latex_error(title: str, log_text: str, result=None, parent=None) -> None:
    MAX_CHARS = 3000
    summary_parts = []

    if result is not None:
        try:
            if result.stdout:
                summary_parts.append("=== stdout ===\n" + result.stdout.decode("utf-8", errors="ignore"))
        except Exception:
            pass
        try:
            if result.stderr:
                summary_parts.append("=== stderr ===\n" + result.stderr.decode("utf-8", errors="ignore"))
        except Exception:
            pass

    if log_text:
        summary_parts.append("=== LaTeX log (extrait) ===\n" + log_text)

    summary = "\n\n".join(p for p in summary_parts if p.strip())
    if not summary:
        summary = "LaTeX a échoué mais aucun détail n’a pu être récupéré."

    if len(summary) > MAX_CHARS:
        summary = summary[:MAX_CHARS] + "\n\n[... contenu tronqué ...]\n"

    try:
        messagebox.showerror(title, summary, parent=parent)
    except Exception:
        messagebox.showerror(title, summary)
# -----------------------------------------------------------------------------


class MarkovApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Markov Premium")
        self.root.minsize(1080, 700)
        self.n_states = 3
        self.steps = 10
        self.state_names = ["S1", "S2", "S3"]
        self.undo_stack = []
        self.redo_stack = []
        self.examples = {
            "Identité": (["A", "B", "C"], np.eye(3), [1,0,0]),
            "SI (Susceptibles/ Infectés)": (["S", "I"], np.array([[0.8,0.2],[0,1]]), [1,0]),
            "File d’attente": (["Vide", "Un client", "Deux clients"], np.array([[0.7,0.3,0],[0.2,0.5,0.3],[0,0.4,0.6]]), [1,0,0]),
            "Jeu de l’oie (boucle)": (["Début", "Milieu", "Fin"], np.array([[0.5,0.5,0],[0,0.8,0.2],[0,0,1]]), [1,0,0])
        }
        self.build_layout()
        self.update_matrices()
        self.save_undo_state()

    # ----- UI BUILD -----
    def build_layout(self):
        self.left_frame = tk.Frame(self.root, width=260)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        # ----- Responsive grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=4)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        # --- Exemple selector
        ttk.Label(self.left_frame, text="Exemples pré-remplis :").pack(anchor="w")
        self.example_var = tk.StringVar()
        self.example_box = ttk.Combobox(self.left_frame, textvariable=self.example_var, state="readonly",
                                        values=list(self.examples.keys()))
        self.example_box.pack(fill="x")
        self.example_box.bind("<<ComboboxSelected>>", self.load_example)
        # --- Undo/Redo
        btn_frame = tk.Frame(self.left_frame)
        btn_frame.pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Undo", command=self.undo).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(btn_frame, text="Redo", command=self.redo).pack(side="left", expand=True, fill="x", padx=2)
        # --- State names
        tk.Label(self.left_frame, text="Noms des états :").pack(anchor="w")
        self.name_frame = tk.Frame(self.left_frame)
        self.name_frame.pack(fill="x", pady=2)
        self.name_entries = []
        # --- Nb états slider
        tk.Label(self.left_frame, text="Nombre d'états :").pack(anchor="w")
        self.n_states_var = tk.IntVar(value=self.n_states)
        self.slider = tk.Scale(self.left_frame, from_=2, to=10, orient=tk.HORIZONTAL,
                               variable=self.n_states_var, command=self.on_n_states_changed)
        self.slider.pack(fill=tk.X, pady=5)
        # --- Matrice
        tk.Label(self.left_frame, text="Matrice de transition :").pack(anchor="w")
        self.trans_matrix_frame = tk.Frame(self.left_frame)
        self.trans_matrix_frame.pack(pady=2)
        self.trans_entries = []
        tk.Button(self.left_frame, text="Générer aléatoirement", command=self.randomize_transition_matrix).pack(fill="x", pady=2)
        # --- Vecteur initial
        tk.Label(self.left_frame, text="Condition initiale :").pack(anchor="w")
        self.init_frame = tk.Frame(self.left_frame)
        self.init_frame.pack(pady=2)
        self.init_entries = []
        tk.Button(self.left_frame, text="Cond. initiales aléatoires", command=self.randomize_initial_conditions).pack(fill="x", pady=2)
        # --- Steps
        tk.Label(self.left_frame, text="Nombre d'étapes :").pack(anchor="w")
        self.steps_var = tk.IntVar(value=self.steps)
        self.steps_entry = tk.Entry(self.left_frame, textvariable=self.steps_var)
        self.steps_entry.pack(fill=tk.X, pady=5)
        # --- Simuler
        tk.Button(self.left_frame, text="Simuler", command=self.simulate).pack(fill="x", pady=8)
        # --- Export
        tk.Button(self.left_frame, text="Exporter GRAPHE en PNG", command=self.export_graph_png).pack(fill="x", pady=2)
        tk.Button(self.left_frame, text="Exporter COURBE en PNG", command=self.export_curve_png).pack(fill="x", pady=2)

        # --- Affichage principal à droite
        self.graph_label = tk.Label(self.right_frame, text="(Graphe Markov)", bg="#f8f8f8")
        self.graph_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.curve_canvas = None

    # ----- STATE SNAPSHOT -----
    def get_state(self):
        names = [e.get() for e in self.name_entries]
        matrix = np.array([[float(e.get()) for e in row] for row in self.trans_entries])
        v0 = np.array([float(e.get()) for e in self.init_entries])
        steps = self.steps_var.get()
        return (names, matrix, v0, steps)

    def set_state(self, state):
        names, matrix, v0, steps = state
        self.n_states = len(names)
        self.slider.set(self.n_states)
        for i, name in enumerate(names):
            self.name_entries[i].delete(0, tk.END)
            self.name_entries[i].insert(0, name)
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.trans_entries[i][j].delete(0, tk.END)
                self.trans_entries[i][j].insert(0, f"{matrix[i][j]:.2f}")
            self.init_entries[i].delete(0, tk.END)
            self.init_entries[i].insert(0, f"{v0[i]:.2f}")
        self.steps_var.set(steps)

    def save_undo_state(self):
        self.undo_stack.append(self.get_state())
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.set_state(self.undo_stack[-1])

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.set_state(state)
            self.undo_stack.append(state)

    # ----- NOMS, MATRICE, INIT -----
    def on_n_states_changed(self, event=None):
        self.n_states = self.n_states_var.get()
        self.state_names = [f"S{i+1}" for i in range(self.n_states)]
        self.update_matrices()
        self.save_undo_state()

    def update_matrices(self):
        # Names
        for child in self.name_frame.winfo_children():
            child.destroy()
        self.name_entries.clear()
        for i in range(self.n_states):
            e = tk.Entry(self.name_frame, width=8, justify="center")
            e.pack(side="left", padx=2)
            e.insert(0, self.state_names[i] if i < len(self.state_names) else f"S{i+1}")
            self.name_entries.append(e)
        # Matrix
        for child in self.trans_matrix_frame.winfo_children():
            child.destroy()
        self.trans_entries.clear()
        for i in range(self.n_states):
            row = []
            for j in range(self.n_states):
                e = tk.Entry(self.trans_matrix_frame, width=5, justify="center")
                e.grid(row=i, column=j, padx=1, pady=1)
                e.insert(0, "0" if i != j else "1")
                row.append(e)
            self.trans_entries.append(row)
        # Init
        for child in self.init_frame.winfo_children():
            child.destroy()
        self.init_entries.clear()
        for j in range(self.n_states):
            e = tk.Entry(self.init_frame, width=5, justify="center")
            e.grid(row=0, column=j, padx=1, pady=1)
            e.insert(0, "1" if j == 0 else "0")
            self.init_entries.append(e)

    def randomize_transition_matrix(self):
        n = self.n_states
        for i in range(n):
            random_row = np.random.rand(n)
            random_row /= random_row.sum()
            row_vals = []
            for j in range(n-1):
                val = round(random_row[j], 2)
                row_vals.append(val)
            last_val = max(0, min(1, round(1 - sum(row_vals), 2)))
            row_vals.append(last_val)
            total = sum(row_vals)
            if total != 1.0:
                diff = round(1.0 - total, 2)
                row_vals[-1] = round(row_vals[-1] + diff, 2)
            for j in range(n):
                self.trans_entries[i][j].delete(0, tk.END)
                self.trans_entries[i][j].insert(0, f"{row_vals[j]:.2f}")
        self.save_undo_state()

    def randomize_initial_conditions(self):
        n = self.n_states
        random_vec = np.random.rand(n)
        random_vec /= random_vec.sum()
        vec_vals = []
        for j in range(n-1):
            val = round(random_vec[j], 2)
            vec_vals.append(val)
        last_val = max(0, min(1, round(1 - sum(vec_vals), 2)))
        vec_vals.append(last_val)
        total = sum(vec_vals)
        if total != 1.0:
            diff = round(1.0 - total, 2)
            vec_vals[-1] = round(vec_vals[-1] + diff, 2)
        for j in range(n):
            self.init_entries[j].delete(0, tk.END)
            self.init_entries[j].insert(0, f"{vec_vals[j]:.2f}")
        self.save_undo_state()

    def load_example(self, event=None):
        key = self.example_var.get()
        if key in self.examples:
            names, matrix, v0 = self.examples[key]
            self.n_states = len(names)
            self.slider.set(self.n_states)
            self.state_names = names
            self.update_matrices()
            for i, name in enumerate(names):
                self.name_entries[i].delete(0, tk.END)
                self.name_entries[i].insert(0, name)
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.trans_entries[i][j].delete(0, tk.END)
                    self.trans_entries[i][j].insert(0, f"{matrix[i][j]:.2f}")
                self.init_entries[i].delete(0, tk.END)
                self.init_entries[i].insert(0, f"{v0[i]:.2f}")
            self.save_undo_state()

    # ----- SIMULATION -----
    def read_transition_matrix(self):
        n = self.n_states
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    A[i, j] = float(self.trans_entries[i][j].get())
                except ValueError:
                    raise ValueError(f"Entrée invalide [{i+1},{j+1}]")
        for i, row in enumerate(A):
            if not np.isclose(np.sum(row), 1.0):
                raise ValueError(f"La somme de la ligne {i+1} de la matrice doit être 1.")
        return A

    def simulate(self):
        try:
            A = self.read_transition_matrix()
        except Exception as ex:
            messagebox.showerror("Erreur", str(ex))
            return
        self.save_undo_state()
        self.draw_tikz_png(A)

    # ----- TIKZ, PDF, PNG -----
    def generate_tikz_code(self, matrix, state_labels=None):
        n = len(matrix)
        if state_labels is None:
            state_labels = [f"S{i+1}" for i in range(n)]
        tikz_nodes = []
        tikz_edges = []
        arrow_style = "postaction={decorate}, decoration={markings, mark=at position 0.5 with {\\arrow{stealth}}}"
        for i, label in enumerate(state_labels):
            angle = 360 * i / n
            tikz_nodes.append(f"\\node[state] (S{i}) at ({angle}:3cm) {{{label}}};")
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    if i == j:
                        tikz_edges.append(
                            f"(S{i}) edge[loop above, {arrow_style}] node{{\\scriptsize {matrix[i][j]:.2f}}} (S{j})"
                        )
                    else:
                        tikz_edges.append(
                            f"(S{i}) edge[bend left, {arrow_style}] node{{\\scriptsize {matrix[i][j]:.2f}}} (S{j})"
                        )
        tikz_code = r"""\documentclass{standalone}
\usepackage{tikz}
\usetikzlibrary{automata, positioning, decorations.markings}
\begin{document}
\begin{tikzpicture}[auto, semithick, node distance=3cm]
\tikzstyle{every state}=[fill=white,draw=black,thick,text=blue,scale=1.25,font=\sffamily]
"""
        tikz_code += "\n  " + "\n  ".join(tikz_nodes) + "\n\n"
        tikz_code += "  \\path\n    " + "\n    ".join(tikz_edges) + ";\n"
        tikz_code += r"""\end{tikzpicture}
\end{document}
"""
        return tikz_code

    def draw_tikz_png(self, A):
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = os.path.join(tmpdir, "graph.tex")
            pdf_file = os.path.join(tmpdir, "graph.pdf")
            tikz_code = self.generate_tikz_code(A, [e.get() for e in self.name_entries])
            with open(tex_file, "w") as f:
                f.write(tikz_code)
            # Compilation LaTeX -> PDF
            result = _run_pdflatex(tex_file, tmpdir)
            if result.returncode != 0:
                log_text = _read_latex_log(tmpdir, tex_file)
                _show_latex_error('Compilation LaTeX échouée', log_text, result=result, parent=self.root if hasattr(self, 'root') else None)
                return
            if not os.path.exists(pdf_file):
                messagebox.showerror("Erreur LaTeX", "LaTeX n'a pas généré de PDF.")
                return
            # Conversion PDF -> PNG via pdftocairo
            try:
                subprocess.run(
                    ["pdftocairo", "-png", pdf_file, os.path.join(tmpdir, "graph")],
                    cwd=tmpdir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except Exception as e:
                messagebox.showerror("Erreur", f"pdftocairo a échoué : {e}")
                return
            # Cherche le PNG généré
            png_candidates = [os.path.join(tmpdir, "graph-1.png"),
                              os.path.join(tmpdir, "graph.png")]
            png_file = None
            for candidate in png_candidates:
                if os.path.exists(candidate):
                    png_file = candidate
                    break
            if png_file is None:
                messagebox.showerror("Erreur", "PNG non généré, vérifiez pdftocairo.")
                return
            # Affichage PNG dans Tkinter
            try:
                img = Image.open(png_file)
                img = self.resize_image_proportionally(img, max_width=500, max_height=500)
                self.tkimg = ImageTk.PhotoImage(img)
                self.graph_label.configure(image=self.tkimg)
                self.graph_label.image = self.tkimg
                self.current_graph_png = img.copy()
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'afficher le PNG généré : {e}")
            # Courbes
            try:
                v0 = np.array([float(e.get()) for e in self.init_entries])
                v0 = v0 / np.sum(v0)
                steps = int(self.steps_var.get())
                self.show_evolution_curves(A, v0, steps, [e.get() for e in self.name_entries])
            except Exception as e:
                messagebox.showwarning("Erreur", f"Impossible d'afficher le graphe des états : {e}")

    # --- Export PNG du graphe
    def export_graph_png(self):
        if not hasattr(self, 'current_graph_png'):
            messagebox.showinfo("Info", "Veuillez d'abord simuler pour générer un graphe.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")],
                                            title="Exporter le graphe PNG")
        if file:
            self.current_graph_png.save(file)

    # --- Export PNG de la courbe
    def export_curve_png(self):
        if not hasattr(self, 'last_curve_fig'):
            messagebox.showinfo("Info", "Veuillez d'abord simuler pour générer une courbe.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")],
                                            title="Exporter la courbe PNG")
        if file:
            self.last_curve_fig.savefig(file, dpi=160)

    def show_evolution_curves(self, A, v0, steps, state_names):
        n = len(v0)
        P = np.zeros((steps+1, n))
        P[0, :] = v0
        for t in range(1, steps+1):
            P[t] = P[t-1] @ A
        fig, ax = plt.subplots(figsize=(6, 2.7), dpi=110)
        for i in range(n):
            ax.plot(range(steps+1), P[:, i], label=state_names[i])
        ax.set_xlabel("Étape")
        ax.set_ylabel("Probabilité")
        ax.set_ylim(0, 0.05 + float(np.max(P)))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        if hasattr(self, 'curve_canvas') and self.curve_canvas is not None:
            self.curve_canvas.get_tk_widget().destroy()
        self.curve_canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        self.curve_canvas.draw()
        self.curve_canvas.get_tk_widget().grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        plt.close(fig)
        self.last_curve_fig = fig

    def resize_image_proportionally(self, img, max_width=500, max_height=500):
        orig_width, orig_height = img.size
        ratio = min(max_width / orig_width, max_height / orig_height)
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def _attach_footer_label(root):
    """Attach a small dark-gray footer at bottom-left of the main window."""
    import tkinter as tk
    import tkinter.font as tkfont
    text = (
        "Terminales générales - mathématiques expertes\n"
        "Développé par Rémi Poison\n" "lycée Léonard de Vinci Calais\n"
        "académie de Lille - mai 2025"
    )
    # Create label
    font = tkfont.nametofont("TkDefaultFont").copy()
    try:
        font.configure(size=max(8, font.cget("size") - 2))
    except Exception:
        pass
    lbl = tk.Label(root, text=text, fg="#455555", bg=root.cget("bg"), font=font, justify="left")
    # Place it at bottom-left, with a little margin
    def _place(_evt=None):
        # Anchor to bottom-left with 8 px padding.
        lbl.place(x=8, y=root.winfo_height()-8, anchor="sw")
    root.bind("<Configure>", _place, add="+")
    # Ensure created and positioned once window is drawn
    root.after(50, _place)
    return lbl

if __name__ == "__main__":
    root = tk.Tk()
    app = MarkovApp(root)
    try:
        _attach_footer_label(root)
    except Exception as e:
        # Non-fatal if footer cannot be attached
        pass
    root.mainloop()

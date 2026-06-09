import matplotlib
matplotlib.use('TkAgg')  # Change to 'Qt5Agg' if TkAgg is unavailable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.df_ops import prepare_df

DATA_DIR    = PROJECT_ROOT / "Data" / "benchmark"
SAVE_PATH   = Path(__file__).resolve().parent / "labels.json"
TOL         = 6
STEP_SMALL  = 1
STEP_LARGE  = 10


class Labeler:
    def __init__(self):
        self.data_files = sorted(DATA_DIR.glob("*.txt"))
        if not self.data_files:
            print(f"No .txt files found in {DATA_DIR}")
            return

        self.file_idx      = 0
        self.all_labels    = {}
        self.history       = []   # undo stack: list of ('maxima'|'minima', day)
        self.mode          = None # 'max' | 'min' | None
        self.cursor_pos    = 0
        self.cursor_line   = None
        self.cursor_marker = None

        self.load_current()

        self.fig, self.ax = plt.subplots(figsize=(20, 5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.redraw()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_current(self):
        fpath = self.data_files[self.file_idx]
        self.fname = fpath.name

        raw_df       = pd.read_csv(fpath, sep=r'\s+', skip_blank_lines=True)
        raw_data_df  = prepare_df(raw_df, add_prefix=False, relative=True)

        med = raw_data_df['sind'].median()
        mad = (raw_data_df['sind'] - med).abs().median()

        self.raw_data_df = raw_data_df
        self.data_df     = raw_data_df[
            (raw_data_df['sind'] - med).abs() < TOL * mad
        ].reset_index(drop=True)

        if self.fname not in self.all_labels:
            self.all_labels[self.fname] = {'maxima': [], 'minima': []}

        self.current_labels = self.all_labels[self.fname]
        self.history        = []
        self.mode           = None
        self.cursor_pos     = len(self.data_df) // 2

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def redraw(self):
        self.ax.cla()

        self.ax.scatter(self.raw_data_df['day'], self.raw_data_df['sind'],
                        color='red', s=10, alpha=0.4, label='Outliers', zorder=1)
        self.ax.scatter(self.data_df['day'], self.data_df['sind'],
                        color='green', s=15, label='Clean', zorder=2)

        y_max = self.data_df['sind'].max()
        y_min = self.data_df['sind'].min()
        y_pad = (y_max - y_min) * 0.02

        for day in self.current_labels['maxima']:
            self.ax.axvline(day, color='royalblue', linestyle='--', linewidth=1.2, alpha=0.8)
            self.ax.text(day, y_max + y_pad, 'MAX', color='royalblue',
                         fontsize=7, ha='center', va='bottom', rotation=90)

        for day in self.current_labels['minima']:
            self.ax.axvline(day, color='darkorange', linestyle='--', linewidth=1.2, alpha=0.8)
            self.ax.text(day, y_min - y_pad, 'MIN', color='darkorange',
                         fontsize=7, ha='center', va='top', rotation=90)

        self.cursor_line   = None
        self.cursor_marker = None
        if self.mode is not None and len(self.data_df) > 0:
            cur_day  = self.data_df['day'].iloc[self.cursor_pos]
            cur_sind = self.data_df['sind'].iloc[self.cursor_pos]
            color = 'royalblue' if self.mode == 'max' else 'darkorange'
            mkr   = '^' if self.mode == 'max' else 'v'
            self.cursor_line   = self.ax.axvline(cur_day, color=color, linewidth=2.5, zorder=5)
            self.cursor_marker = self.ax.scatter([cur_day], [cur_sind],
                                                 color=color, marker=mkr,
                                                 s=250, zorder=6, edgecolors='k')

        n_files  = len(self.data_files)
        n_max    = len(self.current_labels['maxima'])
        n_min    = len(self.current_labels['minima'])
        mode_str = f"Placing: {self.mode.upper()}" if self.mode else "Press  1 = MAX   2 = MIN  to begin"

        self.ax.set_title(
            f"{self.fname}  ({self.file_idx + 1}/{n_files})  |  {mode_str}  |  "
            f"{n_max} maxima · {n_min} minima saved\n"
            "[←/→] move 1 pt   [Shift+←/→] jump 10   [Space] save   "
            "[U] undo   [N] next file   [S] save all   [Q] quit"
        )
        self.ax.set_xlabel("Day")
        self.ax.set_ylabel("S-index")
        self.ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()

    def _move_cursor_artists(self):
        """Update only the cursor without a full redraw — keeps navigation snappy."""
        if self.cursor_line is None or self.cursor_marker is None:
            self.redraw()
            return
        cur_day  = self.data_df['day'].iloc[self.cursor_pos]
        cur_sind = self.data_df['sind'].iloc[self.cursor_pos]
        self.cursor_line.set_xdata([cur_day, cur_day])
        self.cursor_marker.set_offsets([[cur_day, cur_sind]])
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def on_key(self, event):
        k = event.key

        if k == '1':
            self.mode = 'max'
            self.redraw()

        elif k == '2':
            self.mode = 'min'
            self.redraw()

        elif k == 'right' and self.mode:
            self.cursor_pos = min(self.cursor_pos + STEP_SMALL, len(self.data_df) - 1)
            self._move_cursor_artists()

        elif k == 'left' and self.mode:
            self.cursor_pos = max(self.cursor_pos - STEP_SMALL, 0)
            self._move_cursor_artists()

        elif k == 'shift+right' and self.mode:
            self.cursor_pos = min(self.cursor_pos + STEP_LARGE, len(self.data_df) - 1)
            self._move_cursor_artists()

        elif k == 'shift+left' and self.mode:
            self.cursor_pos = max(self.cursor_pos - STEP_LARGE, 0)
            self._move_cursor_artists()

        elif k == ' ' and self.mode:
            self._save_label()

        elif k == 'u':
            self._undo()

        elif k == 'n':
            self._next_file()

        elif k == 's':
            self.save_all()

        elif k == 'q':
            self.save_all()
            plt.close()

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def _save_label(self):
        cur_day = float(self.data_df['day'].iloc[self.cursor_pos])
        key     = 'maxima' if self.mode == 'max' else 'minima'
        self.current_labels[key].append(cur_day)
        self.history.append((key, cur_day))
        # Auto-alternate
        self.mode = 'min' if self.mode == 'max' else 'max'
        self.redraw()

    def _undo(self):
        if not self.history:
            return
        key, day = self.history.pop()
        if day in self.current_labels[key]:
            self.current_labels[key].remove(day)
        self.redraw()

    def _next_file(self):
        self.all_labels[self.fname] = dict(self.current_labels)
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            print("All files processed.")
            self.save_all()
            plt.close()
            return
        self.load_current()
        self.redraw()

    def save_all(self):
        self.all_labels[self.fname] = dict(self.current_labels)
        with open(SAVE_PATH, 'w') as f:
            json.dump(self.all_labels, f, indent=2)
        print(f"Saved → {SAVE_PATH}")


if __name__ == '__main__':
    Labeler()

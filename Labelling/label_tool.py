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

DATA_DIR   = PROJECT_ROOT / "Data" / "benchmark"
SAVE_PATH  = Path(__file__).resolve().parent / "labels.json"
TOL        = 6
STEP_SMALL = 1
STEP_LARGE = 10


class Labeler:
    def __init__(self):
        self.data_files = sorted(DATA_DIR.glob("*.txt"))
        if not self.data_files:
            print(f"No .txt files found in {DATA_DIR}")
            return

        self.file_idx   = 0
        self.all_labels = {}
        self.history    = []
        self.mode       = None
        self.cursor_pos = 0
        self.background = None
        self.cursor_vline = None
        self.cursor_dot   = None

        self.load_current()

        self.fig, self.ax = plt.subplots(figsize=(20, 5), dpi=72)
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

        raw_df      = pd.read_csv(fpath, sep=r'\s+', skip_blank_lines=True)
        raw_data_df = prepare_df(raw_df, add_prefix=False, relative=True)

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
        self.background     = None

    # ------------------------------------------------------------------
    # Drawing — split into static (expensive) and cursor (cheap/blitted)
    # ------------------------------------------------------------------

    def _draw_static(self):
        """Full redraw of everything except the cursor. Captures blit background."""
        self.ax.cla()

        # plot() is much faster than scatter(); rasterized collapses to bitmap
        self.ax.plot(self.raw_data_df['day'], self.raw_data_df['sind'],
                     '.', color='red', markersize=3, alpha=0.4,
                     label='Outliers', rasterized=True)
        self.ax.plot(self.data_df['day'], self.data_df['sind'],
                     '.', color='green', markersize=4,
                     label='Clean', rasterized=True)

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

        # Animated cursor artists — excluded from regular draw(), drawn via blit
        self.cursor_vline, = self.ax.plot([], [], linewidth=2.5, zorder=5, animated=True)
        self.cursor_dot,   = self.ax.plot([], [], linestyle='', markersize=14,
                                           zorder=6, animated=True, mec='k', mew=1.5)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _blit_cursor(self):
        """Restore background pixel buffer and draw only the cursor on top."""
        if self.background is None:
            self._draw_static()
            return

        cur_day  = self.data_df['day'].iloc[self.cursor_pos]
        cur_sind = self.data_df['sind'].iloc[self.cursor_pos]
        y_lim    = self.ax.get_ylim()

        if self.mode is None:
            color, mkr = 'gray', 'o'
        elif self.mode == 'max':
            color, mkr = 'royalblue', '^'
        else:
            color, mkr = 'darkorange', 'v'

        self.cursor_vline.set_data([cur_day, cur_day], list(y_lim))
        self.cursor_vline.set_color(color)

        self.cursor_dot.set_data([cur_day], [cur_sind])
        self.cursor_dot.set_color(color)
        self.cursor_dot.set_marker(mkr)

        self.fig.canvas.restore_region(self.background)
        self.ax.draw_artist(self.cursor_vline)
        self.ax.draw_artist(self.cursor_dot)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def redraw(self):
        self._draw_static()
        self._blit_cursor()

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

        elif k == 'right':
            self.cursor_pos = min(self.cursor_pos + STEP_SMALL, len(self.data_df) - 1)
            self._blit_cursor()

        elif k == 'left':
            self.cursor_pos = max(self.cursor_pos - STEP_SMALL, 0)
            self._blit_cursor()

        elif k == 'shift+right':
            self.cursor_pos = min(self.cursor_pos + STEP_LARGE, len(self.data_df) - 1)
            self._blit_cursor()

        elif k == 'shift+left':
            self.cursor_pos = max(self.cursor_pos - STEP_LARGE, 0)
            self._blit_cursor()

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

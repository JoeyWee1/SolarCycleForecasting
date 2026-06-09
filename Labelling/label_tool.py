import matplotlib
matplotlib.use('TkAgg')  # Change to 'Qt5Agg' if TkAgg is unavailable
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers.df_ops import prepare_df

DATA_DIR  = PROJECT_ROOT / "Data" 
SAVE_PATH = Path(__file__).resolve().parent / "labels.json"
TOL = 6


class Labeler:
    def __init__(self):
        self.data_files = sorted(DATA_DIR.glob("*.txt"))
        if not self.data_files:
            print(f"No .txt files found in {DATA_DIR}")
            return

        self.file_idx   = 0
        self.all_labels = json.load(open(SAVE_PATH)) if SAVE_PATH.exists() else {}
        self.history    = []
        self.mode       = None
        self.cursor_day = 0.0
        self.background = None
        self.cursor_vline = None
        self.cursor_dot   = None

        self.load_current()

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('resize_event', self._on_resize)

        # Query screen size from matplotlib's own Tk window (correct DPI awareness)
        mgr  = plt.get_current_fig_manager()
        sw   = mgr.window.winfo_screenwidth()
        sh   = mgr.window.winfo_screenheight()
        dpi  = self.fig.dpi
        w_in = sw * 0.90 / dpi          # 90% of screen width
        h_in = w_in * sh / sw           # same aspect ratio as screen
        self.fig.set_size_inches(w_in, h_in)

        # CONST button — bottom centre, in its own axes so it survives ax.cla()
        self.fig.subplots_adjust(bottom=0.13)
        ax_btn = self.fig.add_axes([0.44, 0.02, 0.12, 0.06])
        self.btn_const = Button(ax_btn, 'CONST', color='#f0f0f0', hovercolor='#ffe082')
        self.btn_const.on_clicked(self._on_const)

        mgr.window.after(100, self.redraw)
        plt.show()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_current(self):
        fpath = self.data_files[self.file_idx]
        self.fname = fpath.name

        raw_df = pd.read_csv(fpath, sep=r'\s+', skip_blank_lines=True)
        try:
            raw_data_df = prepare_df(raw_df, add_prefix=False, relative=True)
        except Exception:
            raw_data_df = prepare_df(raw_df, add_prefix=True, relative=True)

        med = raw_data_df['sind'].median()
        mad = (raw_data_df['sind'] - med).abs().median()

        # Shift so day 0 = first observation in the raw file
        day_offset = raw_data_df['day'].min()
        raw_data_df = raw_data_df.copy()
        raw_data_df['day'] -= day_offset

        self.raw_data_df = raw_data_df
        # sort by day so np.interp works correctly
        self.data_df = (
            raw_data_df[(raw_data_df['sind'] - med).abs() < TOL * mad]
            .sort_values('day')
            .reset_index(drop=True)
        )

        self.day_min = float(self.data_df['day'].min())
        self.day_max = float(self.data_df['day'].max())
        span = self.day_max - self.day_min

        # Three navigation step sizes in time units (days)
        self.step_fine   = span * 0.002   # ~0.2% — fine
        self.step_medium = span * 0.01    # ~1%   — medium (Shift)
        self.step_coarse = span * 0.05    # ~5%   — coarse (Ctrl)

        if self.fname not in self.all_labels:
            self.all_labels[self.fname] = {'maxima': [], 'minima': [], 'const': False}

        self.current_labels = self.all_labels[self.fname]
        self.history        = []
        self.mode           = None
        self.cursor_day     = (self.day_min + self.day_max) / 2
        self.background     = None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _on_resize(self, event):
        self.background = None
        self.redraw()

    def _draw_static(self):
        self.ax.cla()

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
        is_const = self.current_labels.get('const', False)
        mode_str = "[ CONST ]" if is_const else \
                   (f"Placing: {self.mode.upper()}" if self.mode else "Press  1 = MAX   2 = MIN  to begin")

        self.ax.set_title(
            f"{self.fname}  ({self.file_idx + 1}/{n_files})  |  {mode_str}  |  "
            f"{n_max} maxima · {n_min} minima saved\n"
            "[←/→] fine   [Shift+←/→] medium   [Ctrl+←/→] coarse   "
            "[Space] save   [U] undo   [N] next file   [5] save all   [Q] quit"
        )
        self.ax.set_xlabel("Day")
        self.ax.set_ylabel("S-index")
        self.ax.legend(loc='upper right', fontsize=8)

        # Shade background and update button colour to reflect const state
        if is_const:
            self.ax.set_facecolor('#fff8e1')
            self.btn_const.ax.set_facecolor('#ffb300')
            self.btn_const.label.set_text('CONST  ✓')
        else:
            self.ax.set_facecolor('white')
            self.btn_const.ax.set_facecolor('#f0f0f0')
            self.btn_const.label.set_text('CONST')

        self.cursor_vline, = self.ax.plot([], [], linewidth=2.5, zorder=5, animated=True)
        self.cursor_dot,   = self.ax.plot([], [], linestyle='', markersize=14,
                                           zorder=6, animated=True, mec='k', mew=1.5)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _blit_cursor(self):
        if self.background is None:
            self._draw_static()
            return

        cur_day  = self.cursor_day
        # interpolate y — cursor floats freely between data points
        cur_sind = float(np.interp(cur_day,
                                   self.data_df['day'].values,
                                   self.data_df['sind'].values))
        y_lim = self.ax.get_ylim()

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

    def _move(self, delta):
        self.cursor_day = float(np.clip(self.cursor_day + delta,
                                        self.day_min, self.day_max))
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

        elif k == 'right':            self._move(+self.step_fine)
        elif k == 'left':             self._move(-self.step_fine)
        elif k == 'shift+right':      self._move(+self.step_medium)
        elif k == 'shift+left':       self._move(-self.step_medium)
        elif k == 'ctrl+right':       self._move(+self.step_coarse)
        elif k == 'ctrl+left':        self._move(-self.step_coarse)

        elif k == ' ' and self.mode:
            self._save_label()

        elif k == 'u':
            self._undo()

        elif k == 'n':
            self._next_file()

        elif k == 'c':
            self._on_const()

        elif k == '5':
            self.save_all()

        elif k == 'q':
            self.save_all()
            plt.close()

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    def _on_const(self, _event=None):
        self.current_labels['const'] = not self.current_labels.get('const', False)
        self.redraw()

    def _save_label(self):
        key = 'maxima' if self.mode == 'max' else 'minima'
        self.current_labels[key].append(self.cursor_day)
        self.history.append((key, self.cursor_day))
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

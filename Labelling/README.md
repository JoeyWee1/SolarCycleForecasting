# Label Tool

Interactive tool for manually labelling maxima and minima in S-index time series.

## Usage

```
python labelling/label_tool.py
```

## Workflow

1. Plot appears with clean (green) and outlier (red) data
2. Press `1` to place a **MAX** or `2` to place a **MIN** — a coloured vertical cursor appears
3. `←` / `→` to move one point, `Shift+←/→` to jump 10 points
4. `Space` to save — cursor auto-alternates to the other type
5. `U` to undo the last saved label
6. `N` to move to the next file (loops through all `.txt` files in `Data/benchmark/`)
7. `S` to save at any time, `Q` to save and quit

## Output

Labels are written to `labelling/labels.json`:

```json
{
  "HD201091_Mt_wilson_data.txt": {
    "maxima": [1234.5, 5678.9],
    "minima": [3456.7]
  }
}
```

Days are in the same relative units as `prepare_df(..., relative=True)`.

## Notes

- Outlier rejection uses MAD with `tol=6`
- Default matplotlib backend is `TkAgg` — change line 2 of `label_tool.py` to `Qt5Agg` if unavailable

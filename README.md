# Raytracer

A simple Python raytracer showcasing basic ray tracing functionality, with support for various scene configurations and rendering options.

## Running

To run the **sunset** example, open a terminal in the project’s root directory and run:

```bash
python3.10 -m examples.sunset --outFile sunset.png
```

**Important**:
- By default, the output image will be saved to the `images` folder (if it exists) under the name you specify with `--outFile`.
- If you’re using complex models in your scenes, be aware that rendering may take a long time to finish.

## Extra Command-Line Arguments

You can pass these additional arguments when running the raytracer:

- `--nx` (int): The width of the output image in pixels.  
- `--ny` (int): The height of the output image in pixels.  
- `--white` (float): The whitepoint value for the render.  
- `--outFile` (string): The filename to save the output image. Defaults to saving in the `images` folder.

**Example usage**:

```bash
python3.10 -m examples.sunset --nx 1920 --ny 1080 --white 1.0 --outFile custom_sunset.png
```

## Configuration

Inside **`config.py`**, you can change various settings to customize your renders:

- **Supersampling**  
  - `SUPERSAMPLING_SCALE`: By default, it’s set to `1` (supersampling off).  
    - If set to `N` (an integer ≥ 1), then `N*N` sub-pixels will be computed for each final pixel (for anti-aliasing).  
  - `AA_ALGORITHM`: Determines which anti-aliasing algorithm to use. Default is `ANTI_ALIASING_ALGORITHM.GRID`. You can switch to `GRID_ROTATION` for a rotated sampling pattern, or add your own.  
  - `ROT_GRID_ANGLE` and `ROT_GRID_SCALE`: Specific settings used by the rotated grid algorithm.

- **Reflection Depth**  
  - `MAX_DEPTH`: The maximum number of reflective bounces (default is `4`). Increase for deeper mirror reflections (at the cost of performance).

- **Brightness Threshold**  
  - `BRIGHTNESS_THRESHOLD`: A value between `0.0` and `1.0` (default is `0.4`), used to determine the cut-off for perceived brightness in certain calculations.

Below is an excerpt from **`config.py`** showcasing the defaults:

```python
# MIRROR REFLECTION DEPTH (1 <=)
MAX_DEPTH = 4

# PERCEIVED BRIGHTNESS THRESHOLD (0.0 - 1.0)
BRIGHTNESS_THRESHOLD = 0.4

# SET SUPERSAMPLING_SCALE TO 1 TO TURN OFF 
SUPERSAMPLING_SCALE = 1  # 1 <= 
AA_ALGORITHM = ANTI_ALIASING_ALGORITHM.GRID

# ROTATED GRID ALGORITHM SETTINGS
ROT_GRID_ANGLE = np.arctan(-1/2)
ROT_GRID_SCALE = np.sqrt(5)/2
```


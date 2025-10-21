"""Command-line tool for generating printable coding pages from images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
import webcolors


RGB = Tuple[int, int, int]


try:
    Resampling = Image.Resampling  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - Pillow < 9.1 fallback
    Resampling = Image  # type: ignore[assignment]


CSS3_RGB: Dict[str, RGB] = {
    name: tuple(webcolors.name_to_rgb(name)) for name in webcolors.names(spec="css3")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a printable coding page and preview from an image. "
            "The coding page contains an empty grid with row/column indexes and a legend "
            "mapping colors to grid coordinates."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the source image file")
    parser.add_argument("--width", type=int, default=20, help="Number of columns in the grid (default: 20)")
    parser.add_argument("--height", type=int, help="Number of rows in the grid (defaults to --width)")
    parser.add_argument(
        "--cell-size",
        type=int,
        default=40,
        help="Size of each cell (in pixels) for generated images (default: 40)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where all generated assets will be stored (default: ./output)",
    )
    parser.add_argument(
        "--page-file",
        default="coding_page.png",
        help="Filename for the printable coding page (default: coding_page.png)",
    )
    parser.add_argument(
        "--preview-file",
        default="preview.png",
        help="Filename for the filled preview image (default: preview.png)",
    )
    parser.add_argument(
        "--legend-file",
        default="legend.json",
        help="Filename for the color legend file (default: legend.json)",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        help="Limit the number of colours by quantizing the downsampled image before labelling",
    )
    parser.add_argument(
        "--numbered-page",
        action="store_true",
        help="Write the colour number inside each grid cell instead of leaving it blank",
    )
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def downsample(image: Image.Image, width: int, height: int) -> List[List[RGB]]:
    resized = image.resize((width, height), Resampling.NEAREST)
    return [[resized.getpixel((col, row)) for col in range(width)] for row in range(height)]


def limit_palette(rgb_grid: List[List[RGB]], max_colors: Optional[int]) -> List[List[RGB]]:
    if not max_colors or max_colors <= 0:
        return rgb_grid

    rows = len(rgb_grid)
    cols = len(rgb_grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return rgb_grid

    unique_colors = {pixel for row in rgb_grid for pixel in row}
    if len(unique_colors) <= max_colors:
        return rgb_grid

    palette_image = Image.new("RGB", (cols, rows))
    palette_image.putdata([pixel for row in rgb_grid for pixel in row])
    quantized = palette_image.quantize(colors=max_colors, method=Image.MEDIANCUT)
    quantized_rgb = quantized.convert("RGB")
    return [[quantized_rgb.getpixel((col, row)) for col in range(cols)] for row in range(rows)]


def assign_color_numbers(legend: Dict[str, List[List[int]]]) -> Dict[str, int]:
    return {color: index for index, color in enumerate(legend.keys(), start=1)}


def nearest_css3_color(rgb: RGB) -> Tuple[str, RGB]:
    try:
        name = webcolors.rgb_to_name(rgb, spec="css3")
        return name, CSS3_RGB[name]
    except ValueError:
        pass

    # Choose the closest CSS3 color using squared Euclidean distance in RGB space.
    best_name = "black"
    best_rgb = CSS3_RGB[best_name]
    best_distance = float("inf")
    for name, candidate_rgb in CSS3_RGB.items():
        dr = candidate_rgb[0] - rgb[0]
        dg = candidate_rgb[1] - rgb[1]
        db = candidate_rgb[2] - rgb[2]
        distance = dr * dr + dg * dg + db * db
        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_rgb = candidate_rgb
    return best_name, best_rgb


def build_grid_with_names(rgb_grid: List[List[RGB]]) -> Tuple[List[List[str]], List[List[RGB]]]:
    name_grid: List[List[str]] = []
    css_grid: List[List[RGB]] = []
    for row in rgb_grid:
        name_row: List[str] = []
        css_row: List[RGB] = []
        for rgb in row:
            name, css_rgb = nearest_css3_color(rgb)
            name_row.append(name)
            css_row.append(css_rgb)
        name_grid.append(name_row)
        css_grid.append(css_row)
    return name_grid, css_grid


def build_legend(name_grid: List[List[str]]) -> Dict[str, List[List[int]]]:
    legend: Dict[str, List[List[int]]] = defaultdict(list)
    for row_index, row in enumerate(name_grid, start=1):
        for col_index, name in enumerate(row, start=1):
            legend[name].append([row_index, col_index])
    for positions in legend.values():
        positions.sort()
    return dict(sorted(legend.items(), key=lambda item: (len(item[1]) * -1, item[0])))


def create_coding_page(
    name_grid: List[List[str]],
    cell_size: int,
    output_path: Path,
    *,
    numbered: bool = False,
    color_numbers: Optional[Dict[str, int]] = None,
) -> None:
    rows = len(name_grid)
    cols = len(name_grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError("Grid is empty; nothing to draw")

    if numbered and not color_numbers:
        raise ValueError("Colour numbers are required when numbered page is requested")

    font = ImageFont.load_default()
    padding_left = cell_size
    padding_top = cell_size
    padding_right = cell_size // 2
    padding_bottom = cell_size // 2

    width = padding_left + cols * cell_size + padding_right
    height = padding_top + rows * cell_size + padding_bottom

    page = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(page)

    # Draw column numbers along the top.
    for col in range(cols):
        text = str(col + 1)
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        x = padding_left + col * cell_size + (cell_size - text_width) / 2
        y = (padding_top - font.size) / 2
        draw.text((x, y), text, fill="black", font=font)

    # Draw row numbers along the left.
    for row in range(rows):
        text = str(row + 1)
        bbox = font.getbbox(text)
        text_height = bbox[3] - bbox[1]
        x = (padding_left - bbox[2]) / 2
        y = padding_top + row * cell_size + (cell_size - text_height) / 2
        draw.text((x, y), text, fill="black", font=font)

    # Draw the empty grid.
    for row in range(rows):
        for col in range(cols):
            x0 = padding_left + col * cell_size
            y0 = padding_top + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], outline="black", width=1)
            if numbered and color_numbers:
                color_name = name_grid[row][col]
                number = color_numbers[color_name]
                center_x = x0 + cell_size / 2
                center_y = y0 + cell_size / 2
                draw.text((center_x, center_y), str(number), fill="black", font=font, anchor="mm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path)


def create_preview(css_grid: List[List[RGB]], cell_size: int, output_path: Path) -> None:
    rows = len(css_grid)
    cols = len(css_grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        raise ValueError("Grid is empty; nothing to draw")

    preview = Image.new("RGB", (cols * cell_size, rows * cell_size), color="white")
    draw = ImageDraw.Draw(preview)
    for row in range(rows):
        for col in range(cols):
            color = css_grid[row][col]
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)


def save_legend(
    legend: Dict[str, List[List[int]]],
    output_path: Path,
    *,
    color_numbers: Optional[Dict[str, int]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        if color_numbers:
            numbered_payload = {color: color_numbers[color] for color in legend}
            payload: Dict[str, Any] = {
                "colors": legend,
                "color_numbers": numbered_payload,
            }
            json.dump(payload, fp, indent=2)
        else:
            json.dump(legend, fp, indent=2)


def summarize(legend: Dict[str, List[List[int]]]) -> Iterable[str]:
    for color_name, positions in legend.items():
        yield f"{color_name}: {len(positions)} squares"


def summarize_numbering(color_numbers: Dict[str, int]) -> Iterable[str]:
    for color_name, number in sorted(color_numbers.items(), key=lambda item: item[1]):
        yield f"{number}: {color_name}"


def main() -> None:
    args = parse_args()
    height = args.height or args.width

    source_image = load_image(args.image)
    rgb_grid = downsample(source_image, args.width, height)
    rgb_grid = limit_palette(rgb_grid, args.max_colors)
    name_grid, css_grid = build_grid_with_names(rgb_grid)
    legend = build_legend(name_grid)
    color_numbers = assign_color_numbers(legend) if args.numbered_page else None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    create_coding_page(
        name_grid,
        args.cell_size,
        output_dir / args.page_file,
        numbered=args.numbered_page,
        color_numbers=color_numbers,
    )
    create_preview(css_grid, args.cell_size, output_dir / args.preview_file)
    save_legend(legend, output_dir / args.legend_file, color_numbers=color_numbers)

    legend_summary = "\n".join(summarize(legend))
    numbering_summary = "\n".join(summarize_numbering(color_numbers)) if color_numbers else ""
    print("Generated files:")
    print(f"  Coding page : {output_dir / args.page_file}")
    print(f"  Preview     : {output_dir / args.preview_file}")
    print(f"  Legend      : {output_dir / args.legend_file}")
    if legend_summary:
        print("Legend summary:")
        print(legend_summary)
    if numbering_summary:
        print("Colour numbers:")
        print(numbering_summary)


if __name__ == "__main__":
    main()

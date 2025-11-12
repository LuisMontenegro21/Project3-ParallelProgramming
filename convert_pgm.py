"""
Convertimos los ppm o pmg en graficas o pngs que se puedan ver

Ejemplos de uso:
    Uso individual:
  - python convert_pgm.py output_global.ppm
    Uso comparativo:
  - python convert_pgm.py output_global.ppm output_const.ppm output_shared.ppm
    Guardar resultados - individual:
  - python convert_pgm.py output_global.ppm --save resultados/global.png
    Guardar resultados - comparativo:
  - python convert_pgm.py output_global.ppm output_const.ppm output_shared.ppm --save resultados/comparacion.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys


def _maybe_set_headless_backend(save: bool, show: bool):
    if save and not show:
        import matplotlib
        matplotlib.use("Agg")


def read_pgm_or_ppm(path: Path) -> tuple[np.ndarray, int, int, str]:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P5", b"P6"):
            raise ValueError(f"Formato no soportado {magic!r}; esperado P5 o P6")

        def _read_non_comment_line():
            line = f.readline()
            while line.startswith(b"#") or line.strip() == b"":
                line = f.readline()
            return line

        dims = _read_non_comment_line()
        parts = dims.split()
        while len(parts) < 2:
            parts.extend(_read_non_comment_line().split())
        w, h = map(int, parts[:2])

        maxval = int(_read_non_comment_line().strip())

        if magic == b"P5":
            count = w * h
            data = np.frombuffer(f.read(count), dtype=np.uint8)
            img = data.reshape((h, w))
            mode = "PGM"
        else:
            count = w * h * 3
            data = np.frombuffer(f.read(count), dtype=np.uint8)
            img = data.reshape((h, w, 3))
            mode = "PPM"

        return img, w, h, mode


def make_figure(images: list[tuple[str, np.ndarray]]):
    import matplotlib.pyplot as plt
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for i, (title, img) in enumerate(images):
        ax = axes[0, i]
        if img.ndim == 2:
            im = ax.imshow(img, cmap="gray", aspect="auto")
        else:
            im = ax.imshow(img, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if img.ndim == 2:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Visualiza y compara imágenes PGM/PPM.")
    parser.add_argument("files", nargs="+", type=Path, help="Archivos PGM o PPM a visualizar.")
    parser.add_argument("--save", type=Path, default=None, help="Ruta para guardar la comparación como PNG.")
    parser.add_argument("--no-show", action="store_true", help="No mostrar ventana (solo se guarda).")
    args = parser.parse_args(argv)

    _maybe_set_headless_backend(save=args.save is not None, show=not args.no_show)
    import matplotlib.pyplot as plt

    imgs: list[tuple[str, np.ndarray]] = []
    for p in args.files:
        if not p.exists():
            print(f"Archivo no encontrado: {p}")
            continue
        try:
            img, w, h, mode = read_pgm_or_ppm(p)
            imgs.append((f"{p.name} ({mode}, {w}x{h})", img))
        except Exception as e:
            print(f"Error leyendo {p}: {e}")

    if not imgs:
        print("No se pudieron leer imágenes válidas.")
        return 1

    fig = make_figure(imgs)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Imagen guardada en {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

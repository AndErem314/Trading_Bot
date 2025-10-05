from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _wrap_text(text: str, max_chars: int = 100) -> str:
    lines = []
    for para in text.split('\n'):
        if len(para) <= max_chars:
            lines.append(para)
            continue
        buf = []
        for word in para.split(' '):
            if sum(len(x) for x in buf) + len(buf) + len(word) > max_chars:
                lines.append(' '.join(buf))
                buf = [word]
            else:
                buf.append(word)
        if buf:
            lines.append(' '.join(buf))
    return '\n'.join(lines)


def write_llm_pdf(*,
                  output_dir: str,
                  filename_prefix: str,
                  title: str,
                  text_body: str) -> str:
    """Write a simple, clean PDF containing the provided text.

    Returns the absolute path to the PDF file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = out_dir / f"{filename_prefix}_{ts}_llm.pdf"

    wrapped = _wrap_text(text_body, max_chars=110)

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.axis('off')
        # Use monospace font for consistent alignment
        plt.text(0.06, 0.95, wrapped, transform=fig.transFigure,
                 fontsize=10, fontfamily='monospace', va='top')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return str(pdf_path)

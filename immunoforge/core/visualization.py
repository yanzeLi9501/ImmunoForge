"""
Visualization & Report Generation Module.

Generates interactive HTML reports with embedded Plotly charts and
Markdown-formatted summary documents for ImmunoForge pipeline runs.

Features:
    - Candidate scatter plots (K_D vs composite score)
    - K_D distribution histograms
    - Benchmark calibration plots (predicted vs experimental)
    - Structure quality heatmaps
    - Developability radar charts
    - Full pipeline HTML report generator
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Plotly is optional — use lightweight inline SVG fallback if unavailable
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ═══════════════════════════════════════════════════════════════════
# Chart generators (Plotly-based with HTML fallback)
# ═══════════════════════════════════════════════════════════════════

def kd_vs_score_scatter(
    candidates: list[dict],
    title: str = "Candidate K<sub>D</sub> vs Composite Score",
) -> str:
    """Generate interactive scatter plot of K_D vs composite score.

    Args:
        candidates: List of dicts with keys:
            binder_id, predicted_kd_nM, composite_score, [quality]
    Returns:
        HTML string (Plotly div or static table fallback).
    """
    if HAS_PLOTLY:
        ids = [c.get("binder_id", f"C{i}") for i, c in enumerate(candidates)]
        kds = [c.get("predicted_kd_nM", 0) for c in candidates]
        scores = [c.get("composite_score", 0) for c in candidates]
        colors = [c.get("quality", "MEDIUM") for c in candidates]
        color_map = {"HIGH": "#2ecc71", "MEDIUM": "#f39c12", "LOW": "#e74c3c"}

        fig = go.Figure()
        for q in ["HIGH", "MEDIUM", "LOW"]:
            mask = [i for i, c in enumerate(colors) if c == q]
            if mask:
                fig.add_trace(go.Scatter(
                    x=[kds[i] for i in mask],
                    y=[scores[i] for i in mask],
                    mode="markers",
                    marker=dict(size=10, color=color_map.get(q, "#999")),
                    text=[ids[i] for i in mask],
                    name=q,
                    hovertemplate="<b>%{text}</b><br>K<sub>D</sub>: %{x:.1f} nM<br>Score: %{y:.3f}",
                ))

        fig.update_layout(
            title=title, xaxis_title="Predicted K<sub>D</sub> (nM)",
            yaxis_title="Composite Score", template="plotly_white",
            xaxis_type="log",
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    # Fallback: simple HTML table
    return _table_fallback(candidates, ["binder_id", "predicted_kd_nM", "composite_score"])


def kd_distribution_histogram(
    kd_values: list[float],
    title: str = "K<sub>D</sub> Distribution",
) -> str:
    """Generate K_D distribution histogram."""
    if HAS_PLOTLY:
        import math
        log_kds = [math.log10(max(k, 0.001)) for k in kd_values]
        fig = go.Figure(data=[go.Histogram(
            x=log_kds, nbinsx=20,
            marker_color="#3498db", opacity=0.8,
        )])
        fig.update_layout(
            title=title, xaxis_title="log₁₀ K<sub>D</sub> (nM)",
            yaxis_title="Count", template="plotly_white",
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    return f"<p>K_D values (n={len(kd_values)}): min={min(kd_values):.2f}, max={max(kd_values):.2f}</p>"


def calibration_plot(
    experimental: list[float],
    predicted: list[float],
    labels: list[str] | None = None,
    title: str = "Benchmark Calibration: Predicted vs Experimental K<sub>D</sub>",
) -> str:
    """Generate calibration scatter with y=x reference line."""
    if HAS_PLOTLY:
        import math
        log_exp = [math.log10(max(v, 0.001)) for v in experimental]
        log_pred = [math.log10(max(v, 0.001)) for v in predicted]
        lo = min(min(log_exp), min(log_pred)) - 0.5
        hi = max(max(log_exp), max(log_pred)) + 0.5

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=log_exp, y=log_pred, mode="markers",
            marker=dict(size=10, color="#2980b9"),
            text=labels or [f"Entry {i}" for i in range(len(experimental))],
            hovertemplate="<b>%{text}</b><br>Exp: 10^%{x:.1f} nM<br>Pred: 10^%{y:.1f} nM",
        ))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(dash="dash", color="gray"), showlegend=False,
        ))
        fig.update_layout(
            title=title, xaxis_title="log₁₀ Experimental K<sub>D</sub> (nM)",
            yaxis_title="log₁₀ Predicted K<sub>D</sub> (nM)",
            template="plotly_white",
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    return "<p>Calibration data: {} entries</p>".format(len(experimental))


def developability_radar(
    scores: dict[str, float],
    binder_id: str = "Candidate",
    title: str = "Developability Profile",
) -> str:
    """Generate radar/spider chart for developability properties."""
    if HAS_PLOTLY:
        categories = list(scores.keys())
        values = list(scores.values())
        values.append(values[0])  # Close the polygon
        categories.append(categories[0])

        fig = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill="toself",
            marker_color="#27ae60", name=binder_id,
        ))
        fig.update_layout(
            title=f"{title} — {binder_id}",
            polar=dict(radialaxis=dict(range=[0, 1])),
            template="plotly_white",
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

    return _table_fallback([{"property": k, "score": f"{v:.3f}"} for k, v in scores.items()],
                           ["property", "score"])


# ═══════════════════════════════════════════════════════════════════
# Full pipeline report
# ═══════════════════════════════════════════════════════════════════

def generate_pipeline_report(
    pipeline_results: dict,
    output_path: str | None = None,
    title: str = "ImmunoForge Pipeline Report",
) -> str:
    """Generate comprehensive HTML report from pipeline results.

    Args:
        pipeline_results: Dict with keys from pipeline steps:
            - candidates: list of candidate dicts
            - benchmark: optional benchmark results
            - config: pipeline configuration
        output_path: If provided, write HTML to this file.

    Returns:
        Complete HTML string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    candidates = pipeline_results.get("candidates", [])
    benchmark = pipeline_results.get("benchmark", None)
    config = pipeline_results.get("config", {})

    sections = []

    # Header
    sections.append(f"""
    <header style="background:#2c3e50;color:white;padding:20px;border-radius:8px;margin-bottom:20px">
        <h1>{title}</h1>
        <p>Generated: {timestamp}</p>
        <p>Candidates: {len(candidates)} | Target: {config.get('target', 'N/A')}</p>
    </header>
    """)

    # Summary stats
    if candidates:
        kds = [c.get("predicted_kd_nM", 0) for c in candidates if c.get("predicted_kd_nM")]
        scores = [c.get("composite_score", 0) for c in candidates if c.get("composite_score")]

        sections.append("""
        <section><h2>Summary Statistics</h2>
        <table style="border-collapse:collapse;width:60%">
        <tr><th style="text-align:left;padding:8px;border-bottom:1px solid #ddd">Metric</th>
            <th style="text-align:right;padding:8px;border-bottom:1px solid #ddd">Value</th></tr>
        """)
        if kds:
            sections.append(f"<tr><td>Best K<sub>D</sub> (nM)</td><td style='text-align:right'>{min(kds):.2f}</td></tr>")
            sections.append(f"<tr><td>Median K<sub>D</sub> (nM)</td><td style='text-align:right'>{sorted(kds)[len(kds)//2]:.2f}</td></tr>")
        if scores:
            sections.append(f"<tr><td>Best composite score</td><td style='text-align:right'>{max(scores):.3f}</td></tr>")
        sections.append("</table></section>")

        # K_D scatter
        sections.append(f"<section><h2>K<sub>D</sub> vs Composite Score</h2>{kd_vs_score_scatter(candidates)}</section>")

        # K_D distribution
        if kds:
            sections.append(f"<section><h2>K<sub>D</sub> Distribution</h2>{kd_distribution_histogram(kds)}</section>")

    # Benchmark calibration
    if benchmark and "predictions" in benchmark:
        preds = benchmark["predictions"]
        exp_kds = [p["experimental_kd_nM"] for p in preds if p.get("predicted_kd_nM")]
        pred_kds = [p["predicted_kd_nM"] for p in preds if p.get("predicted_kd_nM")]
        labels = [f"{p['target']}/{p['binder']}" for p in preds if p.get("predicted_kd_nM")]

        if exp_kds:
            sections.append(f"<section><h2>Benchmark Calibration</h2>{calibration_plot(exp_kds, pred_kds, labels)}</section>")

        metrics = benchmark.get("metrics", {})
        if metrics:
            sections.append("<section><h2>Benchmark Metrics</h2><table style='border-collapse:collapse;width:60%'>")
            for k, v in metrics.items():
                sections.append(f"<tr><td style='padding:4px'>{k}</td><td style='text-align:right;padding:4px'>{v}</td></tr>")
            sections.append("</table></section>")

    # Candidate table
    if candidates:
        sections.append("<section><h2>Top Candidates</h2>")
        top = sorted(candidates, key=lambda c: c.get("composite_score", 0), reverse=True)[:20]
        sections.append(_candidate_table_html(top))
        sections.append("</section>")

    html = _wrap_html(title, "\n".join(sections))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)

    return html


# ═══════════════════════════════════════════════════════════════════
# Markdown report
# ═══════════════════════════════════════════════════════════════════

def generate_markdown_report(pipeline_results: dict) -> str:
    """Generate Markdown summary report."""
    candidates = pipeline_results.get("candidates", [])
    config = pipeline_results.get("config", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# ImmunoForge Pipeline Report",
        f"*Generated: {timestamp}*",
        "",
        "## Configuration",
        f"- Target: {config.get('target', 'N/A')}",
        f"- Species: {config.get('species', 'N/A')}",
        f"- Candidates: {len(candidates)}",
        "",
    ]

    if candidates:
        kds = [c.get("predicted_kd_nM", 0) for c in candidates if c.get("predicted_kd_nM")]
        scores = [c.get("composite_score", 0) for c in candidates if c.get("composite_score")]

        lines.append("## Summary")
        if kds:
            lines.append(f"- Best K_D: {min(kds):.2f} nM")
            lines.append(f"- Median K_D: {sorted(kds)[len(kds)//2]:.2f} nM")
        if scores:
            lines.append(f"- Best composite score: {max(scores):.3f}")
        lines.append("")

        lines.append("## Top 10 Candidates")
        lines.append("| Rank | ID | K_D (nM) | Score | Quality |")
        lines.append("|------|-----|----------|-------|---------|")
        top = sorted(candidates, key=lambda c: c.get("composite_score", 0), reverse=True)[:10]
        for i, c in enumerate(top, 1):
            lines.append(
                f"| {i} | {c.get('binder_id', 'N/A')} | "
                f"{c.get('predicted_kd_nM', 'N/A')} | "
                f"{c.get('composite_score', 'N/A'):.3f} | "
                f"{c.get('quality', 'N/A')} |"
            )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _table_fallback(rows: list[dict], columns: list[str]) -> str:
    """Generate simple HTML table when Plotly is unavailable."""
    header = "".join(f"<th style='padding:6px;border:1px solid #ddd'>{c}</th>" for c in columns)
    body = ""
    for row in rows:
        cells = "".join(
            f"<td style='padding:6px;border:1px solid #ddd'>{row.get(c, '')}</td>"
            for c in columns
        )
        body += f"<tr>{cells}</tr>"
    return f"<table style='border-collapse:collapse'><tr>{header}</tr>{body}</table>"


def _candidate_table_html(candidates: list[dict]) -> str:
    """Generate HTML table for candidate list."""
    columns = ["binder_id", "predicted_kd_nM", "composite_score", "quality"]
    header = "".join(f"<th style='padding:8px;border-bottom:2px solid #2c3e50;text-align:left'>{c}</th>" for c in columns)
    rows = ""
    for c in candidates:
        cells = "".join(
            f"<td style='padding:6px;border-bottom:1px solid #eee'>{c.get(col, 'N/A')}</td>"
            for col in columns
        )
        rows += f"<tr>{cells}</tr>"
    return f"<table style='border-collapse:collapse;width:100%'><tr>{header}</tr>{rows}</table>"


def _wrap_html(title: str, body: str) -> str:
    """Wrap content in full HTML document."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f6fa; color: #2c3e50; }}
        section {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
        table {{ margin: 10px 0; }}
    </style>
</head>
<body>
{body}
</body>
</html>"""

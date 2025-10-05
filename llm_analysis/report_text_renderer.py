import json
from typing import Tuple, Dict, Any


def parse_llm_output(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse LLM output which should be:
    <JSON>\n---MEMO---\n<memo text>
    Returns (json_obj, memo_text). If JSON parsing fails, returns ({}, text).
    """
    if not text:
        return {}, ""
    # Split on the first occurrence of line with ---MEMO---
    parts = text.split("---MEMO---", 1)
    json_str = parts[0].strip()
    memo = parts[1].strip() if len(parts) > 1 else ""
    try:
        # Some models wrap JSON in code fences; strip them
        if json_str.startswith("```"):
            json_str = json_str.strip('`')
            # Remove possible language hint
            first_newline = json_str.find('\n')
            if first_newline != -1:
                json_str = json_str[first_newline+1:]
        obj = json.loads(json_str)
    except Exception:
        return {}, text
    return obj, memo


def _format_param_changes(pc: Dict[str, Any]) -> str:
    lines = []
    if not pc:
        return ""
    ichi = pc.get("ichimoku", {})
    for k in ("tenkan_period","kijun_period","senkou_b_period","chikou_offset","senkou_offset"):
        if k in ichi:
            cur = ichi[k].get("current")
            sug = ichi[k].get("suggested")
            rat = ichi[k].get("rationale", "")
            lines.append(f"- {k}: current={cur}, suggested={sug}. {rat}")
    sl = pc.get("signal_logic", {})
    if sl:
        lines.append("- Signal logic:")
        for kk in ("buy_logic","sell_logic"):
            if kk in sl:
                lines.append(f"  • {kk}: {sl[kk].get('current')} -> {sl[kk].get('suggested')} ({sl[kk].get('rationale','')})")
        ac = sl.get("add_conditions", [])
        rc = sl.get("remove_conditions", [])
        if ac:
            lines.append(f"  • add_conditions: {ac}")
        if rc:
            lines.append(f"  • remove_conditions: {rc}")
    rm = pc.get("risk_management", {})
    if rm:
        lines.append("- Risk management:")
        for kk in ("stop_loss_pct","take_profit_pct","position_sizing"):
            if kk in rm:
                v = rm[kk]
                if isinstance(v, dict):
                    lines.append(f"  • {kk}: {v.get('current')} -> {v.get('suggested')} ({v.get('rationale','')})")
                else:
                    lines.append(f"  • {kk}: {v}")
    return "\n".join(lines)


def build_final_text(title: str, json_obj: Dict[str, Any], memo_text: str) -> str:
    """Compose final plaintext for PDF: title, memo bullets, and parameter changes list."""
    lines = []
    lines.append(title)
    lines.append("=")
    # Memo
    if memo_text:
        lines.append("")
        lines.append(memo_text.strip())
    # Parameter changes summary
    pc = json_obj.get("parameter_changes", {}) if json_obj else {}
    pc_block = _format_param_changes(pc)
    if pc_block:
        lines.append("")
        lines.append("Recommended setting changes:")
        lines.append(pc_block)
    # Experiments (optional)
    exps = json_obj.get("experiments", []) if json_obj else []
    if exps:
        lines.append("")
        lines.append("Experiments to run:")
        for e in exps[:5]:
            name = e.get("name", "Experiment")
            desc = e.get("description", "")
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)

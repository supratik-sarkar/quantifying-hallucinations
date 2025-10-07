def summarize_runtime(runtimes):
    # runtimes: dict{name: seconds}
    rows = ["| Method | Runtime (s) |", "|---|---:|"]
    for k,v in runtimes.items():
        rows.append(f"| {k} | {v:.2f} |")
    return "\n".join(rows)


def dbg(msg, mode="debug"):
    colors = {
        "debug": "\033[92m[*]\t",
        "warning": "\033[93m[!]\t",
        "error": "\033[91m[!]\t",
        "clear": "\033[0m",
    }
    if mode == "crit":
        raise Exception(colors["error"] + str(msg) + colors["clear"])
    print(colors[mode] + str(msg) + colors["clear"])

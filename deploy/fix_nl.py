from pathlib import Path
for name in ("remote_switch.sh", "restart.sh"):
    p = Path(__file__).with_name(name)
    if p.is_file():
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
        print(name, p.stat().st_size)

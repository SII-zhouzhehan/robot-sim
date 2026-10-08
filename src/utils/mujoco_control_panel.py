"""Optional localhost slider panel, using only Python's standard library."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import webbrowser


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>机器人控制 · MuJoCo</title>
<style>
body{font:16px system-ui;background:#f3f5f7;color:#182531;margin:0;padding:30px}
main{max-width:640px;margin:auto;background:white;border-radius:20px;padding:28px;box-shadow:0 8px 40px #15243412}
h1{margin:0 0 8px;font-size:27px}p{color:#596b7a;line-height:1.6}label{display:block;margin:22px 0 8px}
input[type=range]{width:100%;accent-color:#237a78}output{float:right;font-variant-numeric:tabular-nums}
.buttons{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}button{border:0;border-radius:9px;padding:11px 16px;cursor:pointer;background:#eaf0f3;color:#182531}
button.stop{background:#b84242;color:white}.switches{display:flex;gap:25px}.switches label{display:flex;gap:8px}
pre{font:13px ui-monospace,monospace;background:#f3f5f7;padding:16px;border-radius:10px;white-space:pre-wrap}
</style><main><h1>机器人控制 · MuJoCo</h1><p>速度使用世界坐标系。变形会平滑过渡，停止按钮立即清零行驶指令。</p>
<label>形态 Φ <output id="phi_out">1.50</output></label><input id="phi" type="range" min="0" max="3" step="0.01" value="1.5">
<label>X 速度（m/s）<output id="vx_out">0.00</output></label><input id="vx" type="range" min="-0.427" max="0.427" step="0.01" value="0">
<label>Y 速度（m/s）<output id="vy_out">0.00</output></label><input id="vy" type="range" min="-0.427" max="0.427" step="0.01" value="0">
<label>转向速度（rad/s）<output id="omega_out">0.00</output></label><input id="omega" type="range" min="-1.5" max="1.5" step="0.05" value="0">
<div class="buttons"><button onclick="shape(0)">8 形 · Φ 0</button><button onclick="shape(1.5)">O 形 · Φ 1.5</button><button onclick="shape(2)">拱形 · Φ 2</button><button onclick="shape(3)">1 形 · Φ 3</button></div>
<div class="switches"><label><input id="gnn" type="checkbox" checked> GNN 形态控制</label><label><input id="pid" type="checkbox" checked> PID 速度闭环</label></div>
<div class="buttons"><button class="stop" onclick="stop()">停止行驶</button><button onclick="send({reset:true})">重置机器人</button></div>
<pre id="status">正在连接仿真…</pre></main>
<script>
const ids=['phi','vx','vy','omega'];let sending=false,dirty=false;
function update(){ids.forEach(k=>document.getElementById(k+'_out').textContent=Number(document.getElementById(k).value).toFixed(2))}
async function send(extra={}){const state={};ids.forEach(k=>state[k]=Number(document.getElementById(k).value));state.source=document.getElementById('gnn').checked?'gnn':'formula';state.pid=document.getElementById('pid').checked;Object.assign(state,extra);try{const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state)});if(!r.ok)throw Error(await r.text())}catch(e){document.getElementById('status').textContent=e.message}}
function shape(phi){document.getElementById('phi').value=phi;update();send()}
function stop(){['vx','vy','omega'].forEach(k=>document.getElementById(k).value=0);update();send()}
ids.forEach(k=>document.getElementById(k).oninput=()=>{update();dirty=true});['gnn','pid'].forEach(k=>document.getElementById(k).onchange=()=>send());
setInterval(async()=>{if(dirty&&!sending){dirty=false;sending=true;await send();sending=false}},50);
setInterval(async()=>{try{const s=await(await fetch('/state')).json();document.getElementById('status').textContent=`Φ ${s.actual_phi.toFixed(2)} / ${s.phi.toFixed(2)}   ${s.source.toUpperCase()}   PID ${s.pid?'开':'关'}\n位置 (${s.pose.map(v=>v.toFixed(3)).join(', ')})\n速度 (${s.velocity.map(v=>v.toFixed(3)).join(', ')})\n实时速度 ${s.realtime.toFixed(2)}×   轮地接触 ${s.wheel_contacts}/4`;document.getElementById('gnn').disabled=!s.gnn_available;if(!s.gnn_available)document.getElementById('gnn').checked=false}catch(e){document.getElementById('status').textContent='仿真已关闭或连接中断'}},250);
fetch('/state').then(r=>r.json()).then(s=>{ids.forEach(k=>document.getElementById(k).value=s[k]);document.getElementById('gnn').checked=s.source==='gnn';document.getElementById('pid').checked=s.pid;update()});
</script></html>"""


class ControlState:
    def __init__(self, phi=1.5, vx=0.0, vy=0.0, omega=0.0, source="gnn", pid=True):
        self.lock = threading.Lock()
        self.values = dict(phi=phi, vx=vx, vy=vy, omega=omega, source=source, pid=pid, reset=False,
                           actual_phi=1.5, pose=[0, 0, 0], velocity=[0, 0, 0], realtime=0.0,
                           wheel_contacts=0, gnn_available=source == "gnn")

    def snapshot(self):
        with self.lock:
            return self.values.copy()

    def update(self, values=None, **kwargs):
        with self.lock:
            if values is not None:
                self.values.update(values)
            self.values.update(kwargs)

    def command(self, values):
        if not isinstance(values, dict):
            raise ValueError("Command must be an object")
        validated = {}
        for name, limit in (("phi", 3), ("vx", 0.427), ("vy", 0.427), ("omega", 1.5)):
            if name in values:
                value = float(values[name])
                if not math.isfinite(value):
                    raise ValueError(f"Invalid {name}")
                validated[name] = min(limit, max(0 if name == "phi" else -limit, value))
        if "source" in values:
            if values["source"] not in ("gnn", "formula"):
                raise ValueError("Unknown morphology source")
            if values["source"] == "gnn" and not self.snapshot()["gnn_available"]:
                raise ValueError("GNN was not loaded")
            validated["source"] = values["source"]
        for name in ("pid", "reset"):
            if name in values:
                if not isinstance(values[name], bool):
                    raise ValueError(f"{name} must be a boolean")
                validated[name] = values[name]
        self.update(validated)

    def consume_reset(self):
        with self.lock:
            reset = self.values["reset"]
            self.values["reset"] = False
            return reset

    def key(self, keycode):
        key = chr(keycode).upper() if 0 <= keycode < 128 else ""
        with self.lock:
            values = self.values
            for button, name, delta in (("W", "vx", .05), ("S", "vx", -.05), ("A", "vy", .05),
                                         ("D", "vy", -.05), ("Q", "omega", .1), ("E", "omega", -.1),
                                         ("Z", "phi", -.1), ("X", "phi", .1)):
                if key == button:
                    limit = {"vx": .427, "vy": .427, "omega": 1.5, "phi": 3}[name]
                    values[name] = min(limit, max(0 if name == "phi" else -limit, values[name] + delta))
            if key == " ":
                values.update(vx=0, vy=0, omega=0)
            elif key == "R":
                values["reset"] = True
            elif key == "P":
                values["pid"] = not values["pid"]
            elif key == "G" and values["gnn_available"]:
                values["source"] = "formula" if values["source"] == "gnn" else "gnn"


class ControlPanel:
    def __init__(self, state, open_browser=True):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body, content_type = PAGE.encode(), "text/html; charset=utf-8"
                elif self.path == "/state":
                    body, content_type = json.dumps(state.snapshot()).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/command":
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("Invalid command size")
                    state.command(json.loads(self.rfile.read(size)))
                except (ValueError, TypeError, OverflowError) as error:
                    self.send_error(400, str(error))
                    return
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        if open_browser:
            webbrowser.open(self.url)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

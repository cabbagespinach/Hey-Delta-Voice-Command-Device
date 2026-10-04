"""
Live dashboard of the assistant's home, served by the Pi, opened in any browser on the same network (the laptop on
demo day; owner 2026-10-02: no monitor at the demo). Standard library only; the page is self-contained (no internet).

    http://<pi-ip>:8080/                the page (number-style address, printed at start) (updates twice a second)
    http://<pi-ip>:8080/state           the same data as JSON (Home.snapshot())

Read-only: it shows the home's state and the last commands; it cannot control anything. It shows the contact's
name, never the number.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, socket, subprocess, threading, time

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hey Delta Home</title>
<style>
:root{--bg:#f6f6f3;--card:#ffffff;--ink:#141413;--ink2:#55534e;--muted:#8a877f;--line:#e4e2dc;--accent:#2a78d6;
--good:#1f8a4c;--warn:#b26a00;--track:#ecebe6}
@media (prefers-color-scheme: dark){:root{--bg:#121211;--card:#1d1d1b;--ink:#f4f3ee;--ink2:#c3c2b7;--muted:#8f8d84;
--line:#2f2e2b;--accent:#5b9bea;--good:#4cc27e;--warn:#e0a03c;--track:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:18px 20px 6px}
h1{font-size:22px;margin:0}.sub{color:var(--muted);font-size:13px}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);font-weight:600}
.dot{width:10px;height:10px;border-radius:50%;background:var(--muted)}.live .dot{background:var(--good);box-shadow:0 0 0 4px color-mix(in srgb,var(--good) 25%,transparent)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;padding:12px 20px 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px;display:flex;justify-content:space-between}
.tag{font-size:10px;letter-spacing:.04em;padding:2px 7px;border-radius:999px;border:1px solid var(--line);color:var(--ink2);text-transform:none}
.big{font-size:44px;font-weight:700;line-height:1}.mid{font-size:20px;font-weight:650}
.row{display:flex;align-items:center;gap:14px}.muted{color:var(--muted)}
.bulb{width:64px;height:64px;border-radius:50%;border:2px solid var(--line);flex:none;transition:all .3s}
.bar{height:8px;border-radius:4px;background:var(--track);overflow:hidden;margin-top:8px}.bar>i{display:block;height:100%;background:var(--accent);border-radius:4px}
ul{list-style:none;margin:0;padding:0}li{padding:6px 0;border-top:1px solid var(--line)}li:first-child{border-top:0}
.wide{grid-column:1/-1}.cmd{font-weight:650}.reply{color:var(--ink2)}.time{color:var(--muted);font-variant-numeric:tabular-nums;font-size:13px}
.err{color:var(--warn)}
</style></head><body>
<header><div><h1>Hey Delta Home</h1><div class="sub" id="devices">connecting…</div></div>
<div class="pill" id="status"><span class="dot"></span><span id="statustext">…</span></div></header>
<main>
 <section class="card"><h2>Lights <span class="tag" id="lighttag"></span></h2>
  <div class="row"><div class="bulb" id="bulb"></div><div><div class="mid" id="lighttext"></div><div class="muted" id="lightsub"></div></div></div></section>
 <section class="card"><h2>Thermostat <span class="tag">simulated</span></h2><div class="big" id="temp"></div><div class="muted">set point</div></section>
 <section class="card"><h2>Music</h2><div class="mid" id="music"></div><div class="muted" id="track"></div>
  <div class="muted" style="margin-top:10px">Volume <span id="vol"></span></div><div class="bar"><i id="volbar"></i></div></section>
 <section class="card"><h2>Timers</h2><ul id="timers"></ul></section>
 <section class="card"><h2>Alarms</h2><ul id="alarms"></ul></section>
 <section class="card"><h2>Reminders</h2><ul id="reminders"></ul></section>
 <section class="card"><h2>Phone <span class="tag" id="phonetag"></span></h2><div id="phone"></div></section>
 <section class="card"><h2>Weather <span class="tag" id="wtag"></span></h2><div id="weather"></div></section>
 <section class="card wide"><h2>Last commands</h2><ul id="events"></ul></section>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const COLORS={white:"#fff6dc",red:"#e5484d",blue:"#3b82f6",green:"#22a35a"};
const list=(el,items,empty)=>{$(el).innerHTML=items.length?items.map(x=>`<li>${x}</li>`).join(""):`<li class="muted">${empty}</li>`};
const fmtS=s=>{s=Math.ceil(s);return `${Math.floor(s/60)}:${String(s%60).padStart(2,"0")}`};
const fmtT=t=>new Date(t*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
const fmtIso=s=>s?new Date(s).toLocaleString([], {weekday:"short",hour:"numeric",minute:"2-digit"}):"";
async function tick(){
 try{
  const d=await (await fetch("/state",{cache:"no-store"})).json();
  const on=x=>x?"real":"simulated";
  $("devices").textContent=`${d.clock} · bulb ${on(d.devices.bulb)} · phone ${d.devices.phone?"set up":"not set up"} · weather ${d.devices.weather?"live":"offline"}`;
  $("status").className="pill"+(d.listening?" live":"");$("statustext").textContent=d.listening?"Listening…":"Say “Hey Delta”";
  const L=d.lights,c=COLORS[L.color]||"#fff6dc";
  $("bulb").style.background=L.on?c:"transparent";$("bulb").style.opacity=L.on?(0.35+0.65*L.brightness/100):1;
  $("bulb").style.boxShadow=L.on?`0 0 ${10+L.brightness/3}px ${c}`:"none";
  $("lighttext").textContent=L.on?"On":"Off";$("lightsub").textContent=`${L.brightness}% · ${L.color}`;$("lighttag").textContent=L.real?"Tapo bulb":"simulated";
  $("temp").textContent=`${d.thermostat}°C`;
  $("music").textContent=d.music.state[0].toUpperCase()+d.music.state.slice(1);$("track").textContent=d.music.track||"";
  $("vol").textContent=`${d.volume}%`;$("volbar").style.width=`${d.volume}%`;
  list("timers",d.timers.map(t=>`<span class="mid">${fmtS(t.seconds_left)}</span> <span class="muted">${esc(t.name)} timer</span>`),"No timers");
  list("alarms",d.alarms.map(a=>esc(fmtIso(a))),"No alarms");
  list("reminders",d.reminders.map(esc),"No reminders");
  $("phonetag").textContent=d.devices.phone?"Bluetooth":"not set up";
  $("phone").innerHTML=d.devices.phone?`<div class="mid">${esc(d.phone.contact)}</div><div class="muted">last call: ${esc(fmtIso(d.phone.last_call))||"–"}<br>last text: ${esc(fmtIso(d.phone.last_message))||"–"}</div>`:`<div class="muted">Calls and texts are not set up.</div>`;
  $("wtag").textContent=d.devices.weather?"Open-Meteo":"offline";$("weather").innerHTML=d.weather?esc(d.weather):`<span class="muted">Ask “What's the weather?”</span>`;
  list("events",d.events.map(e=>`<span class="time">${fmtT(e.t)}</span> &nbsp;<span class="cmd">${esc(e.command)}</span>${e.confidence!=null?` <span class="muted">(${Math.round(e.confidence*100)}%)</span>`:""}<br><span class="reply">“${esc(e.reply)}”</span>`),"Nothing yet");
 }catch(e){$("statustext").textContent="Assistant not reachable";$("status").className="pill";$("devices").innerHTML='<span class="err">Is assistant.py running on the Pi?</span>'}
}
tick();setInterval(tick,500);
</script></body></html>"""


def serve(home, port: int = 8080, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    """Start the dashboard in a background thread; returns the server (call .shutdown() to stop)."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/state":
                body, ctype = json.dumps(home.snapshot()).encode(), "application/json"
            elif self.path.split("?")[0] in ("/", "/index.html"):
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):                      # keep the console for the assistant
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def lan_ips() -> list[str]:
    """The Pi's number-style (IPv4) addresses on the network, e.g. ['192.168.1.23']; [] while not connected."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))                # no packet is sent; picks the outgoing interface
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:                                                # every interface (Wi-Fi + Ethernet), Linux only
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
        ips += [ip for ip in out if ip.count(".") == 3]
    except (OSError, subprocess.SubprocessError):
        pass
    return [ip for i, ip in enumerate(ips) if ip not in ips[:i] and not ip.startswith(("127.", "169.254.", "172.17."))]


def addresses(port: int) -> list[str]:
    """Number-style addresses first (they work on every network), the .local name last."""
    return [f"http://{ip}:{port}" for ip in lan_ips()] + [f"http://{socket.gethostname()}.local:{port}"]


def instructions(port: int) -> str:
    """What to type on the laptop / phone to open the dashboard, and what to do if it does not open."""
    ips = lan_ips()
    if ips:
        main = "\n".join(f"        http://{ip}:{port}" for ip in ips)
        head = "On your laptop or phone, connected to the SAME Wi-Fi / hotspot as this Pi, open a browser and type\n" \
               "this into the address bar exactly (http, not https):\n\n" + main
    else:
        head = ("This Pi is not on a network yet: no address to show. Connect it to the Wi-Fi / hotspot; the address\n"
                "is printed here as soon as it has one (or run: hostname -I).")
    return f"""
================================ DASHBOARD ================================
{head}

  If the page does not open:
  1. "Can't provide a secure connection" / the address changed to https://:
     the dashboard has no https. Type http:// yourself, then
       - Chrome / Edge: click "Continue to site", or Settings > Privacy and security >
         Security > turn off "Always use secure connections" (or use an Incognito window).
       - Firefox: click "Continue to HTTP Site", or Settings > Privacy & Security >
         HTTPS-Only Mode > Manage Exceptions > add the address above.
  2. Mac (macOS 15 or newer): System Settings > Privacy & Security > Local Network >
     turn ON your browser, then reload. iPhone / iPad: Settings > Privacy & Security >
     Local Network > turn ON your browser.
  3. Same network: the laptop/phone must be on the same Wi-Fi or hotspot as the Pi,
     not on mobile data, VPN off. Guest networks often block devices from seeing each other.
  4. Still nothing: the address changes when the Pi joins another network. Check it again
     on the Pi with: hostname -I   (use the first number, add :{port})
===========================================================================
"""


def announce(port: int, every: float = 10.0) -> threading.Thread:
    """Print the instructions now, and again whenever the Pi's address changes (e.g. the hotspot connects after
    an autostart at boot): the address is always the last one printed (journalctl when autostarted)."""
    def loop():
        last = None
        while True:
            ips = lan_ips()
            if ips != last:
                print(instructions(port), flush=True)
                last = ips
            time.sleep(every)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th

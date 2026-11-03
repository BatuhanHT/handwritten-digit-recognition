// Electron ana süreci: pencereyi açar, src/predict.py'yi --serve kipinde başlatır ve
// Python'un stdout satırlarını olduğu gibi arayüzdeki terminal paneline iletir.
const { app, BrowserWindow, ipcMain, screen } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const readline = require('node:readline');
const { spawn } = require('node:child_process');

const ROOT = path.resolve(__dirname, '..');
let win = null;
let py = null;

function pythonPath() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = path.join(ROOT, '.venv', 'bin', 'python');
  return fs.existsSync(venv) ? venv : 'python3';
}

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

// Her sayfa yüklemesinde (ilk açılış ya da Cmd+R) Python baştan başlatılır;
// böylece ölmüş bir süreç yenilemeyle kurtarılır ve ekranda eski "hazır" bilgisi kalmaz.
function startPython() {
  if (py) {
    const old = py;
    py = null;
    old.kill();
  }
  const exe = pythonPath();
  const shown = path.isAbsolute(exe) ? path.relative(ROOT, exe) : exe;
  send('py-line', { stream: 'cmd', text: `$ ${shown} -u src/predict.py --serve` });

  const child = spawn(exe, ['-u', 'src/predict.py', '--serve'], {
    cwd: ROOT,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  py = child;
  const own = (fn) => (...args) => { if (py === child) fn(...args); }; // eski süreçten gelenleri yok say

  readline.createInterface({ input: child.stdout }).on('line', own((text) => send('py-line', { stream: 'stdout', text })));
  readline.createInterface({ input: child.stderr }).on('line', own((text) => send('py-line', { stream: 'stderr', text })));
  child.stdin.on('error', () => {}); // EPIPE: süreç öldüyse bunu 'exit' olayı bildirir
  child.on('error', own((err) => {
    py = null;
    send('py-line', { stream: 'stderr', text: `Python başlatılamadı (${exe}): ${err.message}` });
    send('py-exit', { code: null, signal: null });
  }));
  child.on('exit', own((code, signal) => {
    py = null;
    send('py-exit', { code, signal });
  }));
}

function createWindow() {
  // Ekranın kullanılabilir alanını doldur; tam ekran için Ctrl+Cmd+F (Görünüm menüsü)
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  win = new BrowserWindow({
    width,
    height,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#0E1726',
    title: 'El Yazısı Rakam Tanıma — Canlı Demo',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.webContents.on('did-finish-load', startPython);
  win.webContents.on('will-navigate', (event) => event.preventDefault()); // ör. pencereye dosya sürüklenirse
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.loadFile(path.join(__dirname, 'index.html'));
}

ipcMain.on('predict', (_event, { id, png }) => {
  if (!py) {
    send('py-line', { stream: 'stdout', text: `@@ERROR ${JSON.stringify({ id, message: 'Python süreci çalışmıyor' })}` });
    return;
  }
  py.stdin.write(`${JSON.stringify({ id, png })}\n`);
});

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
app.on('will-quit', () => {
  if (py) py.kill();
});

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('demo', {
  predict: (id, png) => ipcRenderer.send('predict', { id, png }),
  onLine: (cb) => ipcRenderer.on('py-line', (_event, msg) => cb(msg)),
  onExit: (cb) => ipcRenderer.on('py-exit', (_event, msg) => cb(msg)),
});

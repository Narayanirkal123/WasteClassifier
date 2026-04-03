const CACHE_VERSION = 'v1.0.1';
const CACHE_NAME = `wastescan-cache-${CACHE_VERSION}`;

const CORE_ASSETS = [
    './',
    './index.html',
    './style.css',
    './script.js',
    './static/icon.webp',
    './models/EfficientNetB2_fp16.onnx'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(
                keys
                    .filter((key) => key.startsWith('wastescan-cache-') && key !== CACHE_NAME)
                    .map((key) => caches.delete(key))
            )
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
        return;
    }

    if (CORE_ASSETS.some((asset) => url.pathname.endsWith(asset.replace('./', '')))) {
        event.respondWith(
            caches.match(event.request).then((cached) => {
                if (cached) return cached;
                return fetch(event.request).then((response) => {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseClone));
                    return response;
                });
            })
        );
        return;
    }

    event.respondWith(
        fetch(event.request).catch(() => caches.match(event.request))
    );
});

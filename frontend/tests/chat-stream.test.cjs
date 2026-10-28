const { test, after } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');

const out = mkdtempSync(path.join(tmpdir(), 'flygpt-stream-'));
execFileSync(process.execPath, [require.resolve('typescript/bin/tsc'),
  'lib/api.ts', '--outDir', out, '--module', 'commonjs', '--target', 'ES2020',
  '--skipLibCheck', '--strict']);
const { sendChatStream } = require(path.join(out, 'api.js'));
after(() => rmSync(out, { recursive: true, force: true }));
const encode = (value) => new TextEncoder().encode(value);
const event = (value) => encode('data: ' + JSON.stringify(value) + '\n\n');
const result = { type: 'result', response: { answer: '안녕하세요!', type: 'dispatch' } };

function fastDeadline(t) {
  const set = global.setTimeout;
  t.mock.method(global, 'setTimeout', (fn, ms, ...args) =>
    set(fn, ms >= 1000 ? 20 : ms, ...args));
}

test('final result resolves without waiting for EOF and cancels the body', async (t) => {
  let cancelled = false;
  t.mock.method(global, 'fetch', async () => new Response(new ReadableStream({
    start(c) { c.enqueue(event(result)); },
    cancel() { cancelled = true; },
  })));
  const events = [];
  assert.equal((await sendChatStream('안녕', 's', (e) => events.push(e))).answer, '안녕하세요!');
  assert.equal(events.length, 1);
  assert.ok(cancelled);
});

test('no response headers reaches deadline without an automatic second request', async (t) => {
  fastDeadline(t);
  const fetch = t.mock.method(global, 'fetch', (_, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  }));
  await assert.rejects(sendChatStream('안녕', 's', () => {}), /대기 시간이 초과/);
  assert.equal(fetch.mock.callCount(), 1);
});

test('heartbeats cannot extend the deadline; explicit retry makes one fresh request', async (t) => {
  fastDeadline(t);
  let call = 0;
  let heartbeat;
  t.mock.method(global, 'fetch', async (_, { signal }) => {
    call++;
    if (call === 2) return new Response(event(result));
    return new Response(new ReadableStream({
      start(c) {
        c.enqueue(event({ type: 'status', phase: 'processing', request_budget_ms: 1000 }));
        heartbeat = setInterval(() => c.enqueue(encode(': keep-alive\n\n')), 2);
        signal.addEventListener('abort', () => {
          clearInterval(heartbeat);
          c.error(signal.reason);
        }, { once: true });
      },
      cancel() { clearInterval(heartbeat); },
    }));
  });
  try {
    await assert.rejects(sendChatStream('안녕', 's', () => {}), /대기 시간이 초과/);
    assert.equal(call, 1);
    assert.equal((await sendChatStream('안녕', 's', () => {})).answer, '안녕하세요!');
    assert.equal(call, 2);
  } finally { clearInterval(heartbeat); }
});

test('incomplete stream and server errors reject rather than returning a partial answer', async (t) => {
  t.mock.method(global, 'fetch', async () => new Response(event({ type: 'status', phase: 'generation' })));
  await assert.rejects(sendChatStream('안녕', 's', () => {}), /최종 응답 없이/);
  global.fetch.mock.mockImplementation(async () => new Response(event({ type: 'error', message: '요청 실패' })));
  await assert.rejects(sendChatStream('안녕', 's', () => {}), /요청 실패/);
});

test('HTTP error does not leak a raw provider body', async (t) => {
  t.mock.method(global, 'fetch', async () => new Response('secret internal JSON', { status: 504 }));
  await assert.rejects(sendChatStream('안녕', 's', () => {}), (e) =>
    e.message.includes('504') && !e.message.includes('secret'));
});

"""Behavioral regressions for the optional Studio; no GPU or packages needed."""
from pathlib import Path
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

WEBUI = Path(__file__).resolve().parent
sys.path.insert(0, str(WEBUI))
import server as S
import ownership


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    connection.request(method, path, body, headers or {})
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


class StudioHTTP(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state = patch.object(ownership, 'STATE_PATH', Path(self.temporary.name) / 'state.json')
        self.state.start()
        self.server = S.StudioServer(('127.0.0.1', 0), S.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.state.stop()
        self.temporary.cleanup()

    def test_only_public_assets_are_served(self):
        fixture = Path(self.temporary.name) / 'private.txt'
        fixture.write_text('private fixture', encoding='utf-8')
        for path in ('/' + fixture.as_posix(), '/../private.txt', '/..\\private.txt',
                     '/studio.config.json', '/server.py', '/.state/launcher.json',
                     '/%2e%2e/private.txt'):
            with self.subTest(path=path):
                self.assertEqual(request(self.port, 'GET', path)[0], 404)
        for path in ('/', '/app.js', '/app.css', '/icons.svg', '/fonts/outfit-latin-wght.woff2'):
            with self.subTest(path=path):
                self.assertEqual(request(self.port, 'GET', path)[0], 200)

    def test_host_origin_and_browser_header_are_required(self):
        with patch.object(S, 'stop_engine') as stop:
            for headers in ({'Origin': 'https://foreign.example', 'X-Ninfer-Studio': '1'},
                            {'Host': 'foreign.example', 'X-Ninfer-Studio': '1'},
                            {'Content-Type': 'text/plain'}):
                self.assertEqual(request(self.port, 'POST', '/api/engine/stop', '', headers)[0], 403)
            stop.assert_not_called()
        self.assertEqual(request(self.port, 'GET', '/api/ping', headers={'Host': 'foreign.example'})[0], 403)

    def test_stop_cannot_claim_a_foreign_listener(self):
        with patch.object(S, 'engine_pid', return_value=[str(os.getpid())]), \
                patch.object(ownership, 'terminate') as terminate:
            status, body = request(self.port, 'POST', '/api/engine/stop',
                                   json.dumps({'only_pids': [str(os.getpid())]}),
                                   {'X-Ninfer-Studio': '1'})
            self.assertEqual(status, 409)
            self.assertEqual(json.loads(body)['killed'], [])
            terminate.assert_not_called()

    def test_recycled_pid_is_not_owned(self):
        ownership.save({'engine': {'owned': {str(os.getpid()): 'stale-birth'}}})
        with patch.object(ownership, 'terminate') as terminate:
            result = ownership.stop('engine', [str(os.getpid())])
            self.assertFalse(result['ok'])
            terminate.assert_not_called()

    def test_start_during_model_load_does_not_spawn_twice(self):
        ownership.record('engine', os.getpid())
        with patch.object(S, 'engine_pid', return_value=[]), patch.object(S.subprocess, 'Popen') as spawn:
            result = S.start_engine()
            self.assertTrue(result['already'])
            spawn.assert_not_called()

    def test_recycled_parent_cannot_claim_an_older_orphan(self):
        ownership.save({'engine': {'owned': {'10': '100'}}})
        births = {'10': '100', '20': '300', '30': '200'}
        with patch.object(ownership, 'identity', side_effect=lambda p: births.get(str(p))), \
                patch.object(ownership, 'parents', return_value={20: 10, 30: 20}):
            owned = ownership.claim('engine', ['30'])
            self.assertIn('20', owned)
            self.assertNotIn('30', owned)

    def test_stream_delivers_first_event_before_upstream_finishes(self):
        release = threading.Event()
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                self.wfile.flush()
                release.wait(4)
                self.wfile.write(b'data: [DONE]\n\n')
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        worker = threading.Thread(target=upstream.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1.5)
        try:
            with patch.object(S, 'UPSTREAM', 'http://127.0.0.1:%d' % upstream.server_port):
                connection.request('POST', '/api/chat', json.dumps({'model': 'fixture'}),
                                   {'Content-Type': 'application/json', 'X-Ninfer-Studio': '1'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(5), b'data:')
                self.assertFalse(release.is_set())
                release.set()
                self.assertIn(b'[DONE]', response.read())
        finally:
            release.set()
            connection.close()
            upstream.shutdown()
            upstream.server_close()
            worker.join()


DUMMY_ENGINE = '''import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body = json.dumps({"data": [{"id": "fixture"}]} if self.path == "/v1/models" else {"version": "fixture"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer(("127.0.0.1", int(os.environ["NINFER_PORT"])), Handler).serve_forever()
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class LauncherLifecycle(unittest.TestCase):
    def test_custom_config_owned_restart_and_foreign_engine(self):
        with tempfile.TemporaryDirectory(prefix='studio test ') as temporary:
            root = Path(temporary)
            shutil.copytree(WEBUI, root / 'webui', ignore=shutil.ignore_patterns('.state', '__pycache__'))
            dummy = root / 'dummy_engine.py'
            dummy.write_text(DUMMY_ENGINE, encoding='utf-8')
            engine_port, ui_port = free_port(), free_port()
            while ui_port == engine_port:
                ui_port = free_port()
            config = root / 'custom.json'
            config.write_text(json.dumps({
                'engine': {'port': engine_port, 'start_command': [sys.executable, str(dummy)]},
                'ui': {'port': ui_port},
            }), encoding='utf-8')
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('NINFER_', 'STUDIO_'))}
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0
            command = [sys.executable, str(root / 'webui' / 'launcher.py')]
            def launch(action):
                result = subprocess.run(command + [action, '--config', str(config), '--no-browser', '--timeout', '15'],
                                        cwd=WEBUI.parent, env=env, capture_output=True, text=True,
                                        timeout=40, creationflags=flags)
                return result
            foreign = None
            try:
                started = launch('start')
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                status, body = request(ui_port, 'GET', '/api/ping')
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)['engine'], 'http://127.0.0.1:%d' % engine_port)
                status, body = request(ui_port, 'POST', '/api/engine/stop', '', {'X-Ninfer-Studio': '1'})
                self.assertEqual(status, 200, body)
                time.sleep(0.3)
                status, body = request(ui_port, 'POST', '/api/engine/start', '', {'X-Ninfer-Studio': '1'})
                self.assertEqual(status, 200, body)
                self.assertTrue(json.loads(body)['ok'])
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        if request(engine_port, 'GET', '/health')[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.2)
                else:
                    self.fail('Restarted fixture engine did not become ready')
                stopped = launch('stop')
                self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
                foreign_env = dict(env, NINFER_PORT=str(engine_port))
                foreign = subprocess.Popen([sys.executable, str(dummy)], env=foreign_env,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           creationflags=flags)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        if request(engine_port, 'GET', '/health')[0] == 200:
                            break
                    except OSError:
                        pass
                    time.sleep(0.2)
                started = launch('start')
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                self.assertEqual(request(ui_port, 'POST', '/api/engine/stop', '', {'X-Ninfer-Studio': '1'})[0], 409)
                stopped = launch('stop')
                self.assertEqual(stopped.returncode, 1)
                self.assertIsNone(foreign.poll())
                self.assertEqual(request(engine_port, 'GET', '/health')[0], 200)
            finally:
                launch('stop')
                if foreign is not None:
                    foreign.terminate()
                    foreign.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()

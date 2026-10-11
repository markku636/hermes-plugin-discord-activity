import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from _load import discord_api as api


class FakeDiscord(BaseHTTPRequestHandler):
    script = []  # list of (status, body) consumed in order
    seen = []

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        FakeDiscord.seen.append((self.command, self.path, dict(self.headers), body))
        status, payload = FakeDiscord.script.pop(0) if FakeDiscord.script else (200, {"id": "m1"})
        data = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_POST = do_PATCH = do_DELETE = _handle

    def log_message(self, *a):
        pass


class DiscordApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeDiscord)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeDiscord.script = []
        FakeDiscord.seen = []
        self.sleeps = []
        self.client = api.DiscordClient("tok", api_base=self.base, sleep=self.sleeps.append)

    def test_create_payload(self):
        FakeDiscord.script = [(200, {"id": "123"})]
        self.assertEqual(self.client.create("c1", "hello", reply_to="u9"), "123")
        method, path, headers, body = FakeDiscord.seen[0]
        self.assertEqual((method, path), ("POST", "/channels/c1/messages"))
        self.assertEqual(headers["Authorization"], "Bot tok")
        self.assertEqual(body["allowed_mentions"], {"parse": [], "replied_user": False})
        self.assertEqual(body["message_reference"]["message_id"], "u9")
        self.assertTrue(body["enforce_nonce"])

    def test_429_retries_with_same_nonce(self):
        FakeDiscord.script = [(429, {"retry_after": 0.3}), (200, {"id": "m2"})]
        self.assertEqual(self.client.create("c1", "x"), "m2")
        self.assertEqual(self.sleeps, [0.3])
        nonces = [s[3]["nonce"] for s in FakeDiscord.seen]
        self.assertEqual(len(set(nonces)), 1)

    def test_5xx_then_gives_up(self):
        FakeDiscord.script = [(502, {}), (502, {}), (502, {})]
        with self.assertRaises(api.TransientError):
            self.client.edit("c1", "m1", "x")
        self.assertEqual(len(FakeDiscord.seen), 3)

    def test_403_is_permanent_without_retry(self):
        FakeDiscord.script = [(403, {"message": "Missing Permissions"})]
        with self.assertRaises(api.PermanentError):
            self.client.create("c1", "x")
        self.assertEqual(len(FakeDiscord.seen), 1)

    def test_delete_404_is_fine(self):
        FakeDiscord.script = [(404, {"message": "Unknown Message"})]
        self.client.delete("c1", "m1")

    def test_delete_204(self):
        FakeDiscord.script = [(204, None)]
        self.client.delete("c1", "m1")
        self.assertEqual(FakeDiscord.seen[0][0], "DELETE")


if __name__ == "__main__":
    unittest.main()

"""VLC Timestamp Injector + Snapshot Tool

Hotkeys:
    Shift+Enter:       Timestamp + send
    Ctrl+Shift+Enter:  Snapshot + timestamp + send
    Ctrl+Backslash:    Snapshot + attach to message (no send)

Uses Windows RegisterHotKey API — no global keyboard hook, no modifier
key interference.
"""
import ctypes
import ctypes.wintypes as wt
import threading
import requests
import time
import glob
import os
import logging
from io import BytesIO

import win32clipboard
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vlc_remote")

VLC_URL = "http://192.168.1.150:8080/requests/status.json"
VLC_PASS = "password"
SNAPSHOT_DIR = r"\\HTPC\vlcpic"

CLIPBOARD_RETRIES = 5
CLIPBOARD_RETRY_DELAY = 0.05

# Windows API constants
MOD_SHIFT = 0x0004
MOD_CTRL = 0x0002
MOD_NOREPEAT = 0x4000
VK_RETURN = 0x0D
VK_OEM_5 = 0xDC  # backslash key
WM_HOTKEY = 0x0312

# Hotkey IDs
HK_SHIFT_ENTER = 1
HK_CTRL_SHIFT_ENTER = 2
HK_CTRL_BACKSLASH = 3


# ---------------------------------------------------------------------------
# Clipboard helpers
# ---------------------------------------------------------------------------

def _open_clipboard(retries: int = CLIPBOARD_RETRIES) -> None:
    """Open the clipboard with retries (another app may hold the lock)."""
    for attempt in range(retries):
        try:
            win32clipboard.OpenClipboard()
            return
        except win32clipboard.error:
            if attempt == retries - 1:
                raise
            time.sleep(CLIPBOARD_RETRY_DELAY)


def set_clipboard_image(image_path: str) -> None:
    """Place an image on the clipboard as CF_DIB."""
    img = Image.open(image_path)
    buf = BytesIO()
    img.convert("RGB").save(buf, "BMP")
    bmp_data = buf.getvalue()[14:]  # strip BMP file header
    buf.close()
    _open_clipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, bmp_data)
    finally:
        win32clipboard.CloseClipboard()


# ---------------------------------------------------------------------------
# Key injection via SendInput (replaces keyboard library entirely)
# ---------------------------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_MENU = 0x12       # Alt
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_BACK = 0x08


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]
    _fields_ = [
        ("type", wt.DWORD),
        ("_input", _INPUT),
    ]


def _send_input(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    ctypes.windll.user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT))


def _key_down(vk):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp._input.ki.wVk = vk
    return inp


def _key_up(vk):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp._input.ki.wVk = vk
    inp._input.ki.dwFlags = KEYEVENTF_KEYUP
    return inp


def _unicode_down(char):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp._input.ki.wScan = ord(char)
    inp._input.ki.dwFlags = KEYEVENTF_UNICODE
    return inp


def _unicode_up(char):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp._input.ki.wScan = ord(char)
    inp._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
    return inp


def type_text(text: str) -> None:
    """Type a string via Unicode SendInput events (no keyboard hook needed)."""
    inputs = []
    for ch in text:
        inputs.append(_unicode_down(ch))
        inputs.append(_unicode_up(ch))
    _send_input(*inputs)


def send_key(vk: int) -> None:
    """Press and release a single virtual key."""
    _send_input(_key_down(vk), _key_up(vk))


def send_hotkey(modifier_vk: int, key_vk: int) -> None:
    """Press a modifier+key combination."""
    _send_input(
        _key_down(modifier_vk),
        _key_down(key_vk),
        _key_up(key_vk),
        _key_up(modifier_vk),
    )


# ---------------------------------------------------------------------------
# VLC interaction
# ---------------------------------------------------------------------------

def get_vlc_time() -> str | None:
    """Return the current VLC playback position as HH:MM:SS or MM:SS."""
    try:
        r = requests.get(VLC_URL, auth=("", VLC_PASS), timeout=1)
        data = r.json()
        seconds = int(data.get("time", 0))
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    except requests.RequestException as exc:
        log.error("VLC request failed: %s", exc)
        return None


def take_snapshot() -> str | None:
    """Tell VLC to take a snapshot, wait for the new file, return its path."""
    pattern = os.path.join(SNAPSHOT_DIR, "vlcsnap-*.png")
    before = set(glob.glob(pattern))

    try:
        requests.get(VLC_URL, params={"command": "snapshot"},
                     auth=("", VLC_PASS), timeout=2)
    except requests.RequestException as exc:
        log.error("Snapshot request failed: %s", exc)
        return None

    for _ in range(20):
        time.sleep(0.1)
        new_files = set(glob.glob(pattern)) - before
        if new_files:
            path = max(new_files, key=os.path.getmtime)
            log.info("Snapshot captured: %s", os.path.basename(path))
            return path

    log.warning("Timed out waiting for snapshot file")
    return None


# ---------------------------------------------------------------------------
# Hotkey handlers (run on worker threads to avoid blocking the message loop)
# ---------------------------------------------------------------------------

def on_shift_enter() -> None:
    """Type timestamp + send."""
    timestamp = get_vlc_time()
    if timestamp is None:
        return

    time.sleep(0.15)
    type_text(f" [{timestamp}]")
    time.sleep(0.05)
    send_key(VK_RETURN)
    log.info("Sent timestamp [%s]", timestamp)


def on_ctrl_shift_enter() -> None:
    """Snapshot + type timestamp + attach image + send."""
    timestamp = get_vlc_time()
    snap = take_snapshot()

    time.sleep(0.15)

    if timestamp:
        type_text(f" [{timestamp}]")
        time.sleep(0.05)

    if snap:
        set_clipboard_image(snap)
        send_hotkey(VK_MENU, ord('V'))  # Alt+V to attach
        time.sleep(1.0)

    send_key(VK_RETURN)
    log.info("Sent snapshot + timestamp [%s]", timestamp)


def on_ctrl_backslash() -> None:
    """Snapshot + attach image (no send)."""
    snap = take_snapshot()
    if not snap:
        return

    time.sleep(0.15)

    set_clipboard_image(snap)
    send_hotkey(VK_MENU, ord('V'))  # Alt+V to attach
    time.sleep(0.3)
    log.info("Attached snapshot (no send)")


# ---------------------------------------------------------------------------
# Hotkey dispatch via Windows message loop (replaces keyboard library)
# ---------------------------------------------------------------------------

HANDLERS = {
    HK_SHIFT_ENTER: on_shift_enter,
    HK_CTRL_SHIFT_ENTER: on_ctrl_shift_enter,
    HK_CTRL_BACKSLASH: on_ctrl_backslash,
}


def main():
    user32 = ctypes.windll.user32

    # Register hotkeys — RegisterHotKey suppresses them from reaching the app
    if not user32.RegisterHotKey(None, HK_SHIFT_ENTER, MOD_SHIFT | MOD_NOREPEAT, VK_RETURN):
        log.error("Failed to register Shift+Enter")
    if not user32.RegisterHotKey(None, HK_CTRL_SHIFT_ENTER, MOD_CTRL | MOD_SHIFT | MOD_NOREPEAT, VK_RETURN):
        log.error("Failed to register Ctrl+Shift+Enter")
    if not user32.RegisterHotKey(None, HK_CTRL_BACKSLASH, MOD_CTRL | MOD_NOREPEAT, VK_OEM_5):
        log.error("Failed to register Ctrl+Backslash")

    log.info("VLC Timestamp Injector running")
    log.info("  Shift+Enter:       Timestamp + send")
    log.info("  Ctrl+Shift+Enter:  Snapshot + timestamp + send")
    log.info("  Ctrl+Backslash:    Snapshot + attach (no send)")
    log.info("Ctrl+C to quit")

    PM_REMOVE = 0x0001
    msg = wt.MSG()
    try:
        while True:
            # Poll for messages, yielding to Python between checks so
            # KeyboardInterrupt (Ctrl+C in the console) can be raised.
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                if msg.message == WM_HOTKEY:
                    handler = HANDLERS.get(msg.wParam)
                    if handler:
                        threading.Thread(target=handler, daemon=True).start()
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        user32.UnregisterHotKey(None, HK_SHIFT_ENTER)
        user32.UnregisterHotKey(None, HK_CTRL_SHIFT_ENTER)
        user32.UnregisterHotKey(None, HK_CTRL_BACKSLASH)
        log.info("Hotkeys unregistered, exiting")


if __name__ == "__main__":
    main()

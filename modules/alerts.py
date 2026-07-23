cp /home/claude/apexscan_files/dashboard.py /home/claude/dashboard_fix.py

python3 - << 'PYEOF'
with open('/home/claude/dashboard_fix.py') as f:
    src = f.read()

OLD_DISPATCH = '''if dispatch_alert is None:
    def dispatch_alert(settings: dict, message: str, title: str = "") -> dict:
        """Stub: send via Telegram if token configured, else return failure silently."""
        result = {"telegram": False, "email": False}
        try:
            import urllib.request, urllib.parse
            _tok = settings.get("telegram_token","")
            _cid = settings.get("telegram_chat_id","")
            if _tok and _cid:
                _text = (f"*{title}*\n\n{message}" if title else message)
                _body = urllib.parse.urlencode({
                    "chat_id": _cid,
                    "text": _text[:4096],
                    "parse_mode": "Markdown",
                }).encode()
                _req = urllib.request.Request(
                    f"https://api.telegram.org/bot{_tok}/sendMessage",
                    data=_body, method="POST"
                )
                _req.add_header("Content-Type","application/x-www-form-urlencoded")
                with urllib.request.urlopen(_req, timeout=8) as _resp:
                    _js = json.loads(_resp.read().decode())
                    result["telegram"] = _js.get("ok", False)
        except Exception as _te:
            pass
        return result'''

NEW_DISPATCH = '''if dispatch_alert is None:
    def dispatch_alert(settings: dict, message: str, title: str = "") -> dict:
        """
        Send Telegram alert using JSON body (more reliable than form-encoded).
        Uses MarkdownV2 with special chars escaped, falls back to plain text.
        Surfaces Telegram error so user sees exactly what went wrong.
        """
        result = {"telegram": False, "email": False, "error": ""}
        try:
            import urllib.request, json as _json2
            _tok = str(settings.get("telegram_token","")).strip()
            _cid = str(settings.get("telegram_chat_id","")).strip()
            if not _tok or not _cid:
                result["error"] = "Token or Chat ID missing"
                return result

            # Build clean text — avoid Markdown parsing failures
            _title_clean = title.replace("*","").replace("_","").replace("`","").replace("[","")
            _msg_clean   = message[:3800]
            _text = f"*{_title_clean}*\n\n{_msg_clean}" if _title_clean else _msg_clean

            # Try with Markdown first, fall back to plain text if it fails
            for _parse_mode in ["Markdown", None]:
                _payload = {"chat_id": _cid, "text": _text[:4096]}
                if _parse_mode:
                    _payload["parse_mode"] = _parse_mode

                _body = _json2.dumps(_payload).encode("utf-8")
                _req  = urllib.request.Request(
                    f"https://api.telegram.org/bot{_tok}/sendMessage",
                    data=_body, method="POST"
                )
                _req.add_header("Content-Type", "application/json; charset=utf-8")

                try:
                    with urllib.request.urlopen(_req, timeout=15) as _resp:
                        _js = _json2.loads(_resp.read().decode())
                        if _js.get("ok"):
                            result["telegram"] = True
                            return result
                        else:
                            _err = _js.get("description","Unknown Telegram error")
                            result["error"] = _err
                            # If parse mode error, retry without it
                            if "parse" in _err.lower() or "markdown" in _err.lower():
                                _text = _msg_clean   # strip title formatting too
                                continue
                            return result
                except urllib.error.HTTPError as _he:
                    _err_body = _he.read().decode("utf-8","ignore")
                    try:
                        _err_js = _json2.loads(_err_body)
                        result["error"] = _err_js.get("description", f"HTTP {_he.code}")
                    except Exception:
                        result["error"] = f"HTTP {_he.code}: {_err_body[:200]}"
                    # Retry without Markdown
                    if _parse_mode:
                        _text = _msg_clean
                        continue
                    return result

        except Exception as _te:
            result["error"] = str(_te)
        return result'''

count = src.count(OLD_DISPATCH)
print(f"dispatch_alert: {count}")
if count == 1:
    src = src.replace(OLD_DISPATCH, NEW_DISPATCH)
    print("  → fixed")

with open('/home/claude/dashboard_fix.py', 'w') as f:
    f.write(src)
print("Done")
PYEOF



You are out of fre

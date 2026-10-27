"""
제공사 공통 Batch 인터페이스 (Claude / OpenAI / Gemini). DeepSeek은 Batch API가 없어 --workers 병렬 호출 사용.

  submit(model_name, requests, workdir) -> job dict  (batch_state.json에 그대로 저장)
      requests: [(custom_id, RequestSpec), ...]
  status(job) -> (ended: bool, label: str, counts: {"done", "total", "succeeded", "failed"})
  results(job) -> 반복자 (custom_id, "succeeded" | 실패 종류, {"thinking","text","meta"} 또는 오류 문자열)

공통 사항
  - 요청은 서로 독립적으로 처리됨 (호출마다 새 대화)
  - 결과는 순서가 보장되지 않으므로 항상 custom_id(또는 Gemini의 key)로 다시 짝지음
  - 할인: 세 제공사 모두 50%
  - 결과 보관: Claude 29일, OpenAI 출력 파일 30일, Gemini 6주 (이 기간 안에는 fetch_batch_responses.py로 다시 받을 수 있음)
"""
import json
import os

try:
    from src.llm_requests import parse_response_body
except ImportError:
    from llm_requests import parse_response_body


def _clients():
    try:
        import src.api_client as ac
    except ImportError:
        import api_client as ac
    return ac


# =========================================================================
# 제출
# =========================================================================
def submit(model_name, requests, workdir):
    providers = {spec.provider for _, spec in requests}
    if len(providers) != 1:
        raise ValueError(f"한 batch에는 한 종류의 요청만 넣을 수 있습니다: {providers}")
    provider = providers.pop()
    ac = _clients()
    base_meta = requests[0][1].meta

    if provider == "claude":
        b = ac.client_claude.messages.batches.create(
            requests=[{"custom_id": cid, "params": spec.body} for cid, spec in requests])
        return {"provider": "claude", "id": b.id, "base_meta": base_meta}

    if provider in ("openai_responses", "openai_chat"):
        endpoint = requests[0][1].endpoint
        path = os.path.join(workdir, f"openai_batch_input_{len(os.listdir(workdir))}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for cid, spec in requests:
                f.write(json.dumps({"custom_id": cid, "method": "POST", "url": endpoint, "body": spec.body},
                                   ensure_ascii=False) + "\n")
        up = ac.client_gpt.files.create(file=open(path, "rb"), purpose="batch")
        b = ac.client_gpt.batches.create(input_file_id=up.id, endpoint=endpoint, completion_window="24h")
        return {"provider": provider, "id": b.id, "input_file_id": up.id, "endpoint": endpoint,
                "base_meta": base_meta}

    if provider == "gemini":
        from google.genai import types
        path = os.path.join(workdir, f"gemini_batch_input_{len(os.listdir(workdir))}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for cid, spec in requests:
                f.write(json.dumps({"key": cid, "request": spec.body}, ensure_ascii=False) + "\n")
        up = ac.client_gemini_genai.files.upload(
            file=path, config=types.UploadFileConfig(display_name=os.path.basename(path), mime_type="jsonl"))
        job = ac.client_gemini_genai.batches.create(model=model_name, src=up.name,
                                                    config={"display_name": os.path.basename(path)})
        return {"provider": "gemini", "id": job.name, "input_file": up.name, "base_meta": base_meta}

    raise ValueError(f"Batch를 지원하지 않는 제공사입니다: {provider} (--workers 병렬 호출을 사용하세요)")


# =========================================================================
# 상태
# =========================================================================
def status(job):
    ac = _clients()
    p = job.get("provider", "claude")   # 이전 버전 상태 파일은 provider가 없음 = Claude

    if p == "claude":
        b = ac.client_claude.messages.batches.retrieve(job["id"])
        rc = b.request_counts
        c = {k: getattr(rc, k, 0) or 0 for k in ("processing", "succeeded", "errored", "canceled", "expired")}
        done = c["succeeded"] + c["errored"] + c["canceled"] + c["expired"]
        return (b.processing_status == "ended", b.processing_status,
                {"done": done, "total": done + c["processing"], "succeeded": c["succeeded"],
                 "failed": c["errored"] + c["canceled"] + c["expired"]})

    if p in ("openai_responses", "openai_chat"):
        b = ac.client_gpt.batches.retrieve(job["id"])
        rc = b.request_counts
        total, completed, failed = (getattr(rc, "total", 0) or 0, getattr(rc, "completed", 0) or 0,
                                    getattr(rc, "failed", 0) or 0)
        ended = b.status in ("completed", "failed", "expired", "cancelled")
        return ended, b.status, {"done": completed + failed, "total": total, "succeeded": completed, "failed": failed}

    if p == "gemini":
        j = ac.client_gemini_genai.batches.get(name=job["id"])
        state = j.state.name if hasattr(j.state, "name") else str(j.state)
        st = getattr(j, "batch_stats", None)
        total = getattr(st, "request_count", None) if st else None
        ok = getattr(st, "successful_request_count", None) if st else None
        bad = getattr(st, "failed_request_count", None) if st else None
        ended = state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED")
        done = (ok or 0) + (bad or 0)
        return ended, state, {"done": done, "total": total, "succeeded": ok, "failed": bad}

    raise ValueError(p)


# =========================================================================
# 결과
# =========================================================================
def results(job):
    ac = _clients()
    p = job.get("provider", "claude")
    base_meta = job.get("base_meta") or {}

    if p == "claude":
        try:
            from src.api_client import parse_claude_message
        except ImportError:
            from api_client import parse_claude_message
        for item in ac.client_claude.messages.batches.results(job["id"]):
            o = item.result
            if o.type == "succeeded":
                yield item.custom_id, "succeeded", parse_claude_message(o.message, base_meta)
            else:
                err = getattr(o, "error", None)
                yield item.custom_id, o.type, str(err) if err is not None else o.type
        return

    if p in ("openai_responses", "openai_chat"):
        b = ac.client_gpt.batches.retrieve(job["id"])
        seen = set()
        for fid, is_err in ((getattr(b, "output_file_id", None), False), (getattr(b, "error_file_id", None), True)):
            if not fid:
                continue
            content = ac.client_gpt.files.content(fid).text
            for line in content.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec.get("custom_id")
                resp = rec.get("response") or {}
                if not is_err and resp.get("status_code") == 200 and resp.get("body"):
                    seen.add(cid)
                    yield cid, "succeeded", parse_response_body(p, resp["body"], base_meta)
                elif cid not in seen:
                    yield cid, "errored", json.dumps(rec.get("error") or resp, ensure_ascii=False)[:500]
        return

    if p == "gemini":
        j = ac.client_gemini_genai.batches.get(name=job["id"])
        dest = getattr(j, "dest", None)
        fname = getattr(dest, "file_name", None) if dest else None
        if not fname:
            raise RuntimeError(f"Gemini batch 결과 파일이 없습니다 (state={getattr(j, 'state', None)})")
        raw = ac.client_gemini_genai.files.download(file=fname)
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec.get("key")
            if rec.get("response"):
                yield cid, "succeeded", parse_response_body("gemini", rec["response"], base_meta)
            else:
                yield cid, "errored", json.dumps(rec.get("error") or rec, ensure_ascii=False)[:500]
        return

    raise ValueError(p)


# =========================================================================
# 취소 (진행 중인 batch를 멈춤. 이미 끝난 요청의 결과는 남고 비용도 그만큼만 청구됨)
# =========================================================================
def cancel(job):
    ac = _clients()
    p = job.get("provider", "claude")
    if p == "claude":
        return ac.client_claude.messages.batches.cancel(job["id"]).processing_status
    if p in ("openai_responses", "openai_chat"):
        return ac.client_gpt.batches.cancel(job["id"]).status
    if p == "gemini":
        ac.client_gemini_genai.batches.cancel(name=job["id"])
        return "cancel requested"
    raise ValueError(p)

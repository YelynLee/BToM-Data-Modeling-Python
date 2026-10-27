"""
결과 폴더의 batch_state.json에 있는, 아직 결과를 가져오지 않은 batch를 취소함.

- 취소해도 이미 처리된 요청의 결과는 남고, 비용도 처리된 만큼만 청구됨
- 취소 후 같은 main_experiment.py 명령을 다시 실행하면 끝난 결과만 CSV에 반영하고,
  못 받은 요청은 새 batch로 다시 제출함 (원하지 않으면 그 폴더로 다시 실행하지 마세요)

사용 예:
  python -m src.cancel_batch --dir results/claude-opus-4-6/vanilla/singlestep
  python -m src.cancel_batch --dir ... --dry_run
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import batch_api  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    path = os.path.join(args.dir, "batch_state.json")
    if not os.path.exists(path):
        sys.exit(f"{path}가 없습니다.")
    state = json.load(open(path))
    pending = [b for b in state.get("batches", []) if not b.get("collected")]
    if not pending:
        print("결과를 가져오지 않은 batch가 없습니다.")
        return
    for b in pending:
        ended, label, c = batch_api.status(b)
        print(f"{b.get('provider', 'claude')} {b['id']}: {label} | 완료 {c.get('done')}/{c.get('total')}")
        if ended or args.dry_run:
            continue
        print(f"   -> 취소 요청: {batch_api.cancel(b)}")


if __name__ == "__main__":
    main()

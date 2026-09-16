import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig
from deepeval.metrics import ContextualPrecisionMetric, FaithfulnessMetric, GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# DeepEval Environment Overrides for Rate Limiting & Timeouts
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "6")
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "10")
os.environ.setdefault("DEEPEVAL_RETRY_EXP_BASE", "2")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "90")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "30")


class RateLimiter:
    """Sliding-window rate limiter for client-side API throttling."""
    def __init__(self, max_calls: int = 12, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self._lock = threading.Lock()
        self._alock = asyncio.Lock()

    def _wait_time(self) -> float:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.period:
            self.calls.popleft()
        return self.period - (now - self.calls[0]) if len(self.calls) >= self.max_calls else 0.0

    def acquire(self):
        with self._lock:
            while (wait := self._wait_time()) > 0:
                time.sleep(wait)
            self.calls.append(time.monotonic())

    async def a_acquire(self):
        async with self._alock:
            while (wait := self._wait_time()) > 0:
                await asyncio.sleep(wait)
            self.calls.append(time.monotonic())


class RateLimitedGeminiModel(GeminiModel):
    """GeminiModel with automated rate limiting before execution."""
    def __init__(self, *args, limiter: RateLimiter = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.limiter = limiter or RateLimiter()

    def generate(self, *args, **kwargs):
        self.limiter.acquire()
        return super().generate(*args, **kwargs)

    async def a_generate(self, *args, **kwargs):
        await self.limiter.a_acquire()
        return await super().a_generate(*args, **kwargs)


def run_eval(
    results_path: str | Path = "utils/reports/response.json",
    report_path: str | Path = "utils/reports/report.json",
    model_name: str = "gemini-3.1-flash-lite",
    rate_limit_per_min: int = 12,
) -> dict:
    """Runs Faithfulness, Contextual Precision, and Answer Correctness metrics

    over saved RAG results, displays summary statistics, and saves report.json.
    """
    results_path, report_path = Path(results_path), Path(report_path)
    limiter = RateLimiter(max_calls=rate_limit_per_min)
    judge_model = RateLimitedGeminiModel(
        model=model_name,
        api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0,
        limiter=limiter,
    )

    with open(results_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    test_cases = [
        LLMTestCase(
            input=r["question"],
            actual_output=r["response"]["answer"],
            retrieval_context=[r["response"]["context"]],
            expected_output=r.get("expected_answer"),
        )
        for r in records
    ]

    metrics = [
        FaithfulnessMetric(model=judge_model, threshold=0.5),
        ContextualPrecisionMetric(model=judge_model, threshold=0.5),
        GEval(
            name="Answer Correctness",
            criteria=(
                "Determine whether the 'actual output' is factually correct and "
                "semantically equivalent to the 'expected output'. Penalize missing "
                "or contradictory facts; do not penalize differences in phrasing."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=judge_model,
            threshold=0.5,
        ),
    ]

    async_cfg = AsyncConfig(max_concurrent=1, throttle_value=1)

    def execute_eval(use_cache: bool):
        return evaluate(
            test_cases=test_cases,
            metrics=metrics,
            async_config=async_cfg,
            cache_config=CacheConfig(use_cache=use_cache, write_cache=use_cache),
        )

    try:
        result = execute_eval(use_cache=True)
    except AttributeError as e:
        if any(k in str(e) for k in ("get_cached_api_test_case", "test_cases_lookup_map")):
            print("Stale DeepEval cache detected — clearing cache and retrying...")
            for f in (".deepeval-cache.json", ".temp-deepeval-cache.json"):
                Path(f).unlink(missing_ok=True)
            result = execute_eval(use_cache=False)
        else:
            raise

    # Aggregate metric scores
    scores_by_metric = defaultdict(list)
    for tr in result.test_results:
        for md in tr.metrics_data:
            scores_by_metric[md.name].append(md.score)

    summary = {
        name: {"average_score": round(mean(scores), 4) if scores else None, "n": len(scores)}
        for name, scores in scores_by_metric.items()
    }

    # Print summary table
    print(f"\n{'Metric':30s} | {'Avg Score':10s} | {'N':5s}\n" + "-" * 50)
    for name, stats in summary.items():
        avg_str = f"{stats['average_score']}" if stats["average_score"] is not None else "N/A"
        print(f"{name:30s} | {avg_str:<10} | {stats['n']:<5}")

    # Build and save report
    report = {
        "summary": summary,
        "per_case": [
            {
                "input": tc.input,
                "actual_output": tc.actual_output,
                "expected_source": r.get("expected_source"),
                "citations": r.get("response", {}).get("citations"),
                "metrics": {
                    md.name: {"score": md.score, "reason": md.reason, "success": md.success}
                    for md in tr.metrics_data
                },
            }
            for tc, tr, r in zip(test_cases, result.test_results, records)
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved detailed report to {report_path.resolve()}")
    return report["summary"]


if __name__ == "__main__":
    run_eval()
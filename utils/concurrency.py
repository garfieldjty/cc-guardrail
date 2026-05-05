import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def interruptible_map(fn, items, max_workers):
    """Like executor.map() but cancels pending futures on KeyboardInterrupt."""
    if not items:
        return []
    results = [None] * len(items)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(fn, item): i for i, item in enumerate(items)}
    try:
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
    except KeyboardInterrupt:
        logger.info("Interrupted — cancelling pending tasks")
        for f in futures:
            f.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results

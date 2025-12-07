import time

class TimingLogger:
    def __init__(self):
        self.timings = {}
        self.start_times = {}

    def start(self, key):
        """Start a timer for the given key."""
        self.start_times[key] = time.time()

    def stop(self, key):
        """Stop the timer for the given key and record the duration."""
        if key in self.start_times:
            duration = (time.time() - self.start_times[key]) * 1000  # Convert to ms
            self.timings[key] = duration
            del self.start_times[key]
            return duration
        return 0

    def add(self, key, duration_ms):
        """Directly add a duration for a key."""
        if key in self.timings:
            self.timings[key] += duration_ms
        else:
            self.timings[key] = duration_ms

    def get_total_time(self):
        return self.timings.get("total_inspection_time", 0)

    def print_report(self):
        """Print the formatted timing report."""
        if not self.timings:
            return

        total_time = self.get_total_time()
        if total_time == 0:
            # Fallback if total wasn't explicitly tracked
            total_time = sum(self.timings.values())

        print(f"Total Processing Time: {total_time:.2f} ms ({total_time/1000:.3f} s)")
        print("-" * 60)

        # Sort by duration descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for key, duration in sorted_items:
            # Skip total_inspection_time in the list if we want to avoid duplication, 
            # but the user's example includes it.
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{key:.<50} {duration:7.2f} ms ({percentage:5.1f}%)")
        print()

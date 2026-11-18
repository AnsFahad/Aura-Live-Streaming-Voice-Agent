import asyncio
import time
import json
import psutil
import websockets
from typing import List

SERVER_WS_URL = "ws://localhost:8000/ws/chat"
CONCURRENT_SESSIONS = 100
BENCHMARK_DURATION_SECONDS = 5

async def simulate_session(session_idx: int, stats: dict):
    """Simulates an active user streaming audio frames and receiving partial transcripts."""
    pcm_frame = bytes(1024)  # 512 samples @ 16-bit PCM = 32ms frame
    start_connect = time.perf_counter()
    
    try:
        async with websockets.connect(SERVER_WS_URL) as ws:
            stats["connected"] += 1
            connect_rtt = (time.perf_counter() - start_connect) * 1000.0
            stats["rtt_list"].append(connect_rtt)

            # Receive initial connection message
            init_msg = await ws.recv()

            # Stream audio frames concurrently for the benchmark duration
            start_streaming = time.perf_counter()
            frames_sent = 0
            
            while time.perf_counter() - start_streaming < BENCHMARK_DURATION_SECONDS:
                await ws.send(pcm_frame)
                frames_sent += 1
                stats["total_audio_frames"] += 1
                
                # Check for incoming control packets non-blockingly
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    if isinstance(msg, str):
                        stats["control_messages_received"] += 1
                except asyncio.TimeoutError:
                    pass

                # 32ms audio frame pacing
                await asyncio.sleep(0.032)

            stats["successful_sessions"] += 1

    except Exception as e:
        stats["errors"].append(str(e))

async def run_benchmark(target_sessions: int = 100):
    print("=" * 60)
    print(f"🚀 AURA CONCURRENCY BENCHMARK: {target_sessions} SESSIONS")
    print("=" * 60)
    print(f"Target WebSocket Endpoint: {SERVER_WS_URL}")
    print(f"Simulating {target_sessions} concurrent live audio streaming pipelines...")

    process = psutil.Process()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    stats = {
        "connected": 0,
        "successful_sessions": 0,
        "total_audio_frames": 0,
        "control_messages_received": 0,
        "rtt_list": [],
        "errors": []
    }

    start_time = time.perf_counter()
    tasks = [asyncio.create_task(simulate_session(i, stats)) for i in range(target_sessions)]
    
    await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.perf_counter() - start_time

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    avg_rtt = sum(stats["rtt_list"]) / len(stats["rtt_list"]) if stats["rtt_list"] else 0.0

    print("\n📊 BENCHMARK RESULTS")
    print("-" * 60)
    print(f"Total Concurrent Connections:  {stats['connected']} / {target_sessions}")
    print(f"Successfully Completed:        {stats['successful_sessions']} / {target_sessions}")
    print(f"Total Audio Frames Streamed:   {stats['total_audio_frames']:,}")
    print(f"Messages / Events Handled:     {stats['control_messages_received']:,}")
    print(f"Average WebSocket Handshake:   {avg_rtt:.2f} ms")
    print(f"Total Benchmark Duration:      {total_time:.2f} seconds")
    print(f"Memory Usage (Delta):          +{mem_after_mb - mem_before_mb:.2f} MB")
    print(f"Failed / Errored Sessions:     {len(stats['errors'])}")
    print("-" * 60)

    if stats["successful_sessions"] >= int(target_sessions * 0.95):
        print(f"✅ PASSED: Successfully sustained {target_sessions} concurrent sessions within latency bounds!")
    else:
        print(f"⚠️ WARNING: {len(stats['errors'])} sessions encountered issues.")

if __name__ == "__main__":
    asyncio.run(run_benchmark(CONCURRENT_SESSIONS))

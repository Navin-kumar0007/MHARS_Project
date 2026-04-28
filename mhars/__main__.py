import argparse
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser(description="MHARS (Multi-modal Hybrid Adaptive Response System) CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train all models sequentially")
    train_parser.add_argument("--quick", action="store_true", help="Run a quick 50K timestep training for PPO")
    
    # Demo command
    subparsers.add_parser("demo", help="Run the full pipeline demo")
    
    # Benchmark command
    subparsers.add_parser("benchmark", help="Run latency benchmarks for Edge vs Cloud")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Start the live dashboard visualization")

    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == "train":
        print("╔══════════════════════════════════════════════════════╗")
        print("║   MHARS Global Training Pipeline                     ║")
        print("╚══════════════════════════════════════════════════════╝")
        t0 = time.time()
        
        # Train Stage 2 (ML Models)
        from stage2_ml.run_stage2 import main as run_stage2
        run_stage2()
        
        # Train Stage 3 (PPO Agent)
        from stage3_ai.run_stage3 import main as run_stage3
        import sys
        
        # Temporarily manipulate sys.argv for the --quick flag in run_stage3
        old_argv = sys.argv
        sys.argv = ['run_stage3.py']
        if args.quick:
            sys.argv.append('--quick')
        run_stage3()
        sys.argv = old_argv
        
        print(f"\n[DONE] Global training finished in {time.time() - t0:.1f}s.")
        
    elif args.command == "demo":
        from demo import main as run_demo
        run_demo()
        
    elif args.command == "benchmark":
        from benchmarks.run_latency_test import BenchmarkRunner
        # Note: Depending on latency benchmark setup, you may need a wrapper or 
        # to import the class from latency_test if renamed. 
        # But we can just use subprocess if we want to isolate.
        import subprocess
        subprocess.run([sys.executable, "benchmarks/run_latency_test.py"])
        
    elif args.command == "dashboard":
        from mhars.dashboard import main as run_dashboard
        run_dashboard()
        
    else:
        print("Please specify a command. Use --help to see available commands.")
        sys.exit(1)

if __name__ == "__main__":
    main()

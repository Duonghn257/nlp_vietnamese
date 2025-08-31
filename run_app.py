#!/usr/bin/env python3
"""
Script to run the Vietnamese Poem Generator Streamlit app
"""

import subprocess
import sys
import os


def main():
    """Run the Streamlit app"""
    try:
        # Check if streamlit is installed
        subprocess.run(
            [sys.executable, "-m", "streamlit", "--version"],
            check=True,
            capture_output=True,
        )

        # Run the app
        print("🚀 Starting Vietnamese Poem Generator Chat App...")
        print("📱 The app will open in your default browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("\n" + "=" * 50)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
            ]
        )

    except subprocess.CalledProcessError:
        print("❌ Streamlit is not installed. Please install it first:")
        print("pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

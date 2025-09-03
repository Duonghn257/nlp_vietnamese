#!/usr/bin/env python3
"""
Test script for streaming Vietnamese poem generation
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import VietnamesePoem
import time


def test_streaming_generation():
    """Test the streaming generation functionality"""

    print("🌸 Testing Vietnamese Poem Streaming Generation")
    print("=" * 50)

    # Initialize the model
    print("Loading model...")
    try:
        poem_generator = VietnamesePoem(config_path="config.yaml", device="cpu")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test prompts
    test_prompts = [
        "thơ lục bát: về tình yêu",
        "thơ bảy chữ: về thiên nhiên",
        "thơ bốn chữ: về cuộc sống",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("-" * 30)

        # Test streaming generation
        print("🔄 Streaming generation:")
        try:
            full_response = ""
            start_time = time.time()

            for chunk in poem_generator.streaming_generate_poem(
                prompt=prompt, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
                time.sleep(0.1)  # Simulate real-time display

            end_time = time.time()
            print(f"\n\n⏱️  Streaming took: {end_time - start_time:.2f} seconds")
            print(f"📊 Total characters: {len(full_response)}")

        except Exception as e:
            print(f"❌ Error during streaming: {e}")

        # Test regular generation for comparison
        print("\n📄 Regular generation:")
        try:
            start_time = time.time()
            regular_response = poem_generator.generate_poem(
                prompt=prompt, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9
            )
            end_time = time.time()

            print(regular_response)
            print(f"\n⏱️  Regular generation took: {end_time - start_time:.2f} seconds")
            print(f"📊 Total characters: {len(regular_response)}")

        except Exception as e:
            print(f"❌ Error during regular generation: {e}")

        print("\n" + "=" * 50)


def test_api_streaming():
    """Test the API streaming endpoint"""

    print("\n🌐 Testing API Streaming Endpoint")
    print("=" * 50)

    import requests
    import json

    # Test the streaming API
    url = "http://localhost:8000/generate/stream"

    test_data = {
        "prompt": "thơ lục bát: Ai ơi bưng bát ",
        "max_new_tokens": 150,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
    }

    print(f"📡 Making request to: {url}")
    print(f"📝 Prompt: {test_data['prompt']}")

    try:
        response = requests.post(url, json=test_data, stream=True)

        if response.status_code == 200:
            print("✅ API request successful!")
            print("🔄 Receiving streaming response:")
            print("-" * 30)

            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        try:
                            data = json.loads(line_str[6:])
                            if data["type"] == "content":
                                chunk = data["chunk"]
                                print(chunk, end="", flush=True)
                                full_response += chunk
                            elif data["type"] == "done":
                                print("\n✅ Streaming completed!")
                                break
                            elif data["type"] == "error":
                                print(f"\n❌ Error: {data['chunk']}")
                                break
                        except json.JSONDecodeError:
                            continue

            print(f"\n📊 Total response length: {len(full_response)} characters")

        else:
            print(f"❌ API request failed with status: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Could not connect to API server. Make sure the server is running on localhost:8000"
        )
    except Exception as e:
        print(f"❌ Error testing API: {e}")


if __name__ == "__main__":
    print("🚀 Vietnamese Poem Streaming Test Suite")
    print("=" * 60)

    # Test local streaming
    test_streaming_generation()

    # Test API streaming
    test_api_streaming()

    print("\n🎉 Test suite completed!")

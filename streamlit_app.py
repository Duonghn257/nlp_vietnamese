import streamlit as st
import json
from datetime import datetime
from src import VietnamesePoem
import os
import time

# Page configuration
st.set_page_config(
    page_title="Vietnamese Poem Generator Chat", page_icon="🌸", layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "poem_generator" not in st.session_state:
    st.session_state.poem_generator = None

# Title and description
st.title("🌸 Vietnamese Poem Generator Chat")
st.markdown("Chat with your Vietnamese poem generator and create beautiful poetry!")

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Generation Parameters")

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Maximum number of tokens to generate",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Controls randomness in generation (higher = more random)",
    )

    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Number of highest probability tokens to consider",
    )

    top_p = st.slider(
        "Top-P",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.1,
        help="Nucleus sampling parameter",
    )

    st.divider()

    # Poem type selection
    poem_type = st.selectbox(
        "Poem Type",
        options=[
            "thơ bốn chữ",
            "thơ bảy chữ",
            "thơ lục bát",
            "thơ tám chữ",
            "thơ năm chữ",
        ],
        index=2,  # Default to "thơ lục bát"
        help="Select the type of Vietnamese poem to generate",
    )

    st.divider()

    # Device selection
    device = st.selectbox(
        "Device",
        options=["cpu", "cuda"],
        index=0,
        help="Select the device to run the model on",
    )

    # Streaming option
    use_streaming = st.checkbox(
        "Enable Streaming",
        value=True,
        help="Enable real-time streaming generation (word by word)",
    )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize model
if st.session_state.poem_generator is None:
    with st.spinner("Loading model..."):
        try:
            st.session_state.poem_generator = VietnamesePoem(
                config_path="config.yaml", device=device
            )
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"**{message.get('poem_type', '')}**: {message['content']}")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your prompt here..."):
    # Combine poem type with user prompt
    full_prompt = f"{poem_type}: {prompt}"

    # Add user message (show original prompt, not the full prompt)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat(),
            "poem_type": poem_type,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(f"**{poem_type}**: {prompt}")

    # Generate response
    with st.chat_message("assistant"):
        if use_streaming:
            # Streaming generation
            message_placeholder = st.empty()
            full_response = ""

            try:
                for (
                    text_chunk
                ) in st.session_state.poem_generator.streaming_generate_poem(
                    prompt=full_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ):
                    full_response += text_chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)  # Small delay for better visual effect

                # Final update without cursor
                message_placeholder.markdown(full_response)

                # Add assistant message to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now().isoformat(),
                        "parameters": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "top_k": top_k,
                            "top_p": top_p,
                            "poem_type": poem_type,
                            "streaming": True,
                        },
                    }
                )

            except Exception as e:
                st.error(f"❌ Error during streaming generation: {str(e)}")
        else:
            # Non-streaming generation
            with st.spinner("Generating poem..."):
                try:
                    response = st.session_state.poem_generator.generate_poem(
                        prompt=full_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

                    st.markdown(response)

                    # Add assistant message to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now().isoformat(),
                            "parameters": {
                                "max_new_tokens": max_new_tokens,
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p,
                                "poem_type": poem_type,
                                "streaming": False,
                            },
                        }
                    )

                except Exception as e:
                    st.error(f"❌ Error generating poem: {str(e)}")

# Display chat statistics in sidebar
if st.session_state.messages:
    st.sidebar.divider()
    st.sidebar.header("📊 Chat Statistics")
    st.sidebar.metric("Total Messages", len(st.session_state.messages))

    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    assistant_messages = len(
        [m for m in st.session_state.messages if m["role"] == "assistant"]
    )

    st.sidebar.metric("User Messages", user_messages)
    st.sidebar.metric("AI Responses", assistant_messages)

    if assistant_messages > 0:
        streaming_count = len(
            [
                m
                for m in st.session_state.messages
                if m["role"] == "assistant"
                and m.get("parameters", {}).get("streaming", False)
            ]
        )
        st.sidebar.metric("Streaming Responses", streaming_count)

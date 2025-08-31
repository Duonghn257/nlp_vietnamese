import streamlit as st
import json
from datetime import datetime
from src import VietnamesePoem
import os

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
        options=["cpu", "cuda", "mps"],
        index=0,
        help="Select the device for model inference",
    )

    # Model initialization
    if st.button("🚀 Initialize Model"):
        with st.spinner("Loading model..."):
            try:
                st.session_state.poem_generator = VietnamesePoem(
                    config_path="config.yaml", device=device
                )
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

    st.divider()

    # Conversation management
    st.header("💾 Conversation")

    if st.button("💾 Save Conversation"):
        if st.session_state.messages:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)

            st.success(f"Conversation saved to {filename}")
        else:
            st.warning("No conversation to save")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if st.session_state.poem_generator is None:
    st.info("👈 Please initialize the model in the sidebar first!")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "parameters" in message:
                poem_type_info = (
                    f", type={message['parameters']['poem_type']}"
                    if "poem_type" in message["parameters"]
                    else ""
                )
                st.caption(
                    f"Parameters: tokens={message['parameters']['max_new_tokens']}, temp={message['parameters']['temperature']}{poem_type_info}"
                )

    # Chat input
    if prompt := st.chat_input("Enter your prompt here..."):
        # Combine poem type with user prompt
        full_prompt = f"{poem_type}: {prompt}"

        # Add user message (show original prompt, not the full prompt)
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()}
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(f"**{poem_type}**: {prompt}")

        # Generate response
        with st.chat_message("assistant"):
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
                            },
                        }
                    )

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

# Footer
st.divider()
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Vietnamese Poem Generator Chat App</p>
        <p>Built with Streamlit and Vietnamese NLP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

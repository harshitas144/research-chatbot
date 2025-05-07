#!/usr/bin/env python3
"""
Debug helper script for testing Gemini API connectivity.
Run this script to verify your Gemini API key and connection.
"""

import os
import sys

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Google Generative AI package is installed")
except ImportError:
    GEMINI_AVAILABLE = False
    print("❌ Google Generative AI package is NOT installed")
    print("   Run: pip install google-generativeai")
    sys.exit(1)

# Check for API key
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("❌ GEMINI_API_KEY environment variable is not set")
    print("   Set it with: export GEMINI_API_KEY='your-api-key'")
    sys.exit(1)
else:
    print(f"✅ GEMINI_API_KEY is set (length: {len(api_key)})")

# Try to configure the API
try:
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured successfully")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    sys.exit(1)

# Try a test query
try:
    print("Sending test query to Gemini API...")
    model = genai.GenerativeModel(model_name="gemini-pro")
    response = model.generate_content("What is the capital of France?")
    
    if response and hasattr(response, 'text'):
        print(f"✅ Received response: {response.text[:50]}...")
        print("\nAPI connection is working correctly!")
    else:
        print("❌ Received empty or invalid response")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error querying Gemini API: {e}")
    sys.exit(1)
"""
Test script for OpenAI embedding functionality.
Tests the proxy_call.embedding() method with various inputs.
"""

import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Load environment variables from project root
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

from model.utils.proxy_call import OpenaiCall
from openai import OpenAI


def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_environment_setup():
    """Test 1: Check environment configuration."""
    print_section("Test 1: Environment Configuration")
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    print(f"📁 .env file location: {env_path}")
    print(f"📁 .env file exists: {'✅' if env_path.exists() else '❌'}")
    print(f"\n🔑 OPENAI_API_KEY: {'✅ Set (' + api_key[:20] + '...)' if api_key else '❌ Not set'}")
    print(f"🌐 OPENAI_BASE_URL: {base_url if base_url else '❌ Not set'}")
    
    if not api_key or not base_url:
        print("\n❌ Configuration incomplete! Please check your .env file.")
        return False
    
    print("\n✅ Environment configuration OK")
    return True


def test_direct_openai_client():
    """Test 2: Test direct OpenAI client."""
    print_section("Test 2: Direct OpenAI Client")
    
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        
        print("📝 Input: 'Your text string goes here'")
        print("🔄 Creating embedding...")
        
        response = client.embeddings.create(
            input="Your text string goes here",
            model="text-embedding-3-small"
        )
        
        embedding = response.data[0].embedding
        print(f"\n✅ Success!")
        print(f"📊 Embedding dimension: {len(embedding)}")
        print(f"📦 Model: {response.model}")
        print(f"💰 Tokens used: {response.usage.total_tokens}")
        print(f"🔢 First 10 values: {embedding[:10]}")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_proxy_call_single_text():
    """Test 3: Test OpenaiCall.embedding() with single text."""
    print_section("Test 3: OpenaiCall - Single Text")
    
    try:
        proxy = OpenaiCall()
        
        text = "外滩美术馆位于虎丘路，展示着丰富的艺术作品。"
        print(f"📝 Input: {text}")
        print("🔄 Creating embedding via proxy...")
        
        response = proxy.embedding(input_data=[text])
        
        embedding = response.data[0].embedding
        print(f"\n✅ Success!")
        print(f"📊 Embedding dimension: {len(embedding)}")
        print(f"📦 Model: {response.model}")
        print(f"💰 Tokens used: {response.usage.total_tokens}")
        print(f"🔢 First 5 values: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_proxy_call_multiple_texts():
    """Test 4: Test OpenaiCall.embedding() with multiple texts."""
    print_section("Test 4: OpenaiCall - Multiple Texts")
    
    try:
        proxy = OpenaiCall()
        
        texts = [
            "外滩美术馆位于虎丘路，展示着丰富的艺术作品。",
            "上海大剧院位于人民大道，是举办各类精彩演出的场所。",
            "思南公馆位于复兴中路，环境优美，适合休闲散步。"
        ]
        
        print(f"📝 Number of texts: {len(texts)}")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text[:40]}...")
        
        print("\n🔄 Creating embeddings via proxy...")
        
        response = proxy.embedding(input_data=texts)
        
        print(f"\n✅ Success!")
        print(f"📊 Number of embeddings: {len(response.data)}")
        print(f"📏 Embedding dimension: {len(response.data[0].embedding)}")
        print(f"📦 Model: {response.model}")
        print(f"💰 Tokens used: {response.usage.total_tokens}")
        
        # Convert to numpy array (as done in POI_construct)
        embeddings = [np.array(record.embedding) for record in response.data]
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        print(f"\n🔢 NumPy array shape: {embeddings_array.shape}")
        print(f"🔢 Data type: {embeddings_array.dtype}")
        
        # Calculate cosine similarity
        from numpy.linalg import norm
        sim = np.dot(embeddings_array[0], embeddings_array[1]) / (norm(embeddings_array[0]) * norm(embeddings_array[1]))
        print(f"📐 Cosine similarity (text 1 vs 2): {sim:.4f}")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_poi_format():
    """Test 5: Test with POI-formatted text (as used in POI_construct)."""
    print_section("Test 5: POI Format Text")
    
    try:
        proxy = OpenaiCall()
        
        texts = [
            '外滩美术馆: Integration of POI data via the Amap API. Address: 虎丘路20号(南京东路地铁站6号口步行430米), Coordinates: (31.241209, 121.487612), Category: 科教文化服务;美术馆;美术馆, Rating: N/A. Details: 外滩美术馆位于虎丘路，展示着丰富的艺术作品，是欣赏艺术和文化的好去处。',
            '上海大剧院: Integration of POI data via the Amap API. Address: 人民大道300号, Coordinates: (31.229400, 121.471825), Category: 体育休闲服务;影剧院;剧场, Rating: N/A. Details: 上海大剧院位于人民大道，是举办各类精彩演出的场所，体验高品质文艺表演的绝佳选择。'
        ]
        
        print(f"📝 Number of POI texts: {len(texts)}")
        print(f"📏 Text length: ~{len(texts[0])} chars")
        
        print("\n🔄 Creating embeddings...")
        
        response = proxy.embedding(input_data=texts)
        
        # Simulate POI_construct pipeline
        embeddings = [np.array(record.embedding) for record in response.data]
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        print(f"\n✅ Success!")
        print(f"📊 Embeddings created: {len(embeddings_array)}")
        print(f"🔢 Array shape: {embeddings_array.shape}")
        print(f"💰 Tokens used: {response.usage.total_tokens}")
        print(f"✅ Ready for POI database storage!")
        
        return True
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("  EMBEDDING FUNCTIONALITY TEST SUITE")
    print("🧪" * 40)
    
    results = []
    
    # Run tests in order
    results.append(("Environment Setup", test_environment_setup()))
    
    if not results[0][1]:
        print("\n❌ Skipping remaining tests due to configuration issues.")
        return 1
    
    results.append(("Direct OpenAI Client", test_direct_openai_client()))
    results.append(("OpenaiCall - Single Text", test_proxy_call_single_text()))
    results.append(("OpenaiCall - Multiple Texts", test_proxy_call_multiple_texts()))
    results.append(("POI Format Text", test_poi_format()))
    
    # Print summary
    print_section("TEST SUMMARY")
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\n📊 Result: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

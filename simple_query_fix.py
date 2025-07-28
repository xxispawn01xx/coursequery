#!/usr/bin/env python3
"""
Simple query fix to handle embedding errors gracefully.
This provides a fallback response when local models fail.
"""

def create_fallback_response(query: str, course_name: str = "") -> dict:
    """Create a helpful fallback response when models fail."""
    
    # Check if this looks like a real estate or finance question
    real_estate_keywords = [
        'property', 'real estate', 'valuation', 'appraisal', 'mortgage', 
        'market', 'investment', 'rental', 'cash flow', 'cap rate',
        'NOI', 'ROI', 'financing', 'leverage', 'equity'
    ]
    
    query_lower = query.lower()
    is_real_estate_query = any(keyword in query_lower for keyword in real_estate_keywords)
    
    if is_real_estate_query:
        response = f"""I understand you're asking about: "{query}"

**Current Status**: Local AI models are not fully loaded. For the best responses to your real estate questions, I recommend:

**Option 1: Cloud APIs (Recommended)**
- Upload your course materials to Google Drive
- Use ChatGPT Plus ($20/month) or Perplexity Pro ($20/month)
- Get superior responses with current market data

**Option 2: Local Setup**
- Ensure all dependencies are installed
- Load Llama 2 and Mistral models locally
- Process your question offline

**For your specific question about {course_name}:**
This appears to be a {course_name} related question. The course materials would help provide context-specific answers about real estate principles, valuation methods, or investment strategies.

Would you like me to help you set up cloud API access for better responses?"""
    
    else:
        response = f"""I see you're asking: "{query}"

**Technical Issue**: The local embedding model is not loaded properly, which is needed to search through your course materials.

**Quick Solutions**:
1. **Use Cloud APIs**: Configure OpenAI or Perplexity API keys for immediate access
2. **Fix Local Setup**: Ensure all AI dependencies are properly installed
3. **Upload to Cloud**: Put materials in Google Drive and query via ChatGPT

**About your question**: To provide an accurate answer about {course_name}, I need access to either:
- Local AI models (currently having technical issues)
- Cloud API keys for online processing
- Direct access to your course materials

Which option would you prefer to pursue?"""
    
    return {
        'answer': response,
        'sources': [],
        'method': 'fallback',
        'cached': False,
        'error': None
    }

if __name__ == "__main__":
    # Test the fallback
    test_query = "What are the key principles of real estate valuation?"
    result = create_fallback_response(test_query, "Real Estate Fundamentals")
    print("Fallback Response:")
    print(result['answer'])
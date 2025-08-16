"""
Course Vector Index Sharing System
Export processed course content for sharing with others
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil

from course_indexer import CourseIndexer
from config import Config

logger = logging.getLogger(__name__)

class CourseVectorExporter:
    """Export course vector indexes for sharing with others."""
    
    def __init__(self):
        self.config = Config()
        self.course_indexer = CourseIndexer()
        
    def export_course_for_sharing(self, course_name: str, output_path: str = "shared_courses") -> Dict[str, Any]:
        """
        Export a complete course index that can be shared and used by others.
        
        This exports:
        - Vector embeddings
        - Original text chunks  
        - Metadata
        - Embedding model info
        - Simple query interface
        
        Args:
            course_name: Name of the course to export
            output_path: Directory to save the exported course
            
        Returns:
            Export information and instructions
        """
        logger.info(f"🎁 Exporting course for sharing: {course_name}")
        
        try:
            # Check if course exists
            course_index_path = self.config.indexed_courses_dir / course_name
            if not course_index_path.exists():
                raise ValueError(f"Course '{course_name}' not found in indexed courses")
            
            # Create export directory
            export_dir = Path(output_path)
            export_dir.mkdir(exist_ok=True)
            
            course_export_dir = export_dir / f"{course_name}_shared"
            if course_export_dir.exists():
                shutil.rmtree(course_export_dir)
            course_export_dir.mkdir()
            
            # 1. Copy the entire LlamaIndex storage
            logger.info(" Copying vector index and text data...")
            index_source = course_index_path / "index"
            index_dest = course_export_dir / "index"
            
            if index_source.exists():
                shutil.copytree(index_source, index_dest)
            else:
                raise ValueError(f"No index found for course '{course_name}'")
            
            # 2. Copy course metadata
            metadata_source = course_index_path / "metadata.json"
            metadata_dest = course_export_dir / "metadata.json"
            
            if metadata_source.exists():
                shutil.copy2(metadata_source, metadata_dest)
                
                # Load metadata for export info
                with open(metadata_source) as f:
                    course_metadata = json.load(f)
            else:
                course_metadata = {"course_name": course_name}
            
            # 3. Create embedding model configuration
            embedding_config = {
                "model_name": self.config.model_config.get('embeddings', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                "model_type": "huggingface",
                "requires_download": True,
                "instructions": "Your friend needs to install: pip install sentence-transformers"
            }
            
            config_path = course_export_dir / "embedding_config.json"  
            with open(config_path, 'w') as f:
                json.dump(embedding_config, f, indent=2)
            
            # 4. Create a simple query interface for the friend
            self._create_simple_query_interface(course_export_dir, course_name)
            
            # 5. Create README with setup instructions
            self._create_setup_instructions(course_export_dir, course_name, course_metadata)
            
            # 6. Create a single ZIP file for easy sharing
            zip_path = export_dir / f"{course_name}_shared.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in course_export_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(course_export_dir)
                        zipf.write(file_path, arcname)
            
            # Clean up temporary directory
            shutil.rmtree(course_export_dir)
            
            export_info = {
                "course_name": course_name,
                "export_date": datetime.now().isoformat(),
                "zip_file": str(zip_path),
                "file_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
                "course_metadata": course_metadata,
                "embedding_model": embedding_config["model_name"],
                "sharing_instructions": {
                    "file_to_send": str(zip_path),
                    "recipient_setup": "Unzip and run: python query_course.py",
                    "requirements": "Python + sentence-transformers + llama-index"
                }
            }
            
            logger.info(f" Course exported successfully")
            logger.info(f" ZIP file: {zip_path}")
            logger.info(f" Size: {export_info['file_size_mb']} MB")
            
            return export_info
            
        except Exception as e:
            logger.error(f"Failed to export course: {e}")
            return {"error": str(e)}
    
    def _create_simple_query_interface(self, export_dir: Path, course_name: str):
        """Create a simple query script for the recipient."""
        
        query_script = f'''"""
Simple Query Interface for {course_name}
Your friend shared this course content with you!
"""

import os
import json
from pathlib import Path
from typing import List, Dict

try:
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings
    print(" LlamaIndex available")
except ImportError:
    print(" Please install: pip install llama-index sentence-transformers")
    exit(1)

class SharedCourseQuery:
    """Query interface for shared course content."""
    
    def __init__(self):
        self.course_name = "{course_name}"
        self.index = None
        self.load_course()
    
    def load_course(self):
        """Load the shared course index."""
        try:
            # Load embedding configuration
            with open("embedding_config.json") as f:
                config = json.load(f)
            
            # Set up the same embedding model used for indexing
            embed_model = HuggingFaceEmbedding(
                model_name=config["model_name"]
            )
            Settings.embed_model = embed_model
            
            # Load the vector index
            storage_context = StorageContext.from_defaults(persist_dir="index")
            self.index = load_index_from_storage(storage_context)
            
            print(f" Loaded course: {{self.course_name}}")
            print(" You can now ask questions about this course content!")
            
        except Exception as e:
            print(f" Error loading course: {{e}}")
            print("Make sure you have the required dependencies installed")
    
    def query(self, question: str, num_results: int = 3) -> str:
        """Query the course content."""
        if not self.index:
            return "Course not loaded. Please check the installation."
        
        try:
            # Query the index
            query_engine = self.index.as_query_engine(
                similarity_top_k=num_results,
                response_mode="compact"
            )
            
            response = query_engine.query(question)
            return str(response)
            
        except Exception as e:
            return f"Query failed: {{e}}"
    
    def interactive_mode(self):
        """Start interactive question-answer session."""
        print(f"\\n Interactive Query Mode for {{self.course_name}}")
        print("Ask questions about the course content (type 'quit' to exit)\\n")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(" Searching course content...")
            answer = self.query(question)
            print(f"\\n Answer:\\n{{answer}}\\n")
            print("-" * 50)

def main():
    """Main function to run the query interface."""
    course_query = SharedCourseQuery()
    
    if len(os.sys.argv) > 1:
        # Command line question
        question = " ".join(os.sys.argv[1:])
        answer = course_query.query(question)
        print(f"Question: {{question}}")
        print(f"Answer: {{answer}}")
    else:
        # Interactive mode
        course_query.interactive_mode()

if __name__ == "__main__":
    main()
'''
        
        script_path = export_dir / "query_course.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(query_script)
    
    def _create_setup_instructions(self, export_dir: Path, course_name: str, metadata: Dict):
        """Create setup instructions for the recipient."""
        
        instructions = f"""# {course_name} - Shared Course Content

Your friend has shared processed course content with you! This includes:
- All lecture transcriptions and content
- Vector embeddings for semantic search
- AI-powered question answering

## What's Included

- **Course**: {course_name}
- **Documents**: {metadata.get('document_count', 'Unknown')} files processed
- **Content Types**: {', '.join(metadata.get('document_types', {}).keys()) if metadata.get('document_types') else 'Mixed content'}
- **Total Content**: {metadata.get('total_content_length', 0):,} characters
- **Processed**: {metadata.get('last_indexed', 'Unknown date')}

## Quick Setup (2 minutes)

### 1. Install Dependencies
```bash
pip install llama-index sentence-transformers
```

### 2. Run Query Interface
```bash
# Interactive mode
python query_course.py

# Single question
python query_course.py "What is the main topic of this course?"
```

## Example Questions You Can Ask

- "What are the key concepts covered?"
- "Explain [specific topic] from the lectures"
- "What assignments or projects were mentioned?"
- "Summarize the main points of lecture 5"
- "What programming languages or tools are used?"

## How This Works

1. **Vector Embeddings**: The course content has been converted to numerical representations that capture meaning
2. **Semantic Search**: Your questions are matched to relevant course content using AI
3. **Same Model**: Uses the same embedding model as the original indexing for compatibility
4. **Offline**: Runs completely on your computer, no internet required after setup

## Alternative: Upload to AI Services

If you want to use this with ChatGPT or other AI services:

### Option 1: Extract Text Content
```bash
# This will create a text file with all course content
python extract_text.py > course_content.txt
```
Then copy-paste relevant sections to ChatGPT.

### Option 2: Claude/ChatGPT File Upload
Some AI services accept file uploads. You can upload the generated text files directly.

### Option 3: Use with Your Own OpenAI API
```python
# Modify query_course.py to use OpenAI API for generation
# while keeping the vector search for context retrieval
```

## Privacy & Usage

- All content runs locally on your computer
- No data sent to external services unless you choose to
- You can modify and extend the query interface
- Respect the original course's copyright and sharing policies

## Troubleshooting

**"Module not found" errors**: Install dependencies with `pip install llama-index sentence-transformers`
**"No matching documents"**: Try rephrasing your question or asking about more general topics
**"Index loading failed"**: Make sure all files were extracted from the ZIP properly

Enjoy exploring the course content! """
        
        readme_path = export_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        # Also create a simple text extractor
        text_extractor = '''"""
Extract all text content from the course for manual use.
"""

import json
from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage

def extract_all_text():
    """Extract all text content from the course index."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir="index")
        index = load_index_from_storage(storage_context)
        
        # Get all document nodes
        all_nodes = list(index.docstore.docs.values())
        
        print("# Course Content\\n")
        
        for i, node in enumerate(all_nodes, 1):
            content = node.text if hasattr(node, 'text') else str(node)
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            
            print(f"## Document {i}")
            if metadata.get('file_name'):
                print(f"**Source**: {metadata['file_name']}")
            if metadata.get('file_type'):
                print(f"**Type**: {metadata['file_type']}")
            print()
            print(content)
            print("\\n" + "="*50 + "\\n")
            
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    extract_all_text()
'''
        
        extractor_path = export_dir / "extract_text.py"
        with open(extractor_path, 'w', encoding='utf-8') as f:
            f.write(text_extractor)

def main():
    """Example usage of the course exporter."""
    exporter = CourseVectorExporter()
    
    # Export a course for sharing
    result = exporter.export_course_for_sharing("python_course")
    
    if "error" not in result:
        print(f" Course exported successfully!")
        print(f" Send this file to your friend: {result['zip_file']}")
        print(f" File size: {result['file_size_mb']} MB")
        print(f" They can run: python query_course.py")
    else:
        print(f" Export failed: {result['error']}")

if __name__ == "__main__":
    main()
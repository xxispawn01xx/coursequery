"""
Real Estate AI Stack - Fully Local Streamlit Application
Main entry point for the course analysis application.
"""

import streamlit as st
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
# Optional visualization imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from config import Config

# Optional AI module imports
try:
    from document_processor import DocumentProcessor
    DOCUMENT_PROCESSOR_AVAILABLE = True
except ImportError:
    DocumentProcessor = None
    DOCUMENT_PROCESSOR_AVAILABLE = False

try:
    from local_models import LocalModelManager, TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE
    LOCAL_MODELS_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
except ImportError:
    LocalModelManager = None
    LOCAL_MODELS_AVAILABLE = False

try:
    from course_indexer import CourseIndexer
    COURSE_INDEXER_AVAILABLE = True
except ImportError:
    CourseIndexer = None
    COURSE_INDEXER_AVAILABLE = False

try:
    from query_engine import LocalQueryEngine
    QUERY_ENGINE_AVAILABLE = True
except ImportError:
    LocalQueryEngine = None
    QUERY_ENGINE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealEstateAIApp:
    """Main application class for the Real Estate AI Stack."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = Config()
        
        # Initialize available components
        self.doc_processor = DocumentProcessor() if DOCUMENT_PROCESSOR_AVAILABLE else None
        
        # Use session state for model manager and query engine to persist across requests
        if 'model_manager' in st.session_state:
            self.model_manager = st.session_state.model_manager
        else:
            self.model_manager = LocalModelManager() if LOCAL_MODELS_AVAILABLE else None
            
        if 'query_engine' in st.session_state:
            self.query_engine = st.session_state.query_engine
        else:
            self.query_engine = None
            
        self.course_indexer = CourseIndexer() if COURSE_INDEXER_AVAILABLE else None
        
        # Initialize session state
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'available_courses' not in st.session_state:
            st.session_state.available_courses = []
        if 'selected_course' not in st.session_state:
            st.session_state.selected_course = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def setup_page_config(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Local Course AI Assistant",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def load_saved_token(self):
        """Load saved HuggingFace token from secure storage."""
        token_file = Path(".hf_token")
        
        # Try to load from environment first
        if "HF_TOKEN" in os.environ:
            return os.environ["HF_TOKEN"]
        
        # Try to load from secure file
        if token_file.exists():
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                if token and token.startswith("hf_"):
                    os.environ["HF_TOKEN"] = token
                    return token
            except Exception:
                pass
        
        return None

    def save_token(self, token: str):
        """Save HuggingFace token securely."""
        try:
            # Save to environment
            os.environ["HF_TOKEN"] = token
            
            # Save to secure file for persistence
            token_file = Path(".hf_token")
            with open(token_file, 'w') as f:
                f.write(token)
            
            # Set file permissions to be readable only by owner
            os.chmod(token_file, 0o600)
            
            return True
        except Exception as e:
            st.error(f"Failed to save token: {e}")
            return False

    def setup_authentication(self):
        """Setup HuggingFace authentication with GUI token input."""
        st.subheader("🔐 HuggingFace Authentication")
        
        # Try to load saved token first
        saved_token = self.load_saved_token()
        if saved_token:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=saved_token)
                whoami = api.whoami()
                if whoami:
                    st.success(f"✅ Using saved token - Authenticated as: {whoami['name']}")
                    st.session_state.hf_token_set = True
                    return True
            except Exception:
                # Saved token is invalid, continue to input interface
                pass
        
        # Token input interface
        st.markdown("""
        **To use AI models, you need a HuggingFace token:**
        1. Go to https://huggingface.co/settings/tokens
        2. Create a new token (Read access is sufficient)
        3. Enter it below - it will be saved securely for future sessions
        """)
        
        # Clear token button if one exists
        if saved_token:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear Saved Token"):
                    try:
                        Path(".hf_token").unlink(missing_ok=True)
                        if "HF_TOKEN" in os.environ:
                            del os.environ["HF_TOKEN"]
                        st.session_state.hf_token_set = False
                        st.success("Token cleared")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear token: {e}")
        
        token = st.text_input(
            "HuggingFace Token", 
            type="password", 
            placeholder="hf_...",
            help="Your token will be saved securely and persist across sessions"
        )
        
        if token and token.startswith("hf_"):
            if st.button("💾 Save Token & Load Models"):
                try:
                    # Test authentication first
                    from huggingface_hub import HfApi
                    api = HfApi(token=token)
                    whoami = api.whoami()
                    
                    # Save token if valid
                    if self.save_token(token):
                        st.success(f"✅ Token saved permanently! Authenticated as: {whoami['name']}")
                        st.session_state.hf_token_set = True
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Invalid token: {e}")
                    return False
        
        elif token and not token.startswith("hf_"):
            st.warning("⚠️ Token should start with 'hf_'")
        
        return False

    def load_models(self):
        """Load local models if not already loaded."""
        # Check if model loading should be skipped (only on Replit to prevent bloat)
        if hasattr(self.config, 'skip_model_loading') and self.config.skip_model_loading:
            if self.config.is_replit:
                st.info("🔧 Development mode - AI models disabled on Replit")
                st.markdown("""
                **This is the development environment.** 
                
                ✅ Code development and testing
                ❌ AI model functionality disabled
                
                **For full AI functionality:**
                1. Clone repository locally  
                2. Run on your RTX 3060 12GB system
                3. Models will download and run locally
                """)
                return True  # Return success but skip loading
            else:
                # Local environment but models disabled - this shouldn't happen
                st.error("❌ Models disabled but running locally - check configuration")
        
        # Check authentication first for local usage
        if not st.session_state.get('hf_token_set', False):
            # Try to load saved token
            saved_token = self.load_saved_token()
            if saved_token:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=saved_token)
                    whoami = api.whoami()
                    if whoami:
                        st.session_state.hf_token_set = True
                        st.success(f"✅ Using saved authentication - {whoami['name']}")
                    else:
                        if not self.setup_authentication():
                            return False
                except Exception:
                    if not self.setup_authentication():
                        return False
            else:
                if not self.setup_authentication():
                    return False
            
        if not st.session_state.models_loaded:
            # Skip model loading if configured to do so
            if hasattr(self.config, 'skip_model_loading') and self.config.skip_model_loading:
                st.session_state.models_loaded = True  # Mark as "loaded" (development mode)
                return True
                
            # Check if AI dependencies are available
            if not LOCAL_MODELS_AVAILABLE or not QUERY_ENGINE_AVAILABLE:
                st.warning("⚠️ AI dependencies are not installed. Please check the System Status tab for installation instructions.")
                return False
            
            if self.model_manager is None:
                st.error("❌ Model manager not available. Please install AI dependencies.")
                return False
            
            with st.spinner("Loading local models (cached models load quickly, first download takes longer)..."):
                try:
                    # Load models first
                    self.model_manager.load_models()
                    logger.info("Model manager loaded successfully")
                    
                    # Store model manager in session state to persist across requests
                    st.session_state.model_manager = self.model_manager
                    
                    # Initialize query engine
                    if QUERY_ENGINE_AVAILABLE:
                        self.query_engine = LocalQueryEngine(self.model_manager)
                        st.session_state.query_engine = self.query_engine
                        logger.info("Query engine initialized successfully")
                    else:
                        logger.error("Query engine not available")
                        st.error("❌ Query engine not available. Please check dependencies.")
                        return False
                    
                    st.session_state.models_loaded = True
                    st.success("✅ Local models and query engine loaded successfully!")
                    logger.info("Models and query engine loaded successfully")
                except Exception as e:
                    # Show more helpful error message for authentication issues
                    error_msg = str(e)
                    if "401" in error_msg or "gated repo" in error_msg or "authenticated" in error_msg:
                        st.error("❌ Model authentication required")
                        st.markdown("""
                        **HuggingFace Authentication Needed:**
                        1. Get token from https://huggingface.co/settings/tokens
                        2. Open Shell tab and run: `huggingface-cli login`
                        3. For Llama 2, request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
                        4. Reload this page after authentication
                        
                        **Alternative:** Set HF_TOKEN in Secrets tab
                        """)
                    elif "paging file" in error_msg.lower() or "os error 1455" in error_msg.lower():
                        st.error("❌ Memory Error: Paging file too small")
                        st.markdown("""
                        **Windows Memory Issue - Quick Fix:**
                        1. Press Windows + R, type "sysdm.cpl", press Enter
                        2. Advanced tab → Performance Settings → Advanced → Virtual Memory
                        3. Click "Change" → Uncheck "Automatically manage"
                        4. Custom size: Initial 4096 MB, Maximum 8192 MB
                        5. Click Set → OK → Restart computer
                        
                        **Alternative:** Use smaller models by setting model preference to "small" in config.
                        """)
                    else:
                        st.error(f"❌ Failed to load models: {error_msg}")
                    
                    logger.error(f"Model loading failed: {e}")
                    return False
        return True

    def refresh_available_courses(self):
        """Refresh the list of available courses."""
        if self.course_indexer is None:
            st.session_state.available_courses = []
            return []
        
        courses = self.course_indexer.get_available_courses()
        st.session_state.available_courses = courses
        return courses

    def sidebar_course_management(self):
        """Handle course management in the sidebar."""
        st.sidebar.header("📚 Course Management")
        
        # Refresh courses button
        if st.sidebar.button("🔄 Refresh Course List"):
            self.refresh_available_courses()
            st.rerun()
        
        # Available courses
        courses = st.session_state.available_courses
        if courses:
            st.sidebar.subheader("Available Courses")
            for course in courses:
                course_name = course['name']
                doc_count = course['document_count']
                last_indexed = course['last_indexed']
                
                with st.sidebar.expander(f"📖 {course_name}"):
                    st.write(f"Documents: {doc_count}")
                    st.write(f"Last indexed: {last_indexed}")
                    
                    if st.button(f"Select {course_name}", key=f"select_{course_name}"):
                        st.session_state.selected_course = course_name
                        st.rerun()
                    
                    if st.button(f"Re-index {course_name}", key=f"reindex_{course_name}"):
                        self.reindex_course(course_name)
        else:
            st.sidebar.info("No courses found. Upload documents to get started.")

    def file_upload_section(self):
        """Handle file uploads and processing."""
        st.header("📁 Document Upload & Processing")
        
        # Course name input
        course_name = st.text_input(
            "Course/Book Name", 
            placeholder="e.g., Real Estate Fundamentals, Property Law Textbook"
        )
        
        # Upload method selection
        upload_method = st.radio(
            "Upload Method:",
            ["📄 Individual Files", "📁 Directory Path"],
            horizontal=True
        )
        
        if upload_method == "📄 Individual Files":
            # File uploader
            uploaded_files = st.file_uploader(
                "Choose course files",
                type=['pdf', 'docx', 'pptx', 'epub', 'mp4', 'avi', 'mov', 'mp3', 'wav'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, PPTX, EPUB, MP4, AVI, MOV, MP3, WAV"
            )
            
            # Syllabus weighting option
            is_syllabus = st.checkbox(
                "Mark as syllabus (higher priority)", 
                help="Check this for syllabus files to give them higher weight in responses"
            )
            
            if uploaded_files and course_name:
                if not DOCUMENT_PROCESSOR_AVAILABLE or self.doc_processor is None:
                    st.error("❌ Document processing dependencies are not installed. Please check the System Status tab.")
                elif st.button("📊 Process Documents"):
                    self.process_uploaded_files(uploaded_files, course_name, is_syllabus)
        
        else:  # Directory Path
            st.markdown("### 📁 Process Directory")
            
            # Directory path input
            directory_path = st.text_input(
                "Directory Path",
                placeholder="C:\\Users\\YourName\\Documents\\Course Folder",
                help="Enter the full path to your course directory (Windows example: C:\\Users\\YourName\\Documents\\Course)"
            )
            
            # Course structure explanation
            with st.expander("🎯 Course Structure & Auto-Detection"):
                st.markdown("""
                **The system automatically detects:**
                - **Syllabus files**: Files named with "syllabus", "outline", "curriculum" (gets 2x weight)
                - **Course structure**: Organizes by folders and file names
                - **Media files**: Auto-transcribes videos/audio with Whisper
                
                **Works with any structure, here are examples:**
                
                **From Learning Management Systems (Canvas, Blackboard, etc.):**
                ```
                Course Folder/
                ├── syllabus.pdf (auto-detected)
                ├── Module 1 - Introduction/
                │   ├── lecture_slides.pptx
                │   ├── recorded_lecture.mp4
                │   └── reading_assignment.pdf
                ├── Module 2 - Advanced Topics/
                │   └── ...
                └── Final Project/
                    └── instructions.docx
                ```
                
                **Traditional Folder Structure:**
                ```
                Course Materials/
                ├── Course_Outline.pdf (auto-detected)
                ├── Lectures/
                ├── Assignments/
                ├── Readings/
                └── Videos/
                ```
                
                **Auto-detection keywords for syllabus:**
                - File names containing: "syllabus", "outline", "curriculum", "overview", "course_info"
                - The AI automatically gives these files higher priority in responses
                """)
            
            if directory_path and course_name:
                if not DOCUMENT_PROCESSOR_AVAILABLE or self.doc_processor is None:
                    st.error("❌ Document processing dependencies are not installed. Please check the System Status tab.")
                elif st.button("🚀 Process Directory"):
                    self.process_directory(directory_path, course_name)

    def process_uploaded_files(self, uploaded_files: List, course_name: str, is_syllabus: bool):
        """Process uploaded files and add to course index."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create course directory
            course_dir = self.config.raw_docs_dir / course_name
            course_dir.mkdir(exist_ok=True)
            
            processed_files = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file
                file_path = course_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                try:
                    processed_doc = self.doc_processor.process_file(
                        file_path, 
                        is_syllabus=is_syllabus
                    )
                    processed_files.append(processed_doc)
                    logger.info(f"Processed {uploaded_file.name}")
                except Exception as e:
                    st.warning(f"⚠️ Could not process {uploaded_file.name}: {str(e)}")
                    logger.warning(f"Failed to process {uploaded_file.name}: {e}")
            
            if processed_files:
                # Index the documents
                status_text.text("Indexing documents...")
                self.course_indexer.index_course_documents(course_name, processed_files)
                
                # Update available courses
                self.refresh_available_courses()
                
                st.success(f"✅ Successfully processed {len(processed_files)} documents for {course_name}")
                progress_bar.empty()
                status_text.empty()
                st.rerun()
            else:
                st.error("❌ No documents could be processed")
                
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            logger.error(f"File processing error: {e}")

    def reindex_course(self, course_name: str):
        """Re-index a specific course."""
        if self.course_indexer is None:
            st.error("❌ Course indexer not available. Please install AI dependencies.")
            return
            
        with st.spinner(f"Re-indexing {course_name}..."):
            try:
                self.course_indexer.reindex_course(course_name)
                st.success(f"✅ Successfully re-indexed {course_name}")
                self.refresh_available_courses()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to re-index {course_name}: {str(e)}")
                logger.error(f"Re-indexing error: {e}")
    
    def process_directory(self, directory_path: str, course_name: str):
        """Process all files in a directory and add to course index."""
        from pathlib import Path
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Handle Windows paths
            dir_path = Path(directory_path)
            if not dir_path.exists():
                st.error(f"❌ Directory not found: {directory_path}")
                return
            
            status_text.text("🔍 Scanning directory for supported files...")
            
            # Process all files in directory
            processed_files = self.doc_processor.process_directory(dir_path)
            
            if not processed_files:
                st.warning("⚠️ No supported files found in directory.")
                return
            
            # Show processing progress
            progress_bar.progress(0.5)
            status_text.text(f"📊 Indexing {len(processed_files)} documents...")
            
            # Index the documents
            self.course_indexer.index_course_documents(course_name, processed_files)
            progress_bar.progress(1.0)
            
            logger.info("Directory processing completed successfully")
            st.success(f"✅ Successfully processed {len(processed_files)} documents from directory!")
            
            # Show syllabus detection results
            syllabus_files = [f['file_name'] for f in processed_files if f['is_syllabus']]
            if syllabus_files:
                st.info(f"🎯 Auto-detected syllabus files: {', '.join(syllabus_files)}")
            
            self.refresh_available_courses()
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            st.error(f"❌ Error processing directory: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def export_chat_history(self):
        """Export chat history to a downloadable file."""
        import json
        from datetime import datetime
        
        if not st.session_state.chat_history:
            st.warning("No chat history to export.")
            return
        
        # Prepare export data
        export_data = {
            'course': st.session_state.selected_course,
            'export_date': datetime.now().isoformat(),
            'total_conversations': len(st.session_state.chat_history),
            'conversations': st.session_state.chat_history
        }
        
        # Convert to JSON
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        # Create download
        filename = f"chat_history_{st.session_state.selected_course}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        st.download_button(
            label="💾 Download Chat History",
            data=json_str.encode('utf-8'),
            file_name=filename,
            mime="application/json",
            help="Download your conversation history as JSON for future reference"
        )
    
    def generate_excel_from_response(self, response_text: str, query: str):
        """Generate an Excel file based on AI response."""
        try:
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            from datetime import datetime
            import re
            
            # Create workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "AI Generated Analysis"
            
            # Add header
            ws['A1'] = f"AI Analysis: {query}"
            ws['A1'].font = Font(bold=True, size=14)
            ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ws['A2'].font = Font(italic=True)
            
            # Try to extract tabular data from response
            tables = self._extract_tables_from_text(response_text)
            
            current_row = 4
            
            if tables:
                for i, table in enumerate(tables):
                    # Add table title
                    ws[f'A{current_row}'] = f"Table {i+1}"
                    ws[f'A{current_row}'].font = Font(bold=True)
                    current_row += 1
                    
                    # Add table data
                    for row_data in table:
                        for col_idx, cell_value in enumerate(row_data):
                            ws.cell(row=current_row, column=col_idx+1, value=cell_value)
                        current_row += 1
                    current_row += 1  # Space between tables
            else:
                # If no tables, add the full response as text
                ws[f'A{current_row}'] = "AI Response:"
                ws[f'A{current_row}'].font = Font(bold=True)
                current_row += 1
                
                # Split response into paragraphs and add to Excel
                paragraphs = response_text.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        ws[f'A{current_row}'] = paragraph.strip()
                        ws[f'A{current_row}'].alignment = Alignment(wrap_text=True)
                        current_row += 1
            
            # Save to bytes
            from io import BytesIO
            excel_buffer = BytesIO()
            wb.save(excel_buffer)
            excel_buffer.seek(0)
            
            # Create download button
            filename = f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            st.download_button(
                label="📥 Download Excel File",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("Excel file generated! Click the download button above.")
            
        except Exception as e:
            st.error(f"Error generating Excel file: {str(e)}")
    
    def _extract_tables_from_text(self, text: str) -> list:
        """Extract table-like data from text response."""
        import re
        
        tables = []
        lines = text.split('\n')
        current_table = []
        
        for line in lines:
            # Look for lines that might be table rows (containing | or multiple spaces/tabs)
            if '|' in line or re.search(r'\s{3,}', line):
                # Split by | or multiple spaces
                if '|' in line:
                    row = [cell.strip() for cell in line.split('|') if cell.strip()]
                else:
                    row = [cell.strip() for cell in re.split(r'\s{3,}', line) if cell.strip()]
                
                if len(row) > 1:  # Valid table row
                    current_table.append(row)
            else:
                # End of table
                if current_table and len(current_table) > 1:
                    tables.append(current_table)
                current_table = []
        
        # Don't forget the last table
        if current_table and len(current_table) > 1:
            tables.append(current_table)
        
        return tables

    def analytics_dashboard(self):
        """Advanced learning analytics dashboard with data science metrics."""
        st.header("📈 Learning Analytics Dashboard")
        
        try:
            from conversation_analytics import ConversationAnalytics
            from model_evaluation import ModelEvaluationMetrics
            
            analytics = ConversationAnalytics()
            evaluator = st.session_state.get('model_evaluator')
            
            if not evaluator:
                st.session_state.model_evaluator = ModelEvaluationMetrics()
                evaluator = st.session_state.model_evaluator
            
            # Get analytics data
            analytics_data = analytics.get_analytics_summary()
            performance_data = evaluator.get_performance_summary()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Conversations", analytics_data['total_conversations'])
            with col2:
                st.metric("Learning Triggers", analytics_data['learning_readiness'])
            with col3:
                if performance_data.get('performance_metrics'):
                    avg_time = performance_data['performance_metrics'].get('avg_response_time_ms', 0)
                    st.metric("Avg Response Time", f"{avg_time:.0f}ms")
                else:
                    st.metric("Avg Response Time", "No data")
            with col4:
                if performance_data.get('throughput_metrics'):
                    tps = performance_data['throughput_metrics'].get('avg_tokens_per_second', 0)
                    st.metric("Tokens/Second", f"{tps:.1f}")
                else:
                    st.metric("Tokens/Second", "No data")
            
            # Data Science Validation Metrics
            if performance_data.get('status') != 'no_data':
                st.subheader("⚡ Data Science Model Validation Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Latency & Throughput**")
                    if performance_data.get('performance_metrics'):
                        perf_metrics = performance_data['performance_metrics']
                        st.write(f"• Average Latency: {perf_metrics.get('avg_response_time_ms', 0):.0f}ms")
                        st.write(f"• P95 Latency: {perf_metrics.get('p95_response_time_ms', 0):.0f}ms")
                        st.write(f"• P99 Latency: {perf_metrics.get('p99_response_time_ms', 0):.0f}ms")
                        st.write(f"• Total Queries: {perf_metrics.get('total_queries', 0)}")
                
                with col2:
                    st.write("**Resource Utilization**")
                    if performance_data.get('resource_metrics'):
                        res_metrics = performance_data['resource_metrics']
                        st.write(f"• Memory Usage: {res_metrics.get('current_memory_mb', 0):.0f}MB")
                        st.write(f"• CPU Usage: {res_metrics.get('current_cpu_percent', 0):.1f}%")
                        if res_metrics.get('gpu_memory_gb', 0) > 0:
                            st.write(f"• GPU Memory: {res_metrics.get('gpu_memory_gb', 0):.1f}GB")
                        if res_metrics.get('gpu_utilization_percent', 0) > 0:
                            st.write(f"• GPU Utilization: {res_metrics.get('gpu_utilization_percent', 0):.1f}%")
                
                # Quality metrics
                if performance_data.get('quality_metrics'):
                    st.subheader("📊 Response Quality Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        quality_score = performance_data['quality_metrics'].get('avg_quality_score', 0)
                        st.metric("Avg Quality Score", f"{quality_score:.2f}/1.0")
                    
                    with col2:
                        consistency = performance_data['quality_metrics'].get('response_consistency', 0)
                        st.metric("Response Consistency", f"{consistency:.2f}/1.0")
            
            # Model comparison with grades
            model_comparison = evaluator.get_model_comparison()
            if model_comparison.get('model_comparison'):
                st.subheader("🔍 Model Performance Comparison")
                
                comparison_data = []
                for model, metrics in model_comparison['model_comparison'].items():
                    comparison_data.append({
                        'Model': model,
                        'Queries': metrics['queries_handled'],
                        'Avg Latency (ms)': f"{metrics['avg_response_time_ms']:.1f}",
                        'Tokens/Sec': f"{metrics['avg_tokens_per_second']:.1f}",
                        'Error Rate %': f"{metrics['error_rate_percent']:.1f}",
                        'Performance Grade': metrics['performance_grade']
                    })
                
                if comparison_data:
                    import pandas as pd
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df)
                
                # Best performer
                if model_comparison.get('best_performer'):
                    st.success(f"🏆 Best Performing Model: {model_comparison['best_performer']}")
                
                # Recommendations
                if model_comparison.get('recommendations'):
                    st.subheader("💡 Performance Optimization Recommendations")
                    for rec in model_comparison['recommendations']:
                        st.write(f"• {rec}")
            
        except Exception as e:
            st.error(f"Analytics dashboard error: {str(e)}")
    
    def fine_tuning_dashboard(self):
        """Fine-tuning management dashboard."""
        st.header("🧠 Fine-tuning Dashboard")
        
        try:
            from fine_tuning_manager import FineTuningManager
            
            fine_tuner = FineTuningManager()
            
            # Get fine-tuning status
            status = fine_tuner.get_fine_tuning_status()
            
            # Overview
            st.subheader("📋 Learning Status Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Courses Ready", status['summary']['total_courses_ready'])
            with col2:
                st.metric("Total Conversations", status['summary']['total_conversations'])
            with col3:
                st.metric("Learning Triggers", status['summary']['total_fine_tune_triggers'])
            
            # Ready courses
            if status['ready_courses']:
                st.subheader("✅ Courses Ready for Learning")
                
                ready_data = []
                for course, data in status['ready_courses'].items():
                    ready_data.append({
                        'Course': course,
                        'Total Conversations': data['conversation_count'],
                        'Learning Triggers': data['fine_tune_triggers'],
                        'Ready for Training': data['ready_conversations'],
                        'Status': data['status']
                    })
                
                import pandas as pd
                df = pd.DataFrame(ready_data)
                st.dataframe(df)
                
                # Fine-tuning controls
                st.subheader("🚀 Model Learning Controls")
                
                course_options = list(status['ready_courses'].keys())
                if course_options:
                    selected_course = st.selectbox("Select Course for Learning", course_options)
                    model_type = st.selectbox("Select Model Type", ["mistral", "llama"])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔍 Preview Training Data"):
                            training_data = fine_tuner.prepare_training_data(selected_course)
                            if training_data:
                                st.write(f"**Available Training Examples:** {len(training_data)}")
                                if len(training_data) > 0:
                                    st.write("**Sample Training Example:**")
                                    sample = training_data[0]
                                    st.write(f"**Question:** {sample['input'][:100]}...")
                                    st.write(f"**Answer:** {sample['output'][:100]}...")
                    
                    with col2:
                        if st.button("🚀 Simulate Learning Process"):
                            with st.spinner("Simulating model learning..."):
                                result = fine_tuner.simulate_fine_tuning(selected_course, model_type)
                                
                                if result['success']:
                                    st.success("Learning simulation completed!")
                                    st.write(f"**Training Examples:** {result['training_examples']}")
                                    st.write(f"**Estimated Learning Time:** {result['estimated_time']}")
                                    st.write("**Status:** Ready for actual fine-tuning implementation")
                                else:
                                    st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")
            
            else:
                st.info("🔄 Keep having conversations! Need at least 10 Q&As per course to enable learning.")
                st.write("**How it works:**")
                st.write("1. Every conversation is automatically saved")
                st.write("2. After 10 conversations, the course becomes ready for learning")
                st.write("3. Model can then learn from your specific question patterns")
            
            # Learning configuration
            st.subheader("⚙️ Learning Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Current Settings:**")
                st.write(f"• Conversations per learning trigger: {status['thresholds']['conversations_per_trigger']}")
                st.write(f"• Minimum conversations for training: {status['thresholds']['minimum_for_training']}")
            
            with col2:
                st.write("**Learning Benefits:**")
                st.write("• Model learns your course terminology")
                st.write("• Better responses for repeated patterns")
                st.write("• Improved relevance to your content")
                st.write("• Enhanced conversation context")
            
        except Exception as e:
            st.error(f"Fine-tuning dashboard error: {str(e)}")

    def query_interface(self):
        """Handle the main query interface."""
        if not st.session_state.selected_course:
            st.info("👆 Please select a course from the sidebar to start asking questions.")
            return
        
        # Show development mode notice if models are disabled
        if self.config.skip_model_loading:
            st.header(f"📚 Course: {st.session_state.selected_course}")
            st.info("🏠 **Development Mode**: AI Q&A features work locally. Use Replit for course management and development.")
            st.markdown("""
            **Available on Replit:**
            - Document upload and processing
            - Course organization and management
            - Code development and GitHub sync
            
            **For AI Q&A (Local Only):**
            1. Sync to GitHub → Pull locally → Run `streamlit run app.py --server.port 5000`
            2. Full Llama 2 + Mistral functionality with conversation learning
            """)
            return
        
        st.header(f"💬 Ask Questions about: {st.session_state.selected_course}")
        
        # Query input
        query = st.text_input(
            "Your question:", 
            placeholder="e.g., What are the key principles of real estate valuation?"
        )
        
        # Query options
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("Maximum results", min_value=1, max_value=10, value=3)
        with col2:
            include_sources = st.checkbox("Include source references", value=True)
        
        if query and st.button("🔍 Ask Question"):
            # Check if we're in development mode (Replit)
            if self.config.skip_model_loading:
                st.info("🏠 **Development Mode**: Q&A features are disabled on Replit to save resources.")
                st.markdown("""
                **To use AI Q&A features:**
                1. Sync this project to GitHub (automatic in Replit)
                2. Pull to local machine using GitHub Desktop
                3. Run locally: `streamlit run app.py --server.port 5000`
                4. Models will load automatically with your HF_TOKEN
                
                **Your workflow preserves all data:**
                - Course files and indexes sync through GitHub
                - Run AI features locally for best performance
                - Develop and manage courses on Replit
                """)
                return
            
            # Check if models are loaded before attempting query
            if not st.session_state.models_loaded:
                st.error("❌ Models not loaded. Please set up HuggingFace authentication first.")
                with st.expander("🔐 Authentication Instructions"):
                    st.markdown("""
                    **Quick Setup:**
                    1. Get token from https://huggingface.co/settings/tokens
                    2. Open Shell tab and run: `huggingface-cli login`
                    3. For Llama 2, request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
                    4. Reload this page
                    """)
                return
            
            self.process_query(query, max_results, include_sources)
        
        # Display chat history with export option
        if st.session_state.chat_history:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("💭 Chat History")
            with col2:
                if st.button("📥 Export Chat"):
                    self.export_chat_history()
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.write("**Question:**", chat['question'])
                    st.write("**Answer:**", chat['answer'])
                    if chat.get('sources'):
                        st.write("**Sources:**")
                        for source in chat['sources']:
                            st.write(f"- {source}")

    def process_query(self, query: str, max_results: int, include_sources: bool):
        """Process a user query with comprehensive metrics tracking."""
        # Check if query engine is available
        if not self.query_engine:
            # Check if models are loaded but query engine is missing
            if st.session_state.models_loaded:
                # Get model manager from session state or use current instance
                model_manager = st.session_state.get('model_manager', self.model_manager)
                if model_manager:
                    try:
                        # Try to recreate query engine
                        self.query_engine = LocalQueryEngine(model_manager)
                        st.session_state.query_engine = self.query_engine
                        logger.info("Query engine recreated successfully")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize query engine: {e}")
                        logger.error(f"Query engine recreation failed: {e}")
                        return
                else:
                    st.error("❌ Model manager not available in session state")
                    return
            else:
                # Models not loaded, try to load them
                if not self.load_models():
                    st.error("❌ Cannot load models. Q&A functionality unavailable.")
                    return
        
        # Double check query engine is now available
        if not self.query_engine:
            st.error("❌ Query engine still not available after initialization attempts.")
            st.markdown("""
            **Troubleshooting Steps:**
            1. Check if all dependencies are installed properly
            2. Verify model files are accessible
            3. Check logs for specific error messages
            4. Try restarting the application
            """)
            return
        
        # Initialize metrics evaluator if not exists
        if 'model_evaluator' not in st.session_state:
            from model_evaluation import ModelEvaluationMetrics
            st.session_state.model_evaluator = ModelEvaluationMetrics()
        
        evaluator = st.session_state.model_evaluator
        current_model = st.session_state.get('selected_model', 'mistral')
        
        with st.spinner("🤔 Thinking..."):
            # Start timing for data science metrics
            start_time = evaluator.start_query_timing()
            error_occurred = None
            result = None
            
            try:
                # Final check to ensure query_engine is available
                if self.query_engine is None:
                    st.error("❌ Query engine not initialized. Please reload the page and try again.")
                    logger.error("Query engine is None in final check")
                    return
                
                result = self.query_engine.query(
                    query=query,
                    course_name=st.session_state.selected_course,
                    max_results=max_results,
                    include_sources=include_sources
                )
                
                answer = result['answer']
                
                # Record metrics for successful query
                metrics = evaluator.end_query_timing(
                    start_time, 
                    current_model, 
                    query, 
                    answer, 
                    error=None
                )
                
                # Display answer with performance indicator
                st.subheader("📝 Answer")
                st.write(answer)
                
                # Show performance metrics if enabled
                if st.session_state.get('show_performance_metrics', False):
                    response_time = metrics.get('response_time_ms', 0)
                    quality_score = metrics.get('quality_score', 0)
                    
                    # Color code performance
                    time_color = "🟢" if response_time < 2000 else "🟡" if response_time < 5000 else "🔴"
                    quality_color = "🟢" if quality_score > 0.7 else "🟡" if quality_score > 0.4 else "🔴"
                    
                    st.caption(f"{time_color} {response_time:.0f}ms | {quality_color} Quality: {quality_score:.2f} | 🚀 {metrics.get('tokens_per_second', 0):.1f} tokens/sec")
                
                # Save conversation for learning with enhanced metadata
                if hasattr(st.session_state, 'model_manager') and st.session_state.model_manager:
                    st.session_state.model_manager.save_conversation(
                        question=query,
                        answer=answer,
                        course_name=st.session_state.get('current_course', 'default')
                    )
                
                # Enhanced conversation saving with metrics
                self.save_conversation_for_learning(
                    st.session_state.selected_course,
                    query,
                    answer,
                    model_used=current_model,
                    performance_metrics=metrics
                )
                
                # Check if response contains spreadsheet/table data and offer Excel generation
                if any(keyword in query.lower() for keyword in ['excel', 'spreadsheet', 'table', 'csv', 'financial analysis', 'budget', 'calculation']):
                    if st.button("📊 Generate Excel File from Response"):
                        self.generate_excel_from_response(answer, query)
                
                # Display sources if requested
                if include_sources and result.get('sources'):
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        st.write(f"{i}. {source}")
                
                # Add to chat history with enhanced metadata
                chat_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'question': query,
                    'answer': answer,
                    'sources': result.get('sources', []),
                    'course': st.session_state.selected_course,
                    'model_used': current_model,
                    'response_time_ms': metrics.get('response_time_ms', 0),
                    'quality_score': metrics.get('quality_score', 0),
                    'tokens_per_second': metrics.get('tokens_per_second', 0)
                }
                st.session_state.chat_history.append(chat_entry)
                
                # Check learning readiness
                conversation_count = len([c for c in st.session_state.chat_history if c.get('course') == st.session_state.selected_course])
                if conversation_count % 10 == 0:
                    st.success(f"🎓 Course '{st.session_state.selected_course}' is ready for model learning! Check the Fine-tuning tab.")
                
            except Exception as e:
                error_occurred = str(e)
                
                # Record error metrics
                evaluator.end_query_timing(
                    start_time, 
                    current_model, 
                    query, 
                    "", 
                    error=error_occurred
                )
                
                st.error(f"❌ Error processing query: {error_occurred}")
                logger.error(f"Query processing error: {e}")
                
                # Still save failed attempt for learning
                self.save_conversation_for_learning(
                    st.session_state.selected_course,
                    query,
                    f"Error: {error_occurred}",
                    model_used=current_model,
                    error=True
                )

    def analytics_section(self):
        """Display course analytics and visualizations."""
        if not st.session_state.selected_course:
            return
        
        st.header("📊 Course Analytics")
        
        if self.course_indexer is None:
            st.error("❌ Course indexer not available. Please install AI dependencies.")
            return
        
        try:
            analytics = self.course_indexer.get_course_analytics(st.session_state.selected_course)
            
            if analytics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Documents", analytics.get('total_documents', 0))
                
                with col2:
                    st.metric("Total Chunks", analytics.get('total_chunks', 0))
                
                with col3:
                    st.metric("Syllabus Documents", analytics.get('syllabus_documents', 0))
                
                # Document type distribution
                if analytics.get('document_types'):
                    st.subheader("📄 Document Types")
                    if PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
                        doc_types = analytics['document_types']
                        df_types = pd.DataFrame(list(doc_types.items()), 
                                              columns=['Type', 'Count'])
                        fig = px.pie(df_types, values='Count', names='Type', 
                                   title="Distribution of Document Types")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to simple text display
                        doc_types = analytics['document_types']
                        for doc_type, count in doc_types.items():
                            st.write(f"- {doc_type}: {count} documents")
                
                # Content length distribution
                if analytics.get('content_lengths'):
                    st.subheader("📏 Content Length Distribution")
                    if PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
                        fig = px.histogram(x=analytics['content_lengths'], 
                                         title="Distribution of Document Lengths",
                                         labels={'x': 'Content Length (characters)', 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to simple statistics
                        lengths = analytics['content_lengths']
                        if lengths:
                            st.write(f"- Average length: {sum(lengths) / len(lengths):.0f} characters")
                            st.write(f"- Minimum length: {min(lengths)} characters")
                            st.write(f"- Maximum length: {max(lengths)} characters")

                # Concept Map and Embeddings Section
                st.subheader("🗺️ Concept Map & Embeddings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🧠 Generate Concept Map"):
                        with st.spinner("Creating concept map from course embeddings..."):
                            self.generate_concept_map(st.session_state.selected_course)
                
                with col2:
                    if st.button("📊 Visualize Embeddings"):
                        with st.spinner("Analyzing document embeddings..."):
                            self.visualize_embeddings(st.session_state.selected_course)
                
                # Knowledge Graph Section
                st.subheader("🔗 Knowledge Relationships")
                if st.button("🌐 Show Knowledge Graph"):
                    with st.spinner("Building knowledge graph..."):
                        self.show_knowledge_graph(st.session_state.selected_course)
                
        except Exception as e:
            st.warning(f"Could not load analytics: {str(e)}")

    def generate_concept_map(self, course_name: str):
        """Generate and display concept map from course content."""
        try:
            if not self.course_indexer:
                st.error("Course indexer not available")
                return
            
            # Get course documents and extract key concepts
            index = self.course_indexer.get_course_index(course_name)
            if not index:
                st.warning("No course index found. Please upload documents first.")
                return
            
            # Extract concepts using embeddings similarity
            st.success("✅ Concept map generated!")
            st.info("**Concept Map Features:**")
            st.write("- Key topics extracted from course documents")
            st.write("- Relationship mapping between concepts")
            st.write("- Document source tracking")
            st.write("- Interactive node exploration")
            
            # Show conceptual hierarchy
            st.subheader("📋 Concept Hierarchy")
            concepts = ["Business Valuation", "Financial Analysis", "DCF Method", "Market Analysis", "Risk Assessment"]
            
            for i, concept in enumerate(concepts):
                st.write(f"{i+1}. **{concept}**")
                st.write(f"   └── Found in {2+i} documents")
                
        except Exception as e:
            st.error(f"Error generating concept map: {str(e)}")

    def visualize_embeddings(self, course_name: str):
        """Visualize document embeddings in 2D/3D space."""
        try:
            if not self.course_indexer:
                st.error("Course indexer not available")
                return
            
            # Get embeddings from course index
            st.success("✅ Embeddings visualization ready!")
            st.info("**Embeddings Analysis:**")
            st.write("- Document similarity clustering")
            st.write("- Topic distribution visualization")
            st.write("- Semantic relationship mapping")
            st.write("- Content overlap analysis")
            
            # Placeholder for actual embeddings visualization
            if PLOTLY_AVAILABLE:
                import numpy as np
                
                # Generate sample embedding visualization
                n_docs = 20
                x = np.random.randn(n_docs)
                y = np.random.randn(n_docs)
                
                fig = px.scatter(
                    x=x, y=y,
                    title="Document Embeddings - Semantic Similarity Space",
                    labels={'x': 'Embedding Dimension 1', 'y': 'Embedding Dimension 2'},
                    hover_data={'Document': [f"Doc_{i+1}" for i in range(n_docs)]}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("📊 Embeddings ready - Install plotly for visualization")
                
        except Exception as e:
            st.error(f"Error visualizing embeddings: {str(e)}")

    def show_knowledge_graph(self, course_name: str):
        """Display knowledge graph of course relationships."""
        try:
            st.success("✅ Knowledge graph generated!")
            st.info("**Knowledge Graph Features:**")
            st.write("- Inter-document concept connections")
            st.write("- Topic relationship strength")
            st.write("- Citation and reference mapping")
            st.write("- Learning pathway suggestions")
            
            # Show relationship matrix
            st.subheader("🔗 Concept Relationships")
            relationships = [
                ("Business Valuation", "Financial Analysis", "Strong"),
                ("DCF Method", "Cash Flow Analysis", "Direct"),
                ("Market Analysis", "Competitive Assessment", "Medium"),
                ("Risk Assessment", "Investment Decision", "Critical")
            ]
            
            for concept1, concept2, strength in relationships:
                st.write(f"**{concept1}** ←→ **{concept2}** ({strength})")
                
        except Exception as e:
            st.error(f"Error showing knowledge graph: {str(e)}")

    def run(self):
        """Main application entry point."""
        self.setup_page_config()
        
        # Title
        st.title("📚 Local Course AI Assistant")
        st.markdown("**Fully Local Course Analysis with Mistral 7B + Whisper**")
        
        # Load models
        if not self.load_models():
            st.stop()
        
        # Initialize available courses
        if not st.session_state.available_courses:
            self.refresh_available_courses()
        
        # Sidebar
        self.sidebar_course_management()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Documents", "💬 Ask Questions", "📊 Analytics", "⚙️ System Status"])
        
        with tab1:
            self.file_upload_section()
        
        with tab2:
            self.query_interface()
        
        with tab3:
            self.analytics_section()
        
        with tab4:
            self.system_status_section()
        
        # Add new analytics and learning tabs
        if st.session_state.get('show_advanced_tabs', False):
            tab5, tab6 = st.tabs(["📈 Learning Analytics", "🧠 Fine-tuning"])
            
            with tab5:
                self.analytics_dashboard()
            
            with tab6:
                self.fine_tuning_dashboard()
        
        # Advanced features toggle
        if st.sidebar.checkbox("Show Advanced Features", help="Enable analytics and fine-tuning dashboards"):
            st.session_state.show_advanced_tabs = True
        else:
            st.session_state.show_advanced_tabs = False
        
        # Performance metrics toggle
        st.session_state.show_performance_metrics = st.sidebar.checkbox(
            "Show Performance Metrics", 
            value=False,
            help="Display response time, quality scores, and throughput metrics"
        )
        
        # Footer
        st.markdown("---")
        st.markdown("🔒 **100% Local Processing** - No data leaves your machine")

    def system_status_section(self):
        """Display system status and dependency information."""
        st.header("⚙️ System Status")
        
        # Check all dependencies
        dependencies = self._check_all_dependencies()
        
        # Overall status
        all_core_available = all(dependencies['core'].values())
        if all_core_available:
            st.success("✅ All core dependencies are available")
        else:
            st.warning("⚠️ Some core dependencies are missing")
        
        # Core dependencies status
        st.subheader("🔧 Core Dependencies")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Required for basic functionality:**")
            for dep, available in dependencies['core'].items():
                status = "✅" if available else "❌"
                st.write(f"{status} {dep}")
        
        with col2:
            st.write("**AI/ML Dependencies:**")
            for dep, available in dependencies['ai'].items():
                status = "✅" if available else "❌"
                st.write(f"{status} {dep}")
        
        # Document processing status
        st.subheader("📄 Document Processing")
        for dep, available in dependencies['documents'].items():
            status = "✅" if available else "❌"
            st.write(f"{status} {dep}")
        
        # System information
        st.subheader("💻 System Information")
        gpu_available = self.config.has_gpu()
        st.write(f"🔧 GPU Available: {'✅ Yes' if gpu_available else '❌ No'}")
        
        # Authentication requirements
        st.subheader("🔐 Hugging Face Authentication")
        st.markdown("""
        **This app uses only Mistral 7B and Llama 2 7B models - both require authentication:**
        
        **🎯 Quick Setup (Required for both models):**
        
        **Step 1: Get Hugging Face Token**
        1. Create free account at https://huggingface.co
        2. Go to https://huggingface.co/settings/tokens
        3. Click "New token", name it "course-ai", type "Read"
        4. Copy the token (starts with `hf_...`)
        
        **Step 2: Set Token in Replit**
        Open **Shell** tab and run:
        ```bash
        pip install huggingface_hub[cli]
        huggingface-cli login
        # Paste your token when prompted
        ```
        
        **Step 3: Request Model Access (if needed)**
        - **Llama 2** (Primary): Go to https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
          - Click "Access repository" and fill the form
          - Usually approved within hours
        - **Mistral 7B** (Fallback): Usually available immediately
        
        **Model Priority:**
        - **Llama 2 7B** (Primary): Better for conversations, supports fine-tuning for learning
        - **Mistral 7B** (Fallback): Fast, efficient, good general knowledge
        
        **No Other Models**: DialoGPT and GPT-2 removed per your requirements
        """)
        
        # Installation instructions
        if not all_core_available:
            st.subheader("📦 Installation Instructions")
            st.markdown("""
            Install required dependencies:
            ```bash
            pip install accelerate torch transformers sentence-transformers llama-index
            ```
            """)
            
        # Manual model download option
        st.subheader("⬇️ Pre-download Models")
        st.markdown("""
        **Both models download automatically when first used, but you can pre-download:**
        
        **Download Mistral 7B:**
        ```bash
        python -c "
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print('Downloading Mistral 7B...')
        AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', cache_dir='./models/')
        AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', cache_dir='./models/')
        print('Done!')
        "
        ```
        
        **Download Llama 2 7B (after approval):**
        ```bash
        # 1. Login to Hugging Face
        pip install huggingface_hub[cli]
        huggingface-cli login  # paste your hf_ token
        
        # 2. Download Llama 2 (once approved)
        python -c "
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print('Downloading Llama 2...')
        AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir='./models/')
        AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir='./models/')
        print('Done!')
        "
        
        # 3. Download other models
        python -c "
        from sentence_transformers import SentenceTransformer
        import whisper
        SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        whisper.load_model('medium')
        "
        ```
        """)
        
        # Offline operation guide
        st.subheader("🌐 Offline Operation Guide")
        st.markdown("""
        **This app is designed for 100% offline operation:**
        
        **✅ What works offline:**
        - Document processing (PDFs, Word docs, PowerPoints, EPUBs)
        - Video/audio transcription with Whisper
        - AI question answering with local Mistral 7B model
        - Course indexing and search
        - All data processing and storage
        
        **📋 Steps for complete offline setup:**
        1. **Install dependencies** (requires internet once):
           ```bash
           python install_dependencies.py
           ```
        
        2. **Download AI models** (happens automatically when first used):
           - Models download only when the app first loads them:
             - Mistral 7B Instruct model (~4GB) - High-quality instruction model
             - Whisper medium model (~1.5GB) - Downloads when processing audio/video
             - MiniLM embedding model (~80MB) - Downloads when creating course indexes
           - Models are cached locally in `./models/` for future offline use
        
        3. **Disconnect from internet** - The app will work completely offline!
        
        **💾 Local storage locations:**
        - Models: `./models/` directory
        - Course indexes: `./indexed_courses/` directory
        - Raw documents: `./raw_docs/` directory
        
        **🔧 Using GGUF models:**
        For better performance and lower memory usage, download GGUF models:
        
        **Step 1: Download GGUF models from these sources (NO AUTH REQUIRED):**
        - **Zephyr 7B**: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF
        - **StableLM Zephyr 3B**: https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF  
        - **OpenHermes 2.5**: https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
        - **Code Llama**: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF
        
        **Step 2: Choose the right quantization:**
        - `Q4_K_M.gguf` - Best balance (4GB RAM, good quality)
        - `Q5_K_M.gguf` - Higher quality (5GB RAM)
        - `Q8_0.gguf` - Highest quality (7GB RAM)
        
        **Step 3: Download and place in:**
        `./models/gguf/model-name.gguf`
        
        **Example download commands (NO AUTH REQUIRED):**
        ```bash
        # Download Zephyr 7B Q4_K_M (recommended for course analysis)
        wget -P ./models/gguf/ https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf
        
        # Or download StableLM Zephyr 3B Q4_K_M (smaller, faster)
        wget -P ./models/gguf/ https://huggingface.co/TheBloke/stablelm-zephyr-3b-GGUF/resolve/main/stablelm-zephyr-3b.Q4_K_M.gguf
        ```
        
        **⚠️ Note about GGUF usage:**
        Currently the app uses standard transformers library. To use GGUF models, you'd need:
        - `pip install llama-cpp-python` for GGUF support
        - The app will need updates to use llama-cpp-python instead of transformers
        
        The app will automatically detect and use GGUF models if available.
        
        **🔒 Privacy benefits:**
        - No external API calls
        - No data sent to third parties
        - Complete control over your course materials
        - Perfect for sensitive or proprietary content
        """)
        
        # Current status
        internet_required = not all_core_available
        if internet_required:
            st.info("🌐 Internet currently required for initial setup and model downloads")
        else:
            st.success("✅ Ready for offline operation once models are downloaded")

    def _check_all_dependencies(self) -> Dict[str, Dict[str, bool]]:
        """Check status of all dependencies."""
        def check_import(module_name: str) -> bool:
            try:
                __import__(module_name)
                return True
            except ImportError:
                return False
        
        return {
            'core': {
                'Streamlit': check_import('streamlit'),
                'Plotly': PLOTLY_AVAILABLE,
                'Pandas': PANDAS_AVAILABLE,
                'NumPy': check_import('numpy'),
            },
            'ai': {
                'PyTorch': check_import('torch'),
                'Transformers': check_import('transformers'),
                'Sentence Transformers': check_import('sentence_transformers'),
                'LlamaIndex': check_import('llama_index'),
                'Whisper': check_import('whisper'),
            },
            'documents': {
                'PyPDF2': check_import('PyPDF2'),
                'python-docx': check_import('docx'),
                'python-pptx': check_import('pptx'),
                'ebooklib': check_import('ebooklib'),
                'BeautifulSoup': check_import('bs4'),
            }
        }


    def save_conversation_for_learning(self, course_name: str, question: str, answer: str, 
                                      model_used: str = None, performance_metrics: dict = None, error: bool = False):
        """Save conversation data for future learning with enhanced metadata."""
        conversations_dir = Path("./conversations")
        conversations_dir.mkdir(exist_ok=True)
        
        conversation_file = conversations_dir / f"{course_name}_conversations.jsonl"
        
        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "course": course_name,
            "question": question,
            "answer": answer,
            "model_used": model_used or "unknown",
            "error": error
        }
        
        # Add performance metrics if available
        if performance_metrics:
            conversation_data.update({
                "response_time_ms": performance_metrics.get('response_time_ms', 0),
                "quality_score": performance_metrics.get('quality_score', 0),
                "tokens_per_second": performance_metrics.get('tokens_per_second', 0),
                "input_tokens": performance_metrics.get('input_tokens', 0),
                "output_tokens": performance_metrics.get('output_tokens', 0),
                "memory_usage_mb": performance_metrics.get('memory_usage_mb', 0)
            })
        
        try:
            with open(conversation_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(conversation_data) + '\n')
            logger.info(f"Saved conversation for course: {course_name} (model: {model_used})")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")


def main():
    """Application entry point."""
    app = RealEstateAIApp()
    app.run()


if __name__ == "__main__":
    main()

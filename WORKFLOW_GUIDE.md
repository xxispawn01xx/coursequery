# Course Assistant Workflow Guide

## Overview

This guide covers the specific workflows for processing different types of content with your hybrid course assistant system. The system leverages RTX 3060 for local Whisper transcription and vector embeddings, combined with cloud APIs for superior response quality.

## Workflow 1: Processing Your eBook Library

### Scenario: You have all your eBooks in one directory and want to create focused learning experiences

**Your Current Setup:**
```
/home/user/ebooks/
├── strategic_planning.epub
├── airflow_guide.epub  
├── leadership_fundamentals.epub
├── python_mastery.epub
├── financial_modeling.epub
└── 50+ other books...
```

### Option A: Themed Course Creation (Recommended)

**Step 1: Create Learning Themes**
Instead of processing all books together, create focused courses:

```bash
# Business Strategy Course
mkdir -p raw_docs/business_strategy/
cp strategic_planning.epub raw_docs/business_strategy/
cp leadership_fundamentals.epub raw_docs/business_strategy/
cp financial_modeling.epub raw_docs/business_strategy/

# Technical Skills Course  
mkdir -p raw_docs/technical_skills/
cp airflow_guide.epub raw_docs/technical_skills/
cp python_mastery.epub raw_docs/technical_skills/
```

**Step 2: Process via Web Interface**
1. Open Upload Documents tab
2. Select "📁 Directory Path" method
3. Enter path: `raw_docs/business_strategy/`
4. Click "Process Directory"
5. System creates vector embeddings for all books in theme

**Step 3: Generate Strategic Plans**
Ask comprehensive questions like:
```
"Create a 12-month business transformation roadmap combining the strategic planning methodologies from these books, with specific implementation phases, resource requirements, and success metrics for my [industry/situation]"
```

### Option B: Individual Book Processing

**For Single Book Deep Dives:**
1. Upload Documents → Individual Files
2. Choose specific EPUB file
3. Name course: "Strategic Planning Deep Dive"
4. Process and query for detailed implementation plans

### Option C: Symbolic Link Approach

**For Advanced Users:**
```bash
# Create course-specific links without duplicating files
mkdir -p raw_docs/leadership_mastery/
ln -s /home/user/ebooks/leadership_fundamentals.epub raw_docs/leadership_mastery/
ln -s /home/user/ebooks/management_excellence.epub raw_docs/leadership_mastery/
```

## Workflow 2: Processing Udemy Courses

### Scenario: You have a complete Udemy course with videos, code files, and PDFs

**Typical Udemy Course Structure:**
```
/downloads/python_course/
├── 01_Introduction/
│   ├── lecture1.mp4
│   ├── lecture2.mp4
│   └── slides.pdf
├── 02_Fundamentals/
│   ├── lesson3.mp4
│   ├── code_examples.zip
│   └── exercises.pdf
└── resources/
    ├── cheatsheet.pdf
    └── final_project.zip
```

### Step 1: Organize Course Materials
```bash
# Place entire course folder in raw_docs
cp -r /downloads/python_course/ raw_docs/python_mastery/
```

### Step 2: Process via Directory Method
1. Upload Documents → Directory Path
2. Enter: `raw_docs/python_mastery/`
3. System automatically:
   - Transcribes all .mp4 files using RTX 3060 Whisper
   - Extracts text from PDFs
   - Processes code files
   - Creates unified vector embeddings

### Step 3: Apply Learning to Problem Sets
```
"Based on this Python course content, create a detailed implementation plan for solving [specific problem]. Include code structure, step-by-step approach, potential challenges, and testing strategies."
```

## Workflow 3: Multi-Page Strategic Plan Generation

### Use Case: Generate comprehensive business plans from books

**Best Practices for Multi-Page Responses:**

1. **Upload Strategic Business Books**
   - Business strategy guides
   - Industry-specific handbooks
   - Case study collections

2. **Ask Comprehensive Questions**
   ```
   "Generate a complete market entry strategy using the frameworks from these books, tailored to [your specific situation]. Include market analysis, competitive positioning, resource requirements, timeline, risk assessment, and success metrics."
   ```

3. **Request Structured Outputs**
   ```
   "Create a detailed 5-year business plan with quarterly milestones, financial projections, and implementation roadmap based on the methodologies in these business books."
   ```

### Expected Outputs:
- **Multi-page detailed responses** (1000-3000 words)
- **Automatic Excel generation** with timelines and budgets
- **Google Sheets creation** for complex financial models
- **Specific book references** and methodology citations

## Workflow 4: Bulk Audio/Video Processing

### Scenario: Process multiple podcasts, lectures, or video series

**Recommended Structure:**
```
raw_docs/podcast_series/
├── episode_001.mp3
├── episode_002.mp3
├── episode_050.mp3
└── show_notes.pdf
```

**Processing Steps:**
1. Place all audio/video files in course folder
2. Use Directory Path upload method
3. RTX 3060 processes all files with Whisper
4. Creates searchable transcription database
5. Query across all episodes for insights

## Cost Optimization Strategies

### Smart Caching Usage
- Similar questions use cached responses (free)
- Provider-specific caching (OpenAI vs Perplexity)
- 24-hour cache duration for cost savings

### Local vs Cloud Balance
- **RTX 3060 Local**: Whisper transcription, vector embeddings
- **Cloud APIs**: Final response generation for quality
- **Typical Cost**: $0.001-0.01 per strategic query

### Provider Selection
- **OpenAI GPT-4**: Best for detailed analysis and planning
- **Perplexity**: Excellent for research-heavy queries with citations
- **Caching**: Reduces repeat query costs significantly

## Advanced Tips

### For Business Books:
- Upload related books together for comprehensive frameworks
- Ask for implementation roadmaps with specific timelines
- Request Excel outputs for financial planning and tracking

### For Technical Courses:
- Process video lectures with accompanying code files
- Query for problem-solving approaches and debugging strategies
- Generate project templates based on course content

### For Multi-Source Learning:
- Combine books + videos + podcasts in single course
- Cross-reference methodologies from different sources
- Generate synthesis reports comparing approaches

This workflow maximizes your RTX 3060 capabilities while leveraging cloud APIs for superior response quality, creating a cost-effective learning and planning system.